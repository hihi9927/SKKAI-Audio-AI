#!/usr/bin/env python3
"""ACL 60/60 장문 채점 — StreamLAAL(공식 구현) + 재분절 BLEU + 진단.

    .venv-streamlaal/bin/python evaluation/ast/score_acl6060.py \\
        --tag 20260827_002522 --split dev

지표 계산은 **공식 구현에 위임**한다(`simulstream`, StreamLAAL v2 = mweralign 기반).
우리가 하는 일은 커밋 스트림을 그쪽 입력으로 옮기고(`streamlaal_adapter`), 옮기는
과정이 조용히 새는지 감시하는 것이다.

`assert` 는 개수만 지키고 의미는 안 지킨다. 그래서 런마다 아래 넷을 남기고, 첫 런의
값을 기준선으로 저장해 이후 런이 얼마나 벗어났는지 **숫자로** 판단한다.

    as_wer                  재분절이 달성한 mWER. **절대값으로 읽으면 안 된다** — 시스템
                            번역은 참조 번역과 원래 많이 다르므로 70~80% 가 정상이다
                            (합성 케이스에서 0 이 나온 건 가설을 참조와 동일하게 넣었기
                            때문이다). 쓸모는 **축 간 비교**에 있다: BLEU 가 오르면 AS-WER
                            도 같이 내려가야 정상이고, BLEU 는 올랐는데 AS-WER 이 그대로면
                            그때 재분절을 의심한다.
    n_negative_delays       문장 시작 전에 나온 단위 수(anticipation). 경계에서 드물어야.
    n_violation_intra       조각 **내부** 단조 위반. 합성에서 항상 0 — 0이 아니면 펼치기 누수.
    n_violation_boundary    조각 **경계** 역전. 커밋 타이밍의 정상 특성일 수 있다.
    null_rate               가설이 하나도 안 붙은 참조 문장 비율.

null 문장 처리
--------------
주지표는 **제외**한다 — 공식 구현이 그렇게 하고(`skipped_sentences`), IWSLT 순위도
그 값이다. 다만 제외는 selection bias 가 있다(어려운 문장을 안 내면 지연이 좋아진다).
그래서 `--null-penalty-sec` 로 벌점을 준 **보조 컬럼**을 함께 낸다. 기본 10초는
null alignment 에 10초를 물리는 최근 관행을 따른 것이고, 임의값이므로 반드시 함께 보고한다.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mweralign  # noqa: E402
from simulstream.metrics.readers import text_items  # noqa: E402
from simulstream.metrics.scorers.latency import LatencyScoringSample  # noqa: E402
from simulstream.metrics.scorers.latency.mwersegmenter import (  # noqa: E402
    ResegmentedLatencyScoringSample)
from simulstream.metrics.scorers.latency.stream_laal import StreamLaal  # noqa: E402
from simulstream.metrics.readers import OutputWithDelays  # noqa: E402

from streamlaal_adapter import (  # noqa: E402
    Commit, Diagnostics, build_output_with_delays, build_reference_defs)

LANG_UNIT = {"de": "word", "ja": "char", "zh": "char"}
BLEU_TOK = {"de": "13a", "ja": "ja-mecab", "zh": "zh"}


@contextlib.contextmanager
def capture_fd_stdout():
    """**파일 디스크립터 수준**으로 stdout/stderr 를 가로챈다.

    mweralign 은 AS-WER 을 네이티브 확장에서 직접 fd 에 쓴다(실측: fd 2). 그래서
    `contextlib.redirect_stdout`(파이썬 `sys.stdout` 만 바꾼다)으로는 안 잡힌다.
    """
    import os
    import tempfile
    # 호출자는 블록을 **빠져나온 뒤** sink[0] 에서 읽는다. tempfile 은 with 를 나가며
    # 닫히므로 파일 객체를 그대로 넘겨주면 읽을 수 없다.
    sink: List[str] = [""]
    # mweralign 은 AS-WER 을 **stderr** 로 낸다("loading reference…" 도 마찬가지).
    # 둘 다 잡아야 놓치지 않는다.
    sys.stdout.flush(); sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            yield sink
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved_out, 1); os.close(saved_out)
            os.dup2(saved_err, 2); os.close(saved_err)
            tmp.seek(0)
            sink[0] = tmp.read().decode("utf-8", "replace")


def _stat(xs: List[float]) -> Optional[dict]:
    """FTL 요약. 평균·중앙·p90·음수 개수를 한 벌로 낸다."""
    if not xs:
        return None
    ys = sorted(xs)
    return {
        "mean": round(statistics.mean(ys), 4),
        "median": round(statistics.median(ys), 4),
        "p90": round(ys[min(len(ys) - 1, int(len(ys) * 0.9))], 4),
        "n": len(ys),
        "n_negative": sum(1 for x in ys if x < 0),
    }


def load_manifest(path: Path) -> Dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                out[e["utt_id"]] = e
    return out


def commits_from_row(row: dict) -> List[Commit]:
    """`metric.json` 의 세그먼트 → 커밋. **결정 시점 순서**로 정렬한다.

    `segment_id` 는 서버가 매기는 발행 순서라 대체로 시간순이지만, 지연의 근거는
    `decision_audio_sec` 이므로 그걸 1순위로 둔다(동률이면 segment_id).
    """
    segs = [s for s in row["segments"]
            if s.get("decision_audio_sec") is not None
            and s.get("recv_elapsed_sec") is not None]
    segs.sort(key=lambda s: (s["decision_audio_sec"], s["segment_id"] or 0))
    return [Commit(text=(s["translation"] or "").strip(),
                   ideal_delay=float(s["decision_audio_sec"]),
                   ca_delay=float(s["recv_elapsed_sec"]))
            for s in segs]


def score_one(run_dir: Path, manifest: Dict[str, dict], lang: str,
              null_penalty_sec: float) -> dict:
    unit = LANG_UNIT[lang]
    scorer = StreamLaal(argparse.Namespace(latency_unit=unit))

    metric = json.loads((run_dir / "metric.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    # 축 라벨 교차검증 — 서버가 판정한 값과 클라이언트에 준 이름이 어긋나면 곡선의
    # 점이 뒤바뀐다. 값으로 조용히 틀리는 종류라 여기서 세게 막는다.
    scfg = meta.get("server_config") or {}
    axis_server, axis_client = scfg.get("axis"), meta["args"]["model"]
    if axis_server and axis_server != axis_client.split("-c")[0]:
        raise SystemExit(
            f"!! 축 라벨 불일치: 서버 판정 {axis_server!r} vs 클라이언트 {axis_client!r}\n"
            f"   {run_dir}")
    if scfg.get("axis") is None:
        print(f"   [주의] server_config 없음(구버전 런) — 축 교차검증 생략: {run_dir.name}")

    reseg_pairs, per_sentence = [], []
    ftl: List[float] = []
    ftl_ca: List[float] = []
    diag_total = Diagnostics()
    as_wers, n_null, n_ref_total = [], 0, 0
    n_neg = 0

    samples = []
    for row in metric["rows"]:
        item = manifest[row["utt_id"]]
        sentences = item["sentences"]
        refs = build_reference_defs(sentences)
        d = Diagnostics()
        owd = build_output_with_delays(commits_from_row(row), unit, d)
        diag_total.n_commits += d.n_commits
        diag_total.n_empty_commits_dropped += d.n_empty_commits_dropped
        diag_total.n_units += d.n_units
        diag_total.n_violation_intra += d.n_violation_intra
        diag_total.n_violation_boundary += d.n_violation_boundary
        diag_total.notes += d.notes

        # 재분절 — 공식 구현의 메서드를 그대로 부른다(우리가 다시 짜지 않는다).
        # mweralign 이 AS-WER 을 stderr 로 뱉으므로 잡아서 진단으로 쓴다(해석은 위 docstring).
        with capture_fd_stdout() as cap:
            hypo_tok = scorer._tokenize([owd.final_text])
            refs_tok = scorer._tokenize([r.content for r in refs])
            reseg = mweralign.mweralign.align_texts(refs_tok, hypo_tok).split("\n")
        for ln in cap[0].splitlines():
            if "AS-WER" in ln:
                with contextlib.suppress(ValueError):
                    as_wers.append(float(ln.rsplit(":", 1)[1]))
        if scorer.segmenter is not None:
            reseg = [h.replace(" ", "").replace("_", " ") for h in reseg]

        # 지연을 재분절 경계로 자른다 — 여기서 개수가 어긋나면 공식 구현이 assert 로 막는다.
        ideal_sp = scorer._split_delays_by_segmented_text(owd.ideal_delays, reseg)
        ca_sp = scorer._split_delays_by_segmented_text(owd.computational_aware_delays, reseg)

        hyp_objs = [OutputWithDelays(t, i, c) for t, i, c in zip(reseg, ideal_sp, ca_sp)]
        samples.append(ResegmentedLatencyScoringSample(row["utt_id"], hyp_objs, refs))

        for h, r, s in zip(hyp_objs, refs, sentences):
            n_ref_total += 1
            if not h.ideal_delays:
                n_null += 1
            else:
                n_neg += sum(1 for x in h.ideal_delays if x - r.start_time < 0)
                # 문장 단위 FTL — 그 문장의 **첫 글자**가 나오기까지 걸린 시간.
                # 발표 단위 FTL(클라이언트가 재는 값)은 12분 발화의 첫 커밋 하나만
                # 보므로 발표당 표본이 1개다(5발표 → 5표본). 축 비교에 못 쓴다.
                # 여기 값은 StreamLAAL 이 평균 내는 재료의 **첫 항**과 같다.
                ftl.append(h.ideal_delays[0] - r.start_time)
                ftl_ca.append(h.computational_aware_delays[0] - r.start_time)
            reseg_pairs.append({"utt_id": row["utt_id"], "seg_id": s["seg_id"],
                                "src": s["src"], "ref": r.content, "hyp": h.final_text})
            # 문장별 LAAL — `_do_score` 는 평균만 돌려주므로 같은 식(`_sentence_level_laal`)
            # 을 문장마다 한 번 더 부른다. **재구현이 아니라 같은 메서드 호출**이다.
            # 평균이 `stream_laal_sec` 과 일치하는지 아래에서 검산하므로, 어긋나면
            # 즉시 터진다. 평균만 보면 꼬리가 숨는다 — 분포를 봐야 seg 를 제대로 읽는다.
            _laal = _laal_ca = None
            if h.ideal_delays:
                _tl = len(text_items(r.content, unit))
                _d = [x - r.start_time for x in h.ideal_delays]
                _dc = [x - r.start_time for x in h.computational_aware_delays]
                _laal = StreamLaal._sentence_level_laal(_d, r.duration, _tl)
                _laal_ca = StreamLaal._sentence_level_laal(_dc, r.duration, _tl)
            per_sentence.append({
                "utt_id": row["utt_id"], "seg_id": s["seg_id"],
                "n_units": len(h.ideal_delays),
                "ref_units": len(text_items(r.content, unit)),
                "ref_start_sec": round(r.start_time, 3),
                "ref_dur_sec": round(r.duration, 3),
                "laal_sec": round(_laal, 4) if _laal is not None else None,
                "laal_ca_sec": round(_laal_ca, 4) if _laal_ca is not None else None,
            })

    # ── 지연: 공식 구현에 위임 ───────────────────────────────────────────────
    warn = io.StringIO()
    with contextlib.redirect_stderr(warn):
        scores = scorer._do_score(samples)

    # 검산: 문장별 LAAL 의 평균은 공식 구현의 집계와 일치해야 한다. 어긋나면
    # 우리가 문장을 잘못 짝지었다는 뜻이므로 조용히 넘기지 않는다.
    _sl = [x["laal_sec"] for x in per_sentence if x["laal_sec"] is not None]
    if _sl:
        _diff = abs(statistics.mean(_sl) - scores.ideal_latency)
        if _diff > 1e-3:
            raise SystemExit(
                f"!! 문장별 LAAL 평균({statistics.mean(_sl):.4f})이 집계"
                f"({scores.ideal_latency:.4f})와 다르다 — 문장 짝짓기를 확인할 것\n"
                f"   {run_dir}")

    # ── 보조: null 에 벌점을 준 변형 ────────────────────────────────────────
    # 주지표(제외)는 어려운 문장을 안 내면 지연이 좋아지는 selection bias 가 있다.
    penalized = None
    if n_ref_total:
        kept = n_ref_total - n_null
        if kept:
            penalized = (scores.ideal_latency * kept
                         + null_penalty_sec * n_null) / n_ref_total

    # ── 품질: 재분절 후 corpus BLEU ─────────────────────────────────────────
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize=BLEU_TOK[lang])
    bs = bleu.corpus_score([p["hyp"] for p in reseg_pairs],
                           [[p["ref"] for p in reseg_pairs]])

    return {
        "run_dir": str(run_dir),
        # **클라이언트 라벨을 쓴다.** 서버 판정(`axis_server`)은 커밋 정책만 보므로
        # 청크 스윕이 전부 "static" 으로 뭉개진다. 그러면 `reseg_*.json` 의 키
        # (`{axis}/{lang}`)가 static@2s/4s/6s 끼리 충돌해 COMET 채점에서 두 조건이
        # 조용히 사라진다(실측: c4·c6 가 통째로 누락). 위에서 이미
        # `axis_server == axis_client.split("-c")[0]` 로 교차검증했으므로 라벨을
        # 믿어도 된다.
        "axis": axis_client or axis_server,
        "lang": lang,
        "latency_unit": unit,
        "stream_laal_sec": round(scores.ideal_latency, 4),
        "stream_laal_ca_sec": round(scores.computational_aware_latency, 4),
        "stream_laal_null_penalized_sec": (
            round(penalized, 4) if penalized is not None else None),
        "null_penalty_sec": null_penalty_sec,
        # 평균만 쓰면 seg 를 과대평가한다 — seg 는 대체로 빠르지만 SEG 가뭄에서
        # 크게 늦는 꼬리가 있다(실측 dev/ja: 평균 2.99s 인데 p90 9.05s, static@6s 는
        # 평균 3.17s / p90 6.21s). p90 을 반드시 함께 보고할 것.
        # 음수 FTL 은 자르지 않는다 — 자르면 재분절 오배정이 정시로 위장된다.
        "ftl_sec": _stat(ftl),
        "ftl_ca_sec": _stat(ftl_ca),
        "bleu": round(bs.score, 2),
        "bleu_signature": str(bleu.get_signature()),
        "n_talks": len(metric["rows"]),
        "n_ref_sentences": n_ref_total,
        "n_null_sentences": n_null,
        "null_rate": round(n_null / n_ref_total, 4) if n_ref_total else None,
        "diagnostics": {
            "as_wer_mean": round(statistics.mean(as_wers), 3) if as_wers else None,
            "as_wer_max": round(max(as_wers), 3) if as_wers else None,
            "n_negative_delays": n_neg,
            "n_violation_intra": diag_total.n_violation_intra,
            "n_violation_boundary": diag_total.n_violation_boundary,
            "n_commits": diag_total.n_commits,
            "n_units": diag_total.n_units,
            "n_empty_commits_dropped": diag_total.n_empty_commits_dropped,
            "notes": diag_total.notes[:10],
        },
        "_reseg": reseg_pairs,
        "_per_sentence": per_sentence,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="ACL 60/60 StreamLAAL 채점")
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    p.add_argument("--manifest-dir", default=str(HERE / "manifests"))
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--axes", nargs="+", default=None)
    p.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--null-penalty-sec", type=float, default=10.0,
                   help="보조 컬럼에서 null 문장 하나에 물리는 벌점(초). 임의값이므로 함께 보고할 것")
    p.add_argument("--baseline-out", default=None,
                   help="첫 런 진단을 기준선으로 저장할 경로")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve()
    axes = a.axes or sorted(d.name for d in root.iterdir() if d.is_dir())

    results = []
    for axis in axes:
        for lang in a.langs:
            run_dir = root / axis / f"{a.split}-{lang}" / a.tag
            if not (run_dir / "metric.json").exists():
                continue
            man = load_manifest(Path(a.manifest_dir) / f"acl6060_{a.split}_en-{lang}.jsonl")
            print(f"── {axis}/{lang}")
            r = score_one(run_dir, man, lang, a.null_penalty_sec)
            results.append(r)
            dg = r["diagnostics"]
            print(f"   StreamLAAL {r['stream_laal_sec']:7.3f}s  "
                  f"CA {r['stream_laal_ca_sec']:7.3f}s  BLEU {r['bleu']:5.2f}")
            if r["ftl_sec"]:
                f_, fc = r["ftl_sec"], r["ftl_ca_sec"]
                print(f"   FTL 평균 {f_['mean']:.3f}s 중앙 {f_['median']:.3f}s "
                      f"p90 {f_['p90']:.3f}s 음수 {f_['n_negative']}/{f_['n']} | "
                      f"CA 평균 {fc['mean']:.3f}s p90 {fc['p90']:.3f}s")
            print(f"   null {r['n_null_sentences']}/{r['n_ref_sentences']} "
                  f"({r['null_rate']*100:.1f}%) → 벌점판 "
                  f"{r['stream_laal_null_penalized_sec']}s")
            print(f"   진단: AS-WER 평균 {dg['as_wer_mean']} 최대 {dg['as_wer_max']} | "
                  f"음수 {dg['n_negative_delays']} | "
                  f"조각내부위반 {dg['n_violation_intra']} | "
                  f"경계역전 {dg['n_violation_boundary']} | "
                  f"커밋 {dg['n_commits']} 단위 {dg['n_units']}")
            if dg["n_violation_intra"]:
                print("   !! 조각 내부 단조 위반 — 펼치기가 새고 있다. 즉시 확인할 것")

    if not results:
        print("채점할 결과가 없습니다."); return 2

    out = Path(a.out) if a.out else root / f"streamlaal_{a.split}_{a.tag}.json"
    reseg = {f"{r['axis']}/{r['lang']}": r.pop("_reseg") for r in results}
    sents = {f"{r['axis']}/{r['lang']}": r.pop("_per_sentence") for r in results}
    out.write_text(json.dumps({"tag": a.tag, "split": a.split, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / f"reseg_{a.split}_{a.tag}.json").write_text(
        json.dumps(reseg, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out}")
    print(f"      {out.parent / f'reseg_{a.split}_{a.tag}.json'}  (COMET 채점용)")
    (out.parent / f"laal_sentences_{a.split}_{a.tag}.json").write_text(
        json.dumps(sents, ensure_ascii=False), encoding="utf-8")
    print(f"      {out.parent / f'laal_sentences_{a.split}_{a.tag}.json'}  (문장별 LAAL 분포)")

    if a.baseline_out:
        Path(a.baseline_out).write_text(json.dumps(
            {"tag": a.tag, "split": a.split,
             "baseline_from": results[0]["run_dir"],
             "diagnostics": results[0]["diagnostics"],
             "null_rate": results[0]["null_rate"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      기준선: {a.baseline_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
