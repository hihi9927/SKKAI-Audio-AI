"""타깃 언어를 바꿔가며 **같은 분절**을 재채점한다 — 분절의 타깃 일반성 검사.

루프는 (소스, 타깃) 한 쌍에서만 돈다. 그런데 `<SEG:n>` 은 **소스 언어의 표층형만
보고** 찍힌다 — 프롬프트의 `[When to Segment]`·`[Priority Rules]` 어디에도 타깃 언어가
없다. 그렇다면 ko-en 으로 최적화한 분절이 ko-ja·ko-zh 에서도 통해야 하고, 안 통한다면
"경계는 소스만 보고 정할 수 있다"는 설계 전제가 틀린 것이다. 그것을 재는 스크립트다.

**분절은 다시 하지 않는다.** 같은 프롬프트·같은 문장이므로 분절 캐시가 그대로 맞고,
LLM 호출이 0 이다. 타깃마다 바뀌는 것은 번역과 채점뿐이다.

  PYTHONPATH=. python -m core.meaning_segmentator.autoseg.multilingual_check \
      --run-id ko-en/run04 --split dev --limit 100 \
      --targets English German Spanish Japanese Chinese

## 읽는 법 — 절대값이 아니라 baseline 격차를 본다

타깃 간 절대값 비교에는 네 가지가 섞인다.
  1. 분절이 그 타깃에 실제로 맞는가          ← 재려는 것
  2. CometKiwi 의 언어쌍별 캘리브레이션      ← 교란
  3. NLI 모델 차이                            ← `--nli` 를 전 타깃 통일해 제거
  4. 번역기 품질의 언어쌍 차이                ← 교란

2·4 는 **같은 타깃 안에서 비교하면 대부분 상쇄된다.** 그래서 주 판독값은

    gap(타깃, T) = effective(ours @ T) − effective(mechanical_8)

이고, "의미 기반 분절이 이 타깃에서 기계 분절을 이기는가"를 직접 답한다. 기계 분절은
소스를 8자마다 자르므로 **모든 타깃에서 동일한 분절**이고, 따라서 타깃 내 차이는
분절 품질 차이만 남는다.

무료 sanity check 도 같이 낸다: `format_pass_rate` 와 `missing_boundaries` 는 소스와
`seg_text` 만으로 계산되므로 **모든 타깃에서 완전히 같은 값이어야 한다.** 다르게 나오면
하네스가 의도대로 안 돌고 있다는 뜻이다.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

from . import data, metrics
from .gateway import Gateway
from .loop import (compare_baselines, load_contra_floor, score_split,
                   target_is_spaced)
from .pipeline import (GoogleTranslator, JsonCache, blocks_scoring, coverage_need,
                       normalize_tags, segment_batch, to_lang_code, truncate,
                       validate)

_HERE = Path(__file__).resolve().parent


def _cell(v, spec: str, dash: str = "—") -> str:
    return format(v, spec) if v is not None else dash


def score_one_target(tgt_name: str, sentences, texts, seg_texts, spaced: bool,
                     t_grid: list[int], adequacy, consistency, contradiction,
                     tr_cache: JsonCache, out_dir: Path, workers: int,
                     tgt_code: str | None, use_context: bool, log) -> dict:
    """타깃 하나에서 번역 + 채점. 분절은 이미 정해져 있고 여기서 안 바뀐다."""
    code = tgt_code or to_lang_code(tgt_name)
    tgt_spaced = target_is_spaced(tgt_name)
    translator = GoogleTranslator(tgt_code=code, cache=tr_cache,
                                  workers=min(workers, 4), use_context=use_context)
    try:
        log(f"[{tgt_name}] 번역 시작 (gtx:{code}, 타깃 띄어쓰기={tgt_spaced})")
        full = translator.full(texts)

        rows = [{"id": s.id, "text": s.text, "seg_text": seg,
                 "full_trans": f, "by_T": {}}
                for s, seg, f in zip(sentences, seg_texts, full)]

        by_T: dict[str, dict] = {}
        for T in t_grid:
            cut = [truncate(seg, T, spaced) for seg in seg_texts]
            cut_texts = [c[0] for c in cut]
            missings = [c[1] for c in cut]
            sp = score_split(cut_texts, texts, full, translator, adequacy,
                             consistency, spaced, tgt_spaced, contradiction)
            for i, r in enumerate(rows):
                r["by_T"][str(T)] = {
                    "seg_text": cut_texts[i], "k": sp.k[i],
                    "missing_boundaries": missings[i],
                    "pieces_src": sp.pieces_src[i], "pieces_tgt": sp.pieces_tgt[i],
                    "pieces_contra": sp.pieces_contra[i],
                    "effective": sp.effective[i], "adequacy": sp.adequacy[i],
                    "contradiction": sp.contradiction[i],
                    "consistency": sp.consistency[i], "laal_words": sp.laal_words[i],
                }
            by_T[str(T)] = metrics.aggregate_split(
                T, sp.effective, sp.adequacy, sp.contradiction, sp.consistency,
                sp.chrf, sp.laal_words, sp.k, missings).to_dict()
            s = by_T[str(T)]
            log(f"[{tgt_name}] T={T}  eff={_cell(s['effective'], '.4f')} "
                f"adq={s['adequacy']:.4f} contra={_cell(s['contradiction'], '.4f')} "
                f"cons={s['consistency']:.4f} laal={s['laal_words']:.2f} "
                f"k={s['chunks_per_sentence']:.2f}")

        # 순위 진단 — 바닥은 타깃마다 다르다 (full 번역이 다르므로)
        low_t = min(t_grid)
        floor_fn = load_contra_floor(out_dir, rows, contradiction,
                                     filename=f"contra_floor_{code}.json",
                                     tgt_spaced=tgt_spaced)
        gaps = metrics.rank_contra_gaps(rows, low_t, floor_fn=floor_fn,
                                        tgt_spaced=tgt_spaced)
        gap = statistics.mean(gaps) if gaps else None
        # 점추정만 내면 오독을 부른다 — dev 150문장에서 se 가 격차와 같은 규모다.
        gap_se = (statistics.stdev(gaps) / len(gaps) ** 0.5) if len(gaps) > 1 else None
        gap_n = len(gaps)
        sp_corr, sp_n = metrics.rank_contra_spearman(rows, low_t)
        log(f"[{tgt_name}] 순위진단(T={low_t}) gap={_cell(gap, '+.4f')}"
            f"±{_cell(gap_se, '.4f')}(n={gap_n}) Spearman={_cell(sp_corr, '+.3f')}(n={sp_n})")

        # 비교군 — 소스 분절이라 전 타깃 동일. 타깃 내 격차가 주 판독값이다.
        baselines = compare_baselines(translator, adequacy, consistency, sentences,
                                      spaced, tgt_spaced, workers,
                                      contradiction=contradiction)
        for name, b in baselines.items():
            log(f"[{tgt_name}] {name:14s} eff={_cell(b['effective'], '.4f')} "
                f"adq={b['adequacy']:.4f} cons={b['consistency']:.4f} "
                f"laal={b['laal_words']:.2f}")

        return {"code": code, "tgt_spaced": tgt_spaced, "by_T": by_T,
                "baselines": baselines,
                "rank_contra_gap": round(gap, 4) if gap is not None else None,
                "rank_contra_gap_se": round(gap_se, 4) if gap_se is not None else None,
                "rank_contra_gap_n": gap_n,
                "rank_contra_spearman": round(sp_corr, 4) if sp_corr is not None else None,
                "gtx_context_line_mismatches": translator.context_line_mismatches,
                "rows": rows}
    finally:
        translator.close()


def build_report(res: dict) -> str:
    targets = res["targets"]
    t_grid = res["t_grid"]
    names = list(targets)

    def gap_to(name: str, T: str, base: str = "mechanical_8"):
        o = targets[name]["by_T"][T].get("effective")
        b = targets[name]["baselines"][base].get("effective")
        return None if (o is None or b is None) else o - b

    L = [
        f"# 타깃 언어 변동성 — {res['src_lang']} 분절의 다국어 일반성",
        "",
        f"- 기준 런 `{res['reference_run']}` / 프롬프트 `{Path(res['prompt_file']).name}`",
        f"- 데이터 `{res['split']}` {res['n']}문장 (분절은 **재사용**, LLM 호출 {res['seg_llm_calls']}회)",
        f"- 번역기 `gtx (use_context={res['use_context']})` / adequacy `{res['adequacy_backend']}`",
        f"- NLI **전 타깃 통일** `{res['nli_backend']}` — 모델 차이 교란을 제거하기 위함. "
        f"이 때문에 **run04 절대값과 비교 불가**",
        f"- 노브 격자 {t_grid}",
        "",
        "> **관문 미실시.** NLI 백엔드를 바꿨으나 `validity_check.py`/`judge_check.py` 를 "
        "돌리지 않았다. 참고로 `premature_cases.json` 6건에 대한 순위 조건은 "
        "`mdeberta-xnli` 도 6/6 통과하나, 영어 케이스의 마진이 크게 좁다 "
        "(ko-en-p01 +0.9750 → +0.0175). 또 zh/es/de 타깃용 fixture 는 아예 없다. "
        "**절대값은 신뢰하지 말고 아래 격차 표만 읽을 것.**",
        "",
        "## 1. Sanity — 타깃 무관 지표는 같아야 한다",
        "",
        "`format_pass_rate` 와 `missing_boundaries` 는 소스와 `seg_text` 만으로 계산된다. "
        "타깃이 달라도 **같은 값**이어야 하고, 다르면 하네스 버그다.",
        "",
        "| 타깃 | format_pass_rate | " + " | ".join(f"missing (T={t})" for t in t_grid) + " |",
        "|---|---|" + "---|" * len(t_grid),
    ]
    for n in names:
        row = [f"| {n} | {res['format_pass_rate']:.4f} "]
        for t in t_grid:
            row.append(f"| {targets[n]['by_T'][str(t)]['missing_boundaries']:.4f} ")
        L.append("".join(row) + "|")

    miss_sets = [tuple(round(targets[n]["by_T"][str(t)]["missing_boundaries"], 6)
                       for t in t_grid) for n in names]
    L += ["", f"→ 전 타깃 동일: **{'예' if len(set(miss_sets)) == 1 else '아니오 — 확인 필요'}**", ""]

    L += ["## 2. 주 판독값 — mechanical_8 대비 effective 격차", "",
          "타깃 내 비교라 QE 캘리브레이션·번역기 품질 차이가 대부분 상쇄된다. "
          "**양수 = 의미 기반 분절이 기계 분절을 이김.**", "",
          "| 타깃 | " + " | ".join(f"T={t}" for t in t_grid) + " | 평균 |",
          "|---|" + "---|" * (len(t_grid) + 1)]
    gaps_by_target: dict[str, list[float]] = {}
    for n in names:
        gs = [gap_to(n, str(t)) for t in t_grid]
        ok = [g for g in gs if g is not None]
        gaps_by_target[n] = ok
        L.append(f"| {n} | " + " | ".join(_cell(g, '+.4f') for g in gs)
                 + f" | **{_cell(statistics.mean(ok) if ok else None, '+.4f')}** |")

    means = [statistics.mean(v) for v in gaps_by_target.values() if v]
    if len(means) > 1:
        L += ["",
              f"- 타깃 간 격차 평균 **{statistics.mean(means):+.4f}**, "
              f"표준편차 **{statistics.stdev(means):.4f}**, "
              f"범위 {min(means):+.4f} ~ {max(means):+.4f}",
              f"- 전 타깃 양수: **{'예' if all(m > 0 for m in means) else '아니오'}** "
              "— 아니면 그 타깃에서 소스 기반 분절이 기계 분절만 못하다는 뜻"]

    L += ["", "## 3. 절대값 (교란 포함 — 타깃 간 직접 비교 금지)", "",
          "| 타깃 | T | effective | adequacy | contradiction | consistency | laal_words | k |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        for t in t_grid:
            s = targets[n]["by_T"][str(t)]
            L.append(f"| {n} | {t} | {_cell(s['effective'], '.4f')} | {s['adequacy']:.4f} | "
                     f"{_cell(s['contradiction'], '.4f')} | {s['consistency']:.4f} | "
                     f"{s['laal_words']:.2f} | {s['chunks_per_sentence']:.2f} |")
        for bn, b in targets[n]["baselines"].items():
            L.append(f"| {n} | *{bn}* | {_cell(b['effective'], '.4f')} | {b['adequacy']:.4f} | "
                     f"{_cell(b['contradiction'], '.4f')} | {b['consistency']:.4f} | "
                     f"{b['laal_words']:.2f} | {b['chunks_per_sentence']:.2f} |")

    L += ["", "## 4. 순위 일반성 — 소스 기반 순위가 타깃마다 유효한가", "",
          "`rank_contra_gap` 은 순위 하위 절반 − 상위 절반의 경계 contradiction 차 "
          "(잡음 바닥 보정). **0 이하면 그 타깃에서 순위가 정보를 주지 않는다** — "
          "절단이 위험을 못 덜어내므로 노브가 그 타깃에서 무력하다는 뜻이다.", "",
          "**se 를 먼저 볼 것.** dev 150문장에서 문장별 분산이 커 se 가 격차와 같은 "
          "규모다. `|gap| < 2·se` 면 그 타깃에서는 **판단 유보**이지 `0` 이 아니다.", "",
          "| 타깃 | rank_contra_gap | ± se | t = gap/se | 판정 | n | Spearman (raw) | gtx 줄수 불일치 |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        t = targets[n]
        g, se = t["rank_contra_gap"], t.get("rank_contra_gap_se")
        tv = (g / se) if (g is not None and se) else None
        verdict = ("—" if tv is None else
                   "정렬됨" if tv > 2 else "역전" if tv < -2 else "판단 유보")
        L.append(f"| {n} | **{_cell(g, '+.4f')}** | {_cell(se, '.4f')} | {_cell(tv, '+.2f')} | "
                 f"{verdict} | {t['rank_contra_gap_n']} | "
                 f"{_cell(t['rank_contra_spearman'], '+.3f')} | {t['gtx_context_line_mismatches']} |")
    L += ["", "`gtx 줄수 불일치` 가 0 이 아니면 컨텍스트 번역에서 마지막 줄 추출이 실패한 "
          "건수다 — 그 타깃의 조각 번역이 일부 오염됐을 수 있다.", ""]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(
        description="타깃 언어를 바꿔가며 같은 분절을 재채점 (분절 재사용, LLM 호출 0)")
    p.add_argument("--run-id", required=True, help="기준 런 경로 (runs/ 이하). 예: ko-en/run04")
    p.add_argument("--prompt", default=None, help="미지정 시 기준 런의 best_prompt.txt")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--limit", type=int, default=None, help="앞에서 N문장만 (비용 제어)")
    p.add_argument("--targets", nargs="+",
                   default=["English", "German", "Spanish", "Japanese", "Chinese"])
    p.add_argument("--t-grid", type=int, nargs="+", default=None,
                   help="미지정 시 기준 런의 final_t_grid")
    p.add_argument("--nli", default="xlmr-anli", choices=sorted(metrics.NLI_MODELS),
                   help="**전 타깃 통일.** 타깃마다 다른 모델을 쓰면 모델 차이가 "
                        "언어 차이로 오독된다")
    p.add_argument("--adequacy-backend", default=None, help="미지정 시 기준 런에서 상속")
    p.add_argument("--no-google-context", action="store_true")
    p.add_argument("--comet-batch-size", type=int, default=16)
    p.add_argument("--out-id", default=None, help="출력 디렉토리 이름. 기본 multilingual/<run>")
    p.add_argument("--model", default="gpt-5-mini", help="분절 캐시 미스 시에만 쓰인다")
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    # httpx 가 요청마다 URL 전체(원문이 percent-encoding 된 채)를 INFO 로 찍는다.
    # 문장당 수십 호출 × 5 타깃이면 로그가 수백 MB 가 되고 진행 상황이 파묻힌다.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    run_dir = _HERE.parent / "runs" / args.run_id
    if not run_dir.exists():
        print(f"런 디렉토리 없음: {run_dir}", file=sys.stderr)
        return 1
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    measured = json.loads((run_dir / "measured_profile.json").read_text(encoding="utf-8"))
    spaced, trailing_punct = data.profile_settings(measured)

    t_grid = sorted(set(args.t_grid or cfg.get("final_t_grid") or [2, 3, 4, 6]))
    coverage_t = cfg.get("min_boundaries_per") or min(t_grid)
    prompt_path = Path(args.prompt) if args.prompt else (run_dir / "best_prompt.txt")
    prompt = prompt_path.read_text(encoding="utf-8")

    sentences = data.read_split(run_dir / "data" / f"{args.split}.json")
    if args.limit:
        sentences = sentences[: args.limit]
    texts = [s.text for s in sentences]

    out_dir = _HERE.parent / "runs" / (args.out_id or f"multilingual/{Path(args.run_id).name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gw = Gateway(model=args.model, budget=args.budget)
    # 분절 캐시는 **기준 런 것을 그대로** 쓴다 — 같은 프롬프트·같은 문장이라 전부 적중하고
    # LLM 호출이 0 이다. 번역 캐시도 공유한다 (키에 tgt_code 가 들어가 충돌하지 않고,
    # 영어는 기준 런 결과를 그대로 재사용한다).
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        # ── 분절 1회 (타깃 무관) ─────────────────────────────────────────
        need = lambda txt: coverage_need(txt, coverage_t, spaced, 0)
        before = gw.usage.snapshot()["calls"]
        seg_texts, first_pass = segment_batch(
            gw, prompt, texts, cache=seg_cache, workers=args.workers,
            validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct,
                                                True, need(t)),
            normalize_fn=lambda out: normalize_tags(out, spaced, trailing_punct),
        )
        seg_calls = gw.usage.snapshot()["calls"] - before

        valid_flags, scored_flags, violations = [], [], []
        for s, seg in zip(sentences, seg_texts):
            vs = validate(s.id, s.text, seg, spaced, trailing_punct, True, need(s.text))
            valid_flags.append(not vs)
            scored_flags.append(not blocks_scoring(vs))
            violations.extend({"id": v.id, "rule": v.rule, "detail": v.detail} for v in vs)
        fmt = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
        log(f"[분절] {len(texts)}문장 재사용 — LLM 호출 {seg_calls}회, "
            f"포맷 통과율 {fmt:.4f} (1차 {sum(first_pass)/len(first_pass):.4f}), "
            f"위반 {len(violations)}건")

        # ── 백엔드는 전 타깃 공유 ────────────────────────────────────────
        adequacy = metrics.make_adequacy_backend(
            args.adequacy_backend or cfg.get("adequacy_backend", "cometkiwi"),
            batch_size=args.comet_batch_size)
        contradiction = metrics.make_contradiction_backend(args.nli)
        consistency = metrics.make_backend("nli", model_name=metrics.NLI_MODELS[args.nli],
                                           batch_size=args.comet_batch_size)

        res = {
            "reference_run": args.run_id, "prompt_file": str(prompt_path),
            "src_lang": cfg.get("src_lang", "Korean"),
            "split": args.split, "n": len(texts), "t_grid": t_grid,
            "seg_llm_calls": seg_calls,
            "format_pass_rate": round(fmt, 4),
            "format_pass_rate_no_retry": round(sum(first_pass) / len(first_pass), 4),
            "n_violations": len(violations),
            "adequacy_backend": adequacy.name, "nli_backend": args.nli,
            "use_context": not args.no_google_context,
            "gate_run": False,
            "targets": {},
        }
        for tgt in args.targets:
            res["targets"][tgt] = score_one_target(
                tgt, sentences, texts, seg_texts, spaced, t_grid, adequacy,
                consistency, contradiction, tr_cache, out_dir, args.workers,
                None, not args.no_google_context, log)

        rows_out = {t: res["targets"][t].pop("rows") for t in res["targets"]}
        (out_dir / f"{args.split}_rows_by_target.json").write_text(
            json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"{args.split}_summary.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        report = build_report(res)
        (out_dir / f"{args.split}_report.md").write_text(report, encoding="utf-8")
        log(f"\n{report}")
        log(f"[done] {out_dir}")
        return 0
    finally:
        seg_cache.flush()
        tr_cache.flush()
        gw.close()


if __name__ == "__main__":
    sys.exit(main())
