"""참조 기반 BLEU 로 분절 조건들을 비교한다 — 결정론적, 번역 캐시 재사용.

루프의 목적함수(`effective = adequacy × (1 − contradiction)`)는 참조가 없다. 이 스크립트는
같은 분절·같은 번역기 위에서 **corpus BLEU / chrF2** 를 낸다. 조건은 셋이다.

    unsegmented    문장 통째 번역 (같은 번역기에서의 상한)
    auto @T        prompt_eval 산출 분절을 T 로 절단한 것
    mechanical_8   의미를 무시한 8자 절단 (하한)

**언어 간 절대 BLEU 비교는 하지 말 것** — 토크나이저가 다르다 (de `13a`, zh `zh`, ja `ja-mecab`).
언어를 가로지르는 판독은 자기 상한으로 정규화한 `retention = BLEU(auto)/BLEU(unseg)` 로 하고,
토크나이저 선택이 없는 `chrF2` 를 주 근거로 둔다.

    python -m core.meaning_segmentator.autoseg.bleu_eval \
        --run-id en-multi/clean500 --label auto_best --targets de ja zh \
        --manifest-tag clean500 --t-grid 4 6 8 12
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sacrebleu

from . import data, metrics
from .baselines import coarsen
from .pipeline import GoogleTranslator, JsonCache, split_segments, to_lang_code

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.ast import metrics_ast as M  # noqa: E402

# gtx 타깃 코드는 파이프라인 규칙을 따르고(zh → zh-CN), 토크나이저는 sacrebleu 규칙을 따른다.
TGT_NAME = {"de": "German", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "es": "Spanish"}
SPACED = {"de": True, "es": True, "ja": False, "zh": False, "ko": True}


def load_refs(tag: str, tgt: str) -> dict[str, str]:
    return {k: v[0] for k, v in load_manifest(tag, tgt).items()}


def load_manifest(tag: str, tgt: str) -> dict[str, tuple[str, str]]:
    """utt_id → (참조 번역, talk_id). talk_id 로 FLEURS TSV 의 발화 길이를 찾는다."""
    path = (_REPO_ROOT / "evaluation" / "ast" / "manifests"
            / f"fleurs_nway_en-{tgt}_{tag}.jsonl")
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            out[e["utt_id"]] = (e["tgt_text"], str(e.get("talk_id", "")))
    return out


def load_durations(src_lang_dir: str = "en_us") -> dict[str, float]:
    """FLEURS TSV 6번 열(샘플 수) → 발화 길이 ms. 오디오 파일은 필요 없다.

    한 문장 id 에 화자별 녹음이 여러 개라 **중앙값**을 쓴다. 16kHz 고정.
    """
    base = Path.home() / "datasets" / "fleurs" / "data" / src_lang_dir
    by_id: dict[str, list[float]] = {}
    for split in ("train", "dev", "test"):
        f = base / f"{split}.tsv"
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) < 6 or not c[5].isdigit():
                    continue
                by_id.setdefault(c[0], []).append(int(c[5]) / 16000.0 * 1000.0)
    return {k: statistics.median(v) for k, v in by_id.items()}


def load_wordtimes(tag: str, source: str) -> dict[str, list[float]]:
    """talk_id → 어절 종료 시각(ms). 강제정렬 산출물.

    `interp` 는 타임스탬프 없이 발화 내 균일 발화속도로 보간하던 옛 방식이다 —
    실측 대비 도입부 묵음을 통째로 놓친다(17어절 7.4초 문장에서 첫 어절 실측 1.80s
    vs 보간 0.43s). 비교용으로만 남긴다.
    """
    if source == "interp":
        return {}
    suffix = "" if source == "ctc" else f"_{source}"
    path = (_REPO_ROOT / "evaluation" / "ast" / "manifests"
            / f"fleurs_nway_en_{tag}_wordtimes{suffix}.json")
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["word_end_ms"] for k, v in blob.items()}


def laal_ms(pieces_src: list[str], pieces_tgt: list[str], dur_ms: float,
            ref_text: str, spaced: bool, tgt_unit: str,
            word_end_ms: list[float] | None = None) -> float | None:
    """ms 단위 LAAL.

    `word_end_ms` 가 있으면 조각 경계 시각을 **강제정렬 실측**에서 읽는다. 없으면
    발화 내 균일 발화속도 보간으로 물러난다(구 방식, 각주 필요).

    기계분절은 문자 단위로 잘라 어절과 안 맞으므로 인덱스를 범위로 자른다.
    """
    if not dur_ms or not pieces_src:
        return None
    unit = (lambda x: len(x.split())) if spaced else (lambda x: len("".join(x.split())))
    counts = [unit(x) for x in pieces_src]
    total = sum(counts)
    if total <= 0:
        return None
    segs, cum = [], 0
    for src, tgt in zip(counts, pieces_tgt):
        cum += src
        if word_end_ms:
            i = min(max(cum, 1), len(word_end_ms)) - 1
            segs.append((tgt, float(word_end_ms[i])))
        else:
            segs.append((tgt, cum / total * dur_ms))
    return M.laal_for_utterance(segs, dur_ms, ref_text, tgt_unit)


def corpus_chrf2(hyps: list[str], refs: list[str]) -> float:
    """참조 기반 chrF2. 문자 n-gram 이라 토크나이저 선택이 없다."""
    m = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2)
    return m.corpus_score(hyps, [refs]).score


def paired_bootstrap(hyp_a: list[str], hyp_b: list[str], refs: list[str],
                     tokenize: str, n_boot: int = 1000, seed: int = 20260820) -> dict:
    """같은 문장 위에서의 corpus BLEU 차이. corpus BLEU 는 문장별 SE 가 없으므로 재표집한다.

    **문장별 n-gram 통계를 한 번만 뽑고 재표집에서는 합만 낸다** (sacrebleu 의
    `PairedTest` 와 같은 방식). 재표집마다 corpus_score 를 다시 부르면 토크나이즈가
    n_boot 배로 반복돼 ja-mecab 에서 몇 시간이 된다.
    """
    rng = random.Random(seed)
    # 새로 만들지 않는다 — ja-mecab 에서 Tagger 가 사전 4개를 mmap 하고 안 놓는다.
    bleu = M._bleu_metric(tokenize, False)
    sa = bleu._extract_corpus_statistics(hyp_a, [refs])
    sb = bleu._extract_corpus_statistics(hyp_b, [refs])
    N = len(refs)

    def score(stats: list[list], idx: list[int]) -> float:
        agg = [0.0] * len(stats[0])
        for i in idx:
            row = stats[i]
            for j, v in enumerate(row):
                agg[j] += v
        return bleu._compute_score_from_stats(agg).score

    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        deltas.append(score(sa, idx) - score(sb, idx))
    deltas.sort()
    lo, hi = deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]
    mean_d = statistics.mean(deltas)
    # 재표집 중 부호가 평균과 반대로 나온 비율. 낮을수록 방향이 확고하다.
    n_cross = (sum(1 for d in deltas if d <= 0) if mean_d > 0
               else sum(1 for d in deltas if d >= 0))
    return {"delta": round(mean_d, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "p_sign_flip": round(n_cross / n_boot, 4)}


def build_conditions(rows: list[dict], t_grid: list[int], spaced: bool,
                     mech_every: int) -> dict[str, list[dict]]:
    """조건 이름 → 문장별 {seg_text, pieces}. pieces 가 1개면 무분절과 같다."""
    out: dict[str, list[dict]] = {}
    out["unsegmented"] = [{"seg_text": r["text"], "pieces": [r["text"]]} for r in rows]
    for T in t_grid:
        key = str(T)
        cond = []
        for r in rows:
            cell = r["by_T"][key]
            cond.append({"seg_text": cell["seg_text"], "pieces": cell["pieces_src"]})
        out[f"auto_T{T}"] = cond
    for T in t_grid:
        cond = []
        for r in rows:
            all_pieces = split_segments(r["seg_text"]) or [r["text"]]
            pc = coarsen(all_pieces, T, spaced)
            cond.append({"seg_text": " <SEG> ".join(pc), "pieces": pc})
        out[f"auto_greedy_T{T}"] = cond
    mech = []
    for r in rows:
        seg = metrics.mechanical_split(r["text"], mech_every, spaced)
        mech.append({"seg_text": seg, "pieces": split_segments(seg) or [r["text"]]})
    out[f"mechanical_{mech_every}"] = mech
    return out


def load_baseline_conditions(run_dir: Path, policies: list[str], tgt: str,
                             split: str, rows: list[dict], t_grid: list[int],
                             spaced: bool) -> dict[str, list[dict]]:
    """`baselines/` 산출을 조건으로 읽는다. 타깃별 파일이 없으면 타깃 독립본을 쓴다.

    causal_align·mu_prefix 는 **타깃마다 분절이 다르다** (정렬 대상 / NMT 가 타깃별이다).
    자동 분절·기계 분절과 달리 `k` 가 타깃 간에 같지 않은 이유가 이것이고, 그 자체가
    Table 1a "외부 의존" 열이 재려는 비용이다.
    """
    out: dict[str, list[dict]] = {}
    for pol in policies:
        path = run_dir / "baselines" / f"{pol}_{tgt}_{split}.json"
        if not path.exists():
            path = run_dir / "baselines" / f"{pol}_all_{split}.json"
        if not path.exists():
            print(f"[{tgt}] 비교군 {pol} 산출 없음 — 건너뜀 ({path.name})")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        by_id = {r["id"]: r["pieces"] for r in blob["rows"]}
        missing = [r["id"] for r in rows if r["id"] not in by_id]
        if missing:
            raise SystemExit(f"[{tgt}] {pol}: 라벨 없는 문장 {len(missing)}건 "
                             f"(예: {missing[:3]}) — build 를 같은 --limit 없이 다시 돌릴 것")
        base = [by_id[r["id"]] or [r["text"]] for r in rows]
        out[pol] = [{"seg_text": " <SEG> ".join(x), "pieces": x} for x in base]
        for T in t_grid:                     # 정책 경계의 부분집합으로 지연을 옮긴다
            cond = []
            for x in base:
                pc = coarsen(x, T, spaced)
                cond.append({"seg_text": " <SEG> ".join(pc), "pieces": pc})
            out[f"{pol}_T{T}"] = cond
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="분절 조건별 참조 기반 BLEU")
    p.add_argument("--run-id", required=True, help="runs/ 이하 경로 (config·캐시·분절 출처)")
    p.add_argument("--label", default="auto_best", help="prompt_eval 산출 라벨")
    p.add_argument("--split", default="test")
    p.add_argument("--targets", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--manifest-tag", default="clean500")
    p.add_argument("--t-grid", type=int, nargs="+", default=[4, 6, 8, 12])
    p.add_argument("--mech-every", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="gtx 는 비공식 엔드포인트 — 낮게 둔다")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--conditions", nargs="+", default=None, help="미지정 시 전부")
    p.add_argument("--wordtimes", default="qwen", choices=["qwen", "ctc", "interp"],
                   help="지연 시각의 출처. 기본 `qwen` — Table 3/4 가 Qwen3-ASR 를 쓰므로 "
                        "지연축을 같은 자로 재고, 비영어 소스로 확장할 때도 쓸 수 있다. "
                        "`ctc`(wav2vec2, 영어 전용)는 독립 교차검증용이며 조건 LAAL 이 "
                        "22ms 이내로 일치한다. `interp` 는 타임스탬프 없던 옛 방식")
    p.add_argument("--baselines", nargs="+", default=[],
                   help="baselines/ 산출을 조건으로 추가 (punct causal_align mu_prefix)")
    args = p.parse_args()

    run_dir = _HERE.parent / "runs" / args.run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    spaced = bool(json.loads((run_dir / "measured_profile.json").read_text(
        encoding="utf-8"))["uses_spaces_between_words"])

    ev = json.loads((run_dir / "prompt_eval" / f"{args.label}_{args.split}.json"
                     ).read_text(encoding="utf-8"))
    rows = ev["rows"]
    print(f"분절 출처: {args.label}_{args.split}.json ({len(rows)}문장), 소스 띄어쓰기 {spaced}")

    conds = build_conditions(rows, args.t_grid, spaced, args.mech_every)

    out_dir = run_dir / "bleu"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for tgt in args.targets:
        code = to_lang_code(TGT_NAME[tgt])
        tgt_spaced = SPACED[tgt]
        tokenize = M.resolve_tokenize(tgt)
        man = load_manifest(args.manifest_tag, tgt)
        refs_map = {k: v[0] for k, v in man.items()}
        durs = load_durations()
        wtimes = load_wordtimes(args.manifest_tag, args.wordtimes)
        tgt_unit = "word" if tgt_spaced else "char"
        ids = [r["id"] for r in rows]
        missing = [i for i in ids if i not in refs_map]
        if missing:
            print(f"[{tgt}] 참조 없는 문장 {len(missing)}건 제외")
        keep = [i for i, sid in enumerate(ids) if sid in refs_map]
        refs = [refs_map[ids[i]] for i in keep]

        conds_t = dict(conds)
        conds_t.update(load_baseline_conditions(
            run_dir, args.baselines, tgt, args.split, rows, args.t_grid, spaced))
        if args.conditions:
            conds_t = {k: v for k, v in conds_t.items() if k in args.conditions}

        cache = JsonCache(run_dir / "cache" / f"translate_{code}.json")
        tr = GoogleTranslator(tgt_code=code, cache=cache, workers=args.workers,
                              use_context=True)
        full_trans = tr.full([rows[i]["text"] for i in keep])
        per_cond = {}
        try:
            joiner = " " if tgt_spaced else ""

            def translate_one(pieces: list[str]) -> list[str]:
                # 문장 **안**은 직렬이어야 한다 (조각 i 는 앞 조각들을 붙여 번역한다).
                # 문장 **간**은 독립이므로 병렬로 돌린다 — 이걸 안 하면 조각 하나가
                # gtx 왕복 하나라 전체가 1콜/초로 떨어진다 (실측).
                # ms LAAL 이 조각별 번역 길이를 요구하므로 합치지 않고 리스트로 돌린다.
                return list(tr.streaming_segments(pieces))

            for name, cond in conds_t.items():
                pieces_all = [cond[i]["pieces"] for i in keep]
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    per_piece = list(ex.map(translate_one, pieces_all))
                hyps = [joiner.join(x for x in pp if x) for pp in per_piece]
                ks = [len(x) for x in pieces_all]
                laal = [metrics.laal_words(cond[i]["seg_text"], None, full_trans[j],
                                           spaced, tgt_spaced)
                        for j, i in enumerate(keep)]
                laal_ms_vals = []
                for j, i in enumerate(keep):
                    d = durs.get(man[ids[i]][1])
                    v = laal_ms(pieces_all[j], per_piece[j], d or 0.0,
                                refs[j], spaced, tgt_unit,
                                wtimes.get(man[ids[i]][1]))
                    if v is not None:
                        laal_ms_vals.append(v)
                bleu, sig = M.corpus_bleu_score(hyps, refs, tokenize)
                per_cond[name] = {
                    "bleu": round(bleu, 3) if bleu is not None else None,
                    "bleu_signature": sig,
                    "chrf2": round(corpus_chrf2(hyps, refs), 3),
                    "mean_sentence_bleu": round(statistics.mean(
                        M.sentence_bleu_score(h, r, tokenize) or 0.0
                        for h, r in zip(hyps, refs)), 3),
                    "k": round(statistics.mean(ks), 2),
                    "laal_words": round(statistics.mean(laal), 2),
                    "laal_ms": (round(statistics.mean(laal_ms_vals), 1)
                                if laal_ms_vals else None),
                    "n_laal_ms": len(laal_ms_vals),
                    "n": len(hyps),
                    "hyps": hyps,
                }
                _ms = per_cond[name]["laal_ms"]
                print(f"[{tgt}] {name:18s} BLEU {per_cond[name]['bleu']:6.2f}  "
                      f"chrF2 {per_cond[name]['chrf2']:6.2f}  k {per_cond[name]['k']:5.2f}  "
                      f"laal {per_cond[name]['laal_words']:5.2f}w  "
                      f"{'—' if _ms is None else format(_ms, '7.0f') + 'ms'}")
                cache.flush()
        finally:
            cache.flush()
            tr.close()

        base = per_cond.get("unsegmented")
        for name, cell in per_cond.items():
            if base and cell["bleu"] is not None and base["bleu"]:
                cell["retention_bleu"] = round(cell["bleu"] / base["bleu"], 4)
                cell["retention_chrf2"] = round(cell["chrf2"] / base["chrf2"], 4)
            if name != "unsegmented" and base:
                cell["paired_vs_unseg"] = paired_bootstrap(
                    cell["hyps"], base["hyps"], refs, tokenize, args.bootstrap)
        if f"mechanical_{args.mech_every}" in per_cond:
            mech = per_cond[f"mechanical_{args.mech_every}"]
            for name, cell in per_cond.items():
                if name.startswith("auto_"):
                    cell["paired_vs_mech"] = paired_bootstrap(
                        cell["hyps"], mech["hyps"], refs, tokenize, args.bootstrap)

        report[tgt] = {"tokenize": tokenize, "n": len(refs),
                       "translator": f"google:{code}:ctx=True",
                       "context_line_mismatches": tr.context_line_mismatches,
                       "conditions": per_cond}
        (out_dir / f"{tgt}.json").write_text(
            json.dumps(report[tgt], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{tgt}] gtx 컨텍스트 줄수 불일치 {tr.context_line_mismatches}건 "
              f"→ {out_dir / f'{tgt}.json'}")

    write_report(out_dir / "report.md", report, args, cfg)
    print(f"\n{(out_dir / 'report.md')}")
    return 0


def write_report(path: Path, report: dict, args, cfg) -> None:
    L = [f"# 분절 조건별 참조 기반 BLEU — {args.run_id}", "",
         f"- 분절: `{args.label}` / 번역기 gtx(컨텍스트) / 부트스트랩 {args.bootstrap}회",
         "- **언어 간 절대 BLEU 비교 금지** — 토크나이저가 다르다. 언어를 가로지르는 판독은 "
         "`retention`(자기 무분절 대비)과 `chrF2` 로 한다.",
         "- 타깃 de·ja·zh 는 프롬프트 최적화 시 목적함수에 포함됐던 언어다 (미출현 타깃 아님).",
         "- `k`(조각 수)는 자동·기계 분절에서는 세 타깃이 **같은 분절**을 쓰므로 동일하다. "
         "단 `causal_align`·`mu_prefix` 는 타깃별 자원(정렬 대상·NMT)에 의존하므로 타깃마다 다르다. "
         "`laal_words` 는 정의상 "
         "타깃 길이가 들어가고 ja·zh 는 타깃 단위가 문자라 값이 커진다 — **타깃 간 laal 비교 금지**, "
         "같은 타깃 안에서 T 방향만 읽을 것.",
         f"- `laal_ms` 는 **강제정렬 실측**이다 (`--wordtimes {args.wordtimes}`). 독립 정렬기 "
         "둘(wav2vec2 CTC / Qwen3-ForcedAligner)이 조건 수준 LAAL 에서 22ms 이내로 일치한다. "
         "구 방식(발화 내 균일속도 보간)은 지연을 64~131ms 과소평가했고 정책마다 편차가 있었다.",
         "- `*_T*` 비교군 점은 정책의 경계 **부분집합**이다 (좌→우 탐욕). 제안 `auto_T*` 만 "
         "LLM 순위로 남길 경계를 고르므로, 순위 이득을 뺀 대조는 `auto_greedy_T*` 다.", ""]
    for tgt, blob in report.items():
        L += [f"## en→{tgt} (n={blob['n']}, tok:{blob['tokenize']})", "",
              "| 조건 | k | laal_ms ↓ | laal_words ↓ | BLEU ↑ | chrF2 | retention(BLEU) | Δ vs unseg [95% CI] |",
              "|---|---|---|---|---|---|---|---|"]
        for name, c in blob["conditions"].items():
            pv = c.get("paired_vs_unseg")
            d = (f"{pv['delta']:+.2f} [{pv['ci95'][0]:+.2f}, {pv['ci95'][1]:+.2f}]"
                 if pv else "—")
            ms = "—" if c.get("laal_ms") is None else f"{c['laal_ms']:.0f}"
            L.append(f"| {name} | {c['k']:.2f} | {ms} | {c['laal_words']:.2f} | {c['bleu']:.2f} | "
                     f"{c['chrf2']:.2f} | {c.get('retention_bleu', 1.0):.4f} | {d} |")
        L.append("")
    L += ["## 언어 간 안정성 (retention = 조건 / 무분절)", "",
          "| 조건 | " + " | ".join(f"{t} BLEU" for t in report) + " | "
          + " | ".join(f"{t} chrF2" for t in report) + " | BLEU 폭 | chrF2 폭 |",
          "|---" * (2 * len(report) + 3) + "|"]
    names: list[str] = []
    for blob in report.values():                 # 타깃마다 조건이 다를 수 있다
        for name in blob["conditions"]:
            if name not in names:
                names.append(name)
    for name in names:
        if name == "unsegmented":
            continue
        cells = [report[t]["conditions"].get(name) for t in report]
        rb = [c.get("retention_bleu") if c else None for c in cells]
        rc = [c.get("retention_chrf2") if c else None for c in cells]
        # 비교군은 타깃별 자원이 있어야 나온다 — 없는 타깃은 폭 계산에서 뺀다.
        got_b = [x for x in rb if x is not None]
        got_c = [x for x in rc if x is not None]
        fmt = lambda xs: " | ".join("—" if x is None else f"{x:.4f}" for x in xs)
        span_b = f"{max(got_b) - min(got_b):.4f}" if len(got_b) > 1 else "—"
        span_c = f"{max(got_c) - min(got_c):.4f}" if len(got_c) > 1 else "—"
        L.append(f"| {name} | {fmt(rb)} | {fmt(rc)} | {span_b} | {span_c} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
