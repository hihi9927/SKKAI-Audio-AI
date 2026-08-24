"""로컬 모델(ollama)로 매니페스트 원문에 순위 <SEG:n> 를 단다 — OpenAI 키 없이.

  python -m core.meaning_segmentator.autoseg.label_local \
      --manifest evaluation/ast/manifests/covost2_en-de_sample300_src.jsonl \
      --prompt core/meaning_segmentator/runs/en-multi/run06/best_prompt.txt \
      --out core/meaning_segmentator/runs/covost2/sample300/prompt_eval/auto_best_test.json \
      --cache core/meaning_segmentator/runs/covost2/sample300/cache/segment_ollama.json \
      --model llama3.3:70b --workers 8


산출은 `prompt_eval/auto_best_test.json` 스키마의 부분집합이다 — bleu_eval /
comet_score 가 읽는 필드(id, text, seg_text, by_T[T].{seg_text, pieces_src})를 전부 채운다.
분절 프롬프트·검증기(validate)·정규화(normalize_tags)·절단기(truncate)는 autoseg 원본을
그대로 쓴다. 달라진 것은 호출 모델뿐이다 (gpt-5-mini → ollama llama3.3:70b).

로컬 모델용으로 추가한 것은 결정론적 복구 한 층뿐이다 (`asg.repair`) — 코드펜스/라벨
제거, 랭크 재번호, 원문 훼손 시 앵커 복원. 경계 위치는 만들지도 옮기지도 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import repair_local as R
from .ollama_gateway import Gateway
from .pipeline import (JsonCache, chunk_budget, normalize_tags, split_segments,
                       truncate, unit_count, validate)

SPACED = True
TRAILING_PUNCT = "!')-,./:;?’"      # en-multi/clean500 measured_profile 과 같은 집합
SEG_MAX_TOKENS = 2048

RETRY_TMPL = (
    "{text}\n\n"
    "[Your previous answer violated the output rules: {detail}]\n"
    "[Previous answer: {prev}]\n"
    "[Re-emit the ORIGINAL text above, character for character — keep every quotation "
    "mark — with only <SEG:n> tags inserted, numbered 1..N by confidence with no gaps or "
    "duplicates. Do not shorten, rewrite, or add anything. If the violation says too few "
    "tags, add boundaries at the next-safest positions and rank them last — marking one is "
    "free, withholding it is not. If it says a piece is too short, move that boundary so "
    "every piece has at least 3 words.]"
)


def coverage_need(text: str, min_t: int, spaced: bool, min_gap: int) -> int:
    need = max(0, chunk_budget(text, min_t, spaced) - 1)
    if min_gap > 0:
        need = min(need, max(0, unit_count(text, spaced) // min_gap - 1))
    return need


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--model", default="llama3.3:70b")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3, help="검증 실패 시 LLM 재호출 횟수")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--t-grid", type=int, nargs="+", default=[4, 6, 8, 12])
    p.add_argument("--min-gap", type=int, default=3)
    p.add_argument("--candidate-t", type=int, default=4)
    args = p.parse_args()

    rows_in = [json.loads(l) for l in
               Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows_in = rows_in[args.offset:]
    if args.limit:
        rows_in = rows_in[:args.limit]
    prompt = Path(args.prompt).read_text(encoding="utf-8")

    gw = Gateway(model=args.model, num_ctx=args.num_ctx)
    cache = JsonCache(Path(args.cache))
    prompt_hash = JsonCache.key(prompt)

    def check(text: str, seg: str):
        return validate("", text, seg, SPACED, TRAILING_PUNCT, True,
                        coverage_need(text, args.candidate_t, SPACED, args.min_gap),
                        None, args.min_gap)

    def one(text: str) -> tuple[str, bool, list]:
        k = JsonCache.key("ollama-seg1", prompt_hash, gw.model, str(args.min_gap),
                          str(args.candidate_t), text)
        hit = cache.get(k)
        if hit is not None and (hit[0] or "").strip():
            return hit[0], hit[1], hit[2]

        # **재시도는 최선을 고른다, 마지막을 쓰지 않는다.** 로컬 모델은 재시도에서
        # 원문을 더 크게 고쳐 쓰는 일이 있고, 그러면 앵커 복원이 실패해 무분절로
        # 떨어진다 — 그 결과로 이전 시도의 멀쩡한 경계까지 잃는다 (24문장 스모크에서
        # 통과율 0.54 → 0.25, 무분절 0% → 20.8%). 위반 수가 적은 쪽을, 같으면 앞선
        # 시도를 남긴다.
        first_ok, first_viol = True, []
        best, best_cost = None, None
        user = text
        for attempt in range(args.retries + 1):
            raw = gw.chat(system=prompt, user=user, max_tokens=SEG_MAX_TOKENS,
                          purpose="segment" if attempt == 0 else "segment_retry")
            seg = normalize_tags(R.repair(text, raw), SPACED, TRAILING_PUNCT)
            vs = check(text, seg)
            if attempt == 0:
                first_ok = not vs
                first_viol = [{"rule": v.rule, "detail": v.detail} for v in vs]
            # 태그 0개(복원 실패)는 위반 수와 무관하게 최하위로 민다
            cost = (len(vs), 1 if "<SEG:" not in seg else 0)
            if best_cost is None or cost < best_cost:
                best, best_cost = seg, cost
            if not vs:
                break
            user = RETRY_TMPL.format(
                text=text, prev=seg,
                detail="; ".join(f"{v.rule}: {v.detail}" for v in vs))
        seg = best if (best or "").strip() else text
        cache.put(k, [seg, first_ok, first_viol])
        return seg, first_ok, first_viol

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(one, [r["text"] for r in rows_in]))
    cache.flush()
    seg_texts = [r[0] for r in results]
    first_pass = [r[1] for r in results]
    first_viol = [v for r in results for v in r[2]]

    violations, valid_flags = [], []
    for r, seg in zip(rows_in, seg_texts):
        vs = check(r["text"], seg)
        valid_flags.append(not vs)
        violations.extend({"id": r["id"], "rule": v.rule, "detail": v.detail,
                           "text": r["text"], "seg_text": seg} for v in vs)

    out_rows = []
    for r, seg, ok in zip(rows_in, seg_texts, valid_flags):
        cell = {}
        for T in args.t_grid:
            cut, missing = truncate(seg, T, SPACED, args.min_gap)
            pieces = split_segments(cut) or [r["text"]]
            cell[str(T)] = {"seg_text": cut, "k": len(pieces),
                            "missing_boundaries": missing, "pieces_src": pieces}
        out_rows.append({"id": r["id"], "text": r["text"], "seg_text": seg,
                         "valid": ok, "full_trans": None, "by_T": cell})

    n = len(out_rows)
    by_T_stats = {str(T): {
        "target_chunk_words": T, "n": n,
        "chunks_per_sentence": round(
            sum(len(x["by_T"][str(T)]["pieces_src"]) for x in out_rows) / n, 4),
        "missing_boundaries": round(
            sum(x["by_T"][str(T)]["missing_boundaries"] for x in out_rows) / n, 4),
        "unsegmented_rate": round(
            sum(1 for x in out_rows if len(x["by_T"][str(T)]["pieces_src"]) == 1) / n, 4),
    } for T in args.t_grid}

    blob = {
        "prompt_file": args.prompt,
        "split": "test",
        "t_grid": args.t_grid,
        "tgt_lang": "German",
        "min_gap": args.min_gap,
        "candidate_t": args.candidate_t,
        "priority_depth": None,
        "batch_size": 1,
        "require_priority": True,
        "segmenter": f"ollama:{args.model}",
        "adequacy_backend": None,
        "consistency_backend": None,
        "contradiction_backend": None,
        "translator": None,
        "metrics": {
            "n": n,
            "format_pass_rate": round(sum(valid_flags) / n, 4),
            "format_pass_rate_no_retry": round(sum(first_pass) / n, 4),
            "violation_counts": dict(Counter(v["rule"] for v in violations)),
            "first_pass_violation_counts": dict(Counter(v["rule"] for v in first_viol)),
            "by_T": by_T_stats,
        },
        "violations": violations,
        "rows": out_rows,
        "usage": {"calls": gw.calls, "retries": gw.retries, "empty": gw.empty},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    gw.close()

    print(f"n={n}  포맷 통과율 {blob['metrics']['format_pass_rate']:.4f} "
          f"(재시도 없이 {blob['metrics']['format_pass_rate_no_retry']:.4f})  "
          f"위반 {len(violations)}건  호출 {gw.calls} (재시도 {gw.retries}, 빈출력 {gw.empty})")
    print("  최종 위반:", blob["metrics"]["violation_counts"] or "없음")
    print("  1차 위반:", blob["metrics"]["first_pass_violation_counts"] or "없음")
    for T in args.t_grid:
        s = by_T_stats[str(T)]
        print(f"  T={T:<3} k {s['chunks_per_sentence']:.2f}  부족경계 "
              f"{s['missing_boundaries']:.2f}  무분절 {s['unsegmented_rate']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
