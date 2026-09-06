#!/usr/bin/env python3
"""`label_covost2.py` 산출 라벨을 `prompt_eval/<label>_<split>.json` 형식으로 바꾼다.

**왜 변환이 필요한가.** 비교·작도 파이프라인(`bleu_eval` → `baselines/plot_tradeoff`)은
분절 출처를 `runs/<run-id>/prompt_eval/<label>_<split>.json` 한 곳에서만 읽는다. 그 파일은
원래 `eval_prompt.py` 가 만드는데, 그쪽은 분절을 **다시 호출하고** 번역·COMET/NLI 채점까지
한 벌 돌린다. 라벨이 이미 있는 상태에서 그걸 다시 돌리는 것은 API 비용을 두 번 내는 일이다
(CoVoST2 en→X 3,000문장 실측 $7.20).

그래서 이 스크립트는 **절단만** 재사용한다. `bleu_eval.build_conditions` 가 `by_T` 에서
실제로 읽는 것은 `seg_text` 와 `pieces_src` 둘뿐이므로(`bleu_eval.py:148-150`), 그 둘을
`pipeline.truncate` / `pipeline.split_segments` 로 만든다 — 루프가 쓰는 것과 같은 함수라
절단 규칙이 갈라지지 않는다.

**채점에 쓰이는 필드는 만들지 않는다.** `valid` / `full_trans` 는 `eval_prompt` 가 채우는
자리인데 `bleu_eval` 은 안 읽는다. 여기서 그럴듯한 값을 지어 넣으면 나중에 이 파일을
`eval_prompt` 산출물로 착각할 수 있으므로 `null` 로 둔다.

    python core/meaning_segmentator/tools/covost2_label/to_prompt_eval.py \
        --labels core/meaning_segmentator/experiment/artifacts/covost2/n3000/labels/covost2_n3000_run13.jsonl \
        --run-id covost2/n3000 --label auto_run13 \
        --min-gap 3 --t-grid 4 6 8 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.meaning_segmentator.autoseg.runtime import pipeline as P
from core.meaning_segmentator.autoseg.loop import target_is_spaced


def main() -> int:
    ap = argparse.ArgumentParser(description="라벨 jsonl -> prompt_eval json")
    ap.add_argument("--labels", required=True, help="label_covost2.py 산출 jsonl")
    ap.add_argument("--run-id", required=True, help="runs/ 이하 경로")
    ap.add_argument("--label", required=True, help="산출 파일 이름에 쓸 라벨")
    ap.add_argument("--split", default="test")
    ap.add_argument("--t-grid", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--min-gap", type=int, required=True,
                    help="라벨링에 쓴 값과 **같아야 한다** — 다르면 절단 규칙이 갈린다")
    ap.add_argument("--spaced", default=None, choices=["yes", "no"],
                    help="기본은 라벨의 src_lang 으로 판정")
    ap.add_argument("--prompt-file", default=None, help="기록용 (채점에는 안 쓰임)")
    a = ap.parse_args()

    rows_in = [json.loads(l) for l in open(a.labels, encoding="utf-8") if l.strip()]
    if not rows_in:
        print(f"라벨이 비어 있습니다: {a.labels}", file=sys.stderr)
        return 2

    # `label_covost2` 와 같은 규칙 — `zh-CN` 의 지역 접미사를 떼고 판정한다.
    if a.spaced is None:
        codes = {(r.get("src_lang") or "").split("-")[0].lower() for r in rows_in}
        if len(codes) != 1:
            print(f"소스 언어가 섞여 있습니다: {sorted(codes)}. --spaced 로 지정하세요",
                  file=sys.stderr)
            return 2
        spaced = target_is_spaced(codes.pop())
    else:
        spaced = a.spaced == "yes"

    t_grid = sorted(set(a.t_grid))
    rows_out = []
    for r in rows_in:
        # **`src_text` 를 쓴다 — `src_text_raw` 가 아니다.** 라벨러가 CSV 이스케이프를
        # 푼 뒤의 문장에 태그를 달았으므로, 원본을 쓰면 태그를 떼도 문장이 안 맞는다.
        text, seg = r["src_text"], r["seg_text"]
        by_T = {}
        for T in t_grid:
            cut, missing = P.truncate(seg, T, spaced, a.min_gap)
            pieces = P.split_segments(cut) or [text]
            by_T[str(T)] = {"seg_text": cut, "k": len(pieces),
                            "missing_boundaries": missing, "pieces_src": pieces}
        rows_out.append({"id": r["utt_id"], "text": text, "seg_text": seg,
                         "valid": None, "full_trans": None, "by_T": by_T})

    out_dir = (Path(__file__).resolve().parents[2] / "core" / "meaning_segmentator"
               / "runs" / a.run_id / "prompt_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.label}_{a.split}.json"
    out_path.write_text(json.dumps({
        "prompt_file": a.prompt_file,
        "source": f"to_prompt_eval.py <- {a.labels}",
        "split": a.split,
        "t_grid": t_grid,
        "min_gap": a.min_gap,
        "src_spaced": spaced,
        "rows": rows_out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ks = [r["by_T"][str(t_grid[0])]["k"] for r in rows_out]
    miss = sum(r["by_T"][str(t_grid[0])]["missing_boundaries"] for r in rows_out)
    print(f"{len(rows_out)}문장 -> {out_path}")
    print(f"  소스 띄어쓰기={spaced}  min_gap={a.min_gap}  격자={t_grid}")
    print(f"  T={t_grid[0]} 에서 평균 조각수 {sum(ks)/len(ks):.2f}, "
          f"무분절 {sum(1 for k in ks if k == 1)}/{len(ks)}, 예산 미달 경계 {miss}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
