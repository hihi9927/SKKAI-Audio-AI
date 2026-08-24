"""CoVoST2 평가 표본 매니페스트를 만든다 — clean500 과 같은 층화 규칙.

전체 9,424문장에서 그대로 뽑으면 두 가지가 실험을 망친다:

1. **이어붙인 클립.** `n_clips ≥ 2` 가 4,513문장(48%)인데, 서로 무관한 Common Voice
   문장을 이은 것이라 그 이음매가 자명한 경계다. 정책 간 변별력이 죽으므로 `n_clips == 1`
   만 쓴다.
2. **짧은 문장.** 어절 중앙이 13 (FLEURS 20)이라 `min_gap` 하에서 경계를 못 놓는 문장이
   많다. `--min-words` 로 하한을 건다.

표본은 **길이 3분위 × 구두점 유무 층화 정렬의 prefix** 로 뽑는다 (`data.split_data` 와 같은
규칙). prefix 로 뽑아야 나중에 표본을 늘릴 때 분절·번역 캐시가 그대로 산다.

    python -m core.meaning_segmentator.autoseg.baselines.make_covost_sample \\
        --n 300 --min-words 12 --tag sample300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))

MAN = _REPO_ROOT / "evaluation" / "ast" / "manifests"
_PUNCT = set(",;:!?—–")


def main() -> int:
    p = argparse.ArgumentParser(description="CoVoST2 표본 매니페스트")
    p.add_argument("--src", default="covost2_en-de_spk.jsonl")
    p.add_argument("--tag", default="sample300")
    p.add_argument("--tgt", default="de")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--min-words", type=int, default=12)
    p.add_argument("--max-clips", type=int, default=1,
                   help="n_clips 상한. 1 = 이어붙이지 않은 것만")
    p.add_argument("--seed", type=int, default=20260821)
    args = p.parse_args()

    rows = []
    with (MAN / args.src).open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("n_clips", 1) > args.max_clips:
                continue
            w = len(e["src_text"].split())
            if w < args.min_words:
                continue
            if not Path(e["wav"]).exists():
                continue
            rows.append((w, e))
    if len(rows) < args.n:
        raise SystemExit(f"조건 만족 {len(rows)}문장 — {args.n}개를 못 뽑는다")

    # 길이 3분위 × 문장내 구두점 유무로 층을 만들고, 층 안에서 섞은 뒤 라운드로빈.
    ws = sorted(w for w, _ in rows)
    q1, q2 = ws[len(ws) // 3], ws[2 * len(ws) // 3]
    strata: dict[tuple[int, bool], list] = {}
    for w, e in rows:
        band = 0 if w <= q1 else (1 if w <= q2 else 2)
        inner = any(c in _PUNCT for c in e["src_text"][:-1])
        strata.setdefault((band, inner), []).append(e)
    rng = random.Random(args.seed)
    for v in strata.values():
        rng.shuffle(v)

    order, keys = [], sorted(strata)
    while any(strata[k] for k in keys):
        for k in keys:
            if strata[k]:
                order.append(strata[k].pop())
    picked = order[: args.n]

    dest = MAN / f"covost2_en-{args.tgt}_{args.tag}.jsonl"
    with dest.open("w", encoding="utf-8") as f:
        for e in picked:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    pw = [len(e["src_text"].split()) for e in picked]
    pw.sort()
    print(f"모집단 {len(rows)}문장 (n_clips≤{args.max_clips}, ≥{args.min_words}어절) "
          f"→ 표본 {len(picked)}")
    print(f"  어절 중앙 {pw[len(pw) // 2]}  평균 {sum(pw) / len(pw):.1f}  "
          f"최소 {pw[0]}  최대 {pw[-1]}")
    print(f"  총 길이 {sum(e['duration'] for e in picked) / 60:.1f}분")
    print(f"→ {dest}")

    # 분절기에 넣을 평문도 같이 낸다 — 라벨링을 다른 환경에서 할 수 있게.
    txt = MAN / f"covost2_en-{args.tgt}_{args.tag}_src.jsonl"
    with txt.open("w", encoding="utf-8") as f:
        for e in picked:
            f.write(json.dumps({"id": e["utt_id"], "text": e["src_text"]},
                               ensure_ascii=False) + "\n")
    print(f"→ {txt}  (분절 라벨링 입력용: id + text 만)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
