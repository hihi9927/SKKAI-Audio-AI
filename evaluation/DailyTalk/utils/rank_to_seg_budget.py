#!/usr/bin/env python3
"""랭크 라벨(<SEG:n>) → 학습용 평문 <SEG>. 랭크 컷이 아니라 **조각 크기 예산 T** 로 자른다.

`rank_to_seg.py` 는 `rank <= T` 인 태그만 남긴다 — 상위 몇 개를 남길지의 문제라
문장 길이를 안 본다. 그래서 만들어진 라벨이 autoseg 의 T 축 위에 점을 갖지 못한다
(문장당 경계를 전부 쓰므로 어떤 T 로도 재현이 안 된다).

이 스크립트는 `pipeline.truncate` 를 그대로 쓴다 — 평가·논문 곡선과 같은 절단기다:
문장마다 `k = max(1, round(단위수 / T))` 를 구하고 순위 상위 `k-1` 개만 남긴다.
`--min-gap` 을 주면 이미 고른 경계·문장 양끝과 그 단위 수 미만인 자리는 건너뛴다.

    python evaluation/DailyTalk/utils/rank_to_seg_budget.py -T 5
    python evaluation/DailyTalk/utils/rank_to_seg_budget.py -T 4 --min-gap 2
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.meaning_segmentator.autoseg.runtime import pipeline  # noqa: E402

TAG = re.compile(r"\s*<SEG:?\d*>\s*")


def to_plain(seg_text: str) -> str:
    """`<SEG:n>` → `<SEG>`. 순위 정보를 버린다 (학습 라벨에는 순위가 없다)."""
    return re.sub(r"\s+", " ", TAG.sub(" <SEG> ", seg_text)).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="evaluation/DailyTalk/transcribe/new_seg_all.json")
    ap.add_argument("--output", default=None, help="기본: 입력 옆에 _T{T}[_mg{g}].json")
    ap.add_argument("-T", type=int, required=True, help="목표 조각 크기 (어절)")
    ap.add_argument("--min-gap", type=int, default=0, help="경계 간 최소 어절 수. 0 = 끔")
    ap.add_argument("--stats-only", action="store_true", help="파일을 쓰지 않는다")
    a = ap.parse_args()

    src = json.loads(Path(a.input).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    units: list[int] = []
    ks: list[int] = []
    pieces: list[int] = []
    missing_total = 0

    for gk, v in src.items():
        rows = []
        for e in v["data"]:
            raw = (e.get("seg_text") or e["text"]).strip()
            cut, missing = pipeline.truncate(raw, a.T, spaced=True, min_gap=a.min_gap)
            missing_total += missing
            plain = to_plain(cut)
            parts = [p for p in plain.split("<SEG>") if p.strip()]
            units.append(len(TAG.sub(" ", raw).split()))
            ks.append(len(parts))
            pieces += [len(p.split()) for p in parts]
            rows.append({"file": e["file"], "text": e["text"], "seg_text": plain})
        out[gk] = {"data": rows}

    n = len(units)
    print(f"T={a.T} min_gap={a.min_gap}  문장 {n}")
    print(f"  어절: 중앙 {st.median(units):.0f} 평균 {sum(units) / n:.2f}")
    print(f"  조각 수 k: 평균 {sum(ks) / n:.3f} 중앙 {st.median(ks):.0f} "
          f"무분절 {sum(1 for k in ks if k <= 1)} ({sum(1 for k in ks if k <= 1) / n:.1%})")
    print(f"  조각 길이: 평균 {sum(pieces) / len(pieces):.2f} 중앙 {st.median(pieces):.0f} "
          f"1어절 {sum(1 for p in pieces if p == 1)} ({sum(1 for p in pieces if p == 1) / len(pieces):.1%})")
    print(f"  missing_boundaries 합 {missing_total}")

    if a.stats_only:
        return 0
    suffix = f"_T{a.T}" + (f"_mg{a.min_gap}" if a.min_gap else "")
    dst = Path(a.output or str(Path(a.input).with_suffix("")) + suffix + ".json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
