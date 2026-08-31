#!/usr/bin/env python3
"""gpt_seg_en.py 출력(<SEG:n> 랭크 라벨)을 label_covost2.py 와 같은 잣대로 채점한다.

두 경로를 한 표에 올리려면 지표 정의가 같아야 한다 — coverage_need / min_gap /
t_floor 기본값을 label_covost2.py 와 맞춰 두었다.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.meaning_segmentator.autoseg import pipeline as P

TAG = re.compile(r"\s*<SEG:(\d+)>\s*")


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", TAG.sub(" ", s)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True, help="gpt_seg_en.py 출력 JSON")
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--t-floor", type=int, default=4)
    ap.add_argument("--dump", type=int, default=0, help="예시 n개 출력")
    a = ap.parse_args()

    raw = json.loads(Path(a.seg).read_text(encoding="utf-8"))
    rows = [e for gk in raw for e in raw[gk]["data"]]

    n = done = preserved = cov = rank_ok = 0
    bs: list[int] = []
    reqs: list[int] = []
    for e in rows:
        n += 1
        seg = e.get("seg_text")
        if not seg:
            continue
        done += 1
        need = P.coverage_need(e["text"], a.t_floor, True, a.min_gap)
        nums = [int(x) for x in TAG.findall(seg)]
        bs.append(len(nums))
        reqs.append(need)
        if strip_tags(seg) == re.sub(r"\s+", " ", e["text"]).strip():
            preserved += 1
        if len(nums) >= need:
            cov += 1
        if sorted(nums) == list(range(1, len(nums) + 1)):
            rank_ok += 1

    d = max(done, 1)
    print(json.dumps({
        "seg_file": a.seg,
        "n": n,
        "labeled": done,
        "text_preserved": round(preserved / d, 3),
        "rank_wellformed": round(rank_ok / d, 3),
        "coverage_met": round(cov / d, 3),
        "mean_boundaries": round(sum(bs) / max(len(bs), 1), 2),
        "mean_required": round(sum(reqs) / max(len(reqs), 1), 2),
    }, ensure_ascii=False, indent=1))

    for e in rows[:a.dump]:
        print(" ", (e.get("seg_text") or "(없음)")[:140])


if __name__ == "__main__":
    main()
