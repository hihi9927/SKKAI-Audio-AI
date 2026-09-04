#!/usr/bin/env python3
"""랭크 라벨(<SEG:n>) → 학습용 평문 <SEG> 변환. rank<=T만 유지."""
import argparse, json, re
from pathlib import Path

TAG = re.compile(r"\s*<SEG:(\d+)>\s*")

def convert(seg_text: str, T: int) -> str:
    parts = TAG.split(seg_text)
    texts, nums = parts[0::2], [int(x) for x in parts[1::2]]
    out = texts[0].strip()
    for t, n in zip(texts[1:], nums):
        out += (" <SEG> " if n <= T else " ") + t.strip()
    return re.sub(r"\s+", " ", out).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="evaluation/DailyTalk/transcribe/new_seg_all.json")
    ap.add_argument("--output", default="evaluation/DailyTalk/transcribe/new_seg_all_t2.json")
    ap.add_argument("-T", type=int, default=2, help="유지할 최대 랭크 (기본 2)")
    a = ap.parse_args()

    src = json.loads(Path(a.input).read_text(encoding="utf-8"))
    out = {}; n = kept = dropped = notag = 0
    for gk, v in src.items():
        rows = []
        for e in v["data"]:
            st = e.get("seg_text") or e["text"]
            nums = [int(x) for x in re.findall(r"<SEG:(\d+)>", st)]
            kept += sum(1 for x in nums if x <= a.T)
            dropped += sum(1 for x in nums if x > a.T)
            conv = convert(st, a.T)
            if "<SEG>" not in conv: notag += 1
            rows.append({"file": e["file"], "text": e["text"], "seg_text": conv})
            n += 1
        out[gk] = {"data": rows}
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"변환 {n}문장 | SEG 유지 {kept} 버림 {dropped} | 분절없는 문장 {notag}")
    print(f"저장: {a.output}")

if __name__ == "__main__":
    main()
