#!/usr/bin/env python3
"""최종 train/val/test jsonl 조립.

- full    : audio/{file}.wav          + seg_text + 끝-SEG
- partial : split_audio/split_{file}.wav + partial_seg (끝-SEG 없음)
- partial 생성 실패분은 full로 폴백
"""
import argparse, json
from pathlib import Path

PREFIX = "language English<asr_text>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg",     default="evaluation/DailyTalk/transcribe/new_seg_all_t2.json")
    ap.add_argument("--assign",  default="evaluation/DailyTalk/transcribe/split_assign.json")
    ap.add_argument("--partial", default="evaluation/DailyTalk/transcribe/partial_all.json")
    ap.add_argument("--audio-dir", default="Qwen3-ASR/finetuning/data/DailyTalk/audio")
    ap.add_argument("--split-dir", default="Qwen3-ASR/finetuning/data/DailyTalk/split_audio")
    ap.add_argument("--outdir",  default="Qwen3-ASR/finetuning/data/DailyTalk")
    a = ap.parse_args()

    seg = json.loads(Path(a.seg).read_text(encoding="utf-8"))
    segmap = {e["file"]: e for gk in seg for e in seg[gk]["data"]}
    assign = json.loads(Path(a.assign).read_text(encoding="utf-8"))
    pmap = {}
    if Path(a.partial).exists():
        pj = json.loads(Path(a.partial).read_text(encoding="utf-8"))
        pmap = {e["original_file"]: e for gk in pj for e in pj[gk]["data"]}
    audio_dir = Path(a.audio_dir).resolve()
    split_dir = Path(a.split_dir).resolve()

    buckets = {"train": [], "val": [], "test": []}
    stat = {"full": 0, "partial": 0, "fallback": 0}
    for f, meta in assign.items():
        e = segmap.get(f)
        if e is None:
            continue
        if meta["mode"] == "partial" and f in pmap:
            p = pmap[f]
            wav = split_dir / f"split_{f}.wav"
            if wav.exists():
                buckets[meta["split"]].append(
                    {"audio": str(wav), "text": PREFIX + p["seg_text"].strip()})
                stat["partial"] += 1
                continue
            stat["fallback"] += 1
        elif meta["mode"] == "partial":
            stat["fallback"] += 1
        else:
            stat["full"] += 1
        t = e["seg_text"].rstrip()
        if not t.endswith("<SEG>"):
            t += " <SEG>"
        buckets[meta["split"]].append(
            {"audio": str(audio_dir / f"{f}.wav"), "text": PREFIX + t})

    out = Path(a.outdir)
    for name, rows in buckets.items():
        p = out / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}.jsonl  {len(rows)}줄  → {p}")
    print(f"full {stat['full']} | partial {stat['partial']} | partial실패→full 폴백 {stat['fallback']}")

if __name__ == "__main__":
    main()
