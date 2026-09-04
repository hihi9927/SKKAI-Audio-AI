#!/usr/bin/env python3
"""train/val/test 분할 + 각 셋의 절반을 미완결(partial) 대상으로 지정.

출력:
  split_assign.json          {file: {"split": train|val|test, "mode": full|partial}}
  partial_input.json         generate_split_data.py 입력용 (partial 지정분만)
"""
import argparse, json, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="evaluation/DailyTalk/transcribe/new_seg_all_t2.json")
    ap.add_argument("--assign", default="evaluation/DailyTalk/transcribe/split_assign.json")
    ap.add_argument("--partial-input", default="evaluation/DailyTalk/transcribe/partial_input.json")
    ap.add_argument("--audio-dir", default="Qwen3-ASR/finetuning/data/DailyTalk/audio")
    ap.add_argument("--val",  type=int, default=1500)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--partial-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src = json.loads(Path(a.input).read_text(encoding="utf-8"))
    have = {p.stem for p in Path(a.audio_dir).glob("*.wav")}
    items = [(gk, e) for gk in src for e in src[gk]["data"] if e["file"] in have]
    missing = sum(len(src[gk]["data"]) for gk in src) - len(items)
    if missing:
        print(f"경고: 오디오 없는 {missing}건 제외")

    rng = random.Random(a.seed)
    rng.shuffle(items)
    test  = items[:a.test]
    val   = items[a.test:a.test + a.val]
    train = items[a.test + a.val:]
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    assign = {}
    partial_groups = {}
    for name, group in (("train", train), ("val", val), ("test", test)):
        idx = list(range(len(group)))
        rng.shuffle(idx)
        n_part = int(len(group) * a.partial_ratio)
        part = set(idx[:n_part])
        for i, (gk, e) in enumerate(group):
            mode = "partial" if i in part else "full"
            assign[e["file"]] = {"split": name, "mode": mode, "group": gk}
            if mode == "partial":
                partial_groups.setdefault(gk, {"data": []})["data"].append(e)
        print(f"  {name}: full {len(group)-n_part} / partial {n_part}")

    Path(a.assign).write_text(json.dumps(assign, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(a.partial_input).write_text(json.dumps(partial_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    n_part_total = sum(len(v["data"]) for v in partial_groups.values())
    print(f"저장: {a.assign} / {a.partial_input} (partial 대상 {n_part_total}건)")

if __name__ == "__main__":
    main()
