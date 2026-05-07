"""
core/cif/data/dailytalk_10000.jsonl → train.jsonl (9000) / val.jsonl (500) / test.jsonl (500)

실행:
  python core/cif/utils/split_dataset.py
"""

import json
import random
from pathlib import Path

SEED = 42
N_TRAIN = 9000
N_VAL = 500
N_TEST = 500

def main():
    src = Path(__file__).resolve().parents[1] / "data" / "dailytalk_10000.jsonl"
    out_dir = Path(__file__).resolve().parents[1] / "data"

    with open(src) as f:
        records = [json.loads(l) for l in f if l.strip()]

    total = len(records)
    print(f"총 {total}개 샘플")

    if total < N_TRAIN + N_VAL + N_TEST:
        raise ValueError(
            f"샘플 수 부족: {total} < {N_TRAIN + N_VAL + N_TEST}"
        )

    rng = random.Random(SEED)
    rng.shuffle(records)

    splits = {
        "train.jsonl": records[:N_TRAIN],
        "val.jsonl":   records[N_TRAIN: N_TRAIN + N_VAL],
        "test.jsonl":  records[N_TRAIN + N_VAL: N_TRAIN + N_VAL + N_TEST],
    }

    for fname, subset in splits.items():
        path = out_dir / fname
        with open(path, "w") as f:
            for rec in subset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {fname}: {len(subset)}개 → {path}")

if __name__ == "__main__":
    main()
