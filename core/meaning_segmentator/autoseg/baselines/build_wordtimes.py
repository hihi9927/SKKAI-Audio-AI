"""clean500 소스 문장의 **어절 종료 시각**을 강제정렬로 뽑아 저장한다.

산출: `evaluation/ast/manifests/fleurs_nway_en_<tag>_wordtimes.json`
      {talk_id: {"wav":…, "dur_ms":…, "word_end_ms":[…]}}

`bleu_eval` 이 이 값으로 조각 경계 시각을 잡아 ms LAAL 을 낸다 — 발화 내 균일 발화속도
보간을 대체한다. FLEURS 는 문장마다 화자별 녹음이 여럿이라 **길이 중앙값 녹음 하나**를
쓴다 (기존 duration 집계와 같은 규칙).

    python -m core.meaning_segmentator.autoseg.baselines.build_wordtimes --tag clean500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from core.meaning_segmentator.autoseg.baselines.align_audio import Aligner  # noqa: E402


def load_tsv(base: Path, split: str) -> dict[str, list[tuple[str, int]]]:
    """id → [(wav, 샘플수)]. TSV 는 따옴표 이스케이프가 없어 QUOTE_NONE 필수."""
    out: dict[str, list[tuple[str, int]]] = {}
    f = base / f"{split}.tsv"
    if not f.exists():
        return out
    with f.open(encoding="utf-8") as fh:
        for c in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(c) >= 6 and c[5].isdigit():
                out.setdefault(c[0], []).append((c[1], int(c[5])))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="clean500 어절 타임스탬프 추출")
    p.add_argument("--tag", default="clean500")
    p.add_argument("--lang-dir", default="en_us")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    base = Path.home() / "datasets" / "fleurs" / "data" / args.lang_dir
    man_dir = _REPO_ROOT / "evaluation" / "ast" / "manifests"
    entries = []
    with (man_dir / f"fleurs_nway_en-de_{args.tag}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            entries.append((str(e["talk_id"]), e["src_text"], e["fleurs_split"]))
    if args.limit:
        entries = entries[: args.limit]

    tsv = {s: load_tsv(base, s) for s in ("train", "dev", "test")}
    al = Aligner(device=args.device)

    out: dict[str, dict] = {}
    fails: list[str] = []
    t0 = time.time()
    for i, (tid, text, split) in enumerate(entries):
        recs = sorted(tsv.get(split, {}).get(tid, []), key=lambda r: r[1])
        if not recs:
            fails.append(f"{tid}:tsv")
            continue
        wav, n = recs[len(recs) // 2]                 # 길이 중앙값 녹음
        path = base / "audio" / split / wav
        if not path.exists():
            fails.append(f"{tid}:wav")
            continue
        try:
            ends = al.word_end_times(path, text)
        except Exception as exc:                       # noqa: BLE001
            fails.append(f"{tid}:{type(exc).__name__}")
            continue
        if ends is None:
            fails.append(f"{tid}:align")
            continue
        out[tid] = {"wav": wav, "split": split, "dur_ms": n / 16000 * 1000,
                    "word_end_ms": [round(e * 1000, 1) for e in ends]}
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(entries)}  {el:.0f}s  "
                  f"ETA {el / (i + 1) * (len(entries) - i - 1) / 60:.1f}m", flush=True)

    dest = man_dir / f"fleurs_nway_en_{args.tag}_wordtimes.json"
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"정렬 성공 {len(out)}/{len(entries)}, 실패 {len(fails)}")
    if fails:
        print("  실패 예:", fails[:8])
    print(f"→ {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
