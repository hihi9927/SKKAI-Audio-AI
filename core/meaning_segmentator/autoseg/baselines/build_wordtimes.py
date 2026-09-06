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
import json
import sys
import time

from ..paths import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

from core.meaning_segmentator.autoseg.baselines import datasets as _ds  # noqa: E402
from core.meaning_segmentator.autoseg.baselines.align_audio import Aligner  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="clean500 어절 타임스탬프 추출")
    p.add_argument("--tag", default="clean500")
    p.add_argument("--dataset", default="fleurs", choices=["fleurs", "covost2"])
    p.add_argument("--ref-tgt", default="de", help="매니페스트를 고르기 위한 타깃 (소스는 동일)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    jobs = _ds.alignment_jobs(args.dataset, args.tag, args.ref_tgt, args.limit)
    al = Aligner(device=args.device)

    out: dict[str, dict] = {}
    fails: list[str] = []
    t0 = time.time()
    for i, (key, text, wav, dur_ms) in enumerate(jobs):
        try:
            ends = al.word_end_times(wav, text)
        except Exception as exc:                       # noqa: BLE001
            fails.append(f"{key}:{type(exc).__name__}")
            continue
        if ends is None:
            fails.append(f"{key}:align")
            continue
        out[key] = {"wav": wav.name, "dur_ms": dur_ms,
                    "word_end_ms": [round(e * 1000, 1) for e in ends]}
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(jobs)}  {el:.0f}s  "
                  f"ETA {el / (i + 1) * (len(jobs) - i - 1) / 60:.1f}m", flush=True)

    dest = _ds.get(args.dataset).wordtimes_path(args.tag, "ctc")
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"정렬 성공 {len(out)}/{len(jobs)}, 실패 {len(fails)}")
    if fails:
        print("  실패 예:", fails[:8])
    print(f"→ {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
