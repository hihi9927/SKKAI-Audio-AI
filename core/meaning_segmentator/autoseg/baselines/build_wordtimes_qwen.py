"""Qwen3-ForcedAligner 로 같은 어절 타임스탬프를 뽑는다 — CTC 정렬과 대조용.

CTC(`build_wordtimes.py`)와 **같은 녹음·같은 텍스트**를 써야 차이가 정렬기에서만 나온다.
산출 형식도 같다: {talk_id: {"wav","split","dur_ms","word_end_ms"}}

Qwen 은 어절이 아니라 자체 토큰 단위로 span 을 내므로, 문자 오프셋으로 어절에 되매핑한다.
**반드시 `.venv`(qwen_asr 설치본)로 실행할 것** — `.venv-autoseg` 에는 qwen_asr 가 없다.

    .venv/bin/python -m core.meaning_segmentator.autoseg.baselines.build_wordtimes_qwen
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import torch

from ..paths import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

from core.meaning_segmentator.autoseg.baselines import datasets as _ds  # noqa: E402

ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
_WS = re.compile(r"\s+")


def to_word_ends(items, text: str, dur_s: float) -> list[float]:
    """Qwen span 들을 어절 종료 시각으로 접는다.

    span.text 를 원문에서 순차 탐색해 문자 위치를 찾고, 그 위치가 속한 어절의 끝 시각을
    갱신한다. 못 찾은 span 은 건너뛴다 (구두점 제거 등으로 원문과 안 맞을 수 있다).
    """
    words = _WS.split(text.strip())
    w_end, pos = [], 0
    for w in words:
        j = text.index(w, pos)
        pos = j + len(w)
        w_end.append(pos)

    ends = [0.0] * len(words)
    cur = 0
    for it in items:
        t = (it.text or "").strip()
        if not t:
            continue
        j = text.find(t, cur)
        if j < 0:
            j = text.lower().find(t.lower(), cur)
        if j < 0:
            continue
        cur = j + len(t)
        for i, e in enumerate(w_end):
            if e >= cur:
                ends[i] = max(ends[i], float(it.end_time))
                break
    run = 0.0
    for i, e in enumerate(ends):
        run = max(run, e)
        ends[i] = run
    ends[-1] = max(ends[-1], dur_s * 0.999)
    return ends


def main() -> int:
    p = argparse.ArgumentParser(description="Qwen 강제정렬 어절 타임스탬프")
    p.add_argument("--tag", default="clean500")
    p.add_argument("--dataset", default="fleurs", choices=["fleurs", "covost2"])
    p.add_argument("--ref-tgt", default="de", help="매니페스트를 고르기 위한 타깃")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    jobs = _ds.alignment_jobs(args.dataset, args.tag, args.ref_tgt, args.limit)
    al = Qwen3ForcedAligner.from_pretrained(ALIGNER, device_map="cuda:0",
                                            dtype=torch.bfloat16)

    out: dict[str, dict] = {}
    fails: list[str] = []
    t0 = time.time()
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i: i + args.batch]
        try:
            res = al.align(audio=[str(c[2]) for c in chunk],
                           text=[c[1] for c in chunk],
                           language=["English"] * len(chunk))
        except Exception as exc:                       # noqa: BLE001
            fails += [f"{c[0]}:{type(exc).__name__}" for c in chunk]
            continue
        for c, r in zip(chunk, res):
            key, text, wav, dur_ms = c
            try:
                ends = to_word_ends(list(r), text, dur_ms / 1000)
            except Exception as exc:                   # noqa: BLE001
                fails.append(f"{key}:{type(exc).__name__}")
                continue
            out[key] = {"wav": wav.name, "dur_ms": dur_ms,
                        "word_end_ms": [round(e * 1000, 1) for e in ends]}
        done = i + len(chunk)
        if done % 80 == 0 or done >= len(jobs):
            el = time.time() - t0
            print(f"  {done}/{len(jobs)}  {el:.0f}s  "
                  f"ETA {el / done * (len(jobs) - done) / 60:.1f}m", flush=True)

    dest = _ds.get(args.dataset).wordtimes_path(args.tag, "qwen")
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"정렬 성공 {len(out)}/{len(jobs)}, 실패 {len(fails)}")
    if fails:
        print("  실패 예:", fails[:8])
    print(f"→ {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
