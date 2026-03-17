#!/usr/bin/env python3
"""
Verify SmartTurn-track endpoint behavior offline.

Separated from verify_architecture_vad.py so we can iterate independently.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np

# Ensure project root is importable when this script is run directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

from evaluation.smartturn.turn_detector import SmartTurnV3Detector

SAMPLING_RATE = 16000
FRAME_SIZE = 512


def run_smartturn_trace(
    audio: np.ndarray,
    threshold_on: float,
    threshold_off: float,
    min_silence_ms: int,
) -> tuple[dict, str]:
    detector = SmartTurnV3Detector(
        sampling_rate=SAMPLING_RATE,
        threshold_on=threshold_on,
        threshold_off=threshold_off,
        min_silence_duration_ms=min_silence_ms,
    )
    detector.reset_states()

    prob_times_sec: list[float] = []
    probs: list[float] = []
    speech_start_samples: list[int] = []
    speech_end_samples: list[int] = []

    sample_cursor = 0
    while sample_cursor + FRAME_SIZE <= audio.size:
        frame = audio[sample_cursor : sample_cursor + FRAME_SIZE]
        event = detector.process(frame)
        t_sec = (sample_cursor + FRAME_SIZE) / SAMPLING_RATE
        prob_times_sec.append(float(t_sec))
        probs.append(float(event.prob or 0.0))

        if event.start is not None:
            speech_start_samples.append(int(event.start))
        if event.end is not None:
            speech_end_samples.append(int(event.end))

        sample_cursor += FRAME_SIZE

    trace = {
        "prob_times_sec": prob_times_sec,
        "probs": probs,
        "speech_start_samples": speech_start_samples,
        "speech_end_samples": speech_end_samples,
    }
    return trace, detector.score_backend


def main() -> int:
    p = argparse.ArgumentParser(description="SmartTurn architecture trace (offline)")
    p.add_argument("--audio", required=True, help="Path to input audio file")
    p.add_argument("--threshold-on", type=float, default=0.55)
    p.add_argument("--threshold-off", type=float, default=0.45)
    p.add_argument("--min-silence-ms", type=int, default=800)
    p.add_argument(
        "--output-json",
        default=str(
            Path("evaluation") / "smartturn" / "results" / "architecture_smartturn_trace.json"
        ),
    )
    args = p.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    audio, _ = librosa.load(str(audio_path), sr=SAMPLING_RATE, mono=True)
    audio = audio.astype(np.float32, copy=False)

    trace, score_backend = run_smartturn_trace(
        audio=audio,
        threshold_on=args.threshold_on,
        threshold_off=args.threshold_off,
        min_silence_ms=args.min_silence_ms,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "sampling_rate": SAMPLING_RATE,
                "frame_size": FRAME_SIZE,
                "smartturn_config": {
                    "threshold_on": args.threshold_on,
                    "threshold_off": args.threshold_off,
                    "min_silence_ms": args.min_silence_ms,
                },
                "score_backend": score_backend,
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote trace: {out_path}")
    print(f"[backend] score_backend={score_backend}")
    print(
        f"[events] start={len(trace['speech_start_samples'])}, "
        f"end={len(trace['speech_end_samples'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
