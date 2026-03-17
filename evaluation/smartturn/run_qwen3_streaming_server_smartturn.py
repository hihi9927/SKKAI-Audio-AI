#!/usr/bin/env python3
"""
Run the original Qwen3 streaming server with SmartTurn-backed VAD shim.

This keeps the original server code untouched:
  Qwen3-ASR/examples/streaming_websocket_server.py

How it works:
- This script prepends `evaluation/smartturn` to sys.path.
- The server imports `silero_vad`; Python resolves it to
  `evaluation/smartturn/silero_vad.py` first.
- The shim exposes Silero-compatible APIs but routes logic to SmartTurn.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--st-prob-mode", default="speech")
    parser.add_argument("--st-threshold-on", default="0.55")
    parser.add_argument("--st-threshold-off", default="0.35")
    parser.add_argument("--st-min-silence-ms", default="1400")
    parser.add_argument("--st-min-utterance-ms", default="2500")
    parser.add_argument("--st-end-cooldown-ms", default="800")
    parser.add_argument("--st-ema-alpha", default="0.15")
    parser.add_argument(
        "--st-score-hook",
        default="evaluation.smartturn.pipecat_hook:score",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper_args, passthrough_argv = _parse_wrapper_args(sys.argv[1:])

    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    qwen3_root = repo_root / "Qwen3-ASR"
    target = qwen3_root / "examples" / "streaming_websocket_server.py"

    if not target.exists():
        raise FileNotFoundError(f"Target server script not found: {target}")

    # Ensure import order:
    # 1) this folder (contains silero_vad shim)
    # 2) Qwen3-ASR package root
    # 3) project root
    sys.path.insert(0, str(here))
    sys.path.insert(1, str(qwen3_root))
    sys.path.insert(2, str(repo_root))

    # SmartTurn defaults can be overridden by wrapper args.
    os.environ["STITY_SMARTTURN_SCORE_HOOK"] = str(wrapper_args.st_score_hook)
    os.environ["STITY_SMARTTURN_PROB_MODE"] = str(wrapper_args.st_prob_mode)
    os.environ["STITY_SMARTTURN_THRESHOLD_ON"] = str(wrapper_args.st_threshold_on)
    os.environ["STITY_SMARTTURN_THRESHOLD_OFF"] = str(wrapper_args.st_threshold_off)
    os.environ["STITY_SMARTTURN_MIN_SILENCE_MS"] = str(wrapper_args.st_min_silence_ms)
    os.environ["STITY_SMARTTURN_MIN_UTTERANCE_MS"] = str(wrapper_args.st_min_utterance_ms)
    os.environ["STITY_SMARTTURN_END_COOLDOWN_MS"] = str(wrapper_args.st_end_cooldown_ms)
    os.environ["STITY_SMARTTURN_EMA_ALPHA"] = str(wrapper_args.st_ema_alpha)

    # Hand over execution to the original script with current CLI args.
    sys.argv = [str(target), *passthrough_argv]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
