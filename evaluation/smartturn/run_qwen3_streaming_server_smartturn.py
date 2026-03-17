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

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
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

    # Optional defaults for SmartTurn detector behavior.
    os.environ.setdefault("STITY_SMARTTURN_THRESHOLD_ON", "0.5")
    os.environ.setdefault("STITY_SMARTTURN_THRESHOLD_OFF", "0.4")

    # Hand over execution to the original script with current CLI args.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

