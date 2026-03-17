#!/usr/bin/env python3
"""
Run Qwen3 Simul-Streaming mode server (SmartTurn experiment track).

This file is intentionally separated from run_qwen_pipeline.py so we can
iterate on SmartTurn integration without touching the current baseline path.
"""
import os
import sys

# 1) Current directory: evaluation/smartturn
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 2) Project root: STiTy
base_path = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, base_path)

# 3) Existing LibriSpeech server dependencies
librispeech_dir = os.path.join(base_path, "evaluation", "LibriSpeech")
sys.path.insert(0, librispeech_dir)
sys.path.insert(0, os.path.join(librispeech_dir, "servers"))
sys.path.insert(0, os.path.join(base_path, "SimulStreaming"))

# 4) Qwen3-ASR
qwen_asr_path = os.path.join(base_path, "Qwen3-ASR")
sys.path.insert(0, qwen_asr_path)

from server_test import main_websocket_server
from evaluation.smartturn.librispeech_qwen_smartturn import (
    qwen_streaming_args_smartturn,
    qwen_streaming_factory_smartturn,
)


if __name__ == "__main__":
    if "--port" not in sys.argv:
        sys.argv.extend(["--port", "8011"])

    print("Starting Qwen3 Streaming Server (SmartTurn track, port 8011)")
    main_websocket_server(
        qwen_streaming_factory_smartturn,
        add_args=qwen_streaming_args_smartturn,
    )
