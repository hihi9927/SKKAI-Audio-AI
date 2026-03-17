from __future__ import annotations

import argparse
import os
from typing import Any


def qwen_streaming_args_smartturn(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    SmartTurn track argument extension.

    We keep baseline args and add SmartTurn-only options behind a separate
    entrypoint so baseline scripts stay untouched.
    """
    from librispeech_qwen import qwen_streaming_args

    parser = qwen_streaming_args(parser)
    parser.add_argument("--turn-detector", default="smartturn-v3")
    parser.add_argument("--smartturn-threshold-on", type=float, default=0.55)
    parser.add_argument("--smartturn-threshold-off", type=float, default=0.45)
    parser.add_argument("--smartturn-min-silence-ms", type=int, default=800)
    return parser


def qwen_streaming_factory_smartturn(*args: Any, **kwargs: Any) -> Any:
    """
    SmartTurn track model/server factory.

    Current behavior:
    - Delegates to existing qwen_streaming_factory.
    - Stamps SmartTurn-related values into env vars, so downstream code can
      read them without touching baseline pipeline yet.

    Next step:
    - Wire these values into the actual server VAD/turn endpoint module.
    """
    from librispeech_qwen import qwen_streaming_factory

    # Best-effort extraction from argparse namespace if present.
    ns = None
    if args:
        candidate = args[0]
        if hasattr(candidate, "__dict__"):
            ns = candidate

    if ns is not None:
        os.environ["STITy_TURN_DETECTOR"] = str(getattr(ns, "turn_detector", "smartturn-v3"))
        os.environ["STITy_SMARTTURN_THRESHOLD_ON"] = str(
            getattr(ns, "smartturn_threshold_on", 0.55)
        )
        os.environ["STITy_SMARTTURN_THRESHOLD_OFF"] = str(
            getattr(ns, "smartturn_threshold_off", 0.45)
        )
        os.environ["STITy_SMARTTURN_MIN_SILENCE_MS"] = str(
            getattr(ns, "smartturn_min_silence_ms", 800)
        )

    return qwen_streaming_factory(*args, **kwargs)

