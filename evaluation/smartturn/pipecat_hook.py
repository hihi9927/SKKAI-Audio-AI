from __future__ import annotations

import os
from typing import Optional

import numpy as np

_ANALYZER = None
_SR: Optional[int] = None
_BUFFER = np.array([], dtype=np.float32)
_MAX_SAMPLES = 8 * 16000


def _get_analyzer():
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER

    try:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    except Exception as e:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "pipecat SmartTurn is not available. "
            "Install with: pip install 'pipecat-ai[local-smart-turn-v3]'"
        ) from e

    model_path = os.getenv("SMARTTURN_MODEL_PATH") or None
    _ANALYZER = LocalSmartTurnAnalyzerV3(smart_turn_model_path=model_path)
    return _ANALYZER


def score(frame: np.ndarray, sampling_rate: int = 16000) -> float:
    """
    SmartTurn score hook for STiTy detector.

    Input:
      frame: float32 mono chunk
      sampling_rate: audio sample rate

    Output:
      probability in [0, 1]
    """
    global _BUFFER, _SR, _MAX_SAMPLES

    analyzer = _get_analyzer()
    frame = np.ravel(frame).astype(np.float32, copy=False)

    if _SR != sampling_rate:
        _SR = sampling_rate
        _MAX_SAMPLES = int(8 * sampling_rate)
        _BUFFER = np.array([], dtype=np.float32)

    if frame.size:
        _BUFFER = np.concatenate([_BUFFER, frame])
        if _BUFFER.size > _MAX_SAMPLES:
            _BUFFER = _BUFFER[-_MAX_SAMPLES:]

    # Pipecat analyzer keeps sample rate on the instance.
    if hasattr(analyzer, "_sample_rate"):
        analyzer._sample_rate = sampling_rate

    if hasattr(analyzer, "_predict_endpoint"):
        out = analyzer._predict_endpoint(_BUFFER)
        prob = float(out.get("probability", 0.0))
    else:  # pragma: no cover - future API variants
        prob = float(analyzer(_BUFFER))

    return max(0.0, min(1.0, prob))

