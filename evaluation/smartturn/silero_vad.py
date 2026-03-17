from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from evaluation.smartturn.turn_detector import SmartTurnV3Detector


def _env_float(*keys: str, default: float) -> float:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


class _SmartTurnModel:
    """
    Silero-like model shim.

    The original server expects:
      prob = model(window_tensor, sampling_rate).item()
    """

    def __init__(self, sampling_rate: int = 16000) -> None:
        self.sampling_rate = sampling_rate
        threshold_on = _env_float(
            "STITY_SMARTTURN_THRESHOLD_ON",
            "STITy_SMARTTURN_THRESHOLD_ON",
            default=0.5,
        )
        threshold_off = _env_float(
            "STITY_SMARTTURN_THRESHOLD_OFF",
            "STITy_SMARTTURN_THRESHOLD_OFF",
            default=max(0.0, threshold_on - 0.1),
        )
        self._detector = SmartTurnV3Detector(
            sampling_rate=sampling_rate,
            threshold_on=threshold_on,
            threshold_off=threshold_off,
            min_silence_duration_ms=800,
        )

    def _score(self, frame: np.ndarray) -> float:
        return float(self._detector._score(frame))

    def __call__(self, window_tensor, sampling_rate: Optional[int] = None):
        frame = _to_numpy(window_tensor)
        prob = self._score(frame)
        return torch.tensor(prob, dtype=torch.float32)


class VADIterator:
    """
    Silero VADIterator compatibility wrapper backed by SmartTurnV3Detector.
    """

    def __init__(
        self,
        model=None,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 800,
        speech_pad_ms: int = 0,  # kept for API compatibility
        min_speech_duration_ms: int = 0,  # kept for API compatibility
        **_: object,
    ) -> None:
        threshold_off = _env_float(
            "STITY_SMARTTURN_THRESHOLD_OFF",
            "STITy_SMARTTURN_THRESHOLD_OFF",
            default=max(0.0, threshold - 0.1),
        )
        self.detector = SmartTurnV3Detector(
            sampling_rate=sampling_rate,
            threshold_on=threshold,
            threshold_off=threshold_off,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        self.sampling_rate = sampling_rate
        self.model = model or _SmartTurnModel(sampling_rate=sampling_rate)
        self.speech_pad_ms = int(speech_pad_ms)
        self.min_speech_duration_ms = int(min_speech_duration_ms)

    def reset_states(self) -> None:
        self.detector.reset_states()

    def __call__(self, x, return_seconds: bool = False):
        frame = _to_numpy(x)
        event = self.detector.process(frame)
        if event.start is not None:
            value = event.start / self.sampling_rate if return_seconds else int(event.start)
            return {"start": value}
        if event.end is not None:
            value = event.end / self.sampling_rate if return_seconds else int(event.end)
            return {"end": value}
        return None


def load_silero_vad(*_, **__):
    """
    Silero loader compatibility function expected by the original server code.
    """
    return _SmartTurnModel()


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.ravel(arr).astype(np.float32, copy=False)

