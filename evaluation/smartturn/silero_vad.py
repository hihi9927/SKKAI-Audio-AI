from __future__ import annotations

import os
import importlib
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


def _env_int(*keys: str, default: int) -> int:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        try:
            return int(value)
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
        self.sampling_rate = sampling_rate
        self.model = model or _SmartTurnModel(sampling_rate=sampling_rate)
        self.speech_pad_ms = int(speech_pad_ms)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.threshold_on = threshold
        self.threshold_off = threshold_off

        # Stabilization knobs for SmartTurn->Silero compatibility.
        self.ema_alpha = _env_float("STITY_SMARTTURN_EMA_ALPHA", default=0.2)
        min_silence_override = _env_int("STITY_SMARTTURN_MIN_SILENCE_MS", default=-1)
        self.min_silence_duration_ms = (
            min_silence_override if min_silence_override > 0 else int(min_silence_duration_ms)
        )
        self.min_silence_samples = int(
            (self.min_silence_duration_ms / 1000.0) * self.sampling_rate
        )
        self.min_utterance_ms = _env_int("STITY_SMARTTURN_MIN_UTTERANCE_MS", default=1800)
        self.min_utterance_samples = int((self.min_utterance_ms / 1000.0) * self.sampling_rate)
        self.end_cooldown_ms = _env_int("STITY_SMARTTURN_END_COOLDOWN_MS", default=400)
        self.end_cooldown_samples = int((self.end_cooldown_ms / 1000.0) * self.sampling_rate)

        self._reset_runtime_state()

    def reset_states(self) -> None:
        self._reset_runtime_state()
        if hasattr(self.model, "_detector"):
            self.model._detector.reset_states()
        self._reset_hook_state()

    def __call__(self, x, return_seconds: bool = False):
        frame = _to_numpy(x)
        n = int(frame.size)
        if n <= 0:
            return None

        raw_prob = float(self.model(frame, self.sampling_rate).item())
        if self._ema_prob is None:
            self._ema_prob = raw_prob
        else:
            a = min(1.0, max(0.0, self.ema_alpha))
            self._ema_prob = a * raw_prob + (1.0 - a) * self._ema_prob
        prob = self._ema_prob

        event = None
        if not self._in_speech and prob >= self.threshold_on:
            self._in_speech = True
            self._speech_start_sample = self._sample_cursor
            self._silence_run_samples = 0
            event = {"start": self._speech_start_sample}
        elif self._in_speech:
            if prob < self.threshold_off:
                self._silence_run_samples += n
            else:
                self._silence_run_samples = 0

            utterance_samples = self._sample_cursor + n - self._speech_start_sample
            can_end = utterance_samples >= self.min_utterance_samples
            cooldown_ok = (
                self._sample_cursor + n - self._last_end_sample >= self.end_cooldown_samples
            )

            if can_end and cooldown_ok and self._silence_run_samples >= self.min_silence_samples:
                end_sample = self._sample_cursor + n
                self._in_speech = False
                self._silence_run_samples = 0
                self._speech_start_sample = 0
                self._last_end_sample = end_sample
                self._reset_hook_state()
                event = {"end": end_sample}

        self._sample_cursor += n

        if event is not None and return_seconds:
            key = "start" if "start" in event else "end"
            return {key: event[key] / self.sampling_rate}
        if event is not None:
            return event
        return None

    def _reset_runtime_state(self) -> None:
        self._sample_cursor = 0
        self._in_speech = False
        self._speech_start_sample = 0
        self._silence_run_samples = 0
        self._last_end_sample = -10**12
        self._ema_prob: Optional[float] = None

    def _reset_hook_state(self) -> None:
        hook = os.getenv("STITY_SMARTTURN_SCORE_HOOK", "")
        if ":" not in hook:
            return
        module_name, _ = hook.split(":", 1)
        try:
            mod = importlib.import_module(module_name)
            reset_fn = getattr(mod, "reset_state", None)
            if callable(reset_fn):
                reset_fn(self.sampling_rate)
        except Exception:
            return


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
