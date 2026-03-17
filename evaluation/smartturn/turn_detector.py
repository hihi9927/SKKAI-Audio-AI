from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class TurnEvent:
    """Turn detector event for a frame."""

    start: Optional[int] = None
    end: Optional[int] = None
    prob: Optional[float] = None


class SmartTurnV3Detector:
    """
    Adapter wrapper for SmartTurn-v3.

    Notes:
    - This class isolates SmartTurn API differences from the rest of the code.
    - It currently contains a conservative fallback state machine so the module
      can be imported before SmartTurn wiring is complete.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold_on: float = 0.55,
        threshold_off: float = 0.45,
        min_silence_duration_ms: int = 800,
        score_hook: Optional[str] = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.min_silence_duration_ms = min_silence_duration_ms

        self._in_speech = False
        self._sample_cursor = 0
        self._silence_samples = 0
        self._min_silence_samples = int(
            (min_silence_duration_ms / 1000.0) * sampling_rate
        )

        self._score_backend = "rms_fallback"
        self._score_fn = self._build_score_fn(
            score_hook=score_hook or os.getenv("STITY_SMARTTURN_SCORE_HOOK", "")
        )
        if self._score_fn is not None:
            self._score_backend = "smartturn"

    def reset_states(self) -> None:
        self._in_speech = False
        self._sample_cursor = 0
        self._silence_samples = 0

    @property
    def score_backend(self) -> str:
        return self._score_backend

    def _build_score_fn(self, score_hook: str) -> Optional[Callable[[np.ndarray], float]]:
        # 1) Explicit hook: module.submodule:function
        if score_hook:
            if ":" not in score_hook:
                raise ValueError(
                    "STITY_SMARTTURN_SCORE_HOOK must be in 'module:function' format."
                )
            module_name, fn_name = score_hook.split(":", 1)
            module = importlib.import_module(module_name)
            fn = getattr(module, fn_name)
            return lambda frame: self._invoke_callable(fn, frame)

        # 2) Auto-discovery for common SmartTurn module/class/function names
        module_candidates = [
            "smartturn",
            "smartturn_v3",
            "smart_turn",
            "smart_turn_v3",
        ]

        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            fn = self._find_callable_in_module(module)
            if fn is not None:
                return lambda frame, _fn=fn: self._invoke_callable(_fn, frame)

        return None

    def _find_callable_in_module(self, module: Any) -> Optional[Callable[..., Any]]:
        class_names = [
            "SmartTurnV3",
            "SmartTurn",
            "TurnDetector",
            "EndpointDetector",
        ]
        method_names = [
            "predict_proba",
            "score",
            "infer",
            "__call__",
        ]
        function_names = [
            "predict_proba",
            "score",
            "infer",
            "predict",
        ]

        for cls_name in class_names:
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            try:
                try:
                    obj = cls(sampling_rate=self.sampling_rate)
                except TypeError:
                    obj = cls()
            except Exception:
                continue

            for method_name in method_names:
                method = getattr(obj, method_name, None)
                if callable(method):
                    return method

        for fn_name in function_names:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                return fn

        return None

    def _invoke_callable(self, fn: Callable[..., Any], frame: np.ndarray) -> float:
        frame = frame.astype(np.float32, copy=False)
        attempts = [
            lambda: fn(frame, sampling_rate=self.sampling_rate),
            lambda: fn(frame, sample_rate=self.sampling_rate),
            lambda: fn(frame, sr=self.sampling_rate),
            lambda: fn(frame),
            lambda: fn(frame[None, :], sampling_rate=self.sampling_rate),
            lambda: fn(frame[None, :]),
        ]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                out = attempt()
                return self._extract_prob(out)
            except TypeError as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to invoke SmartTurn scoring callable.")

    def _extract_prob(self, out: Any) -> float:
        if isinstance(out, dict):
            for key in ("prob", "probability", "speech_prob", "turn_prob", "score"):
                if key in out:
                    return self._clip01(float(out[key]))
            raise ValueError(f"Unsupported dict output keys: {list(out.keys())}")

        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise ValueError("Empty output from SmartTurn callable.")
            return self._extract_prob(out[0])

        if isinstance(out, np.ndarray):
            if out.size == 0:
                raise ValueError("Empty ndarray output from SmartTurn callable.")
            return self._clip01(float(np.ravel(out)[0]))

        return self._clip01(float(out))

    def _clip01(self, x: float) -> float:
        return min(1.0, max(0.0, x))

    def _score(self, frame: np.ndarray) -> float:
        """
        Return a speech/turn probability in [0, 1].
        """
        if self._score_fn is not None:
            return self._score_fn(frame)

        # Fallback (if SmartTurn runtime is unavailable): simple energy proxy.
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        return self._clip01(rms * 8.0)

    def process(self, frame: np.ndarray) -> TurnEvent:
        frame = frame.astype(np.float32, copy=False)
        frame_size = int(frame.size)

        prob = self._score(frame)
        start: Optional[int] = None
        end: Optional[int] = None

        if not self._in_speech and prob >= self.threshold_on:
            self._in_speech = True
            self._silence_samples = 0
            start = self._sample_cursor

        if self._in_speech:
            if prob < self.threshold_off:
                self._silence_samples += frame_size
            else:
                self._silence_samples = 0

            if self._silence_samples >= self._min_silence_samples:
                self._in_speech = False
                end = self._sample_cursor + frame_size
                self._silence_samples = 0

        self._sample_cursor += frame_size
        return TurnEvent(start=start, end=end, prob=prob)
