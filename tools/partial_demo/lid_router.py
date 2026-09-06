"""발화 시작을 VAD 로 잡고, 그 뒤 일정 길이만 듣고 언어를 판정한다.

**왜 VAD 가 먼저인가.** 클립 앞머리의 무음을 그대로 LID 에 넣으면 판정이 무너진다.
166개(FLEURS 읽기 ko/en 각 60, 자연발화 ko 23 / 낭독 en 23)로 실측한 값:

    창 0.5초 정확도        무음 포함 -> 무음 제거
      voxlingua107-ecapa    1.2%   ->  30.7%
      whisper-tiny         53.0%   ->  73.5%

ECAPA 의 1.2% 는 우연(50%)보다 낮다. 무음을 주면 엉뚱한 언어를 일관되게 뱉기
때문이다. 그래서 VAD 가 발화 시작을 잡은 뒤부터 창을 센다.

**왜 whisper-base 인가.** 같은 조건에서 창 길이별 정확도(VAD 로 무음 제거):

    모델                크기   0.5초   1초    2초    3초   추론
    whisper-base        74M   82.5%  92.8%  100%   99.4%  9.1ms
    whisper-tiny        39M   73.5%  90.4%  96.4%  98.2%  8.4ms
    mms-lid-126        300M   38.6%  80.1%  95.8%  98.8% 10.3ms
    voxlingua107-ecapa   7M   30.7%  68.7%  94.0%  98.2%  3.4ms

추론 시간은 전부 10ms 안쪽이라 변수가 아니다. 지연을 정하는 건 오직 **판정에
필요한 오디오 길이**다. whisper 는 입력을 30초로 패딩하는 구조라 창을 줄여도
추론 시간이 줄지 않는다 — 그래도 10ms 라 문제되지 않는다.

대조군으로 지금 쓰는 Qwen3-ASR 이 보고하는 language 를 같은 166개로 재면 창 2초에서
ko 파인튜닝 93.4%, en 파인튜닝 53.0% 다. en 쪽은 사실상 동전 던지기로, 파인튜닝이
언어 감지를 망가뜨렸다. 그래서 ASR 의 자기 신고를 라우팅 근거로 쓰지 않는다.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("lid-router")

SR = 16000


class LidRouter:
    """VAD 로 발화 시작을 찾고, 그 뒤 `window_sec` 만큼 듣고 언어를 판정한다.

    스트림 하나가 `feed()` 로 오디오를 계속 넣고, 판정이 서면 언어 코드가 나온다.
    아직 모자라면 None 이다. 모델은 프로세스에 한 번만 올라가고 스트림마다
    `Session` 이 자기 버퍼를 갖는다.
    """

    def __init__(self, model_name: str = "openai/whisper-base",
                 window_sec: float = 1.0, max_wait_sec: float = 5.0,
                 device: str = "cuda"):
        self.model_name = model_name
        self.window_sec = window_sec
        self.max_wait_sec = max_wait_sec
        self.device = device
        self._gpu_lock = threading.Lock()   # GPU 호출은 한 번에 하나만
        self._load()

    def _load(self) -> None:
        import torch
        from silero_vad import load_silero_vad
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._torch = torch
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        logger.info(f"[lid] loading {self.model_name} on {self.device} ({dtype})")
        self.proc = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, dtype=dtype).to(self.device).eval()
        tok = self.proc.tokenizer
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        # 언어 토큰은 <|xx|> 꼴이다. 이 집합 안에서만 argmax 를 잡는다.
        self.lang_ids = {
            tok.convert_tokens_to_ids(t): t.strip("<|>")
            for t in tok.additional_special_tokens
            if len(t) == 6 and t.startswith("<|") and t.endswith("|>")
        }
        self.vad = load_silero_vad()
        logger.info("[lid] ready")

    # ── 판정 ──────────────────────────────────────────────────────────────────
    def _speech_span(self, audio: np.ndarray):
        """첫 발화 구간 (시작, 끝, 그 구간이 이미 닫혔는지)."""
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(self._torch.from_numpy(audio), self.vad,
                                   sampling_rate=SR)
        if not ts:
            return None
        first = ts[0]
        # 뒤에 다른 구간이 있거나, 끝난 뒤로 무음이 충분히 이어졌으면 닫힌 것으로 본다.
        closed = len(ts) > 1 or (len(audio) - first["end"]) >= SR * 0.3
        return first["start"], first["end"], closed

    def _classify(self, audio: np.ndarray) -> str:
        torch = self._torch
        feats = self.proc.feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        dec = torch.tensor([[self.sot]], device=self.device)
        with torch.inference_mode():
            logits = self.model(feats, decoder_input_ids=dec).logits[0, -1]
        ids = list(self.lang_ids)
        return self.lang_ids[ids[int(torch.argmax(logits[ids]))]]

    def _decide_sync(self, audio: np.ndarray, flush: bool = False):
        """(언어코드 또는 None, 사유). None 이면 아직 더 들어야 한다.

        `flush` 는 클라이언트가 스트림을 끝냈다는 뜻이다 — 더 기다려도 오디오가
        안 오므로 있는 것으로 판정한다.
        """
        with self._gpu_lock:
            span = self._speech_span(audio)
            if span is None:
                if flush or len(audio) >= SR * self.max_wait_sec:
                    return None, "no-speech"
                return None, "no-speech-yet"
            start, end, closed = span
            speech = audio[start:end] if closed else audio[start:]
            if len(speech) >= SR * self.window_sec:
                return self._classify(speech[: int(SR * self.window_sec)]), "ok"
            # 창을 못 채웠다. 발화가 이미 끝났으면 더 기다릴 게 없으니 있는 걸로 판정한다.
            # 짧은 맞장구("어.", "그지.")가 5초씩 묶이는 걸 막는다.
            if closed or flush or len(audio) >= SR * self.max_wait_sec:
                if len(speech) < SR * 0.2:
                    return None, "speech-too-short"
                return self._classify(speech), f"short-speech({len(speech)/SR:.1f}s)"
            return None, "collecting"

    async def decide(self, audio: np.ndarray, flush: bool = False):
        return await asyncio.to_thread(self._decide_sync, audio, flush)


class Session:
    """스트림 하나가 쌓아 두는 오디오. 판정이 설 때까지만 산다."""

    def __init__(self, router: LidRouter):
        self.router = router
        self._chunks: list[bytes] = []
        self._samples = 0

    @property
    def seconds(self) -> float:
        return self._samples / SR

    def add(self, pcm_bytes: bytes) -> None:
        self._chunks.append(pcm_bytes)
        self._samples += len(pcm_bytes) // 2

    @property
    def raw(self) -> list[bytes]:
        """모아 둔 오디오 원본. 위쪽 서버에 그대로 흘려 보낸다."""
        return self._chunks

    def _float(self) -> np.ndarray:
        joined = b"".join(self._chunks)
        return np.frombuffer(joined, dtype="<i2").astype(np.float32) / 32768.0

    async def decide(self, flush: bool = False):
        if self._samples < SR * 0.2 and not flush:
            return None, "too-short"
        return await self.router.decide(self._float(), flush)

    def timed_out(self) -> bool:
        return self.seconds >= self.router.max_wait_sec


# ── 흐르는 스트림에 대한 연속 판정 ────────────────────────────────────────────
# 스트림당 한 번 판정하는 Session 과 달리, 이쪽은 오디오가 흐르는 내내 VAD 로
# 발화 구간을 찾아 구간마다 언어를 남긴다. 두 ASR 서버에 오디오를 동시에 보내
# 놓고 어느 쪽 결과를 쓸지 고를 때 쓴다.
#
# **왜 발화 구간마다인가.** 라우팅을 스트림당 한 번만 하면 화자가 도중에 언어를
# 바꿔도 모델이 안 바뀐다. 실제로 한국어 모델 세션에 영어를 말했더니
# '헬로 나이스 미트 유.' 처럼 한글로 받아썼다. 구간마다 판정을 남겨 두면
# 그 구간의 결과를 낸 서버만 통과시킬 수 있다.

class VerdictTracker:
    """오디오를 받아 두고 발화 구간마다 언어 판정을 쌓는다.

    판정은 (구간 시작 초, 구간 끝 초 또는 None, 언어) 로 남는다. 끝이 None 이면
    아직 말하는 중이라는 뜻이다. 구간이 닫히면 같은 항목을 확정값으로 갱신한다.

    버퍼는 최근 `keep_sec` 만큼만 들고 있는다. VAD 를 매번 전체에 돌리면 길이의
    제곱으로 늘어난다. 잘라낸 만큼은 `_offset` 에 누적해 절대 시각을 유지한다.
    """

    def __init__(self, router: "LidRouter", keep_sec: float = 20.0,
                 step_sec: float = 0.4):
        self.router = router
        self.keep_sec = keep_sec
        self.step_sec = step_sec
        self._buf = np.zeros(0, dtype=np.float32)
        self._offset = 0.0          # 잘라낸 앞부분의 길이(초)
        self._audio_sec = 0.0       # 지금까지 받은 전체 오디오 길이(초)
        self._last_run = -1.0
        self.verdicts: list[list] = []   # [시작초, 끝초 또는 None, 언어]

    @property
    def audio_sec(self) -> float:
        return self._audio_sec

    def feed(self, pcm_bytes: bytes) -> None:
        x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])
        self._audio_sec += len(x) / SR
        keep = int(SR * self.keep_sec)
        if len(self._buf) > keep:
            drop = len(self._buf) - keep
            self._offset += drop / SR
            self._buf = self._buf[drop:]

    def _segments_sync(self):
        from silero_vad import get_speech_timestamps

        with self.router._gpu_lock:
            ts = get_speech_timestamps(self.router._torch.from_numpy(self._buf),
                                       self.router.vad, sampling_rate=SR)
        out = []
        for t in ts:
            start = self._offset + t["start"] / SR
            end = self._offset + t["end"] / SR
            # 버퍼 끝에 붙어 있으면 아직 말하는 중으로 본다.
            closed = (len(self._buf) - t["end"]) >= SR * 0.3
            out.append((start, end, closed))
        return out

    def _classify_span(self, start: float, end: float) -> Optional[str]:
        a = int((start - self._offset) * SR)
        b = int((end - self._offset) * SR)
        span = self._buf[max(0, a):max(0, b)]
        if len(span) < SR * 0.2:
            return None
        span = span[: int(SR * self.router.window_sec)]
        with self.router._gpu_lock:
            return self.router._classify(span)

    # 0.5초 창 정확도가 whisper-base 기준 82.5% 다. 그보다 짧은 조각의 판정은
    # 못 믿는다 — 실제로 한국어 발화 앞머리 0.4초가 en 으로 잘못 찍혔다.
    MIN_SPEECH_SEC = 0.5

    def _find_overlap(self, start: float, end: float):
        """이미 있는 판정 중 이 구간과 겹치는 것.

        시작점으로 맞추면 안 된다 — 오디오가 쌓이면서 silero 가 같은 발화의 경계를
        조금씩 다르게 잡아, 한 발화가 매번 새 항목으로 쌓인다. 실제로 발화 4개짜리
        스트림에서 판정이 29개까지 늘었다. 겹침으로 보면 하나로 모인다.
        """
        for v in self.verdicts:
            v_end = v[1] if v[1] is not None else self._audio_sec
            if start < v_end and v[0] < end:
                return v
        return None

    def _update_sync(self) -> None:
        for start, end, closed in self._segments_sync():
            if (end - start) <= self.MIN_SPEECH_SEC:
                continue          # 숨소리·잡음 조각. 여기서 나온 판정은 못 믿는다.
            existing = self._find_overlap(start, end)
            if existing is not None and existing[1] is not None:
                continue                      # 이미 확정된 구간
            # 창을 채웠거나 구간이 닫혔을 때만 판정한다. 그 전에는 근거가 모자란다.
            if not closed and (end - start) < self.router.window_sec:
                continue
            lang = self._classify_span(start, end)
            if lang is None:
                continue
            if existing is None:
                self.verdicts.append([start, end if closed else None, lang])
            else:
                existing[0] = min(existing[0], start)
                existing[1] = end if closed else None
                existing[2] = lang
        self.verdicts.sort(key=lambda v: v[0])

    async def update(self) -> None:
        """새 오디오가 `step_sec` 만큼 쌓였을 때만 실제로 돌린다."""
        if self._audio_sec - self._last_run < self.step_sec:
            return
        self._last_run = self._audio_sec
        await asyncio.to_thread(self._update_sync)

    def lang_at(self, t: Optional[float]) -> Optional[str]:
        """시각 t 에 유효한 판정. t 가 None 이면 가장 최근 판정."""
        if not self.verdicts:
            return None
        if t is None:
            return self.verdicts[-1][2]
        # t 보다 앞에서 시작한 구간 중 가장 늦은 것. 커밋은 발화보다 뒤에 오므로
        # 약간의 여유를 둔다.
        picked = [v for v in self.verdicts if v[0] <= t + 0.5]
        return (picked[-1] if picked else self.verdicts[0])[2]
