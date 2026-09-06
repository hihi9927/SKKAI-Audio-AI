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
