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
                 device: str = "cuda", known_langs=None):
        self.model_name = model_name
        # 라우팅 표에 있는 언어들. 조기 확정을 잠글지 판단하는 데 쓴다.
        self.known_langs = set(known_langs or ())
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

    def _classify(self, audio: np.ndarray, with_conf: bool = False):
        """언어 코드. `with_conf` 면 (언어, 확신도) 를 준다.

        확신도는 언어 토큰들 안에서만 정규화한 확률이다. argmax 만 쓰면 짧은 창을
        못 쓰지만, 확신도로 거르면 쓸 수 있다 — 166개 실측(whisper-base):

            창     argmax     확신 0.8 이상만
            0.3s   71.1%      28.3% 를 97.9% 로
            0.5s   82.5%      45.8% 를 100% 로
            0.7s   88.0%      60.8% 를 98.0% 로
            1.0s   92.8%      76.5% 를 99.2% 로

        0.5초에 절반 가까이가 확정된다는 뜻이다. 나머지만 더 들으면 된다.
        """
        torch = self._torch
        feats = self.proc.feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        dec = torch.tensor([[self.sot]], device=self.device)
        with torch.inference_mode():
            logits = self.model(feats, decoder_input_ids=dec).logits[0, -1]
        ids = list(self.lang_ids)
        sub = logits[ids].float()
        probs = torch.softmax(sub, dim=-1)
        k = int(torch.argmax(probs))
        lang = self.lang_ids[ids[k]]
        return (lang, float(probs[k])) if with_conf else lang

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
        self._last_segments: list[tuple] = []   # 마지막 VAD 스캔 결과 (재사용용)
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

    # 발화 시작 후 이만큼 들었을 때 이 확신도를 넘으면 그 자리에서 확정한다.
    # 위 실측표에서 오답이 없거나 1건 수준인 조합만 골랐다. 어느 단계도 못 넘기면
    # 창(window_sec)을 다 채운 뒤 argmax 로 확정한다.
    EARLY_STEPS = ((0.3, 0.90), (0.5, 0.85), (0.7, 0.80))

    def _classify_span(self, start: float, end: float, early: bool = True):
        """(언어, 얼마나 듣고 정했나) 또는 (None, None).

        짧은 창부터 올라가며 확신도가 임계를 넘으면 즉시 확정한다. whisper 는 입력을
        30초로 패딩하므로 창을 줄여도 추론 시간이 같다(9ms) — 여러 창을 시도해도
        비용이 거의 안 는다.
        """
        a = int((start - self._offset) * SR)
        b = int((end - self._offset) * SR)
        span = self._buf[max(0, a):max(0, b)]
        if len(span) < SR * 0.2:
            return None, None
        avail = len(span) / SR
        with self.router._gpu_lock:
            if early:
                for secs, need in self.EARLY_STEPS:
                    if secs > avail or secs >= self.router.window_sec:
                        continue
                    lang, conf = self.router._classify(
                        span[: int(SR * secs)], with_conf=True)
                    if conf >= need:
                        return lang, secs
            lang = self.router._classify(span[: int(SR * self.router.window_sec)])
        return lang, min(avail, self.router.window_sec)

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
        self._last_segments = self._segments_sync()
        for start, end, closed in self._last_segments:
            if (end - start) <= self.MIN_SPEECH_SEC:
                continue          # 숨소리·잡음 조각. 여기서 나온 판정은 못 믿는다.
            existing = self._find_overlap(start, end)
            if existing is not None and existing[1] is not None:
                continue                      # 이미 닫힌 구간
            if existing is not None and len(existing) > 3 and existing[3]:
                # 확신도로 일찍 확정한 값은 다시 뒤집지 않는다. 더 긴 창이 늘 더
                # 정확하지는 않다 — 1.0초 argmax 는 92.8% 인데, 0.5초에서 확신
                # 0.85 를 넘긴 부분집합은 이 표본에서 오답이 없었다.
                existing[1] = end if closed else None
                continue
            # 창을 채웠거나 구간이 닫혔으면 판정한다. 아직 말하는 중이라도 조기
            # 확정 단계의 최소 길이(EARLY_STEPS 의 첫 값)를 넘겼으면 시도한다 —
            # 확신도가 임계를 넘으면 창을 다 안 채워도 확정된다.
            earliest = min((step for step, _ in self.EARLY_STEPS),
                           default=self.router.window_sec)
            if not closed and (end - start) < min(earliest, self.router.window_sec):
                continue
            lang, used = self._classify_span(start, end)
            if lang is None:
                continue
            # 창을 다 안 채우고 정해졌다면 확신도로 일찍 확정된 것이다.
            # 다만 라우팅 표에 없는 언어로 나왔으면 잠그지 않는다 — 한국어 발화가
            # 0.5초 만에 zh 로 확정돼 그대로 굳는 일이 실제로 있었다. 그런 값은
            # 더 들으면 바뀔 여지를 남긴다.
            locked = (used is not None and used < self.router.window_sec
                      and (not self.router.known_langs
                           or lang in self.router.known_langs))
            if existing is None:
                self.verdicts.append([start, end if closed else None, lang, locked])
                logger.debug(f"[verdict] {start:.1f}s {lang} ({used:.2f}s 듣고"
                             f"{', 확정' if locked else ''})")
            else:
                existing[0] = min(existing[0], start)
                existing[1] = end if closed else None
                existing[2] = lang
                if len(existing) > 3:
                    existing[3] = locked
                else:
                    existing.append(locked)
        self.verdicts.sort(key=lambda v: v[0])

    async def update(self, force: bool = False) -> None:
        """새 오디오가 `step_sec` 만큼 쌓였을 때만 실제로 돌린다.

        `force` 는 판정을 기다리는 쪽에서 쓴다 — 오디오가 더 안 들어와도 이미 닫힌
        구간을 지금 판정해야 대기가 풀린다.
        """
        if not force and self._audio_sec - self._last_run < self.step_sec:
            return
        self._last_run = self._audio_sec
        await asyncio.to_thread(self._update_sync)

    def settled(self, end: Optional[float]) -> bool:
        """`end` 이전에 끝난 발화가 전부 판정됐나.

        **"end 를 품는 판정이 있나" 로 물으면 안 된다.** end 는 커밋 경계라 발화
        사이 침묵에 떨어지는 일이 흔하고, 그러면 어떤 발화 구간도 그걸 품지 않아
        조건이 영영 참이 되지 않는다. 실제로 그렇게 물었더니 거의 모든 final 이
        1초 타임아웃을 그대로 물었다.

        정작 막아야 하는 건 **아직 판정 안 된 발화가 앞에 있는데 먼저 도착한 final
        이 그보다 더 앞선 판정을 타고 나가는 것**이다. 그래서 end 이전에 닫힌 발화
        구간이 모두 판정을 가졌는지만 본다. 너무 짧아 애초에 판정하지 않는 조각은
        기다려도 안 생기므로 제외한다.
        """
        if end is None:
            return True
        for start, seg_end, closed in self._last_segments:
            if not closed or seg_end > end:
                continue
            if (seg_end - start) <= self.MIN_SPEECH_SEC:
                continue
            if self._find_overlap(start, seg_end) is None:
                return False
        return True

    def lang_at(self, t: Optional[float]) -> Optional[str]:
        """시각 t 에 유효한 판정. t 가 None 이면 가장 최근 판정."""
        if not self.verdicts:
            return None
        if t is None:
            return self.verdicts[-1][2]
        return self.lang_for_end(t)

    def lang_for_range(self, start: Optional[float], end: Optional[float]):
        """구간 [start, end] 를 가장 많이 덮는 판정.

        **끝점 하나로는 못 가른다.** 커밋 경계는 발화가 끝난 한참 뒤에 찍힐 수 있고,
        앞뒤 발화 사이 침묵이 서버 VAD 기준(800ms)보다 짧으면 아예 다음 발화 한복판에
        떨어진다. 실제로 영어 발화의 커밋이 7.0초에 찍혔는데 그 시각은 5.8초에 시작한
        한국어 구간 안이라, 영어 서버 결과가 한국어 판정으로 몰려 버려졌다. 같은
        발화를 낸 한국어 서버 것도 (영어 판정이라) 버려져 발화가 통째로 사라졌다.

        서버가 start 를 안 채우므로(항상 0.0) 호출하는 쪽에서 **직전 final 의 end**
        를 시작점으로 넘겨 준다. 그 구간과 가장 많이 겹치는 판정을 고른다.
        """
        if not self.verdicts:
            return None
        if end is None:
            return self.verdicts[-1][2]
        a = 0.0 if start is None else min(start, end)
        best, best_overlap = None, 0.0
        for v in self.verdicts:
            v_end = v[1] if v[1] is not None else self._audio_sec
            overlap = min(end, v_end) - max(a, v[0])
            if overlap > best_overlap:
                best, best_overlap = v, overlap
        if best is not None:
            return best[2]
        return self.lang_for_end(end)

    def lang_for_end(self, end: Optional[float]):
        """커밋 경계 `end` 로 그 발화의 판정을 찾는다.

        **이 서버는 final.start 를 안 채운다** — segment_start_time 이 0 으로
        초기화된 뒤 갱신되지 않아 항상 0.0 이 온다. 그래서 구간이 아니라 끝점
        하나로 골라야 한다.

        **앞쪽으로만 본다.** 종전에는 `v[0] <= end + 0.5` 로 뒤쪽 여유를 뒀는데,
        커밋 경계는 발화가 끝난 뒤 침묵까지 밀리므로 앞 발화의 final 이 다음 발화의
        판정에 걸렸다. 한국어가 3.6초에 끝나고 영어가 4.5초에 시작할 때 한국어
        final 의 end 가 4.4 면 영어 판정으로 잡혀 **버려지고**, 같은 자리에서 영어
        서버가 낸 엉뚱한 문장이 대신 통과한다. 유실과 중첩이 한꺼번에 생긴다.

        순서: end 를 품는 구간이 있으면 그것, 없으면(침묵에서 끊긴 커밋) end 이전에
        **끝난** 구간 중 마지막.
        """
        if not self.verdicts:
            return None
        if end is None:
            return self.verdicts[-1][2]

        for v in self.verdicts:
            v_end = v[1] if v[1] is not None else self._audio_sec
            if v[0] <= end <= v_end:
                return v[2]

        ended = [v for v in self.verdicts
                 if (v[1] if v[1] is not None else self._audio_sec) <= end]
        return (ended[-1] if ended else self.verdicts[0])[2]
