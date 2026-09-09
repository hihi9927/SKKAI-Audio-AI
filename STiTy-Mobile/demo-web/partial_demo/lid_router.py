"""발화 시작을 VAD 로 잡고, 그 뒤 일정 길이만 듣고 언어를 판정한다.

**왜 VAD 가 먼저인가.** 클립 앞머리의 무음을 그대로 LID 에 넣으면 판정이 무너진다.
166개(FLEURS 읽기 ko/en 각 60, 자연발화 ko 23 / 낭독 en 23)로 실측한 값:

    창 0.5초 정확도        무음 포함 -> 무음 제거
      voxlingua107-ecapa    1.2%   ->  30.7%
      whisper-tiny         53.0%   ->  73.5%

ECAPA 의 1.2% 는 우연(50%)보다 낮다. 무음을 주면 엉뚱한 언어를 일관되게 뱉기
때문이다. 그래서 VAD 가 발화 시작을 잡은 뒤부터 창을 센다.

**왜 whisper 인가.** 같은 조건에서 창 길이별 정확도(VAD 로 무음 제거):

    모델                크기   0.5초   1초    2초    3초   추론
    whisper-base        74M   82.5%  92.8%  100%   99.4%  9.1ms
    whisper-tiny        39M   73.5%  90.4%  96.4%  98.2%  8.4ms
    mms-lid-126        300M   38.6%  80.1%  95.8%  98.8% 10.3ms
    voxlingua107-ecapa   7M   30.7%  68.7%  94.0%  98.2%  3.4ms

**왜 base 가 아니라 small 이 기본값인가.** ko/en/es 226클립에서 small 은 1초 98.2%,
2초 99.1% 로 base 의 96.9% / 98.7% 보다 낫다. 원어민 음성에서의 차이는 이렇게 작고,
값을 치르는 이유는 억양 있는 제2언어 발화다 — 한국인이 말한 스페인어가 base 에서
ko 로 찍혀 엉뚱한 서버로 갔다. 대신 VRAM 이 536MiB 에서 884MiB 로 는다. 위 창 길이
표와 아래 확신도 임계값은 전부 base 로 잰 값이다. `--lid-model openai/whisper-base`
로 되돌릴 수 있다.

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

    def __init__(self, model_name: str = "openai/whisper-small",
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
        if self.device.startswith("cuda"):
            # 같은 GPU 에 ASR 서버 셋(각 0.25)과 번역기가 함께 산다. 이 프로세스가 캐시
            # 할당자로 자라면 여유(약 0.4~0.9GB)를 먹고 vLLM 엔진이 죽는다 — 실측에서
            # 후보 6개 배치 디코딩을 넣은 뒤 ko 서버 EngineCore 가 세그폴트로 죽었다
            # (프록시 1.3GB 시점). 상한을 두면 초과 시 이 프로세스가 OOM 을 받지,
            # 이웃이 죽지 않는다.
            torch.cuda.set_per_process_memory_fraction(self.GPU_MEM_FRACTION)
        logger.info(f"[lid] loading {self.model_name} on {self.device} ({dtype})")
        self.proc = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, dtype=dtype).to(self.device).eval()
        tok = self.proc.tokenizer
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.transcribe_id = tok.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
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

    def _lang_token_ids(self, allowed=None) -> list:
        """argmax 를 볼 토큰 집합.

        **후보를 좁히면 엉뚱한 언어로 새는 걸 막는다.** 전체 100여 개 언어에서 고르면
        한국어 발화가 zh·tr·ja 로 찍히는 일이 생기는데, 이 파이프라인에는 그 언어를
        맡을 서버가 없다. 웹이 고른 소스 언어(langMap 의 키)만 남기면 애초에 그
        선택지가 사라진다. 실측 오답 12건 중 7건이 표에 없는 언어였다.
        """
        if not allowed:
            return list(self.lang_ids)
        ids = [i for i, code in self.lang_ids.items() if code in allowed]
        return ids or list(self.lang_ids)

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """판정 창의 음량을 맞춘다(피크 0.5). 멀리서 작게 말하면 판정이 무너진다 —
        아랍어·한국어 녹음 23구간을 -20dB 로 낮추면 후보 4개에서 16/23 이 13/23 으로,
        후보 2개에서 22/23 이 21/23 으로 떨어지는데, 정규화하면 둘 다 원래 값으로
        돌아온다(16/23, 22/23). whisper 의 log-mel 이 최댓값 기준이라 크게 흔들리진
        않지만 낮은 음량에서 후보 간 로그확률 차이가 좁아진다."""
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        return audio / peak * 0.5 if peak > 1e-4 else audio

    def _classify(self, audio: np.ndarray, with_conf: bool = False, allowed=None):
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
            self._normalize(audio), sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        dec = torch.tensor([[self.sot]], device=self.device)
        with torch.inference_mode():
            logits = self.model(feats, decoder_input_ids=dec).logits[0, -1]
        # **고르는 건 좁히되, 확신도는 전체 언어 기준으로 잰다.**
        # 후보 안에서 softmax 를 다시 하면 확률이 몰려 임계값의 의미가 달라진다.
        # 실측(창 0.5초, 임계 0.8 에서 그 부분집합의 정확도):
        #     전체 언어  100%  /  5개 언어  97.1%  /  ko·en 둘  95.5%
        # 후보 수마다 임계를 다시 잡아야 하는 셈이라 취약하다. 전체 분포에서 잰
        # 확률을 쓰면 EARLY_STEPS 의 보정값이 후보 수와 무관하게 유지된다.
        all_ids = list(self.lang_ids)
        probs = torch.softmax(logits[all_ids].float(), dim=-1)
        pick_ids = self._lang_token_ids(allowed)
        pos = [all_ids.index(i) for i in pick_ids]
        k = pos[int(torch.argmax(probs[pos]))]
        lang = self.lang_ids[all_ids[k]]
        return (lang, float(probs[k])) if with_conf else lang

    # 강제 디코딩 로그확률 판정이 한 후보당 읽는 토큰 수. 12개면 문장 하나 분량이고
    # 그 뒤는 점수에 거의 안 더해진다.
    LP_MAX_TOKENS = 12
    # 이 프로세스가 쓸 수 있는 GPU 메모리 비율. whisper-small fp16 + 판정 활성화로
    # 0.9GB 안팎이 실측이라 24GB 카드에서 0.06(1.47GB)이면 넉넉하다.
    GPU_MEM_FRACTION = 0.06

    def _classify_lp(self, audio: np.ndarray, allowed=None, with_scores: bool = False):
        """언어 코드. 후보 언어마다 그 언어로 **강제 디코딩**해 토큰 평균 로그확률이
        가장 높은 쪽을 고른다.

        **언어 토큰 argmax 는 억양 있는 화자에서 무너진다.** 한국인이 말한 스페인어·영어를
        whisper-small 의 언어 토큰으로 판정하면 실제 녹음 3개(94발화)에서 1.5초 창 93.6%,
        어려운 30개만 추리면 80% 다 — `Tres meses` 는 en, `Nice to meet you` 는 ko 로 간다.
        medium 은 77%, large-v3-turbo 도 87% 라 모델을 키워도 안 풀린다.

        대신 각 후보 언어로 짧게 받아쓰게 하고 그 확률을 비교하면, 같은 30개에서 1.5초
        93.3%, 구간 전체 100% 다. 언어 토큰 하나의 분포보다 "이 언어의 문장으로 읽힐 수
        있는가" 가 억양에 훨씬 덜 흔들린다. 값은 인코더 한 번 + 후보당 최대 12토큰
        greedy 디코딩이라 GPU 에서 100~200ms 다. 스캔(0.25초마다)에는 비싸서 안 쓰고,
        발화당 한 번인 잠금 판정과 확정 판정에만 쓴다.
        """
        import time as _time
        _t0 = _time.perf_counter()
        torch = self._torch
        codes = sorted(allowed) if allowed else sorted(self.known_langs) or ["en"]
        feats = self.proc.feature_extractor(
            self._normalize(audio), sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        tok = self.proc.tokenizer
        scores = {}
        with torch.inference_mode():
            enc = self.model.model.encoder(feats)
            # 후보마다 따로 강제 디코딩한다. 후보를 한 배치로 묶으면 6개 후보에서 224ms 가
            # 129ms 로 줄지만, 디코더가 cross-attention K/V 를 행마다 다시 만들어 호출당
            # 약 300MB 를 더 잡는다. ASR 서버 셋과 번역기가 카드를 거의 다 쓰는 상태라
            # 그 300MB 가 없어 프록시가 OOM 으로 세션을 끊었다(14:49). 느려도 메모리가
            # 일정한 쪽을 택한다.
            for code in codes:
                lang_id = tok.convert_tokens_to_ids(f"<|{code}|>")
                dec = torch.tensor([[self.sot, lang_id, self.transcribe_id,
                                     self.notimestamps_id]], device=self.device)
                total, n = 0.0, 0
                for _ in range(self.LP_MAX_TOKENS):
                    logits = self.model(encoder_outputs=enc,
                                        decoder_input_ids=dec).logits[0, -1].float()
                    logp = torch.log_softmax(logits, -1)
                    nxt = int(torch.argmax(logp))
                    if nxt == self.eot:
                        break
                    total += float(logp[nxt])
                    n += 1
                    dec = torch.cat([dec, torch.tensor([[nxt]], device=self.device)], dim=1)
                scores[code] = total / n if n else -99.0
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()      # 배치 크기가 호출마다 달라 예약 메모리가 자란다
        best = max(scores, key=scores.get)
        logger.info(f"[lid-lp] {len(audio)/SR:.1f}s -> {best} "
                    f"{' '.join(f'{k}={v:.2f}' for k, v in scores.items())} "
                    f"({(_time.perf_counter() - _t0) * 1000:.0f}ms)")
        return (best, scores) if with_scores else best

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
                 step_sec: float = 0.4, allowed=None,
                 scan_win: float = 0.0, scan_hop: float = 0.25,
                 scan_confirm: int = 2, early: bool = True):
        self.router = router
        # EARLY_STEPS(0.3초부터 확신도로 조기 확정)를 쓸지. 오디오를 담당 서버에만
        # 보내는 구조에서는 첫 판정이 그 발화의 운명이라, 창(window_sec)을 다 듣고
        # 정하는 쪽이 낫다. 억양 있는 화자의 첫 0.5초는 native 실측값보다 못 믿는다.
        self.early = early
        # 이 스트림에서 나올 수 있는 언어. 클라이언트가 고른 소스 언어를 받는다.
        self.allowed = set(allowed or ())
        self.keep_sec = keep_sec
        self.step_sec = step_sec
        # 발화 구간 안을 계속 훑는 스캔. scan_win 이 0 이면 끈다.
        self.scan_win = scan_win
        self.scan_hop = scan_hop
        self.scan_confirm = max(1, scan_confirm)
        self._scan_pos = 0          # 다음에 볼 격자 칸 (절대 시각 = pos * scan_hop)
        self._scan_lang: Optional[str] = None   # 지금 유효하다고 보는 언어
        self._scan_run = 0          # 그와 다른 답이 연속으로 나온 횟수
        self._buf = np.zeros(0, dtype=np.float32)
        self._offset = 0.0          # 잘라낸 앞부분의 길이(초)
        self._audio_sec = 0.0       # 지금까지 받은 전체 오디오 길이(초)
        self._last_run = -1.0
        self._last_segments: list[tuple] = []   # 마지막 VAD 스캔 결과 (재사용용)
        self.verdicts: list[list] = []   # [시작초, 끝초 또는 None, 언어]
        self._ambiguous: set = set()     # 첫 판정이 애매했던 verdict 의 id
        self._waiting: set = set()       # 다음 발화를 기다린다고 이미 적은 조각의 시작
        self._pending: dict = {}         # 기다리는 조각의 시작 -> 지금까지의 최선 언어

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

    # 발화가 닫히면 구간 전체(최대 이 길이)를 강제 디코딩 로그확률로 다시 판정한다.
    # 잠금 판정은 1.5초만 듣고 정한 값이라 실제 녹음(어려운 30발화)에서 93.3% 인데,
    # 구간 전체를 들으면 100% 였다. 오디오는 이미 잠금 쪽으로 갔으므로 이 값은
    # 라우팅을 바꾸지 못하고, Dispatcher 가 이 값과 잠금이 다를 때 그 발화를 맞는
    # 서버로 다시 보내는 교정(correction)에 쓴다.
    CONFIRM_SEC = 6.0

    def _classify_span(self, start: float, end: float, early: bool = True,
                       secs: Optional[float] = None, join_end: Optional[float] = None):
        """(언어, 얼마나 듣고 정했나, 여유) 또는 (None, None, None).

        여유는 로그확률 판정에서 1등과 2등의 차이다. 확신도로 조기 확정한 값은
        여유가 없으므로 None 이다. `join_end` 가 있으면 그 시각까지의 다음 발화
        음성을 이 구간 뒤에 이어 붙여 판정한다(`_join_range` 참고).

        짧은 창부터 올라가며 확신도가 임계를 넘으면 즉시 확정한다. whisper 는 입력을
        30초로 패딩하므로 창을 줄여도 추론 시간이 같다(9ms) — 여러 창을 시도해도
        비용이 거의 안 는다.
        """
        a = int((start - self._offset) * SR)
        b = int((end - self._offset) * SR)
        span = self._buf[max(0, a):max(0, b)]
        if len(span) < SR * 0.2:
            return None, None, None
        if join_end is not None:
            c = int((join_end - self._offset) * SR)
            span = np.concatenate([span, self._buf[max(0, b):max(0, c)]])
        avail = len(span) / SR
        if secs is not None:                  # 확정 판정: 정해진 길이로 한 번만
            with self.router._gpu_lock:
                lang, scores = self.router._classify_lp(span[: int(SR * secs)],
                                                        allowed=self.allowed,
                                                        with_scores=True)
            return lang, min(avail, secs), self._margin(scores)
        with self.router._gpu_lock:
            if early:
                for secs, need in self.EARLY_STEPS:
                    if secs > avail or secs >= self.router.window_sec:
                        continue
                    lang, conf = self.router._classify(
                        span[: int(SR * secs)], with_conf=True, allowed=self.allowed)
                    if conf >= need:
                        return lang, secs, None
            take = self.router.window_sec + (self.AMBIG_EXTRA_SEC if avail > self.router.window_sec + 0.2 else 0.0)
            lang, scores = self.router._classify_lp(
                span[: int(SR * take)], allowed=self.allowed, with_scores=True)
        return lang, min(avail, take), self._margin(scores)

    @staticmethod
    def _margin(scores: dict) -> float:
        vals = sorted(scores.values(), reverse=True)
        return (vals[0] - vals[1]) if len(vals) > 1 else 99.0

    # 로그확률 판정에서 1등과 2등의 차이가 이보다 작으면 못 믿는다. 아랍어·한국어
    # 실측에서 오답 3건의 여유는 0.05·0.21·0.32 였고, 정답의 대부분은 0.5 이상이었다.
    # 정답 중에도 0.12~0.35 가 몇 건 있어 여유만으로는 못 가르므로, 이 아래에서는
    # 더 들어서(다음 발화를 이어 붙여) 다시 판정한다. 0.4 로 두니 실제 마이크
    # 세션에서 판정의 77% 가 애매로 잡혀 대부분의 발화가 1초씩 더 기다렸다(6개 언어
    # 세션 89발화 중 61건). 0.25 면 알려진 오답(0.04~0.21)은 다 잡고 대기는 줄어든다.
    AMBIG_MARGIN = 0.25

    # 창(window_sec)보다 짧게 닫힌 조각이 애매하면, 이만큼 안에 다음 발화가
    # 시작하는지 기다렸다가 그 앞머리를 이어 붙여 판정한다. silero 는 한 문장의
    # 숨 고르는 자리에서도 끊는다 — `إنت ساكن فين؟` 의 앞 1.1초가 그렇게 홀로
    # 떨어져 ko 로 갔다(여유 0.21). 이어 붙이면 아랍어로 읽힌다.
    JOIN_GAP = 1.0
    # 이어 붙일 다음 발화의 최소 길이. 0.4초만 붙이면 여전히 애매했고(0.21 → 0.22),
    # 1.5초를 붙인 확정 판정은 여유 1.3 으로 갈렸다.
    JOIN_NEXT_SEC = 0.8
    # 붙여서 나온 값이 조각만 본 값과 다를 때 받아들이는 최소 여유. 실측 5건: 틀린 뒤집기
    # (`El sábado` + `Saturday works` → en) 는 0.08·0.11, 맞는 뒤집기는 0.31·1.00·1.31.
    # AMBIG_MARGIN(0.4) 을 그대로 쓰면 0.31 짜리 `¿Es tu primera vez aquí?` 가 en 으로
    # 남았다가 2초 뒤 교정으로 바뀐다.
    JOIN_ACCEPT_MARGIN = 0.25
    # 말하는 중인 발화의 첫 창이 애매하면 이만큼 더 듣고 잠근다. 창을 채우는 판정은
    # 창 길이(window_sec)까지만 보므로, 여기서 늘린 만큼이 실제로 더 들리는 길이다.
    AMBIG_EXTRA_SEC = 1.0

    def pending_lang(self, start: float) -> Optional[str]:
        """다음 발화를 기다리는 중인 조각의 지금까지 최선 언어. Dispatcher 가 더 못
        기다릴 때(hold timeout) 직전 발화 언어 대신 이걸 쓴다 — `Saturday works for
        me` 가 앞 스페인어를 물려받아 베이스 서버로 가서 사라진 일이 있다."""
        for key, lang in self._pending.items():
            if abs(key - start) < 0.4:
                return lang
        return None

    def _join_range(self, start: float, end: float):
        """짧게 닫힌 조각 [start, end] 뒤에 이어 붙일 다음 발화의 끝 시각, 또는 None.

        다음 발화가 JOIN_GAP 안에 시작하지 않으면 None 이다. 아직 그만큼 시간이
        안 지났으면 "wait" 를 돌려 판정을 미루게 한다.
        """
        if (end - start) >= self.router.window_sec:
            return None
        for s, e, _closed in self._last_segments:
            if s <= end:
                continue
            if s - end > self.JOIN_GAP:
                return None
            take = min(e, s + self.router.window_sec)
            if take - s < self.JOIN_NEXT_SEC:
                return "wait"                 # 다음 발화가 아직 너무 짧다
            return take
        if self._audio_sec - end < self.JOIN_GAP + 0.3:
            return "wait"
        return None

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

    def _clamp_end(self, v: list, end: float) -> float:
        """뒤에 다른 판정이 이미 있으면 그 시작을 넘겨 늘리지 않는다.

        확정 판정은 구간 끝까지 늘리는데, 스캔이 그 구간을 이미 쪼갠 뒤라면 뒤쪽
        판정을 통째로 덮어 버린다.
        """
        nxt = min((x[0] for x in self.verdicts if x is not v and x[0] > v[0]),
                  default=None)
        return end if nxt is None else min(end, nxt)

    def _split_at(self, at: float, lang: str, seg_end: Optional[float]) -> None:
        """시각 `at` 부터 언어가 `lang` 으로 바뀐 것으로 판정 목록을 자른다."""
        v = self._find_overlap(at, at + 1e-3)
        if v is not None and v[2] == lang:
            return
        if v is not None and at - v[0] < self.scan_hop:
            v[2] = lang                 # 구간 머리에서 바뀌면 그 항목을 고친다
            return
        # 새 판정은 쪼갠 판정의 끝을 그대로 물려받는다. **아직 말하는 중이면
        # None 이어야 한다** — 그 자리의 구간 끝으로 굳혀 두면 구간이 더 자라도
        # 늘지 않아, 뒤쪽 오디오가 어느 판정에도 안 걸린다.
        new_end = v[1] if v is not None else None
        if v is not None:
            v[1] = at
        # 스캔이 낸 값은 확정으로 본다. 창 1.5초는 확정 판정(2초)과 같은 급이라
        # 구간 단위 확정 판정이 다시 덮어쓰면 안 된다. 여섯째 자리는 "발화 한복판을
        # 쪼갠 것" 이라는 표시다 — 이 경우에만 서버가 앞 언어 텍스트를 커밋해야 한다.
        self.verdicts.append([at, new_end, lang, True, True, True])
        self.verdicts.sort(key=lambda x: x[0])
        logger.debug(f"[scan] {at:.2f}s -> {lang}")

    def _scan_sync(self) -> None:
        """발화 구간 안을 창으로 계속 훑어 언어가 바뀌는 자리를 잡는다.

        **구간 단위 판정만으로는 구간 안의 전환을 못 본다.** 앞머리에서 한 번 정하고
        나면 뒤에서 언어가 바뀌어도 다시 보지 않는다. 여기서는 절대 시각 격자 위를
        `scan_hop` 마다 한 칸씩 나아가며 직전 `scan_win` 초를 판정한다.

        실측(합성 전환 120쌍, ko/en/es): 창 1.5초·홉 0.25초·연속 2회 확인에서
        감지 100%, 헛플립 쌍당 0.12건, 전환 인지 지연 중앙 1.07초, 지점 오차 p90
        0.62초. 창을 2.0초로 키우면 헛플립이 0 에 가까워지는 대신 지연이 1.68초로
        늘고 오차 p90 도 1.09초로 넓어진다.

        **연속 확인은 바뀔 때만 요구한다.** 처음 언어를 정하는 건 구간 단위 조기
        판정(EARLY_STEPS)이 이미 훨씬 빨리 해내므로, 여기서는 그 값을 출발점으로
        삼고 그와 다른 답이 `scan_confirm` 번 연달아 나올 때만 전환으로 본다.
        한 번 튄 값으로 슬롯을 자르면 멀쩡한 문장이 두 동강 난다.
        """
        W, H, K = self.scan_win, self.scan_hop, self.scan_confirm
        buf_end = self._offset + len(self._buf) / SR
        while True:
            t = self._scan_pos * H
            if t + W > buf_end:
                break
            if t < self._offset:            # 버퍼에서 이미 밀려난 자리
                self._scan_pos += 1
                continue
            # 창이 발화 구간 안에 온전히 들어갈 때만 본다. 침묵이 섞이면 판정이
            # 뒤집힌다 — 무음은 하나의 확신 있는 오답으로 매핑된다.
            seg = next(((s, e) for s, e, _c in self._last_segments
                        if s <= t and t + W <= e), None)
            if seg is None:
                self._scan_pos += 1
                self._scan_run = 0
                continue
            a = int((t - self._offset) * SR)
            with self.router._gpu_lock:
                lang = self.router._classify(self._buf[a:a + int(W * SR)],
                                             allowed=self.allowed)
            cur = self._scan_lang or self.lang_for_end(t + W)
            if lang == cur or cur is None:
                self._scan_lang = lang if cur is None else cur
                self._scan_run = 0
            else:
                self._scan_run += 1
                if self._scan_run >= K:
                    # 언어 토큰 argmax 두 번으로는 부족하다. 아랍어 `ببطء ممكن` 한복판이
                    # 그렇게 ko 로 잘려 ko 서버가 `알람이 빛을` 을 냈다. 자르기 전에
                    # 같은 창을 강제 디코딩 로그확률로 한 번 더 본다.
                    with self.router._gpu_lock:
                        lp, scores = self.router._classify_lp(
                            self._buf[a:a + int(W * SR)], allowed=self.allowed,
                            with_scores=True)
                    margin = self._margin(scores)
                    if lp != lang or margin < self.AMBIG_MARGIN:
                        logger.info(f"[scan] {t + W / 2:.2f}s {cur} -> {lang} rejected "
                                    f"(lp says {lp}, margin {margin:.2f})")
                        self._scan_run = 0
                        self._scan_pos += 1
                        continue
                    # 추정식은 실측에 쓴 것과 같다 — 마지막 옛 언어 창과 첫 새 언어
                    # 창의 중심을 잇는 중간점. 확인이 K 회면 그 사이가 홉 하나다.
                    self._split_at(t + W / 2 - H / 2, lang, seg[1])
                    self._scan_lang = lang
                    self._scan_run = 0
            self._scan_pos += 1
        # 구간이 닫혔는데 끝이 안 정해진 판정은 그 구간 끝으로 닫는다. 구간 단위
        # 로직은 구간마다 판정 하나만 손보므로, 쪼개서 생긴 뒤쪽 판정이 계속 열린
        # 채 남는다. 열린 판정은 지금까지 받은 오디오 전체를 덮는 것으로 쳐서
        # 뒤따르는 다른 발화까지 집어삼킨다.
        for s, e, closed in self._last_segments:
            if not closed:
                continue
            for v in self.verdicts:
                if v[1] is None and s - 1e-3 <= v[0] < e:
                    v[1] = e

    def _update_sync(self) -> None:
        self._last_segments = self._segments_sync()
        for start, end, closed in self._last_segments:
            if (end - start) <= self.MIN_SPEECH_SEC:
                continue          # 숨소리·잡음 조각. 여기서 나온 판정은 못 믿는다.
            existing = self._find_overlap(start, end)
            if existing is not None and len(existing) > 4 and existing[4]:
                continue                      # 이미 확정 판정까지 끝난 구간
            if closed and existing is not None:
                # **발화가 닫혔으면 더 듣고 다시 정한다.** 조기 판정은 앞 1초 안쪽만
                # 보므로 짧은 근거로 정해진 값이다. final 은 발화가 닫힌 뒤에 오니
                # 여기서 고쳐도 늦지 않다.
                # 첫 판정이 애매했던 조각만 다음 발화를 붙여 본다. 확신 있던 짧은
                # 발화("네.")까지 기다리면 그 final 이 1초씩 늦어진다.
                join_end = (self._join_range(start, end)
                            if id(existing) in self._ambiguous else None)
                if join_end == "wait":
                    continue              # 짧은 조각. 다음 발화를 붙일 수 있는지 기다린다
                lang, used, margin = self._classify_span(
                    start, end, secs=self.CONFIRM_SEC, join_end=join_end)
                if (lang is not None and existing[2] != lang
                        and margin is not None and margin < self.AMBIG_MARGIN):
                    # 애매한 값으로 잠금을 뒤집으면 맞는 final 이 교정에 버려진다 —
                    # 실측에서 `وين ساكن؟` 이 여유 0.05 짜리 ko 확정에 지워졌다.
                    logger.info(f"[confirm] {start:.1f}s keep {existing[2]} "
                                f"(ambiguous {lang}, margin {margin:.2f})")
                    lang = existing[2]
                if lang is not None and existing[2] != lang:
                    # 잠금은 이미 오디오를 보냈으니 못 바꾼다. 교정은 Dispatcher 몫이다.
                    existing_prev = existing[2]
                    logger.info(f"[confirm] {start:.1f}s {existing_prev} -> {lang}")
                if lang is not None:
                    existing[0] = min(existing[0], start)
                    # 스캔이 이 구간을 쪼갰으면 뒤쪽 판정 시작을 넘겨 늘리지 않는다.
                    existing[1] = self._clamp_end(existing, end)
                    existing[2] = lang
                    while len(existing) < 5:
                        existing.append(False)
                    existing[4] = True
                continue
            if existing is not None and existing[1] is not None:
                continue                      # 닫혔는데 판정도 못 한 구간
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
            if not self.early and not closed and (end - start) < self.router.window_sec:
                continue          # 창을 채울 때까지 기다린다
            lang, used, margin = self._classify_span(start, end, early=self.early)
            if lang is None:
                continue
            ambiguous = margin is not None and margin < self.AMBIG_MARGIN
            if (ambiguous and not closed
                    and (end - start) < self.router.window_sec + self.AMBIG_EXTRA_SEC):
                # 말하는 중인데 첫 창(1.5초)이 애매하다. 조금 더 듣고 정한다 — 한국어
                # `안녕하세요. 저희 대학교에…` 의 앞 1.5초가 여유 0.08 로 en 에 잠겨
                # 영어 서버가 문장 전체를, 스캔 전환으로 한국어 서버가 꼬리를 한 번 더 냈다.
                key = round(start, 1)
                if key not in self._waiting:
                    self._waiting.add(key)
                    logger.info(f"[lid-ambig] {start:.1f}s {lang} margin {margin:.2f}, "
                                f"listening up to {self.router.window_sec + self.AMBIG_EXTRA_SEC:.1f}s")
                continue
            if closed and ambiguous:
                join_end = self._join_range(start, end)
                key = round(start, 1)
                if join_end == "wait":
                    self._pending[key] = lang
                    if key not in self._waiting:
                        self._waiting.add(key)
                        logger.info(f"[lid-ambig] {start:.1f}s {lang} margin {margin:.2f}, "
                                    f"waiting for the next segment")
                    continue
                self._pending.pop(key, None)
                if join_end is not None:
                    lang2, used, margin2 = self._classify_span(
                        start, end, secs=self.CONFIRM_SEC, join_end=join_end)
                    # **붙인 결과도 애매하면 조각만 본 값을 지킨다.** 다음 발화가 다른
                    # 언어면 붙인 오디오는 두 언어가 섞인 것이라 답이 뒤쪽 언어로 넘어간다
                    # — `El sábado a las dos` + `Saturday works for me` 가 여유 0.08 로
                    # en 이 돼 스페인어가 영어 서버로 갔다. 붙여서 확실해질 때만
                    # (아랍어 조각: 0.21 → 1.31, 0.05 → 0.48) 그 값을 쓴다.
                    accept = (lang2 is not None and margin2 is not None
                              and margin2 >= self.JOIN_ACCEPT_MARGIN)
                    logger.info(f"[lid-join] {start:.1f}s {lang} (margin {margin:.2f}) "
                                f"+ next to {join_end:.1f}s -> {lang2} "
                                f"(margin {margin2:.2f}){'' if accept else ', kept ' + lang}")
                    if accept:
                        lang, margin = lang2, margin2
                # 직전 발화의 언어를 물려받는 쪽도 해 봤는데, 두 사람이 문장마다 언어를
                # 바꾸는 대화에서 `El sábado a las dos` 가 앞 한국어를 물려받아 ko 서버로
                # 갔다(4건 오답). 애매하면 조각 자체의 값을 쓴다.
            # 창을 다 안 채우고 정해졌다면 확신도로 일찍 확정된 것이다.
            # 다만 라우팅 표에 없는 언어로 나왔으면 잠그지 않는다 — 한국어 발화가
            # 0.5초 만에 zh 로 확정돼 그대로 굳는 일이 실제로 있었다. 그런 값은
            # 더 들으면 바뀔 여지를 남긴다.
            known = self.allowed or self.router.known_langs
            locked = (used is not None and used < self.router.window_sec
                      and (not known or lang in known))
            # 창을 다 듣고 로그확률로 정한 값(early 꺼짐)은 그대로 잠근다. 갱신마다
            # 다시 돌리면 발화가 이어지는 동안 0.25초마다 200ms 를 GPU 에 쓴다 —
            # 어차피 오디오는 첫 판정으로 갔고, 다시 보는 건 확정 판정 몫이다.
            if not self.early:
                locked = True
            if existing is None:
                self.verdicts.append([start, end if closed else None, lang, locked])
                if ambiguous:
                    self._ambiguous.add(id(self.verdicts[-1]))
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
        if self.scan_win:
            self._scan_sync()

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
            v = self._find_overlap(start, seg_end)
            if v is None:
                return False
            if len(v) < 5 or not v[4]:
                return False                  # 확정 판정이 아직이다
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
