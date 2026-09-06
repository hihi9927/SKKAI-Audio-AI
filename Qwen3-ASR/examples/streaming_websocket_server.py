# coding=utf-8
"""
Qwen3-ASR Real-time Streaming WebSocket Server

WhisperLiveKit 앱과 호환되는 실시간 오디오 스트리밍 서버.
짧은 간격으로 오디오를 수신하고, 일정 청크만큼 모아서 transcribe 수행.

Usage:
    python streaming_websocket_server.py --host 0.0.0.0 --port 8765

Client protocol (WhisperLiveKit app compatible):
    1. Connect to WebSocket
    2. Send JSON: {"type": "start", "lang": "auto", "polish": true, "translate": true}
    3. Send binary audio chunks (PCM s16le, 16kHz, mono)
    4. Receive JSON: {"type": "partial", "text", "language", "seq"}  (미확정 가설, 통째 교체.
       text 가 빈 문자열이면 화면을 비우라는 신호)
       또는 {"type": "final", "start", "end", "original", "translation", "language", "commitReason"}
    5. Send JSON: {"type": "stop"} or {"type": "finish"} to end session
"""

import argparse
import asyncio
import contextlib
import difflib
import http
import json
import logging
import os
import re
import time
import traceback
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any

import io
import numpy as np
import aiohttp
import torch
import subprocess

try:
    import websockets
except ImportError:
    raise ImportError("websockets 패키지가 필요합니다: pip install websockets")

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import warmup_streaming
from qwen_asr.inference.sentence_boundary import (
    DOT_COMMIT_BOUNDARY_RE,
    count_dot_commit_boundaries,
    split_sentences,
)

try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from core.translator import GPTCorrector
    from core.translator import GPTTranslator
    _CORRECTOR_AVAILABLE = True
except Exception:
    _CORRECTOR_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
# silero-vad: 최신 API (v3+) 사용 — VADIterator 기반 스트리밍
# ──────────────────────────────────────────────────────────────────────
try:
    from silero_vad import load_silero_vad, VADIterator
    _SILERO_VAD_AVAILABLE = True
except ImportError:
    _SILERO_VAD_AVAILABLE = False

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_SERVER_DIR, "../logs/asr_server.log")


_BASELINE_MODEL_IDS = {
    "qwen/qwen3-asr-1.7b",
    "qwen3-asr-1.7b",
    "baseline",
    "baseline(1.0.0)",
}


class _JsonFormatter(logging.Formatter):
    """JSON 한 줄 포맷 로그 포매터"""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def _configure_logging(use_json: bool = False, log_file: Optional[str] = None) -> None:
    fmt: logging.Formatter = (
        _JsonFormatter() if use_json
        else logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    target_log_file = log_file or _LOG_FILE
    os.makedirs(os.path.dirname(os.path.abspath(target_log_file)), exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(target_log_file),
    ]
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
MAX_AUDIO_ACCUM_SEC = 90.0          # audio_accum 강제 리셋 임계값 (초)
MAX_SEED_COMMITTED_SENTENCES = 1    # 강제 리셋 시 seed_text에 포함할 직전 committed 문장 수
PARTIAL_MIN_INTERVAL_SEC = 0.12     # partial(미확정 가설) 전송 최소 간격 (초)

# VADIterator 설정
VAD_THRESHOLD = 0.5
# 발화 종료 판정까지 필요한 침묵 길이. --vad-min-silence 로 덮어쓴다.
# 짧게 잡을수록 발화가 자주 끊겨 slot 이 자주 초기화되고(언어 전환이 빨리 풀린다)
# 문장은 잘게 쪼개진다. 길게 잡으면 그 반대다.
VAD_MIN_SILENCE_MS = 800
VAD_SPEECH_PAD_MS = 160        # 발화 경계에 추가하는 패딩
VAD_WINDOW_SIZE_SAMPLES = 512  # 16kHz 기준 silero 권장 윈도우 크기


def _infer_dot_commit_default(model_path: str) -> bool:
    """Enable dot commit by default for baseline model variants."""
    normalized = (model_path or "").replace("\\", "/").strip().lower()
    if not normalized:
        return False
    if normalized in _BASELINE_MODEL_IDS:
        return True
    model_name = normalized.rsplit("/", 1)[-1]
    return model_name in _BASELINE_MODEL_IDS or "baseline" in model_name


@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    # 모델 설정
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    gpu_memory_utilization: float = 0.8
    # per-chunk 생성 토큰 상한. 32는 dense/긴 발화에서 꼬리가 잘려(truncation) 128로 상향.
    max_new_tokens: int = 128

    # LoRA 어댑터 경로 (examples/ 디렉토리 기준 상대경로)
    adapter_en: str = "../finetuning/finetuning-out-en-plus/checkpoint-420_vllm"
    adapter_ko: str = "../finetuning/finetuning-out-ko-plus/checkpoint-300_vllm"
    no_lora: bool = True  # True면 어댑터 경로 무시하고 기본 모델만 사용
    max_lora_rank: int = 128  # 학습 시 사용한 LoRA rank

    # 스트리밍 설정
    chunk_size_sec: float = 2.0
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5

    # 빔 서치 설정
    beam_size: int = 2  # 1이면 greedy, 2+ 이면 beam search

    # VAD 설정
    no_vad: bool = False  # True면 silero-vad 비활성화 (VAD 없이 SEG/finish 커밋만 사용)

    # vLLM 컴파일 설정
    enforce_eager: bool = False  # True면 Triton 컴파일 우회 (sm_121a 등 미지원 GPU)

    # Commit 방식 설정
    enable_dot_commit: bool = False  # True면 온점/느낌표/물음표(dot) 기반 seg commit 활성화
    always_commit: bool = False  # True면 SEG/dot 트리거 없이 매 청크 디코딩 결과를 그대로 커밋 (모드2)
    # True면 dot을 감지 즉시 커밋하지 않고 "확정"된 뒤에만 커밋한다.
    # 모델은 청크 버퍼가 끝나면 문장이 미완성이어도 마침표를 붙여 마무리짓는 습성이 있어,
    # 프론티어 마침표는 정보량이 없다(P(마침표|버퍼 끝) ≈ 1). 확정 경로는 3개:
    #   1) 문맥 확정 — 마침표 뒤 토큰이 unfixed_token_num보다 많으면 롤백 창 밖 → 즉시 커밋
    #   2) 합의 확정 — 프론티어 마침표는 보류했다가 다음 청크에서도 동일하면 커밋
    #   3) 정체 확정 — 오디오는 계속 들어오는데 미커밋 가설의 길이가 N청크 동안 그대로면
    #      발화가 끝난 것으로 보고 커밋 (규칙 2가 문구 미세수정 때문에 못 잡는 경우 보완)
    #   4) finish — 스트림 종료 시 flush (안전망. 위 3개가 동작하면 발생하지 않아야 함)
    dot_commit_confirm: bool = False
    # 규칙 3(정체 확정)이 발동하기까지 필요한 "가설이 자라지 않은" 연속 청크 수.
    # 0이면 규칙 3 비활성화. 오디오 에너지를 보지 않으므로 VAD 의존성은 없다.
    dot_commit_stall_chunks: int = 1
    # 한 번의 추출에서 같은 문장이 연속으로 나올 때 뒤엣것을 버린다. 모델이 반복 루프에
    # 빠지는 경우(ko-silence 파인튜닝 실측 8,047회)를 막는 방어라 프로덕션에선 켜 둔다.
    # 다만 루프가 없는 모델에선 정상 발화를 깎는다 — baseline 27건 전부 오탐이었고
    # ('Ha!' 25 / 'Yes.' 1 / 'I hate him.' 1) 낭독체의 실제 반복이었다. 게다가 스킵이
    # 커서 갭을 만들어 뒤 문장이 finish로 밀리는 부작용까지 있었다. 평가에선 끌 수 있게 둔다.
    rep_dedup: bool = True


    # 언어 제한 설정
    restrict_languages: bool = True  # True면 앱 설정 두 언어 외 토큰 차단

    # LLM 후처리 설정
    enable_correction: bool = False
    correction_model: str = "gpt-5.4-mini"
    api_key: Optional[str] = None

    # GPT 번역 설정 (활성화 시 GPTCorrector + Google Translate 대신 단일 GPT 호출)
    enable_gpt_translation: bool = False
    translation_model: str = "gpt-5.4-mini"
    context_window: int = 5

    # Google Translate 컨텍스트 활성화 (--google-context 플래그, 문장 수는 context_window 공유)
    google_context: bool = False

    # 오디오 녹음 설정 (로그 분석용)
    record_audio: bool = False  # True면 수신 PCM을 세션별 WAV로 저장

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8765
    no_idle_shutdown: bool = False
    idle_shutdown_sec: int = 60
    close_timeout: Optional[int] = None


def format_time(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


LANG_NAME_TO_CODE = {
    "Korean": "ko", "English": "en", "Japanese": "ja",
    "Cantonese": "zh", "Chinese": "zh",
    "Indonesian": "id", "Vietnamese": "vi", "Thai": "th",
    "Spanish": "es", "French": "fr", "German": "de",
    "Arabic": "ar", "Portuguese": "pt", "Italian": "it",
    "Russian": "ru", "Turkish": "tr", "Hindi": "hi",
    "Malay": "ms", "Dutch": "nl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Polish": "pl",
    "Czech": "cs", "Filipino": "tl", "Persian": "fa",
    "Greek": "el", "Romanian": "ro", "Hungarian": "hu",
    "Macedonian": "mk",
}


LANG_CODE_TO_NAME = {v: k for k, v in LANG_NAME_TO_CODE.items()}


def lang_to_code(lang: str) -> str:
    """언어 이름을 코드로 변환 (Korean -> ko, Australian English -> en)"""
    if not lang:
        return ""
    mapped = LANG_NAME_TO_CODE.get(lang)
    if mapped:
        return mapped
    lower = lang.lower()
    # "Australian English", "British English" 등 variant 처리
    for keyword, code in (
        ("english", "en"), ("korean", "ko"), ("japanese", "ja"),
        ("chinese", "zh"), ("mandarin", "zh"), ("cantonese", "zh"),
        ("french", "fr"), ("german", "de"), ("spanish", "es"),
        ("vietnamese", "vi"), ("indonesian", "id"), ("thai", "th"),
        ("arabic", "ar"), ("portuguese", "pt"), ("italian", "it"),
        ("russian", "ru"), ("turkish", "tr"), ("hindi", "hi"),
        ("malay", "ms"), ("dutch", "nl"), ("swedish", "sv"),
        ("danish", "da"), ("finnish", "fi"), ("polish", "pl"),
        ("czech", "cs"), ("filipino", "tl"), ("persian", "fa"),
        ("greek", "el"), ("romanian", "ro"), ("hungarian", "hu"),
        ("macedonian", "mk"),
    ):
        if keyword in lower:
            return code
    logger.warning(f"[lang_to_code] 알 수 없는 언어명: {lang!r} — 빈 문자열 반환")
    return ""


def lang_code_to_name(code: str) -> Optional[str]:
    """언어 코드를 이름으로 변환 (ko -> Korean). auto이거나 매핑 없으면 None."""
    if not code or code == "auto":
        return None
    return LANG_CODE_TO_NAME.get(code)


def _norm_lang_code(value: object) -> str:
    """언어 코드로 정규화. 이름("Korean")으로 와도 받는다. 모르면 빈 문자열."""
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if not v or v == "auto":
        return ""
    if v in LANG_CODE_TO_NAME:
        return v
    return lang_to_code(v)


def parse_lang_map(raw: object) -> dict[str, str]:
    """{"ko": "en", "ja": "ko"} 형태의 감지 언어별 번역 목표를 정규화한다.

    "내 언어 <-> 상대 언어" 쌍으로는 세 언어 이상을 각각 다른 곳으로 보낼 수 없다.
    이 매핑이 있으면 감지 언어를 키로 목표를 바로 고른다(_correct_and_translate).
    자기 자신으로 가는 항목과 모르는 코드는 버린다.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        src = _norm_lang_code(k)
        dst = _norm_lang_code(v)
        if not src or not dst or src == dst:
            continue
        out[src] = dst
    return out


class SessionLogger:
    """앱에서 수신한 로그를 세션별 JSON 파일로 저장"""

    def __init__(self, client_id: int = 0, logs_dir: str = os.path.join(_SERVER_DIR, "../logs/asr_logs")):
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.client_id = client_id
        self.path = os.path.join(logs_dir, f"session_{ts}.json")
        self._entries: list[dict] = []
        self._lock = asyncio.Lock()
        logger.info(f"[C{client_id}] [session-log] log file: {self.path}")

    async def append(self, time: str, text: str, translation: str) -> None:
        async with self._lock:
            self._entries.append({
                "client": f"C{self.client_id}",
                "time": time,
                "text": text,
                "translation": translation,
            })
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)

    async def append_tts(self, text: str, lang: str, start: str, end: str) -> None:
        async with self._lock:
            self._entries.append({
                "client": f"C{self.client_id}",
                "type": "tts",
                "text": text,
                "lang": lang,
                "start": start,
                "end": end,
            })
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)


class AudioRecorder:
    """수신 PCM(s16le, 16kHz mono)을 세션별 WAV 파일로 저장.

    세션 로그(session_{ts}.json)와 동일한 타임스탬프 stem을 사용해
    로그 ↔ 오디오를 1:1로 매칭할 수 있게 한다.
    """

    def __init__(
        self,
        session_log_path: str,
        client_id: int = 0,
        sample_rate: int = SAMPLING_RATE,
        audio_dir: str = os.path.join(_SERVER_DIR, "../logs/asr_audio"),
    ):
        os.makedirs(audio_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(session_log_path))[0]
        self.path = os.path.join(audio_dir, f"{stem}.wav")
        self.client_id = client_id
        self._closed = False
        self._wav = wave.open(self.path, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)  # s16le = 2 bytes/sample
        self._wav.setframerate(sample_rate)
        logger.info(f"[C{client_id}] [audio-rec] wav file: {self.path}")

    def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            self._wav.writeframes(pcm)
        except Exception as e:
            logger.warning(f"[C{self.client_id}] [audio-rec] write failed: {e}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._wav.close()


class PairingHub:
    """Lightweight in-memory signaling hub for 2-earphone room pairing."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._rooms: dict[str, dict[str, Any]] = {}
        self._ws_to_room: dict[Any, str] = {}

    async def _safe_send(self, websocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"[pairing] send failed: {e}")

    async def _detach_locked(self, websocket, notify_peer: bool) -> None:
        room_id = self._ws_to_room.pop(websocket, None)
        if not room_id:
            return

        room = self._rooms.get(room_id)
        if not room:
            return

        peer = None
        if room.get("host_ws") is websocket:
            room["host_ws"] = None
            room["host_cfg"] = None
            peer = room.get("guest_ws")
        elif room.get("guest_ws") is websocket:
            room["guest_ws"] = None
            peer = room.get("host_ws")

        if notify_peer and peer is not None:
            await self._safe_send(peer, {"type": "pair_peer_left", "roomId": room_id})

        if room.get("host_ws") is None and room.get("guest_ws") is None:
            self._rooms.pop(room_id, None)

    async def register_host(
        self, websocket, room_id: str, my_lang: str, target_lang: str, mode: str
    ) -> None:
        async with self._lock:
            await self._detach_locked(websocket, notify_peer=False)
            room = self._rooms.setdefault(room_id, {"host_ws": None, "guest_ws": None, "host_cfg": None})
            room["host_ws"] = websocket
            room["host_cfg"] = {
                "myLang": my_lang,
                "targetLang": target_lang,
                "mode": mode or "mode-2",
            }
            self._ws_to_room[websocket] = room_id

        await self._safe_send(websocket, {"type": "pair_hosted", "roomId": room_id})
        logger.info(f"[pairing] host registered room={room_id} my={my_lang} target={target_lang}")

    async def join_room(self, websocket, room_id: str, guest_my_lang: str) -> None:
        async with self._lock:
            await self._detach_locked(websocket, notify_peer=False)
            room = self._rooms.get(room_id)
            if not room or room.get("host_ws") is None or room.get("host_cfg") is None:
                await self._safe_send(
                    websocket,
                    {"type": "pair_error", "roomId": room_id, "message": "room_not_found"},
                )
                return

            prev_guest = room.get("guest_ws")
            if prev_guest is not None and prev_guest is not websocket:
                self._ws_to_room.pop(prev_guest, None)
                await self._safe_send(
                    prev_guest,
                    {"type": "pair_error", "roomId": room_id, "message": "replaced_by_new_guest"},
                )

            room["guest_ws"] = websocket
            self._ws_to_room[websocket] = room_id

            host_ws = room["host_ws"]
            host_cfg = room["host_cfg"]
            host_my_lang = host_cfg["myLang"]

        host_payload = {
            "type": "pair_connected",
            "roomId": room_id,
            "role": "host",
            "mode": host_cfg["mode"],
            "myLang": host_my_lang,
            "targetLang": guest_my_lang,
        }
        guest_payload = {
            "type": "pair_connected",
            "roomId": room_id,
            "role": "guest",
            "mode": host_cfg["mode"],
            "myLang": guest_my_lang,
            "targetLang": host_my_lang,
        }
        await asyncio.gather(
            self._safe_send(host_ws, host_payload),
            self._safe_send(websocket, guest_payload),
        )
        logger.info(
            f"[pairing] connected room={room_id} host_my={host_my_lang} guest_my={guest_my_lang}"
        )

    async def leave(self, websocket) -> None:
        async with self._lock:
            await self._detach_locked(websocket, notify_peer=True)


# ── 번역 백엔드 ────────────────────────────────────────────────────────────────
# gtx 는 구글 번역 위젯이 쓰는 비공식 엔드포인트다. 키 없이 공짜로 되지만 할당량이
# 문서화돼 있지 않고, 호출이 몰리면 429 와 함께 HTML 차단 페이지를 돌려준다. 그러면
# `resp.json()` 이 터지고 → 아래 except 가 빈 문자열을 반환하고 → **앱에는 원문만 뜨고
# 번역칸만 빈다.** 서버는 죽지 않으므로 로그를 보지 않으면 원인을 알기 어렵다.
#
# 키가 있으면 공식 Cloud Translation Basic(v2) 로 간다. 월 50만 자 무료, 이후 백만 자당
# $20. 평가 경로(`evaluation/ast/trans_guard.py`)가 쓰던 것과 같은 엔드포인트다.
GTX_URL = "https://translate.googleapis.com/translate_a/single"
V2_URL = "https://translation.googleapis.com/language/translate/v2"

# 환경변수로 주는 게 기본. `--google-api-key` 로도 덮어쓸 수 있다(키는 커밋 금지).
GOOGLE_TRANSLATE_API_KEY: str = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip()


def set_google_translate_api_key(key: Optional[str], local_translation: bool = False) -> None:
    """Google 번역 키를 반영하고 실제로 어떤 번역 경로를 타는지 한 줄 남긴다.

    `local_translation` 이 True 면 모든 번역이 로컬 모델로 가므로(google_translate_async
    가 LOCAL_TRANSLATOR 를 먼저 본다) 키가 없다고 경고하지 않는다. 경고를 그대로 두면
    실제로는 쓰지 않는 gtx 를 쓰는 것처럼 읽혀 원인 추적을 헷갈리게 한다.
    """
    global GOOGLE_TRANSLATE_API_KEY
    if key:
        GOOGLE_TRANSLATE_API_KEY = key.strip()
    if local_translation:
        logger.info(
            "[translate] 로컬 번역 모델 사용 (--local-translation) — Google 경로는 "
            "로컬 번역기 초기화가 실패했을 때만 폴백으로 쓴다"
        )
    elif GOOGLE_TRANSLATE_API_KEY:
        logger.info("[translate] Cloud Translation v2 사용 (API 키 감지)")
    else:
        logger.warning(
            "[translate] GOOGLE_TRANSLATE_API_KEY 가 없어 무료 gtx 엔드포인트를 씁니다 — "
            "호출이 몰리면 429 로 막혀 번역이 빈 채로 나갑니다"
        )


async def _translate_v2(
    session: aiohttp.ClientSession, text: str, target_lang: str
) -> tuple[str, str]:
    """공식 Cloud Translation Basic(v2).

    `source` 를 넘기지 않고 자동 감지에 맡긴다 — gtx 의 `sl=auto` 와 같은 조건이라야
    `detectedSourceLanguage` 를 받아 기존 언어 판정 로직이 그대로 돈다.
    """
    body = {"q": text, "target": target_lang, "format": "text"}
    async with session.post(V2_URL, params={"key": GOOGLE_TRANSLATE_API_KEY}, json=body) as resp:
        payload = await resp.json(content_type=None)
        if resp.status != 200:
            err = (payload or {}).get("error", {})
            raise RuntimeError(f"HTTP {resp.status} {str(err)[:200]}")
    tr = payload["data"]["translations"][0]
    # format=text 면 이스케이프하지 않는 게 문서상 동작이지만 실제로는 `&#39;` 가 섞여
    # 나오는 사례가 있다. 자막에 그대로 보이면 안 되므로 푼다.
    import html as _html
    return _html.unescape(tr.get("translatedText") or ""), tr.get("detectedSourceLanguage") or ""


async def _translate_gtx(
    session: aiohttp.ClientSession, text: str, target_lang: str
) -> tuple[str, str]:
    params = {"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text}
    async with session.get(
        GTX_URL, params=params, headers={"User-Agent": "Mozilla/5.0"}
    ) as resp:
        if resp.status != 200:
            await resp.read()          # 본문을 비워야 keep-alive 커넥션이 재사용된다
            raise RuntimeError(f"HTTP {resp.status}")
        data = await resp.json(content_type=None)
    translated = "".join(item[0] for item in data[0] if item and item[0])
    detected_lang = data[2] if len(data) > 2 else ""
    return translated, detected_lang


# 로컬 번역기(NLLB). --local-translation 으로 켜며, 켜져 있으면 Google 경로 대신
# 이 인스턴스를 쓴다. 모든 번역 호출이 google_translate_async 를 지나므로
# 여기 한 곳만 갈아끼우면 컨텍스트 번역·평가 서버까지 함께 적용된다.
LOCAL_TRANSLATOR = None


def set_local_translator(translator) -> None:
    global LOCAL_TRANSLATOR
    LOCAL_TRANSLATOR = translator


async def google_translate_async(
    session: aiohttp.ClientSession, text: str, target_lang: str,
    source_lang: Optional[str] = None,
) -> tuple[str, str]:
    """Async Google Translate call.
    Returns: (translated_text, detected_source_lang_code)

    키가 있으면 v2, 없으면 gtx. 일시적 실패(429/타임아웃)를 한 번 흡수하되, 실시간
    자막이라 오래 붙들 수 없으므로 재시도는 1회 · 0.4초로 끊는다.

    --local-translation 으로 로컬 번역기가 켜져 있으면 외부 호출 없이 그걸 쓴다.
    source_lang 은 ASR 이 감지한 소스 언어 코드로, 로컬 번역기에만 쓰인다
    (Google 은 소스를 스스로 감지한다).
    """
    if not text.strip() or not target_lang:
        return "", ""
    if LOCAL_TRANSLATOR is not None:
        return await LOCAL_TRANSLATOR.translate(text, target_lang, source_lang)
    call = _translate_v2 if GOOGLE_TRANSLATE_API_KEY else _translate_gtx
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            return await call(session, text, target_lang)
        except Exception as e:
            last_err = e
            if attempt == 0:
                await asyncio.sleep(0.4)
    logger.warning(f"Translation failed: {last_err}")
    return "", ""


async def google_translate_with_context_async(
    session: aiohttp.ClientSession,
    text: str,
    target_lang: str,
    context_originals: list[str],
) -> tuple[str, str]:
    """컨텍스트 문장을 함께 전송해 Google Translate의 문맥 인식을 활용.

    이전 N개 원문을 현재 문장 앞에 붙여 한 번에 번역 요청하고,
    줄바꿈으로 분리한 뒤 마지막 줄만 현재 문장의 번역으로 반환.
    """
    if not context_originals:
        return await google_translate_async(session, text, target_lang)

    parts = context_originals + [text]
    combined = "\n".join(parts)
    translated_combined, detected_lang = await google_translate_async(session, combined, target_lang)

    lines = [l.strip() for l in translated_combined.split("\n") if l.strip()]
    translation = lines[-1] if lines else translated_combined
    return translation, detected_lang


class _ClientAdapter(logging.LoggerAdapter):
    """클라이언트 ID 태그를 자동으로 앞에 붙이는 LoggerAdapter"""

    def process(self, msg, kwargs):
        return f"[C{self.extra['cid']}] {msg}", kwargs


class Qwen3ASRStreamingHandler:
    """WebSocket 연결 당 하나의 스트리밍 핸들러"""

    def __init__(
        self,
        websocket,
        asr_model: Qwen3ASRModel,
        config: StreamingConfig,
        pairing_hub: PairingHub,
        get_streaming_id=None,
        lora_request_en=None,
        lora_request_ko=None,
        vad_model_bytes: Optional[bytes] = None,
        corrector=None,
        gpt_translator=None,
    ):
        self.websocket = websocket
        self._get_streaming_id = get_streaming_id
        self.log = _ClientAdapter(logger, {"cid": "-"})
        self.asr = asr_model
        self.config = config
        self.pairing_hub = pairing_hub
        self.lora_request_en = lora_request_en
        self.lora_request_ko = lora_request_ko
        self.state = None
        self.running = False

        # 클라이언트 옵션
        self.client_lang = "auto"
        self.client_target_lang = ""
        # 감지 언어별 번역 목표. 비어 있으면 기존 "내 언어 <-> 상대 언어" 쌍으로 돈다.
        self.client_lang_map: dict[str, str] = {}

        # 밖에서 지정한 디코딩 언어(정규 이름, 예: "Spanish"). lang_hint 로 들어온다.
        # 파인튜닝 모델은 자기 언어 밖에서 언어 태그부터 틀린다 — 베이스라인이
        # 스페인어를 ko 로 보고 한글로 받아쓰는 식이다. 오디오를 보고 판정한 쪽이
        # 있으면 그 값으로 못박는 편이 낫다.
        self.forced_language: Optional[str] = None
        # lang_hint 로 좁힌 허용 언어(정규 이름 하나). force 를 안 쓰는 기본 경로다 —
        # 로짓 바이어스는 언어 이름 토큰만 건드리므로 출력 형식이 안 바뀐다.
        self.hinted_language: Optional[str] = None

        # 타임스탬프 추적
        self.segment_start_time = 0.0
        self.current_time = 0.0

        self.asr_lock = asyncio.Lock()
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_logger: Optional[SessionLogger] = None
        self.recorder: Optional[AudioRecorder] = None
        self.corrector = corrector
        self.gpt_translator = gpt_translator
        # 번역 컨텍스트: 최근 N 세그먼트의 (corrected_original, translation) 보관
        _ctx = (gpt_translator.max_context if gpt_translator
                else config.context_window if config.google_context
                else 5)
        self._segment_history: deque[tuple[str, str, str]] = deque(maxlen=_ctx)
        # 첫 번째 발화는 Google Translate, 두 번째부터 GPT로 전환
        self._committed_utterance_count: int = 0

        # generate() 루프 내 병렬 GPT 처리
        self._in_generate_loop: bool = False
        self._pending_gpt_tasks: list = []
        # 진행 중인(fire-and-forget) GPT flush 태스크들 — VAD/finish emit 전 await용.
        # **집합이어야 한다.** 핸들 하나에 대입하면 이전 flush가 아직 돌고 있을 때 그 핸들이
        # 사라지고, _drain_pending_gpt가 마지막 것만 기다리게 된다. 놓친 flush가 스트림
        # 종료 뒤에 끝나면 그 final은 다음 발화로 밀려 나가거나 유실된다.
        # (평가 실측: 40발화 중 28회 중첩 발생, 29발화가 final을 못 받음)
        self._gpt_flush_tasks: set[asyncio.Task] = set()
        self._last_generate_end_time: float = 0.0  # generate() 완료 직후 perf_counter

        # Commit 방식 설정
        self.enable_dot_commit: bool = config.enable_dot_commit
        self.always_commit: bool = config.always_commit
        self.dot_commit_confirm: bool = config.dot_commit_confirm
        self.dot_commit_stall_chunks: int = config.dot_commit_stall_chunks
        self.rep_dedup: bool = config.rep_dedup

        # VAD / stream alignment
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        self.active_slot = "A"
        self.standby_slot = "B"
        self.stream_slots: dict[str, dict] = {}
        self.vad_last_speech_start_sample: int = 0  # 마지막 VAD speech_start 글로벌 샘플 위치

        # partial(토큰 단위 미확정 가설) 스트리밍 상태
        self._last_partial_text: Optional[str] = None
        self._last_partial_time: float = 0.0
        self._partial_seq: int = 0

        # ── silero-vad 초기화 (VADIterator 사용) ──
        # 서버에서 미리 로드한 vad_model_bytes로 클라이언트마다 독립 인스턴스 생성.
        self.vad_enabled = False
        self.vad_iterator = None
        if _SILERO_VAD_AVAILABLE and vad_model_bytes is not None:
            try:
                vad_model = torch.jit.load(io.BytesIO(vad_model_bytes))
                self.vad_iterator = VADIterator(
                    model=vad_model,
                    threshold=VAD_THRESHOLD,
                    sampling_rate=SAMPLING_RATE,
                    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                    speech_pad_ms=VAD_SPEECH_PAD_MS,
                )
                self.vad_enabled = True
                self.log.info("Silero VAD (VADIterator) loaded successfully")
            except Exception as e:
                self.log.warning(f"Silero VAD disabled: {e}")
        elif not _SILERO_VAD_AVAILABLE:
            self.log.warning(
                "silero-vad 패키지가 설치되지 않았습니다. "
                "VAD 없이 동작합니다. 설치: pip install silero-vad"
            )

    async def send_message(self, msg_type: str, **kwargs):
        """JSON 메시지 전송"""
        message = {"type": msg_type, **kwargs}
        if msg_type == "final":
            # 방금 확정된 구간은 더 이상 미확정이 아니다. partial 캐시를 무효화해 두면
            # 다음 _emit_partial이 같은 문자열이더라도 반드시 다시 나간다.
            self._last_partial_text = None
        try:
            await self.websocket.send(json.dumps(message, ensure_ascii=False))
            self.log.debug(f"Sent: {msg_type}")
        except Exception as e:
            self.log.error(f"Failed to send message: {e}")

    def init_streaming_state(self):
        """스트리밍 상태 초기화"""
        self.stream_slots = {
            "A": self._new_stream_slot(),
            "B": self._new_stream_slot(),
        }
        self.active_slot = "A"
        self.standby_slot = "B"
        self.state = self.stream_slots[self.active_slot]["state"]
        self.segment_start_time = 0.0
        self.current_time = 0.0
        self._last_partial_text = None
        self._last_partial_time = 0.0
        self._partial_seq = 0
        self.log.info("Streaming state initialized")

        # reset VAD
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        if self.vad_iterator is not None:
            self.vad_iterator.reset_states()
        self._segment_history.clear()
        self._committed_utterance_count = 0
        self._in_generate_loop = False
        self._pending_gpt_tasks = []

    def _new_stream_slot(self, seed_text: str = "", context: str = "") -> dict:
        allowed_languages = None
        if self.config.restrict_languages:
            langs = []
            if self.client_lang_map:
                # langMap 의 키가 곧 "말할 언어"다. 목표 언어는 출력일 뿐이므로 열지 않는다 —
                # 넣으면 쓰지도 않을 언어까지 허용해 ASR 오감지를 늘린다.
                # 슬롯을 만들 때 계산하므로, 시연 중 매핑을 바꾸면 다음 커밋부터 반영된다.
                for code in self.client_lang_map:
                    name = lang_code_to_name(code)
                    if name and name not in langs:
                        langs.append(name)
            elif self.client_lang and self.client_lang != "auto":
                src_name = lang_code_to_name(self.client_lang)
                if src_name:
                    langs.append(src_name)
                if self.client_target_lang:
                    tgt_name = lang_code_to_name(self.client_target_lang)
                    if tgt_name and tgt_name not in langs:
                        langs.append(tgt_name)
            if langs:
                allowed_languages = langs
        # lang_hint 가 왔으면 그 하나로 더 조인다. langMap 이 넓어도(제3언어를
        # 둘 이상 고른 경우) 판정된 언어만 남는다. restrict_languages 를 껐어도
        # 밖에서 준 판정은 존중한다.
        if self.hinted_language:
            allowed_languages = [self.hinted_language]

        # lang_hint 가 있으면 프롬프트에 "language X<asr_text>" 로 박아 디코딩
        # 언어를 못박는다. 이때 allowed_languages 의 로짓 바이어스는 적용되지
        # 않는다(둘 중 하나만 걸린다).
        state = self.asr.init_streaming_state(
            unfixed_chunk_num=self.config.unfixed_chunk_num,
            unfixed_token_num=self.config.unfixed_token_num,
            chunk_size_sec=self.config.chunk_size_sec,
            allowed_languages=allowed_languages,
            language=self.forced_language,
            context=context,
        )
        # [Fix 2] SEG 커밋 후 remaining 텍스트를 새 슬롯에 이식:
        # audio_accum은 빈 상태로 시작하지만, 모델이 이미 디코딩한 remaining은
        # _raw_decoded / text seed로 넘겨 prefix로 즉시 활용한다.
        if seed_text:
            state._raw_decoded = seed_text
            state.text = seed_text
            state.chunk_id = state.unfixed_chunk_num  # prefix 즉시 활성화

        return {
            "state": state,
            "flush_lock": asyncio.Lock(),
            "last_text": seed_text,  # 이미 처리된 것으로 마킹해 중복 커밋 방지
            "last_text_lang": "",
            "committed_len": 0,
            "committed_prefix": "",
            "committed_display": "",
            "committed_seg_count": 0,
            "audio_anchor_sec": self.current_time,
            "committed_asr_set": set(),  # 세그먼트 내 커밋된 문장 전체 (공백 정규화 후)
            "committed_fuzzy_keys": [],  # 위와 같은 문장의 유사도 비교용 키 (_fuzzy_key)
        }

    def _reset_stream_slot(self, slot_key: str, seed_text: str = "", context: str = ""):
        self.stream_slots[slot_key] = self._new_stream_slot(seed_text=seed_text, context=context)

    async def _apply_lang_hint(self, code: Optional[str],
                               from_sec: Optional[float] = None,
                               force: bool = False) -> None:
        """밖에서 판정한 언어로 디코딩 언어를 못박고, 활성 슬롯을 잘라 낸다.

        **못박기만 해서는 모자란다.** force_language 를 바꿔도 슬롯의 누적 텍스트가
        프롬프트 프리픽스에 그대로 남아, 모델은 "앞 언어로 쓰인 문장을 이어 쓰되
        새 언어로" 라는 모순된 지시를 받는다. 그래서 슬롯을 새로 만들어 프리픽스를
        비운다.

        `from_sec` 는 새 언어 발화가 시작된 오디오 시각이다. 그 지점부터의 오디오를
        새 슬롯에 옮겨 담는다. 안 옮기면 판정이 서기까지의 앞부분(보통 0.5~1초)이
        통째로 날아가고, 뒤따르는 VAD 도 침묵 구간을 잃어 발화 끝을 못 잡는다 —
        SEG 커밋 경로가 carry_audio 로 같은 문제를 피하고 있다.
        """
        name = lang_code_to_name(code) if code else None
        cur = self.forced_language if force else self.hinted_language
        if not name or name == cur:
            return
        prev = cur
        # **기본은 바이어스다.** force_language 는 프롬프트에 언어를 박아 넣어
        # 모델이 언어 이름을 아예 생성하지 않게 하는데, 그러면 출력 형식이 바뀌어
        # (language 접두 없이 본문만) 커밋·SEG 판정이 달라진다. 실측에서 짧은
        # 맞장구가 뭉개졌다. 바이어스는 언어 이름 토큰만 -100 으로 막으므로 형식이
        # 그대로다. 어느 쪽도 본문 글자까지 강제하지는 못한다 — 태그만 고정한다.
        if force:
            self.forced_language = name
            self.hinted_language = None
        else:
            self.hinted_language = name
            self.forced_language = None

        slot_key = self.active_slot
        old_state = self._slot(slot_key)["state"]
        anchor = self._slot(slot_key)["audio_anchor_sec"]
        accum = old_state.audio_accum
        # **buffer 도 같이 옮겨야 한다.** audio_accum 은 청크로 소비된 오디오만
        # 담고, 아직 한 청크를 못 채운 꼬리는 buffer 에 있다. accum 만 보고
        # 자르면 그 꼬리가 통째로 날아간다 — 판정 직후라 accum 이 아예 비어
        # 있는 일도 흔해서, 실제로 발화 앞부분('Esto parece')이 사라졌다.
        tail = old_state.buffer
        carry = None
        if accum is not None and accum.shape[0] > 0:
            offset = 0
            if from_sec is not None:
                offset = int(max(0.0, float(from_sec) - anchor) * SAMPLING_RATE)
            if offset < accum.shape[0]:
                carry = accum[offset:].copy()

        self._reset_stream_slot(slot_key)
        new_slot = self._slot(slot_key)
        if carry is not None and carry.shape[0] > 0:
            new_slot["state"].audio_accum = carry
            new_slot["audio_anchor_sec"] = (
                float(from_sec) if from_sec is not None else anchor)
        if tail is not None and tail.shape[0] > 0:
            new_slot["state"].buffer = tail.copy()
        if slot_key == self.active_slot:
            self.state = new_slot["state"]
        self.log.info(
            f"[LANG-HINT] slot={slot_key} {prev or '-'} -> {name} "
            f"mode={'force' if force else 'bias'} "
            f"from={from_sec} "
            f"carry={0 if carry is None else round(carry.shape[0] / SAMPLING_RATE, 2)}s "
            f"tail={0 if tail is None else round(tail.shape[0] / SAMPLING_RATE, 2)}s"
        )

    def _slot(self, slot_key: Optional[str] = None) -> dict:
        key = slot_key or self.active_slot
        return self.stream_slots[key]

    @staticmethod
    def _committed_cursor(text: str, committed_display: str,
                          committed_seg_count: int = 0) -> int:
        """committed 영역 끝의 커서 위치(int)를 반환.

        정규화 수준을 단계적으로 높여가며 prefix 매칭을 시도한 뒤,
        모두 실패하면 SEG 카운트 기준으로 fallback한다.
        모든 fallback 실패 시 -1(sentinel) 반환.
        """
        seg_tag = "<SEG>"
        seg_len = len(seg_tag)
        _punct = '.,!?;:。？！'
        _quote_colon = re.compile("[\"\u201C\u201D'\u2018\u2019\uFF1A:]+")

        # SEG 제거 + 공백 정규화 — prefix 매칭 전체에서 공통 사용
        text_no_seg = re.sub(r'\s+', ' ', text.replace(seg_tag, "")).strip()

        # ── 공통 커서 워커 ───────────────────────────────────────────────
        def _walk(target_len: int, skip_re=None, advance_punct: bool = False) -> int:
            """raw text에서 target_len개의 표시 문자를 소비하는 커서 위치 반환."""
            pos, disp_pos = 0, 0
            _prev_space = False
            while pos < len(text) and disp_pos < target_len:
                if text[pos:pos + seg_len] == seg_tag:
                    pos += seg_len
                    _prev_space = True
                elif text[pos] == ' ' and _prev_space:
                    pos += 1  # SEG 제거 후 남는 연속 공백 스킵
                elif skip_re and skip_re.match(text[pos]):
                    pos += 1  # 정규화로 제거된 문자(따옴표 등) 건너뜀
                else:
                    _prev_space = (text[pos] == ' ')
                    disp_pos += 1
                    pos += 1
            if advance_punct and pos < len(text) and text[pos] in _punct:
                pos += 1
            return pos

        # ── prefix 매칭: 정규화 수준을 단계적으로 높여가며 시도 ────────────
        if committed_display:
            _norm_p = lambda s: re.sub(r'[.,!?;:。？！、，]', '.', s)
            candidates = [
                # (committed_norm,                           text_norm,                         advance_punct, skip_re)
                (committed_display,                          text_no_seg,                        False, None),
                (_norm_p(committed_display),                 _norm_p(text_no_seg),                False, None),
                (committed_display.rstrip(_punct),           text_no_seg,                        True,  None),
                (_quote_colon.sub('', committed_display),    _quote_colon.sub('', text_no_seg),  False, _quote_colon),
            ]
            # 대소문자만 다른 재디코딩(커밋 후 'Oh, Papa!' → 'Oh, papa!')에도 prefix가
            # 붙도록 소문자 후보를 마지막에 덧붙인다. 이게 없으면 커서가 -1로 떨어지고
            # committed_seg_count==0인 dot 커밋에서는 SEG fallback도 막혀 있어서
            # 텍스트 전체가 미커밋으로 취급된다 — 같은 문장을 매 콜백마다 재검출하고
            # finish flush가 이미 커밋한 구간까지 다시 내보낸다(실측: 1688-142285-0016).
            # _walk가 원본 텍스트 기준 길이를 쓰므로 길이가 보존될 때만 사용한다.
            _lower_extra = []
            for committed_norm, text_norm, advance_punct, skip_re in candidates:
                if (len(committed_norm.lower()) == len(committed_norm)
                        and len(text_norm.lower()) == len(text_norm)):
                    _lower_extra.append(
                        (committed_norm.lower(), text_norm.lower(), advance_punct, skip_re)
                    )
            candidates.extend(_lower_extra)

            for committed_norm, text_norm, advance_punct, skip_re in candidates:
                if not committed_norm:
                    continue
                if not text_norm.startswith(committed_norm):
                    continue
                end = len(committed_norm)
                if end < len(text_norm) and text_norm[end].isalpha():
                    continue  # 부분 단어 매칭 방지
                return _walk(len(committed_norm), skip_re=skip_re, advance_punct=advance_punct)

        # ── SEG 카운트 fallback ──────────────────────────────────────────
        # committed_seg_count=0이고 committed_display가 있으면 skip:
        # DOT commit은 SEG count를 올리지 않으므로 pos=0을 리턴하면
        # uncommitted가 전체 텍스트가 됨
        if committed_seg_count > 0 or not committed_display:
            pos, found = 0, 0
            all_segs_found = True
            while found < committed_seg_count:
                idx = text.find(seg_tag, pos)
                if idx == -1:
                    all_segs_found = False
                    break
                pos = idx + seg_len
                found += 1
            if all_segs_found and pos < len(text):
                # 스트리밍 재디코딩으로 committed_display가 raw text에서 통째로
                # 사라진 경우(prefix rollback으로 이전 SEG가 밀려나는 등), 지금 센
                # SEG는 옛 커밋 경계가 아니라 새 문장의 SEG일 수 있다. skip한 구간이
                # committed_display와 실제로 닮아 있을 때만(=revision) 신뢰한다.
                candidate = re.sub(r'\s+', ' ', text[:pos].replace(seg_tag, "")).strip()
                similarity = difflib.SequenceMatcher(None, candidate, committed_display).ratio()
                if similarity < 0.5:
                    return -1
                return pos

        return -1

    @staticmethod
    def _uncommitted_from(current_text: str, committed_display: str,
                          committed_seg_count: int = 0) -> str:
        """committed 경계 이후의 current_text를 반환.

        _committed_cursor의 3-way fallback으로 커서를 구한 뒤
        해당 위치 이후 텍스트를 반환한다. 모든 fallback 실패 시 "" 반환.
        """
        pos = Qwen3ASRStreamingHandler._committed_cursor(
            current_text, committed_display, committed_seg_count
        )
        if pos == -1:
            if committed_display and current_text:
                # revision 감지: committed_seg_count개 SEG 건너뛰고 그 다음부터 반환
                _seg = "<SEG>"
                p, found = 0, 0
                while found < committed_seg_count:
                    idx = current_text.find(_seg, p)
                    if idx == -1:
                        return ""
                    p = idx + len(_seg)
                    found += 1
                return current_text[p:]
            return ""
        return current_text[pos:]

    def _get_lora_request(self, state):
        """언어에 따라 적절한 LoRA 어댑터 반환. 미지원 언어는 None(기본 모델)."""
        lang = state.force_language or state.language  # canonical name e.g. "Korean"
        if not lang and self.client_lang and self.client_lang != "auto":
            lang = lang_code_to_name(self.client_lang) or ""
        if lang == "English":
            return self.lora_request_en
        if lang == "Korean":
            return self.lora_request_ko
        return None

    @staticmethod
    def _strip_asr_text(text: str) -> str:
        """state.text에서 <asr_text> 태그를 제거한다.

        force_language 모드에서 parse_asr_output은 'language X<asr_text>...'를 그대로
        state.text에 넣는다. 첫 번째 <asr_text> 앞이 language 메타데이터이면 버리고,
        나머지 <asr_text>는 공백으로 치환한다.
        """
        if "<asr_text>" not in text:
            result = text
        else:
            first, rest = text.split("<asr_text>", 1)
            if re.match(r'^\s*language\s+\w', first, re.IGNORECASE):
                text = rest
            result = re.sub(r'\s*<asr_text>\s*', ' ', text).strip()
        return Qwen3ASRStreamingHandler._cut_repeats(result)

    @staticmethod
    def _cut_repeats(text: str, max_repeat: int = 4) -> str:
        """동일 토큰이 max_repeat회 이상 연속되면 그 앞에서 잘라냄 (할루시네이션 반복 제거)."""
        tokens = text.split()
        i = 0
        while i < len(tokens):
            j = i + 1
            while j < len(tokens) and tokens[j] == tokens[i]:
                j += 1
            if j - i >= max_repeat:
                return ' '.join(tokens[:i]).strip()
            i = j
        return text

    def _slot_uncommitted_display(self, slot_key: Optional[str] = None, text_snapshot: Optional[str] = None) -> str:
        slot = self._slot(slot_key)
        if text_snapshot is not None:
            current_text = text_snapshot
        else:
            state = slot["state"]
            current_text = self._strip_asr_text((state.text or "").strip() if state else "")
        uncommitted_raw = self._uncommitted_from(
            current_text, slot["committed_display"], slot["committed_seg_count"]
        )
        if uncommitted_raw is None:
            return ""
        return re.sub(r'\s+', ' ', uncommitted_raw.replace("<SEG>", "")).strip()

    async def _emit_partial(self, slot_key: Optional[str] = None, force: bool = False) -> None:
        """미확정 구간(아직 커밋되지 않은 텍스트)을 partial 메시지로 전송.

        보내는 문자열은 항상 통째로 교체할 전체 텍스트다. 모델이 unfixed_token_num
        만큼 롤백 후 재디코딩하므로 꼬리 토큰은 확정이 아니고, append 방식으로는
        정합성을 못 맞춘다.

        빈 문자열은 "화면을 비우라"는 신호로 그대로 전송한다.
        """
        now = time.perf_counter()
        # 시간 게이트를 먼저 본다. _slot_uncommitted_display 는 difflib 비교까지
        # 도는 계산이라, 매 토큰 돌리면 디코딩 루프에 그대로 얹힌다.
        if not force and now - self._last_partial_time < PARTIAL_MIN_INTERVAL_SEC:
            return
        text = self._slot_uncommitted_display(slot_key)
        # 디코딩 중에는 빈 문자열을 보내지 않는다. 청크마다 <asr_text> 직후 잠깐
        # 텍스트가 비는 순간이 있어, 그대로 흘리면 청크 경계마다 화면이 한 번
        # 깜빡인다. 화면 비우기는 커밋/리셋 뒤 force 경로에서만 내보낸다.
        if not text and not force:
            return
        prev = self._last_partial_text
        if not force and prev and text != prev and prev.startswith(text):
            # 재디코딩 램프업. 청크마다 롤백 후 처음부터 다시 디코딩하므로 직전
            # 텍스트의 앞부분만 나오는 순간이 있다(예: 'I really like.' -> 'I').
            # 그대로 흘리면 화면이 줄었다 늘었다 한다. 진짜 수정(prefix 가 아닌
            # 다른 텍스트)은 이 조건에 안 걸리고, 이 억제로 화면이 뒤처지더라도
            # 청크 끝 force 재동기화가 한 청크 안에 바로잡는다.
            return
        if text == prev:
            # 안 보내더라도 시계는 전진시킨다. 안 그러면 텍스트가 멈춰 있는 동안
            # 매 토큰 위 계산을 다시 돌게 된다.
            self._last_partial_time = now
            return
        self._last_partial_text = text
        self._last_partial_time = now
        self._partial_seq += 1
        await self.send_message(
            "partial",
            text=text,
            language=self._slot(slot_key).get("last_text_lang", ""),
            seq=self._partial_seq,
        )

    def _build_forced_reset_seed(self, slot_key: str, remaining: str) -> str:
        """강제 리셋용 seed_text: 마지막 N committed 문장 + remaining"""
        slot = self._slot(slot_key)
        committed = slot.get("committed_display", "")
        sentences = re.split(r'(?<=[。？！?!])', committed)
        sentences = [s for s in sentences if s.strip()]
        last_n = "".join(sentences[-MAX_SEED_COMMITTED_SENTENCES:])
        return last_n + remaining

    def _init_forced_reset_slot(self, slot_key: str, seed_text: str, remaining: str) -> None:
        """강제 리셋 슬롯의 커서/rollback 초기화"""
        slot = self._slot(slot_key)
        committed_part = seed_text[: len(seed_text) - len(remaining)]

        slot["committed_display"] = committed_part   # cursor 위치 고정
        slot["committed_len"] = len(committed_part)  # 이중 커밋 방지
        slot["committed_seg_count"] = committed_part.count("<SEG>")
        slot["last_text"] = seed_text
        slot["state"].unfixed_token_num = 0          # rollback 비활성화 (seed_text는 확정 텍스트)

    async def _asr_streaming_transcribe(self, chunk: np.ndarray, slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        committed_display_before = slot.get("committed_display", "")
        prev_uncommitted = self._slot_uncommitted_display(slot_key)

        async def _on_seg(_):
            # lock 밖에서 호출되므로 _process_slot_updates가 asr_lock 자유롭게 획득 가능
            await self._process_slot_updates(slot_key)

        async def _on_dot(_):
            # <SEG>와 동일하게 마침표 디코딩되는 순간 바로 커밋 시도 — 발화의
            # 마지막 문장도 finish까지 안 기다리고 dot으로 즉시 커밋되게 함.
            await self._process_slot_updates(slot_key)

        async def _on_partial(_):
            # 화면에 뜨는 건 active 슬롯의 텍스트뿐이다. standby 슬롯 가설을
            # 흘리면 두 개가 번갈아 덮어써서 화면이 튄다.
            if slot_key is not None and slot_key != self.active_slot:
                return
            await self._emit_partial(slot_key)

        async with self.asr_lock:
            lora_request = self._get_lora_request(slot["state"])

        # 생성 중 lock 미보유 — state.text는 await 없는 단순 대입이므로 asyncio 안전
        _accum_len_before = slot["state"].audio_accum.shape[0]
        self._in_generate_loop = True
        try:
            await self.asr.streaming_transcribe(
                chunk, slot["state"], lora_request=lora_request, on_seg=_on_seg,
                on_dot=_on_dot if self.enable_dot_commit else None,
                on_partial=_on_partial,
            )
        finally:
            self._in_generate_loop = False
            self._last_generate_end_time = time.perf_counter()
        # 이번 청크에서 SEG 커밋이 발생했으면 committed_token_len 업데이트
        # → 다음 청크의 rollback이 커밋된 SEG 경계를 넘지 못하도록 고정
        if slot.get("committed_display", "") != committed_display_before:
            _cur_ids = self.asr.processor.tokenizer.encode(slot["state"]._raw_decoded)
            slot["state"].committed_token_len = max(
                slot["state"].committed_token_len,
                max(0, len(_cur_ids) - slot["state"].unfixed_token_num),
            )
        # audio_accum이 늘었을 때만 실제 추론이 실행된 것
        if slot["state"].audio_accum.shape[0] > _accum_len_before:
            _decoded_text = self._strip_asr_text((slot["state"].text or "").strip())
            self.log.info(f"[TRANSCRIBE-DECODING] slot={slot_key} text={_decoded_text!r}")

        # 할루시네이션 감지: 반복 직전까지 부분 커밋 후 슬롯 완전 초기화
        if slot["state"].hallucination_detected:
            slot["state"].hallucination_detected = False
            # 환각 컷으로 현재 text가 비었으면 마지막 비어있지 않던 디코드 텍스트로 복원해 커밋한다.
            # (짧은 발화가 침묵 재디코드+반복 환각으로 ''가 되어 통째로 버려지는 것 방지)
            if not (slot["state"].text or "").strip() and getattr(slot["state"], "_last_nonempty_text", ""):
                slot["state"].text = slot["state"]._last_nonempty_text
            _cut_text = self._strip_asr_text((slot["state"].text or "").strip())
            self.log.info(f"[HALLUCINATION-PARTIAL-COMMIT] slot={slot_key} text={_cut_text!r}")
            # generate 루프 안에서 _on_seg로 쌓인 GPT 태스크를 먼저 flush
            if self._pending_gpt_tasks:
                self._spawn_gpt_flush()
            await self.flush_uncommitted(force=True, reason="vad", slot_key=slot_key)
            self._reset_stream_slot(slot_key)
            if slot_key == self.active_slot:
                self.state = self.stream_slots[self.active_slot]["state"]
            async with self.asr_lock:
                self.asr_processed_cursor = self.sample_cursor
            await self._emit_partial(slot_key, force=True)
            return

        _committed_text_snapshot = await self._process_slot_updates(slot_key, chunk_end=True)
        _accum_size_pre_gpt = self._slot(slot_key)["state"].audio_accum.shape[0]
        if self._pending_gpt_tasks:
            self._spawn_gpt_flush()

        _s = self._slot(slot_key)
        # SEG/dot commit 발생 시, 추론 결과 text가 <SEG>로 끝나면 uncommitted 없음 → 슬롯 리셋.
        any_commit = _s.get("committed_display", "") != committed_display_before
        if any_commit:
            _latest_decoded = self._strip_asr_text((_s["state"].text or "").strip())
            ends_with_seg = _latest_decoded.endswith("<SEG>")
            remaining = "" if ends_with_seg else self._slot_uncommitted_display(slot_key, text_snapshot=_committed_text_snapshot)
            audio_sec = _s["state"].audio_accum.shape[0] / SAMPLING_RATE
            force_reset = audio_sec > MAX_AUDIO_ACCUM_SEC

            if ends_with_seg or not remaining.strip() or force_reset:
                if force_reset:
                    seed_text = self._build_forced_reset_seed(slot_key, remaining)
                    self._reset_stream_slot(slot_key, seed_text=seed_text)
                    self._init_forced_reset_slot(slot_key, seed_text, remaining)
                    if slot_key == self.active_slot:
                        self.state = self.stream_slots[self.active_slot]["state"]
                    self.log.info(f"[FORCE-SLOT-SWITCH] slot={slot_key} audio_sec={audio_sec:.1f}s")
                else:
                    last_committed = _s.get("committed_display", "")
                    # GPT 딜레이 동안 쌓인 오디오(주로 trailing silence)를 새 슬롯에 carry-over.
                    # 리셋 후 즉시 삭제하면 VAD 발동에 필요한 침묵 구간이 사라져 발화 누락이 발생함.
                    _accum_now = _s["state"].audio_accum.shape[0]
                    _carry_samples = _accum_now - _accum_size_pre_gpt
                    carry_audio = (
                        _s["state"].audio_accum[-_carry_samples:].copy()
                        if _carry_samples > 0 else None
                    )
                    self._reset_stream_slot(slot_key)
                    if carry_audio is not None:
                        self._slot(slot_key)["state"].audio_accum = carry_audio
                        carry_sec = round(_carry_samples / SAMPLING_RATE, 3)
                        self.log.info(f"[SEG-CARRY-AUDIO] slot={slot_key} carry={carry_sec}s")
                    if last_committed:
                        self._slot(slot_key)["seg_reset_last_committed"] = last_committed
                    if slot_key == self.active_slot:
                        self.state = self.stream_slots[self.active_slot]["state"]
                    self.log.info(f"[SEG-SLOT-RESET] slot={slot_key} audio_sec={audio_sec:.1f}s")
            else:
                # cross-decode trailing-period reset:
                # prev_uncommitted 온점 위치가 이번 commit과 동일한 경우에만 슬롯 리셋.
                # 온점이 이동했으면(임시 추측 온점) DOT-SLOT-SWITCH 불필요.
                prev_strip = prev_uncommitted.strip()
                period_in_place = False
                if prev_strip and re.search(r'[.?!。？！]$', prev_strip):
                    committed_display_after = _s.get("committed_display", "")
                    newly_committed_text = committed_display_after[len(committed_display_before):]
                    prev_core = re.sub(r'\s+', '', prev_strip.rstrip('.?!。？！'))
                    newly_committed_norm = re.sub(r'\s+', '', newly_committed_text)
                    period_in_place = bool(
                        prev_core
                        and re.match(re.escape(prev_core) + r'[.?!。？！]', newly_committed_norm)
                    )
                if period_in_place:
                    # partial-1 오디오(이미 commit된 구간)만 버리고 partial-2 오디오는 새 슬롯에 유지
                    # 오디오가 carry-over되므로 seed 텍스트 불필요 — 모델이 재디코딩함
                    chunk_samples = int(round(self.config.chunk_size_sec * SAMPLING_RATE))
                    old_accum = _s["state"].audio_accum
                    carry_audio = old_accum[-chunk_samples:] if old_accum.shape[0] >= chunk_samples else old_accum.copy()
                    carry_lang = _s.get("last_text_lang", "")
                    prev_committed = _s.get("committed_display", "")
                    # 리셋은 슬롯 dict를 통째로 갈아끼우므로 dot 게이트 상태도 같이 날아간다.
                    # 그러면 발화 마지막 문장이 리셋 직후 후보로 다시 등록되고, 확정에 필요한
                    # 다음 청크가 오기 전에 오디오가 떨어져 finish로 빠진다. carry_audio는
                    # 같은 단어로 재디코딩되므로 직전 청크의 경계 목록은 그대로 유효하다.
                    carry_boundaries = _s.get("prev_boundary_sentences", ())
                    self._reset_stream_slot(slot_key)
                    new_slot = self._slot(slot_key)
                    new_slot["state"].audio_accum = carry_audio
                    if carry_lang:
                        new_slot["last_text_lang"] = carry_lang
                    if prev_committed:
                        new_slot["dot_switch_prev_committed"] = prev_committed
                    if carry_boundaries:
                        new_slot["prev_boundary_sentences"] = carry_boundaries
                        # accum은 리셋으로 줄어드므로 스탬프는 재기준화(-1) — 이미 한 번
                        # 재디코딩을 견딘 경계라 다음 청크에서 바로 확정돼도 안전하다.
                        new_slot["prev_boundary_accum"] = -1
                    if slot_key == self.active_slot:
                        self.state = self.stream_slots[self.active_slot]["state"]
                    self.log.info(
                        f"[DOT-SLOT-SWITCH] slot={slot_key} audio_sec={audio_sec:.1f}s "
                        f"prev={prev_strip!r}"
                    )
                else:
                    self.log.info(
                        f"[COMMIT-PENDING] slot={slot_key} remaining={remaining!r}"
                    )

        # 커밋/슬롯 리셋이 끝난 실제 상태로 화면을 재동기화한다. 리셋이 일어났으면
        # 미확정 구간이 비어 빈 문자열이 나가고, 클라이언트의 유령 텍스트가 지워진다.
        await self._emit_partial(slot_key, force=True)

        async with self.asr_lock:
            self.asr_processed_cursor = self.sample_cursor

    def _spawn_gpt_flush(self) -> None:
        """GPT flush를 백그라운드로 발사하고 핸들을 집합에 등록한다.

        핸들을 변수 하나에 대입하면 안 된다 — 청크가 연달아 오면 이전 flush가 아직
        도는 중에 덮어써져, 그걸 기다릴 방법이 없어진다.
        """
        task = asyncio.create_task(self._flush_pending_gpt_tasks())
        self._gpt_flush_tasks.add(task)
        task.add_done_callback(self._gpt_flush_tasks.discard)

    async def _drain_pending_gpt(self) -> None:
        """진행 중인(fire-and-forget) GPT flush 태스크와 남은 pending을 모두 완료·emit한다.
        VAD/finish 커밋(뒤 오디오)이 emit되기 전에 호출해, 앞 오디오의 SEG 커밋이 먼저
        emit되도록 보장한다(비동기 번역 완료 순서로 인한 세그먼트 역전 방지). fire-and-forget
        태스크가 _pending_gpt_tasks를 이미 가져간 경우 리스트는 비어있으므로, 리스트 체크가
        아니라 태스크 핸들을 await해야 결정적으로 동작한다.

        **진행 중인 flush를 전부 기다린다.** 하나만 기다리면 겹쳐 발사된 앞의 flush가
        스트림 종료 뒤에 끝나면서 그 문장이 다음 발화로 밀려 나가거나 유실된다."""
        while True:
            running = [t for t in self._gpt_flush_tasks if not t.done()]
            if not running:
                break
            await asyncio.gather(*running, return_exceptions=True)
        self._gpt_flush_tasks.clear()
        # 핸들이 없던(직접 발사 안 된) pending이 남아있으면 마저 flush
        if self._pending_gpt_tasks:
            await self._flush_pending_gpt_tasks()

    async def _flush_pending_gpt_tasks(self) -> None:
        """generate() 중 백그라운드로 발사된 GPT 태스크를 순서대로 await하고 emit."""
        tasks = self._pending_gpt_tasks
        self._pending_gpt_tasks = []
        for item in tasks:
            try:
                corrected, translation, lang, extra = await item["task"]
            except Exception as e:
                self.log.warning(f"[GPT-TASK-ERROR] {e} — Google Translate로 폴백")
                try:
                    translation, lang, extra = await self._translate(
                        item["original"], self.client_target_lang, item["audio_end_sec"]
                    )
                    corrected = item["original"]
                except Exception as e2:
                    self.log.error(f"[GPT-FALLBACK-ERROR] {e2} — 세그먼트 스킵")
                    continue
            await self._emit_final_payload(
                slot_key=item["slot_key"],
                original=item["original"],
                translation=translation,
                language=lang,
                reason=item["trigger_reason"],
                audio_end_sec=item["audio_end_sec"],
                extra=extra,
            )
            self.log.info(
                f"[TRANS-SEG-ASYNC] slot={item['slot_key']} lang={lang} "
                f"original='{item['original']}' translation='{translation}'"
            )

    async def _asr_finish_streaming(self, slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        async with self.asr_lock:
            state = slot["state"]
            lora_request = self._get_lora_request(state)
            await self.asr.finish_streaming_transcribe(state, lora_request=lora_request)

    async def flush_uncommitted(self, force=False, reason="flush", slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        slot_flush_lock = slot["flush_lock"]
        # finish/VAD flush는 남은 구간을 통째로 커밋하므로 보류 중인 dot 후보는 무효
        slot.pop("pending_dot_text", None)
        slot.pop("pending_dot_accum", None)

        # 1단계: 텍스트 스냅샷만 lock 안에서 읽기
        async with slot_flush_lock:
            async with self.asr_lock:
                state = slot["state"]
                current_text = self._strip_asr_text((state.text or "").strip() if state else "")
                current_lang = (state.language if state else None) or slot["last_text_lang"] or ""
            snapshot_committed_len = slot["committed_len"]
            snapshot_committed_display = slot["committed_display"]
            snapshot_committed_seg_count = slot["committed_seg_count"]
            uncommitted_raw = self._uncommitted_from(
                current_text, snapshot_committed_display, snapshot_committed_seg_count
            )
            uncommitted_display = re.sub(r'\s+', ' ', uncommitted_raw.replace("<SEG>", "")).strip() if uncommitted_raw is not None else ""
            # 미커밋 누적분은 스트리밍 커밋 경로의 rep-dedup/cross-dedup을 거치지 않고
            # 여기서 통째로 나간다. 반복 루프가 그대로 실리지 않도록 flush 직전에 접는다.
            uncommitted_display = self._collapse_repetition(uncommitted_display)
            # 이미 커밋된 문장의 재방출도 여기서 걸러야 한다. flush는 committed_asr_set을
            # 보지 않으므로, dot으로 확정·커밋된 문장을 모델이 철자만 바꿔 다시 내놓으면
            # finish flush가 그대로 또 커밋한다 (실측: 3538-142836-0017, 7105-2340-0005 등
            # dot→finish로 같은 문장이 두 번). _process_slot_updates의 cross-dedup과
            # 같은 판정을 문장 단위로 적용한다.
            uncommitted_display = self._apply_commit_guards(slot, uncommitted_display, slot_key)
            if not uncommitted_display:
                return
            if not force and len(uncommitted_display) < 2:
                logger.info(f"[COMMIT-SKIP] reason=timeout-skip text='{uncommitted_display}'")
                return
            # DOT-SLOT-SWITCH 후: 이전 커밋 tail이 uncommitted 앞에 붙어있으면 제거
            prev_committed_dot = slot.get("dot_switch_prev_committed", "")
            if prev_committed_dot:
                slot.pop("dot_switch_prev_committed", None)
                _last_sents = re.split(r'(?<=[.!?。！？])\s+', prev_committed_dot.strip())
                _last_sent = _last_sents[-1].strip() if _last_sents else ""
                if _last_sent:
                    _norm_last = re.sub(r'[.,!?;:。？！\s]+', '', _last_sent)
                    _norm_ud = re.sub(r'[.,!?;:。？！\s]+', '', uncommitted_display)
                    if _norm_ud.startswith(_norm_last):
                        _skip = 0
                        _norm_cnt = 0
                        for _ch in uncommitted_display:
                            if _norm_cnt >= len(_norm_last):
                                break
                            _skip += 1
                            if not re.match(r'[.,!?;:。？！\s]', _ch):
                                _norm_cnt += 1
                        while _skip < len(uncommitted_display) and uncommitted_display[_skip] in '.,!?;:。？！ ':
                            _skip += 1
                        uncommitted_display = uncommitted_display[_skip:].strip()
                        self.log.info(f"[COMMIT-SKIP] reason=dot-suffix-dedup slot={slot_key} stripped={uncommitted_display!r}")
            if not uncommitted_display:
                return

        # 2단계: lock 해제 후 I/O (교정 + 번역) 수행 → 다른 슬롯 flush 비차단
        audio_end_sec = self._get_flush_audio_end_sec()

        # 누적분에 문장이 여러 개면 문장 단위로 쪼개 커밋한다. 원래는 통째로 한 커밋에
        # 담아서 21초 발화가 한 덩어리로 나갔다. dot 확정 게이트(규칙 4)로 상당수는
        # 풀렸지만, 약어(Mr./St./S. J.) 때문에 DOT_COMMIT_BOUNDARY_RE가 경계를 못 잡는
        # 문장은 여전히 통째로 여기 온다. 커밋 사유는 finish로 남되 덩어리만 나눈다.
        # 번역도 문장 단위로 돌아 품질에 유리하다.
        # 모드2는 커밋 단위가 문장이 아니라 청크라 제외한다.
        # 분할은 반드시 DOT_COMMIT_BOUNDARY_RE로 한다. 순진하게 [.!?]로 자르면 약어에서
        # 깨진다(실측: 'returned Mr.' + 'Lilburn.'). 이 정규식은 Mr./Mrs./Dr./St./Jr./Sr./
        # vs./No.를 경계에서 제외하므로 dot 커밋 경로와 같은 기준이 유지된다.
        _parts = [uncommitted_display]
        if not self.always_commit:
            _split, _rem = [], uncommitted_display
            while True:
                _m = DOT_COMMIT_BOUNDARY_RE.search(_rem)
                if not _m:
                    break
                _s = _rem[:_m.end()].strip()
                if _s:
                    _split.append(_s)
                _rem = _rem[_m.end():]
            if _rem.strip():
                _split.append(_rem.strip())
            if len(_split) > 1:
                _parts = _split
                self.log.info(
                    f"[FLUSH-SPLIT] slot={slot_key} {len(_parts)}문장으로 분할 "
                    f"text={uncommitted_display[:80]!r}"
                )

        _emits = []
        for _part in _parts:
            _text, _tr, _lang, _extra = await self._correct_and_translate(
                _part, current_lang, audio_end_sec
            )
            if _text:
                _emits.append((_text, _tr, _lang, _extra))
        if not _emits:
            return
        # 커서/로그는 기존 코드가 단일 값을 쓰므로 대표값을 유지한다.
        uncommitted_display, translation, effective_detected, extra = _emits[-1]
        final_lang = effective_detected
        if reason.startswith("vad"):
            commit_reason = "vad"
        elif reason == "finish":
            commit_reason = "finish"
        elif reason == "timeout":
            commit_reason = "timeout"
        else:
            commit_reason = "seg"

        # 3단계: lock 재획득 후 커밋 — committed_len이 변경됐으면 skip (중복 방지)
        async with slot_flush_lock:
            if slot["committed_len"] != snapshot_committed_len:
                self.log.info(
                    f"[COMMIT-SKIP] reason=flush-skip "
                    f"committed_len {snapshot_committed_len}→{slot['committed_len']}"
                )
                return
            for _text, _tr, _lang, _extra in _emits:
                self.log.info(
                    f"[TRANS-VAD] slot={slot_key or self.active_slot} reason={commit_reason} lang={_lang} "
                    f"original='{_text}' translation='{_tr}'"
                )
                await self._emit_final_payload(
                    slot_key=slot_key or self.active_slot,
                    original=_text,
                    translation=_tr,
                    language=_lang,
                    reason=commit_reason,
                    audio_end_sec=audio_end_sec,
                    extra=_extra,
                )
            slot["committed_len"] = len(current_text)
            slot["committed_prefix"] = current_text
            slot["committed_display"] = re.sub(r'\s+', ' ', current_text.replace("<SEG>", "")).strip()
            slot["committed_seg_count"] = current_text.count("<SEG>")
            slot["audio_anchor_sec"] = audio_end_sec

    def _count_tokens(self, text: str) -> int:
        """모델 토크나이저 기준 토큰 수. dot_commit_confirm의 롤백 창 판정에 사용."""
        if not text or not text.strip():
            return 0
        try:
            return len(self.asr.processor.tokenizer.encode(text))
        except Exception:
            # 토크나이저 접근 실패 시 단어 수로 근사 (게이트가 과도하게 커밋하지 않는 방향)
            return len(text.split())

    @staticmethod
    def _boundary_key(sentence: str) -> str:
        """합의 확정(규칙 2) 비교용 정규화 키 — 대소문자·구두점·공백 무시.

        모델은 롤백 창 안에서 문구를 그대로 두고 구두점만 바꾸는 일이 잦다
        (실측: 'house, mother.' ↔ 'house mother.'가 청크마다 번갈아 나와
        문자열 완전 일치 비교로는 영원히 확정되지 않았다). 경계가 같은 자리에
        살아남았는지만 보면 되므로 표기 변형은 무시한다. 커밋 자체는 최신
        텍스트로 하고 비교에만 이 키를 쓴다.
        """
        return re.sub(r'[.,!?;:。？！、，\'"“”‘’\s]+', '', sentence.replace("<SEG>", "")).lower()

    @classmethod
    def _extract_boundary_keys(cls, text: str) -> tuple:
        """`text`에서 경계로 끝나는 문장들을 커밋 루프와 동일하게 자른 뒤 정규화 키로 반환.

        합의 확정(규칙 2)의 비교 대상. 후보 하나(`pending_dot_text`)만 들고 있으면
        한 청크에 경계가 둘 이상 나올 때 뒤쪽 경계는 앞 경계가 커밋될 때까지 평가조차
        되지 않는다 — 발화 마지막 문장이 딱 그 경우라 오디오가 떨어질 때까지 밀린다.
        청크의 경계 전체를 남겨 두면 "직전 청크에도 있던 경계"를 위치와 무관하게 확정할 수 있다.
        """
        out = []
        remaining = text
        while True:
            m = DOT_COMMIT_BOUNDARY_RE.search(remaining)
            if not m:
                break
            key = cls._boundary_key(remaining[:m.end()])
            if key:
                out.append(key)
            remaining = remaining[m.end():]
        return tuple(out)

    # 유사도 dedup 파라미터. 실측 튜닝값 (mode3 full 2939파일):
    #   minw=6/thr=0.85가 WER 5.81 → 4.25%로 최적. thr을 0.95로 올리면 철자 변형을
    #   놓쳐 4.50%, minw를 10으로 올리면 짧은 재방출을 놓쳐 4.39%.
    # 연속 반복 허용 상한. _collapse_repetition의 max_repeats와 같은 값·같은 근거
    # (실제 발화에도 같은 문장 2회 반복이 있다).
    _REP_DEDUP_MAX_REPEATS = 2
    _FUZZY_DEDUP_MIN_WORDS = 6
    _FUZZY_DEDUP_RATIO = 0.85

    @staticmethod
    def _fuzzy_key(text: str) -> str:
        """유사도 비교용 키 — 대소문자·구두점을 전부 버리고 단어 시퀀스만 남긴다."""
        return ' '.join(re.sub(r"[^a-z' ]", ' ', text.lower()).split())

    def _strip_committed_prefix(self, slot: dict, sentence_display: str) -> str:
        """이미 커밋된 문장을 접두사로 포함하면 그 부분만 잘라낸다.

        재디코딩이 앞 문장은 그대로 두고 뒤를 이어붙여 다시 내놓는 경우가 있다.
        실측 6432-63722-0035:
            커밋됨 : 'Before the big wind in Ireland, ... his Irish compatriot.'
            재방출 : 'Before the big wind in Ireland, ... his Irish compatriot,
                      slightly laughed the Colonel.'
        길이가 달라 _cross_dup_match(유사도)를 빠져나가고, 그렇다고 통째로 버리면
        뒤의 새 내용('slightly laughed the Colonel')까지 잃는다. 겹치는 앞부분만 잘라
        뒤만 커밋한다. 규칙 4로 유예가 풀리면서 이 케이스가 늘어 도입했다.
        """
        if self.always_commit or not sentence_display:
            return sentence_display
        words = sentence_display.split()
        norm = [re.sub(r"[^a-z']", '', w.lower()) for w in words]
        for prev in slot.get("committed_fuzzy_keys", ()):
            pn = prev.split()
            # 남는 게 없으면 접두사가 아니라 완전 중복 — 기존 dedup이 처리한다
            if not pn or len(norm) <= len(pn):
                continue
            head = ' '.join(norm[:len(pn)])
            if difflib.SequenceMatcher(None, head, prev).ratio() >= self._FUZZY_DEDUP_RATIO:
                return ' '.join(words[len(pn):]).lstrip(' ,.;:!?')
        return sentence_display

    def _cross_dup_match(self, slot: dict, sentence_display: str) -> Optional[str]:
        """이미 커밋된 문장의 재방출이면 매칭된 기존 문장을, 아니면 None을 반환.

        기존 cross-dedup은 정규화 후 완전 일치만 봤다. 그런데 dot commit이 문장을
        확정·커밋한 뒤 모델이 재디코딩하면서 **고유명사 철자만 바꿔** 같은 문장을 다시
        내놓는 일이 잦다 (실측: 'Alfred Fichencourt' → 'Fychencote' → 'Pichenot',
        'Beresheba Fristoe' → 'Bertha' → 'Brusheba'). 경계 키가 달라져 완전 일치로는
        못 걸러내고 새 문장으로 또 커밋된다 — mode3 삽입 1224단어 중 911단어(74%)가
        이 재방출이었고, WER로 1.56%p였다.

        짧은 문장은 완전 일치만 본다. 'He said yes.'와 'He said no.'처럼 실제로 다른
        짧은 문장이 유사도만으로는 구분되지 않기 때문.
        """
        key = self._fuzzy_key(sentence_display)
        if not key:
            return None
        if len(key.split()) < self._FUZZY_DEDUP_MIN_WORDS:
            return None
        for prev in slot.get("committed_fuzzy_keys", ()):
            if difflib.SequenceMatcher(None, key, prev).ratio() >= self._FUZZY_DEDUP_RATIO:
                return prev
        return None

    # ── 커밋 가드 파이프라인 ────────────────────────────────────────────────
    # 문장이 최종 출력으로 나가는 경로가 둘이다:
    #   A) _process_slot_updates — 청크마다 SEG/dot 트리거로 문장 하나씩
    #   B) flush_uncommitted     — vad/finish/timeout에 미커밋 누적분을 통째로
    # 원래 가드가 A에만 몰려 있었고, A에서 스킵된 텍스트는 커밋만 안 될 뿐
    # state.text에 남아 있다가 B로 그대로 새어나갔다. 실측 3건이 전부 이 패턴:
    # 반복 루프 덤프(545토큰), 커밋된 문장 재방출, dot 미확정분 일괄 커밋.
    # 그래서 판정을 여기 한 곳에 모으고 두 경로가 같은 파이프라인을 태운다.
    #
    # A는 판정 시점이 둘로 나뉜다 — 문장을 뽑는 시점(extract)과 커서를 확정하는
    # 시점(commit). 커서 확정은 flush_lock 안에서 일어나므로 합칠 수 없다.
    # stage로 어느 가드가 도는지만 구분하고, 가드 자체는 한 벌만 둔다.
    #   extract: 같은 청크 안에서 뽑힌 문장에 대한 판정
    #   commit : 이미 커밋된 이력과의 대조
    #   flush  : B는 두 시점이 하나라 전부 돈다
    #
    # 아래 표가 "어떤 가드가 어느 경로에서 도는가"의 단일 출처다. 빈칸(=flush에 없는
    # 가드)은 사고가 아니라 아래 근거로 내린 판단이며, 새 가드를 추가할 때도 여기서
    # 적용 시점을 명시적으로 정해야 한다. 예전엔 이 정보가 코드 여기저기 흩어져 있어
    # flush에 가드가 없다는 사실 자체가 보이지 않았다.
    _GUARD_STAGES = {
        # 한 번의 추출 안에서 같은 문장이 연속으로 나온 경우.
        # flush에서는 돌리지 않는다 — 같은 관심사를 _collapse_repetition이 이미 처리하고,
        # 그쪽은 정상 발화의 2회 반복을 일부러 보존한다(3005-163389-0001의 정답이
        # "TEAR DOWN THE FENCE TEAR DOWN THE FENCE"). 여기서 1회로 깎으면 그게 깨진다.
        "rep-dedup": ("extract",),
        # 경계 단어 하나가 겹쳤다고 문장 전체를 버리는 공격적인 가드.
        # flush는 누적분이라 한 문장이 길고, 잘못 발동하면 손실이 크므로 제외.
        "seg-boundary-dedup": ("extract",),
        # flush에는 누적분 앞머리를 잘라내는 자체 prefix-strip 블록이 따로 있다
        # (dot_switch_prev_committed 소비). 문장 단위 접미사 비교와 대상이 달라 중복 적용하지 않는다.
        "dot-suffix-dedup": ("extract",),
        # 이미 커밋된 문장과의 대조. 두 경로 모두 필요 — flush에 없어서 dot으로 커밋된
        # 문장이 finish로 재방출됐다(3538-142836-0017 등).
        "cross-dedup": ("commit", "flush"),
        "cross-dedup-fuzzy": ("commit", "flush"),
    }

    def _commit_skip_reason(self, slot: dict, sentence_display: str, *, stage: str,
                            trigger: Optional[str] = None,
                            batch_last: Optional[str] = None,
                            batch_repeat: int = 0) -> Optional[str]:
        """`sentence_display`를 커밋하면 안 되는 이유를 반환. 커밋해도 되면 None.

        가드를 새로 추가할 때는 여기에만 넣고 `_GUARD_STAGES`에 적용 시점을 적는다.
        경로별로 따로 넣으면 한쪽에만 들어가 같은 누수가 재발한다.

        주의: `seg_reset_last_committed` / `dot_switch_prev_committed`는 "직전 커밋의
        흔적" 마커라 판정 성공 여부와 무관하게 한 번 보면 소비한다(pop). 원래 코드의
        부수효과를 그대로 유지한 것.
        """
        def _applies(name):
            return stage in self._GUARD_STAGES[name]

        if not sentence_display:
            return None

        # 한 번의 추출/flush 안에서 같은 문장이 연달아 나온 경우
        # 상한형: 연속 반복을 _REP_DEDUP_MAX_REPEATS회까지는 통과시킨다.
        # 예전엔 두 번째부터 전부 버렸는데 낭독체의 실제 반복까지 깎였다
        # (baseline 27건 전부 오탐 — 'Ha!' 25 / 'Yes.' 1 / 'I hate him.' 1).
        if (_applies("rep-dedup") and self.rep_dedup
                and batch_last is not None and sentence_display == batch_last
                and batch_repeat >= self._REP_DEDUP_MAX_REPEATS):
            return "rep-dedup"

        # SEG 리셋 직후: 직전 커밋의 끝 단어가 새 문장 첫 단어로 다시 나옴.
        # 모드2는 커밋 단위가 문장이 아니라 청크라 경계 단어 하나로 청크째 폐기된다 —
        # 실측상 억제 효과도 없었고(141건 중 4건), 2초 강제 커밋의 경계 중복은
        # 모드2의 측정 대상 그 자체라 그대로 노출한다.
        if _applies("seg-boundary-dedup") and "seg_reset_last_committed" in slot and not self.always_commit:
            seg_reset_last = slot.pop("seg_reset_last_committed")
            _strip_p = lambda w: re.sub(r'[.,!?;:。？！]+$', '', w)
            words = sentence_display.split()
            first_word = _strip_p(words[0]) if words else ""
            last_word = _strip_p(seg_reset_last.split()[-1]) if seg_reset_last.split() else ""
            if first_word and last_word and (first_word == last_word or last_word.endswith(first_word)):
                return "seg-boundary-dedup"

        # DOT-SLOT-SWITCH 직후: 이전 슬롯 커밋의 접미사가 통째로 다시 나옴
        if _applies("dot-suffix-dedup"):
            prev_committed = slot.get("dot_switch_prev_committed", "")
            if prev_committed and (trigger == "dot" or stage == "flush"):
                slot.pop("dot_switch_prev_committed", None)
                _norm = lambda s: re.sub(r'[.,!?;:。？！\s]+', '', s)
                if _norm(prev_committed).endswith(_norm(sentence_display)):
                    return "dot-suffix-dedup"

        # 이 발화에서 이미 커밋된 문장 (완전 일치)
        if _applies("cross-dedup"):
            _asr_key = ' '.join(sentence_display.split()).rstrip('.,!?;:。？！').strip().lower()
            if _asr_key in slot.get("committed_asr_set", set()):
                return "cross-dedup"

        # 같은 문장인데 재디코딩으로 철자만 바뀐 경우.
        # 모드2는 seg-boundary-dedup과 같은 이유로 제외 (커밋 단위가 청크).
        if (_applies("cross-dedup-fuzzy") and not self.always_commit
                and self._cross_dup_match(slot, sentence_display) is not None):
            return "cross-dedup-fuzzy"

        return None

    def _apply_commit_guards(self, slot: dict, text: str, slot_key=None) -> str:
        """flush 경로(B): 미커밋 누적분을 문장 단위로 쪼개 커밋 가드를 태운다.

        A는 문장을 하나씩 뽑아 가드에 태우는데, B는 누적분을 통째로 내보내던 탓에
        가드를 하나도 거치지 않았다. 여기서 같은 파이프라인(`_commit_skip_reason`)을
        문장 단위로 적용해 두 경로의 판정을 일치시킨다.
        `always_commit`(모드2)은 커밋 단위가 문장이 아니라 청크라 적용하지 않는다.
        """
        if self.always_commit or not text:
            return text
        kept, batch_last = [], None
        for sent in split_sentences(text):
            disp = sent.strip()
            if not disp:
                continue
            reason = self._commit_skip_reason(slot, disp, stage="flush", batch_last=batch_last)
            if reason:
                self.log.info(f"[COMMIT-SKIP] reason={reason} stage=flush slot={slot_key} text={disp!r}")
                continue
            _trimmed = self._strip_committed_prefix(slot, disp)
            if not _trimmed:
                continue
            if _trimmed != disp:
                self.log.info(
                    f"[COMMIT-TRIM] reason=committed-prefix stage=flush slot={slot_key} "
                    f"kept={_trimmed!r}"
                )
                sent = _trimmed + " "
                disp = _trimmed
            kept.append(sent)
            batch_last = disp
        return "".join(kept).strip()

    @classmethod
    def _collapse_repetition(cls, text: str, max_repeats: int = 2) -> str:
        """연속 반복을 `max_repeats`회까지만 남기고 접는다.

        모델이 반복 루프에 빠지면 같은 문장이 수십 번 이어진다. 스트리밍 커밋 경로는
        rep-dedup/cross-dedup으로 이걸 걸러내지만, 걸러진 텍스트는 커밋되지 않은 채
        state.text에 그대로 남는다. 그 상태로 스트림이 끝나면 flush_uncommitted가
        누적분을 통째로 내보내 반복이 그대로 최종 출력에 실린다
        (실측: 08/04 mode4 ko 실행에서 참조 31단어 발화가 556단어로 커밋, finish
        세그먼트 하나에 545토큰). 그래서 flush 직전에 한 번 더 접는다.

        `max_repeats=2`인 이유: 실제 발화에도 같은 문장을 두 번 잇는 경우가 있다
        (LibriSpeech 3005-163389-0001의 정답이 "TEAR DOWN THE FENCE TEAR DOWN THE
        FENCE"). 2회까지 보존하면 정상 반복은 건드리지 않고 루프만 잘라낸다.

        루프가 항상 한 문장 주기인 것도 아니다 — 실측 9건 중 4건이 "A. B. A. B. ..."처럼
        두 문장을 한 묶음으로 돌았다. 그래서 단일 단위가 아니라 주기 n의 반복을 찾는다.
        """
        if not text:
            return text

        def _collapse(units, keys, max_period):
            """keys가 주기 n으로 max_repeats회를 넘겨 반복되면 앞 max_repeats주기만 남긴다.

            긴 주기부터 시도해야 "A B A B"를 A와 B 각각의 1주기 반복으로 잘못 보지 않는다.
            """
            out, dropped, i = [], 0, 0
            while i < len(units):
                for n in range(min(max_period, (len(units) - i) // (max_repeats + 1)), 0, -1):
                    base = keys[i:i + n]
                    if not any(base):
                        continue
                    j = i + n
                    while keys[j:j + n] == base:
                        j += n
                    if (j - i) // n > max_repeats:
                        out.extend(units[i:i + n * max_repeats])
                        dropped += (j - i) // n - max_repeats
                        i = j
                        break
                else:
                    out.append(units[i])
                    i += 1
            return out, dropped

        # 1) 문장 단위 — 관측된 반복 루프는 전부 구두점으로 끝나는 문장의 반복이었다.
        sentences = split_sentences(text)
        kept, dropped = _collapse(sentences, [cls._boundary_key(s) for s in sentences],
                                  max_period=4)
        result = "".join(kept)

        # 2) 단어 단위 — 구두점 없이 도는 루프까지 덮는다.
        words = result.split()
        kept_w, d2 = _collapse(words, [re.sub(r'[^\w]+', '', w).lower() for w in words],
                               max_period=8)
        if d2:
            dropped += d2
            result = " ".join(kept_w)

        if dropped:
            logger.info("[FLUSH-DEDUP] 반복 %d주기 제거 (%d자 → %d자)",
                        dropped, len(text), len(result))
        return result.strip()

    async def _process_slot_updates(self, slot_key: str, force_reason: Optional[str] = None,
                                    chunk_end: bool = False,
                                    final: bool = False) -> Optional[str]:
        """
        chunk_end:
            True면 한 청크의 디코딩이 완전히 끝난 뒤 호출된 것.
            dot_commit_confirm의 "합의 확정"(규칙 2)은 청크 간 가설 비교이므로
            generate() 루프 중간(on_seg/on_dot 콜백)이 아니라 이 경로에서만 판정한다.
        """
        slot = self._slot(slot_key)
        state = slot["state"]
        current_text = self._strip_asr_text((state.text or "").strip())
        current_lang = state.language or ""
        # dot_commit_confirm에서는 텍스트가 그대로여도 청크 종료 시점엔 항상 게이트를 돌려야 한다.
        #  - 무음 청크처럼 가설이 안 바뀌는 상황이 바로 "합의 확정"이 성립하는 경우고,
        #  - 규칙 3(정체 확정) 카운터도 가설이 안 자란 청크에서 올라가야 하며,
        #  - generate 루프 안의 on_dot 콜백이 이미 last_text를 갱신해버리기 때문에
        #    "pending이 있을 때만" 통과시키면 pending을 등록할 기회 자체가 사라진다
        #    (등록은 chunk_end에서만 하므로 조기 리턴 ↔ 미등록 교착).
        #  - final(스트림 종료, 규칙 4)은 dot_commit_confirm과 무관하게 반드시 통과시킨다.
        #    종료 직전 청크에서 이미 같은 텍스트로 한 번 돌았으면 current_text ==
        #    last_text라 여기서 조기 리턴되고, 잔여가 통째로 flush(finish)로 흘러간다.
        _recheck_pending = chunk_end and (self.dot_commit_confirm or final)
        if not current_text or (current_text == slot["last_text"] and not _recheck_pending):
            return None

        slot["last_text"] = current_text
        slot["last_text_lang"] = current_lang
        if "<SEG>" in current_text:
            self.log.info(f"[SEG-IN-TEXT] slot={slot_key} text={current_text!r}")

        # 문장 단위 commit
        # committed_display/seg_count 기준으로 uncommitted 구간 계산 (모델 텍스트 수정에도 안전)
        uncommitted = self._uncommitted_from(
            current_text, slot["committed_display"], slot["committed_seg_count"]
        )
        sentences_to_commit = []
        remaining = uncommitted
        _last_extracted_display = None  # 한 번의 호출 내 연속 중복 억제용
        _last_extracted_repeat = 0      # 위 문장이 연속으로 몇 번 커밋됐는지

        # ── 규칙 3(정체 확정)용 카운터 ────────────────────────────────────
        # 오디오는 계속 누적되는데 미커밋 가설의 토큰 수가 자라지 않으면 발화가 끝난 것.
        # 문자열 동일이 아니라 "토큰 수 동일"로 보는 이유: 롤백 창 안에서 문구만
        # 미세하게 바뀌는 경우(Anon. → And on.)에도 발화 종료로 인정해야 하기 때문.
        _stall_hit = False
        _prev_sentences: tuple = ()
        _prev_accum = -1
        if chunk_end and self.dot_commit_confirm:
            _accum_c = int(state.audio_accum.shape[0]) if state.audio_accum is not None else 0

            # 규칙 2 비교용 — 직전 청크의 경계 목록을 꺼내 두고 이번 청크 것으로 교체.
            # (이 함수는 아래에서 여러 번 조기 리턴하므로 여기서 미리 갱신한다)
            _prev_sentences = slot.get("prev_boundary_sentences", ())
            _prev_accum = slot.get("prev_boundary_accum", -1)
            slot["prev_boundary_sentences"] = self._extract_boundary_keys(uncommitted)
            slot["prev_boundary_accum"] = _accum_c

            if self.dot_commit_stall_chunks > 0:
                _unc_tokens = self._count_tokens(uncommitted)
                # 직전 청크 이후 커밋이 일어났으면 uncommitted가 가리키는 구간 자체가 달라진다.
                # 그 상태로 토큰 수만 비교하면 서로 다른 텍스트가 우연히 같은 길이일 때
                # 정체로 오판한다(실측: 'My dear," said Miss.'와 '"Pray don't."'가 둘 다 6토큰).
                _committed_len = len(slot["committed_display"])
                _same_base = _committed_len == slot.get("stall_committed_len")
                if (_same_base and _unc_tokens == slot.get("stall_tokens")
                        and _accum_c > slot.get("stall_accum", -1)):
                    slot["stall_count"] = slot.get("stall_count", 0) + 1
                else:
                    slot["stall_count"] = 0
                    slot["stall_tokens"] = _unc_tokens
                slot["stall_committed_len"] = _committed_len
                slot["stall_accum"] = _accum_c
                _stall_hit = slot["stall_count"] >= self.dot_commit_stall_chunks

        while True:
            if self.always_commit:
                # 모드2: SEG/dot 트리거 없이 이번 청크에서 새로 디코딩된 부분을 통째로 커밋
                if not remaining.strip():
                    break
                trigger = "always"
                after = ""
                sentence = remaining.strip()
            else:
                # 우선순위 1: <SEG>
                # 우선순위 2: VAD (flush_uncommitted 에서 처리)
                # 우선순위 3: dot (enable_dot_commit=True일 때만 활성화)
                # dot 패턴: Mr./Mrs./Dr./St./Jr./Sr./vs./No. 등 약어 제외.
                # 문자열 끝 마침표도 경계로 인정 — <SEG>처럼 발화의 마지막 문장도
                # 커밋 가능해야 함 (sentence_boundary.py). 예전엔 뒤에 단어가
                # 더 와야만 매치되는 룩어헤드 때문에 마지막 문장이 항상 finish
                # 커밋으로만 빠졌다.
                if self.enable_dot_commit:
                    match = DOT_COMMIT_BOUNDARY_RE.search(remaining)
                else:
                    match = re.search(r"<SEG>", remaining)
                if not match:
                    if not (final and remaining.strip()):
                        break
                    # 스트림 종료인데 경계를 못 찾은 잔여. 약어(No./S. J.)나 구두점 없는
                    # 짧은 발화('si')가 여기 온다. 오디오가 더 오지 않으므로 이 잔여는
                    # 그 자체로 완결된 단위다 — 경계가 없다고 flush로 흘려보낼 이유가 없다.
                    # 커밋 사유는 축을 따라간다. dot 축이 아닌데 "dot"으로 찍으면
                    # metric의 commit_reason_counts에 남의 축 라벨이 섞인다.
                    trigger = "dot" if self.enable_dot_commit else "seg"
                    after = ""
                    sentence = remaining.strip()
                    self.log.info(
                        f"[COMMIT-RESIDUAL] rule=final slot={slot_key} "
                        f"trigger={trigger} text={sentence!r}"
                    )
                    sentence_display_check = sentence.replace("<SEG>", "").strip()
                    if sentence_display_check:
                        _skip = self._commit_skip_reason(
                            slot, sentence_display_check, stage="extract",
                            trigger=trigger, batch_last=_last_extracted_display,
                            batch_repeat=_last_extracted_repeat,
                        )
                        if _skip:
                            self.log.info(
                                f"[COMMIT-SKIP] reason={_skip} stage=extract slot={slot_key} "
                                f"text={sentence_display_check!r}"
                            )
                            sentences_to_commit.append((sentence, trigger, False))
                        else:
                            sentences_to_commit.append((sentence, trigger, True))
                            slot.pop("pending_dot_text", None)
                            slot.pop("pending_dot_accum", None)
                            _last_extracted_repeat = (
                                _last_extracted_repeat + 1
                                if sentence_display_check == _last_extracted_display else 1)
                            _last_extracted_display = sentence_display_check
                    break
                matched_text = match.group()
                trigger = "seg" if "<SEG>" in matched_text else "dot"
                after = remaining[match.end():]
                # 커서 추적을 위해 raw sentence(<SEG> 포함) 사용
                sentence = remaining[:match.end()].strip()

                if trigger == "dot" and self.dot_commit_confirm:
                    pending = slot.get("pending_dot_text")
                    # 후보를 등록한 시점의 누적 오디오 길이. 합의 확정은 "새 오디오를 더 듣고
                    # 재디코딩한 뒤에도 동일"이어야 하므로 같은 청크 안에서는 확정하지 않는다.
                    _accum_now = int(state.audio_accum.shape[0]) if state.audio_accum is not None else 0
                    _pending_accum = slot.get("pending_dot_accum", -1)
                    tail_tokens = self._count_tokens(after)
                    if tail_tokens > state.unfixed_token_num:
                        # 규칙 1: 마침표가 롤백 창 밖 → 다음 청크에서 수정되지 않음
                        slot.pop("pending_dot_text", None)
                        slot.pop("pending_dot_accum", None)
                        self.log.info(
                            f"[DOT-CONFIRM] rule=context slot={slot_key} "
                            f"tail_tokens={tail_tokens} text={sentence!r}"
                        )
                    elif (chunk_end and self._boundary_key(sentence) in _prev_sentences
                          and _accum_now > _prev_accum):
                        # 규칙 2: 직전 청크 가설에도 있던 경계가 재디코딩 후에도 살아남음 → 확정.
                        # 한 청크에 경계가 여러 개면 각각 독립적으로 판정되므로,
                        # 발화 마지막 문장이 앞 문장 커밋을 기다리다 오디오가 떨어지는 일이 없다.
                        slot.pop("pending_dot_text", None)
                        slot.pop("pending_dot_accum", None)
                        self.log.info(f"[DOT-CONFIRM] rule=stable slot={slot_key} text={sentence!r}")
                    elif final:
                        # 규칙 4: 스트림 종료 — 오디오가 더 오지 않는다는 건 발화가 끝났다는
                        # 가장 강한 증거이므로 더 유예할 이유가 없다.
                        # 규칙 2/3은 "새 오디오를 더 듣고 재디코딩했는데도 같더라"를 조건으로 거는데
                        # (_accum_now > _prev_accum / _accum_c > stall_accum), finish 시점엔 오디오가
                        # 늘지 않으므로 원천적으로 불성립한다. 그래서 마지막 문장이 확정되지 못한 채
                        # flush로 빠져 발화 전체가 finish 한 덩어리로 나갔다(실측: 21초 발화가 26.8초에
                        # 통째로).
                        #
                        # 규칙 3과 달리 마지막 경계로 한정하지 않고 **보류된 모든 경계**를 확정한다.
                        # 커밋 루프는 앞 경계가 미확정이면 거기서 break라 뒤 경계까지 가지 못한다.
                        # 실측: "What's that there? said Dicky."에서 '?'가 롤백 창 안이라 규칙 1이
                        # 불발하고, 뒤에 'said Dicky.'가 있어 마지막 경계도 아니라 규칙 4에서도
                        # 걸러졌다 → 발화 전체가 finish. 스트림이 끝난 뒤엔 어떤 경계도 더 수정될
                        # 일이 없으므로 롤백 창 개념 자체가 무의미하다.
                        slot.pop("pending_dot_text", None)
                        slot.pop("pending_dot_accum", None)
                        self.log.info(f"[DOT-CONFIRM] rule=final slot={slot_key} text={sentence!r}")
                    elif _stall_hit and not after.strip():
                        # 규칙 3: 오디오는 더 들어왔는데 가설이 자라지 않음 → 발화 종료로 판정.
                        # 규칙 2가 놓치는 "롤백 창 안 문구 수정" 케이스를 흡수한다.
                        # "발화가 끝났다"는 판정이므로 마지막 경계(뒤에 아무것도 없음)에만 적용한다.
                        # 중간 경계는 규칙 1/2가 담당.
                        slot.pop("pending_dot_text", None)
                        slot.pop("pending_dot_accum", None)
                        slot["stall_count"] = 0
                        self.log.info(
                            f"[DOT-CONFIRM] rule=stall slot={slot_key} "
                            f"chunks={self.dot_commit_stall_chunks} text={sentence!r}"
                        )
                    else:
                        # 미확정 — 보류하고 이번 호출에서는 커밋하지 않는다.
                        # 보류 후보는 "청크 종료 시점의 프론티어"여야 한다. generate 루프 안의
                        # on_dot 콜백(chunk_end=False)이 중간 가설로 덮어쓰면, 청크 종료 호출은
                        # 직전 청크가 아니라 같은 청크 중간값과 비교하게 되어 규칙 2가 영원히
                        # 성립하지 않는다(실측: 매 청크 pending이 두 문장 사이를 왕복).
                        if chunk_end and pending != sentence:
                            self.log.info(
                                f"[DOT-PENDING] slot={slot_key} "
                                f"{'revised' if pending else 'new'} text={sentence!r}"
                            )
                            slot["pending_dot_text"] = sentence
                            slot["pending_dot_accum"] = _accum_now
                        break
            sentence_display_check = sentence.replace("<SEG>", "").strip()
            if sentence_display_check:
                # 세 가드 모두 스킵 시 아래 `remaining = after`로 흘러 다음 경계를 보므로
                # 제어 흐름은 하나로 합쳐도 동일하다. 판정은 _commit_skip_reason 한 곳.
                _skip = self._commit_skip_reason(
                    slot, sentence_display_check, stage="extract",
                    trigger=trigger, batch_last=_last_extracted_display,
                    batch_repeat=_last_extracted_repeat,
                )
                if _skip:
                    self.log.info(
                        f"[COMMIT-SKIP] reason={_skip} stage=extract slot={slot_key} "
                        f"text={sentence_display_check!r}"
                    )
                    # 스킵은 "안 내보낸다"이지 "원문에서 소비 안 했다"가 아니다.
                    # 목록에서 빼버리면 Phase 1의 tail이 스킵된 문장에서 시작한 채로
                    # 다음 문장의 startswith 검사를 받아 실패 → break, 뒤 문장 전부가
                    # 미커밋으로 남아 flush(finish)로 흘러간다(실측 CoVoST2-spk
                    # en_de_2c41823446e3: 모델이 'Fine.'을 두 번 뱉어 중복 1건이
                    # 뒤의 'I agree.' / 'This could be the case.'를 인질로 잡아
                    # finish 3건). emit=False로 함께 실어 커서만 전진시킨다.
                    # commit 단계 스킵에는 이미 같은 처리가 있다(아래 Phase 1).
                    sentences_to_commit.append((sentence, trigger, False))
                else:
                    sentences_to_commit.append((sentence, trigger, True))
                    slot.pop("pending_dot_text", None)  # 커밋되면 보류 후보는 무효
                    slot.pop("pending_dot_accum", None)
                    _last_extracted_repeat = (
                        _last_extracted_repeat + 1
                        if sentence_display_check == _last_extracted_display else 1)
                    _last_extracted_display = sentence_display_check
            remaining = after

        if not sentences_to_commit:
            return None

        # ── Phase 1: 커서 추적 + committed_len 즉시 확정 (GPT 호출 전) ─────────
        # raw ASR 텍스트 기준으로 커서를 확정하므로 GPT 교정 결과와 무관하게 안전.
        # sentences_to_commit 원소는 (raw, trigger, emit)이며 emit=False는 "소비했지만
        # 배출하지 않는다" — 커서만 전진시킨다.
        committed_items = []  # list of (sentence_display, trigger_reason)
        _consumed_seg = 0     # emit=False로 소비한 <SEG> 수 (committed_seg_count 보정용)
        _consumed_any = False  # 배출은 없어도 커서가 전진했는지
        latest_text: Optional[str] = None  # flush_lock 안에서 읽은 스냅샷, 호출자에게 반환
        # ASR revision으로 trailing punct가 바뀌어도 (거고, → 거고) 같은 문장으로 취급.
        # 대소문자도 같은 범주의 재디코딩 변형이다 — 커밋 후 모델이 'Oh, Papa!'를
        # 'Oh, papa!'로 고쳐 쓰면 대소문자 구분 키로는 dedup을 못 해 같은 문장이
        # 두 번 커밋된다(실측: 1688-142285-0016, WER 0.00% → 18.18%).
        _asr_key = lambda s: ' '.join(s.split()).rstrip('.,!?;:。？！').strip().lower()

        async with slot["flush_lock"]:
            async with self.asr_lock:
                latest_state = slot["state"]
                latest_text = (latest_state.text or "").strip() if latest_state else ""

            if force_reason == "vad":
                _PUNCT = '.,!?;:。？！'
                latest_ns = re.sub(r'\s+', ' ', latest_text.replace("<SEG>", "")).strip()
                cdisp = slot["committed_display"]
                pos = len(cdisp) if (cdisp and latest_ns.startswith(cdisp)) else 0
                for sentence_raw, trigger_reason, _emit in sentences_to_commit:
                    sentence_display = sentence_raw.replace("<SEG>", "").strip()
                    _audio_span = self.current_time - slot.get("audio_anchor_sec", 0.0)
                    if _emit:
                        _skip = self._commit_skip_reason(
                            slot, sentence_display, stage="commit", trigger=trigger_reason)
                        if _skip:
                            self.log.info(
                                f"[COMMIT-SKIP] reason={_skip} stage=commit slot={slot_key} "
                                f"span={_audio_span:.2f}s text={sentence_display!r}"
                            )
                            _emit = False
                    sent_core = sentence_display.rstrip(_PUNCT)
                    tail_ns = latest_ns[pos:].lstrip()
                    lead = len(latest_ns[pos:]) - len(tail_ns)
                    if not (sent_core and tail_ns.startswith(sent_core)):
                        break
                    end = len(sent_core)
                    if end < len(tail_ns) and tail_ns[end].isalpha():
                        break
                    while end < len(tail_ns) and tail_ns[end] in _PUNCT:
                        end += 1
                    pos += lead + end
                    if not _emit:
                        # 소비만 하고 배출하지 않는다. 커서를 두면 다음 문장의
                        # startswith가 실패해 뒤 문장 전부가 flush(finish)로 끌려간다.
                        _consumed_any = True
                        continue
                    # 커서(pos)는 원문 기준으로 이미 전진시켰으므로 여기서 잘라도 안전하다
                    sentence_display = self._strip_committed_prefix(slot, sentence_display)
                    if not sentence_display:
                        continue
                    committed_items.append((sentence_display, trigger_reason))
                if committed_items or _consumed_any:
                    slot["committed_display"] = latest_ns[:pos].strip()
                    slot["committed_len"] = len(latest_text)
                    slot["audio_anchor_sec"] = self.current_time
                    slot["committed_asr_set"].update(_asr_key(t) for t, _ in committed_items)
                    slot["committed_fuzzy_keys"].extend(
                        k for k in (self._fuzzy_key(t) for t, _ in committed_items) if k)
            else:
                cursor = self._committed_cursor(
                    latest_text,
                    slot["committed_display"],
                    slot["committed_seg_count"],
                )
                if cursor == -1:
                    _seg = "<SEG>"
                    _p, _found = 0, 0
                    while _found < slot["committed_seg_count"]:
                        _idx = latest_text.find(_seg, _p)
                        if _idx == -1:
                            return None
                        _p = _idx + len(_seg)
                        _found += 1
                    cursor = _p
                tail = latest_text[cursor:]
                for sentence_raw, trigger_reason, _emit in sentences_to_commit:
                    sentence_display = sentence_raw.replace("<SEG>", "").strip()
                    _audio_span = self.current_time - slot.get("audio_anchor_sec", 0.0)
                    if _emit:
                        _skip = self._commit_skip_reason(
                            slot, sentence_display, stage="commit", trigger=trigger_reason)
                        if _skip:
                            self.log.info(
                                f"[COMMIT-SKIP] reason={_skip} stage=commit slot={slot_key} "
                                f"span={_audio_span:.2f}s text={sentence_display!r}"
                            )
                            _emit = False
                    if not _emit:
                        # 스킵은 "안 내보낸다"이지 "원문에서 소비 안 했다"가 아니다.
                        # 커서를 그대로 두면 다음 문장의 startswith 검사가 실패해 break로
                        # 빠지고, 뒤 문장 전부가 미커밋으로 남아 finish flush로 흘러간다
                        # (실측 1998-29455-0019: 중복 1건이 뒤의 'You are.' 2문장을 인질로
                        # 잡아 finish행). 스킵한 문장만큼 커서를 전진시킨다.
                        # extract 단계 스킵도 emit=False로 여기 도달한다.
                        _st = tail.lstrip()
                        _lw = len(tail) - len(_st)
                        while _st.startswith("<SEG>"):
                            _st = _st[len("<SEG>"):].lstrip()
                            _lw = len(tail) - len(_st)
                        if _st.startswith(sentence_raw):
                            cursor += _lw + len(sentence_raw)
                            tail = latest_text[cursor:]
                            _consumed_seg += sentence_raw.count("<SEG>")
                            _consumed_any = True
                        continue
                    sentence_display = self._strip_committed_prefix(slot, sentence_display)
                    if not sentence_display:
                        continue
                    stripped_tail = tail.lstrip()
                    leading_ws = len(tail) - len(stripped_tail)
                    # <SEG> 토큰이 문장 사이에 있을 때 건너뜀
                    while stripped_tail.startswith("<SEG>"):
                        stripped_tail = stripped_tail[len("<SEG>"):].lstrip()
                        leading_ws = len(tail) - len(stripped_tail)
                    if not stripped_tail.startswith(sentence_raw):
                        # 커서가 어긋나면 포기한다. tail.find()로 재동기화도 해봤으나,
                        # 같은 문구가 반복되면('Ha!' 3연속) 엉뚱한 occurrence로 점프해
                        # 커밋 순서가 뒤집혔다. 커서 갭의 주원인이던 rep-dedup을 상한형으로
                        # 바꾼 뒤로는 재동기화가 한 번도 발동하지 않아(실측 0회) 제거했다.
                        break
                    cursor += leading_ws + len(sentence_raw)
                    tail = latest_text[cursor:]
                    committed_items.append((sentence_display, trigger_reason))
                if committed_items or _consumed_any:
                    slot["committed_len"] = cursor
                    slot["committed_prefix"] = latest_text[:cursor]
                    slot["committed_display"] = re.sub(
                        r'\s+', ' ', slot["committed_prefix"].replace("<SEG>", "")
                    ).strip()
                    slot["committed_seg_count"] += (
                        sum(1 for _, tr in committed_items if tr == "seg") + _consumed_seg)
                    slot["audio_anchor_sec"] = self.current_time
                    slot["committed_asr_set"].update(_asr_key(t) for t, _ in committed_items)
                    slot["committed_fuzzy_keys"].extend(
                        k for k in (self._fuzzy_key(t) for t, _ in committed_items) if k)

        if not committed_items:
            return None

        # ── Phase 2: GPT 번역 ─────────────────────────────────────────────────
        if self._in_generate_loop:
            # generate() 루프 안: GPT를 백그라운드 태스크로 발사하고 즉시 반환.
            # ASR 디코딩과 번역이 병렬 실행된다.
            for sentence_display, trigger_reason in committed_items:
                effective_reason = force_reason or trigger_reason
                task = asyncio.create_task(
                    self._correct_and_translate(sentence_display, current_lang, self.current_time)
                )
                self._pending_gpt_tasks.append({
                    "task": task,
                    "slot_key": slot_key,
                    "original": sentence_display,
                    "trigger_reason": effective_reason,
                    "audio_end_sec": self.current_time,
                })
        else:
            # generate() 루프 밖 (VAD/finish/post-generate): 직접 await
            for sentence_display, trigger_reason in committed_items:
                effective_reason = force_reason or trigger_reason
                corrected, translation, lang, extra = await self._correct_and_translate(
                    sentence_display, current_lang, self.current_time
                )
                await self._emit_final_payload(
                    slot_key=slot_key,
                    original=sentence_display,
                    translation=translation,
                    language=lang,
                    reason=effective_reason,
                    audio_end_sec=self.current_time,
                    extra=extra,
                )
                self.log.info(
                    f"[TRANS-SEG] slot={slot_key} lang={lang} "
                    f"original='{sentence_display}' translation='{translation}'"
                )

        return self._strip_asr_text(latest_text) if latest_text is not None else None

    def _run_vad_sync(self, chunk: np.ndarray, chunk_base_sample: int):
        """VAD 추론을 동기로 실행 (run_in_executor에서 호출).

        Returns:
            list[int]: 발화 종료 local index 목록. 예외 발생 시 None 반환.
        """
        vad_end_local_indices: list[int] = []
        try:
            offset = 0
            while offset + VAD_WINDOW_SIZE_SAMPLES <= chunk.size:
                window = chunk[offset:offset + VAD_WINDOW_SIZE_SAMPLES]
                speech_dict = self.vad_iterator(
                    torch.from_numpy(window),
                    return_seconds=False,
                )
                if speech_dict is not None:
                    window_end_sample = self.sample_cursor - (chunk.size - offset - VAD_WINDOW_SIZE_SAMPLES)
                    if "start" in speech_dict:
                        self.vad_last_speech_start_sample = int(window_end_sample)
                        self.log.info(f"[VAD-DETECT] speech_start target_samples={window_end_sample}")
                    if "end" in speech_dict:
                        end_sample = window_end_sample
                        local_idx = int(end_sample - chunk_base_sample)
                        local_idx = max(0, min(int(chunk.size), local_idx))
                        if not vad_end_local_indices or local_idx > vad_end_local_indices[-1]:
                            vad_end_local_indices.append(local_idx)
                        self.log.info(f"[VAD-DETECT] speech_end target_samples={end_sample}")
                offset += VAD_WINDOW_SIZE_SAMPLES
        except Exception as e:
            self.log.warning(f"[vad] error, disabling for this session: {e}")
            return None
        return vad_end_local_indices

    async def process_audio_chunk(self, audio_data: bytes):
        if not audio_data:
            return

        if self.recorder is not None:
            self.recorder.write(audio_data)

        chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if chunk.size == 0:
            return

        chunk_base_sample = self.sample_cursor
        self.sample_cursor += chunk.size
        vad_end_local_indices: list[int] = []

        # ── VAD: VADIterator로 스트리밍 음성 구간 탐지 ──
        # 이벤트 루프에서 직접 동기 실행 — 512 샘플 윈도우 기준 추론당 < 0.5ms이므로
        # executor 불필요. 공유 스레드풀 사용 시 두 클라이언트가 동시에 PyTorch 추론을
        # 실행해 내부 상태 충돌 → 한 쪽 VAD exception → vad_enabled=False 버그 방지.
        if self.vad_enabled and self.vad_iterator is not None:
            vad_result = self._run_vad_sync(chunk, chunk_base_sample)
            if vad_result is None:
                self.vad_enabled = False
            else:
                vad_end_local_indices = vad_result

        self.current_time += chunk.size / SAMPLING_RATE
        seg_start = 0
        for local_cut in vad_end_local_indices:
            if local_cut <= seg_start:
                continue

            pre_chunk = chunk[seg_start:local_cut]
            # VAD 커밋 직전 buffer 크기 스냅샷 (SEG 리셋 전)
            _active_state_pre = self.stream_slots[self.active_slot]["state"]
            _buf_pre_transcribe = _active_state_pre.buffer.shape[0] + pre_chunk.size
            _accum_pre_transcribe = _active_state_pre.audio_accum.shape[0]
            if pre_chunk.size > 0:
                await self._asr_streaming_transcribe(pre_chunk, self.active_slot)

            target_sample = chunk_base_sample + local_cut
            target_audio_end_sec = target_sample / SAMPLING_RATE

            old_active = self.active_slot
            self.active_slot, self.standby_slot = self.standby_slot, self.active_slot
            self.state = self.stream_slots[self.active_slot]["state"]
            self.stream_slots[self.active_slot]["audio_anchor_sec"] = target_audio_end_sec
            self.log.info(
                f"[VAD-SLOT-SWITCH] old={old_active} new={self.active_slot} at={target_audio_end_sec:.3f}s"
            )

            await self._on_vad_commit(target_audio_end_sec)
            # 스트리밍 상태 스냅샷 (finish_streaming 전)
            _pre_state = self.stream_slots[old_active]["state"]
            _pre_committed = self.stream_slots[old_active]["committed_display"]
            _pre_seg_count = self.stream_slots[old_active]["committed_seg_count"]
            _pre_text = (_pre_state.text or "").strip()
            _pre_uncommitted = self._uncommitted_from(_pre_text, _pre_committed, _pre_seg_count)
            self.log.info(
                f"[VAD-HISTORY] slot={old_active} "
                f"committed={_pre_committed!r} uncommitted={_pre_uncommitted!r}"
            )
            await self._process_slot_updates(old_active, force_reason="vad")
            # SEG 리셋이 일어났으면 _pre_state.buffer는 0이므로, 리셋 전 snapshot 사용
            _cur_buf = _pre_state.buffer.shape[0]
            # audio_accum이 없는 경우(첫 청크 미만)만 short-utterance로 간주
            _is_short_utterance = _accum_pre_transcribe == 0 and _buf_pre_transcribe > 0
            _has_buffered_audio = _cur_buf > 0 or _is_short_utterance
            self.log.info(
                f"[VAD-BUFFER] slot={old_active} cur_buf={_cur_buf} "
                f"buf_pre={_buf_pre_transcribe} accum_pre={_accum_pre_transcribe} "
                f"short={_is_short_utterance} has_buf={_has_buffered_audio}"
            )
            if _pre_text or _has_buffered_audio:
                await self._asr_finish_streaming(old_active)
                _finish_text = (self.stream_slots[old_active]["state"].text or "").strip()
                self.log.info(f"[VAD-FINISH] slot={old_active} text={_finish_text!r}")
                # 결과가 비어있고 VAD speech_start 위치가 있으면 발화 구간만 트림하여 재시도
                if not _finish_text and self.vad_last_speech_start_sample > 0:
                    await self._retry_vad_short_utterance(old_active)
                    _finish_text = (self.stream_slots[old_active]["state"].text or "").strip()
            self.vad_last_speech_start_sample = 0  # 이 VAD 이벤트 처리 완료 후 초기화
            # finish_streaming이 uncommitted를 날려버렸으면 스트리밍 텍스트로 복원
            _post_state = self.stream_slots[old_active]["state"]
            _post_text = (_post_state.text or "").strip()
            _post_uncommitted = self._uncommitted_from(
                _post_text,
                self.stream_slots[old_active]["committed_display"],
                self.stream_slots[old_active]["committed_seg_count"],
            )
            if not _post_uncommitted and _pre_uncommitted:
                _post_state.text = _pre_text
            # slot-switch에서 같은 발화의 SEG 커밋(앞 오디오)이 아직 emit 안 됐을 수 있으므로,
            # VAD 커밋(뒤 오디오) 전에 먼저 완료·emit해 segment_id가 오디오 순서로 부여되게 한다.
            await self._drain_pending_gpt()
            await self.flush_uncommitted(force=True, reason="vad", slot_key=old_active)
            self._reset_stream_slot(self.standby_slot)
            await self._on_vad_done(old_active, tail_samples=chunk.size - local_cut)

            seg_start = local_cut

        tail_chunk = chunk[seg_start:]
        if tail_chunk.size > 0:
            await self._asr_streaming_transcribe(tail_chunk, self.active_slot)

    async def finish_streaming(self):
        # 스트림 종료(finish/stop) 시 VAD가 커밋하지 않은 남은 텍스트를 마저 커밋한다.
        # 짧은 발화 등으로 VAD speech_start가 안 잡히면 디코드된 텍스트가 커밋 없이 버려지는데,
        # finish 시점에 flush해 복구한다(정상 클립은 이미 커밋돼 uncommitted가 없어 no-op).
        #
        # flush로 가기 전에 dot 확정 판정을 한 번 더 태운다(규칙 4). flush는 남은 걸 통째로
        # 한 커밋에 담지만, 확정 게이트를 태우면 문장 단위로 쪼개져 나간다. 커밋 시점은
        # 어차피 같은 순간이라 지연 손해는 없고, 클라이언트/번역이 문장 단위로 받는다.
        #
        # 규칙 4는 **모든 축**에 태운다. 예전엔 dot 확정 게이트 전용이라
        # `enable_dot_commit and dot_commit_confirm` 으로 막혀 있었는데, 그러면
        # seg/always 축은 이 블록을 통째로 건너뛰고 잔여가 전부 flush(finish)로
        # 흘러갔다(실측 CoVoST2-spk seg 축: 18,655 세그 중 5,793개가 finish).
        # flush 는 잔여를 dot 경계 정규식으로 쪼개므로 SEG 기반 축에서 분할 기준이
        # 어긋나고, 커서 전진/스킵 가드도 타지 않는다. 잔여는 축의 커밋 경로로 내보낸다.
        await self._process_slot_updates(self.active_slot, chunk_end=True, final=True)
        # 규칙 4로 만들어진 커밋을 여기서 반드시 배출해야 한다. dot 커밋을 지연 전송하는
        # 서브클래스(평가 서버)는 generate 완료 시점에 큐를 비우는데, 규칙 4는 finish
        # 시점에 커밋하므로 뒤따르는 generate가 없어 비울 사람이 없다. 그대로 두면
        # 확정·번역까지 다 해놓고 final 메시지를 안 보내 전사가 통째로 빈다
        # (실측: 6128-63241-0006 등 6건이 빈 전사).
        await self._drain_deferred_commits()
        await self.flush_uncommitted(force=True, reason="finish", slot_key=self.active_slot)
        # 스트림이 끝났으니 클라이언트에 남은 미확정 말풍선을 지운다.
        await self._send_partial_clear()

    async def _send_partial_clear(self) -> None:
        """미확정 말풍선을 비우라는 신호(빈 문자열 partial)를 무조건 보낸다."""
        if self._last_partial_text == "":
            return
        self._last_partial_text = ""
        self._last_partial_time = time.perf_counter()
        self._partial_seq += 1
        await self.send_message("partial", text="", language="", seq=self._partial_seq)

    # ── 서브클래스 훅 ──────────────────────────────────────────────────────────

    async def _drain_deferred_commits(self):
        """지연 전송 큐가 있으면 비운다. base는 dot 커밋을 즉시 보내므로 no-op."""
        return

    async def _translate(
        self, text: str, target_lang: str, audio_end_sec: Optional[float] = None  # noqa: ARG002
    ) -> tuple[str, str, dict]:
        """번역 훅. 서브클래스에서 오버라이드해 타이밍 등 추가 데이터 수집 가능.

        Returns:
            (translation, detected_lang, extra)
            extra: 서브클래스가 _emit_final_payload 에 전달할 임의 데이터.
        """
        # 로컬 번역기는 소스 언어를 스스로 감지하지 않으므로 ASR 이 판정한 언어를
        # 같이 넘긴다. Google 경로에서는 이 인자가 무시된다.
        source_lang = lang_to_code(self._slot(self.active_slot).get("last_text_lang", ""))
        translation, detected_lang = await google_translate_async(
            self.http_session, text, target_lang, source_lang
        )
        return translation, detected_lang or source_lang, {}

    def _maybe_fix_direction(self, detected: str, used_target: str) -> Optional[str]:
        """양방향(non-auto) 모드에서 감지된 소스 언어가 번역 target과 같으면(= 같은 언어로
        번역된 no-op) 올바른 반대편 앱 언어를 반환한다. 수정 불필요 시 None.

        언어 전환 경계(예: 한국어 직후 짧은 영어)에서 스트림 단위 state.language가 아직
        안 넘어가 방향이 틀어진 경우를, 번역 후 신뢰 가능한 감지 결과로 자가교정한다.
        client_lang/client_target_lang/detected는 모두 언어 코드(예: 'en','ko').
        """
        if not detected:
            return None
        if self.client_lang_map:
            # 매핑이 있으면 방향은 감지 언어가 정한다. 목표가 다르면 그쪽으로 한 번 더.
            mapped = self.client_lang_map.get(detected)
            if mapped and mapped != used_target:
                return mapped
            return None
        if not self.client_lang or self.client_lang == "auto":
            return None
        if self.client_lang == self.client_target_lang:
            return None
        if detected != used_target:
            return None
        other = self.client_lang if used_target == self.client_target_lang else self.client_target_lang
        if not other or other == detected:
            return None
        return other

    async def _correct_and_translate(
        self, text: str, current_lang: str, audio_end_sec: float
    ) -> tuple[str, str, str, dict]:
        """교정 + 번역 통합 메서드. flush_uncommitted / _process_slot_updates 공통 경로.

        ASR 감지 언어 기준으로 번역 방향을 결정하고 번역 1회 호출.
        번역 결과의 감지 언어가 target과 같아 무의미한 경우(en→en 등) 반대 방향으로 1회 재시도.

        Returns:
            (corrected_text, translation, detected_lang_code, extra)
        """
        src_code = lang_to_code(current_lang) if current_lang else ""

        # ASR이 언어 감지 실패 시 GPT로 텍스트 기반 언어 감지
        if not src_code and self.gpt_translator and self.client_lang and self.client_lang != "auto":
            src_code = await self.gpt_translator.detect_language(text)

        # 번역 방향 결정: ASR 감지 언어 기준, 1회 번역
        # 언어별 목표(langMap)가 지정돼 있으면 그게 먼저다. 항목이 없으면 쌍 규칙으로 돈다.
        mapped_target = self.client_lang_map.get(src_code) if src_code else ""
        if mapped_target:
            target = mapped_target
        elif not self.client_lang or self.client_lang == "auto":
            target = self.client_target_lang        # 방향 모름 → targetLang으로
        elif src_code == self.client_lang:
            target = self.client_target_lang        # 내 언어 감지 → 상대 언어로
        elif src_code:
            target = self.client_lang               # 상대 언어 감지 → 내 언어로
        else:
            target = self.client_target_lang        # ASR 감지 실패 → targetLang으로

        if self.gpt_translator and self._committed_utterance_count > 0:
            if self._segment_history and src_code:
                ctx_pairs = [(o, t) for o, t, l in self._segment_history if l == src_code]
            else:
                ctx_pairs = [(o, t) for o, t, _ in self._segment_history]
            corrected, translation, gpt_detected = await self.gpt_translator.correct_and_translate(
                text, current_lang, target,
                context=ctx_pairs if ctx_pairs else None,
            )
            # 방향 자가교정: 감지 소스가 번역 target과 같으면(같은언어 no-op) 반대 앱 언어로 1회 재번역
            fixed_target = self._maybe_fix_direction(gpt_detected, target)
            if fixed_target:
                corrected, translation, gpt_detected = await self.gpt_translator.correct_and_translate(
                    text, current_lang, fixed_target,
                    context=ctx_pairs if ctx_pairs else None,
                )
            return corrected, translation, src_code or gpt_detected, {}

        # Google Translate 경로
        if self.corrector:
            text = await self.corrector.correct_text(text, current_lang)
        if self.config.google_context and self._segment_history:
            context_originals = [orig for orig, _, l in self._segment_history if not src_code or l == src_code]
            translation, detected_lang = await google_translate_with_context_async(
                self.http_session, text, target, context_originals
            )
            extra = {}
        else:
            translation, detected_lang, extra = await self._translate(text, target, audio_end_sec)
        # 방향 자가교정: 감지 소스가 번역 target과 같으면 반대 앱 언어로 재번역
        fixed_target = self._maybe_fix_direction(detected_lang or src_code, target)
        if fixed_target:
            translation, detected_lang, extra = await self._translate(text, fixed_target, audio_end_sec)
        effective = detected_lang or src_code
        return text, translation, effective, extra

    def _get_flush_audio_end_sec(self) -> float:
        """flush 시 사용할 오디오 종료 시각(초). 서브클래스에서 오버라이드 가능."""
        return self.current_time

    async def _retry_vad_short_utterance(self, slot_key: str) -> None:
        """finish_streaming이 빈 결과를 반환했을 때, VAD 구간 기준으로 오디오를 트림하여 재시도.

        앞: VAD speech_start - 200ms 패딩
        뒤: VAD 종료 감지 시점 - (VAD_MIN_SILENCE_MS - 100ms 여유)

        뒤쪽 여유를 상수에서 뽑는 이유: 종전에는 0.7 초가 박혀 있었는데 그건
        VAD_MIN_SILENCE_MS 가 800 일 때의 값이다. 침묵 기준을 400 으로 내리면
        아직 안 지난 300ms 를 더 잘라내 발화 끝이 날아간다.
        """
        slot = self._slot(slot_key)
        state = slot["state"]
        full_audio = state.audio_accum  # finish_streaming 완료 후 전체 누적 오디오
        if full_audio is None or full_audio.shape[0] == 0:
            return

        slot_anchor_samples = int(slot["audio_anchor_sec"] * SAMPLING_RATE)
        speech_start = self.vad_last_speech_start_sample - slot_anchor_samples - int(0.2 * SAMPLING_RATE)
        speech_start = max(0, speech_start)
        tail_trim = max(0.0, (VAD_MIN_SILENCE_MS - 100) / 1000.0)
        speech_end = full_audio.shape[0] - int(tail_trim * SAMPLING_RATE)

        if speech_end <= speech_start:
            self.log.info(f"[VAD-RETRY] slot={slot_key} skip: empty range start={speech_start} end={speech_end}")
            return

        speech_audio = full_audio[speech_start:speech_end]
        if speech_audio.shape[0] == 0:
            return

        self.log.info(
            f"[VAD-RETRY] slot={slot_key} "
            f"speech_start_sample={self.vad_last_speech_start_sample} "
            f"trim_from={speech_start} ({speech_start / SAMPLING_RATE:.2f}s) "
            f"trim_to={speech_end} ({speech_end / SAMPLING_RATE:.2f}s) "
            f"audio_len={speech_audio.shape[0]} ({speech_audio.shape[0] / SAMPLING_RATE:.2f}s)"
        )

        fresh_state = self.asr.init_streaming_state(
            unfixed_chunk_num=self.config.unfixed_chunk_num,
            unfixed_token_num=self.config.unfixed_token_num,
            chunk_size_sec=self.config.chunk_size_sec,
            allowed_languages=state.allowed_languages,
        )
        # buffer에 발화 오디오를 넣어 finish_streaming_transcribe가 처리하도록 함
        fresh_state.buffer = speech_audio.copy()

        async with self.asr_lock:
            lora_request = self._get_lora_request(fresh_state)
            await self.asr.finish_streaming_transcribe(fresh_state, lora_request=lora_request)

        retry_text = (fresh_state.text or "").strip()
        self.log.info(f"[VAD-RETRY-RESULT] slot={slot_key} text={retry_text!r}")

        if retry_text:
            slot["state"] = fresh_state

    async def _on_vad_commit(self, audio_end_sec: float) -> None:  # noqa: ARG002
        """VAD 발화 종료 커밋 직전에 호출되는 훅. 서브클래스에서 오버라이드 가능."""
        pass

    async def _on_vad_done(self, slot_key: str, tail_samples: int = 0) -> None:  # noqa: ARG002
        """VAD 발화 종료 플러시 완료 후 호출되는 훅. 서브클래스에서 오버라이드 가능.

        tail_samples: VAD 커트 이후 같은 청크에 남아 있는 샘플 수.
        0이면 서버에 처리할 잔여 오디오가 없음을 의미한다.
        """
        pass

    async def _emit_final_payload(
        self,
        *,
        slot_key: str,  # noqa: ARG002
        original: str,
        translation: str,
        language: str,
        reason: str,
        audio_end_sec: float,
        extra: Optional[dict] = None,  # noqa: ARG002
    ) -> None:
        """최종 세그먼트 전송 훅. 서브클래스에서 오버라이드해 FSL 메타데이터 등 추가 가능."""
        await self.send_message(
            "final",
            start=format_time(self.segment_start_time),
            end=format_time(audio_end_sec),
            original=original,
            translation=translation,
            language=language,
            commitReason=reason,
        )
        if self.session_logger:
            await self.session_logger.append(
                time=datetime.now(timezone.utc).isoformat(),
                text=original,
                translation=translation,
            )
        self._committed_utterance_count += 1
        if original and translation and (self.gpt_translator or self.config.google_context):
            self._segment_history.append((original, translation, language))

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            self.log.info(f"New connection from {remote_addr}")
            await self.send_message(
                "hello",
                message="Qwen3-ASR Streaming Server",
                modelPath=self.config.model_path,
                modelId=self.config.model_path,
                serverConfig={
                    "model": self.config.model_path,
                    "chunk_size_sec": self.config.chunk_size_sec,
                    "enforce_eager": self.config.enforce_eager,
                    "enable_gpt_translation": self.config.enable_gpt_translation,
                    "translation_model": self.config.translation_model,
                    "context_window": self.config.context_window,
                    # GPTTranslator는 교정+번역을 단일 호출로 처리하므로
                    # enable_gpt_translation이 켜지면 correction도 사실상 활성화됨
                    "enable_correction": self.config.enable_correction or self.config.enable_gpt_translation,
                    "correction_model": self.config.correction_model,
                },
            )
            timeout = aiohttp.ClientTimeout(total=3)
            self.http_session = aiohttp.ClientSession(timeout=timeout)

            async for message in self.websocket:
                if isinstance(message, bytes):
                    if self.running:
                        await self.process_audio_chunk(message)
                else:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "start":
                            if self._get_streaming_id is not None:
                                streaming_id = await self._get_streaming_id()
                                self.log.extra["cid"] = streaming_id
                            self.session_logger = SessionLogger(client_id=self.log.extra["cid"])
                            if self.config.record_audio:
                                if self.recorder is not None:
                                    self.recorder.close()
                                self.recorder = AudioRecorder(
                                    self.session_logger.path,
                                    client_id=self.log.extra["cid"],
                                )
                            self.client_lang = data.get("lang", "auto")
                            self.client_target_lang = data.get("targetLang", "")
                            self.client_lang_map = parse_lang_map(data.get("langMap"))
                            self.log.info(
                                f"Received start: lang={self.client_lang}, "
                                f"targetLang={self.client_target_lang}, "
                                f"langMap={self.client_lang_map or '-'}"
                            )

                            self.init_streaming_state()
                            self.running = True
                            await self.send_message(
                                "ready", message="Ready to receive audio"
                            )

                        elif msg_type == "lang_hint":
                            await self._apply_lang_hint(
                                data.get("lang"), data.get("fromSec"),
                                force=bool(data.get("force")))
                        elif msg_type == "config":
                            # 시연 중 번역 방향을 바꾼다. 스트림은 그대로 둔다 —
                            # 다음에 확정되는 문장부터 새 설정으로 번역된다.
                            if "langMap" in data:
                                self.client_lang_map = parse_lang_map(data.get("langMap"))
                            if data.get("lang"):
                                self.client_lang = data["lang"]
                            if data.get("targetLang"):
                                self.client_target_lang = data["targetLang"]
                            self.log.info(
                                f"Received config: lang={self.client_lang}, "
                                f"targetLang={self.client_target_lang}, "
                                f"langMap={self.client_lang_map or '-'}"
                            )
                            await self.send_message(
                                "config_ok",
                                lang=self.client_lang,
                                targetLang=self.client_target_lang,
                                langMap=self.client_lang_map,
                            )

                        elif msg_type in ("stop", "finish"):
                            self.log.info(f"Received {msg_type} command")
                            self.running = False
                            await self.finish_streaming()

                            if msg_type == "stop":
                                break

                            # finish → 상태 리셋 후 재시작
                            self.init_streaming_state()
                            self.running = True

                        elif msg_type == "pair_host":
                            room_id = (data.get("roomId") or "").strip()
                            my_lang = (data.get("myLang") or "").strip()
                            target_lang = (data.get("targetLang") or "").strip()
                            mode = (data.get("mode") or "mode-2").strip()
                            if not room_id or not my_lang or not target_lang:
                                await self.send_message(
                                    "pair_error",
                                    roomId=room_id,
                                    message="invalid_pair_host_payload",
                                )
                            else:
                                await self.pairing_hub.register_host(
                                    self.websocket, room_id, my_lang, target_lang, mode
                                )

                        elif msg_type == "pair_join":
                            room_id = (data.get("roomId") or "").strip()
                            guest_my_lang = (data.get("myLang") or "").strip()
                            if not room_id:
                                await self.send_message(
                                    "pair_error",
                                    roomId=room_id,
                                    message="missing_room_id",
                                )
                            elif not guest_my_lang:
                                await self.send_message(
                                    "pair_error",
                                    roomId=room_id,
                                    message="missing_guest_my_lang",
                                )
                            else:
                                await self.pairing_hub.join_room(
                                    self.websocket, room_id, guest_my_lang
                                )

                        elif msg_type == "log":
                            time = data.get("time", "")
                            text = data.get("text", "")
                            translation = data.get("translation", "")
                            if self.session_logger and text:
                                await self.session_logger.append(time, text, translation)

                        elif msg_type == "tts_log":
                            text = data.get("text", "")
                            lang = data.get("lang", "")
                            start = data.get("start", "")
                            end = data.get("end", "")
                            if self.session_logger and text:
                                await self.session_logger.append_tts(
                                    text=text,
                                    lang=lang,
                                    start=start,
                                    end=end,
                                )

                        elif msg_type == "pair_leave":
                            await self.pairing_hub.leave(self.websocket)

                    except json.JSONDecodeError:
                        self.log.warning(f"Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            self.log.info("Connection closed by client")
        except Exception as e:
            self.log.error(f"Error in handler: {e}")
            traceback.print_exc()
        finally:
            was_running = self.running
            self.running = False
            if was_running:
                await self.finish_streaming()
            if self.http_session is not None:
                with contextlib.suppress(Exception):
                    await self.http_session.close()
                self.http_session = None
            if self.recorder is not None:
                self.recorder.close()
                self.recorder = None
            await self.pairing_hub.leave(self.websocket)
            self.log.info("Connection closed")


class Qwen3ASRStreamingServer:
    """Qwen3-ASR 스트리밍 서버"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.IDLE_SHUTDOWN_SEC = config.idle_shutdown_sec
        self.asr = None
        self.lora_request_en = None
        self.lora_request_ko = None
        self.vad_model_bytes: Optional[bytes] = None
        self.corrector: Optional["GPTCorrector"] = None
        self.gpt_translator: Optional["GPTTranslator"] = None
        self.pairing_hub = PairingHub()
        self.idle_task = None
        self.active_connections = 0
        self._streaming_counter = 0
        self.connection_lock = asyncio.Lock()

    async def next_streaming_id(self) -> int:
        """실제 스트리밍 시작 시에만 호출 — C 번호 할당"""
        async with self.connection_lock:
            self._streaming_counter += 1
            return self._streaming_counter

    def _resolve_adapter_path(self, rel_path: str) -> Optional[str]:
        """상대경로를 examples/ 디렉토리 기준으로 절대경로로 변환. 존재하지 않으면 None."""
        if self.config.no_lora:
            return None
        if not rel_path:
            return None
        abs_path = os.path.abspath(os.path.join(_SERVER_DIR, rel_path))
        if not os.path.isdir(abs_path):
            logger.warning(f"LoRA adapter path not found, skipping: {abs_path}")
            return None
        return abs_path

    def init_model(self):
        """ASR 모델 초기화 (동기 — 이벤트 루프 시작 전 호출)"""
        # LoRA 어댑터 경로 확인
        adapter_en_path = self._resolve_adapter_path(self.config.adapter_en)
        adapter_ko_path = self._resolve_adapter_path(self.config.adapter_ko)
        use_lora = bool(adapter_en_path or adapter_ko_path)

        # VAD 모델을 한 번만 로드해 bytes로 보관 — 클라이언트마다 이 bytes로 독립 인스턴스 생성
        if self.config.no_vad:
            logger.info("VAD disabled via --no-vad flag")
        elif _SILERO_VAD_AVAILABLE:
            try:
                _buf = io.BytesIO()
                torch.jit.save(load_silero_vad(), _buf)
                self.vad_model_bytes = _buf.getvalue()
                logger.info("Silero VAD model loaded (server-level)")
            except Exception as e:
                logger.warning(f"Silero VAD model load failed: {e}")

        logger.info(f"Loading model: {self.config.model_path}")
        self.asr = Qwen3ASRModel.LLM(
            model=self.config.model_path,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_new_tokens=self.config.max_new_tokens,
            max_model_len=4096,
            enable_lora=use_lora,
            max_lora_rank=self.config.max_lora_rank if use_lora else 16,
            enforce_eager=self.config.enforce_eager,
        )
        if self.config.beam_size > 1:
            from vllm import SamplingParams
            try:
                self.asr.sampling_params = SamplingParams(
                    use_beam_search=True,
                    best_of=self.config.beam_size,
                    temperature=0.0,
                    max_tokens=self.config.max_new_tokens,
                    skip_special_tokens=True,
                )
                logger.info(f"Beam search enabled: beam_size={self.config.beam_size}")
            except TypeError:
                logger.warning(
                    f"이 vLLM 버전은 use_beam_search를 지원하지 않습니다. "
                    f"Greedy decoding으로 fallback합니다. (beam_size={self.config.beam_size} 무시)"
                )
                self.asr.sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=self.config.max_new_tokens,
                    skip_special_tokens=True,
                )
                logger.info("Greedy decoding (no repetition_penalty)")
        else:
            from vllm import SamplingParams
            self.asr.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.config.max_new_tokens,
                skip_special_tokens=True,
            )
            logger.info("Greedy decoding (no repetition_penalty)")

        # LoRA 어댑터 등록
        if use_lora:
            try:
                from vllm.lora.request import LoRARequest
                if adapter_en_path:
                    self.lora_request_en = LoRARequest("en", 1, adapter_en_path)
                    logger.info(f"English LoRA adapter loaded: {adapter_en_path}")
                if adapter_ko_path:
                    self.lora_request_ko = LoRARequest("ko", 2, adapter_ko_path)
                    logger.info(f"Korean LoRA adapter loaded: {adapter_ko_path}")
            except ImportError:
                logger.warning("vllm.lora.request를 import할 수 없습니다. LoRA 어댑터를 사용하지 않습니다.")

        logger.info("Model loaded successfully")

        if self.config.enable_gpt_translation:
            if not _CORRECTOR_AVAILABLE:
                logger.warning("GPT translator requested but core.translator import failed — falling back to Google Translate")
            elif not (self.config.api_key or os.environ.get("OPENAI_API_KEY")):
                logger.warning("GPT translator requested but OPENAI_API_KEY not set — falling back to Google Translate")
            else:
                self.gpt_translator = GPTTranslator(
                    model=self.config.translation_model,
                    api_key=self.config.api_key,
                    max_context=self.config.context_window,
                )
                logger.info(f"GPT translator enabled (model={self.config.translation_model})")
        elif self.config.enable_correction:
            if not _CORRECTOR_AVAILABLE:
                logger.warning("GPT corrector requested but core.translator import failed — correction disabled")
            elif not (self.config.api_key or os.environ.get("OPENAI_API_KEY")):
                logger.warning("GPT corrector requested but OPENAI_API_KEY not set — correction disabled")
            else:
                self.corrector = GPTCorrector(model=self.config.correction_model, api_key=self.config.api_key)
                logger.info(f"GPT corrector enabled (model={self.config.correction_model})")

    async def handle_connection(self, websocket):
        """각 연결 처리"""
        async with self.connection_lock:
            self.active_connections += 1
            if self.idle_task and not self.idle_task.done():
                self.idle_task.cancel()
            logger.info(f"Client connected (active={self.active_connections})")

        try:
            handler = Qwen3ASRStreamingHandler(
                websocket, self.asr, self.config, self.pairing_hub,
                get_streaming_id=self.next_streaming_id,
                lora_request_en=self.lora_request_en,
                lora_request_ko=self.lora_request_ko,
                vad_model_bytes=self.vad_model_bytes,
                corrector=self.corrector,
                gpt_translator=self.gpt_translator,
            )
            await handler.handle()
        finally:
            async with self.connection_lock:
                self.active_connections -= 1
                logger.info(f"Client disconnected (active={self.active_connections})")
                if self.active_connections == 0:
                    self._restart_idle_timer()

    async def _handle_http_request(self, connection, request):
        if "Upgrade" not in request.headers:
            return connection.respond(http.HTTPStatus.OK, "OK\n")

    async def start(self):
        logger.info(f"Starting WebSocket server on ws://{self.config.host}:{self.config.port}")
        logger.info("Warming up model...")
        await warmup_streaming(self.asr)
        logger.info("Warmup complete")
        async with websockets.serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=self.config.close_timeout,
            max_size=10 * 1024 * 1024,
            process_request=self._handle_http_request,
        ):
            logger.info(f"Server listening on ws://{self.config.host}:{self.config.port}")
            self._restart_idle_timer()  # start idle timer so server shuts down if no client connects
            await asyncio.Future()  # run forever
        
        

    async def _idle_shutdown_loop(self):
        """접속자 0명이 일정 시간 지속되면 EC2 종료"""
        try:
            while True:
                await asyncio.sleep(self.IDLE_SHUTDOWN_SEC)
                async with self.connection_lock:
                    if self.active_connections == 0:
                        logger.info(
                            f"[idle-shutdown] {self.IDLE_SHUTDOWN_SEC}초간 "
                            f"접속자 없음 — EC2 종료"
                        )
                        subprocess.run(["sudo", "shutdown", "-h", "now"])
                        return
        except asyncio.CancelledError:
            return
        
    def _restart_idle_timer(self):
        if self.config.no_idle_shutdown:
            return
        if self.idle_task and not self.idle_task.done():
            self.idle_task.cancel()
        self.idle_task = asyncio.create_task(self._idle_shutdown_loop())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR Streaming WebSocket Server"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-ASR-1.7B",
        help="Model path or name",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.8,
        help="GPU memory utilization (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Max new tokens per chunk (32은 긴 발화 truncation 유발 → 128)",
    )
    parser.add_argument(
        "--vad-min-silence", type=int, default=VAD_MIN_SILENCE_MS,
        help=(
            "발화 종료 판정까지 필요한 침묵 길이(ms). 짧을수록 발화가 자주 "
            f"끊겨 언어 전환이 빨리 풀리지만 문장이 잘게 쪼개진다 (기본 {VAD_MIN_SILENCE_MS})"
        ),
    )
    parser.add_argument(
        "--chunk-size", type=float, default=2.0,
        help="Chunk size in seconds",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Server port",
    )
    parser.add_argument(
        "--local-translation", action="store_true",
        help="번역을 로컬 NLLB-200 모델로 처리한다. 외부 호출이 없어 gtx 429 에 안 걸린다.",
    )
    parser.add_argument(
        "--local-translation-model", type=str, default="facebook/nllb-200-distilled-600M",
        help="로컬 번역 모델 이름 또는 경로",
    )
    parser.add_argument(
        "--local-translation-device", type=str, default=None,
        help="로컬 번역 모델을 올릴 장치 (cuda / cpu). 미지정 시 자동",
    )
    parser.add_argument(
        "--local-translation-url", type=str, default=None,
        help=(
            "단독 번역 서버(STiTy-Mobile/demo-web/local_translation_server.py) 주소. 주면 이 "
            "프로세스에 번역 모델을 올리지 않고 HTTP 로 부른다 — ASR 서버를 "
            "여러 개 띄울 때 번역 모델이 복제되는 걸 막는다. 예: http://127.0.0.1:8770"
        ),
    )
    parser.add_argument(
        "--no-idle-shutdown", action="store_true",
        help="Disable idle shutdown (use this when running tests)",
    )
    parser.add_argument(
        "--idle-shutdown-sec", type=int, default=60,
        help="Seconds of no connections before server shuts down (default: 60)",
    )
    parser.add_argument(
        "--close-timeout", type=int, default=None,
        help="WebSocket close handshake timeout in seconds (default: websockets 기본값 10초). "
             "추론 중 연결 종료 시 timeout이 발생하면 늘려서 진단 가능 (예: 30)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=2,
        help="Beam search size (1=greedy, 2+=beam search, default: 2)",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Triton 컴파일 우회 (sm_121a 등 미지원 GPU에서 필요)",
    )
    parser.add_argument(
        "--lora", action="store_true",
        help="LoRA 어댑터를 사용 (기본값: 사용 안 함)",
    )
    parser.add_argument(
        "--adapter-en", type=str,
        default="../finetuning/finetuning-out-en-plus/checkpoint-420_vllm",
        help="영어 LoRA 어댑터 경로 (examples/ 기준 상대경로, 없으면 기본 모델 사용)",
    )
    parser.add_argument(
        "--adapter-ko", type=str,
        default="../finetuning/finetuning-out-ko-plus/checkpoint-300_vllm",
        help="한국어 LoRA 어댑터 경로 (examples/ 기준 상대경로, 없으면 기본 모델 사용)",
    )
    parser.add_argument(
        "--max-lora-rank", type=int, default=128,
        help="LoRA 최대 rank (학습 시 사용한 rank와 일치해야 함, 기본값: 128)",
    )
    parser.add_argument(
        "--enable-dot-commit", dest="enable_dot_commit", action="store_true", default=None,
        help="온점/느낌표/물음표(dot) 기반 seg commit 활성화 (기본값: 비활성화, 베이스라인 모델용)",
    )
    parser.add_argument(
        "--disable-dot-commit", dest="enable_dot_commit", action="store_false",
        help="Disable dot-based commit even for baseline models",
    )
    parser.add_argument(
        "--dot-commit-confirm", dest="dot_commit_confirm", action="store_true", default=None,
        help="dot을 감지 즉시 커밋하지 않고 확정된 뒤에만 커밋 "
             "(문맥 확정: 마침표 뒤 토큰 > unfixed_token_num / 합의 확정: 직전 청크에도 있던 경계 / "
             "정체 확정: 가설이 안 자람). 미지정 시 enable_dot_commit을 따라간다 "
             "— 즉 dot commit(모드3)이면 자동으로 켜진다",
    )
    parser.add_argument(
        "--no-dot-commit-confirm", dest="dot_commit_confirm", action="store_false",
        help="dot commit이 켜져 있어도 확정 게이트는 끈다 (감지 즉시 커밋하던 예전 동작)",
    )
    parser.add_argument(
        "--no-rep-dedup", dest="rep_dedup", action="store_false", default=True,
        help="같은 문장 연속 반복 억제(rep-dedup)를 끈다. 반복 루프에 빠지는 모델에는 "
             "필요하지만, 루프가 없는 모델에선 낭독체의 실제 반복('Ha! Ha! Ha!')을 깎는다. "
             "(기본: 켜짐)",
    )
    parser.add_argument(
        "--dot-commit-stall-chunks", type=int, default=1,
        help="정체 확정(규칙 3): 오디오는 누적되는데 미커밋 가설 토큰 수가 이 청크 수만큼 "
             "연속으로 그대로면 발화 종료로 보고 커밋. 0이면 비활성화 (기본 1)",
    )
    parser.add_argument(
        "--always-commit", action="store_true",
        help="SEG/dot 트리거 없이 매 청크 디코딩 결과를 그대로 커밋 (모드2 테스트용, "
             "enable_dot_commit보다 우선)",
    )
    parser.add_argument(
        "--no-restrict-languages", action="store_true",
        help="앱 설정 두 언어 외 언어 차단 비활성화 (기본값: 활성화)",
    )
    parser.add_argument(
        "--correction", action="store_true",
        help="GPT LLM 후처리 활성화 (기본값: 비활성화)",
    )
    parser.add_argument(
        "--correction-model", type=str, default="gpt-5.4-mini",
        help="후처리에 사용할 GPT 모델명 (기본값: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--gpt-translation", action="store_true",
        help="GPT로 교정+번역을 단일 호출 처리 (Google Translate + GPTCorrector 대체)",
    )
    parser.add_argument(
        "--translation-model", type=str, default="gpt-5.4-mini",
        help="GPT 번역에 사용할 모델명 (기본값: gpt-5.4-mini, --gpt-translation 활성화 시 사용)",
    )
    parser.add_argument(
        "--context-window", type=int, default=5,
        help="번역 컨텍스트 문장 수 (기본값: 5, --gpt-translation / --google-context 공유)",
    )
    parser.add_argument(
        "--google-api-key", type=str, default=None,
        help="Cloud Translation v2 API 키. 미지정 시 GOOGLE_TRANSLATE_API_KEY 환경변수를 "
             "쓰고, 그것도 없으면 무료 gtx 엔드포인트로 떨어진다(429 로 막힐 수 있음)",
    )
    parser.add_argument(
        "--google-context", action="store_true",
        help="Google Translate에 이전 N개 원문을 함께 전송해 문맥 인식 개선 (--context-window 문장 수 사용)",
    )
    parser.add_argument(
        "--no-vad", action="store_true",
        help="Silero VAD 비활성화 — SEG/finish 커밋만 사용 (VAD 없이 동작)",
    )
    parser.add_argument(
        "--log-json", action="store_true",
        help="로그를 JSON 형식으로 출력",
    )
    parser.add_argument(
        "--record-audio", action="store_true",
        help="수신 오디오를 세션별 WAV(Qwen3-ASR/logs/asr_audio/session_{ts}.wav)로 저장 — "
             "세션 로그와 동일 타임스탬프로 매칭됨 (기본값: 비활성화)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="로그 파일 경로 (미지정 시 기본 경로 사용)",
    )
    args = parser.parse_args()
    if args.enable_dot_commit is None:
        args.enable_dot_commit = _infer_dot_commit_default(args.model)
    if args.dot_commit_confirm is None:
        # 확정 게이트는 dot commit 전용 로직이므로 dot commit(모드3)이면 기본으로 켠다.
        # 게이트 없는 dot commit은 프론티어 마침표를 그대로 커밋해 매 청크 문장 조각이
        # 나가는 동작이라, 이제 와서 그걸 기본값으로 둘 이유가 없다.
        # --no-dot-commit-confirm으로 예전 동작 복원 가능.
        args.dot_commit_confirm = bool(args.enable_dot_commit)
    return args


def main():
    args = parse_args()
    _configure_logging(use_json=args.log_json, log_file=args.log_file)

    # VADIterator 생성과 꼬리 트림이 모두 이 전역을 읽으므로 여기서 한 번 덮는다.
    if args.vad_min_silence != VAD_MIN_SILENCE_MS:
        globals()["VAD_MIN_SILENCE_MS"] = args.vad_min_silence
        logger.info(f"VAD min silence: {args.vad_min_silence}ms")
    set_google_translate_api_key(
        args.google_api_key,
        local_translation=args.local_translation or bool(args.local_translation_url),
    )

    if args.local_translation_url:
        # 원격이 먼저다. 주소를 준 건 "이 프로세스에 모델을 올리지 말라"는 뜻이므로
        # --local-translation 이 함께 켜져 있어도 모델을 올리지 않는다.
        from core.translator.local_translator import RemoteTranslator

        set_local_translator(RemoteTranslator(args.local_translation_url))
        logger.info(f"Remote translator enabled ({args.local_translation_url})")
    elif args.local_translation:
        try:
            from core.translator.local_translator import make_translator
            # 모델 이름으로 백엔드를 고른다 — madlad 가 들어가면 MADLAD, 아니면 NLLB.
            _local = make_translator(
                model_name=args.local_translation_model,
                device=args.local_translation_device,
            )
            _local.load()   # 첫 번역에서 로딩 지연이 나지 않게 미리 올린다
            set_local_translator(_local)
            logger.info(f"Local translator enabled ({args.local_translation_model})")
        except Exception as e:
            logger.warning(f"Local translator 초기화 실패 — Google Translate 로 폴백: {e!r}")
            # 폴백이 실제로 걸렸으니 이제 Google 경로가 진짜 경로다. 위에서 건너뛴
            # 키 유무 안내를 여기서 다시 남긴다.
            set_google_translate_api_key(None, local_translation=False)

    config = StreamingConfig(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        chunk_size_sec=args.chunk_size,
        host=args.host,
        port=args.port,
        no_idle_shutdown=args.no_idle_shutdown,
        idle_shutdown_sec=args.idle_shutdown_sec,
        close_timeout=args.close_timeout,
        beam_size=args.beam_size,
        adapter_en=args.adapter_en,
        adapter_ko=args.adapter_ko,
        no_lora=not args.lora,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=args.enforce_eager,
        no_vad=args.no_vad,
        enable_dot_commit=args.enable_dot_commit,
        dot_commit_confirm=args.dot_commit_confirm,
        dot_commit_stall_chunks=args.dot_commit_stall_chunks,
        rep_dedup=args.rep_dedup,
        always_commit=args.always_commit,
        restrict_languages=not args.no_restrict_languages,
        enable_correction=args.correction,
        correction_model=args.correction_model,
        api_key=args.api_key,
        enable_gpt_translation=args.gpt_translation,
        translation_model=args.translation_model,
        context_window=args.context_window,
        google_context=args.google_context,
        record_audio=args.record_audio,
    )

    server = Qwen3ASRStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
