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
    4. Receive JSON: {"type": "partial", ...} or {"type": "final", ...}
    5. Send JSON: {"type": "stop"} or {"type": "finish"} to end session
"""

import argparse
import asyncio
import contextlib
import http
import json
import logging
import os
import re
import time
import traceback
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

try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from core.llm_corrector import GPTCorrector
    from core.correct_and_trans import GPTTranslator
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
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs/asr_server.log")


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


def _configure_logging(use_json: bool = False) -> None:
    fmt: logging.Formatter = (
        _JsonFormatter() if use_json
        else logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE),
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
# VADIterator 설정
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 800       # 발화 종료 판정까지 필요한 침묵 길이
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
    max_new_tokens: int = 32

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

    # vLLM 컴파일 설정
    enforce_eager: bool = False  # True면 Triton 컴파일 우회 (sm_121a 등 미지원 GPU)

    # Commit 방식 설정
    enable_dot_commit: bool = False  # True면 온점/느낌표/물음표(dot) 기반 seg commit 활성화


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


class SessionLogger:
    """앱에서 수신한 로그를 세션별 JSON 파일로 저장"""

    def __init__(self, client_id: int = 0, logs_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs/asr_logs")):
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


async def google_translate_async(
    session: aiohttp.ClientSession, text: str, target_lang: str
) -> tuple[str, str]:
    """Async Google Translate call.
    Returns: (translated_text, detected_source_lang_code)
    """
    if not text.strip() or not target_lang:
        return "", ""
    try:
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        async with session.get(
            "https://translate.googleapis.com/translate_a/single",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as resp:
            data = await resp.json(content_type=None)
            translated = "".join(item[0] for item in data[0] if item and item[0])
            detected_lang = data[2] if len(data) > 2 else ""
            return translated, detected_lang
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return "", ""


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

        # 타임스탬프 추적
        self.segment_start_time = 0.0
        self.current_time = 0.0

        self.asr_lock = asyncio.Lock()
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_logger: Optional[SessionLogger] = None
        self.corrector = corrector
        self.gpt_translator = gpt_translator
        self.use_correction = True  # overridden per-session by start message speed field
        # GPT 번역 컨텍스트: 최근 N 세그먼트의 (corrected_original, translation) 보관
        _ctx = gpt_translator.max_context if gpt_translator else 5
        self._segment_history: deque[tuple[str, str]] = deque(maxlen=_ctx)
        # 첫 번째 발화는 Google Translate, 두 번째부터 GPT로 전환
        self._committed_utterance_count: int = 0

        # generate() 루프 내 병렬 GPT 처리
        self._in_generate_loop: bool = False
        self._pending_gpt_tasks: list = []
        self._last_generate_end_time: float = 0.0  # generate() 완료 직후 perf_counter

        # Commit 방식 설정
        self.enable_dot_commit: bool = config.enable_dot_commit

        # VAD / stream alignment
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        self.active_slot = "A"
        self.standby_slot = "B"
        self.stream_slots: dict[str, dict] = {}
        self.vad_last_speech_start_sample: int = 0  # 마지막 VAD speech_start 글로벌 샘플 위치

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
        if self.config.restrict_languages and self.client_lang and self.client_lang != "auto":
            langs = []
            src_name = lang_code_to_name(self.client_lang)
            if src_name:
                langs.append(src_name)
            if self.client_target_lang:
                tgt_name = lang_code_to_name(self.client_target_lang)
                if tgt_name and tgt_name not in langs:
                    langs.append(tgt_name)
            if langs:
                allowed_languages = langs

        state = self.asr.init_streaming_state(
            unfixed_chunk_num=self.config.unfixed_chunk_num,
            unfixed_token_num=self.config.unfixed_token_num,
            chunk_size_sec=self.config.chunk_size_sec,
            allowed_languages=allowed_languages,
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
        }

    def _reset_stream_slot(self, slot_key: str, seed_text: str = "", context: str = ""):
        self.stream_slots[slot_key] = self._new_stream_slot(seed_text=seed_text, context=context)

    def _slot(self, slot_key: Optional[str] = None) -> dict:
        key = slot_key or self.active_slot
        return self.stream_slots[key]

    @staticmethod
    def _committed_cursor(text: str, committed_display: str,
                          committed_seg_count: int = 0) -> int:
        """committed 영역 끝의 커서 위치(int)를 반환.

        3-way fallback으로 커서를 찾는다. 모든 fallback 실패 시 -1(sentinel) 반환.

        1차: committed_display(SEG 제거 기준) prefix 매칭
        2차(fallback): committed_seg_count 번째 <SEG> 이후 위치
        3차(fallback): 후행 구두점 제거 후 prefix 매칭
          — dot commit 후 모델이 "Okay." → "Okay, this is..."로 revision할 때 커버.
        """
        seg_tag = "<SEG>"
        seg_len = len(seg_tag)
        _punct = '.,!?;:。？！'

        # ── 1차: display prefix 매칭 (구두점 동치 포함) ──────────────────
        # 모델이 온점을 쉼표로(또는 반대로) revision해도 같은 위치로 인식
        # text_no_seg는 \s+ 정규화: SEG 제거 후 남는 이중 공백을 committed_display와 동일하게 맞춤
        _norm_punct = lambda s: re.sub(r'[.,!?;:。？！、，]', '.', s)
        if committed_display:
            text_no_seg = re.sub(r'\s+', ' ', text.replace(seg_tag, "")).strip()
            if text_no_seg.startswith(committed_display) or \
                    _norm_punct(text_no_seg).startswith(_norm_punct(committed_display)):
                pos, disp_pos, target = 0, 0, len(committed_display)
                _prev_space = False
                while pos < len(text) and disp_pos < target:
                    if text[pos:pos + seg_len] == seg_tag:
                        pos += seg_len
                        _prev_space = True  # SEG 경계는 공백으로 취급
                    elif text[pos] == ' ' and _prev_space:
                        pos += 1  # SEG 제거 후 남는 연속 공백 스킵 (committed_display와 공백 수 맞춤)
                    else:
                        _prev_space = (text[pos] == ' ')
                        disp_pos += 1
                        pos += 1
                return pos

        # ── 2차 fallback: SEG 카운트 기준 ───────────────────────────────
        # committed_seg_count=0이고 committed_display가 있으면 skip:
        # DOT commit은 SEG count를 올리지 않으므로 pos=0을 리턴하면
        # 3차 fallback이 실행되지 않아 uncommitted가 전체 텍스트가 됨
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
            if all_segs_found:
                return pos

        # ── 3차 fallback: 구두점 제거 후 prefix 매칭 ────────────────────
        if committed_display:
            stripped = committed_display.rstrip(_punct)
            if stripped:
                text_no_seg = text.replace(seg_tag, "")
                if text_no_seg.startswith(stripped):
                    end = len(stripped)
                    # 부분 단어 매칭 방지: 매칭 직후 문자가 알파이면 거짓 매칭
                    if end >= len(text_no_seg) or not text_no_seg[end].isalpha():
                        pos, disp_pos, target = 0, 0, len(stripped)
                        while pos < len(text) and disp_pos < target:
                            if text[pos:pos + seg_len] == seg_tag:
                                pos += seg_len
                                if pos < len(text) and text[pos] == ' ':
                                    pos += 1
                            else:
                                disp_pos += 1
                                pos += 1
                        # committed 구두점이 다른 구두점으로 revision된 경우 한 문자 건너뜀
                        if pos < len(text) and text[pos] in _punct:
                            pos += 1
                        return pos

        # ── 4차 fallback: 따옴표·콜론 제거 후 prefix 매칭 ─────────────────
        # 모델이 "说："要..." → "说要..." 처럼 인용 부호·콜론을 삭제/추가하는
        # revision 시에도 커서를 복원한다.
        if committed_display:
            _quote_colon = re.compile(r'[“”‘’"\'：:]+')
            committed_norm = _quote_colon.sub('', committed_display)
            text_no_seg_norm = _quote_colon.sub('', text.replace(seg_tag, ""))
            if committed_norm and text_no_seg_norm.startswith(committed_norm):
                target = len(committed_norm)
                disp_pos, pos = 0, 0
                while pos < len(text) and disp_pos < target:
                    if text[pos:pos + seg_len] == seg_tag:
                        pos += seg_len
                        if pos < len(text) and text[pos] == ' ':
                            pos += 1
                    elif _quote_colon.match(text[pos]):
                        pos += 1
                    else:
                        disp_pos += 1
                        pos += 1
                return pos

        return -1  # 모든 fallback 실패

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
            return text
        first, rest = text.split("<asr_text>", 1)
        if re.match(r'^\s*language\s+\w', first, re.IGNORECASE):
            text = rest
        return re.sub(r'\s*<asr_text>\s*', ' ', text).strip()

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

        async with self.asr_lock:
            lora_request = self._get_lora_request(slot["state"])

        # 생성 중 lock 미보유 — state.text는 await 없는 단순 대입이므로 asyncio 안전
        _accum_len_before = slot["state"].audio_accum.shape[0]
        self._in_generate_loop = True
        try:
            await self.asr.streaming_transcribe(chunk, slot["state"], lora_request=lora_request, on_seg=_on_seg)
        finally:
            self._in_generate_loop = False
            self._last_generate_end_time = time.perf_counter()
        # audio_accum이 늘었을 때만 실제 추론이 실행된 것
        if slot["state"].audio_accum.shape[0] > _accum_len_before:
            _decoded_text = self._strip_asr_text((slot["state"].text or "").strip())
            self.log.info(f"[TRANSCRIBE-DECODING] slot={slot_key} text={_decoded_text!r}")
        _committed_text_snapshot = await self._process_slot_updates(slot_key)
        if self._pending_gpt_tasks:
            await self._flush_pending_gpt_tasks()

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
                    self._reset_stream_slot(slot_key)
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
                    self._reset_stream_slot(slot_key)
                    new_slot = self._slot(slot_key)
                    new_slot["state"].audio_accum = carry_audio
                    if carry_lang:
                        new_slot["last_text_lang"] = carry_lang
                    if prev_committed:
                        new_slot["dot_switch_prev_committed"] = prev_committed
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

        async with self.asr_lock:
            self.asr_processed_cursor = self.sample_cursor

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

        # 1단계: 텍스트 스냅샷만 lock 안에서 읽기
        async with slot_flush_lock:
            async with self.asr_lock:
                state = slot["state"]
                current_text = self._strip_asr_text((state.text or "").strip() if state else "")
                current_lang = slot["last_text_lang"] or ""
            snapshot_committed_len = slot["committed_len"]
            snapshot_committed_display = slot["committed_display"]
            snapshot_committed_seg_count = slot["committed_seg_count"]
            uncommitted_raw = self._uncommitted_from(
                current_text, snapshot_committed_display, snapshot_committed_seg_count
            )
            uncommitted_display = re.sub(r'\s+', ' ', uncommitted_raw.replace("<SEG>", "")).strip() if uncommitted_raw is not None else ""
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
        uncommitted_display, translation, effective_detected, extra = await self._correct_and_translate(
            uncommitted_display, current_lang, audio_end_sec
        )
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
            self.log.info(
                f"[TRANS-VAD] slot={slot_key or self.active_slot} reason={commit_reason} lang={final_lang} "
                f"original='{uncommitted_display}' translation='{translation}'"
            )
            await self._emit_final_payload(
                slot_key=slot_key or self.active_slot,
                original=uncommitted_display,
                translation=translation,
                language=final_lang,
                reason=commit_reason,
                audio_end_sec=audio_end_sec,
                extra=extra,
            )
            slot["committed_len"] = len(current_text)
            slot["committed_prefix"] = current_text
            slot["committed_display"] = re.sub(r'\s+', ' ', current_text.replace("<SEG>", "")).strip()
            slot["committed_seg_count"] = current_text.count("<SEG>")
            slot["audio_anchor_sec"] = audio_end_sec

    async def _process_slot_updates(self, slot_key: str, force_reason: Optional[str] = None) -> Optional[str]:
        slot = self._slot(slot_key)
        state = slot["state"]
        current_text = self._strip_asr_text((state.text or "").strip())
        current_lang = state.language or ""
        if not current_text or current_text == slot["last_text"]:
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

        while True:
            # 우선순위 1: <SEG>
            # 우선순위 2: VAD (flush_uncommitted 에서 처리)
            # 우선순위 3: dot (enable_dot_commit=True일 때만 활성화)
            # dot 패턴: Mr./Mrs./Dr./St./Jr./Sr./vs./No. 등 약어 제외
            if self.enable_dot_commit:
                match = re.search(
                    r"(?:"
                    r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<!Jr)(?<!Sr)(?<!vs)(?<!No)\.\s+(?=\S)"
                    r"|[?!]\s+(?=\S)"
                    r"|[\u3002\uff1f\uff01](?=\S)"
                    r"|<SEG>"
                    r")",
                    remaining,
                )
            else:
                match = re.search(r"<SEG>", remaining)
            if not match:
                break
            matched_text = match.group()
            trigger = "seg" if "<SEG>" in matched_text else "dot"
            after = remaining[match.end():]
            # 커서 추적을 위해 raw sentence(<SEG> 포함) 사용
            sentence = remaining[:match.end()].strip()
            sentence_display_check = sentence.replace("<SEG>", "").strip()
            if sentence_display_check:
                if sentence_display_check == _last_extracted_display:
                    self.log.info(f"[COMMIT-SKIP] reason=rep-dedup slot={slot_key} text={sentence_display_check!r}")
                else:
                    if "seg_reset_last_committed" in slot:
                        seg_reset_last = slot.pop("seg_reset_last_committed")
                        first_word = sentence_display_check.split()[0] if sentence_display_check.split() else ""
                        _strip_p = lambda w: re.sub(r'[.,!?;:。？！]+$', '', w)
                        last_word = _strip_p(seg_reset_last.split()[-1]) if seg_reset_last.split() else ""
                        first_word = _strip_p(first_word)
                        if first_word and last_word and (first_word == last_word or last_word.endswith(first_word)):
                            self.log.info(f"[COMMIT-SKIP] reason=seg-boundary-dedup slot={slot_key} text={sentence_display_check!r}")
                            remaining = after
                            continue
                    prev_committed = slot.get("dot_switch_prev_committed", "")
                    if trigger == "dot" and prev_committed:
                        slot.pop("dot_switch_prev_committed", None)
                        _norm_c = re.sub(r'[.,!?;:。？！\s]+', '', prev_committed)
                        _norm_s = re.sub(r'[.,!?;:。？！\s]+', '', sentence_display_check)
                        if _norm_c.endswith(_norm_s):
                            self.log.info(f"[COMMIT-SKIP] reason=dot-suffix-dedup slot={slot_key} text={sentence_display_check!r}")
                            remaining = after
                            continue
                    sentences_to_commit.append((sentence, trigger))
                    _last_extracted_display = sentence_display_check
            remaining = after

        if not sentences_to_commit:
            return None

        # ── Phase 1: 커서 추적 + committed_len 즉시 확정 (GPT 호출 전) ─────────
        # raw ASR 텍스트 기준으로 커서를 확정하므로 GPT 교정 결과와 무관하게 안전.
        committed_items = []  # list of (sentence_display, trigger_reason)
        latest_text: Optional[str] = None  # flush_lock 안에서 읽은 스냅샷, 호출자에게 반환

        async with slot["flush_lock"]:
            async with self.asr_lock:
                latest_state = slot["state"]
                latest_text = (latest_state.text or "").strip() if latest_state else ""

            if force_reason == "vad":
                _PUNCT = '.,!?;:。？！'
                latest_ns = re.sub(r'\s+', ' ', latest_text.replace("<SEG>", "")).strip()
                cdisp = slot["committed_display"]
                pos = len(cdisp) if (cdisp and latest_ns.startswith(cdisp)) else 0
                for sentence_raw, trigger_reason in sentences_to_commit:
                    sentence_display = sentence_raw.replace("<SEG>", "").strip()
                    _audio_span = self.current_time - slot.get("audio_anchor_sec", 0.0)
                    if ' '.join(sentence_display.split()) in slot.get("committed_asr_set", set()):
                        self.log.info(
                            f"[COMMIT-SKIP] reason=cross-dedup slot={slot_key} span={_audio_span:.2f}s text={sentence_display!r}"
                        )
                        continue
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
                    committed_items.append((sentence_display, trigger_reason))
                if committed_items:
                    slot["committed_display"] = latest_ns[:pos].strip()
                    slot["committed_len"] = len(latest_text)
                    slot["audio_anchor_sec"] = self.current_time
                    slot["committed_asr_set"].update(' '.join(t.split()) for t, _ in committed_items)
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
                for sentence_raw, trigger_reason in sentences_to_commit:
                    sentence_display = sentence_raw.replace("<SEG>", "").strip()
                    _audio_span = self.current_time - slot.get("audio_anchor_sec", 0.0)
                    if ' '.join(sentence_display.split()) in slot.get("committed_asr_set", set()):
                        self.log.info(
                            f"[COMMIT-SKIP] reason=cross-dedup slot={slot_key} span={_audio_span:.2f}s text={sentence_display!r}"
                        )
                        continue
                    stripped_tail = tail.lstrip()
                    leading_ws = len(tail) - len(stripped_tail)
                    # <SEG> 토큰이 문장 사이에 있을 때 건너뜀
                    while stripped_tail.startswith("<SEG>"):
                        stripped_tail = stripped_tail[len("<SEG>"):].lstrip()
                        leading_ws = len(tail) - len(stripped_tail)
                    if not stripped_tail.startswith(sentence_raw):
                        break
                    cursor += leading_ws + len(sentence_raw)
                    tail = latest_text[cursor:]
                    committed_items.append((sentence_display, trigger_reason))
                if committed_items:
                    slot["committed_len"] = cursor
                    slot["committed_prefix"] = latest_text[:cursor]
                    slot["committed_display"] = re.sub(
                        r'\s+', ' ', slot["committed_prefix"].replace("<SEG>", "")
                    ).strip()
                    slot["committed_seg_count"] += sum(1 for _, tr in committed_items if tr == "seg")
                    slot["audio_anchor_sec"] = self.current_time
                    slot["committed_asr_set"].update(' '.join(t.split()) for t, _ in committed_items)

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
            self.log.info(f"[VAD-DECODING] slot={old_active} text={_pre_text!r}")
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
            await self.flush_uncommitted(force=True, reason="vad", slot_key=old_active)
            self._reset_stream_slot(self.standby_slot)
            await self._on_vad_done(old_active)

            seg_start = local_cut

        tail_chunk = chunk[seg_start:]
        if tail_chunk.size > 0:
            await self._asr_streaming_transcribe(tail_chunk, self.active_slot)

    async def finish_streaming(self):
        pass

    # ── 서브클래스 훅 ──────────────────────────────────────────────────────────

    async def _translate(
        self, text: str, target_lang: str, audio_end_sec: Optional[float] = None  # noqa: ARG002
    ) -> tuple[str, str, dict]:
        """번역 훅. 서브클래스에서 오버라이드해 타이밍 등 추가 데이터 수집 가능.

        Returns:
            (translation, detected_lang, extra)
            extra: 서브클래스가 _emit_final_payload 에 전달할 임의 데이터.
        """
        translation, detected_lang = await google_translate_async(
            self.http_session, text, target_lang
        )
        return translation, detected_lang, {}

    async def _correct_and_translate(
        self, text: str, current_lang: str, audio_end_sec: float
    ) -> tuple[str, str, str, dict]:
        """교정 + 번역 통합 메서드. flush_uncommitted / _process_slot_updates 공통 경로.

        gpt_translator 활성화 시: 단일 GPT 호출로 교정 + 번역 처리.
        비활성화 시: corrector(선택) → Google Translate → 언어 flip 처리.

        Returns:
            (corrected_text, translation, detected_lang_code, extra)
        """
        if self.gpt_translator and self._committed_utterance_count > 0:
            src_code = lang_to_code(current_lang) if current_lang else ""
            # 내 언어로 말하면 → 상대방 언어로, 그 외(상대방·제3자) → 내 언어로
            target = (
                self.client_target_lang
                if src_code and src_code == self.client_lang
                else self.client_lang
            )
            corrected, translation, gpt_detected = await self.gpt_translator.correct_and_translate(
                text, current_lang, target,
                context=list(self._segment_history) if self._segment_history else None,
            )
            return corrected, translation, gpt_detected or src_code, {}

        # Google Translate 경로 (첫 번째 발화 또는 gpt_translator 비활성)
        if self.corrector and self.use_correction:
            text = await self.corrector.correct_text(text, current_lang)
        translation, detected_lang, extra = await self._translate(
            text, self.client_target_lang, audio_end_sec
        )
        # 내 언어가 아니면(상대방·제3자 모두) 내 언어로 번역
        effective = detected_lang or lang_to_code(current_lang)
        if effective and effective != self.client_lang:
            translation, _, extra = await self._translate(
                text, self.client_lang, audio_end_sec
            )
        elif not effective and text and self.client_lang in ("ko", "ja", "zh"):
            # 언어 감지 완전 실패: Latin 문자 비율로 상대방 언어 추측
            # 한중일 클라이언트인데 텍스트가 대부분 Latin → 상대방이 영어권 발화
            alpha = [c for c in text if c.isalpha()]
            if alpha and sum(1 for c in alpha if ord(c) < 128) / len(alpha) > 0.8:
                translation, _, extra = await self._translate(
                    text, self.client_lang, audio_end_sec
                )
                effective = self.client_target_lang
                self.log.info(f"[translate-flip-latin] tl={self.client_lang} -> translation='{translation}'")
        return text, translation, effective, extra

    def _get_flush_audio_end_sec(self) -> float:
        """flush 시 사용할 오디오 종료 시각(초). 서브클래스에서 오버라이드 가능."""
        return self.current_time

    async def _retry_vad_short_utterance(self, slot_key: str) -> None:
        """finish_streaming이 빈 결과를 반환했을 때, VAD speech_start 기준으로 오디오를 트림하여 재시도.

        vad_last_speech_start_sample을 기준으로 슬롯의 audio_accum에서 발화 시작 지점 이후만
        잘라내어 fresh state로 재디코딩. 짧은 발화가 긴 침묵 앞에 묻히는 문제를 완화한다.
        """
        slot = self._slot(slot_key)
        state = slot["state"]
        full_audio = state.audio_accum  # finish_streaming 완료 후 전체 누적 오디오
        if full_audio is None or full_audio.shape[0] == 0:
            return

        slot_anchor_samples = int(slot["audio_anchor_sec"] * SAMPLING_RATE)
        pre_pad_samples = int(0.3 * SAMPLING_RATE)  # 300ms 사전 패딩
        speech_start_in_slot = self.vad_last_speech_start_sample - slot_anchor_samples - pre_pad_samples
        speech_start_in_slot = max(0, speech_start_in_slot)

        # 트리밍 효과가 없을 때(500ms 미만 단축) 건너뜀
        if speech_start_in_slot < int(0.5 * SAMPLING_RATE):
            self.log.info(
                f"[VAD-RETRY] slot={slot_key} skip: trim too small "
                f"speech_start_in_slot={speech_start_in_slot}"
            )
            return

        speech_audio = full_audio[speech_start_in_slot:]
        if speech_audio.shape[0] == 0:
            return

        self.log.info(
            f"[VAD-RETRY] slot={slot_key} "
            f"speech_start_sample={self.vad_last_speech_start_sample} "
            f"trim_from={speech_start_in_slot} ({speech_start_in_slot / SAMPLING_RATE:.2f}s) "
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

    async def _on_vad_done(self, slot_key: str) -> None:  # noqa: ARG002
        """VAD 발화 종료 플러시 완료 후 호출되는 훅. 서브클래스에서 오버라이드 가능."""
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
        if self.gpt_translator and original and translation:
            self._segment_history.append((original, translation))

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
                    "enable_correction": self.config.enable_correction,
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
                            self.client_lang = data.get("lang", "auto")
                            self.client_target_lang = data.get("targetLang", "")
                            self.use_correction = data.get("speed", "accurate") != "fast"
                            self.log.info(
                                f"Received start: lang={self.client_lang}, "
                                f"targetLang={self.client_target_lang}, "
                                f"speed={data.get('speed', 'accurate')}"
                            )

                            self.init_streaming_state()
                            self.running = True
                            await self.send_message(
                                "ready", message="Ready to receive audio"
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
        if _SILERO_VAD_AVAILABLE:
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
            max_model_len=8192,
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
                logger.warning("GPT translator requested but core.llm_corrector import failed — falling back to Google Translate")
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
                logger.warning("GPT corrector requested but core.llm_corrector import failed — correction disabled")
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
        "--max-new-tokens", type=int, default=32,
        help="Max new tokens per chunk",
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
        help="GPT 번역 시 참고할 직전 세그먼트 최대 문장 수 (기본값: 5, --gpt-translation 활성화 시 사용)",
    )
    parser.add_argument(
        "--log-json", action="store_true",
        help="로그를 JSON 형식으로 출력",
    )
    args = parser.parse_args()
    if args.enable_dot_commit is None:
        args.enable_dot_commit = _infer_dot_commit_default(args.model)
    return args


def main():
    args = parse_args()
    _configure_logging(use_json=args.log_json)

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
        enable_dot_commit=args.enable_dot_commit,
        restrict_languages=not args.no_restrict_languages,
        enable_correction=args.correction,
        correction_model=args.correction_model,
        api_key=args.api_key,
        enable_gpt_translation=args.gpt_translation,
        translation_model=args.translation_model,
        context_window=args.context_window,
    )

    server = Qwen3ASRStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
