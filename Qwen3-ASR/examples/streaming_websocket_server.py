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
import json
import logging
import os
import re
import traceback
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
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers)


logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
# VADIterator 설정
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 800       # 발화 종료 판정까지 필요한 침묵 길이
VAD_SPEECH_PAD_MS = 160        # 발화 경계에 추가하는 패딩
VAD_WINDOW_SIZE_SAMPLES = 512  # 16kHz 기준 silero 권장 윈도우 크기


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
    enable_correction: bool = True
    correction_model: str = "gpt-5.4-mini"
    correction_api_key: Optional[str] = None

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8765
    no_idle_shutdown: bool = False
    idle_shutdown_sec: int = 60


def format_time(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


LANG_NAME_TO_CODE = {
    "Korean": "ko", "English": "en", "Japanese": "ja", "Chinese": "zh",
    "Indonesian": "id", "Vietnamese": "vi", "Thai": "th",
    "Spanish": "es", "French": "fr", "German": "de",
}


LANG_CODE_TO_NAME = {v: k for k, v in LANG_NAME_TO_CODE.items()}


def lang_to_code(lang: str) -> str:
    """언어 이름을 코드로 변환 (Korean -> ko)"""
    return LANG_NAME_TO_CODE.get(lang, lang.lower()[:2])


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

        # Commit 방식 설정
        self.enable_dot_commit: bool = config.enable_dot_commit

        # VAD / stream alignment
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        self.active_slot = "A"
        self.standby_slot = "B"
        self.stream_slots: dict[str, dict] = {}

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
    
    def _new_stream_slot(self) -> dict:
        allowed_languages = None
        if self.config.restrict_languages:
            candidates = []
            for code in (self.client_lang, self.client_target_lang):
                if code and code != "auto":
                    name = lang_code_to_name(code)
                    if name:
                        candidates.append(name)
            if len(candidates) >= 2:
                allowed_languages = candidates

        return {
            "state": self.asr.init_streaming_state(
                unfixed_chunk_num=self.config.unfixed_chunk_num,
                unfixed_token_num=self.config.unfixed_token_num,
                chunk_size_sec=self.config.chunk_size_sec,
                allowed_languages=allowed_languages,
            ),
            "flush_lock": asyncio.Lock(),
            "last_text": "",
            "last_text_lang": "",
            "committed_len": 0,
            "committed_prefix": "",
            "committed_display": "",
            "committed_seg_count": 0,
        }

    def _reset_stream_slot(self, slot_key: str):
        self.stream_slots[slot_key] = self._new_stream_slot()
        self.log.info(f"[slot-reset] slot={slot_key}")

    def _slot(self, slot_key: Optional[str] = None) -> dict:
        key = slot_key or self.active_slot
        return self.stream_slots[key]

    @staticmethod
    def _uncommitted_from(current_text: str, committed_display: str,
                          committed_seg_count: int = 0) -> str:
        """committed 경계 이후의 current_text를 반환.

        1차: committed_display(SEG 제거 기준) prefix 매칭
        2차(fallback): committed_seg_count 번째 <SEG> 이후 텍스트
        모델이 이전 텍스트를 수정해 display가 달라져도 SEG 카운트로 안전하게 찾는다.
        """
        seg_tag = "<SEG>"
        seg_len = len(seg_tag)

        # ── 1차: display prefix 매칭 ──────────────────────────────────────
        if committed_display:
            current_no_seg = current_text.replace(seg_tag, "")
            if current_no_seg.startswith(committed_display):
                pos, disp_pos, target = 0, 0, len(committed_display)
                while pos < len(current_text) and disp_pos < target:
                    if current_text[pos:pos + seg_len] == seg_tag:
                        pos += seg_len
                    else:
                        disp_pos += 1
                        pos += 1
                return current_text[pos:]

        # ── 2차 fallback: SEG 카운트 기준 ───────────────────────────────
        pos, found = 0, 0
        while found < committed_seg_count:
            idx = current_text.find(seg_tag, pos)
            if idx == -1:
                return ""  # 커밋된 SEG 수보다 현재 텍스트의 SEG가 적음 → 모두 커밋됨
            pos = idx + seg_len
            found += 1
        return current_text[pos:]

    def _get_lora_request(self, state):
        """언어에 따라 적절한 LoRA 어댑터 반환. 미지원 언어는 None(기본 모델)."""
        lang = state.force_language or state.language  # canonical name e.g. "Korean"
        if lang == "English":
            return self.lora_request_en
        if lang == "Korean":
            return self.lora_request_ko
        return None

    async def _asr_streaming_transcribe(self, chunk: np.ndarray, slot_key: Optional[str] = None):
        slot = self._slot(slot_key)

        async def _on_seg(_):
            # lock 밖에서 호출되므로 _process_slot_updates가 asr_lock 자유롭게 획득 가능
            await self._process_slot_updates(slot_key)

        async with self.asr_lock:
            lora_request = self._get_lora_request(slot["state"])

        # 생성 중 lock 미보유 — state.text는 await 없는 단순 대입이므로 asyncio 안전
        await self.asr.streaming_transcribe(chunk, slot["state"], lora_request=lora_request, on_seg=_on_seg)

        async with self.asr_lock:
            self.asr_processed_cursor = self.sample_cursor

    async def _asr_finish_streaming(self, slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        async with self.asr_lock:
            state = slot["state"]
            partial_lang = slot["last_text_lang"]  # canonical name e.g. "Korean"
            # partial에서 감지된 언어가 있고, state에 force_language가 없으면
            # finish pass에서도 같은 언어를 강제해 hallucination 방지
            if partial_lang and not state.force_language:
                state.prompt_raw = self.asr._build_text_prompt(
                    context=state.context,
                    force_language=partial_lang,
                )
                state.force_language = partial_lang
            lora_request = self._get_lora_request(state)
            await self.asr.finish_streaming_transcribe(state, lora_request=lora_request)

    async def flush_uncommitted(self, force=False, reason="flush", slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        slot_flush_lock = slot["flush_lock"]

        # 1단계: 텍스트 스냅샷만 lock 안에서 읽기
        async with slot_flush_lock:
            async with self.asr_lock:
                state = slot["state"]
                current_text = (state.text or "").strip() if state else ""
                if "<asr_text>" in current_text:
                    current_text = current_text.split("<asr_text>", 1)[-1].strip()
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
                logger.info(f"[timeout-skip] reason={reason} too short: '{uncommitted_display}'")
                return

        # 2단계: lock 해제 후 I/O (번역 API) 수행 → 다른 슬롯 flush 비차단
        audio_end_sec = self._get_flush_audio_end_sec()
        if self.corrector:
            uncommitted_display = await self.corrector.correct_text(uncommitted_display, current_lang)
        translation, detected_lang, extra = await self._translate(
            uncommitted_display, self.client_target_lang, audio_end_sec
        )
        self.log.info(
            f"[translate-flush] reason={reason} sentence='{uncommitted_display}' "
            f"tl={self.client_target_lang} -> detected={detected_lang} "
            f"translation='{translation}'"
        )
        # Google Translate가 같은 언어끼리 번역 시 data[2]를 null로 반환하는 경우
        # detected_lang이 빈 문자열이 될 수 있으므로 ASR 감지 언어를 fallback으로 사용
        effective_detected = detected_lang or lang_to_code(current_lang)
        if effective_detected == self.client_target_lang:
            translation, _, extra = await self._translate(
                uncommitted_display, self.client_lang, audio_end_sec
            )
            self.log.info(
                f"[translate-flush-flip] reason={reason} "
                f"tl={self.client_lang} -> translation='{translation}'"
            )

        final_lang = effective_detected
        if reason.startswith("vad"):
            commit_reason = "vad"
        elif reason == "finish":
            commit_reason = "finish"
        else:
            commit_reason = "seg"
        self.log.info(
            f"[final-flush] slot={slot_key or self.active_slot} reason={reason} lang={final_lang} "
            f"translation='{translation}' original='{uncommitted_display}'"
        )
        if reason.startswith("vad"):
            self.log.info(
                f"[final-vad] slot={slot_key or self.active_slot} lang={final_lang} text='{uncommitted_display}' "
                f"translation='{translation}'"
            )

        # 3단계: lock 재획득 후 커밋 — committed_len이 변경됐으면 skip (중복 방지)
        async with slot_flush_lock:
            if slot["committed_len"] != snapshot_committed_len:
                self.log.info(
                    f"[flush-skip] reason={reason} committed_len advanced "
                    f"({snapshot_committed_len} -> {slot['committed_len']}), skipping duplicate"
                )
                return
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

    async def _process_slot_updates(self, slot_key: str):
        slot = self._slot(slot_key)
        state = slot["state"]
        current_text = (state.text or "").strip()
        if "<asr_text>" in current_text:
            current_text = current_text.split("<asr_text>", 1)[-1].strip()
        current_lang = state.language or ""
        if not current_text or current_text == slot["last_text"]:
            return

        slot["last_text"] = current_text
        slot["last_text_lang"] = current_lang

        # 문장 단위 commit
        # committed_display/seg_count 기준으로 uncommitted 구간 계산 (모델 텍스트 수정에도 안전)
        uncommitted = self._uncommitted_from(
            current_text, slot["committed_display"], slot["committed_seg_count"]
        )
        sentences_to_commit = []
        remaining = uncommitted

        while True:
            # 우선순위 1: <SEG>
            # 우선순위 2: VAD (flush_uncommitted 에서 처리)
            # 우선순위 3: dot (enable_dot_commit=True일 때만 활성화)
            # dot 패턴: Mr./Mrs./Dr./St./Jr./Sr./vs./No. 등 약어 제외
            if self.enable_dot_commit:
                match = re.search(
                    r"(?:"
                    r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<!Jr)(?<!Sr)(?<!vs)(?<!No)\.\s+"
                    r"|[?!\u3002\uff1f\uff01]\s+"
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
            if sentence.replace("<SEG>", "").strip():
                sentences_to_commit.append((sentence, trigger))
            remaining = after

        if sentences_to_commit:
            self.log.info(
                f"[seg-detect] slot={slot_key} raw={current_text[:120]}..."
            )
            translated_payloads = []
            for sentence_raw, trigger_reason in sentences_to_commit:
                # <SEG>는 사용자 출력/번역에서 제거
                sentence_display = sentence_raw.replace("<SEG>", "").strip()
                if self.corrector:
                    sentence_display = await self.corrector.correct_text(sentence_display, current_lang)
                translation, detected_lang, extra = await self._translate(
                    sentence_display, self.client_target_lang, self.current_time
                )
                self.log.info(
                    f"[translate-sentence] sentence='{sentence_display}' "
                    f"tl={self.client_target_lang} -> detected={detected_lang} "
                    f"translation='{translation}'"
                )
                effective_detected = detected_lang or lang_to_code(current_lang)
                if effective_detected == self.client_target_lang:
                    translation, _, extra = await self._translate(
                        sentence_display, self.client_lang, self.current_time
                    )
                    self.log.info(
                        f"[translate-sentence-flip] tl={self.client_lang} "
                        f"-> translation='{translation}'"
                    )
                final_lang = effective_detected
                translated_payloads.append({
                    "original_raw": sentence_raw,   # 커서 추적용 (raw, <SEG> 포함)
                    "original": sentence_display,   # 사용자 출력용
                    "translation": translation,
                    "language": final_lang,
                    "extra": extra,
                    "trigger_reason": trigger_reason,
                })

            ready_to_emit = []
            async with slot["flush_lock"]:
                async with self.asr_lock:
                    latest_state = slot["state"]
                    latest_text = (latest_state.text or "").strip() if latest_state else ""

                cursor = slot["committed_len"]
                tail = latest_text[cursor:]

                for payload in translated_payloads:
                    sentence_raw = payload["original_raw"]
                    stripped_tail = tail.lstrip()
                    leading_ws = len(tail) - len(stripped_tail)
                    if not stripped_tail.startswith(sentence_raw):
                        break
                    cursor += leading_ws + len(sentence_raw)
                    tail = latest_text[cursor:]
                    ready_to_emit.append(payload)

                if ready_to_emit:
                    slot["committed_len"] = cursor
                    slot["committed_prefix"] = latest_text[:slot["committed_len"]]
                    slot["committed_display"] = re.sub(
                        r'\s+', ' ', slot["committed_prefix"].replace("<SEG>", "")
                    ).strip()
                    slot["committed_seg_count"] += len(ready_to_emit)

            for payload in ready_to_emit:
                await self._emit_final_payload(
                    slot_key=slot_key,
                    original=payload["original"],
                    translation=payload["translation"],
                    language=payload["language"],
                    reason=payload["trigger_reason"],
                    audio_end_sec=self.current_time,
                    extra=payload.get("extra"),
                )
                self.log.info(
                    f"[final-sentence/{payload['trigger_reason'].upper()}] slot={slot_key} lang={payload['language']} "
                    f"text={payload['original']}"
                )

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
                if speech_dict is not None and "end" in speech_dict:
                    end_sample = self.sample_cursor - (chunk.size - offset - VAD_WINDOW_SIZE_SAMPLES)
                    local_idx = int(end_sample - chunk_base_sample)
                    local_idx = max(0, min(int(chunk.size), local_idx))
                    if not vad_end_local_indices or local_idx > vad_end_local_indices[-1]:
                        vad_end_local_indices.append(local_idx)
                    self.log.info(
                        f"[vad] speech end detected; target_samples={end_sample}"
                    )
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

        # ASR 처리
        self.current_time += chunk.size / SAMPLING_RATE
        seg_start = 0
        for local_cut in vad_end_local_indices:
            if local_cut <= seg_start:
                continue

            pre_chunk = chunk[seg_start:local_cut]
            if pre_chunk.size > 0:
                await self._asr_streaming_transcribe(pre_chunk, self.active_slot)

            target_sample = chunk_base_sample + local_cut
            target_audio_end_sec = target_sample / SAMPLING_RATE
            self.log.info(
                f"[vad-commit] target_samples={target_sample} "
                f"target_audio_end_sec={target_audio_end_sec:.3f} "
                f"processed={self.asr_processed_cursor} active={self.active_slot}"
            )

            old_active = self.active_slot
            self.active_slot, self.standby_slot = self.standby_slot, self.active_slot
            self.state = self.stream_slots[self.active_slot]["state"]
            self.log.info(
                f"[slot-switch] old_active={old_active} new_active={self.active_slot} "
                f"new_standby={self.standby_slot}"
            )

            # 잔여 버퍼(chunk_size 미만 tail audio)를 모델에 통과시킨 후 flush
            await self._on_vad_commit(target_audio_end_sec)
            await self._asr_finish_streaming(old_active)
            await self._process_slot_updates(old_active)
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

    def _get_flush_audio_end_sec(self) -> float:
        """flush 시 사용할 오디오 종료 시각(초). 서브클래스에서 오버라이드 가능."""
        return self.current_time

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
        """최종 세그먼트 전송 훅. 서브클래스에서 오버라이드해 FCL 메타데이터 등 추가 가능."""
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

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            self.log.info(f"New connection from {remote_addr}")
            await self.send_message("hello", message="Qwen3-ASR Streaming Server")
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
                            self.log.info(
                                f"Received start: lang={self.client_lang}, "
                                f"targetLang={self.client_target_lang}"
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
                    skip_special_tokens=False,  # DEBUG: special token 출력용
                )
                logger.info(f"Beam search enabled: beam_size={self.config.beam_size}")
            except TypeError:
                logger.warning(
                    f"이 vLLM 버전은 use_beam_search를 지원하지 않습니다. "
                    f"Greedy decoding으로 fallback합니다. (beam_size={self.config.beam_size} 무시)"
                )
        else:
            logger.info("Greedy decoding (beam_size=1)")

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

        if self.config.enable_correction:
            if not _CORRECTOR_AVAILABLE:
                logger.warning("GPT corrector requested but core.llm_corrector import failed — correction disabled")
            elif not os.environ.get("OPENAI_API_KEY"):
                logger.warning("GPT corrector requested but OPENAI_API_KEY not set — correction disabled")
            else:
                self.corrector = GPTCorrector(model=self.config.correction_model, api_key=self.config.correction_api_key)
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
            )
            await handler.handle()
        finally:
            async with self.connection_lock:
                self.active_connections -= 1
                logger.info(f"Client disconnected (active={self.active_connections})")
                if self.active_connections == 0:
                    self._restart_idle_timer()

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
            max_size=10 * 1024 * 1024,
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
        "--enable-dot-commit", action="store_true",
        help="온점/느낌표/물음표(dot) 기반 seg commit 활성화 (기본값: 비활성화, 베이스라인 모델용)",
    )
    parser.add_argument(
        "--no-restrict-languages", action="store_true",
        help="앱 설정 두 언어 외 언어 차단 비활성화 (기본값: 활성화)",
    )
    parser.add_argument(
        "--no-correction", action="store_true",
        help="GPT LLM 후처리 비활성화 (기본값: 활성화)",
    )
    parser.add_argument(
        "--correction-model", type=str, default="gpt-5.4-mini",
        help="후처리에 사용할 GPT 모델명 (기본값: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--correction-api-key", type=str, default=None,
        help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--log-json", action="store_true",
        help="로그를 JSON 형식으로 출력",
    )
    return parser.parse_args()


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
        beam_size=args.beam_size,
        adapter_en=args.adapter_en,
        adapter_ko=args.adapter_ko,
        no_lora=not args.lora,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=args.enforce_eager,
        enable_dot_commit=args.enable_dot_commit,
        restrict_languages=not args.no_restrict_languages,
        enable_correction=not args.no_correction,
        correction_model=args.correction_model,
        correction_api_key=args.correction_api_key,
    )

    server = Qwen3ASRStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
