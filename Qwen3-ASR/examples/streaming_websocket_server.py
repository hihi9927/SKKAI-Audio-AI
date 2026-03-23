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

# ──────────────────────────────────────────────────────────────────────
# silero-vad: 최신 API (v3+) 사용 — VADIterator 기반 스트리밍
# ──────────────────────────────────────────────────────────────────────
try:
    from silero_vad import load_silero_vad, VADIterator
    _SILERO_VAD_AVAILABLE = True
except ImportError:
    _SILERO_VAD_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs/asr_server.log")),
    ]
)
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

    # 스트리밍 설정
    chunk_size_sec: float = 2.0
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5

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


def lang_to_code(lang: str) -> str:
    """언어 이름을 코드로 변환 (Korean -> ko)"""
    return LANG_NAME_TO_CODE.get(lang, lang.lower()[:2])


class SessionLogger:
    """앱에서 수신한 로그를 세션별 JSON 파일로 저장"""

    def __init__(self, logs_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs/asr_logs")):
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(logs_dir, f"session_{ts}.json")
        self._entries: list[dict] = []
        self._lock = asyncio.Lock()
        logger.info(f"[session-log] log file: {self.path}")

    async def append(self, time: str, text: str, translation: str) -> None:
        async with self._lock:
            self._entries.append({
                "time": time,
                "text": text,
                "translation": translation,
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


class Qwen3ASRStreamingHandler:
    """WebSocket 연결 당 하나의 스트리밍 핸들러"""

    def __init__(
        self,
        websocket,
        asr_model: Qwen3ASRModel,
        config: StreamingConfig,
        pairing_hub: PairingHub,
    ):
        self.websocket = websocket
        self.asr = asr_model
        self.config = config
        self.pairing_hub = pairing_hub
        self.state = None
        self.running = False

        # 클라이언트 옵션
        self.client_lang = "auto"
        self.client_target_lang = ""

        # 타임스탬프 추적
        self.segment_start_time = 0.0
        self.current_time = 0.0

        self.flush_lock = asyncio.Lock()
        self.asr_lock = asyncio.Lock()
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_logger: Optional[SessionLogger] = None

        # VAD / stream alignment
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        self.active_slot = "A"
        self.standby_slot = "B"
        self.stream_slots: dict[str, dict] = {}

        # ── silero-vad 초기화 (VADIterator 사용) ──
        self.vad_enabled = False
        self.vad_iterator = None
        if _SILERO_VAD_AVAILABLE:
            try:
                vad_model = load_silero_vad()
                self.vad_iterator = VADIterator(
                    model=vad_model,
                    threshold=VAD_THRESHOLD,
                    sampling_rate=SAMPLING_RATE,
                    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                    speech_pad_ms=VAD_SPEECH_PAD_MS,
                )
                self.vad_enabled = True
                logger.info("Silero VAD (VADIterator) loaded successfully")
            except Exception as e:
                logger.warning(f"Silero VAD disabled: {e}")
        else:
            logger.warning(
                "silero-vad 패키지가 설치되지 않았습니다. "
                "VAD 없이 동작합니다. 설치: pip install silero-vad"
            )

    async def send_message(self, msg_type: str, **kwargs):
        """JSON 메시지 전송"""
        message = {"type": msg_type, **kwargs}
        try:
            await self.websocket.send(json.dumps(message, ensure_ascii=False))
            logger.debug(f"Sent: {msg_type}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

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
        logger.info("Streaming state initialized")

        # reset VAD
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        if self.vad_iterator is not None:
            self.vad_iterator.reset_states()
    
    def _new_stream_slot(self) -> dict:
        return {
            "state": self.asr.init_streaming_state(
                unfixed_chunk_num=self.config.unfixed_chunk_num,
                unfixed_token_num=self.config.unfixed_token_num,
                chunk_size_sec=self.config.chunk_size_sec,
            ),
            "last_text": "",
            "last_text_lang": "",
            "committed_len": 0,
            "committed_prefix": "",
        }

    def _reset_stream_slot(self, slot_key: str):
        self.stream_slots[slot_key] = self._new_stream_slot()
        logger.info(f"[slot-reset] slot={slot_key}")

    def _slot(self, slot_key: Optional[str] = None) -> dict:
        key = slot_key or self.active_slot
        return self.stream_slots[key]

    async def _asr_streaming_transcribe(self, chunk: np.ndarray, slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        async with self.asr_lock:
            await asyncio.to_thread(self.asr.streaming_transcribe, chunk, slot["state"])
            self.asr_processed_cursor = self.sample_cursor

    async def _asr_finish_streaming(self, slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        async with self.asr_lock:
            await asyncio.to_thread(self.asr.finish_streaming_transcribe, slot["state"])

    async def flush_uncommitted(self, force=False, reason="flush", slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        async with self.flush_lock:
            async with self.asr_lock:
                state = slot["state"]
                current_text = (state.text or "").strip() if state else ""
                current_lang = slot["last_text_lang"] or ""
            uncommitted = current_text[slot["committed_len"]:].strip()
            if not uncommitted:
                return
            if not force and len(uncommitted) < 2:
                logger.info(f"[timeout-skip] reason={reason} too short: '{uncommitted}'")
                return

            translation, detected_lang = await google_translate_async(
                self.http_session, uncommitted, self.client_target_lang
            )
            logger.info(
                f"[translate-flush] reason={reason} sentence='{uncommitted}' "
                f"tl={self.client_target_lang} -> detected={detected_lang} "
                f"translation='{translation}'"
            )
            if detected_lang == self.client_target_lang:
                translation, _ = await google_translate_async(
                    self.http_session, uncommitted, self.client_lang
                )
                logger.info(
                    f"[translate-flush-flip] reason={reason} "
                    f"tl={self.client_lang} -> translation='{translation}'"
                )

            final_lang = detected_lang or lang_to_code(current_lang)
            logger.info(
                f"[final-flush] slot={slot_key or self.active_slot} reason={reason} lang={final_lang} "
                f"translation='{translation}' original='{uncommitted}'"
            )
            if reason.startswith("vad"):
                logger.info(
                    f"[final-vad] slot={slot_key or self.active_slot} lang={final_lang} text='{uncommitted}' "
                    f"translation='{translation}'"
                )
            await self.send_message(
                "final",
                start=format_time(self.segment_start_time),
                end=format_time(self.current_time),
                original=uncommitted,
                translation=translation,
                language=final_lang,
                commitReason=("vad" if reason.startswith("vad") else "seg"),
            )
            if self.session_logger:
                await self.session_logger.append(
                    time=datetime.now(timezone.utc).isoformat(),
                    text=uncommitted,
                    translation=translation,
                )
            slot["committed_len"] = len(current_text)
            slot["committed_prefix"] = current_text

    async def _process_slot_updates(self, slot_key: str, emit_partial: bool = True):
        slot = self._slot(slot_key)
        state = slot["state"]
        current_text = (state.text or "").strip()
        current_lang = state.language or ""
        if not current_text or current_text == slot["last_text"]:
            return

        slot["last_text"] = current_text
        slot["last_text_lang"] = current_lang

        if emit_partial:
            await self.send_message(
                "partial",
                original=current_text,
                last_translation="",
            )
            logger.info(f"[partial] slot={slot_key} text={current_text[:80]}...")

        # 문장 단위 commit
        uncommitted = current_text[slot["committed_len"]:]
        sentences_to_commit = []
        remaining = uncommitted

        while True:
            match = re.search(r"[.?!\u3002\uff1f\uff01]\s+", remaining)
            if not match:
                break
            after = remaining[match.end():]
            if after.strip():
                sentence = remaining[:match.end()].strip()
                if sentence:
                    sentences_to_commit.append(sentence)
                remaining = after
            else:
                break

        if sentences_to_commit:
            translated_payloads = []
            for sentence in sentences_to_commit:
                translation, detected_lang = await google_translate_async(
                    self.http_session, sentence, self.client_target_lang
                )
                logger.info(
                    f"[translate-sentence] sentence='{sentence}' "
                    f"tl={self.client_target_lang} -> detected={detected_lang} "
                    f"translation='{translation}'"
                )
                if detected_lang == self.client_target_lang:
                    translation, _ = await google_translate_async(
                        self.http_session, sentence, self.client_lang
                    )
                    logger.info(
                        f"[translate-sentence-flip] tl={self.client_lang} "
                        f"-> translation='{translation}'"
                    )
                final_lang = detected_lang or lang_to_code(current_lang)
                translated_payloads.append({
                    "original": sentence,
                    "translation": translation,
                    "language": final_lang,
                })

            ready_to_emit = []
            async with self.flush_lock:
                async with self.asr_lock:
                    latest_state = slot["state"]
                    latest_text = (latest_state.text or "").strip() if latest_state else ""

                cursor = slot["committed_len"]
                tail = latest_text[cursor:]

                for payload in translated_payloads:
                    sentence = payload["original"]
                    stripped_tail = tail.lstrip()
                    leading_ws = len(tail) - len(stripped_tail)
                    if not stripped_tail.startswith(sentence):
                        break
                    cursor += leading_ws + len(sentence)
                    tail = latest_text[cursor:]
                    ready_to_emit.append(payload)

                if ready_to_emit:
                    slot["committed_len"] = cursor
                    slot["committed_prefix"] = latest_text[:slot["committed_len"]]

            for payload in ready_to_emit:
                await self.send_message(
                    "final",
                    start=format_time(self.segment_start_time),
                    end=format_time(self.current_time),
                    original=payload["original"],
                    translation=payload["translation"],
                    language=payload["language"],
                    commitReason="seg",
                )
                if self.session_logger:
                    await self.session_logger.append(
                        time=datetime.now(timezone.utc).isoformat(),
                        text=payload["original"],
                        translation=payload["translation"],
                    )
                logger.info(
                    f"[final-sentence] slot={slot_key} lang={payload['language']} "
                    f"text={payload['original']}"
                )

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
        if self.vad_enabled and self.vad_iterator is not None:
            try:
                # VADIterator는 고정 크기(512 samples @16kHz)씩 처리해야 함
                # 청크를 윈도우 크기로 분할하여 순차 처리
                offset = 0
                while offset + VAD_WINDOW_SIZE_SAMPLES <= chunk.size:
                    window = chunk[offset:offset + VAD_WINDOW_SIZE_SAMPLES]
                    speech_dict = self.vad_iterator(
                        torch.from_numpy(window),
                        return_seconds=False,
                    )
                    if speech_dict is not None:
                        # speech_dict = {'start': N} 또는 {'end': N}
                        if "end" in speech_dict:
                            # 발화 종료 감지 → 현재 청크의 경계 인덱스로 변환
                            end_sample = self.sample_cursor - (chunk.size - offset - VAD_WINDOW_SIZE_SAMPLES)
                            local_idx = int(end_sample - chunk_base_sample)
                            local_idx = max(0, min(int(chunk.size), local_idx))
                            if not vad_end_local_indices or local_idx > vad_end_local_indices[-1]:
                                vad_end_local_indices.append(local_idx)
                            logger.info(
                                f"[vad] speech end detected; "
                                f"target_samples={end_sample}"
                            )
                    offset += VAD_WINDOW_SIZE_SAMPLES
            except Exception as e:
                logger.warning(f"[vad] error, disabling for this session: {e}")
                self.vad_enabled = False

        # ASR 처리
        self.current_time += chunk.size / SAMPLING_RATE
        seg_start = 0
        for local_cut in vad_end_local_indices:
            if local_cut <= seg_start:
                continue

            pre_chunk = chunk[seg_start:local_cut]
            if pre_chunk.size > 0:
                await self._asr_streaming_transcribe(pre_chunk, self.active_slot)
                await self._process_slot_updates(self.active_slot, emit_partial=True)

            target_sample = chunk_base_sample + local_cut
            logger.info(
                f"[vad-commit] target_samples={target_sample} "
                f"processed={self.asr_processed_cursor} active={self.active_slot}"
            )

            old_active = self.active_slot
            self.active_slot, self.standby_slot = self.standby_slot, self.active_slot
            self.state = self.stream_slots[self.active_slot]["state"]
            logger.info(
                f"[slot-switch] old_active={old_active} new_active={self.active_slot} "
                f"new_standby={self.standby_slot}"
            )

            await self.flush_uncommitted(force=True, reason="vad", slot_key=old_active)
            self._reset_stream_slot(self.standby_slot)

            seg_start = local_cut

        tail_chunk = chunk[seg_start:]
        if tail_chunk.size > 0:
            await self._asr_streaming_transcribe(tail_chunk, self.active_slot)
            await self._process_slot_updates(self.active_slot, emit_partial=True)

    async def finish_streaming(self):
        if not self.stream_slots:
            return
        await self._asr_finish_streaming(self.active_slot)
        await self.flush_uncommitted(force=True, reason="finish", slot_key=self.active_slot)

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            logger.info(f"New connection from {remote_addr}")
            await self.send_message("hello", message="Qwen3-ASR Streaming Server")
            timeout = aiohttp.ClientTimeout(total=3)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            self.session_logger = SessionLogger()

            async for message in self.websocket:
                if isinstance(message, bytes):
                    if self.running:
                        await self.process_audio_chunk(message)
                else:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "start":
                            self.client_lang = data.get("lang", "auto")
                            self.client_target_lang = data.get("targetLang", "")
                            logger.info(
                                f"Received start: lang={self.client_lang}, "
                                f"targetLang={self.client_target_lang}"
                            )

                            self.init_streaming_state()
                            self.running = True
                            await self.send_message(
                                "ready", message="Ready to receive audio"
                            )

                        elif msg_type in ("stop", "finish"):
                            logger.info(f"Received {msg_type} command")
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

                        elif msg_type == "pair_leave":
                            await self.pairing_hub.leave(self.websocket)

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by client")
        except Exception as e:
            logger.error(f"Error in handler: {e}")
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
            logger.info("Connection closed")


class Qwen3ASRStreamingServer:
    """Qwen3-ASR 스트리밍 서버"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.IDLE_SHUTDOWN_SEC = config.idle_shutdown_sec
        self.asr = None
        self.pairing_hub = PairingHub()
        self.idle_task = None
        self.active_connections = 0          
        self.connection_lock = asyncio.Lock()

    def init_model(self):
        """ASR 모델 초기화"""
        logger.info(f"Loading model: {self.config.model_path}")
        self.asr = Qwen3ASRModel.LLM(
            model=self.config.model_path,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_new_tokens=self.config.max_new_tokens,
            max_model_len=8192,
        )
        warmup_streaming(self.asr)
        logger.info("Model loaded successfully")

    async def handle_connection(self, websocket):
        """각 연결 처리"""
        async with self.connection_lock:
            self.active_connections += 1
            # idle 타이머 취소 (사용자 접속)
            if self.idle_task and not self.idle_task.done():
                self.idle_task.cancel()
            logger.info(f"Client connected ({self.active_connections})")

        try:
            handler = Qwen3ASRStreamingHandler(
                websocket, self.asr, self.config, self.pairing_hub
            )
            await handler.handle()
        finally:
            async with self.connection_lock:
                self.active_connections -= 1
                logger.info(f"Client disconnected ({self.active_connections})")
                if self.active_connections == 0:
                    self._restart_idle_timer()

    async def start(self):
        logger.info(f"Starting WebSocket server on ws://{self.config.host}:{self.config.port}")
        async with websockets.serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            ping_interval=None,
            ping_timeout=None,
            max_size=10 * 1024 * 1024,
        ):
            logger.info(f"Server listening on ws://{self.config.host}:{self.config.port}")
            self._restart_idle_timer()
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
    return parser.parse_args()


def main():
    args = parse_args()

    config = StreamingConfig(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        chunk_size_sec=args.chunk_size,
        host=args.host,
        port=args.port,
        no_idle_shutdown=args.no_idle_shutdown,
        idle_shutdown_sec=args.idle_shutdown_sec,
    )

    server = Qwen3ASRStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
