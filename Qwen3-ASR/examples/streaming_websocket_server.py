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
import re
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import aiohttp
import torch

try:
    import websockets
except ImportError:
    raise ImportError("websockets 패키지가 필요합니다: pip install websockets")

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import warmup_streaming
from silero_vad import load_silero_vad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/asr_server.log"),
    ]
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
VAD_WINDOW_MS = 32           # silero default window size
VAD_HOP_MS = 8               # silero default hop
VAD_SR = 16000
VAD_SILENCE_MS = 320         # need this much silence to close a segment
VAD_PAD_MS = 160             # add tail pad to ensure decoder catches up


@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    # 모델 설정
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    gpu_memory_utilization: float = 0.8
    max_new_tokens: int = 32

    # 스트리밍 설정
    chunk_size_sec: float = 2.0  # 한 번에 처리할 오디오 길이 (초)
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8765


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


async def google_translate_async(session: aiohttp.ClientSession, text: str, target_lang: str) -> tuple[str, str]:
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

    def __init__(self, websocket, asr_model: Qwen3ASRModel, config: StreamingConfig):
        self.websocket = websocket
        self.asr = asr_model
        self.config = config
        self.state = None
        self.running = False
        self.last_text = ""
        self.committed_len = 0
        self.committed_prefix = ""  # committed_len validity tracking
        self.last_text_lang = ""

        # 클라이언트 옵션
        self.client_lang = "auto"
        self.client_target_lang = ""  # 번역 대상 언어

        # 타임스탬프 추적
        self.segment_start_time = 0.0
        self.current_time = 0.0
        # Commit policy: if no new partial arrives for timeout, commit tail.
        self.partial_commit_timeout_sec = 2.0
        self.last_partial_wall_ts = 0.0
        self.partial_watchdog_interval_sec = 0.1
        self.partial_watchdog_task = None
        self.flush_lock = asyncio.Lock()
        self.asr_lock = asyncio.Lock()
        self.http_session: Optional[aiohttp.ClientSession] = None

        # VAD / commit alignment
        self.sample_cursor = 0              # total samples ingested
        self.asr_processed_cursor = 0       # best-effort processed samples
        self.pending_commits: list[int] = []  # target end indices (samples)
        self.commit_task = None
        self.vad_model, self.vad_utils = load_silero_vad()
        self.vad_state = None
        self.vad_silence_run = 0
        self.vad_hangover_samples = int(VAD_SILENCE_MS * VAD_SR / 1000)
        self.vad_pad_samples = int(VAD_PAD_MS * VAD_SR / 1000)

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
        self.state = self.asr.init_streaming_state(
            unfixed_chunk_num=self.config.unfixed_chunk_num,
            unfixed_token_num=self.config.unfixed_token_num,
            chunk_size_sec=self.config.chunk_size_sec,
        )
        self.last_text = ""
        self.committed_len = 0
        self.committed_prefix = ""
        self.last_text_lang = ""
        self.segment_start_time = 0.0
        self.current_time = 0.0
        self.last_partial_wall_ts = 0.0
        logger.info("Streaming state initialized")

        # reset VAD
        self.sample_cursor = 0
        self.asr_processed_cursor = 0
        self.pending_commits.clear()
        self.vad_state = None
        self.vad_silence_run = 0
        self._stop_commit_task()
        self.commit_task = asyncio.create_task(self._commit_loop())

    async def _start_partial_watchdog(self):
        self._stop_partial_watchdog()
        self.partial_watchdog_task = asyncio.create_task(self._partial_watchdog_loop())

    def _stop_partial_watchdog(self):
        if self.partial_watchdog_task and not self.partial_watchdog_task.done():
            self.partial_watchdog_task.cancel()
        self.partial_watchdog_task = None

    def _stop_commit_task(self):
        if self.commit_task and not self.commit_task.done():
            self.commit_task.cancel()
        self.commit_task = None

    async def _partial_watchdog_loop(self):
        try:
            while self.running:
                await asyncio.sleep(self.partial_watchdog_interval_sec)
                await self.maybe_flush_by_partial_timeout(trigger="watchdog")
        except asyncio.CancelledError:
            return

    async def _asr_streaming_transcribe(self, chunk: np.ndarray):
        async with self.asr_lock:
            await asyncio.to_thread(self.asr.streaming_transcribe, chunk, self.state)
            # best-effort: assume decoder processed this chunk fully
            self.asr_processed_cursor = self.sample_cursor

    async def _asr_finish_streaming(self):
        async with self.asr_lock:
            await asyncio.to_thread(self.asr.finish_streaming_transcribe, self.state)

    async def maybe_flush_by_partial_timeout(self, trigger: str):
        if not self.running or self.state is None:
            return
        if self.last_partial_wall_ts <= 0:
            return
        idle_elapsed = time.monotonic() - self.last_partial_wall_ts
        if idle_elapsed < self.partial_commit_timeout_sec:
            return

        await self.flush_uncommitted(reason=f"partial-timeout:{trigger}")
        # Prevent repeated flush loops until a new partial arrives.
        self.last_partial_wall_ts = 0.0

    async def flush_uncommitted(self, force=False, reason="partial-timeout"):
        async with self.flush_lock:
            async with self.asr_lock:
                current_text = (self.state.text or "").strip() if self.state else ""
                current_lang = self.last_text_lang or ""
            uncommitted = current_text[self.committed_len:].strip()
            if not uncommitted:
                return
            if not force and len(uncommitted) < 2:
                logger.info(f"[timeout-skip] reason={reason} too short: '{uncommitted}'")
                return

            translation, detected_lang = await google_translate_async(self.http_session, uncommitted, self.client_target_lang)
            logger.info(f"[translate-timeout] reason={reason} sentence='{uncommitted}' tl={self.client_target_lang} -> detected={detected_lang} translation='{translation}'")
            if detected_lang == self.client_target_lang:
                translation, _ = await google_translate_async(self.http_session, uncommitted, self.client_lang)
                logger.info(f"[translate-timeout-flip] reason={reason} tl={self.client_lang} -> translation='{translation}'")

            final_lang = detected_lang or lang_to_code(current_lang)
            logger.info(f"[final-timeout] reason={reason} lang={final_lang} translation='{translation}' original='{uncommitted}'")
            await self.send_message(
                "final",
                start=format_time(self.segment_start_time),
                end=format_time(self.current_time),
                original=uncommitted,
                translation=translation,
                language=final_lang,
            )
            self.committed_len = len(current_text)
            self.committed_prefix = current_text

    async def process_audio_chunk(self, audio_data: bytes):
        if not audio_data:
            return

        chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if chunk.size == 0:
            return

        # Track absolute cursor before decode
        self.sample_cursor += chunk.size

        # VAD inference (silero expects 16k mono float32)
        with torch.no_grad():
            speech_probs, self.vad_state = self.vad_utils["get_speech_ts_adaptive"](
                torch.from_numpy(chunk),
                self.vad_model,
                self.vad_state,
                sampling_rate=VAD_SR,
                return_seconds=False,
                threshold=0.5,
                min_silence_ms=VAD_SILENCE_MS,
                speech_pad_ms=0,
            )
        # speech_probs is a list of {"start":..., "end":...} relative to this chunk
        # We only care about silence transition; if speech->silence ended in this chunk, enqueue commit target
        if speech_probs:
            last = speech_probs[-1]
            # if segment ended inside this chunk
            if last["end"] < chunk.size:
                end_idx = self.sample_cursor - (chunk.size - last["end"])
                end_idx += self.vad_pad_samples  # pad for decoder lag
                self.pending_commits.append(end_idx)

        # No min_chunk gate: process every incoming chunk immediately.
        self.current_time += chunk.size / SAMPLING_RATE
        await self._asr_streaming_transcribe(chunk)

        current_text = (self.state.text or "").strip()
        current_lang = self.state.language or ""
        if not current_text or current_text == self.last_text:
            await self.maybe_flush_by_partial_timeout(trigger="audio")
            return

        self.last_text = current_text
        self.last_text_lang = current_lang

        await self.send_message(
            "partial",
            original=current_text,
            last_translation=""
        )
        logger.info(f"[partial] text={current_text[:80]}...")
        self.last_partial_wall_ts = time.monotonic()

        uncommitted = current_text[self.committed_len:]
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
            for sentence in sentences_to_commit:
                translation, detected_lang = await google_translate_async(self.http_session, sentence, self.client_target_lang)
                logger.info(f"[translate-sentence] sentence='{sentence}' tl={self.client_target_lang} -> detected={detected_lang} translation='{translation}'")
                if detected_lang == self.client_target_lang:
                    translation, _ = await google_translate_async(self.http_session, sentence, self.client_lang)
                    logger.info(f"[translate-sentence-flip] tl={self.client_lang} -> translation='{translation}'")
                final_lang = detected_lang or lang_to_code(current_lang)
                await self.send_message(
                    "final",
                    start=format_time(self.segment_start_time),
                    end=format_time(self.current_time),
                    original=sentence,
                    translation=translation,
                    language=final_lang
                )
                logger.info(f"[final-sentence] lang={final_lang} text={sentence}")

            if remaining.strip():
                self.committed_len = len(current_text) - len(remaining)
            else:
                self.committed_len = len(current_text)
            self.committed_prefix = current_text[:self.committed_len]

        # Also check from audio path; watchdog handles idle periods.
        await self.maybe_flush_by_partial_timeout(trigger="audio")

    async def _commit_loop(self):
        try:
            while True:
                if self.pending_commits and self.asr_processed_cursor >= self.pending_commits[0]:
                    target = self.pending_commits.pop(0)
                    logger.info(f"[vad-commit] target_samples={target} processed={self.asr_processed_cursor}")
                    await self.flush_uncommitted(force=True, reason="vad")
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return

    async def finish_streaming(self):
        if self.state is None:
            return

        await self._asr_finish_streaming()
        await self.flush_uncommitted(force=True, reason="finish")

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            logger.info(f"New connection from {remote_addr}")
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
                            self.client_lang = data.get("lang", "auto")
                            self.client_target_lang = data.get("targetLang", "")
                            logger.info(f"Received start: lang={self.client_lang}, targetLang={self.client_target_lang}")

                            self.init_streaming_state()
                            self.running = True
                            await self._start_partial_watchdog()
                            # commit loop already started in init_streaming_state
                            await self.send_message("ready", message="Ready to receive audio")

                        elif msg_type == "stop" or msg_type == "finish":
                            logger.info(f"Received {msg_type} command")
                            self.running = False
                            self._stop_partial_watchdog()
                            await self.finish_streaming()

                            if msg_type == "stop":
                                break

                            self.init_streaming_state()
                            self.running = True
                            await self._start_partial_watchdog()

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
            self._stop_partial_watchdog()
            self._stop_commit_task()
            if was_running:
                await self.finish_streaming()
            if self.http_session is not None:
                with contextlib.suppress(Exception):
                    await self.http_session.close()
                self.http_session = None
            logger.info("Connection closed")


class Qwen3ASRStreamingServer:
    """Qwen3-ASR 스트리밍 서버"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.asr = None

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
        handler = Qwen3ASRStreamingHandler(websocket, self.asr, self.config)
        await handler.handle()

    async def start(self):
        """서버 시작"""
        logger.info(f"Starting WebSocket server on ws://{self.config.host}:{self.config.port}")

        async with websockets.serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            ping_interval=None,
            ping_timeout=None,
            max_size=10 * 1024 * 1024,  # 10MB
        ):
            logger.info(f"Server listening on ws://{self.config.host}:{self.config.port}")
            await asyncio.Future()  # run forever


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Streaming WebSocket Server")

    # 모델 설정
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B",
                        help="Model path or name")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="GPU memory utilization (0.0 ~ 1.0)")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Max new tokens per chunk")

    # 스트리밍 설정
    parser.add_argument("--chunk-size", type=float, default=2.0,
                        help="Chunk size in seconds")

    # 서버 설정
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8765,
                        help="Server port")

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
    )

    server = Qwen3ASRStreamingServer(config)
    server.init_model()

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
