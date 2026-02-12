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
import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import urllib.request
import urllib.parse

try:
    import websockets
except ImportError:
    raise ImportError("websockets 패키지가 필요합니다: pip install websockets")

from qwen_asr import Qwen3ASRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    # 모델 설정
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    gpu_memory_utilization: float = 0.8
    max_new_tokens: int = 32

    # 스트리밍 설정
    chunk_size_sec: float = 2.0  # 한 번에 처리할 오디오 길이 (초)
    min_chunk_size_sec: float = 0.5  # 최소 처리 단위 (초)
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class AudioBuffer:
    """오디오 버퍼 관리"""
    sample_rate: int = SAMPLING_RATE
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    total_samples: int = 0
    total_duration: float = 0.0  # 누적 시간 (초)

    def add_pcm_bytes(self, data: bytes) -> None:
        """PCM s16le 바이트를 버퍼에 추가"""
        if len(data) == 0:
            return
        pcm_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, pcm_array])
        self.total_samples += len(pcm_array)
        self.total_duration = self.total_samples / self.sample_rate

    def get_chunk(self, num_samples: int) -> Optional[np.ndarray]:
        """버퍼에서 청크 가져오기"""
        if len(self.buffer) < num_samples:
            return None
        chunk = self.buffer[:num_samples].copy()
        self.buffer = self.buffer[num_samples:]
        return chunk

    def get_all(self) -> np.ndarray:
        """버퍼의 모든 데이터 가져오기"""
        chunk = self.buffer.copy()
        self.buffer = np.array([], dtype=np.float32)
        return chunk

    def has_enough(self, min_samples: int) -> bool:
        """최소 샘플 수 이상 있는지 확인"""
        return len(self.buffer) >= min_samples

    def clear(self) -> None:
        """버퍼 초기화"""
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples = 0
        self.total_duration = 0.0


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


def google_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Google Translate 무료 API를 사용한 번역 (서버 사이드, 저지연)"""
    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={source_lang}&tl={target_lang}&dt=t"
            f"&q={urllib.parse.quote(text)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "".join(item[0] for item in data[0] if item[0])
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return ""


class Qwen3ASRStreamingHandler:
    """WebSocket 연결 당 하나의 스트리밍 핸들러"""

    def __init__(self, websocket, asr_model: Qwen3ASRModel, config: StreamingConfig):
        self.websocket = websocket
        self.asr = asr_model
        self.config = config
        self.audio_buffer = AudioBuffer()
        self.state = None
        self.running = False
        self.last_text = ""
        self.committed_len = 0
        self.last_text_lang = ""

        # 클라이언트 옵션
        self.client_lang = "auto"
        self.client_target_lang = ""  # 번역 대상 언어

        # 타임스탬프 추적
        self.segment_start_time = 0.0
        self.current_time = 0.0

        # 샘플 수 계산
        self.chunk_samples = int(config.chunk_size_sec * SAMPLING_RATE)
        self.min_chunk_samples = int(config.min_chunk_size_sec * SAMPLING_RATE)

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
        self.audio_buffer.clear()
        self.last_text = ""
        self.committed_len = 0
        self.last_text_lang = ""
        self.segment_start_time = 0.0
        self.current_time = 0.0
        logger.info("Streaming state initialized")

    async def process_audio_chunk(self, audio_data: bytes):
        """오디오 청크 처리 - partial 그대로 전송 + 문장 감지 시 final 추가 전송"""
        self.audio_buffer.add_pcm_bytes(audio_data)
        self.current_time = self.audio_buffer.total_duration

        while self.audio_buffer.has_enough(self.min_chunk_samples):
            samples_to_get = min(self.chunk_samples, len(self.audio_buffer.buffer))
            chunk = self.audio_buffer.get_chunk(samples_to_get)

            if chunk is not None and len(chunk) > 0:
                self.asr.streaming_transcribe(chunk, self.state)

                current_text = (self.state.text or "").strip()
                current_lang = self.state.language or ""

                if not current_text:
                    continue

                if current_text == self.last_text:
                    continue

                self.last_text = current_text
                self.last_text_lang = current_lang

                # 1) partial: 항상 전체 누적 텍스트를 그대로 보낸다
                await self.send_message(
                    "partial",
                    original=current_text,
                    last_translation="",
                    language=current_lang
                )
                logger.info(f"[partial] lang={current_lang} text={current_text[:80]}...")

                # 2) 문장 분절: committed_len 이후에서 확정된 문장 찾기
                uncommitted = current_text[self.committed_len:]
                sentences_to_commit = []
                remaining = uncommitted

                # "문장. 다음문장..." → "문장." 확정
                while True:
                    match = re.search(r'[.?!。？！]\s+', remaining)
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

                # 확정 문장 → final 전송 (번역 포함)
                if sentences_to_commit:
                    lang_code = lang_to_code(current_lang)
                    for sentence in sentences_to_commit:
                        translation = ""
                        if self.client_target_lang and self.client_target_lang != lang_code:
                            translation = google_translate(sentence, lang_code, self.client_target_lang)
                        await self.send_message(
                            "final",
                            start=format_time(self.segment_start_time),
                            end=format_time(self.current_time),
                            original=sentence,
                            translation=translation,
                            language=current_lang
                        )
                        logger.info(f"[final] lang={current_lang} text={sentence}")

                    if remaining.strip():
                        self.committed_len = len(current_text) - len(remaining)
                    else:
                        self.committed_len = len(current_text)

    async def finish_streaming(self):
        """스트리밍 종료 처리"""
        if self.state is None:
            return

        # 버퍼에 남은 오디오 처리
        remaining = self.audio_buffer.get_all()
        if len(remaining) > 0:
            self.asr.streaming_transcribe(remaining, self.state)

        # 최종 결과 가져오기
        self.asr.finish_streaming_transcribe(self.state)

        final_text = (self.state.text or "").strip()
        final_lang = self.state.language or ""

        # committed_len 이후의 남은 텍스트만 final로 전송
        uncommitted = final_text[self.committed_len:].strip()
        if uncommitted:
            lang_code = lang_to_code(final_lang)
            translation = ""
            if self.client_target_lang and self.client_target_lang != lang_code:
                translation = google_translate(uncommitted, lang_code, self.client_target_lang)

            await self.send_message(
                "final",
                start=format_time(self.segment_start_time),
                end=format_time(self.current_time),
                original=uncommitted,
                translation=translation,
                language=final_lang
            )
            logger.info(f"[final] lang={final_lang} text={uncommitted}")

    async def handle(self):
        """WebSocket 연결 핸들링"""
        try:
            remote_addr = self.websocket.remote_address
            logger.info(f"New connection from {remote_addr}")

            # Hello 메시지 전송
            await self.send_message("hello", message="Qwen3-ASR Streaming Server")

            async for message in self.websocket:
                if isinstance(message, bytes):
                    # 바이너리 데이터 = 오디오
                    if self.running:
                        await self.process_audio_chunk(message)
                else:
                    # 텍스트 데이터 = JSON 명령
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "start":
                            # 클라이언트 옵션 저장
                            self.client_lang = data.get("lang", "auto")
                            self.client_target_lang = data.get("targetLang", "")

                            logger.info(f"Received start: lang={self.client_lang}, "
                                       f"targetLang={self.client_target_lang}")

                            self.init_streaming_state()
                            self.running = True

                            # Ready 메시지 전송
                            await self.send_message("ready", message="Ready to receive audio")

                        elif msg_type == "stop" or msg_type == "finish":
                            logger.info(f"Received {msg_type} command")
                            await self.finish_streaming()
                            self.running = False

                            # finish의 경우 연결 유지, stop의 경우 종료
                            if msg_type == "stop":
                                break

                            # finish 후 새 세션 준비
                            self.init_streaming_state()
                            self.running = True

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by client")
        except Exception as e:
            logger.error(f"Error in handler: {e}")
            traceback.print_exc()
        finally:
            if self.running:
                await self.finish_streaming()
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
    parser.add_argument("--min-chunk-size", type=float, default=0.5,
                        help="Minimum chunk size in seconds")

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
        min_chunk_size_sec=args.min_chunk_size,
        host=args.host,
        port=args.port,
    )

    server = Qwen3ASRStreamingServer(config)
    server.init_model()

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
