# coding=utf-8
"""
Qwen3-ASR Real-time WebSocket Server (Transformers Backend)

vLLM이 없는 환경에서 transformers 백엔드를 사용하는 pseudo-streaming 서버.
주기적으로 누적된 오디오 전체를 transcribe하여 실시간처럼 동작.

Usage:
    python streaming_websocket_server_transformers.py --host 0.0.0.0 --port 8765

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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

try:
    import websockets
except ImportError:
    raise ImportError("websockets 패키지가 필요합니다: pip install websockets")

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import warmup_streaming

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
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512

    # 스트리밍 설정 (pseudo-streaming)
    transcribe_interval_sec: float = 2.0  # transcribe 호출 간격 (초)
    min_audio_sec: float = 0.5  # 최소 오디오 길이 (초)

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class AudioBuffer:
    """오디오 버퍼 관리"""
    sample_rate: int = SAMPLING_RATE
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    total_samples: int = 0
    total_duration: float = 0.0

    def add_pcm_bytes(self, data: bytes) -> None:
        """PCM s16le 바이트를 버퍼에 추가"""
        if len(data) == 0:
            return
        pcm_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, pcm_array])
        self.total_samples = len(self.buffer)
        self.total_duration = self.total_samples / self.sample_rate

    def get_all(self) -> np.ndarray:
        """버퍼의 모든 데이터 반환 (복사)"""
        return self.buffer.copy()

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


class Qwen3ASRTransformersHandler:
    """WebSocket 연결 당 하나의 핸들러 (Transformers 백엔드)"""

    def __init__(self, websocket, asr_model: Qwen3ASRModel, config: StreamingConfig):
        self.websocket = websocket
        self.asr = asr_model
        self.config = config
        self.audio_buffer = AudioBuffer()
        self.running = False
        self.last_text = ""
        self.last_final_text = ""

        # 클라이언트 옵션
        self.client_lang = "auto"
        self.client_translate = False
        self.client_polish = False

        # 타임스탬프
        self.segment_start_time = 0.0
        self.last_transcribe_time = 0.0

        # transcribe 간격 (샘플 수)
        self.transcribe_interval_samples = int(config.transcribe_interval_sec * SAMPLING_RATE)
        self.min_audio_samples = int(config.min_audio_sec * SAMPLING_RATE)

        # 마지막 transcribe 시점의 샘플 수
        self.last_transcribed_samples = 0

    async def send_message(self, msg_type: str, **kwargs):
        """JSON 메시지 전송"""
        message = {"type": msg_type, **kwargs}
        try:
            await self.websocket.send(json.dumps(message, ensure_ascii=False))
            logger.debug(f"Sent: {msg_type}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def reset_state(self):
        """상태 초기화"""
        self.audio_buffer.clear()
        self.last_text = ""
        self.last_final_text = ""
        self.segment_start_time = 0.0
        self.last_transcribe_time = 0.0
        self.last_transcribed_samples = 0
        logger.info("State reset")

    async def process_audio_chunk(self, audio_data: bytes):
        """오디오 청크 처리"""
        self.audio_buffer.add_pcm_bytes(audio_data)
        current_samples = self.audio_buffer.total_samples

        # transcribe 간격만큼 새 오디오가 쌓였는지 확인
        samples_since_last = current_samples - self.last_transcribed_samples

        if samples_since_last >= self.transcribe_interval_samples and current_samples >= self.min_audio_samples:
            await self._do_transcribe()
            self.last_transcribed_samples = current_samples

    async def _do_transcribe(self):
        """누적된 오디오 전체를 transcribe"""
        audio_data = self.audio_buffer.get_all()
        if len(audio_data) < self.min_audio_samples:
            return

        try:
            # Language hint 처리
            language = None
            if self.client_lang and self.client_lang != "auto":
                lang_map = {"en": "English", "ko": "Korean", "zh": "Chinese", "ja": "Japanese"}
                language = lang_map.get(self.client_lang, None)

            # Transcribe (전체 오디오)
            results = self.asr.transcribe(
                audio=(audio_data, SAMPLING_RATE),
                language=language,
                return_time_stamps=False,
            )

            if results and len(results) > 0:
                current_text = (results[0].text or "").strip()
                current_lang = results[0].language or ""

                # 텍스트가 변경되었을 때만 전송
                if current_text and current_text != self.last_text:
                    await self.send_message(
                        "partial",
                        original=current_text,
                        last_translation="",
                        language=current_lang
                    )
                    self.last_text = current_text
                    logger.info(f"[partial] lang={current_lang} text={current_text[:50]}...")

        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            import traceback
            traceback.print_exc()

    async def finish_streaming(self):
        """스트리밍 종료 처리"""
        audio_data = self.audio_buffer.get_all()
        if len(audio_data) < 100:  # 너무 짧으면 스킵
            return

        try:
            language = None
            if self.client_lang and self.client_lang != "auto":
                lang_map = {"en": "English", "ko": "Korean", "zh": "Chinese", "ja": "Japanese"}
                language = lang_map.get(self.client_lang, None)

            results = self.asr.transcribe(
                audio=(audio_data, SAMPLING_RATE),
                language=language,
                return_time_stamps=False,
            )

            if results and len(results) > 0:
                final_text = (results[0].text or "").strip()
                final_lang = results[0].language or ""

                if final_text and final_text != self.last_final_text:
                    await self.send_message(
                        "final",
                        start=format_time(self.segment_start_time),
                        end=format_time(self.audio_buffer.total_duration),
                        original=final_text,
                        translation="",
                        language=final_lang
                    )
                    self.last_final_text = final_text
                    logger.info(f"[final] lang={final_lang} text={final_text}")

        except Exception as e:
            logger.error(f"Final transcribe error: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """WebSocket 연결 핸들링"""
        try:
            remote_addr = self.websocket.remote_address
            logger.info(f"New connection from {remote_addr}")

            await self.send_message("hello", message="Qwen3-ASR Transformers Server")

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
                            self.client_translate = data.get("translate", False)
                            self.client_polish = data.get("polish", False)

                            logger.info(f"Received start: lang={self.client_lang}, "
                                       f"translate={self.client_translate}, polish={self.client_polish}")

                            self.reset_state()
                            self.running = True

                            await self.send_message("ready", message="Ready to receive audio")

                        elif msg_type == "stop" or msg_type == "finish":
                            logger.info(f"Received {msg_type} command")
                            await self.finish_streaming()
                            self.running = False

                            if msg_type == "stop":
                                break

                            # finish 후 새 세션 준비
                            self.reset_state()
                            self.running = True

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by client")
        except Exception as e:
            logger.error(f"Error in handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.running:
                await self.finish_streaming()
            logger.info("Connection closed")


class Qwen3ASRTransformersServer:
    """Qwen3-ASR 서버 (Transformers 백엔드)"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.asr = None

    def init_model(self):
        """ASR 모델 초기화"""
        logger.info(f"Loading model: {self.config.model_path}")
        logger.info(f"Device: {self.config.device}, dtype: {self.config.dtype}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self.asr = Qwen3ASRModel.from_pretrained(
            self.config.model_path,
            dtype=dtype,
            device_map=self.config.device,
            max_inference_batch_size=1,
            max_new_tokens=self.config.max_new_tokens,
        )
        warmup_streaming(self.asr)
        logger.info("Model loaded successfully")

    async def handle_connection(self, websocket):
        """각 연결 처리"""
        handler = Qwen3ASRTransformersHandler(websocket, self.asr, self.config)
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
            max_size=10 * 1024 * 1024,
        ):
            logger.info(f"Server listening on ws://{self.config.host}:{self.config.port}")
            await asyncio.Future()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR WebSocket Server (Transformers)")

    # 모델 설정
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B",
                        help="Model path or name")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens")

    # 스트리밍 설정
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Transcribe interval in seconds")
    parser.add_argument("--min-audio", type=float, default=0.5,
                        help="Minimum audio length in seconds")

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
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        transcribe_interval_sec=args.interval,
        min_audio_sec=args.min_audio,
        host=args.host,
        port=args.port,
    )

    server = Qwen3ASRTransformersServer(config)
    server.init_model()

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
