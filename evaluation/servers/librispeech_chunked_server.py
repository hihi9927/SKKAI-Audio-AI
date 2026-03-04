#!/usr/bin/env python3
"""
Chunked Upload Server - Processes audio in fixed-size chunks
Simulates real-time input by waiting for chunk duration before processing
"""
from whisper_streaming.whisper_online_main import *

import sys
import argparse
import os
import logging
import numpy as np
import asyncio
import json
import time

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class ChunkedWebSocketHandler:
    """Handles WebSocket connection with chunked upload processing"""

    def __init__(self, websocket, online_asr_proc, chunk_size):
        self.websocket = websocket
        self.online_asr_proc = online_asr_proc
        self.chunk_size = chunk_size  # in seconds
        self.chunk_samples = int(chunk_size * SAMPLING_RATE)
        self.audio_buffer = []
        self.running = False
        self.current_utt_id = None
        self.last_chunk_time = None

    async def send_message(self, message_dict):
        """Send JSON message to client"""
        try:
            await self.websocket.send(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def send_result(self, iteration_output):
        """Send transcription result to client"""
        if iteration_output:
            start_ms = int(iteration_output['start'] * 1000)
            end_ms = int(iteration_output['end'] * 1000)
            text = iteration_output['text'].strip()

            if not text:
                logger.debug("Empty text in segment")
                return

            message = f"{start_ms} {end_ms} {text}"
            print(message, flush=True, file=sys.stderr)

            # Get detected language
            detected_lang = iteration_output.get('language', 'en')

            logger.info(f"Whisper output: {text}")
            logger.info(f"Detected language: {detected_lang}")

            # Send result message
            result_msg = {
                'type': 'final',
                'start': start_ms,
                'end': end_ms,
                'original': text,
                'language': detected_lang
            }

            # Add utterance_id for client matching
            if self.current_utt_id:
                result_msg['utt_id'] = self.current_utt_id

            await self.send_message(result_msg)
        else:
            logger.debug("No text in this segment")

    def convert_to_numpy(self, audio_data):
        """Convert Float32 binary data to numpy array"""
        try:
            audio_float = np.frombuffer(audio_data, dtype=np.float32)
            logger.debug(f"Received Float32 audio: {len(audio_float)} samples")
            return audio_float
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return None

    async def process_audio_chunk(self, audio_data):
        """Process incoming audio data in fixed-size chunks"""
        try:
            # Convert binary Float32 to numpy array
            audio = self.convert_to_numpy(audio_data)

            if audio is None or len(audio) == 0:
                logger.warning("Failed to convert audio or empty audio")
                return

            logger.debug(f"Received audio chunk: {len(audio)} samples")

            # Add to buffer
            self.audio_buffer.append(audio)

            # Calculate total samples in buffer
            total_samples = sum(len(x) for x in self.audio_buffer)

            # Process only when we have exactly one chunk worth of data
            while total_samples >= self.chunk_samples:
                # Simulate real-time by waiting for chunk duration
                current_time = time.time()
                if self.last_chunk_time is not None:
                    elapsed = current_time - self.last_chunk_time
                    wait_time = self.chunk_size - elapsed
                    if wait_time > 0:
                        logger.debug(f"Waiting {wait_time:.3f}s to simulate real-time chunk processing")
                        await asyncio.sleep(wait_time)

                # Extract exactly chunk_samples
                concatenated = np.concatenate(self.audio_buffer)
                chunk_to_process = concatenated[:self.chunk_samples]
                remaining = concatenated[self.chunk_samples:]

                # Update buffer with remaining data
                self.audio_buffer = [remaining] if len(remaining) > 0 else []
                total_samples = len(remaining)

                logger.info(f"Processing chunk: {len(chunk_to_process)} samples ({len(chunk_to_process)/SAMPLING_RATE:.2f}s)")

                # Insert audio into ASR
                self.online_asr_proc.insert_audio_chunk(chunk_to_process)

                # Process and get result
                result = self.online_asr_proc.process_iter()
                await self.send_result(result)

                self.last_chunk_time = time.time()

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """Main handler for WebSocket connection"""
        try:
            logger.info(f"New chunked WebSocket connection from {self.websocket.remote_address}")

            # Send hello message
            await self.send_message({
                'type': 'hello',
                'message': f'Connected to Whisper Chunked Server (chunk size: {self.chunk_size}s)',
                'mode': 'chunked',
                'chunk_size': self.chunk_size
            })

            # Initialize ASR
            self.online_asr_proc.init()
            self.running = True
            self.last_chunk_time = time.time()

            # Process incoming messages
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Binary data = audio
                    if self.running:
                        await self.process_audio_chunk(message)
                else:
                    # Text data = JSON command
                    try:
                        data = json.loads(message)
                        msg_type = data.get('type', '')

                        if msg_type == 'finish':
                            logger.info("Received finish command - processing remaining buffer")
                            # Process remaining audio in buffer
                            if self.audio_buffer:
                                remaining_audio = np.concatenate(self.audio_buffer)
                                if len(remaining_audio) > 0:
                                    logger.info(f"Processing final chunk: {len(remaining_audio)} samples")
                                    self.online_asr_proc.insert_audio_chunk(remaining_audio)
                                    result = self.online_asr_proc.process_iter()
                                    await self.send_result(result)
                                self.audio_buffer = []

                            # Flush ASR processor
                            result = self.online_asr_proc.finish()
                            if result:
                                await self.send_result(result)
                            logger.info("Buffer flushed")
                        elif msg_type == 'stop':
                            logger.info("Received stop command")
                            self.running = False
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message}")

        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("WebSocket connection closing, flushing ASR processor...")
            self.running = False

            try:
                self.audio_buffer = []
                if hasattr(self.online_asr_proc, 'finish'):
                    self.online_asr_proc.finish()
                    logger.info("ASR processor finished and flushed")
            except Exception as e:
                logger.error(f"Error flushing ASR processor: {e}")

            logger.info("WebSocket connection closed")


async def chunked_websocket_server(websocket, online_asr_proc, chunk_size):
    """WebSocket server handler for chunked mode"""
    handler = ChunkedWebSocketHandler(websocket, online_asr_proc, chunk_size)
    await handler.handle()


def main_chunked_server(factory, add_args):
    """
    Main WebSocket server entry point for chunked mode

    factory: function that creates the ASR and online processor object from args and logger.
    add_args: add specific args for the backend
    """
    import websockets

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--chunk-size", type=float, default=2.0,
                        help="Fixed chunk size in seconds for chunked processing")
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.")

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args, logger)

    # setting whisper object by args
    asr, online = asr_factory(args, factory)
    chunk_size = args.chunk_size

    # warm up the ASR
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.warmup(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. "+msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # Start WebSocket server
    async def server_handler(websocket):
        await chunked_websocket_server(websocket, online, chunk_size)

    async def main():
        logger.info(f'Starting Chunked WebSocket server on ws://{args.host}:{args.port}')
        logger.info(f'Chunk size: {chunk_size}s')

        async with websockets.serve(
            server_handler,
            args.host,
            args.port,
            ping_interval=None,
            ping_timeout=None,
            max_size=10 * 1024 * 1024
        ):
            logger.info(f'Chunked WebSocket server listening on ws://{args.host}:{args.port}')
            await asyncio.Future()

    asyncio.run(main())
