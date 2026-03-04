#!/usr/bin/env python3
"""
Whisper Original Server - Processes complete audio file after receiving all data
Simulates real-time input by waiting for full audio duration before processing
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


class OriginalWebSocketHandler:
    """Handles WebSocket connection with full-file processing (Whisper original mode)"""

    def __init__(self, websocket, online_asr_proc):
        self.websocket = websocket
        self.online_asr_proc = online_asr_proc
        self.audio_buffer = []
        self.running = False
        self.current_utt_id = None
        self.start_time = None
        self.total_samples = 0

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

    async def collect_audio_chunk(self, audio_data):
        """Collect incoming audio data without processing"""
        try:
            # Convert binary Float32 to numpy array
            audio = self.convert_to_numpy(audio_data)

            if audio is None or len(audio) == 0:
                logger.warning("Failed to convert audio or empty audio")
                return

            logger.debug(f"Received audio chunk: {len(audio)} samples")

            # Just add to buffer without processing
            self.audio_buffer.append(audio)
            self.total_samples += len(audio)

            # Record start time on first chunk
            if self.start_time is None:
                self.start_time = time.time()

            logger.info(f"Collected {self.total_samples} samples ({self.total_samples/SAMPLING_RATE:.2f}s total)")

        except Exception as e:
            logger.error(f"Error collecting audio chunk: {e}")
            import traceback
            traceback.print_exc()

    async def process_complete_audio(self):
        """Process all collected audio at once (Whisper original mode)"""
        try:
            if not self.audio_buffer:
                logger.warning("No audio to process")
                return

            # Concatenate all audio chunks
            complete_audio = np.concatenate(self.audio_buffer)
            audio_duration = len(complete_audio) / SAMPLING_RATE

            logger.info(f"Complete audio: {len(complete_audio)} samples ({audio_duration:.2f}s)")

            # Simulate real-time: wait for the duration of the audio
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                wait_time = audio_duration - elapsed
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f}s to simulate real-time (audio duration: {audio_duration:.2f}s, elapsed: {elapsed:.2f}s)")
                    await asyncio.sleep(wait_time)
                else:
                    logger.info(f"No wait needed (audio duration: {audio_duration:.2f}s, elapsed: {elapsed:.2f}s)")

            # Process all audio at once
            logger.info("Processing complete audio file...")
            self.online_asr_proc.insert_audio_chunk(complete_audio)

            # Get result
            result = self.online_asr_proc.process_iter()
            await self.send_result(result)

            # Flush to get final result
            final_result = self.online_asr_proc.finish()
            if final_result:
                await self.send_result(final_result)

            logger.info("Complete audio processing finished")

        except Exception as e:
            logger.error(f"Error processing complete audio: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """Main handler for WebSocket connection"""
        try:
            logger.info(f"New original mode WebSocket connection from {self.websocket.remote_address}")

            # Send hello message
            await self.send_message({
                'type': 'hello',
                'message': 'Connected to Whisper Original Server (full file processing)',
                'mode': 'original'
            })

            # Initialize ASR
            self.online_asr_proc.init()
            self.running = True

            # Process incoming messages
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Binary data = audio - just collect it
                    if self.running:
                        await self.collect_audio_chunk(message)
                else:
                    # Text data = JSON command
                    try:
                        data = json.loads(message)
                        msg_type = data.get('type', '')

                        if msg_type == 'finish':
                            logger.info("Received finish command - processing complete audio")
                            await self.process_complete_audio()
                            self.audio_buffer = []
                            self.total_samples = 0
                            self.start_time = None
                            logger.info("Complete audio processed")
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
            logger.info("WebSocket connection closing...")
            self.running = False

            try:
                self.audio_buffer = []
                if hasattr(self.online_asr_proc, 'finish'):
                    self.online_asr_proc.finish()
                    logger.info("ASR processor finished")
            except Exception as e:
                logger.error(f"Error finishing ASR processor: {e}")

            logger.info("WebSocket connection closed")


async def original_websocket_server(websocket, online_asr_proc):
    """WebSocket server handler for original mode"""
    handler = OriginalWebSocketHandler(websocket, online_asr_proc)
    await handler.handle()


def main_original_server(factory, add_args):
    """
    Main WebSocket server entry point for Whisper original mode

    factory: function that creates the ASR and online processor object from args and logger.
    add_args: add specific args for the backend
    """
    import websockets

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.")

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args, logger)

    # setting whisper object by args
    asr, online = asr_factory(args, factory)

    # warm up the ASR
    msg = "Whisper is not warmed up. The first processing may take longer."
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
        await original_websocket_server(websocket, online)

    async def main():
        logger.info(f'Starting Original Mode WebSocket server on ws://{args.host}:{args.port}')

        async with websockets.serve(
            server_handler,
            args.host,
            args.port,
            ping_interval=None,
            ping_timeout=None,
            max_size=50 * 1024 * 1024  # 50MB max for full file
        ):
            logger.info(f'Original Mode WebSocket server listening on ws://{args.host}:{args.port}')
            await asyncio.Future()

    asyncio.run(main())
