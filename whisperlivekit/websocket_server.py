#!/usr/bin/env python3
import asyncio
import json
import logging
import websockets
from deep_translator import GoogleTranslator

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class WebSocketHandler:
    """Handles WebSocket connection and ASR processing for one client"""

    def __init__(self, websocket, transcription_engine, audio_processor):
        self.websocket = websocket
        self.transcription_engine = transcription_engine
        self.audio_processor = audio_processor
        self.running = False

    async def send_message(self, message_dict):
        """Send JSON message to client"""
        try:
            logger.debug(f"[send_message] → Client: {message_dict}")
            await self.websocket.send(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def process_audio_chunk(self, audio_data):
        """Process incoming audio data"""
        try:
            logger.debug(f"Received audio chunk: {len(audio_data)} bytes")
            # Pass audio directly to AudioProcessor (PCM mode)
            await self.audio_processor.process_audio(audio_data)
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            import traceback
            traceback.print_exc()

    def parse_time_to_seconds(self, time_str):
        """Parse time string '0:00:21' to seconds"""
        if not time_str:
            return 0
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    async def translate_text(self, text, source_lang):
        """Translate text using Google Translator"""
        try:
            if not text or not text.strip():
                return ''

            loop = asyncio.get_event_loop()

            if source_lang == 'ko':
                # Korean → English
                result = await loop.run_in_executor(
                    None,
                    lambda: GoogleTranslator(source='ko', target='en').translate(text)
                )
            else:
                # English → Korean
                result = await loop.run_in_executor(
                    None,
                    lambda: GoogleTranslator(source='en', target='ko').translate(text)
                )

            return result or ''

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ''

    async def handle_results(self, results_generator):
        """Handle results from AudioProcessor - send FrontData directly to client"""
        try:
            TIME_WINDOW_SECONDS = 10  # Keep only last 10 seconds of lines

            async for response in results_generator:
                # Get the response as dict
                response_dict = response.to_dict()

                # Filter old lines based on time
                lines = response_dict.get('lines', [])
                if lines:
                    # Find the latest end time
                    latest_end_time = 0
                    for line in lines:
                        if line.get('end'):
                            end_time = self.parse_time_to_seconds(line['end'])
                            if end_time > latest_end_time:
                                latest_end_time = end_time

                    # Remove lines older than TIME_WINDOW_SECONDS
                    cutoff_time = latest_end_time - TIME_WINDOW_SECONDS
                    filtered_lines = []
                    for line in lines:
                        if line.get('end'):
                            end_time = self.parse_time_to_seconds(line['end'])
                            if end_time >= cutoff_time:
                                filtered_lines.append(line)
                        else:
                            filtered_lines.append(line)

                    if len(filtered_lines) < len(lines):
                        logger.info(f"Filtered {len(lines) - len(filtered_lines)} old lines (cutoff: {cutoff_time}s)")

                    response_dict['lines'] = filtered_lines

                    # Translate the combined text from filtered lines + buffer
                    all_texts = []
                    detected_lang = 'en'

                    for line in filtered_lines:
                        text = line.get('text', '').strip()
                        if text:
                            all_texts.append(text)
                            # Get language from the last non-empty line
                            if line.get('detected_language'):
                                detected_lang = line['detected_language']

                    # Add buffer text if present
                    buffer_text = response_dict.get('buffer_transcription', '').strip()
                    if buffer_text:
                        all_texts.append(buffer_text)

                    # Combine all texts
                    combined_text = ' '.join(all_texts)

                    # Translate the combined text
                    if combined_text:
                        translation = await self.translate_text(combined_text, detected_lang)
                        response_dict['translation'] = translation
                        logger.info(f"Translated ({detected_lang}): {combined_text[:50]}... → {translation[:50]}...")
                    else:
                        response_dict['translation'] = ''

                # Send filtered response to client
                await self.send_message(response_dict)
            logger.info("Results generator finished.")
        except Exception as e:
            logger.error(f"Error in results handler: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """Main handler for WebSocket connection"""
        try:
            logger.info(f"New WebSocket connection from {self.websocket.remote_address}")

            # Send hello message
            await self.send_message({
                'type': 'hello',
                'message': 'Connected to WhisperLiveKit Streaming Server'
            })

            # Create tasks for processing
            results_generator = await self.audio_processor.create_tasks()

            # Start background task to handle results
            results_task = asyncio.create_task(self.handle_results(results_generator))

            self.running = True

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

                        if msg_type == 'start':
                            logger.info("Received start command")
                            # Send ready message
                            await self.send_message({
                                'type': 'ready',
                                'message': 'Server ready to receive audio'
                            })

                        elif msg_type == 'finish':
                            logger.info("Received finish command")
                            # Stop processing
                            self.running = False
                            break

                        elif msg_type == 'stop':
                            logger.info("Received stop command")
                            self.running = False
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message}")

            # Cancel results task
            if not results_task.done():
                results_task.cancel()
                try:
                    await results_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("WebSocket connection closing, cleaning up...")
            self.running = False
            await self.audio_processor.cleanup()
            logger.info("WebSocket connection closed")


async def websocket_server(websocket, transcription_engine):
    """WebSocket server handler"""
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)
    handler = WebSocketHandler(websocket, transcription_engine, audio_processor)
    await handler.handle()


async def main_server(args):
    """Main server entry point"""
    logger.info("Initializing transcription engine...")

    # Force pcm_input mode since client sends raw PCM data
    args.pcm_input = True

    transcription_engine = TranscriptionEngine(**vars(args))
    logger.info("Transcription engine initialized")

    async def server_handler(websocket):
        await websocket_server(websocket, transcription_engine)

    logger.info(f'Starting WebSocket server on ws://{args.host}:{args.port}')

    async with websockets.serve(
        server_handler,
        args.host,
        args.port,
        ping_interval=None,  # Disable ping/pong for continuous audio streaming
        ping_timeout=None,
        max_size=10 * 1024 * 1024  # 10MB max message size
    ):
        logger.info(f'WebSocket server listening on ws://{args.host}:{args.port}')
        await asyncio.Future()  # run forever


def main():
    """Entry point for the CLI command."""
    args = parse_args()

    # Override host and port defaults for WebSocket server
    if not hasattr(args, 'host') or args.host is None:
        args.host = '0.0.0.0'
    if not hasattr(args, 'port') or args.port is None:
        args.port = 8001

    logger.setLevel(args.log_level if hasattr(args, 'log_level') else logging.INFO)

    asyncio.run(main_server(args))


if __name__ == "__main__":
    main()
