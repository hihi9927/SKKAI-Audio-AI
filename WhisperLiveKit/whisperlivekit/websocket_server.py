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

        # Translation buffer: accumulate text segments until sentence is complete
        self.translation_buffer = []  # List of {text, start, end, language} dicts
        self.last_translation = ''  # Last completed translation for showing in partials
        self.last_translation_lang = None  # Language of last translation
        self.detected_language = None  # Current detected language

        # Track sent sentences to avoid duplicates (use set for O(1) lookup)
        self.sent_sentences = set()  # Set of sentence strings we've already sent
        self.connection_initialized = False  # Track if we've initialized for this connection

    def initialize_buffers(self):
        """Initialize/clear all buffers for a new connection session"""
        logger.info("[Buffer Init] Clearing all buffers for new connection session")
        self.translation_buffer = []
        self.last_translation = ''
        self.last_translation_lang = None
        self.detected_language = None
        self.sent_sentences.clear()  # Clear the set of sent sentences
        self.connection_initialized = True

        # Clear the transcription backend's internal state if available
        if hasattr(self.audio_processor, 'transcription') and self.audio_processor.transcription:
            if hasattr(self.audio_processor.transcription, 'init_context'):
                logger.info("[Buffer Init] Resetting transcription backend context")
                self.audio_processor.transcription.init_context()

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
            # Initialize buffers on first audio chunk
            if not self.connection_initialized:
                self.initialize_buffers()

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

    def is_sentence_complete(self, text):
        """Check if text ends with sentence-ending punctuation"""
        if not text:
            return False
        text = text.strip()
        sentence_end_punctuation = {'.', '!', '?', '。', '！', '？'}
        return any(text.endswith(p) for p in sentence_end_punctuation)

    def split_into_sentences(self, text):
        """Split text into individual sentences"""
        import re
        # Split by sentence-ending punctuation (including ellipsis ...)
        # Matches: . ! ? 。 ！ ？ or ... followed by space
        # Pattern: sentence-ending punctuation + optional space
        pattern = r'([.]{3}|[.!?。！？])\s+'
        parts = re.split(pattern, text)

        sentences = []
        current = ''

        i = 0
        while i < len(parts):
            part = parts[i]

            # Skip empty parts
            if not part or not part.strip():
                i += 1
                continue

            # Check if this looks like punctuation
            if part in ['.', '!', '?', '。', '！', '？', '...']:
                current += part
                sentences.append(current.strip())
                current = ''
                i += 1
            else:
                # This is text content
                current += part
                # Check if next part is punctuation
                if i + 1 < len(parts) and parts[i + 1] in ['.', '!', '?', '。', '！', '？', '...']:
                    current += parts[i + 1]
                    sentences.append(current.strip())
                    current = ''
                    i += 2
                else:
                    i += 1

        # Add any remaining text (incomplete sentence)
        if current.strip():
            sentences.append(current.strip())

        return [s for s in sentences if s]  # Filter empty strings

    async def flush_translation_buffer(self, trigger_reason="unknown"):
        """Flush translation buffer and send as 'final' message"""
        if not self.translation_buffer:
            return

        logger.info(f"[Translation] Flushing buffer ({len(self.translation_buffer)} segments) - reason: {trigger_reason}")

        # Combine all buffered text
        full_text = ' '.join(seg['text'] for seg in self.translation_buffer)
        first_start = self.translation_buffer[0]['start']
        last_end = self.translation_buffer[-1]['end']
        language = self.translation_buffer[0].get('language', 'ko')

        # Translate the complete text
        translation = await self.translate_text(full_text, language)
        self.last_translation = translation
        self.last_translation_lang = 'en' if language == 'ko' else 'ko'

        logger.info(f"[Translation] Complete - {language}: {full_text[:50]}... → {translation[:50]}...")

        # Send as 'final' message (complete sentence)
        final_msg = {
            'type': 'final',
            'start': first_start,
            'end': last_end,
            'original': full_text,
            'translation': translation,
            'language': language
        }

        await self.send_message(final_msg)

        # Clear the buffer
        self.translation_buffer = []
        logger.info(f"[Translation] Buffer cleared after {trigger_reason}")

    async def handle_results(self, results_generator):
        """Handle results from AudioProcessor - SimulStreaming style (partial vs final)"""
        try:
            async for response in results_generator:
                # Get the response as dict
                response_dict = response.to_dict()

                # Get lines and buffer
                lines = response_dict.get('lines', [])
                buffer_text = response_dict.get('buffer_transcription', '').strip()

                # Detect language from lines
                detected_lang = None
                for line in lines:
                    if line and line.get('detected_language'):
                        detected_lang = line['detected_language']
                        break

                # Don't default to 'ko' - wait for actual language detection
                # This prevents sending wrong language transcriptions as 'final'

                # Update detected language
                if detected_lang and self.detected_language != detected_lang:
                    if self.detected_language is not None:
                        logger.info(f"[Language] Changed from {self.detected_language} to {detected_lang} - clearing buffer")
                        self.translation_buffer = []
                        self.sent_sentences.clear()  # Clear sent sentences on language change
                    self.detected_language = detected_lang

                # Process complete lines (ONLY if language is detected)
                # This prevents sending wrong-language transcriptions as 'final'
                if lines and detected_lang:
                    # Only look at the last line (most recent)
                    last_line = lines[-1]

                    if last_line and last_line.get('text'):
                        text = last_line['text'].strip()

                        if text and self.is_sentence_complete(text):
                            # This line has sentence-ending punctuation
                            # It might contain multiple sentences - split them
                            sentences = self.split_into_sentences(text)

                            # Find new sentences that haven't been sent yet
                            for sentence in sentences:
                                # Skip incomplete sentences (no ending punctuation)
                                if not self.is_sentence_complete(sentence):
                                    logger.debug(f"[Incomplete] Skipping: {sentence[:30]}...")
                                    continue

                                # Check if we already sent this exact sentence (set lookup - O(1))
                                if sentence in self.sent_sentences:
                                    # Silently skip duplicates (no log to reduce noise)
                                    continue

                                # New complete sentence - send it
                                logger.info(f"[New Complete Sentence] {sentence[:50]}...")

                                self.translation_buffer.append({
                                    'text': sentence,
                                    'start': last_line.get('start', ''),
                                    'end': last_line.get('end', ''),
                                    'language': detected_lang
                                })

                                # Mark this sentence as sent
                                self.sent_sentences.add(sentence)
                                await self.flush_translation_buffer("sentence_complete")

                # Send partial for current incomplete text (buffer)
                # Only send if language is detected
                if buffer_text and detected_lang:
                    # Send partial message (no translation yet)
                    partial_msg = {
                        'type': 'partial',
                        'original': buffer_text,
                        'last_translation': self.last_translation,  # Show last completed translation
                        'language': detected_lang
                    }

                    await self.send_message(partial_msg)
                    logger.debug(f"[Partial] Sent: {buffer_text[:50]}...")

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
