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
        self.last_sent_buffer = ''  # Track last sent partial buffer to prevent duplicates
        self.last_logged_state = ''  # Track last logged state to reduce noise

    def initialize_buffers(self):
        """Initialize/clear all buffers for a new connection session"""
        logger.info("[Buffer Init] Clearing all buffers for new connection session")
        self.translation_buffer = []
        self.last_translation = ''
        self.last_translation_lang = None
        self.detected_language = None
        self.sent_sentences.clear()  # Clear the set of sent sentences
        self.last_sent_buffer = ''  # Clear last sent buffer
        self.last_logged_state = ''  # Clear last logged state
        self.connection_initialized = True

        # DO NOT clear AudioProcessor state - it's managed by the transcription engine
        # The buffer_transcription updates automatically with new audio
        # Old state gets cleared by refresh_segment() in simul_whisper when needed

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
        """
        번역 버퍼 flush 및 잔여 버퍼 재구성 - PDF 부록 A3.2 구현

        커밋 시점에 호출되어:
        1) 커밋 구간 확정 및 출력 고정
        2) 번역 버퍼 flush (확정 구간만 번역)
        3) 잔여 버퍼 분리 및 재구성
        4) 상태 동기화
        """
        if not self.translation_buffer:
            return

        logger.info(f"[Translation Flush - PDF A3.2] Flushing buffer ({len(self.translation_buffer)} segments) - reason: {trigger_reason}")

        # 1) 커밋 구간 확정 및 출력 고정
        full_text = ' '.join(seg['text'] for seg in self.translation_buffer)
        first_start = self.translation_buffer[0]['start']
        last_end = self.translation_buffer[-1]['end']
        language = self.translation_buffer[0].get('language', 'ko')

        logger.info(f"[Commit] Fixed output: '{full_text}' (no rewrite after this)")

        # 2) 번역 버퍼 flush (확정 구간만 번역)
        translation = await self.translate_text(full_text, language)
        self.last_translation = translation
        self.last_translation_lang = 'en' if language == 'ko' else 'ko'

        logger.info(f"[Translation Complete] {language}: {full_text[:50]}... → {translation[:50]}...")

        # Send as 'final' message (complete sentence)
        final_msg = {
            'type': 'final',
            'start': first_start,
            'end': last_end,
            'original': full_text,
            'translation': translation,
            'language': language,
            'commit_reason': trigger_reason  # 커밋 사유 추가 (디버깅용)
        }

        await self.send_message(final_msg)

        # 3) 잔여 버퍼 분리 및 재구성
        # 현재 구현에서는 translation_buffer를 완전히 비우고,
        # 다음 입력을 위해 clean slate로 시작
        self.translation_buffer = []

        # 4) 상태 동기화
        # 과도한 문맥 누적 방지 - 최소한의 컨텍스트만 유지
        # (현재 구조에서는 이미 TranscriptionEngine에서 처리됨)

        logger.info(f"[Buffer Management] Buffer cleared and reset for next segment")

    async def handle_results(self, results_generator):
        """Handle results from AudioProcessor - SimulStreaming style (partial vs final)"""
        try:
            async for response in results_generator:
                # Get the response as dict
                response_dict = response.to_dict()

                # Get lines and buffer
                lines = response_dict.get('lines', [])
                buffer_text = response_dict.get('buffer_transcription', '').strip()

                # DEBUG: Log only when content changes (reduce noise)
                current_state = f"{len(lines)}:{buffer_text[:50] if buffer_text else ''}"
                if lines:
                    current_state += ':' + '|'.join(l.get('text', '')[:30] for l in lines)
                if current_state != self.last_logged_state:
                    logger.debug(f"[WebSocket] Received lines={len(lines)}, buffer='{buffer_text[:50] if buffer_text else ''}'")
                    if lines:
                        for idx, line in enumerate(lines):
                            logger.debug(f"[WebSocket] Line {idx}: text='{line.get('text', '')[:50]}', lang={line.get('detected_language')}")
                    self.last_logged_state = current_state

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
                    # Filter out empty lines and get the last non-empty line
                    non_empty_lines = [line for line in lines if line.get('text', '').strip()]

                    if non_empty_lines:
                        last_line = non_empty_lines[-1]
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
                # Send partial even if language is not detected yet (language detection takes ~2 seconds)
                # CRITICAL: Only send if buffer content has CHANGED to prevent infinite loop
                if buffer_text and buffer_text != self.last_sent_buffer:
                    # Send partial message (no translation yet)
                    partial_msg = {
                        'type': 'partial',
                        'original': buffer_text,
                        'last_translation': self.last_translation,  # Show last completed translation
                        'language': self.detected_language or 'unknown'  # Use 'unknown' if not detected yet
                    }

                    await self.send_message(partial_msg)
                    logger.debug(f"[Partial] Sent: {buffer_text[:50]}...")
                    self.last_sent_buffer = buffer_text  # Track this buffer to prevent duplicates

            logger.info("Results generator finished.")
        except Exception as e:
            logger.error(f"Error in results handler: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """Main handler for WebSocket connection"""
        results_task = None
        try:
            logger.info(f"New WebSocket connection from {self.websocket.remote_address}")

            # CRITICAL: Initialize buffers at the START of every connection
            # This ensures old state doesn't persist across connections
            self.initialize_buffers()

            # Send hello message
            await self.send_message({
                'type': 'hello',
                'message': 'Connected to WhisperLiveKit Streaming Server'
            })

            self.running = True
            tasks_created = False

            # Process incoming messages
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Binary data = audio
                    if self.running and tasks_created:
                        await self.process_audio_chunk(message)
                else:
                    # Text data = JSON command
                    try:
                        data = json.loads(message)
                        msg_type = data.get('type', '')

                        if msg_type == 'start':
                            # 클라이언트가 보낸 audioFormat에 따라 처리 모드 설정
                            audio_format = data.get('audioFormat', 'pcm')
                            is_pcm = (audio_format == 'pcm')
                            self.audio_processor.set_pcm_input(is_pcm)

                            # 모바일 클라이언트의 경우 VAD 비활성화 옵션
                            # (청크 녹음 방식은 오디오 갭이 발생하여 VAD 오작동)
                            disable_vad = data.get('disableVAD', False)
                            if disable_vad:
                                self.audio_processor.vac = None
                                logger.info("VAD disabled by client request (chunked audio mode)")

                            logger.info(f"Received start command (audioFormat={audio_format}, pcm_input={is_pcm}, disableVAD={disable_vad})")

                            # Create tasks for processing (after knowing the audio format)
                            if not tasks_created:
                                results_generator = await self.audio_processor.create_tasks()
                                results_task = asyncio.create_task(self.handle_results(results_generator))
                                tasks_created = True

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
            if results_task and not results_task.done():
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

    # Audio format is now auto-detected per connection via client's start message
    # Default to PCM for backward compatibility (browser clients)
    if not hasattr(args, 'pcm_input'):
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
