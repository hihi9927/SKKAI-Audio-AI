#!/usr/bin/env python3
from whisper_streaming.whisper_online_main import *

import sys
import argparse
import os
import logging
import numpy as np
import asyncio
import json
from deep_translator import GoogleTranslator
import spacy

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

# Initialize spaCy models as None (loaded only when --use-spacy is specified)
nlp_ko = None
nlp_en = None


def load_spacy_models():
    """Load spaCy models for sentence boundary detection (optional feature)."""
    global nlp_ko, nlp_en

    try:
        nlp_ko = spacy.load("ko_core_news_sm")
        logger.info("Loaded Korean spaCy model for sentence boundary detection")
    except OSError:
        logger.warning("Korean spaCy model not found. Install with: python -m spacy download ko_core_news_sm")
        nlp_ko = None

    try:
        nlp_en = spacy.load("en_core_web_sm")
        logger.info("Loaded English spaCy model for sentence boundary detection")
    except OSError:
        logger.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        nlp_en = None


def load_denoiser_model():
    """Load SpeechBrain MetricGAN+ denoiser once at server startup."""
    try:
        import torch  # noqa: F401
        from speechbrain.inference import SpectralMaskEnhancement
        from speechbrain.utils.fetching import LocalStrategy
        logger.info("[Denoiser] Loading MetricGAN+ model (one-time)...")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with COPY strategy (no symlinks - Windows compatible)
        model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY  # Use COPY instead of SYMLINK for Windows
        )
        logger.info(f"[Denoiser] Model loaded successfully on {device}!")
        return model
    except ImportError:
        logger.error("[Denoiser] Failed to import speechbrain. Install with: pip install speechbrain")
    except Exception as e:
        logger.error(f"[Denoiser] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    return None


class WebSocketHandler:
    """Handles WebSocket connection and ASR processing for one client"""

    def __init__(self, websocket, online_asr_proc, min_chunk, use_denoiser=False, denoiser_model=None):
        self.websocket = websocket
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.is_first = True
        self.audio_buffer = []
        self.running = False

        # Translation buffer: accumulate text segments until sentence break
        self.translation_buffer = []  # List of text segments
        self.detected_language = None  # Store detected language
        self.nonvoice_count = 0  # Count consecutive nonvoice detections for VAD-based flush
        self.display_mode = 'both'  # Display mode: 'translateOnly', 'transcriptOnly', 'both'
        self.language_hint = 'auto'  # Language hint: 'auto', 'ko', 'en'

        # Store last completed translation for showing in partial results
        self.last_translation = ''  # Last completed translation text
        self.last_translation_lang = None  # Language of last translation (ko or en)

        # Hallucination detection
        self.recent_tokens = []  # Track recent tokens for repetition detection
        self.hallucination_phrases = [
            'speaking in foreign language', 'speaking korean', 'speaking english',
            'speaking in korean', 'speaking in english',
            'thank you for watching', 'thanks for watching',
            '자막', '구독', '좋아요', '알림 설정'  # Common YouTube artifacts
        ]

        # Denoiser setup
        self.use_denoiser = use_denoiser
        self.denoiser_model = denoiser_model if self.use_denoiser else None
        if self.use_denoiser and self.denoiser_model is None:
            logger.error("[Denoiser] Denoiser not available, disabling denoise.")
            self.use_denoiser = False

    async def send_message(self, message_dict):
        """Send JSON message to client"""
        try:
            await self.websocket.send(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def flush_translation_buffer(self, trigger_reason="unknown"):
        """Flush translation buffer and send translation"""
        if not self.translation_buffer:
            return

        logger.info(f"[Translation] Flushing buffer ({len(self.translation_buffer)} segments) - reason: {trigger_reason}")

        # Combine all buffered text
        full_text = ' '.join(seg['text'] for seg in self.translation_buffer)
        first_start = self.translation_buffer[0]['start']
        last_end = self.translation_buffer[-1]['end']

        # transcriptOnly mode: don't translate, just send original text
        if self.display_mode == 'transcriptOnly':
            logger.info(f"[transcriptOnly] Skipping translation, sending original text only")
            result_msg = {
                'type': 'final',
                'start': first_start,
                'end': last_end,
                'original': full_text,
                'polished': full_text,  # Same as original in transcriptOnly mode
                'language': self.detected_language or 'en'
            }
            await self.send_message(result_msg)
            self.translation_buffer = []
            logger.info(f"[transcriptOnly] Buffer cleared after {trigger_reason}")
            return

        # Translate the complete text (for translateOnly and both modes)
        lang, ko_text, en_text = await self.detect_and_translate(full_text, self.detected_language, None)

        logger.info(f"[Translation] Result - lang: {lang}, ko: {ko_text}, en: {en_text}")

        # polished는 번역 결과 (translateOnly 모드에서는 번역 실패 시 빈 문자열)
        if self.display_mode == 'translateOnly':
            # translateOnly mode: only show translation, empty if translation fails
            if lang == 'ko':
                # Check if translation succeeded (not None and not same as original)
                if en_text and en_text.strip() != full_text.strip():
                    polished = en_text
                else:
                    polished = ''
                    logger.warning(f"[Translation] EN translation failed or same as original")
            else:
                # Check if translation succeeded (not None and not same as original)
                if ko_text and ko_text.strip() != full_text.strip():
                    polished = ko_text
                else:
                    polished = ''
                    logger.warning(f"[Translation] KO translation failed or same as original")
        else:
            # Normal mode (both): show translation or fallback to original
            polished = full_text
            if lang == 'ko' and en_text:
                polished = en_text
            elif lang == 'en' and ko_text:
                polished = ko_text

        # Send the complete sentence with translation
        # In translateOnly mode, don't send original text (hide source language)
        if self.display_mode == 'translateOnly':
            result_msg = {
                'type': 'final',
                'start': first_start,
                'end': last_end,
                'original': '',  # Don't show original in translateOnly mode
                'polished': polished,
                'language': lang
            }
        else:
            result_msg = {
                'type': 'final',
                'start': first_start,
                'end': last_end,
                'original': full_text,
                'polished': polished,
                'language': lang
            }

        if ko_text:
            result_msg['ko'] = ko_text
        if en_text:
            result_msg['en'] = en_text

        await self.send_message(result_msg)

        # Store last translation for showing in partial results
        if lang == 'ko' and en_text:
            self.last_translation = en_text
            self.last_translation_lang = 'en'
        elif lang == 'en' and ko_text:
            self.last_translation = ko_text
            self.last_translation_lang = 'ko'
        else:
            # No translation available
            self.last_translation = ''
            self.last_translation_lang = None

        # Clear the buffer
        self.translation_buffer = []
        # Reset token tracking after successful translation
        self.recent_tokens = []
        logger.info(f"[Translation] Buffer cleared after {trigger_reason}")

    def clean_special_characters(self, text):
        """Remove unwanted special characters from text"""
        import re

        # Remove brackets and their contents (e.g., [Music], [Applause])
        text = re.sub(r'\[.*?\]', '', text)

        # Remove diamond with question mark (�) and other common unicode errors
        text = text.replace('�', '')

        # Remove other problematic characters
        text = text.replace('♪', '')  # Music notes
        text = text.replace('♫', '')  # Music notes

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def is_hallucination(self, text):
        """Detect if text is likely a hallucination
        Returns: (is_hallucination: bool, reason: str)"""

        text_lower = text.lower().strip()

        # Check for brackets (often contains noise like [Music], [Applause])
        if '[' in text or ']' in text:
            logger.warning(f"[Hallucination] Detected brackets in text: '{text}'")
            return True, "brackets"

        # Check for known hallucination phrases
        for phrase in self.hallucination_phrases:
            if phrase in text_lower:
                logger.warning(f"[Hallucination] Detected meta phrase: '{phrase}' in '{text}'")
                return True, f"meta_phrase:{phrase}"

        # Check for repeated phrase patterns
        # Split by punctuation (including comma) to find repeated phrases
        import re
        sentences = re.split(r'[.!?,。!?]\s*', text)  # Added comma to split pattern
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 3:
            # Check if we have the same sentence repeated 3+ times
            from collections import Counter
            sentence_counts = Counter(sentences)
            most_common_sentence, count = sentence_counts.most_common(1)[0]

            if count >= 3:
                logger.warning(f"[Hallucination] Detected repeated phrase (x{count}): '{most_common_sentence}'")
                return True, f"repeated_phrase:x{count}"

        # Split into words and check for repetition
        words = text.split()

        if len(words) < 3:
            return False, None

        # Check last 10 words for unique count
        recent_words = words[-10:]
        unique_words = set(recent_words)

        if len(unique_words) < 2:
            logger.warning(f"[Hallucination] Too few unique words in recent segment: {unique_words}")
            return True, "low_unique_words"

        # Check for consecutive identical words (e.g., "1 1 1 1 1")
        consecutive_count = 1
        max_consecutive = 1

        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1

        if max_consecutive >= 5:
            logger.warning(f"[Hallucination] Detected {max_consecutive} consecutive identical words: '{words[0]}'")
            return True, f"consecutive_repetition:{max_consecutive}"

        # Check total token count in recent_tokens (accumulated across segments)
        self.recent_tokens.extend(words)
        if len(self.recent_tokens) > 200:
            logger.warning(f"[Hallucination] Token count exceeded 200: {len(self.recent_tokens)}")
            return True, "token_overflow"

        return False, None

    def find_boundary_in_text(self, text, language):
        """Find the last boundary position within text (not just at the end)
        Returns: (boundary_pos, boundary_type) or (None, None) if no boundary found
        boundary_pos is the character position after the boundary"""

        if not text or not text.strip():
            return None, None

        # Check for sentence-ending punctuation first (highest priority)
        punctuation_marks = ['.', '!', '?', '。', '!', '?']
        last_punct_pos = -1
        for punct in punctuation_marks:
            pos = text.rfind(punct)
            if pos > last_punct_pos:
                last_punct_pos = pos

        if last_punct_pos >= 0:
            logger.info(f"[Boundary Detection] Found punctuation at position {last_punct_pos}: '{text[last_punct_pos]}'")
            return last_punct_pos + 1, "punctuation"

        # Check for phrase boundaries (Korean/English patterns)
        if language == 'ko':
            ko_conjunctions = [
                '그리고', '그런데', '하지만', '그래서', '그러나', '그러니', '그러니까',
                '그러면', '그럼', '그렇지만', '그치만', '그렇다면',
                '또', '또는', '또한', '더욱이', '게다가', '뿐만아니라',
                '근데', '그래도', '그렇더라도', '그런데도', '그래봤자',
                '따라서', '그러므로', '그렇기때문에', '그때문에', '그럼에도', '그럼에도불구하고',
                '즉', '다시말해', '다시말하면', '요컨대',
                '한편', '반면', '반면에', '이에반해', '이와달리',
                '물론', '확실히', '당연히', '분명히',
                '예를들어', '예를들면', '가령',
                '아무튼', '어쨌든', '여하튼', '하여튼',
                '왜냐하면', '왜냐', '외', '그외', '그외에', '그밖에',
                '특히', '무엇보다', '더구나', '심지어',
                '만약', '만일', '혹시', '설령',
                '비록', '설사', '가령', '단',
                '결국', '마침내', '드디어', '끝내',
                '그때', '그순간', '그후', '이후', '그다음', '그뒤'
            ]

            ko_endings = [
                '고', '구', '구요', '고요',
                '며', '면서', '으면서', '면', '으면',
                '지만', '지만요', '긴하지만', '긴한데',
                '는데', '은데', 'ㄴ데', '런데', '인데', '은데요', '는데요',
                '니까', '니깐', '으니까', '으니깐', '니', '으니',
                '어서', '아서', '서', '어서요', '아서요',
                '려고', 'ㄹ려고', '으려고', '려구', '으려구',
                '다가', '다가요', '었다가', '았다가', '였다가',
                '거나', '거나요', '든지', '든가',
                '자마자', '자', '자요',
                '수록', '을수록',
                '때', '을때', '땐', '을땐',
                '도록', '토록',
                '게', '게끔',
                '길래', '기에',
                '채', '은채', '채로', '은채로',
                '듯', '듯이', '듯이요',
                '나', '으나', '냐', '으냐',
                '든', '든지', '든가',
                '랴', '으랴',
                '건', '건만',
                '건데', '거든', '거든요',
                '느라', '느라고',
                '다시피', '다시피요',
                '던데', '더니', '더라',
                '자니', '자니까',
                '기로', '기로서니',
                '으면', '면',
                '지', '을지',
                '는데용', '은데용', '니깡', '어갖고', '아갖고',
                '어가지고', '아가지고', '해가지고',
                '구서', '구서요', '고서', '고서요',
                '걸', '을걸', 'ㄹ걸', '걸요', '을걸요',
                ','
            ]

            # Find last occurrence of any pattern
            last_boundary_pos = -1
            found_pattern = None

            # Check conjunctions (word boundaries)
            for pattern in ko_conjunctions:
                # Find last occurrence
                pos = text.rfind(' ' + pattern + ' ')
                if pos > last_boundary_pos:
                    last_boundary_pos = pos + len(' ' + pattern)
                    found_pattern = pattern
                # Also check if it's at the end
                if text.rstrip().endswith(' ' + pattern):
                    pos = text.rfind(' ' + pattern)
                    if pos > last_boundary_pos:
                        last_boundary_pos = len(text.rstrip())
                        found_pattern = pattern

            # Check endings
            for pattern in ko_endings:
                # Find all occurrences
                idx = text.rfind(pattern)
                while idx >= 0:
                    # Check if it's followed by space or is at the end
                    if idx + len(pattern) < len(text):
                        next_char = text[idx + len(pattern)]
                        if next_char == ' ':
                            if idx + len(pattern) > last_boundary_pos:
                                last_boundary_pos = idx + len(pattern)
                                found_pattern = pattern
                    else:
                        # At the end
                        if idx + len(pattern) > last_boundary_pos:
                            last_boundary_pos = idx + len(pattern)
                            found_pattern = pattern

                    # Find previous occurrence
                    idx = text.rfind(pattern, 0, idx)

            if last_boundary_pos > 0:
                logger.info(f"[Boundary Detection] Found Korean pattern '{found_pattern}' at position {last_boundary_pos}")
                return last_boundary_pos, "phrase"

        return None, None

    def check_phrase_boundary(self, text, language):
        """Check if text ends with a meaningful phrase boundary (faster translation trigger)"""
        text_stripped = text.strip()

        logger.info(f"[DEBUG check_phrase_boundary] Input text: '{text}' -> stripped: '{text_stripped}' (lang: {language})")

        if not text_stripped:
            logger.info(f"[DEBUG check_phrase_boundary] Text is empty after strip")
            return False

        # Korean phrase boundaries
        if language == 'ko':
            # Conjunctions (standalone words)
            ko_conjunctions = [
                # Basic conjunctions
                '그리고', '그런데', '하지만', '그래서', '그러나', '그러니', '그러니까',
                '그러면', '그럼', '그렇지만', '그치만', '그렇다면',
                '또', '또는', '또한', '더욱이', '게다가', '뿐만아니라',
                '근데', '그래도', '그렇더라도', '그런데도', '그래봤자',
                # Formal conjunctions
                '따라서', '그러므로', '그렇기때문에', '그때문에', '그럼에도', '그럼에도불구하고',
                '즉', '다시말해', '다시말하면', '요컨대',
                '한편', '반면', '반면에', '이에반해', '이와달리',
                '물론', '확실히', '당연히', '분명히',
                '예를들어', '예를들면', '가령',
                '아무튼', '어쨌든', '여하튼', '하여튼',
                # Additional
                '왜냐하면', '왜냐', '외', '그외', '그외에', '그밖에',
                '특히', '무엇보다', '더구나', '심지어',
                '만약', '만일', '혹시', '설령',
                '비록', '설사', '가령', '단',
                '결국', '마침내', '드디어', '끝내',
                '그때', '그순간', '그후', '이후', '그다음', '그뒤'
            ]

            # Connecting endings (attached to verbs/adjectives) - 매우 포괄적
            ko_endings = [
                # -고 계열
                '고', '구', '구요', '고요',
                # -며/-면서 계열
                '며', '면서', '으면서', '면', '으면',
                # -지만 계열
                '지만', '지만요', '긴하지만', '긴한데',
                # -는데/-ㄴ데 계열
                '는데', '은데', 'ㄴ데', '런데', '인데', '은데요', '는데요',
                # -니까/-으니까 계열
                '니까', '니깐', '으니까', '으니깐', '니', '으니',
                # -어서/-아서 계열
                '어서', '아서', '서', '어서요', '아서요',
                # -려고/-으려고 계열
                '려고', 'ㄹ려고', '으려고', '려구', '으려구',
                # -다가 계열
                '다가', '다가요', '었다가', '았다가', '였다가',
                # -거나 계열
                '거나', '거나요', '든지', '든가',
                # -자마자 계열
                '자마자', '자', '자요',
                # -수록 계열 (받침 유무 모두 포함)
                '수록', '을수록',
                # -때 계열 (받침 유무 모두 포함)
                '때', '을때', '땐', '을땐',
                # -도록
                '도록', '토록',
                # -게
                '게', '게끔',
                # -길래
                '길래', '기에',
                # -채 계열 (받침 유무 모두 포함)
                '채', '은채', '채로', '은채로',
                # -듯/-듯이
                '듯', '듯이', '듯이요',
                # 기타 연결어미
                '나', '으나', '냐', '으냐',
                '든', '든지', '든가',
                '랴', '으랴',
                '건', '건만',
                '건데', '거든', '거든요',
                '느라', '느라고',
                '다시피', '다시피요',
                '던데', '더니', '더라',
                '자니', '자니까',
                '기로', '기로서니',
                # -면/-으면 계열
                '으면', '면',
                # -ㄹ지/-을지 계열
                '지', '을지',
                # 추가 구어체
                '는데용', '은데용', '니깡', '어갖고', '아갖고',
                '어가지고', '아가지고', '해가지고',
                '구서', '구서요', '고서', '고서요',
                # -ㄹ걸/-을걸 계열 (추측, 후회)
                '걸', '을걸', 'ㄹ걸', '걸요', '을걸요'
            ]

            # Check if ends with standalone conjunction
            for conj in ko_conjunctions:
                if text_stripped.endswith(conj):
                    logger.info(f"[Phrase Boundary] Korean conjunction: {conj}")
                    return True

            # Check if ends with connecting ending (more flexible)
            for ending in ko_endings:
                if text_stripped.endswith(ending):
                    logger.info(f"[Phrase Boundary] Korean ending: {ending}")
                    return True

            # Check for comma (common phrase separator in Korean)
            if text_stripped.endswith(','):
                logger.info(f"[Phrase Boundary] Comma detected")
                return True

        # English phrase boundaries
        elif language == 'en':
            # Split into words for conjunction check
            words = text_stripped.lower().split()
            if not words:
                return False

            # Coordinating conjunctions (FANBOYS)
            en_conjunctions = [
                'and', 'but', 'or', 'so', 'yet', 'for', 'nor'
            ]

            # Subordinating conjunctions (very comprehensive)
            en_subordinating = [
                # Time
                'when', 'whenever', 'while', 'as', 'after', 'before', 'since', 'until', 'till',
                'once', 'as soon as', 'by the time',
                # Cause/Reason
                'because', 'since', 'as', 'now that', 'seeing that', 'given that',
                # Condition
                'if', 'unless', 'provided', 'provided that', 'providing that', 'as long as',
                'so long as', 'in case', 'even if', 'only if',
                # Contrast/Concession
                'although', 'though', 'even though', 'whereas', 'while', 'whilst',
                'despite', 'in spite of', 'regardless of', 'notwithstanding',
                # Purpose
                'so that', 'in order that', 'lest',
                # Comparison
                'than', 'as', 'as if', 'as though', 'like',
                # Place
                'where', 'wherever', 'everywhere', 'anywhere',
                # Manner
                'how', 'however', 'as', 'like'
            ]

            # Correlative conjunctions
            en_correlative = [
                'either', 'neither', 'both', 'not only', 'whether'
            ]

            # Transitional phrases (very comprehensive)
            en_transitions = [
                # Addition
                'moreover', 'furthermore', 'additionally', 'besides', 'also',
                'in addition', 'as well', 'not only', 'what is more', 'equally important',
                'likewise', 'similarly', 'in the same way', 'by the same token',
                # Contrast
                'however', 'nevertheless', 'nonetheless', 'still', 'yet',
                'on the contrary', 'on the other hand', 'in contrast', 'conversely',
                'rather', 'instead', 'alternatively', 'even so', 'despite this',
                'in spite of this', 'notwithstanding', 'regardless',
                # Cause/Result
                'therefore', 'thus', 'hence', 'consequently', 'accordingly',
                'as a result', 'as a consequence', 'for this reason', 'because of this',
                'due to this', 'owing to this', 'so', 'then',
                # Example
                'for example', 'for instance', 'such as', 'namely', 'specifically',
                'to illustrate', 'in particular', 'especially', 'notably',
                'as an illustration', 'case in point',
                # Emphasis
                'indeed', 'in fact', 'actually', 'certainly', 'surely',
                'undoubtedly', 'without doubt', 'obviously', 'clearly',
                'of course', 'naturally', 'definitely', 'absolutely',
                # Summary/Conclusion
                'in conclusion', 'to conclude', 'in summary', 'to summarize',
                'in short', 'in brief', 'to sum up', 'all in all', 'overall',
                'ultimately', 'finally', 'lastly', 'in the end',
                # Time/Sequence
                'meanwhile', 'in the meantime', 'simultaneously', 'at the same time',
                'afterward', 'afterwards', 'subsequently', 'later', 'then',
                'next', 'finally', 'eventually', 'previously', 'formerly',
                'at first', 'initially', 'first of all', 'to begin with',
                # Clarification
                'in other words', 'that is', 'that is to say', 'to put it differently',
                'to clarify', 'to rephrase', 'namely', 'specifically',
                # Condition
                'otherwise', 'if not', 'if so', 'in that case', 'under those circumstances',
                # Comparison
                'likewise', 'similarly', 'in the same way', 'equally', 'correspondingly',
                'by comparison', 'in comparison', 'compared to'
            ]

            # Check last word for conjunction
            last_word = words[-1].rstrip('.,;')
            if last_word in en_conjunctions or last_word in en_subordinating or last_word in en_correlative:
                logger.info(f"[Phrase Boundary] English conjunction: {last_word}")
                return True

            # Check last 2-6 words for transitional/subordinating phrases (descending order for longest match)
            for n in range(min(6, len(words)), 1, -1):
                last_n_words = ' '.join(words[-n:]).rstrip('.,;')

                # Check subordinating conjunctions (multi-word)
                if last_n_words in en_subordinating:
                    logger.info(f"[Phrase Boundary] English subordinating conjunction: {last_n_words}")
                    return True

                # Check transitional phrases
                if last_n_words in en_transitions:
                    logger.info(f"[Phrase Boundary] English transition: {last_n_words}")
                    return True

                # Check correlative conjunctions
                if last_n_words in en_correlative:
                    logger.info(f"[Phrase Boundary] English correlative conjunction: {last_n_words}")
                    return True

            # Check for comma
            if text_stripped.endswith(','):
                logger.info(f"[Phrase Boundary] Comma detected")
                return True

        return False

    def check_sentence_boundary(self, text, language):
        """Check if text contains a complete sentence using spaCy"""
        try:
            # Select appropriate spaCy model
            if language == 'ko' and nlp_ko is not None:
                nlp = nlp_ko
            elif language == 'en' and nlp_en is not None:
                nlp = nlp_en
            else:
                # Fallback: no spaCy model available
                logger.debug(f"[spaCy] No model available for language: {language}")
                return False

            # Process text with spaCy
            doc = nlp(text)
            sentences = list(doc.sents)

            logger.debug(f"[spaCy] Text: '{text}' -> {len(sentences)} sentence(s)")

            # Check if the last sentence ends with proper punctuation
            # spaCy always returns at least 1 sentence, so we need to verify it's actually complete
            if len(sentences) >= 1:
                last_sent = sentences[-1].text.strip()
                has_punctuation = any(last_sent.endswith(p) for p in ['.', '!', '?', '。', '!', '?'])

                logger.debug(f"[spaCy] Last sentence: '{last_sent}' | Has punctuation: {has_punctuation}")

                if has_punctuation:
                    logger.info(f"[spaCy] Detected complete sentence with punctuation")
                    return True

            return False

        except Exception as e:
            logger.error(f"[spaCy] Error in sentence boundary detection: {e}")
            return False

    async def detect_and_translate(self, text, detected_lang, lang_probs):
        """Translate based on language already detected and mapped to EN/KO by Whisper"""
        try:
            # Language is already set to 'en' or 'ko' by simul_whisper.py
            lang = detected_lang.lower() if detected_lang else 'en'

            logger.info(f"[detect_and_translate] Language from Whisper: {lang}")

            ko_text = None
            en_text = None

            # Run translation in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            # 한국어 → 영어 번역
            if lang == 'ko':
                en_text = await loop.run_in_executor(
                    None,
                    lambda: GoogleTranslator(source='ko', target='en').translate(text)
                )
                logger.debug(f"Translated KO→EN: {text} → {en_text}")
            # 영어 → 한국어 번역
            else:
                ko_text = await loop.run_in_executor(
                    None,
                    lambda: GoogleTranslator(source='en', target='ko').translate(text)
                )
                logger.debug(f"Translated EN→KO: {text} → {ko_text}")

            return lang, ko_text, en_text

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return 'en', None, None

    async def send_result(self, iteration_output):
        """Send transcription result to client"""
        # Track VAD status for translation flush
        vad_status = iteration_output.get('vad_status') if iteration_output else None
        if vad_status:
            if vad_status == 'nonvoice':
                self.nonvoice_count += 1
            else:
                self.nonvoice_count = 0

            # Flush translation buffer after 0.5s silence (nonvoice_count == 1)
            # VAD detects silence every 500ms, so count==1 means approximately 0.5s
            if self.nonvoice_count == 1 and len(self.translation_buffer) > 0:
                await self.flush_translation_buffer("VAD silence (0.5s)")

        if iteration_output and iteration_output.get('text'):
            start_ms = int(iteration_output['start'] * 1000)
            end_ms = int(iteration_output['end'] * 1000)
            text = iteration_output['text'].strip()

            if not text:
                logger.debug("Empty text in segment")
                return

            message = f"{start_ms} {end_ms} {text}"
            print(message, flush=True, file=sys.stderr)

            # Get detected language (override with hint if not auto)
            whisper_detected_lang = iteration_output.get('language', 'en')

            # Apply language hint override
            if self.language_hint == 'auto':
                detected_lang = whisper_detected_lang
            else:
                detected_lang = self.language_hint
                if whisper_detected_lang != detected_lang:
                    logger.info(f"[Language Hint] Overriding Whisper detection '{whisper_detected_lang}' with hint '{detected_lang}'")

            # Check if language changed - if so, clear buffer (don't flush) and reset ASR
            if self.detected_language is not None and self.detected_language != detected_lang:
                logger.info(f"[send_result] Language changed from {self.detected_language} to {detected_lang} - clearing buffer and resetting ASR")

                # Clear buffer without translating (previous buffer likely has misdetected language)
                logger.info(f"[send_result] Discarding {len(self.translation_buffer)} segments due to language change")
                self.translation_buffer = []

                # Reset ASR processor to clear KV cache and all internal states
                # This prevents tensor size mismatch errors when language changes
                self.online_asr_proc.init()
                logger.info("[send_result] ASR processor reset complete")

            # Update language
            self.detected_language = detected_lang

            logger.info(f"[send_result] Text segment: {text}")
            logger.info(f"[send_result] Detected language: {detected_lang}")

            # Check for hallucination BEFORE cleaning (to detect brackets)
            is_halluc, halluc_reason = self.is_hallucination(text)
            if is_halluc:
                logger.warning(f"[Hallucination] Detected in segment: '{text}' - reason: {halluc_reason}")
                # Discard this segment and clear buffer
                logger.info(f"[Hallucination] Discarding segment and clearing buffer ({len(self.translation_buffer)} segments)")
                self.translation_buffer = []
                self.recent_tokens = []  # Reset token tracking
                # Reset ASR to clear context
                self.online_asr_proc.init()
                logger.info("[Hallucination] ASR processor reset")
                return  # Skip this segment entirely

            # Clean special characters (�, ♪, etc.)
            text = self.clean_special_characters(text)
            if not text or not text.strip():
                logger.info(f"[send_result] Text is empty after cleaning - skipping")
                return

            # Check if the new segment contains a boundary in the middle
            boundary_pos, boundary_type = self.find_boundary_in_text(text, detected_lang)

            if boundary_pos is not None and boundary_pos < len(text.rstrip()):
                # Boundary found in the middle of the segment - split it
                before_boundary = text[:boundary_pos].strip()
                after_boundary = text[boundary_pos:].strip()

                logger.info(f"[send_result] Boundary found inside segment at position {boundary_pos}")
                logger.info(f"[send_result] Before: '{before_boundary}' | After: '{after_boundary}'")

                # Add the part before boundary to buffer
                if before_boundary:
                    self.translation_buffer.append({
                        'text': before_boundary,
                        'start': start_ms,
                        'end': end_ms  # Approximate - we don't have exact timing
                    })

                # Flush the buffer (translate the part before boundary)
                logger.info(f"[send_result] Flushing buffer due to mid-segment {boundary_type}")
                await self.flush_translation_buffer(f"mid_segment_{boundary_type}")

                # Start new buffer with the part after boundary
                if after_boundary:
                    self.translation_buffer.append({
                        'text': after_boundary,
                        'start': end_ms,  # Approximate
                        'end': end_ms
                    })
                    logger.info(f"[send_result] New buffer started with: '{after_boundary}'")

                    # Send partial for the remaining text
                    if self.display_mode != 'translateOnly':
                        result_msg = {
                            'type': 'partial',
                            'start': end_ms,
                            'end': end_ms,
                            'original': after_boundary,
                            'last_translation': self.last_translation,
                            'language': detected_lang
                        }
                        await self.send_message(result_msg)
            else:
                # No boundary in the middle - proceed with normal logic
                # Add to translation buffer
                self.translation_buffer.append({
                    'text': text,
                    'start': start_ms,
                    'end': end_ms
                })

                logger.info(f"[send_result] Buffer now has {len(self.translation_buffer)} segments")

                # Check for hallucination in the ENTIRE buffer (cumulative text)
                # This prevents partial results from being sent during hallucination
                cumulative_text = ' '.join(seg['text'] for seg in self.translation_buffer)
                is_cumulative_halluc, cumulative_halluc_reason = self.is_hallucination(cumulative_text)

                if is_cumulative_halluc:
                    logger.warning(f"[Hallucination] Detected in cumulative buffer: '{cumulative_text}' - reason: {cumulative_halluc_reason}")
                    logger.info(f"[Hallucination] Clearing buffer ({len(self.translation_buffer)} segments) and resetting ASR")
                    self.translation_buffer = []
                    self.recent_tokens = []
                    self.online_asr_proc.init()
                    return  # Don't send any partial or final message

                # Check if this segment ends with sentence-ending punctuation
                sentence_complete = any(text.endswith(p) for p in ['.', '!', '?', '。', '!', '?'])

                if sentence_complete:
                    # Sentence is complete - translate the entire buffer
                    logger.info(f"[send_result] Sentence complete (punctuation) - flushing")
                    await self.flush_translation_buffer("punctuation")
                else:
                    # Get CIF fire detection status from iteration_output
                    fire_detected = iteration_output.get('fire_detected', None) if iteration_output else None

                    # Priority 1: CIF model boundary detection (most accurate!)
                    if fire_detected:
                        logger.info(f"[send_result] CIF fire boundary detected - flushing for faster translation")
                        await self.flush_translation_buffer("cif_boundary")
                    # Priority 2: Check if spaCy detects a sentence boundary in full text (even without punctuation)
                    # DISABLED: spaCy causes performance bottleneck by processing entire buffer every segment (O(n²) complexity)
                    # CIF model + rule-based patterns are sufficient for boundary detection
                    # elif self.check_sentence_boundary(' '.join(seg['text'] for seg in self.translation_buffer), detected_lang):
                    #     logger.info(f"[spaCy] Sentence boundary detected without punctuation")
                    #     await self.flush_translation_buffer("sentence_boundary")

                    # Priority 2 (was 3): Check if THIS segment (just added) has phrase boundary
                    elif self.check_phrase_boundary(text, detected_lang):
                        logger.info(f"[send_result] Phrase boundary detected in current segment - flushing for faster translation")
                        await self.flush_translation_buffer("phrase_boundary")
                    else:
                        # Sentence not complete yet - send cumulative partial result with translation
                        if self.display_mode != 'translateOnly':
                            # Send cumulative text (all buffered segments combined)
                            first_start = self.translation_buffer[0]['start']

                            logger.info(f"[send_result] Partial cumulative ({len(self.translation_buffer)} segments): {cumulative_text}")

                            # For partial, show last completed translation (not current partial translation)
                            result_msg = {
                                'type': 'partial',
                                'start': first_start,
                                'end': end_ms,
                                'original': cumulative_text,  # Current partial text
                                'last_translation': self.last_translation,  # Last completed translation
                                'language': detected_lang
                            }

                            await self.send_message(result_msg)
                            logger.info(f"[send_result] Partial sent: {cumulative_text} (showing last translation: {self.last_translation})")
                        else:
                            logger.info(f"[send_result] Partial segment skipped (translateOnly mode, buffer: {len(self.translation_buffer)} segments)")
        else:
            logger.debug("No text in this segment")

    def convert_to_numpy(self, audio_data):
        """Convert Int16 binary data to numpy array"""
        try:
            # Client sends Int16 PCM data
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 for Whisper (range: -32768 ~ 32767 -> -1.0 ~ 1.0)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Apply noise reduction if enabled
            if self.use_denoiser and self.denoiser_model is not None:
                try:
                    import torch
                    with torch.no_grad():
                        # FullSubNet+ expects shape: [batch, samples]
                        audio_tensor = torch.from_numpy(audio_float).float().unsqueeze(0)

                        # Apply denoising - enhance_batch expects [batch, time]
                        denoised = self.denoiser_model.enhance_batch(audio_tensor, lengths=torch.tensor([1.0]))

                        # Convert back to numpy
                        audio_float = denoised.squeeze().cpu().numpy()

                        logger.debug(f"[Denoiser] Applied noise reduction to {len(audio_float)} samples")
                except Exception as e:
                    logger.error(f"[Denoiser] Error during noise reduction: {e}")
                    # Continue with original audio if denoising fails

            logger.debug(f"Received Int16 audio: {len(audio_int16)} samples -> {len(audio_float)} float32 samples")
            return audio_float

        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return None

    async def process_audio_chunk(self, audio_data):
        """Process incoming audio data"""
        try:
            # Convert binary Float32 to numpy array
            audio = self.convert_to_numpy(audio_data)

            if audio is None or len(audio) == 0:
                logger.warning("Failed to convert audio or empty audio")
                return

            logger.debug(f"Received audio chunk: {len(audio)} samples")

            # Add to buffer
            self.audio_buffer.append(audio)

            # Process when buffer has enough data
            total_samples = sum(len(x) for x in self.audio_buffer)
            min_samples = self.min_chunk * SAMPLING_RATE

            if total_samples >= min_samples or not self.is_first:
                # Concatenate all buffered audio
                if self.audio_buffer:
                    audio_chunk = np.concatenate(self.audio_buffer)
                    self.audio_buffer = []
                    self.is_first = False

                    # Insert audio into ASR
                    self.online_asr_proc.insert_audio_chunk(audio_chunk)

                    # Process and get result
                    result = self.online_asr_proc.process_iter()
                    await self.send_result(result)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """Main handler for WebSocket connection"""
        try:
            logger.info(f"New WebSocket connection from {self.websocket.remote_address}")

            # Send hello message
            await self.send_message({
                'type': 'hello',
                'message': 'Connected to Whisper Streaming Server'
            })

            # Initialize ASR
            self.online_asr_proc.init()
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
                            # Get display mode from client
                            if 'displayMode' in data:
                                self.display_mode = data['displayMode']
                                logger.info(f"Display mode set to: {self.display_mode}")
                            # Get language hint from client (check both 'lang' and 'languageHint' for backwards compatibility)
                            lang_hint = data.get('lang') or data.get('languageHint')
                            if lang_hint:
                                self.language_hint = lang_hint
                                logger.info(f"Language hint set to: {self.language_hint}")

                                # Force Whisper model to use the specified language
                                # This ensures transcription happens in the correct language
                                if hasattr(self.online_asr_proc, 'model'):
                                    if lang_hint != 'auto':
                                        # Update config language (prevents auto language detection in simul_whisper.py line 521)
                                        self.online_asr_proc.model.cfg.language = lang_hint
                                        # Update decode_options language
                                        if hasattr(self.online_asr_proc.model, 'decode_options'):
                                            self.online_asr_proc.model.decode_options = self.online_asr_proc.model.decode_options._replace(language=lang_hint)
                                        # Force detected_language to match hint (prevents re-detection)
                                        self.online_asr_proc.model.detected_language = lang_hint
                                        # Recreate tokenizer for the specified language
                                        self.online_asr_proc.model.create_tokenizer(lang_hint)
                                        # Reinitialize tokens with the new tokenizer to ensure correct language tokens
                                        self.online_asr_proc.model.init_tokens()
                                        logger.info(f"[Whisper] Forced model language to: {lang_hint} (cfg.language={self.online_asr_proc.model.cfg.language})")
                                    else:
                                        # Auto-detect mode: set language to auto
                                        self.online_asr_proc.model.cfg.language = 'auto'
                                        if hasattr(self.online_asr_proc.model, 'decode_options'):
                                            self.online_asr_proc.model.decode_options = self.online_asr_proc.model.decode_options._replace(language=None)
                                        # Reset detected_language to allow re-detection on next segment
                                        self.online_asr_proc.model.detected_language = None
                                        self.online_asr_proc.model.create_tokenizer(None)
                                        # Reinitialize tokens for auto-detect mode
                                        self.online_asr_proc.model.init_tokens()
                                        logger.info(f"[Whisper] Set to auto-detect language mode (cfg.language={self.online_asr_proc.model.cfg.language})")
                        elif msg_type == 'finish':
                            logger.info("Received finish command - flushing buffer")
                            # Flush remaining audio buffer
                            result = self.online_asr_proc.finish()
                            if result:
                                await self.send_result(result)

                            # Flush remaining translation buffer
                            await self.flush_translation_buffer("finish command")
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
            # 즉시 디코딩 중단 및 정리
            logger.info("WebSocket connection closing, flushing ASR processor...")
            self.running = False

            # ASR 프로세서 정리 (디코딩 즉시 중단)
            try:
                # 버퍼 비우기
                self.audio_buffer = []

                # ASR 디코딩 즉시 종료
                if hasattr(self.online_asr_proc, 'finish'):
                    self.online_asr_proc.finish()
                    logger.info("ASR processor finished and flushed")
                else:
                    logger.info("ASR processor flushed (no finish method)")
            except Exception as e:
                logger.error(f"Error flushing ASR processor: {e}")

            logger.info("WebSocket connection closed")


async def websocket_server(websocket, online_asr_proc, min_chunk, use_denoiser=False, denoiser_model=None):
    """WebSocket server handler"""
    handler = WebSocketHandler(
        websocket,
        online_asr_proc,
        min_chunk,
        use_denoiser=use_denoiser,
        denoiser_model=denoiser_model
    )
    await handler.handle()


def main_websocket_server(factory, add_args):
    """
    Main WebSocket server entry point

    factory: function that creates the ASR and online processor object from args and logger.
    add_args: add specific args for the backend
    """
    import websockets

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. "
            "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

    # Noise reduction option
    parser.add_argument("--denoise", action="store_true",
            help="Enable real-time noise reduction using FullSubNet+ (SpeechBrain). Effective for background conversations and murmuring. Requires: pip install speechbrain")

    # spaCy sentence boundary detection option
    parser.add_argument("--use-spacy", action="store_true", dest="use_spacy",
            help="Enable spaCy-based sentence boundary detection (slower but more accurate). Requires: python -m spacy download ko_core_news_sm en_core_web_sm")

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args, logger)

    # Load spaCy models if enabled
    if args.use_spacy:
        logger.info("[spaCy] Loading spaCy models for sentence boundary detection...")
        load_spacy_models()
    else:
        logger.info("[spaCy] Sentence boundary detection disabled (use --use-spacy to enable)")

    # setting whisper object by args
    asr, online = asr_factory(args, factory)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # Preload denoiser once (if enabled) - BEFORE warmup
    denoiser_model = load_denoiser_model() if args.denoise else None
    if args.denoise and denoiser_model is None:
        logger.warning("[Denoiser] Disabled because model failed to load.")

    # warm up the ASR because the very first transcribe takes more time than the others.
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
        # warmup된 online 객체를 직접 사용 (단일 클라이언트만 지원)
        await websocket_server(
            websocket,
            online,
            min_chunk,
            use_denoiser=args.denoise and denoiser_model is not None,
            denoiser_model=denoiser_model
        )

    async def main():
        denoiser_status = "enabled" if args.denoise and denoiser_model is not None else "disabled"
        logger.info(f'Starting WebSocket server on ws://{args.host}:{args.port} (Denoiser: {denoiser_status})')

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

    asyncio.run(main())
