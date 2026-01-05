"""
Sentence segmentation utilities for WhisperLiveKit.

Provides token-level and text-level detection of sentence boundaries
for real-time ASR applications.
"""

import logging
import re
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Sentence-ending punctuation marks
SENTENCE_END_PUNCTUATION = {'.', '!', '?', '。', '！', '？'}

# Korean sentence-ending patterns (josa/eomi)
# These are common Korean sentence-ending particles
KOREAN_ENDINGS = {
    '다', '요', '까', '니', '죠', '네', '지',
    '습니다', '입니다', '됩니다', '합니다',
    '해요', '가요', '와요', '네요', '군요', '겠어요',
    '어', '아', '지', '죠', '게', '는데', '거든',
}

# English sentence-ending words (less common, mostly for question forms)
ENGLISH_ENDINGS = {
    'right', 'okay', 'ok', 'yeah', 'yes', 'no', 'please'
}


class SentenceSegmenter:
    """
    Real-time sentence segmentation detector.

    Supports both token-level and text-level detection of sentence boundaries.
    """

    def __init__(
        self,
        enable_punctuation: bool = True,
        enable_korean_endings: bool = True,
        enable_english_endings: bool = False,
        min_tokens_before_break: int = 3,
    ):
        """
        Initialize sentence segmenter.

        Args:
            enable_punctuation: Detect sentence-ending punctuation (.!?。！？)
            enable_korean_endings: Detect Korean sentence-ending particles
            enable_english_endings: Detect English sentence-ending words
            min_tokens_before_break: Minimum tokens before allowing a sentence break
        """
        self.enable_punctuation = enable_punctuation
        self.enable_korean_endings = enable_korean_endings
        self.enable_english_endings = enable_english_endings
        self.min_tokens_before_break = min_tokens_before_break

        # Token counter for current segment
        self.token_count = 0

        logger.info(
            f"SentenceSegmenter initialized: punct={enable_punctuation}, "
            f"ko={enable_korean_endings}, en={enable_english_endings}, "
            f"min_tokens={min_tokens_before_break}"
        )

    def reset(self):
        """Reset the token counter for a new segment."""
        self.token_count = 0

    def is_sentence_boundary(self, text: str, token_count: Optional[int] = None) -> bool:
        """
        Check if the given text represents a sentence boundary.

        Args:
            text: Text to check (can be a single token or accumulated text)
            token_count: Optional token count override (uses internal counter if None)

        Returns:
            True if this is a sentence boundary, False otherwise
        """
        if token_count is None:
            token_count = self.token_count

        # Don't break too early
        if token_count < self.min_tokens_before_break:
            return False

        # Normalize text for checking
        text_stripped = text.strip()
        if not text_stripped:
            return False

        # Check for sentence-ending punctuation
        if self.enable_punctuation:
            if any(text_stripped.endswith(p) for p in SENTENCE_END_PUNCTUATION):
                logger.debug(f"Sentence boundary detected (punctuation): '{text_stripped}'")
                return True

        # Check for Korean endings
        if self.enable_korean_endings:
            if self._has_korean_ending(text_stripped):
                logger.debug(f"Sentence boundary detected (Korean ending): '{text_stripped}'")
                return True

        # Check for English endings
        if self.enable_english_endings:
            if self._has_english_ending(text_stripped):
                logger.debug(f"Sentence boundary detected (English ending): '{text_stripped}'")
                return True

        return False

    def _has_korean_ending(self, text: str) -> bool:
        """
        Check if text ends with a Korean sentence-ending particle.

        Korean sentence endings can be:
        1. Direct matches (e.g., "다", "요", "까")
        2. Compound endings (e.g., "습니다", "해요")
        """
        # Check for direct matches
        for ending in KOREAN_ENDINGS:
            if text.endswith(ending):
                # Additional validation: ensure it's not just a substring
                # For short endings like "다", check if preceded by a verb stem
                if len(ending) <= 2 and len(text) > len(ending):
                    # Korean verb stems typically end with specific patterns
                    # This is a simple heuristic
                    return True
                elif len(ending) > 2:
                    # Longer endings are more reliable
                    return True

        return False

    def _has_english_ending(self, text: str) -> bool:
        """
        Check if text ends with an English sentence-ending word.

        This is less reliable than Korean endings and is disabled by default.
        """
        words = text.lower().split()
        if not words:
            return False

        last_word = words[-1].rstrip('.,!?')
        return last_word in ENGLISH_ENDINGS

    def increment_token_count(self):
        """Increment the internal token counter."""
        self.token_count += 1

    def should_break_at_token(self, token_text: str) -> bool:
        """
        Token-level API: Check if decoding should stop after this token.

        This increments the internal counter automatically.

        Args:
            token_text: The text of the current token

        Returns:
            True if decoding should stop after this token
        """
        self.increment_token_count()
        return self.is_sentence_boundary(token_text)


def create_segmenter(
    mode: str = "full",
    min_tokens: int = 3
) -> Optional[SentenceSegmenter]:
    """
    Factory function to create a sentence segmenter.

    Args:
        mode: Segmentation mode
            - "off" or None: No segmentation
            - "punctuation": Only punctuation-based
            - "korean": Punctuation + Korean endings
            - "full": All detection methods (default)
        min_tokens: Minimum tokens before allowing breaks

    Returns:
        SentenceSegmenter instance or None if mode is "off"
    """
    if mode is None or mode.lower() == "off":
        return None

    if mode.lower() == "punctuation":
        return SentenceSegmenter(
            enable_punctuation=True,
            enable_korean_endings=False,
            enable_english_endings=False,
            min_tokens_before_break=min_tokens,
        )
    elif mode.lower() == "korean":
        return SentenceSegmenter(
            enable_punctuation=True,
            enable_korean_endings=True,
            enable_english_endings=False,
            min_tokens_before_break=min_tokens,
        )
    elif mode.lower() == "full":
        return SentenceSegmenter(
            enable_punctuation=True,
            enable_korean_endings=True,
            enable_english_endings=True,
            min_tokens_before_break=min_tokens,
        )
    else:
        logger.warning(f"Unknown segmentation mode '{mode}', using 'full'")
        return SentenceSegmenter(
            enable_punctuation=True,
            enable_korean_endings=True,
            enable_english_endings=True,
            min_tokens_before_break=min_tokens,
        )
