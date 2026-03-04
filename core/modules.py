# coding=utf-8
from typing import Optional
import numpy as np

from .types import (
    AudioSegment,
    RecognizedToken,
    CommittedSentence,
    ValidatedSentence,
    TranslationResult,
)


# A
class ConversationManager:
    def process_audio(self, raw_audio: np.ndarray) -> Optional[AudioSegment]:
        """raw audio → AudioSegment (발화 감지 시에만 반환)"""
        pass


# ASR
class SpeechRecognizer:
    def transcribe(self, segment: AudioSegment) -> Optional[RecognizedToken]:
        """AudioSegment → RecognizedToken (확정 시에만 반환)"""
        pass


# B
class CommitPolicy:
    def process_token(self, token: RecognizedToken) -> Optional[CommittedSentence]:
        """토큰 → CommittedSentence (분절 완료 시에만 반환)"""
        pass


# C
class StabilityFilter:
    def validate(self, sentence: CommittedSentence) -> Optional[ValidatedSentence]:
        """CommittedSentence → ValidatedSentence (유효하면 반환, 무효면 None)"""
        pass


# Translation
class TranslationService:
    def translate(self, sentence: ValidatedSentence) -> TranslationResult:
        """ValidatedSentence → TranslationResult"""
        pass
