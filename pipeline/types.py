# coding=utf-8
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np


# ===== 공통 타입 =====

@dataclass
class SpeakerInfo:
    speaker_id: int
    speaker_language: str


@dataclass
class TimeRange:
    audio_start_time_ms: int
    audio_end_time_ms: int


# ===== A → ASR =====

@dataclass
class AudioSegment:
    speaker: SpeakerInfo
    audio: np.ndarray
    time_range: TimeRange


# ===== ASR → B =====

@dataclass
class RecognizedToken:
    text: str
    time_range: TimeRange
    confidence: float
    language: str
    speaker: SpeakerInfo


# ===== B → C =====

@dataclass
class CommittedSentence:
    sentence_id: int
    text: str
    time_range: TimeRange
    language: str
    speaker: SpeakerInfo
    commit_reason: str


# ===== C → Translation =====

@dataclass
class ValidatedSentence:
    sentence_id: int
    text: str
    time_range: TimeRange
    source_language: str
    target_language: str
    speaker: SpeakerInfo


# ===== Translation → Client =====

@dataclass
class TranslationResult:
    sentence_id: int
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    speaker: SpeakerInfo
    output_timestamp_ms: int


# ===== 로그 =====

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class PipelineLog:
    stage: str
    level: LogLevel
    sentence_id: Optional[int]
    input_time_ms: int
    output_time_ms: int
    latency_ms: int
    message: str
    context: Optional[dict] = None
