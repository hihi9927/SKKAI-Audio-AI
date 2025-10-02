"""
Whisper utility functions for offline real-time STT.
Contains essential utilities for text processing, timestamp formatting, and quality metrics.
"""
import json
import os
import re
import sys
import zlib
from typing import List, Optional

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def exact_div(x, y):
    """Exact integer division with assertion."""
    assert x % y == 0
    return x // y


def str2bool(string):
    """Convert string representation to boolean."""
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    """Convert string to int or None."""
    return None if string == "None" else int(string)


def optional_float(string):
    """Convert string to float or None."""
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    """
    Calculate text compression ratio - useful for detecting repetitive or low-quality transcriptions.
    Higher values indicate more repetitive content.
    """
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    """
    Format seconds as timestamp string (HH:MM:SS.mmm or MM:SS.mmm).
    Useful for real-time STT timestamp display.
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def get_start(segments: List[dict]) -> Optional[float]:
    """Get the start time of the first segment."""
    return next(
        (w["start"] for s in segments for w in s["words"]),
        segments[0]["start"] if segments else None,
    )


def get_end(segments: List[dict]) -> Optional[float]:
    """Get the end time of the last segment."""
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


# Additional utility functions for real-time STT


def filter_segments_by_confidence(segments: List[dict], min_confidence: float = 0.5) -> List[dict]:
    """
    Filter segments by confidence score.
    Useful for real-time STT to remove low-quality transcriptions.
    """
    if not segments:
        return segments
    
    filtered = []
    for segment in segments:
        # Check if segment has confidence/probability information
        if "probability" in segment and segment["probability"] >= min_confidence:
            filtered.append(segment)
        elif "probability" not in segment:
            # If no probability info, keep the segment
            filtered.append(segment)
    
    return filtered


def merge_short_segments(segments: List[dict], min_duration: float = 0.5) -> List[dict]:
    """
    Merge very short segments with adjacent ones.
    Useful for real-time STT to reduce fragmentation.
    """
    if not segments or len(segments) <= 1:
        return segments
    
    merged = []
    current = segments[0].copy()
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        current_duration = current["end"] - current["start"]
        
        if current_duration < min_duration:
            # Merge with next segment
            current["end"] = next_segment["end"]
            current["text"] += " " + next_segment["text"].strip()
            if "words" in current and "words" in next_segment:
                current["words"].extend(next_segment["words"])
        else:
            # Keep current segment and move to next
            merged.append(current)
            current = next_segment.copy()
    
    # Add the last segment
    merged.append(current)
    return merged


def calculate_speaking_rate(segments: List[dict]) -> float:
    """
    Calculate speaking rate (words per minute).
    Useful for real-time STT quality metrics.
    """
    if not segments:
        return 0.0
    
    total_words = 0
    total_duration = 0.0
    
    for segment in segments:
        if "words" in segment:
            total_words += len(segment["words"])
        else:
            # Rough estimation: count words in text
            total_words += len(segment["text"].split())
        
        duration = segment["end"] - segment["start"]
        total_duration += duration
    
    if total_duration == 0:
        return 0.0
    
    # Convert to words per minute
    return (total_words / total_duration) * 60.0


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract potential keywords from transcribed text.
    Useful for real-time STT applications that need key information.
    """
    # Remove punctuation and convert to lowercase
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    
    # Filter by length and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'a', 'an'}
    
    keywords = []
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords
