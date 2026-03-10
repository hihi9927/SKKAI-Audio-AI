#!/usr/bin/env python3
"""
Realtime Meaning Segmenter
===========================
오디오 청크를 push_audio()로 누적하여 실시간으로 의미 분절 경계를 감지.

파이프라인:
  push_audio(chunk, sr)
    → VAD: Silero VAD (512-sample 프레임) → speech/silence 이벤트
    → speech end 감지 시 발화 청크 추출
    → Qwen3ASRModel.transcribe(utterance, return_time_stamps=True)
    → WordToken 생성 (누적 타임스탬프 offset 적용)
    → 피처 계산 (prosody / text / semantic)
    → Causal peak detection → SegmentBoundary 반환

지연(latency):
  min_silence_duration_ms 이상의 묵음이 쌓여야 발화 종료를 감지할 수 있으므로,
  해당 시간만큼의 지연은 구조적으로 불가피.

오프라인 버전(test_segmentation.py)과의 차이:
  - ForcedAligner 불필요 (transcribe() 내부에서 처리)
  - 경계 확정: trailing window window개 이상 쌓이면 causal하게 확정
  - 청크 간 경계(cross-utterance): 오디오 없으므로 gap 기반 prosody만 적용

사용 예:
    from core.meaning_seperator.realtime_segmenter import RealtimeSegmenter
    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", ...)
    seg = RealtimeSegmenter(asr, language="en", nlp=nlp)

    for chunk in audio_stream:           # numpy float32, 16kHz
        boundaries = seg.push_audio(chunk, sr=16000)
        for b in boundaries:
            print(f"[{b.word_before.end:.2f}s] '{b.word_before.text}' | '{b.word_after.text}'  score={b.total_score:.3f}")

    for b in seg.flush():                # 세션 종료 시 나머지 처리
        print(b)
"""

import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_BASE     = _HERE.parent.parent.parent
_QWEN_DIR = _BASE / "Qwen3-ASR"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_QWEN_DIR))

from core.meaning_seperator.test_segmentation import (
    WordToken,
    BoundaryScore,
    prosody_score,
    text_score,
    semantic_score,
    LANGUAGE_MAP,
    DEFAULT_WEIGHTS,
    DEFAULT_PEAK_WINDOW,
    DEFAULT_MIN_PROMINENCE,
    DEFAULT_MIN_FLOOR,
    DEFAULT_MIN_DISTANCE,
)

SAMPLING_RATE = 16000


# ─── 데이터 구조 ───────────────────────────────────────────────────────────────
@dataclass
class SegmentBoundary:
    """실시간으로 확정된 분절 경계."""
    token_index:    int
    word_before:    WordToken
    word_after:     WordToken
    total_score:    float
    prosody_score:  float
    text_score:     float
    semantic_score: float


# ─── RealtimeSegmenter ─────────────────────────────────────────────────────────
class RealtimeSegmenter:
    """
    실시간 의미 분절기.

    Silero VAD로 발화 청크를 감지하고, 각 청크를
    Qwen3ASRModel.transcribe()로 전사한 뒤 피처 계산 +
    causal peak detection으로 분절 경계를 확정합니다.

    VAD 지연: min_silence_duration_ms 이후에 발화 종료를 감지하므로
    그만큼의 지연은 구조적으로 불가피.
    """

    # Silero VAD가 요구하는 고정 프레임 크기 (16kHz 기준)
    _VAD_FRAME_SIZE = 512   # 32ms

    def __init__(
        self,
        asr_model,
        language:      str                = "en",
        weights:       Dict[str, float]   = None,
        okt                               = None,
        nlp                               = None,
        embed_fn:      Optional[Callable] = None,
        use_semantic:  bool               = True,
        # VAD
        vad_threshold:           float = 0.5,   # Silero VAD 민감도 (0~1)
        min_silence_duration_ms: int   = 400,   # 발화 종료 판단 묵음 길이 (ms)
        min_speech_duration_ms:  int   = 100,   # 최소 발화 길이 (ms, 이하 무시)
        # Peak detection
        window:         int   = DEFAULT_PEAK_WINDOW,
        min_prominence: float = DEFAULT_MIN_PROMINENCE,
        min_floor:      float = DEFAULT_MIN_FLOOR,
        min_distance:   int   = DEFAULT_MIN_DISTANCE,
    ):
        self.asr          = asr_model
        self.language     = language
        self.weights      = dict(weights or DEFAULT_WEIGHTS)
        self.okt          = okt
        self.nlp          = nlp
        self.embed_fn     = embed_fn
        self.use_semantic = use_semantic

        self.vad_threshold           = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms  = min_speech_duration_ms

        self.window         = window
        self.min_prominence = min_prominence
        self.min_floor      = min_floor
        self.min_distance   = min_distance

        # Silero VAD 로드
        try:
            from silero_vad import load_silero_vad, VADIterator
            _vad_model = load_silero_vad()
            self._vad_iter = VADIterator(
                _vad_model,
                sampling_rate=SAMPLING_RATE,
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_duration_ms,
                min_speech_duration_ms=min_speech_duration_ms,
            )
        except ImportError:
            raise ImportError("silero-vad 패키지가 필요합니다: pip install silero-vad")

        self._reset_state()

    # ── Public API ──────────────────────────────────────────────────────────────

    def reset(self):
        """새 세션 시작 시 상태 초기화."""
        self._vad_iter.reset_states()
        self._reset_state()

    def push_audio(self, chunk: np.ndarray, sr: int = SAMPLING_RATE) -> List[SegmentBoundary]:
        """
        오디오 청크를 누적합니다.
        Silero VAD가 발화 종료를 감지하면 전사 → 피처 계산 → 경계 확정.

        Args:
            chunk: float32 numpy array (mono, 16kHz)
            sr:    샘플링 레이트 (반드시 16000)

        Returns:
            새로 확정된 분절 경계 목록 (없으면 빈 리스트)
        """
        self._sr = sr
        buf = np.concatenate([self._vad_leftover, chunk.astype(np.float32)])

        results: List[SegmentBoundary] = []
        i = 0
        while i + self._VAD_FRAME_SIZE <= len(buf):
            frame = buf[i: i + self._VAD_FRAME_SIZE]
            event = self._vad_iter(torch.from_numpy(frame), return_seconds=False)

            if event and 'start' in event:
                self._in_speech    = True
                self._speech_buf   = np.array([], dtype=np.float32)

            if self._in_speech:
                self._speech_buf = np.concatenate([self._speech_buf, frame])

            if event and 'end' in event:
                self._in_speech = False
                utterance = self._speech_buf.copy()
                self._speech_buf = np.array([], dtype=np.float32)
                if len(utterance) >= int(0.1 * sr):
                    results.extend(self._process_utterance(utterance))

            i += self._VAD_FRAME_SIZE

        self._vad_leftover = buf[i:]
        return results

    def flush(self) -> List[SegmentBoundary]:
        """
        남은 버퍼를 강제 처리합니다 (세션 종료 시 호출).
        causal 제약 없이 남은 모든 경계를 peak detection합니다.
        """
        results: List[SegmentBoundary] = []

        # 발화 중이었으면 남은 speech_buf 처리
        remaining = np.concatenate([self._speech_buf, self._vad_leftover])
        if len(remaining) >= int(0.05 * self._sr):
            results.extend(self._process_utterance(remaining))

        results.extend(self._confirm_remaining())
        return results

    # ── Private: 상태 초기화 ────────────────────────────────────────────────────

    def _reset_state(self):
        self._sr:            int                  = SAMPLING_RATE
        self._vad_leftover:  np.ndarray           = np.array([], dtype=np.float32)
        self._in_speech:     bool                 = False
        self._speech_buf:    np.ndarray           = np.array([], dtype=np.float32)
        self._tokens:        List[WordToken]       = []
        self._boundaries:    List[BoundaryScore]   = []
        self._confirmed:     List[int]             = []
        self._checked_up_to: int                  = 0
        self._time_offset:   float                = 0.0
        self._gap_mean:      float                = 0.0
        self._gap_std:       float                = 0.1

    # ── Private: 발화 처리 ──────────────────────────────────────────────────────

    def _process_utterance(self, audio: np.ndarray) -> List[SegmentBoundary]:
        """
        발화 청크 처리:
          1. ASR → 토큰 (로컬 타임스탬프)
          2. 글로벌 타임스탬프 변환 & 토큰 누적
          3. gap 통계 업데이트
          4. 피처 계산
             - 청크 내 경계: 청크 오디오 사용 (prosody 전체)
             - 청크 간 경계: gap 기반 prosody만
          5. Causal peak detection
        """
        sr       = self._sr
        duration = len(audio) / sr
        aligner_lang = LANGUAGE_MAP.get(self.language, self.language.capitalize())

        # 1. ASR
        try:
            results = self.asr.transcribe(
                audio=(audio, sr),
                language=aligner_lang,
                return_time_stamps=True,
            )
        except Exception:
            self._time_offset += duration
            return []

        if not results:
            self._time_offset += duration
            return []

        # 2. 토큰 파싱 (로컬) & 글로벌 변환
        new_tokens_local = self._parse_tokens(results[0], duration)
        if not new_tokens_local:
            self._time_offset += duration
            return []

        new_tokens_global = [
            WordToken(t.text, t.start + self._time_offset, t.end + self._time_offset)
            for t in new_tokens_local
        ]

        prev_len = len(self._tokens)
        self._tokens.extend(new_tokens_global)
        self._time_offset += duration

        # 3. gap 통계 업데이트
        all_gaps = [
            max(0.0, self._tokens[i + 1].start - self._tokens[i].end)
            for i in range(len(self._tokens) - 1)
        ]
        self._gap_mean = float(np.mean(all_gaps)) if all_gaps else 0.0
        self._gap_std  = float(np.std(all_gaps))  if all_gaps else 0.1

        # audio_offset: 이 청크의 시작 시각 (글로벌)
        audio_offset = self._time_offset - duration

        # 4. 피처 계산
        # 청크 간 경계: 이전 마지막 토큰 → 새 첫 토큰 (오디오 없음)
        cross_idx = prev_len - 1
        if cross_idx >= 0 and len(self._tokens) > prev_len:
            self._compute_and_store_boundary(cross_idx, audio=None, audio_offset=audio_offset)

        # 청크 내 경계
        for i in range(max(prev_len, 0), len(self._tokens) - 1):
            self._compute_and_store_boundary(i, audio=audio, audio_offset=audio_offset)

        # 5. Causal peak detection
        return self._causal_peak_confirm()

    def _parse_tokens(self, result, duration: float) -> List[WordToken]:
        """ASR 결과에서 WordToken 리스트 추출 (로컬 타임스탬프)."""
        tokens: List[WordToken] = []

        # Case 1: ForcedAlignItem 리스트 (return_time_stamps=True 내부 aligner 결과)
        if isinstance(result, list):
            for item in result:
                tokens.append(WordToken(
                    text=str(item.text),
                    start=float(item.start_time),
                    end=float(item.end_time),
                ))
            return tokens

        # Case 2: word-level timestamps (result.words)
        words = getattr(result, 'words', None)
        if words:
            for w in words:
                tokens.append(WordToken(
                    text=getattr(w, 'word', str(w)),
                    start=float(getattr(w, 'start', 0.0)),
                    end=float(getattr(w, 'end', 0.0)),
                ))
            return tokens

        # Case 3: segment-level → 단어 균등 분배
        segments = getattr(result, 'segments', None)
        if segments:
            for seg in segments:
                seg_words = (getattr(seg, 'text', '') or '').strip().split()
                if not seg_words:
                    continue
                seg_start = float(getattr(seg, 'start', 0.0))
                seg_end   = float(getattr(seg, 'end', duration))
                w_dur     = (seg_end - seg_start) / len(seg_words)
                for j, w in enumerate(seg_words):
                    tokens.append(WordToken(
                        text=w,
                        start=seg_start + j * w_dur,
                        end=seg_start + (j + 1) * w_dur,
                    ))
            return tokens

        # Case 4: text only → 균등 분배
        text = (getattr(result, 'text', '') or '').strip()
        word_list = text.split()
        if not word_list:
            return []
        w_dur = duration / len(word_list)
        for j, w in enumerate(word_list):
            tokens.append(WordToken(
                text=w,
                start=j * w_dur,
                end=(j + 1) * w_dur,
            ))
        return tokens

    def _compute_and_store_boundary(
        self,
        idx:          int,
        audio:        Optional[np.ndarray],
        audio_offset: float,
    ):
        """경계 idx에서 피처 계산 후 self._boundaries에 추가."""
        if idx < 0 or idx >= len(self._tokens) - 1:
            return

        wb = self._tokens[idx]
        wa = self._tokens[idx + 1]

        # ── Prosody ─────────────────────────────────────────────────────────
        if audio is not None and len(audio) > 0:
            # 글로벌 타임스탬프 → 청크 로컬로 변환
            # all_tokens를 로컬로 변환하되 duration은 유지 (final lengthening용)
            all_local = [
                WordToken(t.text, t.start - audio_offset, t.end - audio_offset)
                for t in self._tokens
            ]
            wb_local = all_local[idx]
            wa_local = all_local[idx + 1]

            p_sc, p_det = prosody_score(
                audio, self._sr, wb_local, wa_local, all_local,
                gap_mean=self._gap_mean, gap_std=self._gap_std,
            )
        else:
            # 청크 간 경계: gap 기반만
            gap  = max(0.0, wa.start - wb.end)
            z    = (gap - self._gap_mean) / max(self._gap_std, 1e-6)
            g_sc = min(1.0, max(0.0, (z + 2) / 4))
            p_sc = 0.30 * g_sc
            p_det = {
                "gap_s":  round(gap, 4),
                "gap_z":  round(z, 3),
                "gap_sc": round(g_sc, 3),
                "note":   "cross_utterance_gap_only",
            }

        # ── Text ────────────────────────────────────────────────────────────
        t_sc, t_det = text_score(wb, wa, okt=self.okt, nlp=self.nlp)

        # ── Semantic ────────────────────────────────────────────────────────
        if self.use_semantic and self.embed_fn:
            s_sc, s_det = semantic_score(self._tokens, idx, self.embed_fn)
        else:
            s_sc, s_det = 0.0, {"note": "disabled"}

        # ── Weighted total ──────────────────────────────────────────────────
        w_p = self.weights["prosody"]
        w_t = self.weights["text"]
        w_s = self.weights["semantic"] if (self.use_semantic and self.embed_fn) else 0.0
        total_w = w_p + w_t + w_s
        total   = (w_p * p_sc + w_t * t_sc + w_s * s_sc) / max(total_w, 1e-8)

        self._boundaries.append(BoundaryScore(
            position=idx,
            word_before=wb, word_after=wa,
            prosody_score=round(p_sc, 3),
            text_score=round(t_sc, 3),
            semantic_score=round(s_sc, 3),
            total_score=round(total, 3),
            confirmed=False,
            details={"prosody": p_det, "text": t_det, "semantic": s_det},
        ))

    # ── Private: Peak detection ─────────────────────────────────────────────────

    def _causal_peak_confirm(self) -> List[SegmentBoundary]:
        """
        경계 리스트에서 충분한 trailing context(window개)가 확보된
        위치에 대해 peak 확정 검사. 이미 검사한 구간은 skip.
        """
        n = len(self._boundaries)
        max_confirmable = n - 1 - self.window
        if max_confirmable < self._checked_up_to:
            return []

        return self._peak_scan(self._checked_up_to, max_confirmable + 1, update_cursor=True)

    def _confirm_remaining(self) -> List[SegmentBoundary]:
        """flush() 시 trailing 제약 없이 남은 모든 경계 peak 확정."""
        n = len(self._boundaries)
        return self._peak_scan(self._checked_up_to, n, update_cursor=True)

    def _peak_scan(self, start: int, end: int, update_cursor: bool) -> List[SegmentBoundary]:
        """[start, end) 범위 경계에 대해 prominence peak 검사."""
        scores = [b.total_score for b in self._boundaries]
        n      = len(scores)
        new_confirmed: List[SegmentBoundary] = []

        for i in range(start, min(end, n)):
            s = scores[i]
            if s < self.min_floor:
                continue

            left  = scores[max(0, i - self.window): i]
            right = scores[i + 1: min(n, i + 1 + self.window)]
            lm    = float(np.mean(left))  if left  else 0.0
            rm    = float(np.mean(right)) if right else 0.0
            ctx   = (lm + rm) / 2 if (left and right) else max(lm, rm)

            if s - ctx < self.min_prominence:
                continue

            # NMS
            if any(abs(i - c) < self.min_distance for c in self._confirmed):
                continue

            self._confirmed.append(i)
            self._boundaries[i].confirmed = True
            b = self._boundaries[i]
            new_confirmed.append(SegmentBoundary(
                token_index=i,
                word_before=b.word_before,
                word_after=b.word_after,
                total_score=b.total_score,
                prosody_score=b.prosody_score,
                text_score=b.text_score,
                semantic_score=b.semantic_score,
            ))

        if update_cursor:
            self._checked_up_to = min(end, n)

        return new_confirmed
