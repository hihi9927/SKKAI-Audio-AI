#!/usr/bin/env python3
"""
Meaning Segmentation Test
=========================
Toy dataset(6개 m4a)에 대해 3가지 피처 기반 의미 분절을 테스트합니다.

피처:
  1. Prosody  - gap, final lengthening, f0 drop, energy drop
  2. Text     - 담화표지/접속어미/화제전환 신호
  3. Semantic - sentence-transformers 코사인 비유사도 (다중 창 크기)

경계 확정:
  고정 임계값 대신 local prominence 기반 peak detection 사용.
  각 위치의 점수가 주변(±window) 평균보다 min_prominence 이상 높아야 peak로 확정.
  NMS(non-maximum suppression)로 너무 가까운 peak 중 강한 것만 유지.

사용법:
  cd STiTy/
  python core/meaning_seperator/test_segmentation.py
  python core/meaning_seperator/test_segmentation.py --device cpu
  python core/meaning_seperator/test_segmentation.py --no-semantic
  python core/meaning_seperator/test_segmentation.py --window 4 --min-prominence 0.08
"""

import sys
import os
import re
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_BASE      = _HERE.parent.parent          # STiTy/
_QWEN_DIR  = _BASE / "Qwen3-ASR"
_TEST_DIR  = _HERE / "test"
_AUDIO_DIR = _TEST_DIR / "audio"
_TEXT_DIR  = _TEST_DIR / "text"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_QWEN_DIR))

# ─── 상수 ─────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS        = {"prosody": 0.4, "text": 0.4, "semantic": 0.2}
DEFAULT_PEAK_WINDOW    = 3      # 좌우 몇 개 이웃까지 비교할지
DEFAULT_MIN_PROMINENCE = 0.20   # peak가 이웃 평균보다 얼마나 높아야 하는지
DEFAULT_MIN_FLOOR      = 0.15   # 이 값 미만이면 noise로 간주, peak 후보 제외
DEFAULT_MIN_DISTANCE   = 2      # NMS: peak 간 최소 간격 (토큰 수)
TARGET_SR              = 16000

# 언어 코드 → Qwen3ForcedAligner 언어명 매핑
LANGUAGE_MAP: Dict[str, str] = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

# ── Text Feature: KoNLPy Okt POS 점수표 (한국어) ─────────────────────────────
BEFORE_POS_SCORE: Dict[str, float] = {
    "Eomi":        0.7,   # 어미 (종결/연결 공통)
    "Punctuation": 1.0,   # 구두점
}
BEFORE_VERBAL_EOMI_BONUS = 0.4   # Verb/Adj + Eomi → 절 종결 보너스

AFTER_POS_SCORE: Dict[str, float] = {
    "Exclamation": 0.7,  # 감탄사
    "Adverb":      0.7,  # 접속 부사
}

BEFORE_AUX_VERB_BONUS_UD = 0.2   # AUX + VERB → 한국어 용언+어미 상당

# ── Text Feature: SpaCy POS 점수표 (영어) ─────────────────────────────
BEFORE_POS_SCORE_UD: Dict[str, float] = {
    "PUNCT": 1.0,
    "VERB":  0.4,   # 기본값 낮추고
    "AUX":   0.2,
    "PART":  0.1,   # 'to' 같은 건 오히려 연속 신호
    "NOUN":  0.2,   # NP 종결도 분절 후보
    "PRON":  0.1,
    "ADJ":   0.15,  # predicative adj ("it's great")
    "ADV":   0.2,   # "so", "just" 등 문말 부사
    "NUM":   0.15,
}

# Morphology 기반 보너스 (spaCy token.morph 활용)
BEFORE_MORPH_BONUS: Dict[str, float] = {
    "Tense=Past":    0.3,   # 과거형 동사 → 완결성 높음
    "Tense=Pres":    0.15,
    "VerbForm=Fin":  0.25,  # 정형동사 (finite verb) → 절 종결
    "VerbForm=Inf":  -0.2,  # 부정사 → 뒤에 더 이어짐
    "VerbForm=Part": 0.1,
    "Mood=Ind":      0.15,  # 직설법
    "Mood=Imp":      0.3,   # 명령법 ("Stop!", "Go!")
    "Number=Plur":   0.05,
}

AFTER_POS_SCORE_UD: Dict[str, float] = {
    "INTJ":  0.6,
    "ADV":   0.3,   # 일반 부사는 낮추고
    "CCONJ": 0.4,   # "and", "but" → 새 절 시작 신호로 올림
    "SCONJ": 0.5,
    "DET":   0.2,
    "PRON":  0.3,   # "I", "he", "they" → 새 주어 등장
    "PROPN": 0.25,  # 고유명사로 시작 → 새 화제
    "NUM":   0.15,
}

# 이 패턴이 감지되면 점수를 깎아야 함
CONTINUITY_PENALTY_PATTERNS = [
    # VERB → to (부정사구 시작)
    ("VERB", "PART"),    # -0.4
    # 관계절 시작
    ("NOUN", "REL"),     # -0.3  (which, that, who)
    # 비교급
    ("ADJ", "SCONJ"),    # -0.2  ("better than")
    # 조동사 + 본동사
    ("AUX", "VERB"),     # -0.3
]

# token.dep_ 기반 종결 강도
DEP_SCORE: Dict[str, float] = {
    "ROOT":      0.3,   # 문장 핵심 동사
    "conj":     -0.1,   # 등위 접속 구조 안에 있음
    "advcl":    -0.15,  # 부사절 안에 있음
    "relcl":    -0.2,   # 관계절 안에 있음
    "mark":     -0.1,   # 종속 접속사 마커
    "cc":        0.2,   # 등위 접속사 → 새 절 경계
}

# spaCy 언어 코드 → 모델명
SPACY_MODEL_MAP: Dict[str, str] = {
    "en": "en_core_web_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
}

# ─── 데이터 구조 ───────────────────────────────────────────────────────────────
@dataclass
class WordToken:
    text:  str
    start: float   # seconds
    end:   float   # seconds

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class BoundaryScore:
    position:       int          # i → boundary between tokens[i] and tokens[i+1]
    word_before:    WordToken
    word_after:     WordToken
    prosody_score:  float = 0.0
    text_score:     float = 0.0
    semantic_score: float = 0.0
    total_score:    float = 0.0
    confirmed:      bool  = False
    details:        Dict  = field(default_factory=dict)


@dataclass
class SegmentationResult:
    name:                str
    tokens:              List[WordToken]
    boundaries:          List[BoundaryScore]
    confirmed_positions: List[int]
    gt_positions:        List[int]


# ─── 오디오 로딩 ───────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """m4a / wav 등을 numpy float32로 로드."""
    try:
        import librosa
        audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
        return audio, loaded_sr
    except Exception as e1:
        try:
            from pydub import AudioSegment as _Pydub
            seg = _Pydub.from_file(path).set_frame_rate(sr).set_channels(1)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= 2 ** (seg.sample_width * 8 - 1)
            return samples, sr
        except Exception as e2:
            raise RuntimeError(f"오디오 로드 실패 ({path}): {e1} | {e2}")


# ─── GT 파싱 ───────────────────────────────────────────────────────────────────
def parse_gt(text_with_markers: str) -> Tuple[str, List[str]]:
    """
    '/ 로 표시된 정답 텍스트를 파싱.
    Returns: (clean_text, segments)
    """
    parts = [p.strip() for p in text_with_markers.split("/") if p.strip()]
    clean = " ".join(parts)
    return clean, parts


def get_gt_boundary_indices(tokens: List[WordToken], gt_segments: List[str]) -> List[int]:
    """
    누적 문자 수 기준으로 GT 분절 위치 → 토큰 인덱스로 변환.
    boundary after tokens[i] ↔ 반환 리스트의 원소 i.
    """
    # 토큰 텍스트를 공백 없이 이어붙인 문자열에서 각 토큰의 끝 위치 계산
    char_ends = []
    acc = 0
    for t in tokens:
        acc += len(t.text.replace(" ", ""))
        char_ends.append(acc)

    # 각 세그먼트(마지막 제외)의 끝 문자 위치
    boundary_indices = []
    seg_char_acc = 0
    for seg in gt_segments[:-1]:
        seg_char_acc += len(seg.replace(" ", ""))
        # seg_char_acc에 가장 가까운 토큰 경계 찾기
        best_idx = min(range(len(char_ends)), key=lambda i: abs(char_ends[i] - seg_char_acc))
        boundary_indices.append(best_idx)

    return boundary_indices


# ─── Feature 1: Prosody ────────────────────────────────────────────────────────
def prosody_score(
    audio: np.ndarray, sr: int,
    w_before: WordToken, w_after: WordToken,
    all_tokens: List[WordToken],
    gap_mean: float = 0.0,
    gap_std:  float = 0.1,
) -> Tuple[float, Dict]:
    """
    Gap, Final Lengthening, F0 Drop, Pitch Reset, Energy Drop 5가지 서브피처.

    Gap score: 발화 내 pause 분포의 z-score 기반 → 화자마다/속도마다 자동 정규화.
    Pitch reset: 경계 다음 단어 시작 F0가 직전보다 높으면 ↑ (한국어 억양 재설정).
    """
    details: Dict = {}
    score = 0.0

    # 1. Gap — z-score 기반 (speech rate 자동 정규화) ── 가중 0.30
    gap = max(0.0, w_after.start - w_before.end)
    z = (gap - gap_mean) / max(gap_std, 1e-6)
    # z=0(평균) → 0.5, z=2(매우 긴 pause) → 1.0, z=-2(매우 짧음) → 0
    g_score = min(1.0, max(0.0, (z + 2) / 4))
    details["gap_s"]  = round(gap, 4)
    details["gap_z"]  = round(z, 3)
    details["gap_sc"] = round(g_score, 3)
    score += 0.30 * g_score

    # 2. Final lengthening ── 가중 0.20
    idx = all_tokens.index(w_before)
    local_start = max(0, idx - 3)
    local_end   = min(len(all_tokens), idx + 3)
    local_durs  = [t.duration for t in all_tokens[local_start:local_end] if t.duration > 0.01]
    mean_dur    = float(np.mean(local_durs)) if local_durs else 0.15
    fl_ratio    = w_before.duration / max(mean_dur, 1e-6)
    fl_score    = min(1.0, max(0.0, (fl_ratio - 0.8) / 1.2))
    details["fl_ratio"] = round(fl_ratio, 3)
    details["fl_sc"]    = round(fl_score, 3)
    score += 0.20 * fl_score

    # 3. F0 drop (word_before) + Pitch reset (word_after) ── 합산 가중 0.35
    try:
        import librosa
        seg_b = audio[max(0, int(w_before.start * sr)): int(w_before.end * sr)]
        seg_a = audio[int(w_after.start * sr): min(len(audio), int(w_after.start * sr) + int(0.2 * sr))]

        f0v_b = np.array([])
        f0v_a = np.array([])

        if len(seg_b) >= sr * 0.08:
            f0_b, voiced_b, _ = librosa.pyin(seg_b, fmin=70, fmax=500, sr=sr,
                                              frame_length=512, hop_length=128)
            f0v_b = f0_b[voiced_b] if voiced_b is not None else np.array([])

        if len(seg_a) >= sr * 0.05:
            f0_a, voiced_a, _ = librosa.pyin(seg_a, fmin=70, fmax=500, sr=sr,
                                              frame_length=512, hop_length=128)
            f0v_a = f0_a[voiced_a] if voiced_a is not None else np.array([])

        # F0 drop: word_before 전반 → 후반 하강 ── 가중 0.20
        if len(f0v_b) >= 4:
            half    = len(f0v_b) // 2
            f0_drop = (np.mean(f0v_b[:half]) - np.mean(f0v_b[half:])) / max(np.mean(f0v_b[:half]), 1.0)
            f0_drop_sc = min(1.0, max(0.0, float(f0_drop) * 3))
            details["f0_drop"]    = round(float(f0_drop), 3)
            details["f0_drop_sc"] = round(f0_drop_sc, 3)
            score += 0.20 * f0_drop_sc
        else:
            details["f0_drop_note"] = "too_short"

        # Pitch reset: word_after 시작이 word_before 끝보다 높으면 ↑ ── 가중 0.15
        if len(f0v_b) >= 2 and len(f0v_a) >= 2:
            f0_before_end  = float(np.mean(f0v_b[len(f0v_b) // 2:]))
            f0_after_start = float(np.mean(f0v_a[:max(1, len(f0v_a) // 2)]))
            f0_reset       = f0_after_start - f0_before_end   # 양수 = 피치 재설정 상승
            reset_sc       = min(1.0, max(0.0, f0_reset / 80.0))  # 80 Hz 상승이면 1.0
            details["f0_reset_hz"] = round(f0_reset, 1)
            details["reset_sc"]    = round(reset_sc, 3)
            score += 0.15 * reset_sc
        else:
            details["reset_note"] = "insufficient_f0"

    except Exception as e:
        details["f0_err"] = str(e)

    # 4. Energy drop ── 가중 0.15
    try:
        win     = int(0.1 * sr)
        b_end   = int(w_before.end  * sr)
        a_start = int(w_after.start * sr)
        e_before = float(np.sqrt(np.mean(audio[max(0, b_end - win):b_end] ** 2))) if b_end > 0 else 0
        e_after  = float(np.sqrt(np.mean(audio[a_start:min(len(audio), a_start + win)] ** 2))) if a_start < len(audio) else 0
        e_drop   = max(0.0, e_before - e_after) / max(e_before, 1e-8)
        e_sc     = min(1.0, e_drop * 2)
        details["e_drop"] = round(e_drop, 3)
        details["e_sc"]   = round(e_sc, 3)
        score += 0.15 * e_sc
    except Exception as e:
        details["e_err"] = str(e)

    return min(1.0, score), details


# ─── Feature 2: Text (형태소 분석 기반) ───────────────────────────────────────
def text_score(
    w_before: WordToken, w_after: WordToken,
    okt=None,
    nlp=None,
) -> Tuple[float, Dict]:
    """
    형태소 분석 기반 텍스트 경계 피처.
    - 한국어 (okt): KoNLPy Okt POS — Eomi/Punctuation + 용언+어미 보너스
    - 비한국어 (nlp): spaCy UD POS — VERB/PUNCT + AUX+VERB 보너스
    둘 다 없으면 0.0 반환.
    """
    details: Dict = {}
    score = 0.0

    if okt is not None:
        # ── 한국어: KoNLPy Okt ───────────────────────────────────────────────
        try:
            bm = okt.pos(w_before.text, norm=True)
            pos_b = [p for _, p in bm]
            details["before_pos"] = pos_b
            if pos_b:
                last_pos = pos_b[-1]
                base_sc  = BEFORE_POS_SCORE.get(last_pos, 0.0)
                if last_pos == "Eomi" and len(pos_b) >= 2 and pos_b[-2] in ("Verb", "Adjective"):
                    base_sc += BEFORE_VERBAL_EOMI_BONUS
                    details["before_pattern"] = "verb_eomi"
                else:
                    details["before_pattern"] = last_pos
                score += base_sc
        except Exception as e:
            details["before_err"] = str(e)

        try:
            am = okt.pos(w_after.text, norm=True)
            pos_a = [p for _, p in am]
            details["after_pos"] = pos_a
            if pos_a:
                first_pos = pos_a[0]
                score += AFTER_POS_SCORE.get(first_pos, 0.0)
                details["after_pattern"] = first_pos
        except Exception as e:
            details["after_err"] = str(e)

    elif nlp is not None:
        # ── 비한국어: spaCy UD POS ───────────────────────────────────────────
        _PENALTY_MAP = {
            ("VERB", "PART"):  -0.4,
            ("NOUN", "REL"):   -0.3,
            ("ADJ", "SCONJ"):  -0.2,
            ("AUX", "VERB"):   -0.3,
        }
        _REL_LEMMAS = {"which", "that", "who", "whom", "whose", "where", "when"}

        doc_b = None
        doc_a = None

        try:
            doc_b = nlp(w_before.text)
            pos_b = [t.pos_ for t in doc_b]
            details["before_pos"] = pos_b
            if doc_b:
                last_tok = doc_b[-1]
                last_pos = last_tok.pos_

                base_sc = BEFORE_POS_SCORE_UD.get(last_pos, 0.0)
                if last_pos == "VERB" and len(pos_b) >= 2 and pos_b[-2] == "AUX":
                    base_sc += BEFORE_AUX_VERB_BONUS_UD
                    details["before_pattern"] = "aux_verb"
                else:
                    details["before_pattern"] = last_pos

                # Morphology bonus
                morph_str = str(last_tok.morph)
                morph_bonus = sum(v for k, v in BEFORE_MORPH_BONUS.items() if k in morph_str)
                if morph_bonus != 0.0:
                    details["before_morph_bonus"] = round(morph_bonus, 3)
                    base_sc += morph_bonus

                # DEP score
                dep_sc = DEP_SCORE.get(last_tok.dep_, 0.0)
                if dep_sc != 0.0:
                    details["before_dep"] = f"{last_tok.dep_}({dep_sc:+.2f})"
                    base_sc += dep_sc

                score += base_sc
        except Exception as e:
            details["before_err"] = str(e)

        try:
            doc_a = nlp(w_after.text)
            pos_a = [t.pos_ for t in doc_a]
            details["after_pos"] = pos_a
            if doc_a:
                first_pos = doc_a[0].pos_
                score += AFTER_POS_SCORE_UD.get(first_pos, 0.0)
                details["after_pattern"] = first_pos
        except Exception as e:
            details["after_err"] = str(e)

        # Cross-boundary continuity penalty
        if doc_b and doc_a:
            try:
                last_pos  = doc_b[-1].pos_
                after_key = "REL" if doc_a[0].lemma_.lower() in _REL_LEMMAS else doc_a[0].pos_
                penalty   = _PENALTY_MAP.get((last_pos, after_key), 0.0)
                if penalty < 0:
                    details["continuity_penalty"] = f"({last_pos},{after_key}){penalty:+.2f}"
                    score += penalty
            except Exception:
                pass

    else:
        details["note"] = "no_tagger"
        return 0.0, details

    return min(1.0, score), details


# ─── Feature 3: Semantic ───────────────────────────────────────────────────────
def semantic_score(
    tokens: List[WordToken],
    position: int,
    embed_fn: Optional[Callable],
    window_sizes: List[int] = [1, 3, 6],
) -> Tuple[float, Dict]:
    details: Dict = {}
    if embed_fn is None:
        details["note"] = "no_embed_model"
        return 0.0, details

    dissims = []
    for w in window_sizes:
        before_toks = tokens[max(0, position - w + 1): position + 1]
        after_toks  = tokens[position + 1: min(len(tokens), position + 1 + w)]
        if not before_toks or not after_toks:
            continue
        b_text = " ".join(t.text for t in before_toks)
        a_text = " ".join(t.text for t in after_toks)
        try:
            eb = embed_fn(b_text)
            ea = embed_fn(a_text)
            cos = float(np.dot(eb, ea) / (np.linalg.norm(eb) * np.linalg.norm(ea) + 1e-8))
            d   = max(0.0, 1.0 - cos)
            dissims.append(d)
            details[f"w{w}_dissim"] = round(d, 3)
        except Exception as e:
            details[f"w{w}_err"] = str(e)

    if not dissims:
        return 0.0, details
    final = float(np.mean(dissims))
    details["avg_dissim"] = round(final, 3)
    return min(1.0, final), details


# ─── Peak Detection ────────────────────────────────────────────────────────────
def find_boundary_peaks(
    boundaries:     List[BoundaryScore],
    window:         int   = DEFAULT_PEAK_WINDOW,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_floor:      float = DEFAULT_MIN_FLOOR,
    min_distance:   int   = DEFAULT_MIN_DISTANCE,
) -> List[int]:
    """
    고정 임계값 대신 local prominence 기반으로 분절 경계 확정.

    알고리즘:
      1. total_score < min_floor → noise로 간주, 후보 제외
      2. 좌우 ±window 범위 평균보다 min_prominence 이상 높아야 peak 후보
      3. NMS: min_distance 이하 간격의 후보가 여러 개면 score 높은 것만 유지
    """
    scores = [b.total_score for b in boundaries]
    n = len(scores)

    # 1~2단계: floor + prominence 필터
    candidates: List[Tuple[int, float]] = []
    for i in range(n):
        s = scores[i]
        if s < min_floor:
            continue
        left  = scores[max(0, i - window): i]
        right = scores[i + 1: min(n, i + 1 + window)]
        left_mean  = float(np.mean(left))  if left  else 0.0
        right_mean = float(np.mean(right)) if right else 0.0
        # 양쪽 평균 (한쪽만 있으면 그 쪽만)
        if left and right:
            context_mean = (left_mean + right_mean) / 2
        else:
            context_mean = max(left_mean, right_mean)
        if s - context_mean >= min_prominence:
            candidates.append((i, s))

    if not candidates:
        return []

    # 3단계: NMS - score 높은 순으로 정렬 후 min_distance 이내 중복 제거
    candidates.sort(key=lambda x: x[1], reverse=True)
    confirmed: List[int] = []
    for idx, _ in candidates:
        if all(abs(idx - c) >= min_distance for c in confirmed):
            confirmed.append(idx)
    return sorted(confirmed)


# ─── 메인 파이프라인 ────────────────────────────────────────────────────────────
def run_test_case(
    name:           str,
    audio_path:     str,
    gt_answer:      str,
    aligner,
    embed_fn:       Optional[Callable],
    weights:        Dict[str, float],
    use_semantic:   bool,
    device:         str,
    language:       str   = "ko",
    okt=None,
    nlp=None,
    window:         int   = DEFAULT_PEAK_WINDOW,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_floor:      float = DEFAULT_MIN_FLOOR,
    min_distance:   int   = DEFAULT_MIN_DISTANCE,
) -> SegmentationResult:

    # 1. 오디오 로드
    audio, sr = load_audio(audio_path)

    # 2. GT 파싱
    clean_text, gt_segments = parse_gt(gt_answer)

    # 3. Forced alignment → 토큰별 타임스탬프
    aligner_lang = LANGUAGE_MAP.get(language, language.capitalize())
    align_results = aligner.align(
        audio=(audio, sr),
        text=clean_text,
        language=aligner_lang,
    )
    raw_tokens = align_results[0]
    tokens = [WordToken(t.text, float(t.start_time), float(t.end_time)) for t in raw_tokens]

    # 4. GT 경계 인덱스
    gt_positions = get_gt_boundary_indices(tokens, gt_segments)

    # 5. 발화 전체 gap 분포 계산 (z-score 기반 gap 점수를 위한 사전 통계)
    all_gaps = [max(0.0, tokens[i + 1].start - tokens[i].end) for i in range(len(tokens) - 1)]
    gap_mean = float(np.mean(all_gaps)) if all_gaps else 0.0
    gap_std  = float(np.std(all_gaps))  if all_gaps else 0.1

    # 6. Pass 1: 각 경계에서 피처 점수 계산 (confirmed는 아직 미정)
    boundaries: List[BoundaryScore] = []
    for i in range(len(tokens) - 1):
        wb, wa = tokens[i], tokens[i + 1]

        p_sc, p_det = prosody_score(audio, sr, wb, wa, tokens, gap_mean=gap_mean, gap_std=gap_std)
        t_sc, t_det = text_score(wb, wa, okt=okt, nlp=nlp)
        s_sc, s_det = (semantic_score(tokens, i, embed_fn) if use_semantic
                       else (0.0, {"note": "disabled"}))

        w_s = weights["semantic"] if use_semantic else 0.0
        w_t = weights["text"]
        w_p = weights["prosody"]
        total_w = w_p + w_t + w_s
        total = (w_p * p_sc + w_t * t_sc + w_s * s_sc) / max(total_w, 1e-8)

        boundaries.append(BoundaryScore(
            position=i,
            word_before=wb, word_after=wa,
            prosody_score=round(p_sc, 3),
            text_score=round(t_sc, 3),
            semantic_score=round(s_sc, 3),
            total_score=round(total, 3),
            confirmed=False,
            details={"prosody": p_det, "text": t_det, "semantic": s_det},
        ))

    # 7. Pass 2: 전체 점수 분포를 보고 local prominence peak 확정
    confirmed_positions = find_boundary_peaks(
        boundaries,
        window=window,
        min_prominence=min_prominence,
        min_floor=min_floor,
        min_distance=min_distance,
    )
    confirmed_set = set(confirmed_positions)
    for b in boundaries:
        b.confirmed = b.position in confirmed_set

    return SegmentationResult(
        name=name,
        tokens=tokens,
        boundaries=boundaries,
        confirmed_positions=confirmed_positions,
        gt_positions=gt_positions,
    )


# ─── 평가 ─────────────────────────────────────────────────────────────────────
def evaluate(result: SegmentationResult) -> Dict:
    pred = set(result.confirmed_positions)
    gt   = set(result.gt_positions)
    tp   = len(pred & gt)
    fp   = len(pred - gt)
    fn   = len(gt - pred)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return dict(precision=round(prec,3), recall=round(rec,3), f1=round(f1,3),
                tp=tp, fp=fp, fn=fn,
                pred=sorted(pred), gt=sorted(gt))


# ─── 번역 기반 평가 ────────────────────────────────────────────────────────────
def get_segments_text(tokens: List[WordToken], boundary_positions: List[int]) -> List[str]:
    """토큰 리스트를 경계 위치 기준으로 분절하여 각 분절 텍스트 반환."""
    if not tokens:
        return []
    bounds = sorted(boundary_positions)
    segments: List[str] = []
    prev = 0
    for pos in bounds:
        end = min(pos + 1, len(tokens))
        seg = " ".join(t.text for t in tokens[prev:end])
        if seg.strip():
            segments.append(seg.strip())
        prev = end
    tail = " ".join(t.text for t in tokens[prev:])
    if tail.strip():
        segments.append(tail.strip())
    return segments


def translate_segments(
    segments: List[str],
    source: str = "ko",
    target: str = "en",
) -> List[str]:
    """deep-translator GoogleTranslator로 각 분절 번역."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return []

    results: List[str] = []
    translator = GoogleTranslator(source=source, target=target)
    for seg in segments:
        try:
            translated = translator.translate(seg)
            results.append(translated or seg)
        except Exception as e:
            results.append(f"[번역 오류: {e}]")
    return results


def translation_evaluate(
    result: SegmentationResult,
    source_lang: str = "ko",
    target_lang: str = "en",
    embed_fn: Optional[Callable] = None,
) -> Dict:
    """
    GT 분절 번역 vs 예측 분절 번역 비교 평가.
    GT 경계로 분절한 텍스트를 번역한 결과와
    예측 경계로 분절한 텍스트를 번역한 결과를 비교합니다.
    source_lang과 target_lang이 같으면 번역 없이 원문 그대로 반환합니다.
    """
    gt_segs   = get_segments_text(result.tokens, result.gt_positions)
    pred_segs = get_segments_text(result.tokens, result.confirmed_positions)

    if source_lang == target_lang:
        gt_trans   = list(gt_segs)
        pred_trans = list(pred_segs)
    else:
        gt_trans   = translate_segments(gt_segs,   source=source_lang, target=target_lang)
        pred_trans = translate_segments(pred_segs, source=source_lang, target=target_lang)

    similarity: Optional[float] = None
    if embed_fn and gt_trans and pred_trans:
        gt_full   = " ".join(gt_trans)
        pred_full = " ".join(pred_trans)
        try:
            eg  = embed_fn(gt_full)
            ep  = embed_fn(pred_full)
            cos = float(np.dot(eg, ep) / (np.linalg.norm(eg) * np.linalg.norm(ep) + 1e-8))
            similarity = round(max(0.0, cos), 3)
        except Exception:
            pass

    return {
        "gt_segments":       gt_segs,
        "pred_segments":     pred_segs,
        "gt_translations":   gt_trans,
        "pred_translations": pred_trans,
        "similarity":        similarity,
        "target_lang":       target_lang,
    }


# ─── 출력 ─────────────────────────────────────────────────────────────────────
def print_result(result: SegmentationResult, ev: Dict) -> None:
    W = 72
    print(f"\n{'='*W}")
    print(f"  {result.name}")
    print(f"{'='*W}")

    # 토큰 목록
    print(f"\n[Tokens]  ({len(result.tokens)} 개)")
    for i, t in enumerate(result.tokens):
        gt_mark   = " ←GT"  if i in result.gt_positions else ""
        pred_mark = " ←PRED" if i in result.confirmed_positions else ""
        print(f"  [{i:2d}] {t.text:<18} {t.start:6.3f}s ~ {t.end:6.3f}s{gt_mark}{pred_mark}")

    # 경계 분석 표
    print(f"\n[Boundary Analysis]")
    hdr = f"  {'#':>3} | {'Before':>14} {'After':>14} | {'Pros':>5} {'Text':>5} {'Sem':>5} | {'Total':>6} | GT Pred"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in result.boundaries:
        gt_m   = "✓ " if b.position in result.gt_positions else "  "
        pr_m   = "✓"  if b.confirmed                       else " "
        print(f"  {b.position:>3} | {b.word_before.text:>14} {b.word_after.text:>14} "
              f"| {b.prosody_score:>5.3f} {b.text_score:>5.3f} {b.semantic_score:>5.3f} "
              f"| {b.total_score:>6.3f} | {gt_m}{pr_m}")

    # 평가
    print(f"\n[Evaluation]")
    print(f"  GT   positions : {ev['gt']}")
    print(f"  Pred positions : {ev['pred']}")
    print(f"  Precision={ev['precision']:.3f}  Recall={ev['recall']:.3f}  F1={ev['f1']:.3f}"
          f"   (TP={ev['tp']} FP={ev['fp']} FN={ev['fn']})")

    # 피처 상세 (GT 경계만)
    if result.gt_positions:
        print(f"\n[Feature Detail at GT boundaries]")
        for b in result.boundaries:
            if b.position in result.gt_positions:
                print(f"  pos={b.position}  before='{b.word_before.text}'  after='{b.word_after.text}'")
                for feat, det in b.details.items():
                    print(f"    {feat}: {det}")


def print_translation_result(tr: Dict) -> None:
    """GT 번역 vs 예측 번역 비교 출력."""
    if not tr["gt_translations"] and not tr["pred_translations"]:
        print("\n[Translation] deep-translator 미설치 (pip install deep-translator)")
        return

    lang = tr["target_lang"]
    print(f"\n[Translation Evaluation]  (→ {lang})")

    print(f"\n  GT 분절 ({len(tr['gt_segments'])}개):")
    for i, (src, tgt) in enumerate(zip(tr["gt_segments"], tr["gt_translations"])):
        print(f"  [{i+1}] {src}")
        print(f"       → {tgt}")

    print(f"\n  PRED 분절 ({len(tr['pred_segments'])}개):")
    for i, (src, tgt) in enumerate(zip(tr["pred_segments"], tr["pred_translations"])):
        print(f"  [{i+1}] {src}")
        print(f"       → {tgt}")

    if tr["similarity"] is not None:
        print(f"\n  번역 의미 유사도 (GT vs PRED): {tr['similarity']:.3f}")
    else:
        print(f"\n  번역 의미 유사도: (embed_fn 없음)")


def print_feature_analysis(all_results: List[SegmentationResult]) -> None:
    """GT 경계 vs 비경계 평균 피처 점수 비교."""
    at_gt     = {"prosody": [], "text": [], "semantic": [], "total": []}
    not_at_gt = {"prosody": [], "text": [], "semantic": [], "total": []}

    for result in all_results:
        gt_set = set(result.gt_positions)
        for b in result.boundaries:
            target = at_gt if b.position in gt_set else not_at_gt
            target["prosody"].append(b.prosody_score)
            target["text"].append(b.text_score)
            target["semantic"].append(b.semantic_score)
            target["total"].append(b.total_score)

    print(f"\n{'='*50}")
    print(f"  Feature Analysis (전체 케이스 합산)")
    print(f"{'='*50}")
    print(f"  {'Feature':>12} | {'GT경계 평균':>10} | {'비경계 평균':>10} | {'차이':>8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for feat in ["prosody", "text", "semantic", "total"]:
        gt_mean  = np.mean(at_gt[feat])     if at_gt[feat]     else 0.0
        non_mean = np.mean(not_at_gt[feat]) if not_at_gt[feat] else 0.0
        diff     = gt_mean - non_mean
        print(f"  {feat:>12} | {gt_mean:>10.3f} | {non_mean:>10.3f} | {diff:>+8.3f}")
    print(f"\n  GT 경계 수    : {len(at_gt['total'])}")
    print(f"  비경계 수     : {len(not_at_gt['total'])}")


# ─── 진입점 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Meaning Segmentation Test")
    parser.add_argument("--device",         default="cuda:0",             help="cuda:0 또는 cpu")
    parser.add_argument("--no-semantic",    action="store_true",          help="Semantic 피처 비활성화")
    parser.add_argument("--window",         type=int,   default=DEFAULT_PEAK_WINDOW,    help="Peak 검출 좌우 window 크기")
    parser.add_argument("--min-prominence", type=float, default=DEFAULT_MIN_PROMINENCE, help="Peak 최소 prominence")
    parser.add_argument("--min-floor",      type=float, default=DEFAULT_MIN_FLOOR,      help="Peak 후보 최소 점수 (noise floor)")
    parser.add_argument("--min-distance",   type=int,   default=DEFAULT_MIN_DISTANCE,   help="Peak 간 최소 거리 (NMS)")
    parser.add_argument("--language",       default="ko",                              help="입력 오디오 언어 코드 (기본: ko, 예: en, ja)")
    parser.add_argument("--target-lang",    default="en",                              help="번역 목적 언어 (기본: en)")
    parser.add_argument("--no-translate",   action="store_true",                       help="번역 평가 비활성화")
    args = parser.parse_args()

    # ── Qwen3ForcedAligner 로드 ──────────────────────────────────────────────
    import torch
    from qwen_asr import Qwen3ForcedAligner

    print("=" * 50)
    print("  Qwen3ForcedAligner 로딩 중...")
    print("=" * 50)
    try:
        dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
        aligner = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=dtype,
            device_map=args.device,
        )
        print("  ✓ Aligner 로드 완료")
    except Exception as e:
        print(f"  ✗ Aligner 로드 실패: {e}")
        sys.exit(1)

    # ── Text 피처 형태소 분석기 로드 ────────────────────────────────────────
    okt = None
    nlp = None
    if args.language == "ko":
        try:
            from konlpy.tag import Okt
            print("\n  KoNLPy Okt 로딩 중...")
            okt = Okt()
            print("  ✓ Text 피처 (KoNLPy Okt) 활성화")
        except ImportError:
            print("  ⚠ konlpy 없음. Text 피처 비활성화.")
            print("    설치: pip install konlpy")
    else:
        try:
            import spacy
            model_name = SPACY_MODEL_MAP.get(args.language, f"{args.language}_core_news_sm")
            print(f"\n  spaCy 모델 로딩 중... ({model_name})")
            nlp = spacy.load(model_name)
            print(f"  ✓ Text 피처 (spaCy {model_name}) 활성화")
        except ImportError:
            print("  ⚠ spacy 없음. Text 피처 비활성화.")
            print("    설치: pip install spacy")
        except OSError:
            print(f"  ⚠ spaCy 모델 '{model_name}' 없음. Text 피처 비활성화.")
            print(f"    설치: python -m spacy download {model_name}")

    # ── 가중치 조정 (활성화된 피처에 따라 자동 정규화됨) ────────────────────
    weights = dict(DEFAULT_WEIGHTS)
    if okt is None and nlp is None:
        weights["text"] = 0.0

    # ── Sentence-Transformers 로드 ───────────────────────────────────────────
    embed_fn = None
    if not args.no_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            print("\n  Sentence-Transformers 로딩 중...")
            embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embed_fn = lambda t: embed_model.encode(t, convert_to_numpy=True, show_progress_bar=False)
            print("  ✓ Semantic 피처 활성화")
        except ImportError:
            print("  ⚠ sentence-transformers 없음. Semantic 피처 비활성화.")
            print("    설치: pip install sentence-transformers")

    # ── 테스트 케이스 ────────────────────────────────────────────────────────
    test_cases = [
        ("sentence1", "sentence1.m4a", "answer1.txt"),
        ("sentence2", "sentence2.m4a", "answer2.txt"),
        ("sentence3", "sentence3.m4a", "answer3.txt"),
    ]

    all_results: List[SegmentationResult] = []
    all_evals:   List[Dict] = []

    for name, audio_file, answer_file in test_cases:
        audio_path  = str(_AUDIO_DIR / audio_file)
        answer_path = str(_TEXT_DIR  / answer_file)

        if not os.path.exists(audio_path):
            print(f"\n[SKIP] {name}: 오디오 파일 없음 ({audio_path})")
            continue

        with open(answer_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()

        print(f"\n>> {name} 처리 중...")
        try:
            result = run_test_case(
                name=name,
                audio_path=audio_path,
                gt_answer=gt_text,
                aligner=aligner,
                embed_fn=embed_fn,
                weights=weights,
                use_semantic=(not args.no_semantic and embed_fn is not None),
                device=args.device,
                language=args.language,
                okt=okt,
                nlp=nlp,
                window=args.window,
                min_prominence=args.min_prominence,
                min_floor=args.min_floor,
                min_distance=args.min_distance,
            )
            ev = evaluate(result)
            print_result(result, ev)
            if not args.no_translate:
                tr = translation_evaluate(result, source_lang=args.language, target_lang=args.target_lang, embed_fn=embed_fn)
                print_translation_result(tr)
            all_results.append(result)
            all_evals.append(ev)
        except Exception as e:
            import traceback
            print(f"  ✗ 오류: {e}")
            traceback.print_exc()

    # ── 전체 요약 ────────────────────────────────────────────────────────────
    if all_evals:
        print_feature_analysis(all_results)

        print(f"\n{'='*50}")
        print(f"  OVERALL SUMMARY  ({len(all_evals)}개 케이스)")
        print(f"{'='*50}")
        print(f"  Avg Precision : {np.mean([e['precision'] for e in all_evals]):.3f}")
        print(f"  Avg Recall    : {np.mean([e['recall']    for e in all_evals]):.3f}")
        print(f"  Avg F1        : {np.mean([e['f1']        for e in all_evals]):.3f}")
        print(f"  Peak window   : ±{args.window}")
        print(f"  Min prominence: {args.min_prominence}")
        print(f"  Min floor     : {args.min_floor}")
        print(f"  Min distance  : {args.min_distance}")
        print(f"  Weights       : {DEFAULT_WEIGHTS}")
        print()


if __name__ == "__main__":
    main()
