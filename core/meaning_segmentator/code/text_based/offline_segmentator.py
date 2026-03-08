#!/usr/bin/env python3
"""
Meaning Segmentation Test
=========================
PCM 파일 기반 의미 분절 테스트 (GT 없음, 결과 저장).

피처:
  1. Prosody  - gap, final lengthening, f0 drop, energy drop
  2. Text     - 담화표지/접속어미/화제전환 신호
  3. Semantic - sentence-transformers 코사인 비유사도 (다중 창 크기)
  4. Entropy  - ASR 토큰 생성 시 모델 불확실성 (output_scores 기반)

경계 확정:
  고정 임계값 대신 local prominence 기반 peak detection.
  각 위치의 점수가 주변(±window) 평균보다 min_prominence 이상 높아야 peak.
  NMS(non-maximum suppression)로 너무 가까운 peak 중 강한 것만 유지.

사용법:
  cd STiTy/
  python core/meaning_seperator/code/text_based/test_segmentation.py
  python core/meaning_seperator/code/text_based/test_segmentation.py --num-files 2
  python core/meaning_seperator/code/text_based/test_segmentation.py --no-semantic --no-entropy
  python core/meaning_seperator/code/text_based/test_segmentation.py --output results.json
"""

import sys
import os
import re
import json
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_BASE      = _HERE.parent.parent.parent   # STiTy/
_QWEN_DIR  = _BASE / "Qwen3-ASR"
_DATA_DIR    = _HERE.parent.parent / "data" / "eval_clean"
_RESULTS_DIR = _HERE.parent.parent / "results" / "feature_based"
_TEST_DIR  = _HERE / "test"
_AUDIO_DIR = _TEST_DIR / "audio"
_TEXT_DIR  = _TEST_DIR / "text"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_QWEN_DIR))

# ─── 상수 ─────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS        = {"prosody": 0.30, "text": 0.30, "semantic": 0.15, "entropy": 0.25}
DEFAULT_PEAK_WINDOW    = 3
DEFAULT_MIN_PROMINENCE = 0.20
DEFAULT_MIN_FLOOR      = 0.15
DEFAULT_MIN_DISTANCE   = 2
TARGET_SR              = 16000
DEFAULT_MAX_ENTROPY_REF = 0.5  # 정규화 기준 엔트로피 (nats); 실제 범위: 2~10

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
    "Eomi":        0.7,
    "Punctuation": 1.0,
}
BEFORE_VERBAL_EOMI_BONUS = 0.4

AFTER_POS_SCORE: Dict[str, float] = {
    "Exclamation": 0.7,
    "Adverb":      0.7,
}

BEFORE_AUX_VERB_BONUS_UD = 0.2

# ── Text Feature: SpaCy POS 점수표 (영어) ─────────────────────────────
BEFORE_POS_SCORE_UD: Dict[str, float] = {
    "PUNCT": 1.0,
    "VERB":  0.4,
    "AUX":   0.2,
    "PART":  0.1,
    "NOUN":  0.2,
    "PRON":  0.1,
    "ADJ":   0.15,
    "ADV":   0.2,
    "NUM":   0.15,
}

BEFORE_MORPH_BONUS: Dict[str, float] = {
    "Tense=Past":    0.3,
    "Tense=Pres":    0.15,
    "VerbForm=Fin":  0.25,
    "VerbForm=Inf":  -0.2,
    "VerbForm=Part": 0.1,
    "Mood=Ind":      0.15,
    "Mood=Imp":      0.3,
    "Number=Plur":   0.05,
}

AFTER_POS_SCORE_UD: Dict[str, float] = {
    "INTJ":  0.6,
    "ADV":   0.3,
    "CCONJ": 0.4,
    "SCONJ": 0.5,
    "DET":   0.2,
    "PRON":  0.3,
    "PROPN": 0.25,
    "NUM":   0.15,
}

CONTINUITY_PENALTY_PATTERNS = [
    ("VERB", "PART"),
    ("NOUN", "REL"),
    ("ADJ", "SCONJ"),
    ("AUX", "VERB"),
]

DEP_SCORE: Dict[str, float] = {
    "ROOT":      0.3,
    "conj":     -0.1,
    "advcl":    -0.15,
    "relcl":    -0.2,
    "mark":     -0.1,
    "cc":        0.2,
}

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
    start: float
    end:   float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class BoundaryScore:
    position:       int
    word_before:    WordToken
    word_after:     WordToken
    prosody_score:  float = 0.0
    text_score:     float = 0.0
    semantic_score: float = 0.0
    entropy_score:  float = 0.0
    total_score:    float = 0.0
    confirmed:      bool  = False
    details:        Dict  = field(default_factory=dict)


@dataclass
class SegmentationResult:
    name:                str
    tokens:              List[WordToken]
    boundaries:          List[BoundaryScore]
    confirmed_positions: List[int]
    gt_positions:        List[int] = field(default_factory=list)
    transcript:          str       = ""


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


def load_pcm(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Raw PCM (16-bit signed int, mono, 16kHz) 로드 → float32 [-1, 1]."""
    data = np.fromfile(path, dtype=np.int16)
    audio = data.astype(np.float32) / 32768.0
    return audio, sr


# ─── GT 파싱 ───────────────────────────────────────────────────────────────────
def parse_gt(text_with_markers: str) -> Tuple[str, List[str]]:
    parts = [p.strip() for p in text_with_markers.split("/") if p.strip()]
    clean = " ".join(parts)
    return clean, parts


def get_gt_boundary_indices(tokens: List[WordToken], gt_segments: List[str]) -> List[int]:
    char_ends = []
    acc = 0
    for t in tokens:
        acc += len(t.text.replace(" ", ""))
        char_ends.append(acc)

    boundary_indices = []
    seg_char_acc = 0
    for seg in gt_segments[:-1]:
        seg_char_acc += len(seg.replace(" ", ""))
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
    details: Dict = {}
    score = 0.0

    gap = max(0.0, w_after.start - w_before.end)
    z = (gap - gap_mean) / max(gap_std, 1e-6)
    g_score = min(1.0, max(0.0, (z + 2) / 4))
    details["gap_s"]  = round(gap, 4)
    details["gap_z"]  = round(z, 3)
    details["gap_sc"] = round(g_score, 3)
    score += 0.30 * g_score

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

        if len(f0v_b) >= 4:
            half    = len(f0v_b) // 2
            f0_drop = (np.mean(f0v_b[:half]) - np.mean(f0v_b[half:])) / max(np.mean(f0v_b[:half]), 1.0)
            f0_drop_sc = min(1.0, max(0.0, float(f0_drop) * 3))
            details["f0_drop"]    = round(float(f0_drop), 3)
            details["f0_drop_sc"] = round(f0_drop_sc, 3)
            score += 0.20 * f0_drop_sc
        else:
            details["f0_drop_note"] = "too_short"

        if len(f0v_b) >= 2 and len(f0v_a) >= 2:
            f0_before_end  = float(np.mean(f0v_b[len(f0v_b) // 2:]))
            f0_after_start = float(np.mean(f0v_a[:max(1, len(f0v_a) // 2)]))
            f0_reset       = f0_after_start - f0_before_end
            reset_sc       = min(1.0, max(0.0, f0_reset / 80.0))
            details["f0_reset_hz"] = round(f0_reset, 1)
            details["reset_sc"]    = round(reset_sc, 3)
            score += 0.15 * reset_sc
        else:
            details["reset_note"] = "insufficient_f0"

    except Exception as e:
        details["f0_err"] = str(e)

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


# ─── Feature 2: Text ──────────────────────────────────────────────────────────
def text_score(
    w_before: WordToken, w_after: WordToken,
    okt=None,
    nlp=None,
) -> Tuple[float, Dict]:
    details: Dict = {}
    score = 0.0

    if okt is not None:
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
                morph_str = str(last_tok.morph)
                morph_bonus = sum(v for k, v in BEFORE_MORPH_BONUS.items() if k in morph_str)
                if morph_bonus != 0.0:
                    details["before_morph_bonus"] = round(morph_bonus, 3)
                    base_sc += morph_bonus
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


# ─── Feature 4: Entropy ───────────────────────────────────────────────────────
def get_token_entropies(
    asr_model,
    audio: np.ndarray,
    sr: int,
    language: str = "Korean",
    max_new_tokens: int = 512,
) -> Tuple[str, List[float]]:
    """
    ASR 모델을 output_scores=True로 실행해 전사 텍스트와
    생성된 각 LLM 토큰의 엔트로피(nats)를 반환.

    Returns:
        transcript (str): 전사 텍스트
        entropies (List[float]): 토큰별 엔트로피, len == 생성 토큰 수
    """
    import torch

    hf_model  = asr_model.model      # Qwen3ASRForConditionalGeneration
    processor = asr_model.processor

    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]
    text_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    text_prompt = text_prompt + f"language {language}<asr_text>"

    inputs = processor(
        text=[text_prompt],
        audio=[audio],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(hf_model.device).to(hf_model.dtype)

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
        )

    input_len    = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_len:]
    transcript   = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # 각 생성 스텝의 엔트로피 (nats)
    entropies: List[float] = []
    for step_logits in outputs.scores:          # [batch, vocab_size]
        probs = torch.softmax(step_logits[0].float(), dim=-1)
        ent   = -(probs * torch.log(probs + 1e-9)).sum().item()
        entropies.append(ent)

    return transcript, entropies


def map_entropies_to_words(
    token_entropies: List[float],
    num_words: int,
) -> List[float]:
    """
    T개 LLM 토큰 엔트로피를 W개 단어 엔트로피로 비례 매핑.
    단어 i는 토큰 인덱스 [i*T/W, (i+1)*T/W) 구간의 평균 엔트로피.
    """
    if not token_entropies or num_words == 0:
        return [0.0] * num_words
    T      = len(token_entropies)
    result = []
    for wi in range(num_words):
        t_start = int(wi * T / num_words)
        t_end   = max(t_start + 1, int((wi + 1) * T / num_words))
        t_end   = min(t_end, T)
        result.append(float(np.mean(token_entropies[t_start:t_end])))
    return result


def entropy_score_at_boundary(
    position: int,
    word_entropies: List[float],
    max_entropy_ref: float = DEFAULT_MAX_ENTROPY_REF,
) -> Tuple[float, Dict]:
    """
    경계 position에서의 엔트로피 점수.
    word_before와 word_after의 엔트로피 평균을 max_entropy_ref로 정규화.
    """
    details: Dict = {}
    if not word_entropies or position >= len(word_entropies):
        return 0.0, {"note": "no_entropy"}

    e_before = word_entropies[position]                                       if position     < len(word_entropies) else 0.0
    e_after  = word_entropies[position + 1] if (position + 1) < len(word_entropies) else 0.0
    raw      = (e_before + e_after) / 2.0
    score    = min(1.0, raw / max(max_entropy_ref, 1e-6))

    details["e_before_nats"] = round(e_before, 3)
    details["e_after_nats"]  = round(e_after,  3)
    details["raw_avg_nats"]  = round(raw,       3)
    details["score"]         = round(score,     3)
    return score, details


# ─── Peak Detection ────────────────────────────────────────────────────────────
def find_boundary_peaks(
    boundaries:     List[BoundaryScore],
    window:         int   = DEFAULT_PEAK_WINDOW,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_floor:      float = DEFAULT_MIN_FLOOR,
    min_distance:   int   = DEFAULT_MIN_DISTANCE,
) -> List[int]:
    scores = [b.total_score for b in boundaries]
    n = len(scores)

    candidates: List[Tuple[int, float]] = []
    for i in range(n):
        s = scores[i]
        if s < min_floor:
            continue
        left  = scores[max(0, i - window): i]
        right = scores[i + 1: min(n, i + 1 + window)]
        left_mean  = float(np.mean(left))  if left  else 0.0
        right_mean = float(np.mean(right)) if right else 0.0
        if left and right:
            context_mean = (left_mean + right_mean) / 2
        else:
            context_mean = max(left_mean, right_mean)
        if s - context_mean >= min_prominence:
            candidates.append((i, s))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    confirmed: List[int] = []
    for idx, _ in candidates:
        if all(abs(idx - c) >= min_distance for c in confirmed):
            confirmed.append(idx)
    return sorted(confirmed)


# ─── PCM 파이프라인 (GT 없음) ──────────────────────────────────────────────────
def run_pcm_case(
    name:            str,
    audio_path:      str,
    asr_model,
    aligner,
    embed_fn:        Optional[Callable],
    weights:         Dict[str, float],
    use_semantic:    bool,
    use_entropy:     bool,
    language:        str   = "ko",
    okt=None,
    nlp=None,
    window:          int   = DEFAULT_PEAK_WINDOW,
    min_prominence:  float = DEFAULT_MIN_PROMINENCE,
    min_floor:       float = DEFAULT_MIN_FLOOR,
    min_distance:    int   = DEFAULT_MIN_DISTANCE,
    max_entropy_ref: float = DEFAULT_MAX_ENTROPY_REF,
    max_new_tokens:  int   = 512,
) -> SegmentationResult:
    """
    PCM 오디오 파일 처리:
      1. PCM 로드
      2. ASR (output_scores=True) → 전사 텍스트 + 토큰별 엔트로피
      3. ForcedAligner → 단어별 타임스탬프
      4. 피처 계산 (prosody / text / semantic / entropy)
      5. Peak detection → 분절 경계 확정
    """
    aligner_lang = LANGUAGE_MAP.get(language, language.capitalize())

    # 1. PCM 로드
    audio, sr = load_pcm(audio_path)

    # 2. ASR + entropy
    transcript   = ""
    word_entropies: List[float] = []

    if use_entropy and asr_model is not None:
        transcript, token_entropies = get_token_entropies(
            asr_model, audio, sr,
            language=aligner_lang,
            max_new_tokens=max_new_tokens,
        )
    elif asr_model is not None:
        # entropy 비활성화 — 전사만
        from qwen_asr.inference.utils import normalize_audios
        results = asr_model.transcribe(
            audio=(audio, sr),
            language=aligner_lang,
            return_time_stamps=False,
        )
        transcript   = results[0].text if results else ""
        token_entropies = []
    else:
        raise RuntimeError("asr_model이 필요합니다.")

    if not transcript.strip():
        return SegmentationResult(
            name=name, tokens=[], boundaries=[],
            confirmed_positions=[], transcript=transcript,
        )

    # 3. ForcedAligner → 단어 타임스탬프
    align_results = aligner.align(
        audio=(audio, sr),
        text=transcript,
        language=aligner_lang,
    )
    raw_tokens = align_results[0]
    tokens = [WordToken(t.text, float(t.start_time), float(t.end_time)) for t in raw_tokens]

    if not tokens:
        return SegmentationResult(
            name=name, tokens=[], boundaries=[],
            confirmed_positions=[], transcript=transcript,
        )

    # 4a. 엔트로피 → 단어별 매핑
    if use_entropy and token_entropies:
        word_entropies = map_entropies_to_words(token_entropies, len(tokens))
    else:
        word_entropies = [0.0] * len(tokens)

    # 4b. Gap 통계
    all_gaps = [max(0.0, tokens[i + 1].start - tokens[i].end) for i in range(len(tokens) - 1)]
    gap_mean = float(np.mean(all_gaps)) if all_gaps else 0.0
    gap_std  = float(np.std(all_gaps))  if all_gaps else 0.1

    # 4c. 경계별 피처 계산
    boundaries: List[BoundaryScore] = []
    for i in range(len(tokens) - 1):
        wb, wa = tokens[i], tokens[i + 1]

        p_sc, p_det = prosody_score(audio, sr, wb, wa, tokens, gap_mean=gap_mean, gap_std=gap_std)
        t_sc, t_det = text_score(wb, wa, okt=okt, nlp=nlp)
        s_sc, s_det = (semantic_score(tokens, i, embed_fn)
                       if use_semantic else (0.0, {"note": "disabled"}))
        e_sc, e_det = (entropy_score_at_boundary(i, word_entropies, max_entropy_ref=max_entropy_ref)
                       if use_entropy else (0.0, {"note": "disabled"}))

        w_p = weights.get("prosody",  0.30)
        w_t = weights.get("text",     0.30)
        w_s = weights.get("semantic", 0.15) if use_semantic else 0.0
        w_e = weights.get("entropy",  0.25) if use_entropy  else 0.0
        total_w = w_p + w_t + w_s + w_e
        total   = (w_p * p_sc + w_t * t_sc + w_s * s_sc + w_e * e_sc) / max(total_w, 1e-8)

        boundaries.append(BoundaryScore(
            position=i,
            word_before=wb, word_after=wa,
            prosody_score=round(p_sc, 3),
            text_score=round(t_sc, 3),
            semantic_score=round(s_sc, 3),
            entropy_score=round(e_sc, 3),
            total_score=round(total, 3),
            confirmed=False,
            details={"prosody": p_det, "text": t_det, "semantic": s_det, "entropy": e_det},
        ))

    # 5. Peak detection
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
        gt_positions=[],
        transcript=transcript,
    )


# ─── 결과 출력 ────────────────────────────────────────────────────────────────
def print_pcm_result(result: SegmentationResult) -> None:
    W = 80
    print(f"\n{'='*W}")
    print(f"  {result.name}")
    if result.transcript:
        preview = result.transcript[:80].replace("\n", " ")
        print(f"  전사: {preview}{'...' if len(result.transcript) > 80 else ''}")
    print(f"{'='*W}")

    if not result.tokens:
        print("  (토큰 없음 — 묵음 또는 전사 실패)")
        return

    # 분절 결과
    segments = get_segments_text(result.tokens, result.confirmed_positions)
    print(f"\n[Segments]  ({len(segments)}개)")
    for i, seg in enumerate(segments):
        print(f"  [{i+1:2d}] {seg}")

    # 토큰 목록
    print(f"\n[Tokens]  ({len(result.tokens)} 개)")
    for i, t in enumerate(result.tokens):
        mark = " ←BOUNDARY" if i in result.confirmed_positions else ""
        print(f"  [{i:2d}] {t.text:<18} {t.start:6.3f}s ~ {t.end:6.3f}s{mark}")

    # 경계 분석 표
    print(f"\n[Boundary Scores]")
    hdr = f"  {'#':>3} | {'Before':>14} {'After':>14} | {'Pros':>5} {'Text':>5} {'Sem':>5} {'Ent':>5} | {'Total':>6} | Confirmed"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in result.boundaries:
        mark = "✓" if b.confirmed else " "
        print(f"  {b.position:>3} | {b.word_before.text:>14} {b.word_after.text:>14} "
              f"| {b.prosody_score:>5.3f} {b.text_score:>5.3f} "
              f"{b.semantic_score:>5.3f} {b.entropy_score:>5.3f} "
              f"| {b.total_score:>6.3f} | {mark}")


# ─── 분절 텍스트 추출 ────────────────────────────────────────────────────────
def get_segments_text(tokens: List[WordToken], boundary_positions: List[int]) -> List[str]:
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


# ─── JSON 저장 ────────────────────────────────────────────────────────────────
def result_to_dict(result: SegmentationResult) -> Dict:
    segments = get_segments_text(result.tokens, result.confirmed_positions)
    return {
        "name":       result.name,
        "transcript": result.transcript,
        "num_tokens": len(result.tokens),
        "segments": [
            {
                "index": i + 1,
                "text":  seg,
                "start": result.tokens[result.confirmed_positions[i - 1] + 1].start
                         if i > 0 and (i - 1) < len(result.confirmed_positions)
                         else (result.tokens[0].start if result.tokens else 0.0),
                "end":   result.tokens[result.confirmed_positions[i]].end
                         if i < len(result.confirmed_positions)
                         else (result.tokens[-1].end if result.tokens else 0.0),
            }
            for i, seg in enumerate(segments)
        ],
        "boundaries": [
            {
                "position":      b.position,
                "word_before":   b.word_before.text,
                "word_after":    b.word_after.text,
                "start":         b.word_before.start,
                "end":           b.word_after.end,
                "prosody":       b.prosody_score,
                "text":          b.text_score,
                "semantic":      b.semantic_score,
                "entropy":       b.entropy_score,
                "total":         b.total_score,
                "confirmed":     b.confirmed,
            }
            for b in result.boundaries
        ],
        "confirmed_positions": result.confirmed_positions,
    }


def save_results_json(results: List[SegmentationResult], path: str) -> None:
    data = [result_to_dict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {path}")


# ─── (기존 호환) GT 기반 평가 ─────────────────────────────────────────────────
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


def print_feature_analysis(all_results: List[SegmentationResult]) -> None:
    """전체 피처 분포 요약 (GT 없으므로 전체 평균/분산)."""
    keys = ["prosody", "text", "semantic", "entropy", "total"]
    accum: Dict[str, List[float]] = {k: [] for k in keys}

    for result in all_results:
        for b in result.boundaries:
            accum["prosody"].append(b.prosody_score)
            accum["text"].append(b.text_score)
            accum["semantic"].append(b.semantic_score)
            accum["entropy"].append(b.entropy_score)
            accum["total"].append(b.total_score)

    if not accum["total"]:
        return

    print(f"\n{'='*55}")
    print(f"  Feature Distribution (전체 경계 합산)")
    print(f"{'='*55}")
    print(f"  {'Feature':>12} | {'Mean':>6} | {'Std':>6} | {'Max':>6} | {'Min':>6}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for feat in keys:
        vals = accum[feat]
        if vals:
            print(f"  {feat:>12} | {np.mean(vals):>6.3f} | {np.std(vals):>6.3f} "
                  f"| {np.max(vals):>6.3f} | {np.min(vals):>6.3f}")


# ─── 진입점 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Meaning Segmentation (PCM + Entropy)")
    parser.add_argument("--device",          default="cuda:0",                        help="cuda:0 또는 cpu")
    parser.add_argument("--language",        default="ko",                            help="언어 코드 (기본: ko)")
    parser.add_argument("--num-files",       type=int,  default=None,                 help="처리할 PCM 파일 수 (기본: 전부)")
    parser.add_argument("--audio-dir",       default=None,                            help="PCM 파일 디렉터리 (기본: data/eval_clean)")
    parser.add_argument("--output",          default=None,                            help="결과 JSON 저장 경로")
    parser.add_argument("--no-semantic",     action="store_true",                     help="Semantic 피처 비활성화")
    parser.add_argument("--no-entropy",      action="store_true",                     help="Entropy 피처 비활성화")
    parser.add_argument("--max-entropy-ref", type=float, default=DEFAULT_MAX_ENTROPY_REF, help="엔트로피 정규화 기준값(nats)")
    parser.add_argument("--max-new-tokens",  type=int,   default=512,                 help="ASR 최대 생성 토큰 수")
    parser.add_argument("--window",          type=int,   default=DEFAULT_PEAK_WINDOW,    help="Peak 검출 window 크기")
    parser.add_argument("--min-prominence",  type=float, default=DEFAULT_MIN_PROMINENCE, help="Peak 최소 prominence")
    parser.add_argument("--min-floor",       type=float, default=DEFAULT_MIN_FLOOR,      help="Peak noise floor")
    parser.add_argument("--min-distance",    type=int,   default=DEFAULT_MIN_DISTANCE,   help="Peak 최소 간격 (NMS)")
    args = parser.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    use_entropy  = not args.no_entropy
    use_semantic = not args.no_semantic

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  모델 로딩 중...")
    print("=" * 55)
    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32

    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=dtype,
        device_map=args.device,
    )
    print("  ✓ Qwen3ASRModel 로드 완료")

    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=dtype,
        device_map=args.device,
    )
    print("  ✓ Qwen3ForcedAligner 로드 완료")

    # ── Text 피처 ────────────────────────────────────────────────────────────
    okt = None
    nlp = None
    if args.language == "ko":
        try:
            from konlpy.tag import Okt
            okt = Okt()
            print("  ✓ KoNLPy Okt 활성화")
        except ImportError:
            print("  ⚠ konlpy 없음 → Text 피처 비활성화 (pip install konlpy)")
    else:
        try:
            import spacy
            model_name = SPACY_MODEL_MAP.get(args.language, f"{args.language}_core_news_sm")
            nlp = spacy.load(model_name)
            print(f"  ✓ spaCy {model_name} 활성화")
        except (ImportError, OSError) as e:
            print(f"  ⚠ spaCy 없음 또는 모델 미설치 → Text 피처 비활성화 ({e})")

    # ── Semantic 피처 ────────────────────────────────────────────────────────
    embed_fn = None
    if use_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embed_fn = lambda t: embed_model.encode(t, convert_to_numpy=True, show_progress_bar=False)
            print("  ✓ Sentence-Transformers 활성화")
        except ImportError:
            print("  ⚠ sentence-transformers 없음 → Semantic 비활성화")
            use_semantic = False

    # ── 가중치 ──────────────────────────────────────────────────────────────
    weights = dict(DEFAULT_WEIGHTS)
    if okt is None and nlp is None:
        weights["text"] = 0.0

    # ── PCM 파일 목록 ────────────────────────────────────────────────────────
    audio_dir = Path(args.audio_dir) if args.audio_dir else _DATA_DIR
    pcm_files = sorted(audio_dir.glob("*.pcm"))
    if not pcm_files:
        print(f"\n  ✗ PCM 파일 없음: {audio_dir}")
        sys.exit(1)

    if args.num_files is not None:
        pcm_files = pcm_files[:args.num_files]

    print(f"\n  처리 대상: {len(pcm_files)}개 PCM 파일 ({audio_dir})")
    if use_entropy:
        print(f"  Entropy 피처: ON  (max_ref={args.max_entropy_ref} nats)")
    else:
        print(f"  Entropy 피처: OFF")

    # ── 처리 루프 ────────────────────────────────────────────────────────────
    all_results: List[SegmentationResult] = []

    for pcm_path in pcm_files:
        name = pcm_path.stem
        print(f"\n>> {name} 처리 중...")
        try:
            result = run_pcm_case(
                name=name,
                audio_path=str(pcm_path),
                asr_model=asr_model,
                aligner=aligner,
                embed_fn=embed_fn,
                weights=weights,
                use_semantic=use_semantic,
                use_entropy=use_entropy,
                language=args.language,
                okt=okt,
                nlp=nlp,
                window=args.window,
                min_prominence=args.min_prominence,
                min_floor=args.min_floor,
                min_distance=args.min_distance,
                max_entropy_ref=args.max_entropy_ref,
                max_new_tokens=args.max_new_tokens,
            )
            print_pcm_result(result)
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"  ✗ 오류: {e}")
            traceback.print_exc()

    # ── 요약 ─────────────────────────────────────────────────────────────────
    if all_results:
        print_feature_analysis(all_results)

        total_segs = sum(len(r.confirmed_positions) + 1 for r in all_results if r.tokens)
        total_toks = sum(len(r.tokens) for r in all_results)
        print(f"\n{'='*55}")
        print(f"  SUMMARY  ({len(all_results)}개 파일)")
        print(f"{'='*55}")
        print(f"  총 단어 수       : {total_toks}")
        print(f"  총 분절 수       : {total_segs}")
        print(f"  평균 분절 길이   : {total_toks / max(total_segs, 1):.1f} 단어")
        print(f"  Peak window      : ±{args.window}")
        print(f"  Min prominence   : {args.min_prominence}")
        print(f"  Min floor        : {args.min_floor}")
        print(f"  Min distance     : {args.min_distance}")
        print(f"  Weights          : {weights}")
        print()

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    if args.output:
        save_results_json(all_results, args.output)
    else:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        default_out = str(_RESULTS_DIR / "segmentation_results.json")
        save_results_json(all_results, default_out)


if __name__ == "__main__":
    main()
