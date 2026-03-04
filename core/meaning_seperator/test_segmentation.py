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
DEFAULT_WEIGHTS        = {"prosody": 0.5, "text": 0, "semantic": 0.5}
DEFAULT_PEAK_WINDOW    = 3      # 좌우 몇 개 이웃까지 비교할지
DEFAULT_MIN_PROMINENCE = 0.10   # peak가 이웃 평균보다 얼마나 높아야 하는지
DEFAULT_MIN_FLOOR      = 0.15   # 이 값 미만이면 noise로 간주, peak 후보 제외
DEFAULT_MIN_DISTANCE   = 2      # NMS: peak 간 최소 간격 (토큰 수)
TARGET_SR              = 16000

# 담화표지 / 화제전환 신호 (word_after 기준)
DISCOURSE_MARKERS = {
    "그리고", "그런데", "근데", "그래서", "그럼", "그러면", "또한",
    "하지만", "그러나", "또", "그다음에", "다음에", "이제", "자",
    "한편", "반면", "게다가", "따라서", "그리하여", "그나저나",
}
TOPIC_SIGNALS = {"자", "이제", "그럼", "그러면", "다음으로", "다음은", "이번엔", "이번에는"}

# 접속어미 접미사 (word_before 기준)
CONNECTIVE_SUFFIXES = ["고", "서", "는데", "인데", "지만", "거나", "면서", "다가", "고서"]


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
) -> Tuple[float, Dict]:
    details: Dict = {}
    score = 0.0

    # 1. Gap (silence between words) ── 가중 0.35
    gap = max(0.0, w_after.start - w_before.end)
    g_score = min(1.0, gap / 0.3)   # 0.3 s 이상이면 1.0
    details["gap_s"]    = round(gap, 4)
    details["gap_sc"]   = round(g_score, 3)
    score += 0.35 * g_score

    # 2. Final lengthening ── 가중 0.25
    local_start = max(0, all_tokens.index(w_before) - 3)
    local_end   = min(len(all_tokens), all_tokens.index(w_before) + 3)
    local_durs  = [t.duration for t in all_tokens[local_start:local_end] if t.duration > 0.01]
    mean_dur    = float(np.mean(local_durs)) if local_durs else 0.15
    fl_ratio    = w_before.duration / max(mean_dur, 1e-6)
    fl_score    = min(1.0, max(0.0, (fl_ratio - 0.8) / 1.2))
    details["fl_ratio"] = round(fl_ratio, 3)
    details["fl_sc"]    = round(fl_score, 3)
    score += 0.25 * fl_score

    # 3. F0 drop (마지막 200 ms) ── 가중 0.25
    try:
        import librosa
        seg_start = max(0,           int(w_before.start * sr))
        seg_end   = min(len(audio),  int(w_before.end   * sr))
        seg       = audio[seg_start:seg_end]
        if len(seg) >= sr * 0.08:
            f0, voiced, _ = librosa.pyin(seg, fmin=70, fmax=500, sr=sr,
                                          frame_length=512, hop_length=128)
            f0v = f0[voiced] if voiced is not None else np.array([])
            if len(f0v) >= 4:
                half    = len(f0v) // 2
                f0_drop = (np.mean(f0v[:half]) - np.mean(f0v[half:])) / max(np.mean(f0v[:half]), 1.0)
                f0_sc   = min(1.0, max(0.0, float(f0_drop) * 3))
                details["f0_drop"] = round(float(f0_drop), 3)
                details["f0_sc"]   = round(f0_sc, 3)
                score += 0.25 * f0_sc
            else:
                details["f0_note"] = "voiced_too_short"
        else:
            details["f0_note"] = "seg_too_short"
    except Exception as e:
        details["f0_err"] = str(e)

    # 4. Energy drop ── 가중 0.15
    try:
        win = int(0.1 * sr)
        b_end   = int(w_before.end   * sr)
        a_start = int(w_after.start  * sr)
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


# ─── Feature 2: Text ───────────────────────────────────────────────────────────
def text_score(
    w_before: WordToken, w_after: WordToken,
) -> Tuple[float, Dict]:
    details: Dict = {}
    score = 0.0

    # 담화표지 (word_after)
    after = w_after.text
    dm = after in DISCOURSE_MARKERS or any(after.startswith(m) for m in DISCOURSE_MARKERS)
    details["dm_after"] = dm
    if dm:
        score += 0.5

    # 접속어미 (word_before 끝)
    before = w_before.text
    conn = any(before.endswith(suf) for suf in CONNECTIVE_SUFFIXES)
    details["conn_before"] = conn
    if conn:
        score += 0.3

    # 화제 전환 신호 (word_after)
    ts = after in TOPIC_SIGNALS
    details["topic_sig"] = ts
    if ts:
        score += 0.2

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
    align_results = aligner.align(
        audio=(audio, sr),
        text=clean_text,
        language="Korean",
    )
    raw_tokens = align_results[0]
    tokens = [WordToken(t.text, float(t.start_time), float(t.end_time)) for t in raw_tokens]

    # 4. GT 경계 인덱스
    gt_positions = get_gt_boundary_indices(tokens, gt_segments)

    # 5. Pass 1: 각 경계에서 피처 점수 계산 (confirmed는 아직 미정)
    boundaries: List[BoundaryScore] = []
    for i in range(len(tokens) - 1):
        wb, wa = tokens[i], tokens[i + 1]

        p_sc, p_det = prosody_score(audio, sr, wb, wa, tokens)
        t_sc, t_det = text_score(wb, wa)
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

    # 6. Pass 2: 전체 점수 분포를 보고 local prominence peak 확정
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
        ("sentence1",      "sentence1.m4a",      "answer1.txt"),
        ("sentence1_fast", "sentence1_fast.m4a",  "answer1.txt"),
        ("sentence2",      "sentence2.m4a",       "answer2.txt"),
        ("sentence2_fast", "sentence2_fast.m4a",  "answer2.txt"),
        ("sentence3",      "sentence3.m4a",       "answer3.txt"),
        ("sentence3_fast", "sentence3_fast.m4a",  "answer3.txt"),
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
                weights=DEFAULT_WEIGHTS,
                use_semantic=(not args.no_semantic and embed_fn is not None),
                device=args.device,
                window=args.window,
                min_prominence=args.min_prominence,
                min_floor=args.min_floor,
                min_distance=args.min_distance,
            )
            ev = evaluate(result)
            print_result(result, ev)
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
