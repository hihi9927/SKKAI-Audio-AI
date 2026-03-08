#!/usr/bin/env python3
"""
LibriSpeech 기반 의미 분절 평가
================================
LibriSpeech 폴더 구조(evaluation/data/{speaker}/{book}/)를 탐색하여
화자/책 단위로 의미 분절 성능을 번역 유사도 기준으로 평가합니다.

평가 방식:
  GT   : trans.txt의 각 utterance를 독립 번역 → 전체 연결
  PRED : 예측 분절 경계 기준으로 텍스트 분할 번역 → 전체 연결
  Score: GT 전체 번역 vs PRED 전체 번역의 코사인 유사도 (sentence-transformers)

폴더 구조:
  evaluation/data/{speaker_id}/{book_id}/
    {speaker_id}-{book_id}-0000.flac
    {speaker_id}-{book_id}-0001.flac
    ...
    {speaker_id}-{book_id}.trans.txt

사용법:
  python core/meaning_seperator/eval_librispeech.py
  python core/meaning_seperator/eval_librispeech.py --data-dir evaluation/data
  python core/meaning_seperator/eval_librispeech.py --language en --target-lang ko
  python core/meaning_seperator/eval_librispeech.py --max-utterances 10
"""

import sys
import os
import argparse
import logging
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_BASE     = _HERE.parent.parent   # STiTy/
_QWEN_DIR = _BASE / "Qwen3-ASR"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_QWEN_DIR))
sys.path.insert(0, str(_HERE))   # test_segmentation 직접 import를 위해

# test_segmentation에서 필요한 부분만 가져오기
from test_segmentation import (
    WordToken, BoundaryScore,
    load_audio,
    prosody_score, text_score, semantic_score,
    find_boundary_peaks,
    translate_segments,
    get_segments_text,
    LANGUAGE_MAP, SPACY_MODEL_MAP,
    DEFAULT_WEIGHTS, DEFAULT_PEAK_WINDOW, DEFAULT_MIN_PROMINENCE,
    DEFAULT_MIN_FLOOR, DEFAULT_MIN_DISTANCE, TARGET_SR,
)


# ─── 로그 설정 ─────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("librispeech_eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# ─── LibriSpeech 파싱 ─────────────────────────────────────────────────────────
def parse_trans_file(trans_path: Path) -> List[Tuple[str, str]]:
    """
    {speaker}-{book}.trans.txt 파싱.
    Returns: [(utt_id, text), ...]
      예: [("121-121726-0000", "ALSO A POPULAR CONTRIVANCE..."), ...]
    """
    utterances = []
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                utterances.append((parts[0], parts[1].strip()))
    return utterances


def load_folder(
    folder: Path,
    max_utterances: Optional[int] = None,
) -> Tuple[Optional[Path], List[Tuple[str, str, Path]]]:
    """
    speaker/book 폴더에서 trans.txt와 오디오 파일을 로드.
    Returns: (trans_path, [(utt_id, text, audio_path), ...])
    """
    trans_files = list(folder.glob("*.trans.txt"))
    if not trans_files:
        return None, []

    trans_path = trans_files[0]
    utterances = parse_trans_file(trans_path)
    if max_utterances:
        utterances = utterances[:max_utterances]

    result = []
    for utt_id, text in utterances:
        for ext in (".flac", ".wav", ".mp3", ".m4a"):
            audio_path = folder / f"{utt_id}{ext}"
            if audio_path.exists():
                result.append((utt_id, text, audio_path))
                break

    return trans_path, result


# ─── 오디오 연결 ──────────────────────────────────────────────────────────────
def concatenate_audios(
    audio_paths: List[Path],
    sr: int = TARGET_SR,
    silence_sec: float = 0.3,
) -> Tuple[np.ndarray, List[float]]:
    """
    여러 오디오를 순서대로 이어붙임.
    silence_sec: 발화 사이에 삽입할 무음 길이(초)
    Returns: (concat_audio, utt_end_times) — 각 발화의 끝 시점(초)
    """
    silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
    segments: List[np.ndarray] = []
    utt_end_times: List[float] = []
    cumulative = 0.0

    for path in audio_paths:
        audio, _ = load_audio(str(path), sr=sr)
        segments.append(audio)
        segments.append(silence)
        cumulative += len(audio) / sr + silence_sec
        utt_end_times.append(cumulative - silence_sec)

    if not segments:
        return np.array([], dtype=np.float32), []

    return np.concatenate(segments), utt_end_times


# ─── 분절 파이프라인 ──────────────────────────────────────────────────────────
def run_segmentation(
    audio:         np.ndarray,
    sr:            int,
    full_text:     str,
    aligner,
    embed_fn:      Optional[Callable],
    weights:       Dict[str, float],
    use_semantic:  bool,
    language:      str = "en",
    okt=None,
    nlp=None,
    window:         int   = DEFAULT_PEAK_WINDOW,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_floor:      float = DEFAULT_MIN_FLOOR,
    min_distance:   int   = DEFAULT_MIN_DISTANCE,
) -> Tuple[List[WordToken], List[int], List[BoundaryScore]]:
    """
    오디오 + 텍스트 → (토큰 리스트, 예측 경계 위치, 경계 점수 리스트).
    GT 경계 비교 없이 예측 경계만 반환.
    """
    aligner_lang = LANGUAGE_MAP.get(language, language.capitalize())
    align_results = aligner.align(
        audio=(audio, sr),
        text=full_text,
        language=aligner_lang,
    )
    raw_tokens = align_results[0]
    tokens = [WordToken(t.text, float(t.start_time), float(t.end_time)) for t in raw_tokens]

    if len(tokens) < 2:
        return tokens, [], []

    all_gaps = [max(0.0, tokens[i + 1].start - tokens[i].end) for i in range(len(tokens) - 1)]
    gap_mean = float(np.mean(all_gaps)) if all_gaps else 0.0
    gap_std  = float(np.std(all_gaps))  if all_gaps else 0.1

    boundaries: List[BoundaryScore] = []
    for i in range(len(tokens) - 1):
        wb, wa = tokens[i], tokens[i + 1]

        p_sc, p_det = prosody_score(audio, sr, wb, wa, tokens, gap_mean=gap_mean, gap_std=gap_std)
        t_sc, t_det = text_score(wb, wa, okt=okt, nlp=nlp)
        s_sc, s_det = semantic_score(tokens, i, embed_fn) if use_semantic else (0.0, {})

        w_p = weights["prosody"]
        w_t = weights["text"]
        w_s = weights["semantic"] if use_semantic else 0.0
        total_w = w_p + w_t + w_s
        total   = (w_p * p_sc + w_t * t_sc + w_s * s_sc) / max(total_w, 1e-8)

        boundaries.append(BoundaryScore(
            position=i, word_before=wb, word_after=wa,
            prosody_score=round(p_sc, 3),
            text_score=round(t_sc, 3),
            semantic_score=round(s_sc, 3),
            total_score=round(total, 3),
        ))

    pred_positions = find_boundary_peaks(
        boundaries,
        window=window, min_prominence=min_prominence,
        min_floor=min_floor, min_distance=min_distance,
    )
    return tokens, pred_positions, boundaries


# ─── 유사도 계산 ──────────────────────────────────────────────────────────────
def cosine_similarity(
    text_a: str,
    text_b: str,
    embed_fn: Callable,
) -> float:
    ea = embed_fn(text_a)
    eb = embed_fn(text_b)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


# ─── 폴더 단위 평가 ───────────────────────────────────────────────────────────
def evaluate_folder(
    folder:         Path,
    aligner,
    embed_fn:       Optional[Callable],
    weights:        Dict[str, float],
    use_semantic:   bool,
    language:       str,
    target_lang:    str,
    okt=None,
    nlp=None,
    max_utterances: Optional[int] = None,
    window:         int   = DEFAULT_PEAK_WINDOW,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_floor:      float = DEFAULT_MIN_FLOOR,
    min_distance:   int   = DEFAULT_MIN_DISTANCE,
    logger:         Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """
    하나의 speaker/book 폴더를 평가.
    GT  : 각 utterance 독립 번역 → 연결
    PRED: 예측 분절 기준 번역 → 연결
    Returns: 결과 dict or None (실패 시)
    """
    log = logger.info if logger else print

    trans_path, folder_data = load_folder(folder, max_utterances)
    if not folder_data:
        return None

    speaker_id = folder.parent.name
    book_id    = folder.name
    utt_ids    = [x[0] for x in folder_data]
    utt_texts  = [x[1] for x in folder_data]
    audio_paths = [x[2] for x in folder_data]

    W = 70
    log(f"\n{'='*W}")
    log(f"  Speaker: {speaker_id}  |  Book: {book_id}  |  발화 수: {len(folder_data)}")
    log(f"{'='*W}")

    # 오디오 연결
    try:
        concat_audio, utt_end_times = concatenate_audios(audio_paths)
    except Exception as e:
        log(f"  [오류] 오디오 연결 실패: {e}")
        return None

    if len(concat_audio) == 0:
        log("  [오류] 오디오 없음")
        return None

    full_text   = " ".join(utt_texts)
    audio_dur   = len(concat_audio) / TARGET_SR
    log(f"\n  텍스트: {len(full_text)}자  |  오디오: {audio_dur:.1f}초")

    # 분절 실행
    try:
        tokens, pred_positions, boundaries = run_segmentation(
            concat_audio, TARGET_SR, full_text,
            aligner, embed_fn, weights, use_semantic,
            language=language, okt=okt, nlp=nlp,
            window=window, min_prominence=min_prominence,
            min_floor=min_floor, min_distance=min_distance,
        )
    except Exception as e:
        log(f"  [오류] 분절 실패: {e}")
        log(traceback.format_exc())
        return None

    log(f"  토큰: {len(tokens)}개  |  예측 경계: {len(pred_positions)}개  위치: {pred_positions}")

    # ── GT 번역: 각 utterance 독립 번역 ─────────────────────────────────────
    if language == target_lang:
        gt_translations   = list(utt_texts)
        pred_translations = list(get_segments_text(tokens, pred_positions))
    else:
        gt_translations   = translate_segments(utt_texts, source=language, target=target_lang)
        pred_segs         = get_segments_text(tokens, pred_positions)
        pred_translations = translate_segments(pred_segs, source=language, target=target_lang)

    pred_segs = get_segments_text(tokens, pred_positions)

    # ── 유사도 계산 ──────────────────────────────────────────────────────────
    similarity: Optional[float] = None
    if embed_fn and gt_translations and pred_translations:
        try:
            gt_full   = " ".join(gt_translations)
            pred_full = " ".join(pred_translations)
            similarity = round(max(0.0, cosine_similarity(gt_full, pred_full, embed_fn)), 3)
        except Exception as e:
            log(f"  [경고] 유사도 계산 실패: {e}")

    # ── 출력 ─────────────────────────────────────────────────────────────────
    log(f"\n  ┌─ GT 번역 ({len(gt_translations)}개 utterance → {target_lang})")
    for i, (src, tgt) in enumerate(zip(utt_texts, gt_translations)):
        src_preview = src[:70] + ("..." if len(src) > 70 else "")
        tgt_preview = tgt[:70] + ("..." if len(tgt) > 70 else "") if tgt else ""
        log(f"  │  [{i+1:02d}] {src_preview}")
        log(f"  │       → {tgt_preview}")

    log(f"\n  ├─ PRED 번역 ({len(pred_segs)}개 분절 → {target_lang})")
    for i, (src, tgt) in enumerate(zip(pred_segs, pred_translations)):
        src_preview = src[:70] + ("..." if len(src) > 70 else "")
        tgt_preview = tgt[:70] + ("..." if len(tgt) > 70 else "") if tgt else ""
        log(f"  │  [{i+1:02d}] {src_preview}")
        log(f"  │       → {tgt_preview}")

    if similarity is not None:
        log(f"\n  └─ 번역 유사도 (GT vs PRED): {similarity:.3f}")
    else:
        log(f"\n  └─ 번역 유사도: N/A (embed_fn 없음)")

    # ── 경계 점수 상위 10개 ──────────────────────────────────────────────────
    if boundaries:
        log(f"\n  [경계 점수 상위 10]")
        top = sorted(boundaries, key=lambda b: b.total_score, reverse=True)[:10]
        log(f"  {'pos':>4} | {'before':>14} {'after':>14} | {'Pros':>5} {'Text':>5} {'Sem':>5} | {'Total':>6} | Pred")
        log(f"  {'----':>4}-+-{'----------':>14}-{'----------':>14}-+-{'----':>5}-{'----':>5}-{'----':>5}-+-{'-----':>6}-+-----")
        pred_set = set(pred_positions)
        for b in top:
            mark = " ✓" if b.position in pred_set else "  "
            log(f"  {b.position:>4} | {b.word_before.text:>14} {b.word_after.text:>14} "
                f"| {b.prosody_score:>5.3f} {b.text_score:>5.3f} {b.semantic_score:>5.3f} "
                f"| {b.total_score:>6.3f} |{mark}")

    return {
        "speaker_id":        speaker_id,
        "book_id":           book_id,
        "n_utterances":      len(folder_data),
        "n_tokens":          len(tokens),
        "n_pred_boundaries": len(pred_positions),
        "pred_positions":    pred_positions,
        "similarity":        similarity,
    }


# ─── 진입점 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LibriSpeech 기반 의미 분절 평가")
    parser.add_argument("--data-dir",       default="evaluation/data",
                        help="LibriSpeech 데이터 루트 (기본: evaluation/data)")
    parser.add_argument("--log-dir",        default="core/meaning_seperator/results",
                        help="결과 저장 폴더 (기본: core/meaning_seperator/results)")
    parser.add_argument("--device",         default="cuda:0",
                        help="cuda:0 또는 cpu")
    parser.add_argument("--language",       default="en",
                        help="입력 오디오 언어 코드 (기본: en)")
    parser.add_argument("--target-lang",    default="ko",
                        help="번역 목적 언어 (기본: ko)")
    parser.add_argument("--no-semantic",    action="store_true",
                        help="Semantic 피처 비활성화")
    parser.add_argument("--max-utterances", type=int, default=0,
                        help="폴더당 최대 발화 수 (기본: 0=전체)")
    parser.add_argument("--silence-sec",    type=float, default=0.3,
                        help="발화 사이 무음 길이 초 (기본: 0.3)")
    parser.add_argument("--window",         type=int,   default=DEFAULT_PEAK_WINDOW)
    parser.add_argument("--min-prominence", type=float, default=DEFAULT_MIN_PROMINENCE)
    parser.add_argument("--min-floor",      type=float, default=DEFAULT_MIN_FLOOR)
    parser.add_argument("--min-distance",   type=int,   default=DEFAULT_MIN_DISTANCE)
    parser.add_argument("--speaker",        type=str,   default=None,
                        help="특정 화자 ID만 실행 (예: --speaker 1089). 쉼표로 여러 개 지정 가능: 1089,121")
    parser.add_argument("--book",           type=str,   default=None,
                        help="특정 book ID만 실행 (예: --book 134686). 쉼표로 여러 개 지정 가능: 134686,121726")
    args = parser.parse_args()

    speaker_filter = {s.strip() for s in args.speaker.split(",")} if args.speaker else None
    book_filter    = {b.strip() for b in args.book.split(",")}    if args.book    else None

    max_utt = args.max_utterances if args.max_utterances > 0 else None

    # ── 로그 설정 ─────────────────────────────────────────────────────────────
    data_dir = _BASE / args.data_dir
    log_dir  = _BASE / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"librispeech_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger   = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("  LibriSpeech 의미 분절 평가")
    logger.info("=" * 70)
    logger.info(f"  데이터    : {data_dir}")
    logger.info(f"  언어      : {args.language} → {args.target_lang}")
    logger.info(f"  최대 발화 : {max_utt or '전체'}")
    logger.info(f"  로그 파일 : {log_path}")

    # ── Qwen3ForcedAligner ────────────────────────────────────────────────────
    import torch
    from qwen_asr import Qwen3ForcedAligner

    logger.info("\nQwen3ForcedAligner 로딩 중...")
    try:
        dtype   = torch.bfloat16 if "cuda" in args.device else torch.float32
        aligner = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=dtype, device_map=args.device,
        )
        logger.info("✓ Aligner 로드 완료")
    except Exception as e:
        logger.error(f"✗ Aligner 로드 실패: {e}")
        sys.exit(1)

    # ── Text 피처 형태소 분석기 ───────────────────────────────────────────────
    okt = None
    nlp = None
    if args.language == "ko":
        try:
            from konlpy.tag import Okt
            okt = Okt()
            logger.info("✓ KoNLPy Okt 활성화 (한국어)")
        except ImportError:
            logger.info("⚠ konlpy 없음 → Text 피처 비활성화")
    else:
        try:
            import spacy
            model_name = SPACY_MODEL_MAP.get(args.language, f"{args.language}_core_news_sm")
            nlp = spacy.load(model_name)
            logger.info(f"✓ spaCy 활성화 ({model_name})")
        except ImportError:
            logger.info("⚠ spacy 없음 → Text 피처 비활성화 (pip install spacy)")
        except OSError:
            logger.info(f"⚠ spaCy 모델 없음 → Text 피처 비활성화 (python -m spacy download {model_name})")

    weights = dict(DEFAULT_WEIGHTS)
    if okt is None and nlp is None:
        weights["text"] = 0.0

    # ── Sentence-Transformers ─────────────────────────────────────────────────
    embed_fn = None
    if not args.no_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embed_fn = lambda t: embed_model.encode(t, convert_to_numpy=True, show_progress_bar=False)
            logger.info("✓ Sentence-Transformers 활성화")
        except ImportError:
            logger.info("⚠ sentence-transformers 없음 → Semantic/유사도 비활성화")

    use_semantic = not args.no_semantic and embed_fn is not None
    logger.info(f"\n  가중치: {weights}  |  Semantic: {use_semantic}")

    # ── 폴더 탐색 ─────────────────────────────────────────────────────────────
    all_results: List[Dict] = []

    if not data_dir.exists():
        logger.error(f"[오류] 데이터 폴더 없음: {data_dir}")
        sys.exit(1)

    # evaluation/data/{speaker_id}/{book_id}/
    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        if speaker_filter and speaker_dir.name not in speaker_filter:
            continue
        for book_dir in sorted(speaker_dir.iterdir()):
            if not book_dir.is_dir():
                continue
            if book_filter and book_dir.name not in book_filter:
                continue

            result = evaluate_folder(
                folder=book_dir,
                aligner=aligner,
                embed_fn=embed_fn,
                weights=weights,
                use_semantic=use_semantic,
                language=args.language,
                target_lang=args.target_lang,
                okt=okt,
                nlp=nlp,
                max_utterances=max_utt,
                window=args.window,
                min_prominence=args.min_prominence,
                min_floor=args.min_floor,
                min_distance=args.min_distance,
                logger=logger,
            )
            if result:
                all_results.append(result)

    # ── 전체 요약 ─────────────────────────────────────────────────────────────
    W = 70
    logger.info(f"\n\n{'='*W}")
    logger.info(f"  OVERALL SUMMARY  ({len(all_results)}개 폴더)")
    logger.info(f"{'='*W}")
    logger.info(f"  {'Speaker':>8} {'Book':>8} {'Uttrs':>6} {'Tokens':>7} {'PredBnd':>8} {'Sim':>7}")
    logger.info(f"  {'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")

    sims = []
    for r in all_results:
        sim_str = f"{r['similarity']:.3f}" if r["similarity"] is not None else "N/A"
        logger.info(
            f"  {r['speaker_id']:>8} {r['book_id']:>8} "
            f"{r['n_utterances']:>6} {r['n_tokens']:>7} "
            f"{r['n_pred_boundaries']:>8} {sim_str:>7}"
        )
        if r["similarity"] is not None:
            sims.append(r["similarity"])

    if sims:
        logger.info(f"\n  평균 유사도 : {np.mean(sims):.3f}")
        logger.info(f"  최고 유사도 : {np.max(sims):.3f}")
        logger.info(f"  최저 유사도 : {np.min(sims):.3f}")

    logger.info(f"\n  로그 저장 완료: {log_path}")


if __name__ == "__main__":
    main()
