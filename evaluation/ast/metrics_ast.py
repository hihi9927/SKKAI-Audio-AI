#!/usr/bin/env python3
"""AST 평가 지표 — LAAL(지연) + BLEU(품질).

**모든 시간 단위는 밀리초(ms)다.** 서버가 보내는 값은 초 단위이므로 클라이언트에서
변환해 들어온다.

LAAL (Length-Adaptive Average Lagging, Papi et al. 2022,
"Over-Generation Cannot Be Rewarded: Length-Adaptive Average Lagging for
Simultaneous Speech Translation"):

    LAAL = (1/τ) · Σ_{i=1..τ} [ d_i − (i−1) · T / max(|Y_hyp|, |Y_ref|) ]
    τ    = min{ i : d_i ≥ T },  그런 i가 없으면 |Y_hyp|

d_i 는 i번째 타깃 단위가 나온 시점의 **지연**, T 는 소스 오디오 길이다. AL과 다른 점은
분모가 max(|Y_hyp|, |Y_ref|) 라는 것 하나다. AL은 분모가 |Y_hyp| 라서 짧게 생성할수록
(i−1)·γ 항이 커져 지연이 작게 나오는 — 즉 과소생성이 보상받는 — 구멍이 있다.

우리 파이프라인은 토큰이 아니라 **세그먼트 단위로 커밋**하므로, 한 세그먼트에 속한
타깃 단위는 모두 같은 d를 공유한다 (chunk-level SimulST의 표준 처리).

d를 무엇으로 두느냐에 따라 두 가지를 모두 보고한다:
  - LAAL      (non-computation-aware): d = 커밋을 결정한 순간까지 읽은 소스 오디오 길이
                                       (`decisionAudioSec`). 정책만 평가, 하드웨어 무관.
  - LAAL_CA   (computation-aware)    : d = 클라이언트가 `final`을 받은 실시간 경과.
                                       계산 비용 포함, 실제 체감.
검산: LAAL_CA − LAAL ≈ mean(fsl). 크게 어긋나면 타이밍 배선이 틀린 것이다.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

# ── 비음성 마커 ───────────────────────────────────────────────────────────────
# TED 자막에는 (Laughter) / (Gelächter) 같은 이벤트 표기가 들어 있다. ASR은 이걸
# 내놓지 않으므로 참조에 남겨두면 BLEU가 부당하게 깎인다. 제거 여부는 점수를 바꾸므로
# 반드시 metric.json 에 기록한다.
_NONSPEECH_WORDS = (
    # 영어
    "laughter", "laughs", "applause", "cheers", "cheering", "music", "singing",
    "video", "audio", "recording", "silence", "sighs", "beat", "sniffs",
    "laughter and applause", "clapping", "boos", "gasps",
    # 독일어 (MuST-C en-de 참조측)
    "gelächter", "lachen", "applaus", "beifall", "musik", "gesang", "jubel",
    "stille", "seufzt", "klatschen",
)
_NONSPEECH_RE = re.compile(
    r"[\(\[]\s*(?:" + "|".join(re.escape(w) for w in _NONSPEECH_WORDS) + r")\s*[\)\]]",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


def strip_nonspeech(text: str) -> str:
    """`(Laughter)` / `(Gelächter)` 같은 비음성 이벤트 표기를 제거한다.

    괄호 전체를 지우지 않고 알려진 이벤트 단어만 지운다 — 본문에 등장하는 정상적인
    괄호(예: "(2005)")까지 날리면 참조가 훼손된다.
    """
    if not text:
        return ""
    return _WS_RE.sub(" ", _NONSPEECH_RE.sub(" ", text)).strip()


# ── 단위 세기 ────────────────────────────────────────────────────────────────

def count_units(text: str, unit: str = "word") -> int:
    """LAAL의 |Y| 를 센다.

    word — 공백 어절. en/de/es 등 띄어쓰기 언어용.
    char — 공백 제외 문자. zh/ja 처럼 어절 개념이 없는 언어용.

    무엇을 고르든 LAAL 값이 달라지므로 선택을 meta.json 에 남겨야 비교가 성립한다.
    """
    if not text:
        return 0
    if unit == "word":
        return len(text.split())
    if unit == "char":
        return len(_WS_RE.sub("", text))
    raise ValueError(f"unknown unit: {unit!r} (word|char)")


def expand_delays(
    segments: Sequence[tuple[str, float]], unit: str = "word"
) -> list[float]:
    """[(번역문, 지연ms)] → 타깃 단위별 지연 리스트.

    세그먼트 안의 모든 단위는 같은 지연을 갖는다. 빈 번역은 단위 0개라 자동으로 빠진다.
    """
    delays: list[float] = []
    for text, delay_ms in segments:
        n = count_units(text, unit)
        if n <= 0:
            continue
        delays.extend([float(delay_ms)] * n)
    return delays


# ── LAAL ─────────────────────────────────────────────────────────────────────

def compute_laal(
    delays_ms: Sequence[float],
    src_duration_ms: float,
    n_ref_units: Optional[int] = None,
) -> Optional[float]:
    """LAAL(ms). 계산 불가면 None.

    delays_ms  — 타깃 단위별 지연. 오름차순일 필요는 없으나 실제로는 단조 증가한다.
    n_ref_units — 참조 번역의 단위 수. None이면 AL과 같은 정의(분모 = |Y_hyp|)가 된다.
    """
    n_hyp = len(delays_ms)
    if n_hyp == 0 or src_duration_ms <= 0:
        return None

    denom_units = max(n_hyp, n_ref_units or 0)
    gamma = src_duration_ms / denom_units

    # τ = 소스를 다 읽은 뒤 나온 첫 타깃 단위. 없으면 전체를 쓴다.
    tau = n_hyp
    for i, d in enumerate(delays_ms, start=1):
        if d >= src_duration_ms:
            tau = i
            break

    total = 0.0
    for i in range(1, tau + 1):
        total += delays_ms[i - 1] - (i - 1) * gamma
    return total / tau


def laal_for_utterance(
    segments: Sequence[tuple[str, float]],
    src_duration_ms: float,
    ref_text: str = "",
    unit: str = "word",
) -> Optional[float]:
    """발화 하나의 LAAL(ms). segments = [(세그먼트 번역, 지연ms)]."""
    delays = expand_delays(segments, unit)
    n_ref = count_units(ref_text, unit) if ref_text else None
    return compute_laal(delays, src_duration_ms, n_ref)


# ── BLEU ─────────────────────────────────────────────────────────────────────

# sacrebleu 토크나이저 기본값. 다른 토크나이저로 낸 점수는 서로 비교 불가이므로
# 결과에 반드시 signature 를 남긴다.
DEFAULT_TOKENIZE = {
    "de": "13a", "en": "13a", "es": "13a", "fr": "13a", "it": "13a",
    "nl": "13a", "pt": "13a", "ro": "13a", "ru": "13a", "cs": "13a",
    "zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab",
}


def resolve_tokenize(target_lang: str) -> str:
    return DEFAULT_TOKENIZE.get((target_lang or "").lower(), "13a")


def corpus_bleu_score(
    hypotheses: Sequence[str], references: Sequence[str], tokenize: str = "13a"
) -> tuple[Optional[float], Optional[str]]:
    """corpus BLEU 와 sacrebleu signature 를 돌려준다. 실패 시 (None, 사유)."""
    try:
        import sacrebleu
    except ImportError:
        return None, "sacrebleu-not-installed"
    if not hypotheses:
        return None, "empty-hypotheses"

    try:
        metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
        score = metric.corpus_score(list(hypotheses), [list(references)])
    except Exception as exc:  # ja-mecab / ko-mecab 미설치 등
        if tokenize in ("ja-mecab", "ko-mecab"):
            metric = sacrebleu.metrics.BLEU(tokenize="char")
            score = metric.corpus_score(list(hypotheses), [list(references)])
            return score.score, str(metric.get_signature()) + " [fallback:char]"
        return None, f"sacrebleu-error: {exc}"
    return score.score, str(metric.get_signature())


def sentence_bleu_score(
    hypothesis: str, reference: str, tokenize: str = "13a"
) -> Optional[float]:
    """세그먼트/발화 단위 BLEU. 주지표는 corpus BLEU 이고 이건 보조다 —
    짧은 문장에서 값이 크게 튄다."""
    try:
        import sacrebleu
    except ImportError:
        return None
    if not hypothesis or not reference:
        return None
    try:
        metric = sacrebleu.metrics.BLEU(tokenize=tokenize, effective_order=True)
        return metric.sentence_score(hypothesis, [reference]).score
    except Exception:
        return None


# ── 집계 ─────────────────────────────────────────────────────────────────────

def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
