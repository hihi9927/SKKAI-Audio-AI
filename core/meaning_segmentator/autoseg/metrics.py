"""A5 Scorer — 결정론적.

2축 목적함수를 계산한다. 품질(Q) 단일 축으로 최적화하면 최적해가
"<SEG>를 하나도 넣지 않는 프롬프트"로 수렴하므로 지연축(L)이 반드시 필요하다.

  Q : seg 번역 이어붙임 vs full 번역의 의미 유사도. 분절이 없으면 1.0 (제외하지 않음).
  L : 스트리밍 지연 프록시. 1.0 = 무분절, 0.5 = 무한 세분. 낮을수록 빠름.
  V : 포맷 검증 통과율. 1.0 미만이면 그 프롬프트는 탈락.

품질 백엔드는 교체 가능:
  embed (기본) — 게이트웨이 임베딩 코사인 유사도. 로컬 ML 의존성 불필요.
  chrf         — 문자 n-gram F-score. 보조 지표로 항상 함께 리포트.
  comet        — 미구현. unbabel-comet 설치 시 여기에 연결 (운영 기준 지표).
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, asdict

from .gateway import Gateway
from .pipeline import split_segments

_WS = re.compile(r"\s+")


# ── chrF ─────────────────────────────────────────────────────────────────

def _char_ngrams(s: str, n: int) -> Counter:
    s = _WS.sub("", s)
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """문자 n-gram F-score. 표면형 일치도만 보므로 의미 판정에는 부족하다 —
    Q의 보조 지표로만 쓴다."""
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


# ── 임베딩 코사인 ────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_similarity(gw: Gateway, hyps: list[str], refs: list[str]) -> list[float]:
    """분절 번역과 전체 번역의 의미 유사도. 두 문자열이 동일하면 호출 없이 1.0."""
    pending = [i for i, (h, r) in enumerate(zip(hyps, refs)) if h != r and h and r]
    scores = [1.0] * len(hyps)
    for i, (h, r) in enumerate(zip(hyps, refs)):
        if not h or not r:
            scores[i] = 0.0
    if not pending:
        return scores
    texts = [hyps[i] for i in pending] + [refs[i] for i in pending]
    vecs = gw.embed(texts)
    half = len(pending)
    for j, i in enumerate(pending):
        scores[i] = _cosine(vecs[j], vecs[half + j])
    return scores


# ── 지연 프록시 ──────────────────────────────────────────────────────────

def latency_proxy(seg_text: str, spaced: bool) -> float:
    """세그먼트 i가 나가기까지 통과해야 하는 문장 비율의 평균.

      조각 길이 c_1..c_k, N = Σc_i
      L = Σ_i (c_1+...+c_i) / (k · N)

    k=1 → 1.0 (무분절). k→∞ → 0.5. 낮을수록 먼저 내보낼 수 있다.
    """
    parts = split_segments(seg_text)
    if not parts:
        return 1.0
    lens = [len(p if spaced else _WS.sub("", p)) for p in parts]
    total = sum(lens)
    if total == 0 or len(lens) == 1:
        return 1.0
    cum, acc = 0, 0
    for c in lens:
        cum += c
        acc += cum
    return acc / (len(lens) * total)


# ── 집계 ─────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    n: int
    valid_rate: float          # V (복구 재시도 이후)
    first_pass_valid_rate: float  # 복구 없이 한 번에 포맷을 지킨 비율
    quality: float             # Q 전체 평균 (무분절 = 1.0 포함) — 보고용
    quality_segmented: float   # 분절된 문장만의 Q — 제약은 이쪽에 건다
    quality_segmented_std: float
    n_segmented: int
    quality_chrf: float        # 보조
    latency: float             # L
    latency_gain: float        # 1 - L
    mean_segments: float       # k 평균
    segmented_rate: float      # <SEG> 1개 이상인 문장 비율
    quality_min: float
    quality_p10: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def aggregate(
    quality_scores: list[float],
    chrf_scores: list[float],
    seg_texts: list[str],
    valid_flags: list[bool],
    spaced: bool,
    first_pass_flags: list[bool] | None = None,
) -> Metrics:
    n = len(seg_texts)
    lat = [latency_proxy(s, spaced) for s in seg_texts]
    ks = [max(1, len(split_segments(s))) for s in seg_texts]
    seg_q = [q for q, k in zip(quality_scores, ks) if k > 1]
    q_sorted = sorted(quality_scores) if quality_scores else [0.0]
    p10_idx = max(0, int(0.10 * (len(q_sorted) - 1)))
    return Metrics(
        n=n,
        valid_rate=sum(valid_flags) / n if n else 0.0,
        first_pass_valid_rate=(sum(first_pass_flags) / n) if (first_pass_flags and n) else (sum(valid_flags) / n if n else 0.0),
        quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        quality_segmented=(sum(seg_q) / len(seg_q)) if seg_q else 1.0,
        quality_segmented_std=(statistics.pstdev(seg_q) if len(seg_q) > 1 else 0.0),
        n_segmented=len(seg_q),
        quality_chrf=sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0,
        latency=sum(lat) / len(lat) if lat else 1.0,
        latency_gain=1.0 - (sum(lat) / len(lat) if lat else 1.0),
        mean_segments=sum(ks) / len(ks) if ks else 1.0,
        segmented_rate=sum(1 for k in ks if k > 1) / n if n else 0.0,
        quality_min=q_sorted[0],
        quality_p10=q_sorted[p10_idx],
    )


def quality_lcb(m: Metrics, z: float = 1.0) -> float:
    """분절 품질의 하한 신뢰구간.

    Qs 를 평균만으로 쓰면 표본이 작을 때 판정이 잡음에 좌우된다 (ja 실측: dev 30문장
    중 분절된 것이 4건뿐이라 문장 하나가 Qs 를 0.05 이상 움직였다). 표본이 작을수록
    불리하게 만들어, 루프가 **품질을 충분한 수의 분절 문장에서 입증**하도록 강제한다.
    분절을 적게 하는 프롬프트는 그만큼 입증 부담을 지므로, 지연축과 같은 방향으로
    압력이 걸린다.
    """
    n = m.n_segmented
    if n <= 1:
        return m.quality_segmented
    return m.quality_segmented - z * (m.quality_segmented_std / (n ** 0.5))


def objective(m: Metrics, q_floor: float) -> float:
    """제약 최적화: V=1 이고 Q_segmented>=Q_floor 인 조건에서 지연 이득을 최대화.

    품질 제약을 **분절된 문장에만** 거는 것이 핵심이다. 전 문장 평균을 쓰면
    무분절 문장의 1.0 이 평균을 끌어올려, "적게 분절하면 나쁘게 잘라도 통과"하는
    역인센티브가 생긴다 (ja 실측: 분절률 0.24에서 전체 Q 0.9668, 분절 문장만
    보면 0.8618 로 floor 미달). Q_floor 를 전 문장을 자르는 기계분절로
    캘리브레이션하므로 축을 맞추는 것이기도 하다.

    가중합 대신 제약형을 쓰는 이유는 λ 튜닝이 필요 없고 해석이 명확해서다.
    제약 위반 시에는 위반 정도에 비례한 음수를 주어 방향을 잃지 않게 한다.
    """
    if m.valid_rate < 1.0:
        return -10.0 + m.valid_rate
    if m.n_segmented == 0:
        return 0.0          # 아무것도 자르지 않으면 지연 이득도 0 — 최악과 동치
    lcb = quality_lcb(m)
    if lcb < q_floor:
        return -1.0 + (lcb - q_floor)
    return m.latency_gain


def mechanical_split(text: str, every: int, spaced: bool) -> str:
    """의미를 무시하고 every 글자마다 자르는 하한 앵커.

    Q_floor 를 언어쌍과 무관한 고정 상수로 두면 위험하므로,
    '나쁜 분절'의 Q를 실측해 상대 기준으로 삼는다."""
    body = text if spaced else _WS.sub("", text)
    parts = [body[i : i + every] for i in range(0, len(body), every)]
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 1:
        return text
    return " <SEG> ".join(parts)
