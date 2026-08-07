"""A5 Scorer — 결정론적.

2축 목적함수를 계산한다. 품질(Q) 단일 축으로 최적화하면 최적해가
"<SEG>를 하나도 넣지 않는 프롬프트"로 수렴하므로 지연축(L)이 반드시 필요하다.

  Q : seg 번역 이어붙임 vs full 번역의 의미 유사도. 분절이 없으면 1.0 (제외하지 않음).
  L : 스트리밍 지연 프록시. 1.0 = 무분절, 0.5 = 무한 세분. 낮을수록 빠름.
  V : 포맷 검증 통과율. 1.0 미만이면 그 프롬프트는 탈락.

품질 백엔드는 교체 가능:
  comet (기본) — 운영 기준 지표. unbabel-comet + GPU 필요. 원문(src)도 본다.
  embed        — 게이트웨이 임베딩 코사인 유사도. 로컬 ML 의존성 불필요.
  chrf         — 문자 n-gram F-score. 보조 지표로 항상 함께 리포트.

**백엔드를 바꾸면 반드시 타당도를 다시 측정한다.** 임베딩 코사인은 부정이 뒤집힌
번역(0.9278)을 무해한 표기 차이(0.8401)보다 높게 매겨 순위가 뒤집혀 있었다 —
캘리브레이션은 스케일만 맞출 뿐 타당도를 보증하지 않는다. `validity_check.py` 참조.
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


# ── 품질 백엔드 ──────────────────────────────────────────────────────────

def _identity_shortcut(hyps: list[str], refs: list[str]) -> tuple[list[float], list[int]]:
    """동일 문자열은 1.0, 빈 문자열은 0.0 으로 고정하고 나머지 인덱스를 돌려준다.

    무분절 문장은 seg 번역 = full 번역이라 항상 여기서 걸린다. Q 정의상 1.0 이고
    (제약은 분절된 문장에만 걸리므로 목적함수에 영향이 없다) 모델 호출도 아낀다.
    모든 백엔드가 같은 규약을 쓰게 해야 백엔드 간 수치가 비교 가능하다."""
    scores = [1.0] * len(hyps)
    pending: list[int] = []
    for i, (h, r) in enumerate(zip(hyps, refs)):
        if not h or not r:
            scores[i] = 0.0
        elif h != r:
            pending.append(i)
    return scores, pending


class QualityBackend:
    """Q 계산 백엔드.

    `src` 를 받는 이유는 COMET 이 원문을 입력으로 쓰기 때문이다. embed·chrF 는 무시한다.
    반환은 문장별 점수 리스트로 통일한다."""

    name = "base"

    def score(self, srcs: list[str], hyps: list[str], refs: list[str]) -> list[float]:
        raise NotImplementedError


class EmbedBackend(QualityBackend):
    name = "embed"

    def __init__(self, gw: Gateway):
        self.gw = gw

    def score(self, srcs, hyps, refs):
        return embed_similarity(self.gw, hyps, refs)


class ChrfBackend(QualityBackend):
    name = "chrf"

    def score(self, srcs, hyps, refs):
        scores, pending = _identity_shortcut(hyps, refs)
        for i in pending:
            scores[i] = chrf(hyps[i], refs[i])
        return scores


class CometBackend(QualityBackend):
    """운영 기준 지표.

    모델 로드는 **프로세스당 1회**다. 이터레이션마다 로드하면 런 시간이 배로 든다
    (기존 `utils/comet_eval.py` 가 그렇게 돼 있으니 그쪽 코드를 가져다 쓰지 말 것)."""

    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da",
                 batch_size: int = 16, gpus: int = 1, name: str = "comet"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.name = name
        self._model = None

    def load(self):
        if self._model is None:
            import logging
            from comet import download_model, load_from_checkpoint
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
            self._model = load_from_checkpoint(download_model(self.model_name))
            self._model.eval()
        return self._model

    def unload(self) -> None:
        """GPU 메모리를 돌려준다. 여러 체크포인트를 한 프로세스에서 비교할 때만 필요하다
        (XCOMET-XL 은 14GB 를 잡아 두 개가 동시에 안 올라간다). 루프에서는 부르지 않는다."""
        if self._model is None:
            return
        self._model = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def score(self, srcs, hyps, refs):
        scores, pending = _identity_shortcut(hyps, refs)
        if not pending:
            return scores
        model = self.load()
        batch = [{"src": srcs[i], "mt": hyps[i], "ref": refs[i]} for i in pending]
        out = model.predict(batch, batch_size=self.batch_size, gpus=self.gpus,
                            progress_bar=False)
        for i, s in zip(pending, out.scores):
            scores[i] = float(s)
        return scores


# COMET 계열은 체크포인트만 다르고 인터페이스가 같다. 이름을 나눠 두는 이유는
# baseline.json 에 어느 체크포인트로 캘리브레이션했는지가 남아야 하기 때문이다 —
# 축이 다른 두 백엔드의 Q_floor 를 섞어 쓰면 비교가 무의미해진다.
COMET_CHECKPOINTS = {
    "comet": "Unbabel/wmt22-comet-da",   # XLM-R large 기반. 빠르고 공개 접근 가능
    "xcomet": "Unbabel/XCOMET-XL",       # 오류 구간 탐지형. HF 라이선스 동의 필요
}


def make_backend(name: str, gw: Gateway | None = None, **kw) -> QualityBackend:
    if name == "embed":
        if gw is None:
            raise ValueError("embed 백엔드에는 Gateway 가 필요합니다")
        return EmbedBackend(gw)
    if name == "chrf":
        return ChrfBackend()
    if name in COMET_CHECKPOINTS:
        kw.setdefault("model_name", COMET_CHECKPOINTS[name])
        return CometBackend(name=name, **kw)
    raise ValueError(f"알 수 없는 품질 백엔드: {name}")


# ── 지연 프록시 ──────────────────────────────────────────────────────────

def latency_proxy(seg_text: str, spaced: bool) -> float:
    """**정보 1단위가 평균적으로 기다린 시간.**

      조각 길이 c_1..c_k, N = Σc_i, 누적_i = c_1+...+c_i
      L = Σ_i (c_i / N) · (누적_i / N)

    각 조각의 대기시간을 **그 조각이 담은 정보량 비중으로 가중**한다.

    이전 정의는 조각 수로 평균했다 (`Σ 누적_i / (k·N)`). 그러면 1글자 조각과
    30어절 조각이 같은 무게라, **무의미한 조각을 앞에 만들수록 점수가 좋아진다.**
    실측에서 탐색이 즉시 그 허점을 찾아냈다 — '그 <SEG> 다음에 이거 벤치...' 가
    의미 기반 분절(gain 0.2059)을 제치고 gain 0.4853 으로 1위였다. 사용자가 얻는
    정보는 0인데 지표만 좋아진 것이다. 정보 가중을 하면 같은 분절이 0.0285 로
    무분절과 동등하게 평가된다.

    등분할에서는 두 정의가 정확히 같은 값을 준다 (k=2 → 0.75, k=3 → 0.667, ...).
    갈리는 것은 불균등한 분절뿐이다.

    k=1 → 1.0 (무분절). k→∞(등분할) → 0.5. 낮을수록 정보가 빨리 도착한다.
    """
    parts = split_segments(seg_text)
    if not parts:
        return 1.0
    lens = [len(p if spaced else _WS.sub("", p)) for p in parts]
    total = sum(lens)
    if total == 0 or len(lens) == 1:
        return 1.0
    cum, acc = 0, 0.0
    for c in lens:
        cum += c
        acc += (c / total) * (cum / total)
    return acc


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
