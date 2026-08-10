"""A5 Scorer — 결정론적.

설계 (`../AUTO_PROMPT_LOOP_DESIGN.md`). 축이 세 개다.

  format_pass_rate — 포맷 검증 통과율. 1.0 미만이면 그 프롬프트는 탈락. 하드 게이트.
  adequacy         — QE(조각 원문, 조각 번역). **참조가 없다.** y축 주지표.
  laal_words       — Length-Adaptive Average Lagging, 소스 어절 단위. x축.

지연은 목적함수에 들어가지 않는다. 노브(`target_chunk_words`)가 고정하기 때문이다.
따라서 목적함수는 단일축이고 가중치도 임계값도 없다:

  effective = adequacy × (1 − contradiction)   조기 방출로 뒤집힌 만큼을 뺀다
  score        = T 격자에서의 effective 평균

v1 의 `Q`(= seg 합본 vs full 번역)는 `consistency` 로 이름을 바꿔 **보고 지표로만**
남는다. 가설("나누어 번역해 합쳐도 의미가 크게 달라지지 않는 지점")의 직접 측정값이라
버리지 않지만, 참조가 자기 시스템의 offline 출력이라 어순 편향이 있다 — 자체 실측에서
의미를 보존한 재서술(0.8414)이 부정 뒤집힘(0.8843)보다 낮게 나온다. 그래서 주지표는
참조 없는 `adequacy` 다.

폐기된 것: `L`/`gain`(= Average Proportion. 문헌은 AP 를 쓰지 않는다), `k_eff`,
`Q_floor`/`LCB`/`q_weight`/앵커 캘리브레이션, 달성률.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

from .gateway import Gateway
from .pipeline import split_segments, unit_count

_WS = re.compile(r"\s+")


# ── chrF ─────────────────────────────────────────────────────────────────

def _char_ngrams(s: str, n: int) -> Counter:
    s = _WS.sub("", s)
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """문자 n-gram F-score. 표면형 일치도만 보므로 의미 판정에는 부족하다 —
    보조 지표로만 쓴다."""
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
    """두 문자열이 동일하면 호출 없이 1.0."""
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


# ── consistency 백엔드 (참조 기반) ───────────────────────────────────────

def _identity_shortcut(hyps: list[str], refs: list[str]) -> tuple[list[float], list[int]]:
    """동일 문자열은 1.0, 빈 문자열은 0.0 으로 고정하고 나머지 인덱스를 돌려준다.

    무분절 문장은 seg 번역 = full 번역이라 항상 여기서 걸린다. 모든 백엔드가 같은
    규약을 쓰게 해야 백엔드 간 수치가 비교 가능하다."""
    scores = [1.0] * len(hyps)
    pending: list[int] = []
    for i, (h, r) in enumerate(zip(hyps, refs)):
        if not h or not r:
            scores[i] = 0.0
        elif h != r:
            pending.append(i)
    return scores, pending


class QualityBackend:
    """`consistency` 계산 백엔드 (참조 기반).

    `src` 를 받는 이유는 COMET 이 원문을 입력으로 쓰기 때문이다. embed·chrF 는 무시한다."""

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


class _CometBase:
    """COMET 계열 공통 로더. 모델 로드는 **프로세스당 1회**다."""

    def __init__(self, model_name: str, batch_size: int = 16, gpus: int = 1):
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
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
        """GPU 메모리를 돌려준다. 여러 체크포인트를 한 프로세스에서 비교할 때만 필요하다."""
        if self._model is None:
            return
        self._model = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def _predict(self, batch: list[dict]) -> list[float]:
        out = self.load().predict(batch, batch_size=self.batch_size, gpus=self.gpus,
                                  progress_bar=False)
        return [float(s) for s in out.scores]


class CometBackend(_CometBase, QualityBackend):
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da",
                 batch_size: int = 16, gpus: int = 1, name: str = "comet"):
        _CometBase.__init__(self, model_name, batch_size, gpus)
        self.name = name

    def score(self, srcs, hyps, refs):
        scores, pending = _identity_shortcut(hyps, refs)
        if not pending:
            return scores
        batch = [{"src": srcs[i], "mt": hyps[i], "ref": refs[i]} for i in pending]
        for i, s in zip(pending, self._predict(batch)):
            scores[i] = s
        return scores


# 체크포인트 이름을 나눠 두는 이유는 `config.json` 에 무엇으로 쟀는지가 남아야 하기
# 때문이다. 축이 다른 두 백엔드의 수치를 섞어 쓰면 비교가 무의미해진다.
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
    raise ValueError(f"알 수 없는 consistency 백엔드: {name}")


# ── adequacy 백엔드 (참조 없음) ──────────────────────────────────────────

# **참조를 두지 않는 것이 요점이다.** 참조를 full 번역으로 두면 어순을 단조화한 좋은
# 분절이 감점된다 (ko→en 은 SOV→SVO 라 크게 걸린다). 문헌의 해법(단조 번역 모델,
# 타깃 재배열)은 우리가 "그 절차 없이 된다"고 주장하는 대상이라 쓸 수 없다.
QE_CHECKPOINTS = {
    "cometkiwi": "Unbabel/wmt22-cometkiwi-da",      # HF 게이트 모델. 라이선스 동의 필요
    "cometkiwi-xl": "Unbabel/wmt23-cometkiwi-da-xl",
}


class AdequacyBackend:
    """`adequacy` 계산 백엔드. 입력은 `(원문 조각, 번역 조각)` 쌍뿐이다."""

    name = "base"

    def score(self, srcs: list[str], hyps: list[str]) -> list[float]:
        raise NotImplementedError


class CometKiwiBackend(_CometBase, AdequacyBackend):
    def __init__(self, model_name: str = "Unbabel/wmt22-cometkiwi-da",
                 batch_size: int = 16, gpus: int = 1, name: str = "cometkiwi"):
        _CometBase.__init__(self, model_name, batch_size, gpus)
        self.name = name

    def score(self, srcs, hyps):
        out = [0.0] * len(srcs)
        pending = [i for i, (s, h) in enumerate(zip(srcs, hyps)) if s.strip() and h.strip()]
        if not pending:
            return out
        batch = [{"src": srcs[i], "mt": hyps[i]} for i in pending]
        for i, s in zip(pending, self._predict(batch)):
            out[i] = s
        return out


def make_adequacy_backend(name: str, **kw) -> AdequacyBackend:
    if name in QE_CHECKPOINTS:
        kw.setdefault("model_name", QE_CHECKPOINTS[name])
        return CometKiwiBackend(name=name, **kw)
    raise ValueError(
        f"알 수 없는 adequacy 백엔드: {name}. "
        f"쓸 수 있는 값: {sorted(QE_CHECKPOINTS)}. "
        f"CometKiwi 는 HF 게이트 모델이라 huggingface.co 에서 라이선스에 동의하고 "
        f"`hf auth login` 을 먼저 해야 한다 (huggingface_hub 1.x. 구버전은 huggingface-cli login)."
    )


# ── 조기 방출 — NLI 모순 검사 ────────────────────────────────────────────

# **`adequacy` 는 조기 방출을 원리적으로 검출할 수 없다.** `(조각 원문, 조각 번역)` 만의
# 함수이기 때문이다 — `그건 문제가 -> That's a problem` 은 그 조각의 번역으로는 최선에
# 가깝고, 틀렸다는 정보가 전부 미래에 있다. 실측에서 QE 는 못 잡는 데 그치지 않고
# **조기 방출을 보상했다**: 유창하고 완결된 조각을 선호해서, 반박당하는 방출(0.8653)이
# 정직한 파편(0.7403)보다 높게 나왔다 (케이스 5건 중 4건 순위 위반).
#
# 미래를 끌어들이는 방법 셋 중 이것만 남았다.
#   - QE 에 문장 전체를 src 로 주기 → 기각. 누락도 모순만큼 벌해서 순위 위반 4/6 그대로
#   - LLM 판정자를 점수에 주입 → 기각. 오판 1건이 평균을 0.02 움직여 검출 대상(0.003)을
#     압도한다. 포맷 하드게이트가 신호를 파괴한 것과 같은 구조
#   - NLI 모순 검사 → 채택. **결정론적**이라 목적함수에 잡음이 안 들어가고,
#     불완전함(neutral)과 모순(contradiction)을 구별한다
#
# 실측 (premature_cases.json, contradiction 확률 순위): 위반 0/5. QE 는 4/5 였다.
NLI_MODELS = {
    "deberta-mnli": "microsoft/deberta-large-mnli",              # en 타깃. 분리가 가장 깨끗
    "deberta-anli": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "mdeberta-xnli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # 다국어
}


class ContradictionBackend:
    """`NLI(premise = full 번역, hypothesis = 그 시점까지 방출된 누적 번역)`.

    premise·hypothesis 가 둘 다 **타깃 언어**라 단일 언어 NLI 로 충분하다 —
    소스 언어별 자원이 필요 없으므로 언어 독립 원칙을 지킨다.

    argmax 라벨이 아니라 **contradiction 확률**을 쓴다. 라벨이 `neutral` 로 어긋나도
    확률 순위는 유지되므로(실측) 임계값 없이 연속 점수로 쓸 수 있다."""

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli",
                 batch_size: int = 16, device: int = 0, name: str = "deberta-mnli"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.name = name
        self._pipe = None

    def load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  device=self.device, top_k=None)
        return self._pipe

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        out = [0.0] * len(premises)
        pending = [i for i, (p, h) in enumerate(zip(premises, hypotheses))
                   if p.strip() and h.strip()]
        if not pending:
            return out
        res = self.load()([{"text": premises[i], "text_pair": hypotheses[i]}
                           for i in pending], batch_size=self.batch_size)
        for i, scores in zip(pending, res):
            out[i] = next((s["score"] for s in scores
                           if s["label"].lower().startswith("contr")), 0.0)
        return out


def make_contradiction_backend(name: str, **kw) -> ContradictionBackend:
    if name not in NLI_MODELS:
        raise ValueError(f"알 수 없는 NLI 백엔드: {name}. 쓸 수 있는 값: {sorted(NLI_MODELS)}")
    kw.setdefault("model_name", NLI_MODELS[name])
    return ContradictionBackend(name=name, **kw)


def effective_of(adequacy: float, contradiction: float) -> float:
    """조기 방출한 조각은 점수를 못 번다.

        effective = adequacy × (1 − contradiction)

    곱셈이라 **새 상수가 없다** — 가중치도 임계값도 도입하지 않는다. 모순이 없으면
    (contradiction≈0) 그대로 통과하고, 확실히 반박당하면(≈1) 0 이 된다. 의미도 그대로다:
    사용자가 틀린 것을 본 조각은 지연 이득을 벌지 않는다.
    """
    return adequacy * (1.0 - contradiction)


# ── 지연 — LAAL ──────────────────────────────────────────────────────────

def laal_words(seg_text: str, seg_pieces: list[str] | None, full_translation: str,
               spaced: bool = True, tgt_spaced: bool = True) -> float:
    """Length-Adaptive Average Lagging. 소스 **어절**(비띄어쓰기 언어면 문자) 단위.

        laal = (1/τ) · Σ_{i=1..τ} ( d_i − d*_i )
        d*_i = (i−1) · |X| / max(|Y|, |Y*|)
        τ    = min{ i : d_i = |X| }

    우리 설정에서 조각 j 의 번역은 조각 j 가 완성되는 순간 **한꺼번에** 나가므로,
    그 조각에 속한 모든 목표 토큰의 지연 `d_i` 는 누적 소스 길이 `C_j` 로 같다.
    마지막 조각은 항목 하나만 기여하고 그 뒤는 잘린다.

    v1 의 `gain`(= Average Proportion)을 대체한다. AP 는 Cho & Esipova(2016) 이후
    쓰이지 않는다 — 하한이 0.5 에 묶여 동적 범위 절반이 낭비되고 문장 길이에
    의존한다. AP 와 달리 LAAL 은 **순서에 민감하다**: 앞쪽을 빨리 내면 낮아진다.

    참조 길이 `|Y*|` 는 gold 참조가 없으므로 full 번역 길이로 대신한다.
    단위가 ms 가 아니라 어절이므로 논문 수치와 직접 비교하면 안 된다.
    """
    chunks = split_segments(seg_text)
    if not chunks:
        return 0.0
    c = [unit_count(x, spaced) for x in chunks]
    X = sum(c)
    if X == 0:
        return 0.0

    m = [max(1, unit_count(p, tgt_spaced)) for p in (seg_pieces or [])]
    if len(m) != len(c):
        m = [max(1, x) for x in c]          # 조각 수가 안 맞으면 소스 비중으로 대체
    Yref = unit_count(full_translation or "", tgt_spaced) or sum(m)
    gamma = X / max(sum(m), Yref)

    d: list[int] = []
    cum = 0
    for cj, mj in zip(c, m):
        cum += cj
        d.extend([cum] * mj)
    tau = next((i for i, di in enumerate(d) if di >= X), len(d) - 1) + 1
    return sum(d[i - 1] - (i - 1) * gamma for i in range(1, tau + 1)) / tau


# ── 집계 ─────────────────────────────────────────────────────────────────

@dataclass
class SplitMetrics:
    """노브 값 `T` 하나에서의 지표.

    `effective = adequacy × (1 − contradiction)` 가 목적함수가 보는 값이다.
    `adequacy` 와 `contradiction` 은 각각 따로도 보고해 어느 쪽이 움직였는지 분리한다."""

    target_chunk_words: int
    n: int
    n_scored: int              # 포맷 위반을 제외하고 실제로 채점된 문장 수
    effective: float
    adequacy: float
    contradiction: float
    effective_min: float
    effective_p10: float
    consistency: float
    consistency_chrf: float
    laal_words: float
    chunks_per_sentence: float
    split_ratio: float
    missing_boundaries: float
    premature_rate: float | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class Metrics:
    n: int
    format_pass_rate: float
    format_pass_rate_no_retry: float
    by_T: dict[str, SplitMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "format_pass_rate": round(self.format_pass_rate, 4),
            "format_pass_rate_no_retry": round(self.format_pass_rate_no_retry, 4),
            "by_T": {k: v.to_dict() for k, v in self.by_T.items()},
            "score": round(score(self), 4),
        }

    def at(self, T: int) -> SplitMetrics | None:
        return self.by_T.get(str(T))


def aggregate_split(
    target_chunk_words: int,
    effective_scores: list[float],
    adequacy_scores: list[float],
    contradiction_scores: list[float],
    consistency_scores: list[float],
    chrf_scores: list[float],
    laals: list[float],
    ks: list[int],
    missings: list[int],
    n_total: int | None = None,
) -> SplitMetrics:
    """점수 리스트는 **포맷을 통과한 문장만** 담는다.

    포맷 위반 문장의 분절은 규칙을 어긴 것이라 그 adequacy 에 의미가 없다. 반대로
    위반 1건으로 프롬프트 전체를 폐기하면(v1 의 `-10 + fmt`) 3% 표본 사건이 0.003 규모의
    신호를 압도한다. 위반은 `format_pass_rate` 로 따로 잰다.
    """
    n_scored = len(effective_scores)
    e_sorted = sorted(effective_scores) if effective_scores else [0.0]
    p10 = e_sorted[max(0, int(0.10 * (len(e_sorted) - 1)))]

    def mean(xs, default=0.0):
        return sum(xs) / len(xs) if xs else default

    return SplitMetrics(
        target_chunk_words=target_chunk_words,
        n=n_total if n_total is not None else n_scored,
        n_scored=n_scored,
        effective=mean(effective_scores),
        adequacy=mean(adequacy_scores),
        contradiction=mean(contradiction_scores),
        effective_min=e_sorted[0],
        effective_p10=p10,
        consistency=mean(consistency_scores, 1.0),
        consistency_chrf=mean(chrf_scores),
        laal_words=mean(laals),
        chunks_per_sentence=mean([float(k) for k in ks], 1.0),
        split_ratio=(sum(1 for k in ks if k > 1) / len(ks)) if ks else 0.0,
        missing_boundaries=mean([float(s) for s in missings]),
    )


def aggregate(n: int, valid_flags: list[bool], first_pass_flags: list[bool] | None,
              by_T: dict[str, SplitMetrics]) -> Metrics:
    return Metrics(
        n=n,
        format_pass_rate=(sum(valid_flags) / n) if n else 0.0,
        format_pass_rate_no_retry=(
            (sum(first_pass_flags) / n) if (first_pass_flags and n)
            else ((sum(valid_flags) / n) if n else 0.0)),
        by_T=by_T,
    )


def score(m: Metrics) -> float:
    """**단일축.** T 격자에서의 `effective` 평균.

    가중치가 없는 이유는 두 축을 합치는 게 아니라 **같은 축을 T 별로 평균**하기
    때문이다. 지연은 노브가 고정하므로 목적함수에서 빠진다 — 프롬프트가 덜 잘라서
    점수를 얻는 경로가 구조적으로 없다.

    **포맷 하드 게이트(`-10 + fmt`)를 없앴다.** run01 에서 30문장 중 1건의 표기 위반이
    프롬프트를 `-9.03` 으로 폐기시켰고, 그 크기가 실제 프롬프트 차이(0.003)를 3000배
    압도해 hill climbing 이 "이번에 위반이 났는가"로 결정됐다. 지금은 표기 오류를
    `pipeline.normalize_tags` 가 결정론적으로 고치고, 남은 위반 문장은 채점에서
    제외되며(`n_scored`), `format_pass_rate` 는 별도로 보고된다. 번역 비용 방어는
    `skip_translation_below` 가 계속 담당한다.
    """
    vals = [s.effective for s in m.by_T.values()]
    return sum(vals) / len(vals) if vals else 0.0


def paired_delta(new_rows: list[dict], best_rows: list[dict],
                 t_grid: list[int]) -> dict:
    """이전 best 대비 **문장별** effective 차이. 채택 판정이 이걸로 이뤄진다.

    절대 평균 비교는 문장 난이도 분산에 묻힌다 — run01 dev 실측에서 문장별 sd 0.0496,
    평균의 표준오차 0.0064 인데 검출하려는 차이는 0.0035 였다.

    같은 문장을 쓰므로 쌍체 비교가 가능하고, 결정적으로 **두 프롬프트가 같은 분절을 준
    문장은 차이가 정확히 0** 이다. 프롬프트를 1~2 섹션만 고치므로 대부분이 그렇고,
    따라서 쌍체 차이는 자동으로 **분절이 실제로 바뀐 문장에만** 초점을 맞춘다.
    """
    best = {r["id"]: r for r in best_rows}
    deltas: list[float] = []
    changed = 0
    for r in new_rows:
        b = best.get(r["id"])
        if not b:
            continue
        d, seen = 0.0, 0
        diff = False
        for T in t_grid:
            a = (r.get("by_T") or {}).get(str(T))
            c = (b.get("by_T") or {}).get(str(T))
            if not a or not c:
                continue
            d += a["effective"] - c["effective"]
            seen += 1
            if a["seg_text"] != c["seg_text"]:
                diff = True
        if seen:
            deltas.append(d / seen)
            changed += 1 if diff else 0
    if not deltas:
        return {"mean_delta": None, "se_delta": None, "n_pairs": 0, "n_changed": 0}
    mean_d = sum(deltas) / len(deltas)
    if len(deltas) > 1:
        var = sum((x - mean_d) ** 2 for x in deltas) / len(deltas)
        se = (var / len(deltas)) ** 0.5
    else:
        se = 0.0
    return {"mean_delta": round(mean_d, 5), "se_delta": round(se, 5),
            "n_pairs": len(deltas), "n_changed": changed}


def mechanical_split(text: str, every: int, spaced: bool) -> str:
    """의미를 무시하고 every 글자마다 자르는 비교군 (곡선의 하한 기준선).

    순위를 달지 않는다 — 절단할 근거가 없으므로 곡선 위의 점 하나로만 평가된다."""
    body = text if spaced else _WS.sub("", text)
    parts = [body[i : i + every] for i in range(0, len(body), every)]
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 1:
        return text
    return " <SEG> ".join(parts)
