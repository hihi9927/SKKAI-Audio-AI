"""A6 Scorer — 결정론적.

설계는 `AUTOSEG_SIMPLIFY.md`, 지표 선택의 근거는 `AUTOSEG_DETAILS.md`. 축이 세 개다.

  format_pass_rate — 포맷 검증 통과율. 1.0 미만이면 그 프롬프트는 탈락. 하드 게이트.
  adequacy         — QE(조각 원문, 조각 번역). **참조가 없다.** y축 주지표.
  laal_words       — Length-Adaptive Average Lagging, 소스 어절 단위. x축.

지연은 목적함수에 들어가지 않는다. 노브(`target_chunk_words`)가 고정하기 때문이다.
따라서 목적함수는 단일축이고 가중치도 임계값도 없다:

  contradiction(문장) = 경계 (k−1)개 모순 확률의 평균     # 마지막 조각은 대상 아님
  effective(문장)     = adequacy(문장) × (1 − contradiction(문장))
  score               = T 격자에서의 effective 평균

**집계는 경계 평균이다 — 조각 가중 평균이 아니다.** 조각 가중 평균은 마지막 조각의
구조적 0 을 평균에 넣어 k 가 클수록 contradiction 이 기계적으로 오르고(잡음 기대값
ε 에서 문장 값 ≈ ε·(1 − w_last/W)), 무분절이 "경계가 없어서" 0 점 만점을 받는다 —
게임을 안 뛰면 오류율 0 인 구조. 경계 평균은 iid 잡음 기대값이 k 무관이라 노출이
정규화되고, **무분절(k=1)은 0 이 아니라 정의되지 않음(None)** 이 된다. effective 도
같이 None 이 되어 평균에서 빠진다 (`n_effective` 로 집계 대상 수를 남긴다).

v1 의 `Q`(= seg 합본 vs full 번역)는 `consistency` 로 이름을 바꿔 **보고 지표로만**
남는다. 가설("나누어 번역해 합쳐도 의미가 크게 달라지지 않는 지점")의 직접 측정값이다.
기본 백엔드는 **양방향 NLI**(`nli`) 다 — COMET 은 참조가 자기 시스템의 offline 출력이라
어순 편향이 있다(자체 실측: 의미를 보존한 재서술 0.8414 < 부정 뒤집힘 0.8843). NLI 는
명제만 보므로 어순을 단조화한 좋은 분절이 감점되지 않고, ent(full⇒합본)이 환각을,
ent(합본⇒full)이 누락을 각각 잡는다. COMET 계열은 옵션으로 남긴다.

폐기된 것: `L`/`gain`(= Average Proportion. 문헌은 AP 를 쓰지 않는다), `k_eff`,
`Q_floor`/`LCB`/`q_weight`/앵커 캘리브레이션, 달성률.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

from .pipeline import TAG_RE, split_segments, unit_count

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

    `src` 를 받는 이유는 COMET 이 원문을 입력으로 쓰기 때문이다. NLI 는 무시한다.

    **embed·chrF 백엔드는 삭제했다.** 45개 런에서 한 번도 안 쓰였고(nli 39 / comet 4),
    embed 는 그 하나 때문에 `Gateway`(임베딩 API 호출)를 이 모듈에 끌어들이고 있었다."""

    name = "base"

    def score(self, srcs: list[str], hyps: list[str], refs: list[str]) -> list[float]:
        raise NotImplementedError


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
        # num_workers=0 — DataLoader 워커를 안 띄우고 메인 프로세스에서 돈다.
        # 워커를 쓰면 "DataLoader worker exited unexpectedly" 로 런이 통째로 죽는다
        # (2026-09-02 mg1 COMET, BLEU 는 다 끝난 뒤라 16분을 날렸다).
        out = self.load().predict(batch, batch_size=self.batch_size, gpus=self.gpus,
                                  progress_bar=False, num_workers=0)
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


def make_backend(name: str, **kw) -> QualityBackend:
    if name == "nli":
        kw.setdefault("model_name", NLI_MODEL)
        kw.setdefault("name", name)
        return BidirectionalNliBackend(**kw)
    if name in COMET_CHECKPOINTS:
        kw.setdefault("model_name", COMET_CHECKPOINTS[name])
        return CometBackend(name=name, **kw)
    raise ValueError(f"알 수 없는 consistency 백엔드: {name}. "
                     f"쓸 수 있는 값: nli, {', '.join(sorted(COMET_CHECKPOINTS))}")


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
# **모델은 하나로 고정한다.** 후보 비교는 끝났고, 고를 수 있게 열어 두면 런마다 다른
# 체크포인트로 잰 값이 한 표에 섞인다 (저장된 런에 실제로 셋이 섞여 있었다:
# xlmr-anli 19 / mdeberta-xnli 16 / deberta-mnli 7).
#
# `xlm-roberta-large-xnli-anli` 인 이유 — **다국어 large 여야 한다.** base 급
# `mDeBERTa-v3-base-xnli` 는 ko/zh/ja 타깃에서 consistency 곡선이 뒤집혔다 (T 를 키울수록
# offline 번역에서 멀어진다고 나온다 — 물리적으로 불가능). 같은 데이터에서 comet·chrf 는
# 5/5 정상 방향이라 NLI 쪽 결함이었다. 영어 전용 `deberta-large-mnli` 는 분리가 가장
# 깨끗하지만 비영어 타깃에서 무음으로 틀린 값을 준다.
# 후보 비교 기록: AUTOSEG_DETAILS.md '검토했으나 채택하지 않은 것'
NLI_MODEL = "vicgalle/xlm-roberta-large-xnli-anli"


class _NliBase:
    """NLI pipeline 공통 로더.

    pipeline 은 **(모델, 디바이스)별 프로세스 전역 싱글턴**이다 — contradiction 과
    consistency(nli)가 같은 체크포인트를 쓰는 것이 기본 구성인데, 인스턴스별로
    로드하면 같은 모델이 GPU 에 두 번 올라간다 (deberta-large ≈ 1.6GB 중복).
    run04 에서 다른 실험과 GPU 를 나눠 쓰다 OOM 난 뒤 공유로 바꿨다."""

    _PIPES: dict = {}

    def __init__(self, model_name: str = NLI_MODEL,
                 batch_size: int = 16, device: int = 0, name: str = ""):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.name = name
        self._pipe = None

    def load(self):
        if self._pipe is None:
            key = (self.model_name, self.device)
            if key not in _NliBase._PIPES:
                from transformers import pipeline
                # **truncation 을 반드시 켠다.** xlm-roberta 의 위치 임베딩은 514 라
                # 긴 문장(어절 40+)에서 premise+hypothesis 가 그 한계를 넘으면
                # 조용한 오차가 아니라 RuntimeError 로 런이 죽는다 (clean500 실측:
                # 875 토큰). 잘린 뒤쪽은 어차피 모델이 못 보던 구간이다.
                _NliBase._PIPES[key] = pipeline(
                    "text-classification", model=self.model_name,
                    device=self.device, top_k=None,
                    truncation=True, max_length=512)
            self._pipe = _NliBase._PIPES[key]
        return self._pipe

    @staticmethod
    def _prob(scores: list[dict], prefix: str) -> float:
        return next((s["score"] for s in scores
                     if s["label"].lower().startswith(prefix)), 0.0)


class ContradictionBackend(_NliBase):
    """`NLI(premise = full 번역, hypothesis = 그 시점까지 방출된 누적 번역)`.

    premise·hypothesis 가 둘 다 **타깃 언어**라 단일 언어 NLI 로 충분하다 —
    소스 언어별 자원이 필요 없으므로 언어 독립 원칙을 지킨다.

    argmax 라벨이 아니라 **contradiction 확률**을 쓴다. 라벨이 `neutral` 로 어긋나도
    확률 순위는 유지되므로(실측) 임계값 없이 연속 점수로 쓸 수 있다."""

    def __init__(self, model_name: str = NLI_MODEL,
                 batch_size: int = 16, device: int = 0, name: str = "xlmr-anli"):
        super().__init__(model_name, batch_size, device, name)

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        return self.score_dual(premises, hypotheses)[0]

    def score_dual(self, premises: list[str],
                   hypotheses: list[str]) -> tuple[list[float], list[float]]:
        """`(contradiction, 1 − entailment)` 를 **한 번의 호출로** 둘 다 낸다.

        NLI 는 세 라벨 확률을 한꺼번에 주므로 두 척도를 같이 재는 데 **추가 비용이 0** 이다.
        `1 − entailment` 로 바꾸는 안이 검토 중이라(오프라인 실측이 전부 그쪽을 가리키지만
        표본이 얇다) 목적함수는 `contradiction` 그대로 두고 다른 쪽을 **나란히 기록**한다.
        다음 런의 로그로 판단한다.

        오프라인 실측 (`AUTOSEG_SIMPLIFY.md` "확인해야 할 것"):
          관문 최소 여유 0.0604 → 0.8377 · 긴 방출 구간 신호 +0.0115 → +0.0450
          검출력 |t| 0.49 → 0.80 · 두 척도 순위상관 +0.923 · 채택 결정 11회 중 3회 뒤집힘

        차이: `contradiction` 은 "틀렸다" 만 세고, `1 − entailment` 는 "아직 모르겠다"
        (neutral)까지 센다. 미완성 조각은 원래 neutral 이 정상이라 무해한 미완성이
        벌받을 수 있다 — 관문에서 `benign_incomplete` 가 3~4배 올랐다.
        """
        n = len(premises)
        contra = [0.0] * n
        one_minus_ent = [0.0] * n
        pending = [i for i, (p, h) in enumerate(zip(premises, hypotheses))
                   if p.strip() and h.strip()]
        if not pending:
            return contra, one_minus_ent
        res = self.load()([{"text": premises[i], "text_pair": hypotheses[i]}
                           for i in pending], batch_size=self.batch_size)
        for i, scores in zip(pending, res):
            contra[i] = self._prob(scores, "contr")
            one_minus_ent[i] = 1.0 - self._prob(scores, "entail")
        return contra, one_minus_ent


class BidirectionalNliBackend(_NliBase, QualityBackend):
    """`consistency` 기본 백엔드 — 합본 vs full 번역의 **양방향 entailment**.

        ent(full ⇒ 합본)   합본에 full 이 지지하지 않는 명제가 있는가 (환각·왜곡)
        ent(합본 ⇒ full)   합본이 정보를 빠뜨렸는가 (누락)
        score = min(둘)

    함의는 비대칭이라 한 방향만 재면 반쪽만 본다 — full⇒합본은 누락을 통과시키고
    (약한 명제는 함의되므로), 합본⇒full 은 환각을 통과시킨다. min 이라 어느 쪽
    실패든 잡히고, 두 방향을 따로 보면 실패 유형이 분리된다.

    COMET consistency 를 대체하는 이유: 참조가 offline 출력이라 어순을 단조화한
    좋은 분절이 감점된다 (benign_paraphrase 0.8414 < negation_flip 0.8843). NLI 는
    명제만 보므로 표면 어순·문장 수가 달라도 감점이 없고, 부정 뒤집힘은 양방향
    모두에서 contradiction 으로 잡힌다. 두 입력이 모두 타깃 언어라 소스 언어별
    자원도 필요 없다. 기본 `xlmr-anli` 는 다국어라 **타깃에 따라 바꿀 필요가 없다** —
    예전 처방이던 base 급 다국어 모델은 ko/zh/ja 에서 곡선이 뒤집힌다 (`NLI_MODEL` 주석)."""

    def __init__(self, model_name: str = NLI_MODEL,
                 batch_size: int = 16, device: int = 0, name: str = "nli"):
        super().__init__(model_name, batch_size, device, name)

    def score(self, srcs, hyps, refs):
        scores, pending = _identity_shortcut(hyps, refs)
        if not pending:
            return scores
        items = []
        for i in pending:
            items.append({"text": refs[i], "text_pair": hyps[i]})   # full ⇒ 합본
            items.append({"text": hyps[i], "text_pair": refs[i]})   # 합본 ⇒ full
        res = self.load()(items, batch_size=self.batch_size)
        for j, i in enumerate(pending):
            scores[i] = min(self._prob(res[2 * j], "entail"),
                            self._prob(res[2 * j + 1], "entail"))
        return scores


def make_contradiction_backend(**kw) -> ContradictionBackend:
    kw.setdefault("model_name", NLI_MODEL)
    kw.setdefault("name", "xlmr-anli")
    return ContradictionBackend(**kw)


def effective_of(adequacy: float, contradiction: float) -> float:
    """조기 방출로 반박당한 만큼 점수를 깎는다. **문장 단위**로 적용한다.

        effective(문장) = adequacy(문장) × (1 − mean(경계 contradiction))

    곱셈이라 **새 상수가 없다** — 가중치도 임계값도 도입하지 않는다. 모순이 없으면
    (contradiction≈0) 그대로 통과하고, 확실히 반박당하면(≈1) 0 이 된다. 의미도 그대로다:
    사용자가 틀린 것을 본 방출은 지연 이득을 벌지 않는다.

    contradiction 은 **경계 (k−1)개의 평균**이다. 마지막 조각(미래 없음, 구조적 0)을
    평균에 넣던 이전 집계는 k 가 클수록 문장 값이 기계적으로 올라 무분절이 자동
    만점을 받았다. 경계 평균은 노출이 정규화되고, 무분절은 경계가 없어 contradiction
    자체가 정의되지 않는다 — 호출자가 None 으로 처리한다."""
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

    **문장 단위로는 T 에 대해 단조가 아니다.** `τ` 는 "소스를 다 들은 시점"이라 그 뒤는
    안 세는데(AL/LAAL 표준), 경계를 빼면 마지막 조각이 커져 그 시점이 **앞당겨진다**.
    그러면 가장 오래 기다린 뒤쪽 항이 통째로 집계에서 빠져 평균이 내려갈 수 있다.

    최소 반례 (소스 11어절):
        조각 3개  소스 [3,5,3] 타깃 [4,1,5]  τ=6  항 [3.0,1.9,0.8,−0.3,3.6,5.5]  laal 2.4167
        2·3 합침  소스 [3,8]   타깃 [4,6]    τ=5  항 [3.0,1.9,0.8,−0.3,6.6]      laal 2.4000
    5번 항은 3.6 → 6.6 으로 **올랐는데**(기다림은 실제로 늘었다) 6번 항 5.5 가 사라져
    합이 14.50 → 12.00 이 됐다.

    저장된 런 실측: 문장×T 계열 2,120 중 247(11.7%)이 이 모양이다. **집계 평균은
    17/17 런에서 단조 증가**라 곡선은 안전하다 — 문장마다 방향이 엇갈려도 상쇄된다.
    따라서 `laal_words` 는 **집계에서만 해석한다.** 문장 하나를 놓고 "T 를 키우면 지연이
    준다"고 읽으면 안 된다.
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
    `adequacy` 와 `contradiction` 은 각각 따로도 보고해 어느 쪽이 움직였는지 분리한다.

    `effective`·`contradiction` 은 **경계가 있는 문장(k≥2)에서만 정의**된다. 무분절
    문장은 모순을 낼 노출 자체가 없어 0 이 아니라 None 이고 평균에서 빠진다 —
    빠진 규모는 `n_effective` 로 드러난다. 전 문장이 무분절이면(무분절 비교군) 둘 다
    None 이다."""

    target_chunk_words: int
    n: int
    n_effective: int           # 경계가 있어 effective 가 정의된 문장 수 (k≥2)
    effective: float | None
    adequacy: float
    contradiction: float | None
    effective_min: float | None
    effective_p10: float | None
    consistency: float
    laal_words: float
    chunks_per_sentence: float
    missing_boundaries: float
    rank_contra_spearman: float | None = None      # 순위 vs 실측 contra 정렬도. 양수=정렬
    # 순위 하위 절반 − 상위 절반의 경계 contradiction 차. 양수=정렬, 0 이하=순위 무정보.
    # focus="priority" 판정이 쓰는 값 (`rank_contra_gap`).
    # **se 를 함께 싣는다.** 점추정을 0 과 비교하면 heavy-tail 잡음(관측 범위 −0.03~+0.03)
    # 때문에 조향 방향이 동전 던지기가 된다 — en-de run01/run02 에서 실제로 그랬다.
    rank_contra_gap: float | None = None
    rank_contra_gap_se: float | None = None
    rank_contra_gap_n: int | None = None
    # **병기 지표 — 목적함수에는 안 들어간다.** 같은 NLI 호출에서 나온 `1 − entailment` 로
    # 잰 값이다 (`ContradictionBackend.score_dual`). 교체 여부를 다음 런의 로그로 판단한다.
    contradiction_ent: float | None = None
    effective_ent: float | None = None
    # 다언어 목적함수에서만 채워진다. `effective` 는 **타깃별 원값의 평균**이라 해석·비교가
    # 되고, `effective_z` 는 **분할별로 고정된 기준선**으로 타깃별 z-정규화한 뒤 평균한
    # 값이다 (`loop._zmix`). **보고는 effective, 채택 판정은 effective_z** 로 축을 나눈다.
    #
    # 기준선을 평가마다 다시 잡으면 안 된다 — z 평균이 항상 0 이 되어 쌍체 Δ 가 항등적
    # 으로 0 이 되거나(세트 일치), 서로 다른 문장 추출이 만든 오프셋이 Δ 로 새어 나온다
    # (세트 불일치). run05 에서 둘 다 실제로 일어났다: dev Δ = +0.00000 ±0.06408,
    # train Δ = +0.14897 (기준선 고정 시 +0.08867). `loop._zmix` 참고.
    #
    # 이득은 **타깃 분산 동등 가중**이지 무조건적인 se 감소가 아니다. run05 test 실측
    # 타깃별 raw se: ko 0.0105 / ja 0.0160 / zh 0.0177 / es 0.0174 / de 0.0179.
    # 5타깃 raw 평균의 se 는 0.0120 — 평균적 타깃 대비 −25% 지만 **가장 조용한 타깃
    # (ko) 단독보다는 14% 나쁘다.** 오프라인 예측 −40% 는 평균 타깃 기준이었다.
    #
    # 기준선이 분할마다 다르므로 **런 간·분할 간 절대값 비교에 쓰면 안 된다.**
    effective_z: float | None = None
    n_targets: int | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class Metrics:
    n: int
    format_pass_rate: float
    format_pass_rate_no_retry: float
    by_T: dict[str, SplitMetrics] = field(default_factory=dict)
    # ── 순위 축 진단 (`rank_lift`) ────────────────────────────────────────
    # **순위를 무작위로 섞었을 때 effective 가 얼마나 떨어지는가.** 절단기가 순위를 쓰는
    # 곳은 keep-vs-discard 한 군데뿐이므로, 그 결정만 망가뜨려 값을 재는 것이 순위축의
    # 직접 측정이다. 폐기된 경계는 렌더링이 없어 contra 값 자체가 없으므로
    # `rank_contra_gap`(생존 경계끼리의 순서)으로는 원리적으로 못 보는 양이다.
    #
    # 실측 (순위 셔플 대조, en-de run04 + ko-en run05, 셔플 20회):
    #   폐기율 64% → lift +0.024,  82% → +0.050,  85% → +0.056,  93% → +0.061
    #   real 이 20/20 셔플을 이겼고(순열 p=0.048), 같은 조건에서 `rank_contra_gap` 은
    #   en-de 에서 오히려 무작위보다 낮게(z −0.6, −1.1) 나왔다 — 부호가 언어쌍마다 뒤집힌다.
    #
    # **가장 큰 T 에서 잰다.** 순위의 값은 폐기율과 함께 커지므로 거기가 가장 민감하다.
    # 셔플 1회면 충분하다 — 최대 T 에서 오작동 0/40 (작은 T 에서는 4/20 로 무너진다).
    rank_lift: float | None = None
    rank_lift_se: float | None = None
    rank_lift_t: float | None = None
    rank_lift_n: int | None = None
    rank_lift_T: int | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "format_pass_rate": round(self.format_pass_rate, 4),
            "format_pass_rate_no_retry": round(self.format_pass_rate_no_retry, 4),
            "by_T": {k: v.to_dict() for k, v in self.by_T.items()},
            "score": round(score(self), 4),
            "rank_lift": self.rank_lift,
            "rank_lift_se": self.rank_lift_se,
            "rank_lift_t": self.rank_lift_t,
            "rank_lift_n": self.rank_lift_n,
            "rank_lift_T": self.rank_lift_T,
        }

    def at(self, T: int) -> SplitMetrics | None:
        return self.by_T.get(str(T))


def _opt_mean(xs: list[float | None] | None) -> float | None:
    """None 을 빼고 평균. 값이 하나도 없으면 None (병기 지표가 꺼져 있을 때)."""
    if not xs:
        return None
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def percentile10(xs: list[float]) -> float | None:
    """하위 10% 지점 값. 평균이 못 보는 **꼬리**를 본다.

    같은 식이 `loop` 의 다언어 병합 쪽에도 복붙돼 있었다 — 정의가 갈리면 단일 타깃과
    다언어의 p10 을 나란히 놓을 수 없다."""
    if not xs:
        return None
    e = sorted(xs)
    return e[max(0, int(0.10 * (len(e) - 1)))]


def aggregate_split(
    target_chunk_words: int,
    effective_scores: list[float | None],
    adequacy_scores: list[float],
    contradiction_scores: list[float | None],
    consistency_scores: list[float],
    laals: list[float],
    ks: list[int],
    missings: list[int],
    n_total: int | None = None,
    effective_ent_scores: list[float | None] | None = None,
    contradiction_ent_scores: list[float | None] | None = None,
) -> SplitMetrics:
    """점수 리스트는 **포맷을 통과한 문장만** 담는다.

    포맷 위반 문장의 분절은 규칙을 어긴 것이라 그 adequacy 에 의미가 없다. 반대로
    위반 1건으로 프롬프트 전체를 폐기하면(v1 의 `-10 + fmt`) 3% 표본 사건이 0.003 규모의
    신호를 압도한다. 위반은 `format_pass_rate` 로 따로 잰다.

    `effective_scores`·`contradiction_scores` 는 무분절 문장(k=1)에서 None 을 담는다 —
    경계가 없어 모순 노출 자체가 없으므로 0(무죄)이 아니라 미정의다. None 은 평균에서
    빠지고 그 규모가 `n_effective` 로 남는다.
    """
    n_scored = len(effective_scores)
    eff_vals = [e for e in effective_scores if e is not None]
    con_vals = [c for c in contradiction_scores if c is not None]

    def mean(xs, default=0.0):
        return sum(xs) / len(xs) if xs else default

    return SplitMetrics(
        target_chunk_words=target_chunk_words,
        n=n_total if n_total is not None else n_scored,
        n_effective=len(eff_vals),
        effective=(mean(eff_vals) if eff_vals else None),
        adequacy=mean(adequacy_scores),
        contradiction=(mean(con_vals) if con_vals else None),
        effective_min=(min(eff_vals) if eff_vals else None),
        effective_p10=percentile10(eff_vals),
        consistency=mean(consistency_scores, 1.0),
        laal_words=mean(laals),
        chunks_per_sentence=mean([float(k) for k in ks], 1.0),
        missing_boundaries=mean([float(s) for s in missings]),
        effective_ent=_opt_mean(effective_ent_scores),
        contradiction_ent=_opt_mean(contradiction_ent_scores),
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


# ── 지표 용어집 — Critic 에게 넘길 설명 ──────────────────────────────────
#
# **왜 코드 옆에 두나.** 종전에는 `Critic.review` 안에 손으로 쓴 문단이 있었고, 실려 가는
# 지표 31개 중 **7개만** 설명돼 있었다. 목적함수 `effective` 조차 정의된 적이 없다 —
# 무슨 숫자를 올려야 하는지 모르는 채로 진단하고 있었다는 뜻이다. 그리고 손으로 쓴
# 문단은 필드가 늘어도 안 따라온다.
#
# `describe()` 가 **실제로 들어 있는 키만** 골라 렌더링하고, 용어집에 없는 키는
# `(no description)` 로 드러난다. 필드를 추가하면 설명이 빠진 게 보인다.
#
# **`fixable` 이 핵심이다.** 프롬프트로 움직일 수 없는 것을 고치라고 시키는 게 루프가
# 개선을 못 만든 큰 원인이었다 — `focus="format"` 개정 12회가 포맷을 못 고쳤고(재시도 후
# +0.0021, 1차 −0.0257), 그 위반의 65%가 간격이었는데 그건 이제 정규화가 결정론으로
# 처리한다. 무엇이 프롬프트의 몫이고 무엇이 아닌지 함께 알려준다.
GLOSSARY: dict[str, tuple[str, bool]] = {
    # key: (설명, 프롬프트로 움직일 수 있는가)
    "n": ("sentences scored", False),
    "score": ("THE OBJECTIVE. Mean of `effective` over the T grid. This is the single "
              "number the loop maximises — everything else is diagnosis.", True),
    "format_pass_rate": ("fraction of sentences with no rule violation after one repair "
                         "retry. What still fails here is mostly the model rewriting the "
                         "source text, which more prompt wording does not fix.", False),
    "format_pass_rate_no_retry": ("same, before the repair retry. A low value costs money "
                                  "(one extra LLM call per sentence) but does not lower "
                                  "the score.", False),
    "rank_lift": ("how much `effective` DROPS when the confidence numbers are randomly "
                  "shuffled while the boundary positions stay the same. It isolates what "
                  "the RANKING does. Large positive = the ranking already works, refine it "
                  "only if the cases show mis-ranked boundaries. Near zero = the ranking "
                  "carries no information, so rewriting [Priority Rules] will not help and "
                  "the problem is WHERE boundaries are marked.", True),
    "rank_lift_t": ("`rank_lift` divided by its standard error. Read the sign and the "
                    "magnitude together — below ~1 the lift is not distinguishable from "
                    "noise.", False),
    "rank_lift_se": ("standard error of `rank_lift`.", False),
    "rank_lift_n": ("sentences that contributed to `rank_lift`.", False),
    "rank_lift_T": ("the T at which `rank_lift` was measured.", False),
    "by_T": ("results keyed by target piece size T. A LARGE key means few pieces, so only "
             "the TOP-RANKED boundaries survive; a SMALL key means many pieces, so "
             "lower-ranked boundaries survive too.", False),
    "target_chunk_words": ("the T of this row.", False),
    "effective": ("adequacy x (1 - contradiction). The per-T objective. Undefined (null) "
                  "for sentences with no boundary — those are excluded, not scored zero.",
                  True),
    "adequacy": ("translation quality of each piece against ITS OWN source, with no "
                 "reference translation — word order differences from an offline "
                 "translation do not lower it. Weighted by piece length.", True),
    "contradiction": ("at each boundary, how much the text emitted SO FAR contradicts the "
                      "whole-sentence translation. Averaged over the (k-1) boundaries. "
                      "This is what a boundary placed too early costs.", True),
    "effective_p10": ("the 10th percentile of `effective`. Read it next to the mean — the "
                      "mean is pulled by a few bad sentences, so a mean that rises while "
                      "p10 falls is luck, not improvement.", False),
    "effective_min": ("the worst sentence.", False),
    "n_effective": ("sentences where `effective` is defined (they had at least one "
                    "boundary).", False),
    "consistency": ("similarity of the concatenated pieces to the whole-sentence "
                    "translation. Reported only — not in the objective.", False),
    "laal_words": ("lag in source words, lower is faster. It is SET BY THE LATENCY BUDGET "
                   "(T), not by the prompt. Do not try to move it.", False),
    "chunks_per_sentence": ("mean pieces per sentence. Also set by T — the truncator cuts "
                            "to the budget whenever enough boundaries were marked. "
                            "**Telling the model to segment less does NOT reduce this**; it "
                            "only changes WHICH boundaries survive.", False),
    "missing_boundaries": ("how many boundaries the budget asked for that the prompt never "
                           "marked. Above zero means the knob cannot reach that T.", True),
    "rank_contra_spearman": ("rank correlation between assigned confidence and measured "
                             "contradiction. Superseded by `rank_lift`; reported only.",
                             False),
    "rank_contra_gap": ("bottom-half minus top-half boundary contradiction. Superseded by "
                        "`rank_lift`; its length-noise correction is currently under "
                        "review, so do not act on its sign.", False),
    "rank_contra_gap_se": ("standard error of `rank_contra_gap`.", False),
    "rank_contra_gap_n": ("sentences behind `rank_contra_gap`.", False),
    "effective_ent": ("`effective` recomputed with 1 - entailment instead of contradiction. "
                      "REPORTED ONLY, being trialled — the objective still uses "
                      "`effective`.", False),
    "contradiction_ent": ("`contradiction` measured as 1 - entailment. Reported only.",
                          False),
    "effective_z": ("multi-target only: `effective` z-normalised per target then averaged. "
                    "Used for the adoption test; `effective` is the reported value.", False),
    "n_targets": ("multi-target only: how many target languages were averaged.", False),
}


def describe(d: dict, _seen: set | None = None) -> str:
    """`Metrics.to_dict()` 에 실제로 들어 있는 키만 골라 설명을 만든다.

    프롬프트에 손으로 적으면 필드가 늘어도 안 따라온다. 여기서 생성하면 설명이 없는
    키가 `(no description)` 으로 드러나므로, 지표를 추가할 때 빠뜨린 게 보인다.
    """
    seen = _seen if _seen is not None else set()
    lines: list[str] = []
    for k, v in d.items():
        if k in seen:
            continue
        seen.add(k)
        if isinstance(v, dict) and k == "by_T":
            desc, fix = GLOSSARY.get(k, ("(no description)", False))
            lines.append(f"- {k}: {desc}")
            for sub in v.values():
                if isinstance(sub, dict):
                    lines.append(describe(sub, seen))
            continue
        desc, fix = GLOSSARY.get(k, ("(no description)", False))
        tail = "" if fix else "  [not movable by the prompt]"
        lines.append(f"- {k}: {desc}{tail}")
    return "\n".join(x for x in lines if x)


def score(m: Metrics) -> float:
    """**단일축.** T 격자에서의 `effective` 평균.

    가중치가 없는 이유는 두 축을 합치는 게 아니라 **같은 축을 T 별로 평균**하기
    때문이다. 지연은 노브가 고정하므로 목적함수에서 빠진다 — 프롬프트가 덜 잘라서
    점수를 얻는 경로가 구조적으로 없다.

    **포맷 하드 게이트(`-10 + fmt`)를 없앴다.** run01 에서 30문장 중 1건의 표기 위반이
    프롬프트를 `-9.03` 으로 폐기시켰고, 그 크기가 실제 프롬프트 차이(0.003)를 3000배
    압도해 hill climbing 이 "이번에 위반이 났는가"로 결정됐다. 지금은 표기 오류를
    `pipeline.normalize_tags` 가 결정론적으로 고치고, 남은 위반 문장은 채점에서
    제외되고, `format_pass_rate` 는 별도로 보고된다. 번역 비용 방어는
    `skip_translation_below` 가 계속 담당한다.

    `effective` 가 None 인 T(전 문장 무분절 — 비교군에서만 발생)는 평균에서 뺀다.
    """
    vals = [s.effective for s in m.by_T.values() if s.effective is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """동순위 평균 처리한 Spearman. 분산 0 이면 None."""
    def rankify(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rr = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rr[order[k]] = avg
            i = j + 1
        return rr
    rx, ry = rankify(xs), rankify(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def rank_contra_spearman(rows: list[dict], T: int, min_boundaries: int = 3) -> tuple[float | None, int]:
    """모델이 단 순위(<SEG:n>) vs 실측 경계 contradiction 의 문장 내 Spearman 평균.

    **양수 = 정렬됨** (순위 숫자가 클수록 = 확신 낮을수록 실측 위험이 큼).
    0 근처 = 순위가 위험과 무상관, 음수 = 거꾸로 — 절단이 위험을 줄이지 못하고
    `[Priority Rules]` 를 고치는 것이 근거 없는 일이 된다는 뜻이다.

    **주의 — 이 값은 원시(raw) contradiction 기준이라 길이 교란이 섞여 있다.**
    NLI 잡음 바닥은 hypothesis 길이에 따라 다르고 상위 순위 경계는 문장 앞쪽(짧은
    hypothesis)에 몰린다. run03 test 에서 raw −0.25 가 바닥 보정 후 **+0.14 로 뒤집혔다** —
    역전처럼 보인 것의 대부분이 위치 교란이었다.

    **그 보정의 방향이 현행 백엔드에서 뒤집혔다.** run03 은 `deberta-mnli`(짧을수록 바닥이
    큼: 1-2어절 0.113, 10어절+ 0.003)로 잰 것이고 그 백엔드는 삭제됐다. `xlmr-anli` 는
    반대로 **길수록 크다** (German 0.024 → 0.107, `runs/noise_floor_xlmr/`). 따라서 지금
    보정은 앞쪽 경계의 불리를 없애는 게 아니라 **키울 수 있다.** 이 값의 부호를 읽기 전에
    그 문서를 먼저 볼 것.

    가장 작은 T(경계 최다 생존)에서 재야 표본이 산다. 반환 (평균, 문장 수).
    """
    key = str(T)
    cors: list[float] = []
    for r in rows:
        d = (r.get("by_T") or {}).get(key)
        if not d:
            continue
        ranks = [int(m.group(1)) for m in TAG_RE.finditer(d.get("seg_text") or "")
                 if m.group(1)]
        contras = (d.get("pieces_contra") or [])[:-1]
        if len(ranks) != len(contras) or len(ranks) < min_boundaries:
            continue
        c = _spearman([float(x) for x in ranks], [float(x) for x in contras])
        if c is not None:
            cors.append(c)
    return (sum(cors) / len(cors) if cors else None), len(cors)


def _floor_corrected(d: dict, contras: list[float], floor_fn,
                     tgt_spaced: bool = True) -> list[float]:
    """경계별 contradiction 에서 길이 잡음 바닥 c0(hypothesis 길이)를 뺀다.

    경계 j 의 hypothesis 는 조각 번역 1..j+1 을 이어붙인 것이므로 길이가 j 와 함께
    자란다. 바닥이 길이에 따라 출렁이므로 보정 없이는 특정 위치의 경계가 구조적으로
    불리해진다.

    **방향은 현행 백엔드에서 뒤집혔다.** "짧을수록 크다" 는 삭제된 `deberta-mnli` 성질이고,
    `xlmr-anli` 는 길수록 크다 (German 0.024 → 0.107, `runs/noise_floor_xlmr/`).

    길이는 **타깃 표기 체계**로 센다 (`unit_count`). 어절로 고정하면 무공백 타깃(ja/zh)
    에서 조각을 공백으로 이어붙인 문자열의 `split()` 이 **조각 수**를 세게 되어, 실제
    문자 수와 무관하게 전부 최단 버킷으로 떨어진다 — 바닥이 통째로 엉뚱해진다.
    """
    if floor_fn is None:
        return contras
    pieces_tgt = d.get("pieces_tgt") or []
    out = []
    for j, c in enumerate(contras):
        hyp = " ".join(p for p in pieces_tgt[: j + 1] if p)
        out.append(max(0.0, c - floor_fn(max(1, unit_count(hyp, tgt_spaced)))))
    return out


def rank_contra_gap(rows: list[dict], T: int, min_boundaries: int = 2,
                    floor_fn=None, tgt_spaced: bool = True) -> tuple[float | None, int]:
    """순위 **하위 절반 − 상위 절반**의 경계 contradiction 평균 차. 문장 평균.

    `rank_contra_spearman` 과 같은 축을 다른 통계량으로 잰다. Spearman 은 순위 상관만
    보므로 "정렬은 됐는데 격차가 없다"를 못 가르지만, 이 값은 **크기**를 재므로
    절단이 실제로 위험을 얼마나 덜어내는지가 나온다.

    **양수 = 정렬됨** — 확신 낮다고 매긴 경계가 실제로 더 반박당함. 상위만 남기는
    절단이 위험을 덜어낸다는 뜻.
    **0 이하 = 순위가 정보를 안 준다** — 상위 경계가 하위와 같거나 더 위험하다.
    이 경우 노브를 조여도 품질이 안 오르고, 고칠 곳은 위치가 아니라 `[Priority Rules]`
    다.

    임계값이 **0** 인 것이 이 지표를 쓰는 이유다. 종전의 T 대비
    (`adequacy(작은 T) − adequacy(큰 T) > PRIORITY_MARGIN`)는 두 문제가 있었다.
      1. 두 집합이 **중첩**이라(`keep(큰 T) ⊆ keep(작은 T)`) "하위가 상위보다 나은가"가
         아니라 "상위에 하위를 얹으면 나아지는가"를 쟀다.
      2. 조각 수가 함께 바뀌어 QE 길이 편향이 섞였다 — run04 실측에서 작은 T 가
         일관되게 +0.003~0.005 높았고, 그 부호가 진단과 같아 신호와 편향이 분리되지
         않았다. `PRIORITY_MARGIN = 0.03` 은 그 편향을 덮으려는 잠정 상수였다.
    여기서는 상위/하위가 **배타적이고 동수**라 두 교란이 모두 상쇄되고, "순위에 정보가
    없다"의 기준점이 임의 상수가 아니라 0 이 된다.

    홀수 개일 때 가운데 순위는 버린다 — 어느 쪽에도 속하지 않는 값을 한쪽에 넣으면
    대비가 흐려진다.

    `floor_fn(hyp_words) -> c0` 를 주면 길이 잡음 바닥을 뺀 뒤 계산한다. 안 주면 raw 라
    앞쪽(상위 순위) 경계가 불리해져 **음수 쪽으로 편향**된다 (run03 test: raw −0.25 →
    보정 후 +0.14). 루프는 `loop.load_contra_floor` 로 런당 1회 측정해 넘긴다.

    가장 작은 T(경계 최다 생존)에서 재야 표본이 산다. 반환 (평균, 문장 수).
    **오차막대가 필요하면 `rank_contra_gaps` 로 문장별 값을 받을 것** — dev 150문장에서
    se ≈ 0.018 이라 +0.03 수준의 격차는 2·se 를 못 넘는다. 점추정만 보고 "정렬됐다"고
    읽으면 안 된다.
    """
    gaps = rank_contra_gaps(rows, T, min_boundaries, floor_fn, tgt_spaced)
    return (sum(gaps) / len(gaps) if gaps else None), len(gaps)


def rank_contra_gaps(rows: list[dict], T: int, min_boundaries: int = 2,
                     floor_fn=None, tgt_spaced: bool = True) -> list[float]:
    """`rank_contra_gap` 의 **문장별** 값. 평균 내기 전 분포가 필요할 때 쓴다."""
    key = str(T)
    gaps: list[float] = []
    for r in rows:
        d = (r.get("by_T") or {}).get(key)
        if not d:
            continue
        ranks = [int(m.group(1)) for m in TAG_RE.finditer(d.get("seg_text") or "")
                 if m.group(1)]
        contras = (d.get("pieces_contra") or [])[:-1]
        if len(ranks) != len(contras) or len(ranks) < min_boundaries:
            continue
        vals = _floor_corrected(d, [float(x) for x in contras], floor_fn, tgt_spaced)
        order = sorted(range(len(ranks)), key=lambda i: ranks[i])   # 순위 오름차순
        half = len(order) // 2
        top = [vals[i] for i in order[:half]]                 # 확신 높음 (번호 작음)
        bottom = [vals[i] for i in order[len(order) - half:]]  # 확신 낮음
        gaps.append(sum(bottom) / len(bottom) - sum(top) / len(top))
    return gaps


def priority_audit(rows: list[dict], T: int, floor_fn=None, tgt_spaced: bool = True,
                   min_n: int = 8) -> list[dict]:
    """모델 순위가 **무엇을 과신하는가** 를 특징별로 대조한다.

    `rank_contra_gap` 은 "순위가 정보를 주는가"만 답한다. 0 이하라는 사실만으로는 PE 가
    `[Priority Rules]` 의 **어느 줄**을 고쳐야 하는지 알 수 없어, 눈감고 재작성하다
    실패한다 (en-de run02 iter1~2, 수동 시도 2회 모두 동일).

    여기서는 경계를 표면 특징으로 묶어 **모델이 매긴 순위 백분위**와 **실측 contradiction**
    을 나란히 낸다. 백분위가 낮은데(=확신 높음) contra 가 높으면 그 특징이 과신 대상이다.
    en-de 실측에서 이 대조가 "쉼표 경계: 백분위 0.35 / contra 0.108 vs 그 외 0.64 / 0.062"
    를 뽑아냈고, 그걸 프롬프트에 반영한 것이 순위를 유의하게 개선한 **유일한** 개입이었다
    (gap −0.005 → +0.032, 순열 p=0.005).

    **언어 자원을 쓰지 않는다** — 특징은 구두점(측정 프로파일에서 온다)과 상대 위치뿐이다.
    반환은 `over_trust` 내림차순. 각 항목의 `over_trust` 는
    `(중앙 백분위 − 이 특징 백분위) × contra 비` 로, 양수가 클수록 과신이다.
    """
    key = str(T)
    recs: list[tuple[str, float, float]] = []      # (특징, 순위 백분위, contra)
    for r in rows:
        d = (r.get("by_T") or {}).get(key)
        if not d:
            continue
        ranks = [int(m.group(1)) for m in TAG_RE.finditer(d.get("seg_text") or "")
                 if m.group(1)]
        contras = (d.get("pieces_contra") or [])[:-1]
        src = d.get("pieces_src") or []
        if len(ranks) != len(contras) or len(ranks) < 3 or len(src) - 1 != len(contras):
            continue
        vals = _floor_corrected(d, [float(x) for x in contras], floor_fn, tgt_spaced)
        n = len(ranks)
        order = sorted(range(n), key=lambda i: ranks[i])
        pct = {i: (k + 1) / n for k, i in enumerate(order)}
        for j, c in enumerate(vals):
            tail = (src[j] or "").rstrip()[-1:]
            feat = f"뒤 구두점 {tail!r}" if tail and not tail.isalnum() else "구두점 없음"
            recs.append((feat, pct[j], c))
            recs.append((f"상대위치 {int(j / max(1, n) * 3)}/3", pct[j], c))
    if not recs:
        return []
    base_pct = sum(r[1] for r in recs) / len(recs)
    base_con = sum(r[2] for r in recs) / len(recs) or 1e-9
    out = []
    for feat in sorted({r[0] for r in recs}):
        g = [r for r in recs if r[0] == feat]
        if len(g) < min_n:
            continue
        p = sum(x[1] for x in g) / len(g)
        c = sum(x[2] for x in g) / len(g)
        out.append({"feature": feat, "n": len(g),
                    "rank_percentile": round(p, 3),
                    "contradiction": round(c, 4),
                    "over_trust": round((base_pct - p) * (c / base_con), 4)})
    return sorted(out, key=lambda d: -d["over_trust"])


def paired_delta(new_rows: list[dict], best_rows: list[dict],
                 t_grid: list[int], key: str = "effective") -> dict:
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
            # 무분절 문장은 effective 가 미정의(None)다. 어느 한쪽이라도 None 이면
            # 그 T 는 쌍체 비교에서 뺀다 — 0 으로 치면 "분절 안 함"이 이득이 된다.
            if a.get(key) is None or c.get(key) is None:
                continue
            d += a[key] - c[key]
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


def rank_lift(real: list[float | None], shuffled: list[float | None]) -> dict:
    """순위를 무작위로 섞었을 때 잃는 `effective` — 문장별 쌍체.

    `paired_delta` 와 같은 계산이지만 입력이 **행이 아니라 문장별 값 두 벌**이다.
    비교 대상이 다른 프롬프트가 아니라 **같은 프롬프트의 순위를 망가뜨린 대조군**이라
    id 매칭이 필요 없고, 같은 문장·같은 조각 수·같은 후보 집합이므로 쌍체가 정확하다.

    **se 는 실제 산포를 15~25% 과소평가한다** — 어느 순열을 뽑았는지의 무작위성이 이
    공식에 안 잡히기 때문이다 (실측: 자체 se 0.0168 vs 셔플 190쌍의 산포 0.0210).
    `t` 를 절대적으로 읽지 말 것. 문턱은 실측 발화율로 잡았다 (agents.RANK_LIFT_T_MIN).
    """
    d = [a - b for a, b in zip(real, shuffled) if a is not None and b is not None]
    if len(d) < 2:
        return {"lift": None, "se": None, "t": None, "n": len(d)}
    mean_d = sum(d) / len(d)
    var = sum((x - mean_d) ** 2 for x in d) / len(d)
    se = (var / len(d)) ** 0.5
    return {"lift": round(mean_d, 5), "se": round(se, 5),
            "t": round(mean_d / se, 2) if se else None, "n": len(d)}


def mechanical_split(text: str, every: int, spaced: bool) -> str:
    """의미를 무시하고 every 글자마다 자르는 비교군 (곡선의 하한 기준선).

    순위를 달지 않는다 — 절단할 근거가 없으므로 곡선 위의 점 하나로만 평가된다."""
    body = text if spaced else _WS.sub("", text)
    parts = [body[i : i + every] for i in range(0, len(body), every)]
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 1:
        return text
    return " <SEG> ".join(parts)
