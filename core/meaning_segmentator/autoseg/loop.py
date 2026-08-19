"""A8 Loop Controller — 결정론적 오케스트레이터.

LLM 판단은 없다. 실행 순서, 채택/롤백, 예산, 중단 조건만 관리한다.
설계는 `../AUTO_PROMPT_LOOP_DESIGN.md`.

  python -m core.meaning_segmentator.autoseg.loop \
      --dataset kspon --src-lang Korean --tgt-lang English \
      --translator google --iterations 6 --train 30 --dev 60 --test 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import agents, data, metrics, noise_floor
from .gateway import BudgetExceeded, Gateway
from .pipeline import (GoogleTranslator, JsonCache, Translator, blocks_scoring,
                       chunk_budget, normalize_tags, segment_batch, split_segments,
                       to_lang_code, truncate, unit_count, validate)

_HERE = Path(__file__).resolve().parent

# 타깃 언어의 표기 체계 — LAAL 의 목표측 토큰 수를 세는 단위를 정한다.
_UNSPACED_TARGETS = {"japanese", "chinese", "thai", "ja", "zh", "th"}


def target_is_spaced(name: str) -> bool:
    return name.strip().lower() not in _UNSPACED_TARGETS


# ── 채점 코어 ────────────────────────────────────────────────────────────

@dataclass
class ScoredSplit:
    """분절된 텍스트 한 벌을 채점한 결과. 전부 문장 단위 리스트다 (`pieces_*` 만 중첩).

    `contradiction`·`effective` 는 무분절 문장(k=1)에서 None 이다 — 경계가 없어
    모순 노출 자체가 없으므로 0(무죄)이 아니라 미정의."""
    joined: list[str]                    # 조각 번역 합본
    pieces_src: list[list[str]]          # 조각 원문
    pieces_tgt: list[list[str]]          # 조각 번역
    pieces_contra: list[list[float]]     # 경계별 모순 확률. 마지막 원소는 항상 0.0
    effective: list[float | None]
    adequacy: list[float]
    contradiction: list[float | None]
    consistency: list[float]
    chrf: list[float]
    laal_words: list[float]
    k: list[int]


def score_split(seg_texts: list[str], texts: list[str], full: list[str],
                translator, adequacy: metrics.AdequacyBackend,
                consistency: metrics.QualityBackend, spaced: bool, tgt_spaced: bool,
                contradiction: "metrics.ContradictionBackend | None" = None) -> ScoredSplit:
    """분절 텍스트 한 벌 -> 문장별 지표.

    루프(노브 T 로 절단한 것)와 비교군(무분절·기계분절)이 **같은 자로** 재져야 하므로
    한 함수로 둔다. 예전에는 두 벌로 복제돼 있었고, 한쪽만 고친 탓에
    `NameError: name 'eff' is not defined` 로 최종 리포트 직전에 죽은 적이 있다.

    `seg_texts` 는 `<SEG>` 가 남아 있는 채점 대상, `texts` 는 원문(consistency 의 소스),
    `full` 은 오라클 전체 번역이다.
    """
    joined, pieces = translator.seg_batch(seg_texts, full)

    # adequacy 는 **조각 단위**다. 문장 점수는 조각 길이로 가중 평균한다 —
    # 1어절 조각과 10어절 조각을 같은 무게로 세면 짧은 조각을 많이 만드는 쪽이
    # 유리해진다 (v1 의 지연 프록시가 같은 허점으로 무너졌다).
    pair_src: list[str] = []
    pair_tgt: list[str] = []
    owner: list[int] = []
    chunk_lists: list[list[str]] = []
    for i, (st, ps) in enumerate(zip(seg_texts, pieces)):
        chunks = split_segments(st)
        ps = list(ps)[: len(chunks)] + [""] * max(0, len(chunks) - len(ps))
        chunk_lists.append(chunks)
        for c, p in zip(chunks, ps):
            pair_src.append(c)
            pair_tgt.append(p)
            owner.append(i)
    qe = adequacy.score(pair_src, pair_tgt) if pair_src else []

    # 조기 방출 — 조각 j 뒤의 경계에서 "그 시점까지 방출된 누적 번역"이 오라클(full
    # 번역)과 모순되는가. 마지막 조각 뒤에는 미래가 없으므로 대상이 아니다.
    contra = [0.0] * len(pair_src)
    if contradiction is not None and pair_src:
        prem, hyp, slot = [], [], []
        pos = 0
        for i, ps in enumerate(pieces):
            k_i = len(chunk_lists[i])
            for j in range(k_i):
                if j < k_i - 1:
                    prem.append(full[i])
                    hyp.append(" ".join(x for x in list(ps)[: j + 1] if x).strip())
                    slot.append(pos)
                pos += 1
        for s, v in zip(slot, contradiction.score(prem, hyp)):
            contra[s] = v

    # 경계별 값을 문장에 되돌린다. 문장 평균만 남기면 **어느 경계가** 반박당했는지가
    # 사라지는데, 판정자·비평가 조준에 필요한 것이 바로 그 위치다 (이미 계산된 값).
    # 각 문장 마지막 원소는 뒤에 미래가 없어 항상 0.0 — "안전"이 아니라 "대상 아님".
    contra_rows: list[list[float]] = [[] for _ in texts]
    for i, v in zip(owner, contra):
        contra_rows[i].append(round(v, 4))

    def weighted(vals):
        num = [0.0] * len(texts)
        den = [0.0] * len(texts)
        for i, c, v in zip(owner, pair_src, vals):
            w = float(max(1, unit_count(c, spaced)))
            num[i] += v * w
            den[i] += w
        return [n / d if d else 0.0 for n, d in zip(num, den)]

    # 문장 contradiction = **경계 (k−1)개의 평균.** 마지막 조각 자리(구조적 0)를
    # 평균에 넣으면 k 가 클수록 문장 값이 기계적으로 올라 무분절이 "노출이 없어서"
    # 자동 만점을 받는다. 경계 평균은 iid 잡음 기대값이 k 무관이라 노출이 정규화되고,
    # 무분절은 0 이 아니라 미정의(None)가 되어 집계에서 빠진다.
    adequacy_rows = weighted(qe)
    contradiction_rows: list[float | None] = []
    effective_rows: list[float | None] = []
    for adq, cr in zip(adequacy_rows, contra_rows):
        bounds = cr[:-1]
        if bounds:
            c_mean = sum(bounds) / len(bounds)
            contradiction_rows.append(c_mean)
            effective_rows.append(metrics.effective_of(adq, c_mean))
        else:
            contradiction_rows.append(None)
            effective_rows.append(None)

    return ScoredSplit(
        joined=list(joined),
        pieces_src=chunk_lists,
        pieces_tgt=[list(p) for p in pieces],
        pieces_contra=contra_rows,
        effective=effective_rows,
        adequacy=adequacy_rows,
        contradiction=contradiction_rows,
        consistency=consistency.score(texts, joined, full),
        chrf=[metrics.chrf(h, r) for h, r in zip(joined, full)],
        laal_words=[metrics.laal_words(st, ps, f, spaced, tgt_spaced)
                    for st, ps, f in zip(seg_texts, pieces, full)],
        k=[max(1, len(c)) for c in chunk_lists],
    )


# ── 평가 1회 ─────────────────────────────────────────────────────────────

def evaluate(
    gw: Gateway,
    translator,
    prompt: str,
    sentences: list[data.Sentence],
    spaced: bool,
    seg_cache: JsonCache,
    workers: int,
    adequacy: metrics.AdequacyBackend,
    consistency: metrics.QualityBackend,
    t_grid: list[int],
    trailing_punct: str | None = None,
    skip_translation_below: float = 0.95,
    tgt_spaced: bool = True,
    require_priority: bool = True,
    require_coverage: bool = True,
    contradiction: "metrics.ContradictionBackend | None" = None,
    coverage_t: int | None = None,
    reasoning_effort: str | None = None,
    priority_depth: int | None = None,
    batch_size: int = 1,
    min_gap: int = 0,
) -> tuple[list[dict], metrics.Metrics, list[dict]]:
    """분절 1회 + 노브 값마다 번역·채점.

    분절 호출은 **T 와 무관하게 한 번**이다. 순위 태그가 붙어 나오므로 이후 조각 수
    조절은 결정론적 절단이고, 곡선 전체가 추론 1회로 나온다.
    """
    texts = [s.text for s in sentences]
    # 커버리지 요건은 **가장 조인 예산**에서 온다. 그보다 적게 찍으면 그 T 에서 노브가
    # 무력해지므로, 요건이 곡선에 그릴 최소 T 를 기준으로 잡혀야 격자 전체가 의미를 갖는다.
    #
    # `coverage_t` 는 그래서 `t_grid` 와 분리돼 있다. 둘을 묶어 두면 루프(격자 3 6)와
    # 최종 평가(격자 2 3 4 6)가 **서로 다른 요건**을 쓰게 된다 — run03 에서 실제로
    # T=3 요건으로 최적화된 프롬프트가 T=2 요건으로 심판받아 test 1차 통과율이
    # 0.34 까지 떨어졌다 (train/dev 는 0.63~0.98).
    min_t = coverage_t or (min(t_grid) if t_grid else 3)
    need = lambda txt: (max(0, chunk_budget(txt, min_t, spaced) - 1)
                        if require_coverage else None)

    first_pass_viol: list[dict] = []
    seg_texts, first_pass = segment_batch(
        gw, prompt, texts, cache=seg_cache, workers=workers,
        validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct,
                                            require_priority, need(t), priority_depth),
        normalize_fn=lambda out: normalize_tags(out, spaced, trailing_punct),
        reasoning_effort=reasoning_effort,
        batch_size=batch_size,
        first_pass_sink=first_pass_viol,
    )

    violations: list[dict] = []
    valid_flags: list[bool] = []
    scored_flags: list[bool] = []
    for s, seg in zip(sentences, seg_texts):
        vs = validate(s.id, s.text, seg, spaced, trailing_punct, require_priority,
                      need(s.text), priority_depth)
        valid_flags.append(not vs)
        scored_flags.append(not blocks_scoring(vs))
        violations.extend({"id": v.id, "rule": v.rule, "detail": v.detail,
                           "text": s.text, "seg_text": seg} for v in vs)

    # 저비용 게이트는 **원문 훼손** 비율로 판단한다. 커버리지 미달로 번역을
    # 통째로 건너뛰면 개선 신호가 사라진다 — 커버리지는 채점에서 다뤄야 한다.
    rate = sum(scored_flags) / len(scored_flags) if scored_flags else 0.0
    rows = [{"id": s.id, "text": s.text, "seg_text": seg, "valid": v,
             "full_trans": None, "by_T": {}}
            for s, seg, v in zip(sentences, seg_texts, valid_flags)]

    # 포맷이 무너진 프롬프트에는 번역·채점을 쓰지 않는다 (저비용 게이트 먼저)
    if rate < skip_translation_below:
        for v in first_pass_viol:
            v["first_pass"] = True
        return (rows, metrics.aggregate(len(rows), valid_flags, first_pass, {}),
                violations + first_pass_viol)

    full = translator.full(texts)
    for r, f in zip(rows, full):
        r["full_trans"] = f

    by_T: dict[str, metrics.SplitMetrics] = {}
    for T in t_grid:
        cut = [truncate(seg, T, spaced, min_gap) for seg in seg_texts]
        cut_texts = [c[0] for c in cut]
        missings = [c[1] for c in cut]

        sp = score_split(cut_texts, texts, full, translator, adequacy, consistency,
                         spaced, tgt_spaced, contradiction)

        for i, r in enumerate(rows):
            r["by_T"][str(T)] = {
                "seg_text": cut_texts[i], "k": sp.k[i], "missing_boundaries": missings[i],
                "pieces_src": sp.pieces_src[i], "pieces_tgt": sp.pieces_tgt[i],
                "pieces_contra": sp.pieces_contra[i], "seg_trans": sp.joined[i],
                "effective": (round(sp.effective[i], 4)
                              if sp.effective[i] is not None else None),
                "adequacy": round(sp.adequacy[i], 4),
                "contradiction": (round(sp.contradiction[i], 4)
                                  if sp.contradiction[i] is not None else None),
                "consistency": round(sp.consistency[i], 4),
                "chrf": round(sp.chrf[i], 4), "laal_words": round(sp.laal_words[i], 4),
            }
        # **포맷 위반 문장은 채점에서 뺀다.** 규칙을 어긴 분절의 점수는 의미가 없고,
        # 반대로 위반 1건으로 프롬프트 전체를 폐기하면 표본 잡음이 신호를 압도한다.
        keep = [i for i, v in enumerate(scored_flags) if v]
        pick = lambda xs: [xs[i] for i in keep]
        by_T[str(T)] = metrics.aggregate_split(
            T, pick(sp.effective), pick(sp.adequacy), pick(sp.contradiction),
            pick(sp.consistency), pick(sp.chrf), pick(sp.laal_words), pick(sp.k),
            pick(missings), n_total=len(rows))

    for v in first_pass_viol:
        v["first_pass"] = True
    return (rows, metrics.aggregate(len(rows), valid_flags, first_pass, by_T),
            violations + first_pass_viol)


def load_contra_floor(run_dir, rows: list[dict], backend,
                      filename: str = "contra_floor.json", tgt_spaced: bool = True):
    """경계 contradiction 의 **길이별 잡음 바닥**. 런당 1회 측정하고 디스크에 캐시한다.

    NLI 는 무해한 미완성에도 0 이 아닌 모순 확률을 준다 (짧을수록 크다: run03 실측
    1-2어절 0.113, 10어절+ 0.003). 상위 순위 경계는 문장 앞쪽 = 짧은 hypothesis 에
    몰리므로, 보정 없이 순위별 위험을 비교하면 **상위 순위가 구조적으로 불리**하다.
    run03 test 에서 이 교란만으로 순위 정렬도가 −0.25 로 나왔고 보정 후 +0.14 였다.

    바닥은 (코퍼스, 번역기, NLI 백엔드)의 성질이지 프롬프트의 성질이 아니다 — full
    번역은 이터레이션 간 불변이므로 다시 잴 이유가 없다. 번역 호출은 0 이고 NLI 만 돈다.

    반환: `floor_fn(hyp_words) -> c0`. 잴 수 없으면 None (보정 없이 raw 로 진행).

    `filename` 은 타깃 언어마다 바닥이 다르기 때문에 있다 — full 번역이 달라지면
    바닥도 달라진다 (`multilingual_check.py`).
    """
    if backend is None:
        return None
    fp = run_dir / filename
    if fp.exists():
        floor = json.loads(fp.read_text(encoding="utf-8"))
    else:
        fulls = [r["full_trans"] for r in rows if r.get("full_trans")]
        if not fulls:
            return None
        floor = noise_floor.measure_floor(fulls, backend, tgt_spaced=tgt_spaced)
        fp.write_text(json.dumps(floor, ensure_ascii=False, indent=2), encoding="utf-8")
    return lambda n: noise_floor.floor_lookup(floor, n)


def fmt_by_purpose(u: dict, top: int = 6) -> str:
    """용도별 누적 비용 한 줄. **합계만으로는 병목이 안 보인다** — run05 에서 분절이
    비용의 90% 라는 것을 호출 수로 역산해야 했다 (gateway.Usage.by_purpose 참조)."""
    bp = u.get("by_purpose") or {}
    if not bp:
        return "(없음)"
    total = sum(v["cost"] for v in bp.values()) or 1.0
    parts = [f"{k} ${v['cost']:.3f}({v['cost'] / total * 100:.0f}%, {v['calls']}콜"
             + (f", 사고 {v['reasoning_tokens'] // max(1, v['calls']):,}tok/콜)"
                if v.get("reasoning_tokens") else ")")
             for k, v in list(bp.items())[:top]]
    return "  ".join(parts)


def _cell(v, spec: str, dash: str = "—") -> str:
    """무분절엔 effective/contradiction 이 미정의(None)다. 0 으로 찍으면 오독된다."""
    return format(v, spec) if v is not None else dash


def fmt_metrics(m: metrics.Metrics, tag: str) -> str:
    parts = [f"{tag} fmt={m.format_pass_rate:.2f}(1st {m.format_pass_rate_no_retry:.2f})"]
    for k in sorted(m.by_T, key=int):
        s = m.by_T[k]
        parts.append(f"T{k}: eff={_cell(s.effective, '.4f')} adq={s.adequacy:.4f} "
                     f"contra={_cell(s.contradiction, '.3f')} laal={s.laal_words:.2f} "
                     f"k={s.chunks_per_sentence:.2f} miss={s.missing_boundaries:.2f}")
    parts.append(f"score={metrics.score(m):.4f}")
    return "  ".join(parts)


# ── 메인 ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="의미 분절 프롬프트 자동 생성 루프 (v2)")
    p.add_argument("--dataset", default="kspon", choices=sorted(data.LOADERS))
    p.add_argument("--src-lang", default="Korean")
    p.add_argument("--tgt-lang", default="English")
    p.add_argument("--run-id", default=None)
    p.add_argument("--pair-id", default=None,
                   help="런 디렉토리 이름. 미지정 시 언어명에서 생성")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--judge-model", default=None,
                   help="판정자 모델. 미지정 시 --model. 분절기와 다른 모델을 쓰면 순환이 준다")
    p.add_argument("--translator-model", default="gpt-5-mini")
    # 비용의 98% 가 분절 호출의 사고 토큰이다 (run05 실측). low 는 medium 대비 2.7배
    # 싸면서 태그 수·원문 보존이 같거나 낫고, gpt-5.4-mini 는 명시하지 않으면 사고를
    # 아예 안 해 태그가 필요량의 1/3 로 떨어진다. 자세한 실측은 gateway.Gateway.chat.
    # 에이전트는 이터레이션당 10콜 안팎이라 비용 비중이 작다. 여기서 아끼면 프롬프트
    # 개정 품질이 떨어지므로 gpt-5-mini 의 기본값이던 medium 을 유지한다.
    # 저비용 게이트. 원문 훼손(text_modified) 비율이 이 값을 못 넘으면 번역·채점을
    # 통째로 건너뛴다. **분할 크기에 따라 유효 허용 건수가 달라진다** — 0.95 는
    # train 60 에서 3건, train 40 에서 2건까지만 봐준다. en-de run01 iter0 에서
    # 4/60(=0.933)이 걸려 train 이 통째로 미채점됐고, score=0 이 best 로 기록되면서
    # Critic 이 채점된 train 행 없이 조향했다.
    # v0 후보 개수와 선별 표본. 1 이면 종전 동작(첫 골격 통과본 사용).
    # 이터레이션당 만들 개정 후보 수. 1 이면 종전(제안 1개 = 검증 1개).
    # 2 이상이면 첫 개는 자유 개정, 나머지는 Critic 의 `proposed_rule` 을 하나씩만 반영.
    p.add_argument("--revision-candidates", type=int, default=1,
                   help="이터레이션당 개정 후보 수. 2 이상이면 probe 로 골라 쓴다")
    p.add_argument("--v0-candidates", type=int, default=1,
                   help="prompt_v0 후보 수. 2 이상이면 dev 일부로 골라 시작한다")
    p.add_argument("--v0-probe", type=int, default=40,
                   help="후보 1차 선별에 쓸 dev 문장 수. 실측 1위적중 20문장 36%% / 40문장 58%%")
    # 한 분절 호출에 넣을 문장 수. **비용의 유일한 큰 레버**다 — en-de test 100문장 실측:
    # b=1 $1.05 → b=6 $0.47 (55% 절감), 쌍체 Δ(T6) −0.0026±0.0056 으로 품질 차이 검출 안 됨.
    # **b=12 부터 무너진다**: 1차 통과율 0.75(b=12)·0.27(b=24)로 떨어져 단건 재시도가
    # 폭증하고, 비용이 U자로 되돌아오면서(b=12 $0.53, b=24 $0.80) 품질도 b=12 에서
    # Δ −0.019(t=−2.0)로 유일하게 유의하게 나빠졌다. 6 을 넘기지 말 것.
    # 절단기 최소 간격. T 는 평균이지 하한이 아니라 1~2어절 조각이 섞여 나온다
    # (T=6 에서 47/100 문장). en-de 오프라인 시뮬 최적값 3 (pipeline.truncate 참조).
    # 언어쌍별로 다시 잴 것 — en 소스에서만 검증했다.
    p.add_argument("--min-gap", type=int, default=0,
                   help="절단 시 경계 간 최소 어절 간격. 0=끔. en-de 실측 최적 3")
    p.add_argument("--batch-size", type=int, default=1,
                   help="한 분절 호출에 넣을 문장 수. 실측 최적 6, 12 이상은 역효과")
    p.add_argument("--candidate-t", type=int, default=None,
                   help="후보 마킹 하한 기준 T. 작을수록 많이 찍는다. 미지정 시 min(--final-t-grid)")
    p.add_argument("--skip-translation-below", type=float, default=0.95,
                   help="원문 보존율이 이 값 미만이면 번역·채점 생략 (0 = 항상 채점)")
    p.add_argument("--agent-reasoning-effort", default="medium",
                   choices=["minimal", "low", "medium", "high", "none"],
                   help="Profiler/Judge/Critic/PE 사고량. none = 모델 기본값")
    p.add_argument("--seg-reasoning-effort", default="low",
                   choices=["minimal", "low", "medium", "high", "none"],
                   help="분절 호출 사고량. none = 모델 기본값. 에이전트 호출에는 영향 없음")
    p.add_argument("--translator", default="google", choices=["llm", "google"])
    p.add_argument("--tgt-code", default=None)
    p.add_argument("--no-google-context", action="store_true")
    p.add_argument("--tgt-spaced", default=None, choices=["yes", "no"],
                   help="타깃 언어가 띄어쓰기를 쓰는가. 미지정 시 --tgt-lang 에서 추론 (LAAL 단위)")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--train", type=int, default=30)
    p.add_argument("--train-pool", type=int, default=None)
    p.add_argument("--max-prompt-growth", type=float, default=1.3)
    p.add_argument("--dev", type=int, default=60)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--min-chars", type=int, default=25)
    # 노브. 루프에서는 부분집합만 쓴다 — 조각 번역이 격자 크기에 비례해 늘기 때문이다.
    p.add_argument("--t-grid", type=int, nargs="+", default=[3, 6],
                   help="루프가 쓰는 목표 조각 크기. score 는 이 격자에서의 adequacy 평균이라 "
                        "다른 격자로 잰 score 와 비교할 수 없다")
    p.add_argument("--final-t-grid", type=int, nargs="+", default=[2, 3, 4, 6],
                   help="최종 test 곡선용 격자")
    p.add_argument("--main-t", type=int, default=None,
                   help="판정자가 도는 주 작동점. 미지정 시 --t-grid 의 중앙값")
    p.add_argument("--judge-rows", type=int, default=8,
                   help="이터레이션당 판정할 문장 수 (adequacy 하위부터)")
    p.add_argument("--no-judge", action="store_true", help="판정자를 끄고 adequacy 만으로 조향")
    p.add_argument("--adequacy-backend", default="cometkiwi",
                   choices=sorted(metrics.QE_CHECKPOINTS),
                   help="참조 없는 QE. y축 주지표")
    p.add_argument("--consistency-backend", default="nli",
                   choices=["nli", "comet", "xcomet", "embed", "chrf"],
                   help="가설 검증값(보고용). 기본 nli = 합본 vs full 의 양방향 entailment — "
                        "어순 무관. 모델은 --contradiction-backend 를 따른다. "
                        "comet 계열은 참조 기반이라 어순 편향이 있다")
    # `xlmr-anli` 로 바꾼 근거 (en-de test 100문장 + 관문 6케이스 실측, 2026-08-19):
    #   관문 최소 여유   mdeberta-xnli 0.0027 (통과선상) / deberta-mnli 미측정 / xlmr-anli 0.0994
    #   5개 타깃 곡선   mdeberta 2/5 정상 (ko/zh/ja 역전) / xlmr-anli 5/5
    #   잡음 바닥       mdeberta 0.102 — 실측 신호 0.075 보다 커서 사실상 무정보
    # 대가가 있다: 문장별 분산이 커져 dev 쌍체 se 가 0.0065 -> 0.0144 로 배증한다.
    # 채택 문턱이 `Δ > adopt_se_mult·se` 라 그만큼 보수화되므로 기본 배수를 함께 낮춘다.
    p.add_argument("--contradiction-backend", default="xlmr-anli",
                   choices=sorted(metrics.NLI_MODELS),
                   help="조기 방출 검출 NLI. 다국어. 영어 전용이면 deberta-mnli 도 가능")
    p.add_argument("--no-coverage-rule", action="store_true",
                   help="최소 경계 수 요건을 끈다. 노브가 k 를 통제하지 못하게 된다")
    p.add_argument("--no-contradiction", action="store_true",
                   help="NLI 를 끈다. effective = adequacy 가 되어 조기 방출이 벌받지 않는다")
    p.add_argument("--comet-batch-size", type=int, default=16)
    # 1.0 -> 0.5. `xlmr-anli` 는 지표 타당도가 훨씬 낫지만 문장별 분산이 커서 dev 쌍체
    # se 가 0.0065 -> 0.0144 로 배증한다 (run03 재채점 실측). 배수를 그대로 두면 문턱이
    # 두 배가 되어 채택이 더 어려워진다 — run01~03 이 이미 채택 0회다.
    p.add_argument("--adopt-se-mult", type=float, default=0.5,
                   help="채택 요건: dev 쌍체 Δ > (이 값)·se. 점추정 비교는 오차막대 안 "
                        "잡음까지 채택했다. 0 이면 이전 방식(점 비교)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--final-only", action="store_true",
                   help="이터레이션을 건너뛰고 기존 best_prompt.txt 로 최종 test 평가만 "
                        "다시 돈다. 루프는 끝났는데 마지막 단계가 죽었을 때 쓴다")
    args = p.parse_args()
    for _f in ("seg_reasoning_effort", "agent_reasoning_effort"):
        if getattr(args, _f) == "none":
            setattr(args, _f, None)           # 모델 기본값에 맡긴다

    t_grid = sorted(set(args.t_grid))
    final_grid = sorted(set(args.final_t_grid) | set(t_grid))
    # 커버리지 요건은 **곡선에 그릴 가장 조인 점**에서 온다. 루프 격자가 아니다 —
    # 배포할 프롬프트는 최종 곡선의 모든 점을 지탱해야 하고, 루프가 그보다 느슨한
    # 요건으로 학습하면 마지막 평가에서만 무너진다 (run03: test 1차 통과율 0.34).
    # 검증기(`evaluate`)와 프롬프트 문면(`output_rules`)이 **같은 값**을 써야 한다.
    coverage_t = min(final_grid)
    # 후보 풀 하한. 기본은 곡선의 가장 조인 점이지만, 그 값이면 **후보 수 ≈ 채택 수** 가 되어
    # 순위 절단이 고를 게 없다 (en-de run01: 마킹 7.2 / 채택 6.2, 폐기 1개).
    # 더 작은 값을 주면 더 많이 찍게 강제된다 — 실측에서 밀도 0.348 → 0.529, T=6 품질
    # +0.013 로 **오늘까지 확인된 유일한 품질 레버**다 (docs/RANK_METRIC_DIAGNOSIS.md §8.1).
    # 문면(`initial_prompt`)과 검증기(`need`)가 **같은 값**을 써야 한다 — 어긋나면 전 문장이
    # too_few_tags 로 재시도돼 비용이 두 배가 되고 1차 통과율 신호가 오염된다.
    candidate_t = args.candidate_t or coverage_t
    main_t = args.main_t or t_grid[len(t_grid) // 2]
    if main_t not in t_grid:
        print(f"--main-t {main_t} 가 --t-grid {t_grid} 에 없습니다", file=sys.stderr)
        return 2

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    pair_id = args.pair_id or f"{args.src_lang}-{args.tgt_lang}".lower().replace(" ", "_")
    run_dir = _HERE.parent / "runs" / pair_id / run_id
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gw = Gateway(model=args.model, budget=args.budget,
                 reasoning_effort=args.agent_reasoning_effort)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")
    translator = None
    judge_gw = None

    adequacy = metrics.make_adequacy_backend(
        args.adequacy_backend, batch_size=args.comet_batch_size)
    # consistency 의 nli 모델은 contradiction 백엔드를 따른다 — 둘 다 (합본, full)
    # 타깃 언어 쌍을 재므로 언어 선택 기준이 같다 (타깃이 영어가 아니면 mdeberta-xnli).
    if args.consistency_backend == "nli":
        consistency = metrics.make_backend(
            "nli", model_name=metrics.NLI_MODELS[args.contradiction_backend],
            batch_size=args.comet_batch_size)
    else:
        consistency = metrics.make_backend(
            args.consistency_backend, gw=gw,
            **({"batch_size": args.comet_batch_size}
               if args.consistency_backend in metrics.COMET_CHECKPOINTS else {}))
    contradiction = (None if args.no_contradiction else
                     metrics.make_contradiction_backend(args.contradiction_backend))

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        # ── A0 데이터 ────────────────────────────────────────────────────
        sentences = data.LOADERS[args.dataset]()
        pool_n = max(args.train, args.train_pool or args.train)
        splits = data.split_data(sentences, pool_n, args.dev, args.test,
                                 min_chars=args.min_chars)
        data.write_splits(splits, run_dir / "data")
        log(f"[data] {args.dataset}: 전체 {len(sentences)}, "
            f"train {len(splits['train'])} / dev {len(splits['dev'])} / test {len(splits['test'])}")

        # 측정 프로파일 — 결정론적 코드를 움직이는 필드는 코퍼스에서 직접 잰다
        measured = data.measure_profile([s.text for s in
                                         splits["train"] + splits["dev"] + splits["test"]])
        (run_dir / "measured_profile.json").write_text(
            json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── A1 Language Profiler ────────────────────────────────────────
        profiler = agents.Profiler(gw)
        profile_path = run_dir / "language_profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            samples = [s.text for s in splits["train"][:20]]
            profile = profiler.profile(samples, args.tgt_lang)
            profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

        # **JSON 은 고치지 않는다.** 고치면 prompt_v0 가 달라져 기존 런과 비교가 깨진다.
        # 덮어쓰기는 소비 지점인 여기서만 한다.
        spaced, trailing_punct, warns = data.reconcile_profile(measured, profile)
        for w in warns:
            log(f"[profile] 경고: {w}")
        tgt_spaced = (target_is_spaced(args.tgt_lang) if args.tgt_spaced is None
                      else args.tgt_spaced == "yes")
        log(f"[profile] {profile.get('source_language')} / 어순 {profile.get('word_order')} / "
            f"띄어쓰기 {spaced}(측정) / 타깃 띄어쓰기 {tgt_spaced}")

        if args.translator == "google":
            translator = GoogleTranslator(
                tgt_code=args.tgt_code or to_lang_code(args.tgt_lang),
                cache=tr_cache, workers=min(args.workers, 4),
                use_context=not args.no_google_context)
            translator_id = f"google:{translator.tgt_code}:ctx={translator.use_context}"
        else:
            translator = Translator(gw=gw, src_name=args.src_lang, tgt_name=args.tgt_lang,
                                    model=args.translator_model, cache=tr_cache,
                                    workers=args.workers)
            translator_id = f"llm:{args.translator_model}"
        log(f"[translator] {translator_id}")

        (run_dir / "config.json").write_text(json.dumps({
            **vars(args), "t_grid": t_grid, "final_t_grid": final_grid, "main_t": main_t,
            "translator_id": translator_id, "tgt_spaced": tgt_spaced,
            "adequacy_model": metrics.QE_CHECKPOINTS[args.adequacy_backend],
            "consistency_model": getattr(consistency, "model_name",
                                         args.consistency_backend),
            "judge_prompt_hash": JsonCache.key(agents.JUDGE_SYSTEM),
            "judge_model": args.judge_model or args.model,
            "min_boundaries_per": candidate_t,   # [Output Rules] 에 박히는 값 = 검증기 요건
            "candidate_t": candidate_t,
            "curve_min_t": coverage_t,
            "coverage_required": not args.no_coverage_rule,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        def run_eval(prompt_: str, split_sentences, grid):
            return evaluate(gw, translator, prompt_, split_sentences, spaced, seg_cache,
                            args.workers, adequacy, consistency, grid, trailing_punct,
                            tgt_spaced=tgt_spaced, contradiction=contradiction,
                            require_coverage=not args.no_coverage_rule,
                            coverage_t=candidate_t,
                            reasoning_effort=args.seg_reasoning_effort,
                            skip_translation_below=args.skip_translation_below,
                            batch_size=args.batch_size,
                            min_gap=args.min_gap)

        def prewarm(prompts: list[str], sentences) -> None:
            """여러 프롬프트의 분절을 **한 풀에** 미리 돌려 캐시에 넣는다.

            **검증·재시도를 반드시 함께 건다.** `segment_batch` 는 캐시가 맞으면 검증
            없이 즉시 반환하므로, 검증 안 한 출력을 캐시에 넣으면 뒤따르는 `run_eval`
            이 재시도를 영영 못 한다 — 복구됐어야 할 문장이 안 고쳐지고
            `format_pass_rate_no_retry` 는 거짓으로 1.00 이 된다. 캐시에 들어가는 값은
            정상 경로와 **바이트 단위로 같아야** 한다.
            """
            if len(prompts) <= 1:
                return
            texts = [x.text for x in sentences]

            def one(pr: str):
                # 프롬프트 **안**의 병렬(=콜 수)만으로는 워커를 못 채운다. 프롬프트
                # **사이**에도 병렬을 걸어야 동시 폭이 후보 수만큼 곱해진다.
                min_t = candidate_t
                need = (lambda t: max(0, chunk_budget(t, min_t, spaced) - 1)
                        ) if not args.no_coverage_rule else (lambda t: None)
                segment_batch(
                    gw, pr, texts, cache=seg_cache, workers=args.workers,
                    validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct,
                                                        True, need(t)),
                    normalize_fn=lambda o: normalize_tags(o, spaced, trailing_punct),
                    reasoning_effort=args.seg_reasoning_effort,
                    batch_size=args.batch_size)

            errs = []
            with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
                futs = [ex.submit(one, pr) for pr in prompts]
                for f in futs:
                    try:
                        f.result()
                    except BudgetExceeded:
                        errs.append("budget")
                    except Exception as e:                   # 예열 실패는 치명적이지 않다
                        errs.append(str(e)[:80])
            if "budget" in errs:
                raise BudgetExceeded("예열 중 예산 초과")
            if errs:
                log(f"[prewarm] 일부 실패(무시): {errs[:2]}")
            log(f"[prewarm] 후보 {len(prompts)} × 문장 {len(texts)} 예열 완료 "
                f"(동시 최대 {len(prompts) * args.workers})")

        def select_prompt(cands: list[str], probe_n: int, tag_: str) -> str:
            """후보 여러 개 중 하나를 고른다. **2단계** — probe 로 거르고 상위 2개만 dev 전체.

            probe 하나로 고르면 못 믿는다 (test 100문장·변종 6종 실측): probe 20 은
            1위 적중 36%, 40 은 58%, dev 전체(60)라야 76% 다. 상위 2개 포함률은 probe 40
            에서 77% 이므로, 40 으로 두 개까지 좁힌 뒤 그 둘만 전체로 재는 것이 비용 대비
            가장 낫다.

            **쌍체로 바꿔도 순위는 안 바뀐다** — 후보들이 같은 문장 집합을 쓰면 기준값이
            공통 상수라 argmax 에서 소거된다 (실측 소수점까지 동일). 쌍체의 이득은
            오차막대이지 순위가 아니다.
            """
            if len(cands) <= 1:
                return cands[0] if cands else ""
            probe = splits["dev"][:probe_n]
            # **분절을 먼저 한 풀에 몰아 캐시를 채운다.** 후보를 순차로 `run_eval` 하면
            # 후보마다 probe/batch_size 개(=40/6≈7)의 콜만 던지게 되어 워커를 못 채운다
            # — run04 실측 평균 동시 실행 3.03 / 최대 7 (워커 8). 분절 1콜이 112초라
            # 그 직렬화가 곧 경과 시간이다. 후보 전체의 분절을 한 번에 던지면 동시 폭이
            # 후보 수만큼 늘고, 이후 `run_eval` 은 전부 캐시 히트로 지나간다.
            prewarm(cands, probe)
            scored = []
            for i, c in enumerate(cands):
                _r, _m, _v = run_eval(c, probe, t_grid)
                sc_i = metrics.score(_m)
                scored.append((sc_i, i, c))
                log(f"[{tag_}] 후보 {i}: {len(c)}자 probe({len(probe)}) "
                    f"score={sc_i:.4f} fmt={_m.format_pass_rate:.2f}")
            scored.sort(key=lambda x: -x[0])
            finals = scored[:2]
            if len(splits["dev"]) > probe_n:
                prewarm([c for _s, _i, c in finals], splits["dev"])
                re_scored = []
                for sc_i, i, c in finals:
                    _r, _m, _v = run_eval(c, splits["dev"], t_grid)
                    s2 = metrics.score(_m)
                    re_scored.append((s2, i, c))
                    log(f"[{tag_}] 결선 후보 {i}: dev({len(splits['dev'])}) score={s2:.4f}")
                re_scored.sort(key=lambda x: -x[0])
                finals = re_scored
            log(f"[{tag_}] 후보 {finals[0][1]} 채택 (score={finals[0][0]:.4f})")
            return finals[0][2]

        prompt_path = run_dir / "iter_00" / "prompt.txt"
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
            missing = agents.check_skeleton(prompt)
            if missing:
                log(f"[stop] 재개한 prompt_v0 골격 누락 {missing} — --fresh 로 다시 만들 것")
                return 2
        else:
            # 재시도본을 재검사하지 않으면 잘린 프롬프트가 그대로 통과한다 — run04 에서
            # 1차·재시도 연속 잘림(꼬리 섹션 누락)이 실제로 났다. 결함 prompt_v0 로
            # 돌면 PE 개정본이 골격 검사에 계속 걸려 개정이 전부 거부되므로,
            # 복구 불가면 예산을 태우기 전에 죽는 것이 맞다.
            # **v0 를 여러 개 뽑아 고른다.** 분절기 temperature 를 0 으로 못 박을 수 없어
            # (설계 §8.6-5) v0 품질 분산이 크다 — 같은 데이터·같은 생성기인데 en-de
            # run01 6,968자 / run02 13,198자로 2배 갈렸다. 게다가 루프가 v0 를 못 이기는
            # 일이 잦아(run01 2회, run02 3회 모두 iter_00 채택) **런 결과가 v0 뽑기에
            # 좌우된다.** 후보를 만들어 dev 일부로 고르면 그 분산을 산다.
            candidates: list[str] = []
            for attempt in range(3 * max(1, args.v0_candidates)):
                if len(candidates) >= args.v0_candidates:
                    break
                cand = profiler.initial_prompt(profile, args.tgt_lang, spaced, candidate_t)
                missing = agents.check_skeleton(cand)
                if missing:
                    log(f"[profiler] 골격 누락 {missing} — 재시도 {attempt + 1}")
                    continue
                candidates.append(cand)
            if not candidates:
                log(f"[stop] prompt_v0 가 골격 미달만 반복 — 모델/max_tokens 확인")
                return 2

            for k, cand in enumerate(candidates):
                (run_dir / f"prompt_v0_cand{k}.txt").write_text(cand, encoding="utf-8")
            prompt = select_prompt(candidates, args.v0_probe, "profiler")
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
        prompt_v0_len = len(prompt)
        log(f"[profiler] prompt_v0 작성 완료 ({prompt_v0_len}자)")

        # ── 루프 ────────────────────────────────────────────────────────
        critic = agents.Critic(gw)
        engineer = agents.PromptEngineer(gw)
        compressor = agents.Compressor(gw)
        # 판정자를 다른 모델로 쓰면 별도 Gateway 가 생긴다. 그 사용량이 비용 보고와
        # 예산 가드에서 빠지면 안 되므로 참조를 들고 합산한다.
        if not args.no_judge and args.judge_model:
            judge_gw = Gateway(model=args.judge_model, budget=args.budget,
                               reasoning_effort=args.agent_reasoning_effort)
        judge = None if args.no_judge else agents.Judge(judge_gw or gw)

        def usage_total() -> dict:
            u = dict(gw.usage.snapshot())
            if judge_gw is not None:
                ju = judge_gw.usage.snapshot()
                for k, v in ju.items():
                    if isinstance(v, (int, float)):
                        u[k] = u.get(k, 0) + v
                # 판정자가 별도 게이트웨이면 용도별 집계도 합쳐야 비용 표가 안 비뚤어진다.
                merged = {k: dict(v) for k, v in u.get("by_purpose", {}).items()}
                for k, v in ju.get("by_purpose", {}).items():
                    tgt = merged.setdefault(k, {kk: 0 for kk in v})
                    for kk, vv in v.items():
                        tgt[kk] = tgt.get(kk, 0) + vv
                u["by_purpose"] = merged
            return u

        history: list[dict] = []
        best = {"version": 0, "prompt": prompt, "train_score": None, "dev_score": None}
        best_ctx: dict = {}
        best_critique: dict | None = None
        last_focus: str | None = None
        stale = 0
        floor_fn = None          # contradiction 잡음 바닥 — 첫 평가 후 1회 측정

        # `--final-only` — 이터레이션을 건너뛰고 기존 `best_prompt.txt` 로 최종 test 평가만
        # 다시 돈다. 루프는 끝났는데 마지막 단계에서 죽는 경우가 실제로 있었고(run05,
        # SIGTERM), 그때 루프를 재실행하면 history 가 빈 리스트로 시작해 iter_00 부터
        # 덮어쓴다 — PE 호출이 비결정론적이라 같은 프롬프트 열이 나오지도 않는다.
        # 이터레이션 산출물은 그대로 두고 test/curve/report 만 만들어내는 경로가 필요하다.
        if args.final_only:
            best_path = run_dir / "best_prompt.txt"
            hist_path = run_dir / "history.json"
            if not best_path.exists():
                log(f"[stop] --final-only 인데 {best_path} 가 없다")
                return 2
            best = {"version": 0, "prompt": best_path.read_text(encoding="utf-8"),
                    "train_score": None, "dev_score": None}
            if hist_path.exists():
                history = json.loads(hist_path.read_text(encoding="utf-8"))
                adopted = [h for h in history if h.get("adopted")]
                if adopted:
                    best["version"] = adopted[-1]["version"]
                    best["train_score"] = adopted[-1].get("score_train")
                    best["dev_score"] = adopted[-1].get("score_dev")
            log(f"[final-only] iter_{best['version']:02d} 의 best_prompt "
                f"({len(best['prompt'])}자) 로 최종 평가만 수행한다")
            args.iterations = 0

        for it in range(args.iterations):
            it_dir = run_dir / f"iter_{it:02d}"
            it_dir.mkdir(parents=True, exist_ok=True)
            (it_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            batch = splits["train"]
            if len(batch) > args.train:
                batch = random.Random(20260808 + it).sample(batch, args.train)
            rows, m, viol = run_eval(prompt, batch, t_grid)
            sc = metrics.score(m)

            # ── A6′ Judge — 주 작동점에서만 (비용) ────────────────────────
            judgements: list[dict] = []
            if judge is not None and m.by_T:
                judgements = agents.judge_rows(judge, rows, main_t, args.judge_rows)
                pr = agents.premature_rate(judgements)
                if pr is not None and str(main_t) in m.by_T:
                    m.by_T[str(main_t)].premature_rate = round(pr, 4)
                rs = agents.reference_suspect_rate(judgements)
                if rs is not None and str(main_t) in m.by_T:
                    m.by_T[str(main_t)].reference_suspect_rate = round(rs, 4)
                (it_dir / "judgements.json").write_text(
                    json.dumps(judgements, ensure_ascii=False, indent=2), encoding="utf-8")

            # 순위 진단 — 모델이 단 순위가 실측 위험(경계 contradiction)과 맞는가.
            audit: list[dict] = []      # by_T 가 비면(포맷 붕괴) 아래 블록을 안 타므로 선초기화
            # 경계가 가장 많이 살아남는 최소 T 에서 잰다. 두 통계량을 함께 낸다:
            #   Spearman   순위 상관 (방향만)
            #   gap        하위 절반 − 상위 절반의 위험 차 (크기). focus 판정이 쓰는 값.
            # gap 이 0 이하면 절단이 위험을 못 덜어낸다 = [Priority Rules] 문제.
            if m.by_T:
                low_t = min(t_grid)
                if floor_fn is None:
                    floor_fn = load_contra_floor(run_dir, rows, contradiction,
                                                 tgt_spaced=tgt_spaced)
                sp_corr, sp_n = metrics.rank_contra_spearman(rows, low_t)
                gaps_i = metrics.rank_contra_gaps(rows, low_t, floor_fn=floor_fn,
                                                  tgt_spaced=tgt_spaced)
                gap = sum(gaps_i) / len(gaps_i) if gaps_i else None
                gap_n = len(gaps_i)
                gap_se = (statistics.stdev(gaps_i) / len(gaps_i) ** 0.5
                          if len(gaps_i) > 1 else None)
                if str(low_t) in m.by_T:
                    if sp_corr is not None:
                        m.by_T[str(low_t)].rank_contra_spearman = round(sp_corr, 4)
                    if gap is not None:
                        m.by_T[str(low_t)].rank_contra_gap = round(gap, 4)
                        m.by_T[str(low_t)].rank_contra_gap_n = gap_n
                        if gap_se is not None:
                            m.by_T[str(low_t)].rank_contra_gap_se = round(gap_se, 4)
                # 순위 진단이 음수여도 **어느 특징이 과신되는지**는 gap 이 안 알려준다.
                # 그 조준 정보를 critique 에 실어 PE 가 [Priority Rules] 를 눈감고
                # 재작성하지 않게 한다 (metrics.priority_audit).
                audit = metrics.priority_audit(rows, low_t, floor_fn=floor_fn,
                                               tgt_spaced=tgt_spaced)
                if audit:
                    (it_dir / "priority_audit.json").write_text(
                        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
                    top = audit[0]
                    log(f"[iter {it}] 순위감사 최다 과신 '{top['feature']}' "
                        f"백분위 {top['rank_percentile']:.2f} contra {top['contradiction']:.4f} "
                        f"(n={top['n']})")
                if sp_corr is not None or gap is not None:
                    log(f"[iter {it}] 순위진단(T={low_t}) "
                        f"gap={_cell(gap, '+.4f')}±{_cell(gap_se, '.4f')}(n={gap_n}) "
                        f"Spearman={_cell(sp_corr, '+.3f')}(n={sp_n})"
                        + ("" if floor_fn else "  [바닥 보정 없음 — 음수 편향]"))

            (it_dir / "train_rows.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            # 1차(재시도 전) 위반은 **따로** 남긴다. 섞으면 Critic 이 이미 복구된 것을
            # 현재 결함으로 읽고, `violations.json` 이 최종 상태라는 의미도 잃는다.
            viol, viol_1st = split_first_pass(viol)
            (it_dir / "violations.json").write_text(
                json.dumps(viol, ensure_ascii=False, indent=2), encoding="utf-8")
            if viol_1st:
                (it_dir / "violations_first_pass.json").write_text(
                    json.dumps(viol_1st, ensure_ascii=False, indent=2), encoding="utf-8")
                import collections as _c
                log(f"[iter {it}] 1차 위반 {len(viol_1st)}건 "
                    f"{dict(_c.Counter(v['rule'] for v in viol_1st))}")
            log(f"[iter {it}] {fmt_metrics(m, 'train')}"
                + (f"  premature={m.by_T[str(main_t)].premature_rate}"
                   if m.by_T.get(str(main_t)) and m.by_T[str(main_t)].premature_rate is not None
                   else ""))

            # 쌍체 차이. dev 가 고정 집합이라 `mean(new) − mean(old) = mean(new − old)` 가
            # 항등이고 점추정은 절대값 비교와 동일하다. 얻는 것은 **오차막대와 유효 표본**:
            # 분절이 안 바뀐 문장은 차이가 정확히 0 이라 분산에 기여하지 않으므로
            # se 가 훨씬 작게 나오고, `n_changed` 가 실제로 판정에 참여한 문장 수를 알려준다.
            # 채택 판정이 이 오차막대를 쓴다 (`Δ > adopt_se_mult·se`) — run01~03 실측에서
            # 점 비교는 오차막대 안 잡음(예: −0.013±0.014)까지 채택 후보로 만들었다.
            train_delta = (metrics.paired_delta(rows, best_ctx["rows"], t_grid)
                           if best_ctx else None)
            if train_delta and train_delta["mean_delta"] is not None:
                log(f"[iter {it}] train Δ={train_delta['mean_delta']:+.5f} "
                    f"±{train_delta['se_delta']:.5f} (분절 변경 {train_delta['n_changed']}"
                    f"/{train_delta['n_pairs']}문장)")

            # dev 검증은 train 이 개선됐을 때만 (비용 절약). **판정은 쌍체로 한다** —
            # 절대 점수 비교는 문장 난이도 분산에 묻혀 (문장별 sd 0.05, 검출 목표 0.005)
            # 개선된 개정을 걸러낸다. en-de run02 iter1 실측: 절대 0.7844 < 0.7925 로
            # 막혔으나 쌍체는 +0.00929 로 개선 방향이었다 — 채택 판정(쌍체)을 받아볼
            # 기회 자체가 없었다. 게이트와 채택이 서로 다른 통계를 쓰던 모순을 없앤다.
            # 게이트는 점추정(> 0)만 보고, 오차막대 요구는 dev 채택 판정이 계속 맡는다.
            dev_m = None
            dev_score = None
            dev_delta = None
            dev_rows = None
            if best["train_score"] is None:
                run_dev = True
            elif train_delta and train_delta.get("mean_delta") is not None:
                run_dev = train_delta["mean_delta"] > 0
            else:
                run_dev = sc > best["train_score"]      # 쌍체를 못 잰 경우만 후퇴
            if run_dev:
                dev_rows, dev_m, dev_viol = run_eval(prompt, splits["dev"], t_grid)
                dev_score = metrics.score(dev_m)
                dev_delta = (metrics.paired_delta(dev_rows, best_ctx["dev_rows"], t_grid)
                             if best_ctx.get("dev_rows") else None)
                (it_dir / "dev_rows.json").write_text(
                    json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
                dev_viol, dev_viol_1st = split_first_pass(dev_viol)
                (it_dir / "dev_violations.json").write_text(
                    json.dumps(dev_viol, ensure_ascii=False, indent=2), encoding="utf-8")
                if dev_viol_1st:
                    (it_dir / "dev_violations_first_pass.json").write_text(
                        json.dumps(dev_viol_1st, ensure_ascii=False, indent=2), encoding="utf-8")
                viol = viol + dev_viol
                log(f"[iter {it}] {fmt_metrics(dev_m, 'dev  ')}"
                    + (f"  Δ={dev_delta['mean_delta']:+.5f} ±{dev_delta['se_delta']:.5f} "
                       f"(변경 {dev_delta['n_changed']}/{dev_delta['n_pairs']})"
                       if dev_delta and dev_delta["mean_delta"] is not None else ""))

            # ── 채택 판정 — 첫 후보는 무조건, 이후는 쌍체 Δ 가 오차막대를 넘을 때만.
            # 점추정 비교(> 만)는 노이즈로 오른 개정을 채택한다. 쌍체 se 가 있으면
            # Δ > adopt_se_mult·se 를 요구하고, 없을 때만 점 비교로 후퇴한다.
            adopted = False
            if dev_score is not None:
                if best["dev_score"] is None:
                    adopted = True
                elif dev_delta and dev_delta.get("mean_delta") is not None:
                    adopted = (dev_delta["mean_delta"]
                               > args.adopt_se_mult * (dev_delta.get("se_delta") or 0.0))
                else:
                    adopted = dev_score > best["dev_score"]
            if adopted:
                best = {"version": it, "prompt": prompt,
                        "train_score": sc, "dev_score": dev_score}
                (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
                best_ctx = {"rows": rows, "dev_rows": dev_rows, "metrics": m.to_dict(),
                            "violations": viol, "judgements": judgements,
                            "priority_audit": audit}
                best_critique = None      # 새 best — 비평을 다시 받아야 한다
                adopted = True
                stale = 0
            else:
                stale += 1

            (it_dir / "metrics.json").write_text(json.dumps({
                "train": m.to_dict(),
                "dev": dev_m.to_dict() if dev_m else None,
                "score_train": round(sc, 4),
                "score_dev": round(dev_score, 4) if dev_score is not None else None,
                "paired_train": train_delta, "paired_dev": dev_delta,
                "adopted": adopted,
                "usage": usage_total(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            history.append({
                "version": it, "adopted": adopted,
                "train": m.to_dict(), "dev": dev_m.to_dict() if dev_m else None,
                "score_train": round(sc, 4),
                "score_dev": round(dev_score, 4) if dev_score is not None else None,
                "paired_dev": dev_delta,
                "changelog": history[-1].get("next_changelog") if history else None,
            })
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

            u = usage_total()
            log(f"[iter {it}] adopted={adopted} stale={stale} "
                f"calls={u['calls']} cost={u['cost']:.4f}")
            log(f"[iter {it}] 비용내역 {fmt_by_purpose(u)}")

            if it == args.iterations - 1:
                break
            if stale >= args.patience:
                log(f"[stop] dev 개선 없음 {stale}회 연속 — 조기 종료")
                break

            # 에이전트 호출이 실패해도 런 전체를 버리지 않는다.
            ctx = best_ctx or {"rows": rows, "metrics": m.to_dict(),
                               "violations": viol, "judgements": judgements}
            try:
                # ── A6 Critic ───────────────────────────────────────────
                # 비평 대상과 개정 대상은 반드시 같은 프롬프트여야 한다.
                if best_critique is None:
                    cases = agents.select_cases(ctx["rows"], main_t, ctx.get("judgements"))
                    best_critique = critic.review(
                        cases, ctx["metrics"], ctx["violations"],
                        avoid=last_focus if stale >= 2 else None,
                        priority_audit=ctx.get("priority_audit"))
                critique = best_critique
                # 캐시가 고착 방지를 우회하지 않도록 aggregate 만 다시 계산한다 (LLM 없음).
                if stale >= 2 and last_focus:
                    critique = {**critique, "aggregate": agents.summarize_critique(
                        critique.get("cases") or [], ctx["metrics"],
                        critique.get("summary"), avoid=last_focus,
                        priority_audit=ctx.get("priority_audit"))}
                (it_dir / "critique.json").write_text(
                    json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")
                agg = critique.get("aggregate", {})
                last_focus = agg.get("focus")
                log(f"[iter {it}] critic dominant={agg.get('dominant_error')} "
                    f"focus={last_focus}")

                # ── A7 Prompt Engineer ──────────────────────────────────
                # **제안 1개를 검증 1개로 받던 구조를 K개 생성 -> 선택으로 바꾼다.**
                # 오늘 실측에서 dev 까지 간 개정 3건이 전부 음수였다(t = −0.8 ~ −3.2) —
                # 채택 문턱이 아니라 개정 품질이 원인이다. LLM 은 좋은 개정을 확실히
                # 만들지는 못해도 여러 개 중엔 하나쯤 낸다. 후보는 Critic 이 낸
                # `proposed_rule` 을 **하나씩만** 반영해 만든다 (신용 배분 + 국소성).
                rules = [c.get("proposed_rule") for c in (critique.get("cases") or [])
                         if c.get("proposed_rule")]
                seen, hints = set(), []
                for r in rules:
                    k = r[:80]
                    if k not in seen:
                        seen.add(k); hints.append(r)
                hints = hints[:max(0, args.revision_candidates - 1)] or []
                jobs = [None] + hints            # None = 종전 방식(자유 개정) 1개
                jobs = jobs[:max(1, args.revision_candidates)]

                def make(hint):
                    try:
                        rv = engineer.revise(best["prompt"], critique, history, profile,
                                             t_grid, only_rule=hint)
                    except BudgetExceeded:
                        raise
                    except Exception as e:                      # 후보 하나 실패로 안 죽는다
                        log(f"[iter {it}] 개정 후보 실패: {e}")
                        return None
                    return rv

                cands = []
                for hint in jobs:
                    rv = make(hint)
                    if not rv: continue
                    pr = rv.get("prompt", "")
                    if not pr or agents.check_skeleton(pr): continue
                    if agents.check_revision(best["prompt"], pr, last_focus): continue
                    cands.append((pr, rv))
                if len(cands) > 1:
                    pick = select_prompt([c[0] for c in cands], args.v0_probe, f"iter {it} 개정")
                    revised = next(rv for pr, rv in cands if pr == pick)
                elif cands:
                    revised = cands[0][1]
                else:
                    # 후보가 전멸하면 종전 경로로 한 번 더 — 게이트 사유를 로그에 남긴다.
                    revised = engineer.revise(best["prompt"], critique, history, profile, t_grid)
                log(f"[iter {it}] 개정 후보 {len(cands)}/{len(jobs)} 통과")
                new_prompt = revised.get("prompt", "")
                budget = int(prompt_v0_len * args.max_prompt_growth)
                if new_prompt and len(new_prompt) > budget:
                    log(f"[iter {it}] 개정본 {len(new_prompt)}자 > 예산 {budget}자 — 압축")
                    packed = compressor.compress(
                        new_prompt, budget, revised.get("sections_changed") or [])
                    if (packed and not agents.check_skeleton(packed)
                            and len(packed) <= budget):
                        log(f"[iter {it}] 압축 성공 {len(new_prompt)} -> {len(packed)}자")
                        new_prompt = packed
                    else:
                        log(f"[iter {it}] 압축 실패({len(packed or '')}자) — 개정 거부")
                        new_prompt = ""
                missing = agents.check_skeleton(new_prompt) if new_prompt else []
                # 국소성 하드 게이트 — 지시문 권고만으로는 안 지켜졌다 (agents.check_revision).
                scope = (agents.check_revision(best["prompt"], new_prompt, last_focus)
                         if new_prompt and not missing else [])
                if not new_prompt:
                    log(f"[iter {it}] 개정 없음 — 이전 프롬프트 유지")
                elif missing or len(new_prompt) < 500:
                    log(f"[iter {it}] 개정 프롬프트 골격 누락 {missing} — 이전 프롬프트 유지")
                elif scope:
                    log(f"[iter {it}] 개정 범위 위반 — 거부: {'; '.join(scope)}")
                else:
                    prompt = new_prompt
                (it_dir / "changelog.json").write_text(json.dumps({
                    "sections_changed": revised.get("sections_changed"),
                    "changelog": revised.get("changelog"),
                    "scope_violations": scope,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                history[-1]["next_changelog"] = revised.get("changelog")
                log(f"[iter {it}] 개정: {revised.get('sections_changed')}")
            except BudgetExceeded:
                raise
            except Exception as e:
                log(f"[iter {it}] 에이전트 실패 — 루프 중단하고 최종 평가로 넘어간다: {e}")
                break

        # ── 최종 test 평가 (전체 격자) ───────────────────────────────────
        log(f"[final] test 평가 — 격자 {final_grid} (루프가 한 번도 보지 않은 데이터)")
        test_rows, test_m, test_viol = run_eval(best["prompt"], splits["test"], final_grid)
        test_viol, test_viol_1st = split_first_pass(test_viol)
        if test_viol_1st:
            (run_dir / "test_violations_first_pass.json").write_text(
                json.dumps(test_viol_1st, ensure_ascii=False, indent=2), encoding="utf-8")
        if judge is not None and test_m.by_T:
            # 보고용 test 판정은 **무작위 표본**이다. 루프 중에는 실패 조준 표본
            # (Critic 입력용)이 맞지만, 그 표본으로 잰 premature_rate 는 조건부 상향
            # 추정치라 리포트 수치로 못 쓴다 (run03 의 0.2727 이 그 값).
            tj = agents.judge_rows(judge, test_rows, main_t, args.judge_rows * 2,
                                   sample="random")
            pr = agents.premature_rate(tj)
            if pr is not None and str(main_t) in test_m.by_T:
                test_m.by_T[str(main_t)].premature_rate = round(pr, 4)
            rs = agents.reference_suspect_rate(tj)
            if rs is not None and str(main_t) in test_m.by_T:
                test_m.by_T[str(main_t)].reference_suspect_rate = round(rs, 4)
            (run_dir / "test_judgements.json").write_text(
                json.dumps(tj, ensure_ascii=False, indent=2), encoding="utf-8")
        if test_m.by_T:
            low_t = min(final_grid)
            if floor_fn is None:
                floor_fn = load_contra_floor(run_dir, test_rows, contradiction,
                                            tgt_spaced=tgt_spaced)
            sp_corr, sp_n = metrics.rank_contra_spearman(test_rows, low_t)
            gap, gap_n = metrics.rank_contra_gap(test_rows, low_t, floor_fn=floor_fn,
                                                tgt_spaced=tgt_spaced)
            if str(low_t) in test_m.by_T:
                if sp_corr is not None:
                    test_m.by_T[str(low_t)].rank_contra_spearman = round(sp_corr, 4)
                if gap is not None:
                    test_m.by_T[str(low_t)].rank_contra_gap = round(gap, 4)
            log(f"[final] 순위진단(T={low_t}) gap={_cell(gap, '+.4f')}(n={gap_n}) "
                f"Spearman={_cell(sp_corr, '+.3f')}(n={sp_n})")
        (run_dir / "test_rows.json").write_text(
            json.dumps(test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── 비교군 (노브 없음 — 각 1점) ──────────────────────────────────
        baselines = compare_baselines(translator, adequacy, consistency, splits["test"],
                                      spaced, tgt_spaced, args.workers, contradiction=contradiction)
        curve = {
            "ours": {k: v.to_dict() for k, v in test_m.by_T.items()},
            "baselines": baselines,
            "axes": {"x": "laal_words (source words, lower = faster)",
                     "y": "consistency (bidirectional NLI vs offline translation; "
                          "unsegmented = 1.0 is the axis' own reference point)",
                     "aux": "adequacy / contradiction (2-panel), effective (loop objective)"},
        }
        (run_dir / "curve.json").write_text(
            json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")

        if isinstance(translator, GoogleTranslator) and translator.context_line_mismatches:
            log(f"[경고] gtx 컨텍스트 번역에서 줄 수 불일치 {translator.context_line_mismatches}건 — "
                f"해당 조각 번역이 오염됐을 수 있다 (마지막 줄 추출 실패)")

        report = build_report(args, run_dir, profile, measured, history, best, test_m,
                              test_viol, usage_total(), baselines, t_grid,
                              final_grid, main_t, translator_id)
        (run_dir / "final_report.md").write_text(report, encoding="utf-8")
        log(f"\n{report}")
        log(f"[done] {run_dir}")
        return 0

    except BudgetExceeded as e:
        log(f"[stop] {e}")
        return 2
    finally:
        seg_cache.flush()
        tr_cache.flush()
        if isinstance(translator, GoogleTranslator):
            translator.close()
        if judge_gw is not None:
            judge_gw.close()
        gw.close()


# ── 비교군 ───────────────────────────────────────────────────────────────

def compare_baselines(translator, adequacy, consistency, sentences, spaced,
                      tgt_spaced, workers, mech_every: int = 8,
                      contradiction=None) -> dict:
    """무분절 / 기계 8자분절. 순위가 없어 절단이 불가능하므로 곡선 위의 점 하나씩이다."""
    texts = [s.text for s in sentences]
    full = translator.full(texts)
    out: dict[str, dict] = {}
    for name, segs in (("unsegmented", list(texts)),
                       ("mechanical_8", [metrics.mechanical_split(t, mech_every, spaced)
                                         for t in texts])):
        sp = score_split(segs, texts, full, translator, adequacy, consistency,
                         spaced, tgt_spaced, contradiction)
        out[name] = metrics.aggregate_split(
            0, sp.effective, sp.adequacy, sp.contradiction, sp.consistency,
            sp.chrf, sp.laal_words, sp.k, [0] * len(texts)).to_dict()
    return out


def split_first_pass(viol: list[dict]) -> tuple[list[dict], list[dict]]:
    """재시도 후 최종 위반과 1차(재시도 전) 위반을 가른다."""
    final = [v for v in viol if not v.get("first_pass")]
    first = [v for v in viol if v.get("first_pass")]
    return final, first


def build_report(args, run_dir, profile, measured, history, best, test_m, test_viol,
                 usage, baselines, t_grid, final_grid, main_t, translator_id) -> str:
    lines = [
        f"# 자동 분절 프롬프트 루프 결과 (v2) — {args.src_lang} → {args.tgt_lang}",
        "",
        f"- 데이터셋: `{args.dataset}` (train {args.train} / dev {args.dev} / test {args.test})",
        f"- 분절 모델 `{args.model}` / 판정자 `{args.judge_model or args.model}` / "
        f"번역기 `{translator_id}`",
        f"- adequacy 백엔드: **{args.adequacy_backend}** "
        f"(`{metrics.QE_CHECKPOINTS[args.adequacy_backend]}`, 참조 없음)",
        f"- consistency 백엔드: {args.consistency_backend} (보고용"
        + (", 양방향 entailment 의 min — 어순 무관" if args.consistency_backend == "nli" else "")
        + ")",
        f"- 노브: 목표 조각 크기 T. 루프 격자 {t_grid}, 최종 격자 {final_grid}, 주 작동점 T={main_t}",
        f"- 언어 프로파일: {profile.get('source_language')}, 어순 {profile.get('word_order')} / "
        f"측정: 공백비율 {measured['space_ratio']}, 문말 부호 {measured['trailing_punctuation']}",
        f"- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` "
        f"(가중치·임계값 없음). 채택 판정은 **쌍체 비교**",
        f"- contradiction 백엔드: "
        + ("없음 (조기 방출이 벌받지 않음)" if args.no_contradiction
           else f"**{args.contradiction_backend}** (`{metrics.NLI_MODELS[args.contradiction_backend]}`)"),
        f"- 채택된 프롬프트: iter_{best['version']:02d}",
        "",
        "## 이터레이션 이력",
        "",
        "| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in history:
        t = h["train"]
        pd = h.get("paired_dev") or {}
        d = (f"{pd['mean_delta']:+.5f} ±{pd['se_delta']:.5f}"
             if pd.get("mean_delta") is not None else "—")
        lines.append(
            f"| {h['version']} | {t['format_pass_rate']:.2f} | {h['score_train']:.4f} | "
            + (f"{h['score_dev']:.4f}" if h.get("score_dev") is not None else "—")
            + f" | {d} | {pd.get('n_changed', '—')} | {'O' if h['adopted'] else 'X'} |")

    lines += [
        "",
        "## 최종 test 곡선",
        "",
        "| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(test_m.by_T, key=int):
        s = test_m.by_T[k]
        lines.append(f"| {k} | {s.laal_words:.2f} | **{_cell(s.effective, '.4f')}** | "
                     f"{s.adequacy:.4f} | {_cell(s.contradiction, '.4f')} | "
                     f"{s.consistency:.4f} | "
                     f"{s.chunks_per_sentence:.2f} | {s.missing_boundaries:.2f} |")
    for name, b in baselines.items():
        lines.append(f"| {name} (노브 없음) | {b['laal_words']:.2f} | "
                     f"**{_cell(b['effective'], '.4f')}** | "
                     f"{b['adequacy']:.4f} | {_cell(b['contradiction'], '.4f')} | "
                     f"{b['consistency']:.4f} | {b['chunks_per_sentence']:.2f} | — |")

    main_s = test_m.by_T.get(str(main_t))
    pr = main_s.premature_rate if main_s else None
    rs = main_s.reference_suspect_rate if main_s else None
    low_s = test_m.by_T.get(str(min(final_grid)))
    sp = low_s.rank_contra_spearman if low_s else None
    gap = low_s.rank_contra_gap if low_s else None
    lines += [
        "",
        f"- 포맷 통과율 {test_m.format_pass_rate:.4f} (재시도 없이 "
        f"{test_m.format_pass_rate_no_retry:.4f}), 위반 {len(test_viol)}건",
        f"- premature_rate (T={main_t}, 부록 지표, **무작위 표본**): "
        + (f"**{pr:.4f}**" if pr is not None else "미측정"),
        f"- reference_suspect_rate (T={main_t}): "
        + (f"{rs:.4f}" if rs is not None else "미측정")
        + " — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다",
        f"- **순위 격차 `rank_contra_gap` (T={min(final_grid)}, 바닥 보정)**: "
        + (f"**{gap:+.4f}**" if gap is not None else "미측정")
        + " — 순위 하위 절반 − 상위 절반의 경계 contradiction 차. "
        "양수 = 절단이 실제로 위험을 덜어냄. **0 이하면 순위가 정보를 주지 않는다** "
        "(기준점이 0 인 것은 순위 무정보 시 기대값이 정확히 0 이기 때문 — 임의 상수 아님)",
        f"- 순위정렬 Spearman (T={min(final_grid)}, raw): "
        + (f"{sp:+.4f}" if sp is not None else "미측정")
        + " — 같은 축의 방향만 보는 보조값. **바닥 보정이 없어 음수 쪽으로 편향**된다 "
        "(run03: raw −0.25 → 보정 후 +0.14). 판정은 위의 gap 으로 한다",
        "",
        "`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). "
        "`adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. "
        "`contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — "
        "무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.",
        "",
        "## 비용",
        "",
        f"- 호출 {usage['calls']}회, 입력 토큰 {usage['prompt_tokens']:,} "
        f"(캐시 {usage['cached_tokens']:,}), 출력 토큰 {usage['completion_tokens']:,}",
        f"- 게이트웨이 추정 비용 {usage['cost']:.4f}",
        "",
        "| 용도 | 호출 | 비용 | 비중 | 사고 토큰/콜 |",
        "|---|---|---|---|---|",
    ] + [
        f"| `{k}` | {v['calls']} | {v['cost']:.4f} | "
        f"{v['cost'] / max(1e-9, usage['cost']) * 100:.1f}% | "
        f"{v.get('reasoning_tokens', 0) // max(1, v['calls']):,} |"
        for k, v in (usage.get("by_purpose") or {}).items()
    ] + [
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
