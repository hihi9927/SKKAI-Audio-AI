"""A11 Loop Controller — 결정론적 오케스트레이터.

LLM 판단은 없다. 실행 순서, 채택/롤백, 예산, 중단 조건만 관리한다.
설계는 `AUTOSEG_SIMPLIFY.md`, 그 근거는 `AUTOSEG_DETAILS.md`.

  python -m core.meaning_segmentator.autoseg.loop \
      --dataset kspon --src-lang Korean --tgt-lang English \
      --iterations 6 --train 30 --dev 60 --test 100
"""

from __future__ import annotations

import argparse
import copy
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
from .pipeline import (GoogleTranslator, JsonCache, blocks_scoring,
                       coverage_need, normalize_tags, round_half_up, segment_batch,
                       shuffle_priorities, split_segments, to_lang_code, truncate,
                       unit_count, validate)

_HERE = Path(__file__).resolve().parent

# 타깃 언어의 표기 체계 — LAAL 의 목표측 토큰 수를 세는 단위를 정한다.
_UNSPACED_TARGETS = {"japanese", "chinese", "thai", "ja", "zh", "th"}


class _NoConsistency:
    """`rank_lift` 대조군 전용 — consistency 를 건너뛴다.

    `effective = adequacy × (1 − contradiction)` 이라 consistency 는 대조군 비교에
    쓰이지 않는다. 그런데 `score_split` 은 항상 계산하므로, 끄지 않으면 이터레이션마다
    NLI(또는 COMET) 한 벌이 순수 낭비로 돈다."""

    name = "none"

    def score(self, srcs, hyps, refs):
        return [0.0] * len(hyps)


_NO_CONSISTENCY = _NoConsistency()


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
    # 병기 지표 — 같은 NLI 호출에서 나온 `1 − entailment`. 목적함수에는 안 들어간다.
    pieces_contra_ent: list[list[float]]
    effective_ent: list[float | None]
    contradiction_ent: list[float | None]
    effective: list[float | None]
    adequacy: list[float]
    contradiction: list[float | None]
    consistency: list[float]
    chrf: list[float]
    laal_words: list[float]
    k: list[int]


# ── 다언어 목적함수 ─────────────────────────────────────────────────────────
#
# **분절은 타깃과 무관하다** — 문장당 1회이고 캐시가 타깃을 안 탄다. 그래서 타깃을 N개로
# 늘려도 비용의 90% 를 차지하는 분절 호출은 **그대로**이고, 번역(gtx 무료)과 채점(GPU)만
# N배가 된다. API 비용 증가는 사실상 0 이다.
#
# 얻는 것이 둘이다.
#   1. 타깃 편향 제거. 프롬프트 문면에서 타깃을 막아도(check_target_agnostic) 목적함수가
#      한 언어면 최적화가 그쪽으로 끌려간다. 여러 타깃에서 동시에 좋은 경계라야 통과한다.
#   2. **검출력**. test 100문장 실측에서 문장별 effective 의 타깃 간 Spearman 이
#      +0.50~+0.63 이라 오차가 부분적으로 독립이다. 평균의 se 가
#      1개 0.0170 → 2개 0.0132 → 3개 0.0116 → 5개 0.0102 (−40%) 로 줄었다.
#      루프가 검출하려는 프롬프트 차이가 0.006, dev se 가 0.008 이었으므로 이 감소가
#      "효과 크기와 잡음이 같은 자릿수" 라는 벽을 넘게 해준다.
#
# **타깃별 z-정규화 후 평균**을 쓴다. 원값 평균은 분산이 큰 타깃이 지배한다 (실측 문장별
# sd: zh 0.147 ~ ko 0.196). 절대값 보고는 타깃별 곡선으로 따로 낸다.
#
# 위 −40% 는 **평균적 타깃 기준**이다. run05 test 실측 타깃별 raw se 는 ko 0.0105 /
# ja 0.0160 / zh 0.0177 / es 0.0174 / de 0.0179 로 폭이 넓어, 5타깃 평균의 se 0.0120 은
# 평균 대비 −25% 지만 가장 조용한 타깃(ko) 단독보다는 14% 나쁘다. 다언어의 이득은
# "무조건 잡음이 준다" 가 아니라 **어느 타깃이 조용한지 모르는 상태에서 평균적 타깃보다
# 낫고, 한 언어에 과적합하지 않는다** 는 쪽이다.
#
# z 기준선은 **분할별로 한 번 정해 고정한다** (`_zmix`, `run_dir/z_baseline.json`).
# 평가마다 다시 잡으면 채택 판정이 망가진다 — run05 에서 실제로 그랬다.

DEFAULT_TARGET_POOL = ["English", "Korean", "Japanese", "Chinese", "Spanish", "German"]


def resolve_targets(pool: list[str], src_lang: str) -> list[str]:
    """검증 풀에서 소스 언어를 뺀다. 자기 자신으로 번역하는 점수는 의미가 없다."""
    s = (src_lang or "").strip().lower()
    return [t for t in pool if t.strip().lower() != s]


def _zmix(per_target: dict[str, list[float | None]],
          base: dict[str, tuple[float, float]] | None = None) -> list[float | None]:
    """타깃별 문장 점수를 z-정규화해 평균. 어느 타깃에서도 값이 없으면 None.

    **기준선은 반드시 평가 바깥에서 고정돼야 한다.** 평가 세트 안에서 정규화하면
    어떤 프롬프트를 넣든 그 프롬프트의 z 평균이 정확히 0 이 되고, `paired_delta` 가
    재는 `mean(z_new) - mean(z_old)` 가 **항등적으로 0** 이 된다 — run05 dev 실측에서
    두 이터레이션 모두 `mean_z = +0.00000`, `Δ = +0.00000 ±0.06408` 이 나왔다.
    검출력이 0 인 관문이었다.

    세트가 일부만 겹치면 증상이 반대로 나온다. 두 기준선이 **서로 다른 문장 추출**로
    정해져 상쇄되지 않고 겹친 부분에 임의의 오프셋이 남는다 — run05 train 은 pool 80
    에서 40 을 회전 추출해 겹침이 18/40 이었고, T=6 에서 `mean_new +0.05353` vs
    `mean_old -0.10363` → `Δ +0.157`. 어느 draw 에 쉬운 문장이 더 많았냐가 만든 값이지
    프롬프트 품질이 아니다.

    기준선을 고정하면 평균 항이 쌍체 차이에서 **소거되고 `sd` 만 남는다** — 타깃별
    분산을 맞춰 가중한 원값 차이가 되어 z 의 본래 목적(분산이 다른 타깃의 동등 가중)은
    지키면서 두 인공물이 함께 사라진다.

    `base` 는 `{타깃: (평균, 표준편차)}`. 값이 없는 타깃은 이 세트에서 계산해 **`base`
    에 채워 넣는다** — 호출자가 그대로 다음 평가에 넘겨 재사용한다. `base=None` 이면
    옛 동작(세트 안 정규화)이고, 단일 평가를 그 자체로 볼 때만 쓴다.
    """
    zs: dict[str, list[float | None]] = {}
    for tgt, vals in per_target.items():
        ref = base.get(tgt) if base is not None else None
        if ref is not None:
            m, sd = float(ref[0]), float(ref[1]) or 1.0
        else:
            ok = [v for v in vals if v is not None]
            if len(ok) < 2:
                zs[tgt] = [None] * len(vals)
                continue
            m = sum(ok) / len(ok)
            sd = (sum((v - m) ** 2 for v in ok) / (len(ok) - 1)) ** 0.5 or 1.0
            if base is not None:
                base[tgt] = (m, sd)
        zs[tgt] = [None if v is None else (v - m) / sd for v in vals]
    n = len(next(iter(per_target.values()))) if per_target else 0
    out: list[float | None] = []
    for i in range(n):
        got = [zs[t][i] for t in zs if zs[t][i] is not None]
        out.append(sum(got) / len(got) if got else None)
    return out


class StageTimer:
    """이터레이션의 **단계별 벽시계 시간**을 재서 파일로 남긴다.

    **LangSmith 로는 안 된다.** 그쪽은 LLM 호출 1건씩만 본다 (실측 중앙: `segment`
    161s, `segment_retry` 50s, `judge` 6s, `critic` 63s, `prompt_engineer` 49s).
    루프의 벽시계에는 그 사이가 들어 있다 — CometKiwi·NLI 채점(로컬 GPU), 번역,
    그리고 **호출을 얼마나 겹쳐서 던졌는가**. 마지막 항목이 병목이었다 (run04 실측
    평균 동시 실행 3.03 / 워커 8). 호출 시간을 다 더해도 그건 안 보인다.

    파일로 남기는 이유: 지금까지의 런은 **시간이 하나도 안 남아 있다.** 로그에
    타임스탬프가 없고 디렉토리 mtime 은 체크아웃 시각으로 덮였다. 그래서 "6시간짜리
    런이 어디서 오래 걸리나" 를 물으면 추정밖에 못 했다.
    """

    def __init__(self, path):
        self.path = path
        self.rows: list[dict] = []
        self._t0 = time.time()
        self._stage: str | None = None
        self._start = self._t0

    def mark(self, stage: str | None) -> None:
        """직전 단계를 닫고 새 단계를 연다. `None` 이면 닫기만 한다."""
        now = time.time()
        if self._stage is not None:
            self.rows.append({"stage": self._stage,
                              "sec": round(now - self._start, 2),
                              "at_sec": round(self._start - self._t0, 2)})
            self._flush()
        self._stage, self._start = stage, now

    def _flush(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "total_sec": round(sum(r["sec"] for r in self.rows), 2),
                "stages": self.rows,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:                       # 계측이 런을 죽이면 안 된다
            pass


def judge_distributed(judge, per_rows: dict[str, list[dict]], T: int, frac: float,
                      workers: int = 8):
    """타깃마다 **자기 경계의 `frac` 비율**을 판정한다.

    한 타깃에만 판정자를 돌리면 목적함수에서 뺀 타깃 편향이 Critic 케이스 선정으로
    되돌아온다. 타깃별로 같은 비율을 재면 편향이 없고, 타깃마다 다른 실패(어떤
    언어에서만 깨지는 경계)가 드러나는 이득이 붙는다.

    **고정 개수(종전 8)를 비율로 바꿨다.** 개수는 train 크기와 T 에 따라 의미가 통째로
    달라진다 — 같은 8 이 T=12·30문장에서는 경계의 4.7% 지만 T=6·40문장에서는 1.5% 다.
    게다가 `total // len(targets)` 로 나눠서 나머지를 버렸다 (타깃 5개면 8 이 5가 됐다).
    비율은 두 문제를 한 번에 없앤다.
    """
    out: list[dict] = []
    for t, rows in per_rows.items():
        n = 0
        for r in rows:
            d = (r.get("by_T", {}) or {}).get(str(T)) or {}
            ps = d.get("pieces_tgt") or []
            n += max(0, len(ps) - 1)
        js = agents.judge_top_contra(judge, rows, T, max(1, int(n * frac + 0.5)),
                                     workers=workers)
        for j in js:
            j["target"] = t
        out.extend(js)
    return out


def evaluate_multi(targets: list[str], make_ctx, prompt, sentences, t_grid,
                   zbase: dict | None = None, **kw):
    """타깃마다 `evaluate` 를 돌리고 **z-평균 effective** 로 합친다.

    `make_ctx(tgt) -> (translator, adequacy, consistency, contradiction, tgt_spaced)`.
    분절은 첫 타깃에서 한 번 호출되고 나머지는 전부 캐시 히트다 — 캐시 키에 타깃이 없다.

    `zbase` 는 **분할별로 고정된 z 기준선** `{str(T): {타깃: (평균, sd)}}` 이다. 비어
    있으면 이번 평가에서 채우고, 이후 평가는 그 값을 그대로 쓴다 — 채택 판정이
    기준선 이동이 아니라 실제 점수 차이를 보게 만드는 장치다 (`_zmix` 참고).

    반환 `(merged_rows, Metrics, violations, per_target_metrics, per_target_rows)`.
    `merged_rows[i]["by_T"][T]` 에는 대표 타깃(첫 번째)의 조각 정보가 남고,
    `effective` 만 z-평균으로 덮어쓴다. `by_tgt` 에 타깃별 원값을 함께 싣는다.
    """
    per_rows: dict[str, list[dict]] = {}
    per_m: dict[str, "metrics.Metrics"] = {}
    viol: list[dict] = []
    norm_sink: list[dict] | None = kw.pop("norm_sink", None)
    for k, tgt in enumerate(targets):
        tr, adq, cons, contra, tsp = make_ctx(tgt)
        r, m, v = evaluate(gw=kw["gw"], translator=tr, prompt=prompt, sentences=sentences,
                           spaced=kw["spaced"], seg_cache=kw["seg_cache"],
                           workers=kw["workers"], adequacy=adq, consistency=cons,
                           t_grid=t_grid, trailing_punct=kw["trailing_punct"],
                           tgt_spaced=tsp, contradiction=contra,
                           require_coverage=kw["require_coverage"],
                           coverage_t=kw["coverage_t"],
                           reasoning_effort=kw["reasoning_effort"],
                           batch_size=kw["batch_size"], min_gap=kw["min_gap"],
                           skip_translation_below=kw["skip_translation_below"],
                           # 정규화도 분절의 성질이라 타깃과 무관하다 — 첫 타깃만 모은다.
                           # 나머지는 전부 캐시 히트라 normalize 가 아예 안 돈다.
                           norm_sink=(norm_sink if k == 0 else None))
        per_rows[tgt] = r
        per_m[tgt] = m
        if k == 0:                      # 위반은 분절의 성질이라 타깃과 무관 — 한 번만
            viol = v
    base = per_rows[targets[0]]
    merged = [dict(r) for r in base]
    by_T_keys = sorted({T for r in base for T in (r.get("by_T") or {})}, key=int)
    split_m: dict[str, "metrics.SplitMetrics"] = {}
    for T in by_T_keys:
        per_target = {t: [((r.get("by_T") or {}).get(T) or {}).get("effective")
                          for r in per_rows[t]] for t in targets}
        # 병기 지표도 같은 방식으로 타깃 평균을 낸다. 안 하면 대표 타깃 값이 그대로 남아
        # "다언어 평균" 인 척하게 된다.
        per_target_ent = {t: [((r.get("by_T") or {}).get(T) or {}).get("effective_ent")
                              for r in per_rows[t]] for t in targets}
        mixed_z = _zmix(per_target,
                        None if zbase is None else zbase.setdefault(str(T), {}))
        rep = per_m[targets[0]].by_T.get(T)
        if rep is None:
            continue
        # 대표 타깃의 조각·지연 정보는 그대로 두고 점수만 다언어 값으로 바꾼다.
        #   effective   = 타깃별 **원값** 평균 — 보고·곡선용, 해석 가능
        #   effective_z = 타깃별 z 평균        — 쌍체 Δ·채택 판정용, 검출력
        mixed_raw: list[float | None] = []
        for i, r in enumerate(merged):
            vals = [per_target[t][i] for t in targets if per_target[t][i] is not None]
            raw = sum(vals) / len(vals) if vals else None
            mixed_raw.append(raw)
            d = (r.get("by_T") or {}).get(T)
            if not d:
                continue
            d = dict(d)
            d["by_tgt"] = {t: per_target[t][i] for t in targets}
            d["effective_single"] = d.get("effective")
            d["effective"] = raw
            d["effective_z"] = mixed_z[i]
            ev = [per_target_ent[t][i] for t in targets if per_target_ent[t][i] is not None]
            d["effective_ent"] = (sum(ev) / len(ev)) if ev else None
            r.setdefault("by_T", {})[T] = d
        sm = copy.copy(rep)
        ok = [x for x in mixed_raw if x is not None]
        okz = [x for x in mixed_z if x is not None]
        # **빈 경우는 None 이다 — 0.0 이 아니다.** 단일 타깃(`aggregate_split`)과 규약이
        # 달랐다. `score()` 는 None 을 평균에서 빼지만 0.0 은 넣으므로, 전 문장 무분절인
        # 비교군이 다언어 모드에서만 0 점으로 끌려 내려갔다.
        sm.effective = round(sum(ok) / len(ok), 4) if ok else None
        sm.effective_z = round(sum(okz) / len(okz), 4) if okz else None
        oke = [x for r in merged
               for x in [((r.get("by_T") or {}).get(T) or {}).get("effective_ent")]
               if x is not None]
        sm.effective_ent = round(sum(oke) / len(oke), 4) if oke else None
        sm.contradiction_ent = None      # 타깃마다 다른 값이라 평균이 뜻을 잃는다
        sm.n_effective = len(ok)
        sm.n_targets = len(targets)
        sm.effective_min = round(min(ok), 4) if ok else None
        p10 = metrics.percentile10(ok)
        sm.effective_p10 = round(p10, 4) if p10 is not None else None
        split_m[T] = sm
    m0 = per_m[targets[0]]
    merged_m = metrics.Metrics(n=m0.n, format_pass_rate=m0.format_pass_rate,
                               format_pass_rate_no_retry=m0.format_pass_rate_no_retry,
                               by_T=split_m)
    # **순위축 진단을 병합에서 잃지 않는다.** `rank_lift` 는 타깃과 무관한 성질이다 —
    # 분절이 같으므로 순위를 섞었을 때 잃는 것도 같은 축이다. 새 `Metrics` 를 만들면서
    # 안 옮기면 다언어 런에서는 통째로 사라지고, `focus="priority"` 판정과 Critic 에게
    # 넘길 정보가 둘 다 비게 된다. 대표 타깃(첫 번째) 값을 그대로 싣는다.
    rep_m = per_m[targets[0]]
    merged_m.rank_lift = rep_m.rank_lift
    merged_m.rank_lift_se = rep_m.rank_lift_se
    merged_m.rank_lift_t = rep_m.rank_lift_t
    merged_m.rank_lift_n = rep_m.rank_lift_n
    merged_m.rank_lift_T = rep_m.rank_lift_T
    return merged, merged_m, viol, per_m, per_rows


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
    contra_ent = [0.0] * len(pair_src)
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
        # **한 번의 NLI 호출로 두 척도를 다 받는다** — 세 라벨 확률이 같이 나오므로
        # `1 − entailment` 를 병기하는 데 추가 비용이 0 이다 (`score_dual` 주석 참고).
        if hasattr(contradiction, "score_dual"):
            cs, es = contradiction.score_dual(prem, hyp)
        else:
            cs, es = contradiction.score(prem, hyp), [0.0] * len(prem)
        for s, v, w in zip(slot, cs, es):
            contra[s] = v
            contra_ent[s] = w

    # 경계별 값을 문장에 되돌린다. 문장 평균만 남기면 **어느 경계가** 반박당했는지가
    # 사라지는데, 판정자·비평가 조준에 필요한 것이 바로 그 위치다 (이미 계산된 값).
    # 각 문장 마지막 원소는 뒤에 미래가 없어 항상 0.0 — "안전"이 아니라 "대상 아님".
    contra_rows: list[list[float]] = [[] for _ in texts]
    contra_ent_rows: list[list[float]] = [[] for _ in texts]
    for i, v, w in zip(owner, contra, contra_ent):
        contra_rows[i].append(round(v, 4))
        contra_ent_rows[i].append(round(w, 4))

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
    contradiction_ent_rows: list[float | None] = []
    effective_ent_rows: list[float | None] = []
    for adq, cr, er in zip(adequacy_rows, contra_rows, contra_ent_rows):
        bounds = cr[:-1]
        if bounds:
            c_mean = sum(bounds) / len(bounds)
            contradiction_rows.append(c_mean)
            effective_rows.append(metrics.effective_of(adq, c_mean))
            eb = er[:-1]
            e_mean = sum(eb) / len(eb) if eb else None
            contradiction_ent_rows.append(e_mean)
            effective_ent_rows.append(
                metrics.effective_of(adq, e_mean) if e_mean is not None else None)
        else:
            contradiction_rows.append(None)
            effective_rows.append(None)
            contradiction_ent_rows.append(None)
            effective_ent_rows.append(None)

    return ScoredSplit(
        joined=list(joined),
        pieces_src=chunk_lists,
        pieces_tgt=[list(p) for p in pieces],
        pieces_contra=contra_rows,
        pieces_contra_ent=contra_ent_rows,
        effective_ent=effective_ent_rows,
        contradiction_ent=contradiction_ent_rows,
        effective=effective_rows,
        adequacy=adequacy_rows,
        contradiction=contradiction_rows,
        consistency=consistency.score(texts, joined, full),
        chrf=[metrics.chrf(h, r) for h, r in zip(joined, full)],
        laal_words=[metrics.laal_words(st, ps, f, spaced, tgt_spaced)
                    for st, ps, f in zip(seg_texts, pieces, full)],
        k=[max(1, len(c)) for c in chunk_lists],
    )


# ── 평가 1회 ────────────────────────────────────────────────────────────

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
    batch_size: int = 1,
    min_gap: int = 0,
    rank_lift_seed: int = 0,
    rank_lift_shuffles: int = 3,
    norm_sink: list[dict] | None = None,
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
    need = lambda txt: (coverage_need(txt, min_t, spaced, min_gap)
                        if require_coverage else None)

    first_pass_viol: list[dict] = []

    # **결정론적 수정을 기록한다.** `normalize_tags` 는 산출물을 조용히 바꾸는데(구두점
    # 재배치·태그 삭제·연속 태그 병합·재번호) 그게 어디에도 안 남으면 규칙이 틀렸을 때
    # 영영 모른다 — zh 여는 따옴표 13건이 정확히 그렇게 묻혔다. 검증기의
    # 검증기의 태그-뒤-구두점 검사는 이 함수가 먼저 돌아 고쳐 놓으므로 구조적으로 0건이었다.
    norm_log: list[dict] = []

    def _norm(t: str, out: str) -> str:
        rec: list[dict] = []
        fixed = normalize_tags(out, spaced, trailing_punct, sink=rec, min_gap=min_gap)
        norm_log.extend({"text": t, **e} for e in rec)
        if norm_sink is not None:
            norm_sink.extend({"text": t, **e} for e in rec)
        return fixed

    seg_texts, first_pass = segment_batch(
        gw, prompt, texts, cache=seg_cache, workers=workers,
        validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct,
                                            require_priority, need(t)),
        normalize_fn=_norm,
        reasoning_effort=reasoning_effort,
        batch_size=batch_size,
        first_pass_sink=first_pass_viol,
    )

    violations: list[dict] = []
    valid_flags: list[bool] = []
    scored_flags: list[bool] = []
    for s, seg in zip(sentences, seg_texts):
        vs = validate(s.id, s.text, seg, spaced, trailing_punct, require_priority,
                      need(s.text))
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
    lift_T = max(t_grid) if t_grid else None      # 순위 진단은 폐기율이 가장 높은 곳에서
    real_eff_at_lift_T: list[float | None] = []
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
                "effective_ent": (round(sp.effective_ent[i], 4)
                                  if sp.effective_ent[i] is not None else None),
                "contradiction_ent": (round(sp.contradiction_ent[i], 4)
                                      if sp.contradiction_ent[i] is not None else None),
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
            pick(sp.consistency), pick(sp.laal_words), pick(sp.k),
            pick(missings), n_total=len(rows),
            effective_ent_scores=pick(sp.effective_ent),
            contradiction_ent_scores=pick(sp.contradiction_ent))
        if T == lift_T:
            real_eff_at_lift_T = list(sp.effective)

    m = metrics.aggregate(len(rows), valid_flags, first_pass, by_T)

    # ── 순위 축 진단 ─────────────────────────────────────────────────────
    # 순위 번호만 무작위로 치환한 대조군을 한 벌 더 채점한다. 후보 집합·`want`·`min_gap`
    # 이 전부 같으므로 바뀌는 것은 `truncate` 의 keep 집합뿐이고, 그 차이가 곧 순위의 값이다.
    # **LLM 호출은 0건이다** — 분절을 다시 하지 않고 이미 받은 태그의 번호만 섞는다.
    # consistency 는 `effective` 에 안 들어가는 보고 지표라 대조군에서는 끈다.
    if (lift_T is not None and contradiction is not None and real_eff_at_lift_T
            and rank_lift_shuffles > 0):
        # 시드는 **런 전체에서 고정**한다. 이터레이션마다 다른 순열을 뽑으면 lift 변화가
        # 프롬프트 때문인지 순열 때문인지 섞인다.
        rng = random.Random(rank_lift_seed)
        # **셔플을 여러 벌 평균낸다.** 1벌이면 조향이 도는 train(40~60문장)에서 se 가
        # 0.027~0.038 로 커져 t<1 이 잡음으로 울린다 — 기록된 이터레이션 재생에서 en-de
        # run04 가 iter0 +0.0915 / iter1 +0.0188 로 튀었다(test 실측은 +0.061).
        # 3벌이면 대조군 쪽 산포가 1/√3 로 줄어 그 진동이 문턱 위로 올라온다.
        per_shuf: list[list[float | None]] = []
        for _ in range(rank_lift_shuffles):
            shuf = [shuffle_priorities(seg, rng) for seg in seg_texts]
            shuf_cut = [truncate(s, lift_T, spaced, min_gap)[0] for s in shuf]
            per_shuf.append(score_split(shuf_cut, texts, full, translator, adequacy,
                                        _NO_CONSISTENCY, spaced, tgt_spaced,
                                        contradiction).effective)
        mean_shuf: list[float | None] = []
        for i in range(len(seg_texts)):
            vals = [s[i] for s in per_shuf if s[i] is not None]
            mean_shuf.append(sum(vals) / len(vals) if vals else None)
        keep = [i for i, v in enumerate(scored_flags) if v]
        rl = metrics.rank_lift([real_eff_at_lift_T[i] for i in keep],
                               [mean_shuf[i] for i in keep])
        m.rank_lift, m.rank_lift_se = rl["lift"], rl["se"]
        m.rank_lift_t, m.rank_lift_n = rl["t"], rl["n"]
        m.rank_lift_T = lift_T

    for v in first_pass_viol:
        v["first_pass"] = True
    return (rows, m, violations + first_pass_viol)


def load_contra_floor(run_dir, rows: list[dict], backend,
                      filename: str = "contra_floor.json", tgt_spaced: bool = True):
    """경계 contradiction 의 **길이별 잡음 바닥**. 런당 1회 측정하고 디스크에 캐시한다.

    NLI 는 무해한 미완성에도 0 이 아닌 모순 확률을 준다. 그 크기가 hypothesis 길이에
    따라 달라지므로, 보정 없이 순위별 위험을 비교하면 특정 위치의 경계가 구조적으로
    불리해진다. run03 test 에서 이 교란만으로 순위 정렬도가 −0.25 로 나왔고 보정 후
    +0.14 였다.

    **방향은 백엔드마다 다르다 — 그리고 뒤집혔다.** 위 run03 근거와 종전 주석이 인용하던
    "짧을수록 크다(1-2어절 0.113, 10어절+ 0.003)" 는 `deberta-mnli` 측정이고, 그 백엔드는
    삭제됐다. 현행 `xlmr-anli` 는 반대로 **길수록 크다** — German 0.024 → 0.107,
    English 0.065 → 0.117, Chinese 0.074 → 0.123 (문장 611/350/100 재측정,
    `runs/noise_floor_xlmr/`). 저장된 바닥 파일 9개 중 "짧을수록 큼" 은 0개다.
    따라서 이 보정을 쓰는 진단(`rank_contra_gap`)의 부호 해석을 다시 봐야 한다.

    바닥은 (코퍼스, 번역기, NLI 백엔드)의 성질이지 프롬프트의 성질이 아니다 — full
    번역은 이터레이션 간 불변이므로 다시 잴 이유가 없다. 번역 호출은 0 이고 NLI 만 돈다.

    반환: `floor_fn(hyp_words) -> c0`. 잴 수 없으면 None (보정 없이 raw 로 진행).

    `filename` 은 타깃 언어마다 바닥이 다르기 때문에 있다 — full 번역이 달라지면
    바닥도 달라진다.
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
        # **`p10` 을 평균 옆에 항상 같이 찍는다.** `effective` 평균은 소수의 나쁜 문장에
        # 크게 끌린다 — 실측(en-de/run01 pool_base T=6, 100문장): 평균 0.7760 / 중앙
        # 0.8021 이고 **최악 1문장을 빼면 평균이 +0.0034 오른다.** 프롬프트 간 실제 차이가
        # 0.003~0.007 규모라 같은 크기다. 평균만 보면 "개선"과 "그 배치에 유난히 나쁜
        # 문장이 하나 덜 걸림"을 구분할 수 없다.
        #
        #   평균 ↑ 이고 p10 도 ↑   →  꼬리까지 좋아졌다. 진짜 개선
        #   평균 ↑ 인데 p10 은 ↓   →  꼬리 운이다. 의심할 것
        parts.append(f"T{k}: eff={_cell(s.effective, '.4f')} p10={_cell(s.effective_p10, '.4f')} "
                     f"[ent {_cell(s.effective_ent, '.4f')}] "
                     f"adq={s.adequacy:.4f} "
                     f"contra={_cell(s.contradiction, '.3f')} laal={s.laal_words:.2f} "
                     f"k={s.chunks_per_sentence:.2f} miss={s.missing_boundaries:.2f}")
    parts.append(f"score={metrics.score(m):.4f}")
    # **순위축은 로그에 남긴다.** 지금까지 어느 런에도 기록된 적이 없어서(기능이 마지막
    # 런보다 나중에 들어왔다) 순위가 실제로 값을 하는지 확인할 방법이 없었다.
    if m.rank_lift is not None:
        parts.append(f"rank_lift={m.rank_lift:+.4f}"
                     + (f"(t{m.rank_lift_t:+.1f}, T{m.rank_lift_T})"
                        if m.rank_lift_t is not None else ""))
    return "  ".join(parts)


# ── 메인 ───────────────────────────────────────────────────────────────

# ── T 격자 유도 ──────────────────────────────────────────────────────────

# **`t_floor` 는 하한 두 개 중 큰 쪽이다.** 종전 `ceil(1.3 · min_gap)` 은 둘을 한
# 상수로 뭉뚱그렸고, 주석은 그걸 "포화 바닥"이라고만 설명해 근거가 틀려 있었다.
#
# ① **포화 바닥 = `min_gap + 1`** — 절단기 쪽. 유도된다.
#    `T <= min_gap` 이면 요청 조각 수가 용량(`길이 // min_gap`)을 넘어 전부 같은 결과가
#    나온다. 곡선에 같은 점이 여러 번 찍히므로 격자에 넣을 이유가 없다.
#    저장된 런 5개 재절단에서 "앞 T 와 값이 처음 달라지는 T" 가 **5/5 모두 min_gap+1**:
#        en-de/run01 mg3 -> 4   de-en/run01 mg3 -> 4   ko-en/run04 mg3 -> 4
#        ja-en/run01 mg8 -> 9   zh-en/smoke04 mg6 -> 7
#
# ② **마킹 한계 ≈ `1.25 · min_gap`** — 모델 쪽. 실측이다.
#    검증기가 `길이/t_floor - 1` 개를 요구하는데, 모델이 그만큼 못 찍으면 재시도가
#    성공할 수 없다. 실제 마킹 간격 (길이 / (마크+1)):
#        de 3.39 (mg3, 1.13배)   zh 7.17 (mg6, 1.20배)   ja 9.89 (mg8, 1.24배)
#    **하드 한계는 아니다** — 물리적 상한은 min_gap(=1.0배)이고 모델은 그 77~86% 만
#    찍는다. 즉 1.25 는 "관측된 행동"이지 "능력의 벽"이 아니라, 런에서 더 촘촘히
#    찍는 것이 확인되면 내릴 수 있다. 내리면 후보가 늘어 절단이 고를 여지가 커진다.
#
# 둘 중 어느 쪽이 이기는지는 min_gap 크기에 따라 갈린다 (교차점 min_gap=4~5):
# 어절 언어(mg 3~4)는 ①, 글자 언어(mg 6~8)는 ② 가 결정한다.
#
# 배수 (1, 1.5, 2, 3) 은 기존 기본 격자 [2,3,4,6] 의 비율 그대로다 — `min_gap=0` 이면
# 예전 기본값 ([2,3,4,6] / [3,6])이 정확히 재현되므로 기존 런과의 연속성이 유지된다.
MARK_SPACING_RATIO = 1.25
# **`min_gap` 은 시간이다 — 토큰 수가 아니다.**
#
# 하는 일이 "청자가 알아들을 수 있는 최소 방출량"이라 **문장 길이와 무관**해야 한다.
# 종전 식 `중앙 문장길이 × 0.15` 는 길이 무관한 양을 길이에서 유도했고, 근거로 든
# 네 언어 중 de·ja·zh 는 **같은 240문장**을 세 언어로 옮긴 것이라 독립 관측이 1개였다.
# 성격이 다른 ko(KsponSpeech, 중앙 5어절)에서 유도값 2 가 나와 사람이 쓴 3 과 어긋났다.
#
# 강제정렬로 발화 속도를 재니 손으로 정했던 값들이 **하나의 시간**으로 모인다
# (FLEURS loop240, 발화 구간 기준. `data.units_per_sec` 참고):
#     de 3어절 / 2.43 = 1.23초    zh 6자 / 4.74 = 1.27초
#     ja 8자  / 5.77 = 1.39초    en 3어절 / 2.88 = 1.04초
# 산포가 토큰 축 2.7배 -> 시간 축 1.12배로 준다. 시간 축 재정의가 예고한
# "r=0.15 는 시간 불변성을 토큰 축에서 재구성한 것" 이 수치로 확인된 셈이다.
#
# **1200ms 는 지연 요건에서 나온다 — 지각 하한이 아니다.**
#
# 문헌에 "의미를 파악할 수 있는 최소 덩어리"의 하한은 없다. 있는 것은 **범위**다:
#     의미 덩어리(intonation unit)   1.6~3초   (Chafe; 자발발화 intonational phrase)
#     언어 처리 시간 창              ~2.7초    (Nature Rev. Psych. 2026)
#     청자가 지각하는 덩어리          ~2.5초
#     통역사 EVS                     2~4초
#     사용자 수용 지연               <3.5초, 이상적 <2~3초
# 즉 문헌이 정하는 것은 **T 의 범위**이고, `min_gap` 은 거기서 역산된다:
#     ① 곡선이 수용 구간(<3.5초)을 아래까지 훑으려면 최소 작동점 ~1.5초
#     ② 최소 작동점 = t_floor
#     ③ min_gap = t_floor / 1.25 (마킹 한계 실측) = 1.2초
#
# 실측 검증 — 사람이 손으로 정한 값과 일치한다 (독립 판단 두 개):
#     de  1.2 x 2.43 = 2.92 -> 3   (손값 3)
#     en  1.2 x 2.88 = 3.46 -> 3   (손값 3)
#     ko  1.2 x 2.11 = 2.53 -> 3   (손값 3, KsponSpeech 자발발화)
# ko 는 종전 `0.15 x 중앙길이` 가 2 를 내서 어긋나던 케이스다 — 시간 축이 고쳤다.
# zh 6 / ja 7 은 검증이 아니다: 그 손값(6, 8)이 en 의 3 을 길이 비율로 환산한
# 파생값이기 때문이다 (MULTI2EN_DATASET.md §3).
#
# 문헌 하한(1.6초)에 맞추면 안 된다 — en min_gap 5, 격자가 {2.4, 3.8, 4.9, 7.3}초가
# 되어 **네 점 중 둘이 수용 한계 3.5초 밖**이다. 빠른 구간을 못 재게 된다.
#
# **시간이 프롬프트에 도달하지는 않는다.** 분절기는 텍스트 모델이라 "1.3초마다 표시"를
# 못 알아듣는다. 환산은 여기서 끝나고 아래로는 전부 토큰
# 단위로 흐른다 — 프롬프트 문면도 검증기도 절단기도 종전과 같다.
MIN_GAP_MS = 1200


def derive_min_gap(units_per_sec: float) -> int:
    """발화 속도(단위/초)를 최소 조각 크기(단위)로 환산한다."""
    return max(2, round(MIN_GAP_MS / 1000.0 * units_per_sec))


def derive_t_floor(min_gap: int) -> int:
    """격자 바닥이자 마킹 밀도 기준. 포화 바닥과 마킹 한계 중 **큰 쪽**."""
    mg = max(0, min_gap)
    return max(2, mg + 1, -(-125 * mg // 100))        # ceil(1.25 * mg)


def derive_t_grids(min_gap: int) -> tuple[list[int], list[int]]:
    """(t_grid, final_t_grid). min_gap 아래로는 어차피 못 내려가므로 요청도 안 한다.

    반올림은 `chunk_budget` 과 **같은 규칙**(`round_half_up`)을 쓴다. 예전에는 여기만
    `int(x+0.5)`, 조각 수는 파이썬 `round()`(짝수 반올림)라 같은 사슬 안에서 규칙이
    둘이었다.

    **루프 격자에 `t_floor`(배수 1)를 넣는다 — 종전 `{1.5, 3}` 에서 바꿨다.**

    종전 조합에 근거가 있었던 게 아니다. 옛 기본값이 `--t-grid 3 6` / `--final-t-grid
    2 3 4 6` 이었고, 격자를 `min_gap` 에서 유도하도록 바꿀 때 "`min_gap=0` 이면 옛
    기본값이 그대로 재현되게" 배수를 역산한 결과가 `{1.5, 3}` 이었다. 점을 왜 그 둘로
    골랐는지는 어디에도 없다. 부분집합을 쓰는 이유만 적혀 있다 — 조각 번역 호출이 격자
    크기에 비례해서(설계 §8.5).

    그 사이 격자가 통째로 위로 밀렸다. 옛 `{3,6}` 은 최종 `[2,3,4,6]` 에서 최소값 하나만
    뺀 것이었는데, `min_gap=3` 이면 루프 `[6,12]` / 최종 `[4,6,8,12]` 라 **아래 절반**이
    빠진다. 그래서 세 가지가 어긋나 있었다.

      1. 커버리지 요건은 `coverage_t = min(최종 격자) = t_floor` 로 강제하는데 **점수는
         거기를 안 잰다.** 같은 종류의 불일치가 run03 에서 test 1차 통과율 0.34 사고를
         냈다 (요건 쪽은 그때 고쳤고 점수 쪽이 남아 있었다).
      2. 채택된 프롬프트가 **한 번도 최적화되지 않은 점**(T=t_floor)에서 최종 심판을
         받는다. T 별 프롬프트 순위는 같지 않다 — en-de/run01 프롬프트 9개에서
         Spearman(T6, T12) = +0.70, 1등도 서로 다르다.
      3. `t_floor` 는 논문이 주장할 **최저지연 작동점**이다. 그 점을 안 보고 최적화한다.

    배수 `{1, 1.5, 3}` 은 곡선의 앞·중간·뒤를 잡는다. 양 끝만(`{1,3}`) 쓰면 가운데에서
    무너지는 프롬프트를 못 걸러낸다. 비용은 조각 번역 기준 2점 → 3점으로 +50% 인데,
    기본 번역기가 Google(사실상 0원, 결정론)이라 LLM 번역기를 쓸 때만 실제 부담이다.

    **격자를 바꾸면 `score` 는 이전 런과 비교할 수 없다** — 격자 평균이기 때문이다.
    어떤 격자를 썼는지는 `config.json` 에 남는다.

    부수 효과: `main_t`(판정자 작동점, 기본 `t_grid` 의 중앙)가 최대값에서 가운데 점으로
    옮겨간다 (min_gap=3 이면 12 → 6). 판정자는 대표 작동점을 봐야 하므로 이쪽이 맞다.
    """
    b = derive_t_floor(min_gap)
    final = sorted({round_half_up(b * m) for m in (1, 1.5, 2, 3)})
    loop = sorted({round_half_up(b * m) for m in (1, 1.5, 3)})
    return loop, final


def main() -> int:
    p = argparse.ArgumentParser(description="의미 분절 프롬프트 자동 생성 루프 (v2)")
    p.add_argument("--dataset", default="kspon",
                   help="등록된 이름 또는 매니페스트 경로(.jsonl). 이름 목록은 data.DATASETS")
    p.add_argument("--src-lang", default="Korean")
    p.add_argument("--tgt-lang", default="English")
    p.add_argument("--run-id", default=None)
    p.add_argument("--pair-id", default=None,
                   help="런 디렉토리 이름. 미지정 시 언어명에서 생성")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--judge-model", default=None,
                   help="판정자 모델. 미지정 시 --model. 분절기와 다른 모델을 쓰면 순환이 준다")
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
    p.add_argument("--revision-candidates", type=int, default=3,
                   help="이터레이션당 개정 후보 수. 2 이상이면 probe 로 골라 쓴다")
    p.add_argument("--v0-candidates", type=int, default=1,
                   help="prompt_v0 후보 수. 2 이상이면 dev 일부로 골라 시작한다")
    p.add_argument("--select-n", type=int, default=0,
                   help="후보 선별에 쓸 **train** 문장 수. 0 = train 전체. dev 는 채택 "
                        "판정 전용이라 선별에 쓰지 않는다 (select_prompt 주석 참고). "
                        "정확도는 문장 수만 따른다 — 실측 1위적중 20문장 36%% / 40문장 58%% "
                        "/ 60문장 76%%")
    # 한 분절 호출에 넣을 문장 수. **비용의 유일한 큰 레버**다 — en-de test 100문장 실측:
    # b=1 $1.05 → b=6 $0.47 (55% 절감), 쌍체 Δ(T6) −0.0026±0.0056 으로 품질 차이 검출 안 됨.
    # **b=12 부터 무너진다**: 1차 통과율 0.75(b=12)·0.27(b=24)로 떨어져 단건 재시도가
    # 폭증하고, 비용이 U자로 되돌아오면서(b=12 $0.53, b=24 $0.80) 품질도 b=12 에서
    # Δ −0.019(t=−2.0)로 유일하게 유의하게 나빠졌다. 6 을 넘기지 말 것.
    # 절단기 최소 간격. T 는 평균이지 하한이 아니라 1~2어절 조각이 섞여 나온다
    # (T=6 에서 47/100 문장). en-de 오프라인 시뮬 최적값 3 (pipeline.truncate 참조).
    #
    # 기본값을 0 -> 3 으로 올렸다. **지표는 이 값을 지지하지 않는다. 그럼에도 켜 둔다.**
    #
    # 7언어쌍 스윕 실측 (같은 마킹을 min_gap 만 바꿔 재절단·재채점, mg=0 대비 같은 지연에서.
    # ko->{en,de,es,ja,zh} + en->de + en->ko, xlmr-anli):
    #     effective    mg2 -0.0032  mg3 -0.0029  mg4 -0.0361
    #     consistency  mg2 +0.0162  mg3 -0.0116  mg4 -0.0234
    # mg=3 은 7쌍 x 2지표 = 14번의 비교에서 한 번도 1위를 못 했다. 언어로 일반화되는
    # 최적값도 없다 — 소스가 같은 en-de(mg4 최고)와 en-ko(mg4 최악)가 정반대다.
    #
    # 그런데 **두 지표 모두 이 값이 막으려는 실패를 볼 수 없다.** adequacy 는
    # (조각 원문, 조각 번역) 쌍만 보므로 'What' -> 'What' 을 충실한 번역으로 채점하고,
    # consistency 는 합본만 보므로 어디서 잘랐든 값이 같다. 실제로 관측된 출력:
    #     What <SEG:1> are <SEG:2> you <SEG:3> working <SEG:4> on?
    #   min_gap=0 -> "What / are you working on?"      한 단어를 방출한다
    #   min_gap=3 -> "What are you working on?"        무분절 (옳다)
    # 청자에게 무의미한 한 단어 방출은 지표가 아니라 사용 요건으로 막는다.
    #
    # 부수 효과가 본 기능이기도 하다: 짧은 문장은 min_gap 을 만족하는 자리가 없어
    # 경계 0개 = **무분절**로 나온다. T 가 요청하는 경로와는 별개로, min_gap 이 만드는
    # 상태이고 (pipeline.truncate 참조), 이 경로가 유일하다.
    #
    # 이 값은 T 에 비례하지 않는 **절대 하한**이라, T 를 줄여도 과분절이 안 따라
    # 내려간다 — 저지연 작동점에서 유일하게 듣는 제약이다.
    p.add_argument("--min-gap", type=int, default=None,
                   help="절단 시 경계 간 최소 간격(단위는 측정이 정한다: 어절 또는 문자). "
                        "미지정 시 **코퍼스에서 유도**한다 (중앙 단위수 × 0.15). 0=끔")
    p.add_argument("--batch-size", type=int, default=6,
                   help="한 분절 호출에 넣을 문장 수. 실측 최적 6, 12 이상은 역효과")
    p.add_argument("--units-per-sec", type=float, default=None,
                   help="코퍼스 발화 속도(단위/초). 강제정렬 산출물이 없을 때 직접 준다")
    p.add_argument("--t-floor", type=int, default=None,
                   help="후보 마킹 하한 기준 T. 작을수록 많이 찍는다. 미지정 시 min(--final-t-grid)")
    p.add_argument("--skip-translation-below", type=float, default=0.95,
                   help="원문 보존율이 이 값 미만이면 번역·채점 생략 (0 = 항상 채점)")
    p.add_argument("--agent-reasoning-effort", default="medium",
                   choices=["minimal", "low", "medium", "high", "none"],
                   help="Profiler/Judge/Critic/PE 사고량. none = 모델 기본값")
    p.add_argument("--seg-reasoning-effort", default="medium",
                   choices=["minimal", "low", "medium", "high", "none"],
                   help="분절 호출 사고량. none = 모델 기본값. 에이전트 호출에는 영향 없음")
    p.add_argument("--translate-backend", default=None, choices=["gtx", "v2"],
                   help="번역 백엔드. 미지정 시 GOOGLE_TRANSLATE_API_KEY 가 있으면 v2, "
                        "없으면 gtx. **두 백엔드의 번역문은 같지 않다** — 섞어 비교 금지")
    p.add_argument("--tgt-code", default=None)
    # **목적함수를 다언어로.** 분절은 타깃 무관이라 비용의 90% 가 그대로다 (loop 상단 주석).
    # 소스 언어는 자동 제외한다.
    p.add_argument("--tgt-langs", nargs="+", default=None,
                   help=f"검증 타깃 풀. 기본 {' '.join(DEFAULT_TARGET_POOL)} (소스는 자동 제외)")
    p.add_argument("--no-google-context", action="store_true")
    p.add_argument("--tgt-spaced", default=None, choices=["yes", "no"],
                   help="타깃 언어가 띄어쓰기를 쓰는가. 미지정 시 --tgt-lang 에서 추론 (LAAL 단위)")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--train", type=int, default=30)
    p.add_argument("--train-pool", type=int, default=None)
    # 프롬프트가 v0 대비 커질 수 있는 **유일한** 상한. 품질 노브가 아니라 비용 천장이다 —
    # 프롬프트는 문장마다 다시 보내므로 길이가 곧 토큰 비용이다.
    #
    # **1.3 -> 1.6.** 1.3 은 "적용된 개정 42건의 v0 대비 배수가 중앙 1.06 / 최대 1.29,
    # 예산 발동 2%" 라는 실측에서 나왔는데, 그 42건은 전부 **예전 섹션 관문을 통과한**
    # 개정이다. 관문을 없애 PE 가 자유로워지자 산출이 1.4~1.5배로 올라갔고, run08 에서
    # 예산 발동이 2% 가 아니라 **연속 100%** 가 됐다:
    #   iter2  17,215자 -> 압축 16,334  > 예산 14,652  거부
    #   iter3  15,866자 -> 압축 14,755  > 예산 14,652  거부 (103자 차)
    # 거부되면 프롬프트가 안 바뀌어 다음 이터가 같은 것을 다시 재고(Δ 정확히 0,
    # 변경 0/265) 이터레이션 하나가 통째로 헛돈다.
    #
    # 실제 비용 영향은 작다 — 호출 1건의 사고 토큰이 13.7k 인데 프롬프트는 ~3k 라
    # 1.3 -> 1.6 이 전체 토큰의 5% 남짓이다. 천장을 조금 올려 개정을 살리는 쪽이 맞다.
    p.add_argument("--max-prompt-growth", type=float, default=1.6,
                   help="프롬프트 길이 천장 (v0 대비 배수). 넘치면 압축기가 깎는다")
    p.add_argument("--dev", type=int, default=60)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--seed", type=int, default=data.DEFAULT_SEED,
                   help="층화 분할 시드. config.json 에 기록된다 — 바꾸면 "
                        "train/dev/test 가 통째로 달라져 런 간 비교가 깨진다")
    # 노브. 루프에서는 부분집합만 쓴다 — 조각 번역이 격자 크기에 비례해 늘기 때문이다.
    p.add_argument("--t-grid", type=int, nargs="+", default=None,
                   help="루프가 쓰는 목표 조각 크기. score 는 이 격자에서의 adequacy 평균이라 "
                        "다른 격자로 잰 score 와 비교할 수 없다")
    p.add_argument("--final-t-grid", type=int, nargs="+", default=None,
                   help="최종 test 곡선용 격자")
    p.add_argument("--main-t", type=int, default=None,
                   help="판정자가 도는 주 작동점. 미지정 시 --t-grid 의 중앙값")
    # **개수가 아니라 비율이다.** 판정자가 붙이는 `cause` 는 6개 범주인데, 저장된 43개
    # 이터레이션에서 라벨이 붙은 경계가 **이터당 중앙 5개**였다 — 6범주에 5표라 최빈값이
    # 1 대 1 대 1 이고, Critic 이 "어떤 실패가 지배적인가" 를 읽을 수 없었다. 비율로 두면
    # train 크기와 T 가 바뀌어도 표본이 같이 따라온다.
    p.add_argument("--judge-frac", type=float, default=0.10,
                   help="타깃별로 판정할 경계 비율 (contradiction 상위부터). 0.10 = 10%%")
    p.add_argument("--no-judge", action="store_true",
                   help="판정자를 끈다. 사례에 '왜·어디로' 설명이 안 붙는다")
    p.add_argument("--adequacy-backend", default="cometkiwi",
                   choices=sorted(metrics.QE_CHECKPOINTS),
                   help="참조 없는 QE. y축 주지표")
    p.add_argument("--consistency-backend", default="nli",
                   choices=["nli", "comet", "xcomet"],
                   help="가설 검증값(보고용). 기본 nli = 합본 vs full 의 양방향 entailment — "
                        "어순 무관. 모델은 metrics.NLI_MODEL 고정. "
                        "comet 계열은 참조 기반이라 어순 편향이 있다")
    # `xlmr-anli` 로 바꾼 근거 (en-de test 100문장 + 관문 6케이스 실측, 2026-08-19):
    #   관문 최소 여유   mdeberta-xnli 0.0027 (통과선상) / deberta-mnli 미측정 / xlmr-anli 0.0994
    #   5개 타깃 곡선   mdeberta 2/5 정상 (ko/zh/ja 역전) / xlmr-anli 5/5
    #   잡음 바닥       mdeberta 0.102 — 실측 신호 0.075 보다 커서 사실상 무정보
    # 대가가 있다: 문장별 분산이 커져 dev 쌍체 se 가 0.0065 -> 0.0144 로 배증한다.
    # 채택 문턱이 `Δ > adopt_se_mult·se` 라 그만큼 보수화되므로 기본 배수를 함께 낮춘다.
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
    # **중간 재개.** 3~5시간짜리 런이 이터레이션 도중 죽으면 종전 선택지는 두 개뿐이었다
    # — 처음부터 다시(PE 가 비결정론적이라 앞의 이터레이션이 통째로 무의미해진다) 또는
    # `--final-only`(거기까지의 best 로 test 만 뽑고 끝). run05 가 실제로 SIGTERM 으로
    # 그렇게 죽었다. 분절·번역 캐시는 남아 있으므로 잃는 것은 **루프 상태**뿐이고,
    # 그 상태는 전부 디스크에 있다 (history.json + iter_NN/).
    p.add_argument("--resume", action="store_true",
                   help="같은 run-id 의 history.json 을 이어받아 중단된 이터레이션부터 "
                        "계속한다. --run-id 를 함께 줘야 한다")
    p.add_argument("--final-only", action="store_true",
                   help="이터레이션을 건너뛰고 기존 best_prompt.txt 로 최종 test 평가만 "
                        "다시 돈다. 루프는 끝났는데 마지막 단계가 죽었을 때 쓴다")
    args = p.parse_args()
    for _f in ("seg_reasoning_effort", "agent_reasoning_effort"):
        if getattr(args, _f) == "none":
            setattr(args, _f, None)           # 모델 기본값에 맡긴다

    # **`min_gap` 을 안 주면 코퍼스에서 유도한다.** 이게 있어야 언어별 숫자가 커맨드에서
    # 사라진다 — 격자·`t_floor` 는 이미 `min_gap` 에서 나오므로 이 하나가 마지막 고리다.
    # 격자 결정보다 먼저여야 해서 데이터셋을 여기서 한 번 읽는다 (jsonl 읽기, 무시 가능).
    # **데이터를 여기서 한 번만 읽고 나눈다.** 격자 유도가 발화 속도를 요구하고,
    # 발화 속도는 코퍼스를 봐야 나온다. 예전에는 여기서 한 번, 아래 A0 에서 또 한 번
    # 읽어 `measure_profile` 이 두 번 돌았고 **모집단도 달랐다**.
    #
    # **측정은 train+dev 만 본다.** test 를 넣으면 그 문장의 구두점이 검증기 규칙
    # (`trailing_punct`)에 반영되어 "루프가 한 번도 보지 않은 데이터" 라는 전제가 깨진다.
    sentences = data.load(args.dataset)
    pool_n = max(args.train, args.train_pool or args.train)
    splits = data.split_data(sentences, pool_n, args.dev, args.test, seed=args.seed)
    fit = splits["train"] + splits["dev"]
    measured = data.measure_profile([x.text for x in fit])
    spaced, trailing_punct = data.profile_settings(measured)

    rate_source = "cli:--min-gap"
    if args.min_gap is None:
        _rate, rate_source = data.units_per_sec(args.dataset, fit, spaced)
        if _rate is None and args.units_per_sec:
            _rate, rate_source = args.units_per_sec, "cli:--units-per-sec"
        if _rate is None:
            print(f"[min_gap] {args.dataset}: 발화 속도를 잴 수 없다 — 강제정렬 산출물"
                  f"(`*_unittimes.json`)이 없다. `--units-per-sec` 로 직접 주거나 "
                  f"`--min-gap` 으로 값을 직접 줄 것.\n"
                  f"  정렬 산출: python -m core.meaning_segmentator.autoseg."
                  f"baselines.build_unittimes --lang <de|ja|zh|ko>", flush=True)
            return 2
        args.min_gap = derive_min_gap(_rate)
        print(f"[min_gap] {args.dataset}: {_rate:.2f}{measured['unit']}/초 × "
              f"{MIN_GAP_MS}ms → --min-gap {args.min_gap}  (출처 {rate_source})",
              flush=True)

    # 격자를 안 주면 min_gap 에서 유도한다. min_gap 이 조각 길이 하한이므로 T 하한도
    # 거기서 나온다 — 상수로 박아 두면 min_gap 을 바꿀 때마다 포화점이 생긴다.
    _dl, _df = derive_t_grids(args.min_gap)
    derived = args.t_grid is None or args.final_t_grid is None
    if args.t_grid is None:
        args.t_grid = _dl
    if args.final_t_grid is None:
        args.final_t_grid = _df
    t_grid = sorted(set(args.t_grid))
    final_grid = sorted(set(args.final_t_grid) | set(t_grid))
    if derived:
        print(f"[grid] min_gap={args.min_gap} 에서 유도: --t-grid {t_grid} "
              f"--final-t-grid {final_grid}  (도달 가능한 최소 조각 ≈ {1.3 * args.min_gap:.1f}단위, "
              f"그 아래 T 는 포화한다)", flush=True)       # log() 는 아직 정의 전이다

    # T 는 조각 크기의 **평균**이고 min_gap 은 **최소**다. 절단기가 보충을 안 하므로
    # (pipeline.truncate) T 를 min_gap 아래로 내려도 조각이 더 잘게 쪼개지지 않는다 —
    # 곡선의 그 점이 min_gap 바닥에 눌려 이웃과 겹친다.
    #
    # 실측(min_gap=3, 평균 조각수. 두 코퍼스가 같은 자리에서 갈린다):
    #   ko-en/run05  T=2 3.93  T=3 3.93  T=4 3.49  T=5 2.97  T=6 2.60
    #   en-de/run04  T=2 5.68  T=3 5.68  T=4 5.20  T=5 4.31  T=6 3.69
    # `T <= min_gap` 은 **완전히 같은 점**이고(2 와 3 이 소수점까지 동일), 그 위로
    # 1.5배까지는 부분적으로 눌린다(T=4 는 움직이긴 하나 간격이 좁다).
    #
    # 틀린 값이 아니라 **중복이거나 압축된 점**이므로 막지 않고 경고만 한다 — 격자는
    # 실험 정의라 조용히 바꾸면 안 된다.
    # 하한은 격자 유도와 **같은 식**을 쓴다 (derive_t_grids 의 b). 예전엔 1.5배를 썼는데
    # 그건 "보충 발동률이 낮아지는 지점"을 눈대중한 값이었고, 실측한 도달 가능 바닥은
    # min_gap 의 1.21~1.33배다. 상수가 둘이면 유도 격자가 자기 경고에 걸린다.
    t_floor = derive_t_grids(args.min_gap)[1][0]  # = derive_t_floor(min_gap)
    if args.min_gap > 0:
        dup = [t for t in final_grid if t <= args.min_gap]
        tight = [t for t in final_grid if args.min_gap < t < t_floor]
        if dup:
            print(f"[warn] T={dup} <= min_gap={args.min_gap} — 절단기가 min_gap 아래로는 "
                  f"더 자르지 않으므로 이 점들은 서로 **완전히 같은 분절**이 된다. "
                  f"곡선에 정보를 더하지 않으니 격자에서 빼거나 --min-gap 을 낮출 것",
                  flush=True)          # log() 는 아직 정의 전이다
        if tight:
            print(f"[note] T={tight} 는 min_gap={args.min_gap} 바로 위라 이웃 점과 간격이 "
                  f"좁다. T >= {t_floor} 에서 깨끗하게 벌어진다", flush=True)
    # 커버리지 요건은 **곡선에 그릴 가장 조인 점**에서 온다. 루프 격자가 아니다 —
    # 배포할 프롬프트는 최종 곡선의 모든 점을 지탱해야 하고, 루프가 그보다 느슨한
    # 요건으로 학습하면 마지막 평가에서만 무너진다 (run03: test 1차 통과율 0.34).
    # 검증기(`evaluate`)와 프롬프트 문면(`output_rules`)이 **같은 값**을 써야 한다.
    coverage_t = min(final_grid)
    # `t_floor` = 유도 격자의 바닥이자 **마킹 밀도 기준**. 종전에 `density` 라는
    # 별도 이름·별도 식(`round(min_gap*4/3)`)이었는데 `ceil(1.3*min_gap)` 과 배수가
    # 사실상 같아 min_gap 1~20 중 4곳만 1 차이였고, min_gap=20 에서는 뒤집혀
    # (27 > 26) 전 문장이 too_few_tags 로 걸리는 버그였다. 하나로 합쳤다.
    #
    # 문면(`initial_prompt`)과 검증기(`need`)가 **같은 값**을 써야 한다 — 어긋나면
    # 전 문장이 재시도돼 비용이 두 배가 되고 1차 통과율 신호가 오염된다.
    #
    # **사용자가 준 `--t-grid` 는 따라가지 않는다.** 격자를 크게 주면 마킹 요건이
    # 같이 느슨해지는데, 밀도는 지금까지 확인된 유일한 품질 레버다
    # (밀도 0.348 -> 0.529 에서 T=6 품질 +0.013. AUTOSEG_DETAILS.md '순위 축 진단').
    if args.t_floor:
        t_floor = args.t_floor
    elif args.min_gap <= 0:
        t_floor = min(2, coverage_t)
    main_t = args.main_t or t_grid[len(t_grid) // 2]
    if main_t not in t_grid:
        print(f"--main-t {main_t} 가 --t-grid {t_grid} 에 없습니다", file=sys.stderr)
        return 2

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    pair_id = args.pair_id or f"{args.src_lang}-{args.tgt_lang}".lower().replace(" ", "_")
    run_dir = _HERE.parent / "runs" / pair_id / run_id
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    # **재개 전제는 먼저 본다.** 아래에서 CometKiwi·NLI 를 GPU 에 올리는 데 몇 분이
    # 걸리므로, 이어갈 것이 없다는 사실은 그 전에 알려야 한다.
    if args.resume:
        if not (run_dir / "history.json").exists():
            print(f"[stop] --resume 인데 {run_dir}/history.json 이 없다. "
                  f"--run-id 로 이어갈 런을 지정할 것 (지금 {run_id})", file=sys.stderr)
            return 2
        if args.fresh:
            print("[stop] --resume 과 --fresh 는 같이 못 쓴다", file=sys.stderr)
            return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    gw = Gateway(model=args.model, budget=args.budget,
                 reasoning_effort=args.agent_reasoning_effort)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    translator = None
    judge_gw = None

    adequacy = metrics.make_adequacy_backend(
        args.adequacy_backend, batch_size=args.comet_batch_size)
    # consistency 의 nli 모델은 contradiction 백엔드를 따른다 — 둘 다 (합본, full)
    # 타깃 언어 쌍을 재므로 언어 선택 기준이 같다 — 기본 xlmr-anli 는 다국어라 공통이다.
    if args.consistency_backend == "nli":
        consistency = metrics.make_backend(
            "nli", model_name=metrics.NLI_MODEL,
            batch_size=args.comet_batch_size)
    else:
        consistency = metrics.make_backend(
            args.consistency_backend,
            **({"batch_size": args.comet_batch_size}
               if args.consistency_backend in metrics.COMET_CHECKPOINTS else {}))
    contradiction = (None if args.no_contradiction else
                     metrics.make_contradiction_backend())

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        # ── A0 데이터 ────────────────────────────────────────────────────
        # 로딩·분할·측정은 위에서 이미 끝났다 (격자 유도가 발화 속도를 필요로 해서).
        data.write_splits(splits, run_dir / "data")
        log(f"[data] {args.dataset}: 전체 {len(sentences)}, "
            f"train {len(splits['train'])} / dev {len(splits['dev'])} / test {len(splits['test'])}")
        (run_dir / "measured_profile.json").write_text(
            json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── A1 Language Profiler ────────────────────────────────────────
        profiler = agents.Profiler(gw)
        profile_path = run_dir / "language_profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            samples = [s.text for s in splits["train"][:20]]
            profile = profiler.profile(samples)
            profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

        # **JSON 은 고치지 않는다.** 고치면 prompt_v0 가 달라져 기존 런과 비교가 깨진다.
        # 덮어쓰기는 소비 지점인 여기서만 한다.
        tgt_spaced = (target_is_spaced(args.tgt_lang) if args.tgt_spaced is None
                      else args.tgt_spaced == "yes")
        log(f"[profile] {profile.get('source_language')} / 어순 {profile.get('word_order')} / "
            f"띄어쓰기 {spaced}(측정) / 타깃 띄어쓰기 {tgt_spaced}")

        targets = resolve_targets(args.tgt_langs or DEFAULT_TARGET_POOL, args.src_lang)
        if not targets:
            log("[stop] 검증 타깃이 없다 — --tgt-langs 확인")
            return 2
        # 비교군(무분절·기계분절)과 곡선 대표값이 쓰는 번역기. **첫 타깃**을 대표로 쓰고
        # 캐시 파일도 타깃별로 나눈다 — 하나로 합치면 gtx 결과가 언어별로 섞인다.
        rep_tgt = targets[0]
        rep_code = args.tgt_code or to_lang_code(rep_tgt)
        tr_cache = JsonCache(run_dir / "cache" / f"translate_{rep_code}.json")
        translator = GoogleTranslator(
            tgt_code=rep_code, cache=tr_cache, workers=min(args.workers, 4),
            use_context=not args.no_google_context, backend=args.translate_backend)
        # **백엔드가 id 에 들어간다.** gtx 와 v2 의 번역문은 같지 않다 (기존 캐시 18건
        # 재번역 대조에서 일치 0/18). id 에 안 남기면 다른 자로 잰 점수가 조용히 섞인다.
        translator_id = (f"google:{translator.backend}:{translator.tgt_code}"
                         f":ctx={translator.use_context}")
        log(f"[translator] {translator_id} (대표 타깃 {rep_tgt})")
        log(f"[targets] {len(targets)}개: {', '.join(targets)}  "
            f"(목적함수 = 타깃별 z-정규화 effective 의 평균)")

        # **`--final-only` 는 config 를 덮어쓰지 않는다.** 인자를 안 준 항목이 기본값으로
        # 채워져 저장되면 **그 런을 만든 설정 기록이 사라진다** — ja/run01 실측:
        # `revision_candidates 3 -> 1`, `budget 10 -> 4` 로 덮였다 (당시 `v0_probe` 포함).
        if not (args.final_only and (run_dir / "config.json").exists()):
            (run_dir / "config.json").write_text(json.dumps({
            **vars(args), "t_grid": t_grid, "final_t_grid": final_grid, "main_t": main_t,
            "translator_id": translator_id, "tgt_spaced": tgt_spaced,
            "adequacy_model": metrics.QE_CHECKPOINTS[args.adequacy_backend],
            "consistency_model": getattr(consistency, "model_name",
                                         args.consistency_backend),
            "judge_prompt_hash": JsonCache.key(agents.JUDGE_SYSTEM),
            "judge_model": args.judge_model or args.model,
            "min_boundaries_per": t_floor,   # [Output Rules] 에 박히는 값 = 검증기 요건
            "curve_min_t": coverage_t,
            "coverage_required": not args.no_coverage_rule,
            "min_gap_ms": MIN_GAP_MS,
            "units_per_sec_source": rate_source,
            "t_floor": t_floor,     # max(2, min_gap+1, ceil(1.25*min_gap)). 아래 T 는 포화
            "t_grid_derived": derived,           # 격자를 min_gap 에서 유도했는가
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # 타깃별 컨텍스트. 번역기·바닥·띄어쓰기가 타깃마다 다르고, adequacy/NLI 백엔드는
        # 다국어라 공유한다. 캐시는 타깃별 파일로 분리해야 gtx 결과가 안 섞인다.
        # 대표 타깃의 번역기는 위에서 이미 만들었으므로 그대로 재사용한다.
        # (종전 `if args.translator == "google"` — `--translator` 는 A5 에서 Google 하나로
        #  줄이며 사라진 인자다. 조건만 남아 시작하자마자 AttributeError 로 죽었다.)
        _tr_cache: dict = {rep_code: translator}

        def make_ctx(tgt: str):
            code = args.tgt_code if len(targets) == 1 else None
            code = code or to_lang_code(tgt)
            if code not in _tr_cache:
                _tr_cache[code] = GoogleTranslator(
                    tgt_code=code,
                    cache=JsonCache(run_dir / "cache" / f"translate_{code}.json"),
                    workers=args.workers,
                    use_context=not args.no_google_context)
            return (_tr_cache[code], adequacy, consistency, contradiction,
                    target_is_spaced(tgt))

        _kw = dict(gw=gw, spaced=spaced, seg_cache=seg_cache, workers=args.workers,
                   trailing_punct=trailing_punct,
                   require_coverage=not args.no_coverage_rule, coverage_t=t_floor,
                   reasoning_effort=args.seg_reasoning_effort,
                   batch_size=args.batch_size, min_gap=args.min_gap,
                   skip_translation_below=args.skip_translation_below)

        # **부트스트랩에서는 이 가드를 면제한다.** 가드의 목적은 "쓰레기를 번역하느라 돈
        # 쓰지 말라"인데, 기준선이 아직 없는 첫 평가에서 걸리면 **점수 자체가 안 나와**
        # 루프가 개선할 신호를 못 얻는다 — zh 실측: fmt 0.60 < 0.95 로 채점이 통째로
        # 스킵돼 `score=0.0000`, 개정 조향 불가 (MULTI2EN_DATASET.md §5.3-2).
        # 새 언어의 v0 는 en 만큼 형식을 지키지 못하는 것이 정상이므로, 한 번은 재고
        # 나서 판단한다.
        # **절대 임계값이 아니라 달성 가능한 값 대비로 잰다.** 0.95 는 en 에서 관찰된
        # 수준이라 새 언어에는 구조적으로 높다 — ja 실측 dev fmt 는 0.87~0.92 로, iter 0 만
        # 면제해도 iter 1 부터 다시 걸려 루프가 멈춘다. 기준선의 80% 로 두면 "붕괴한
        # 개정"만 걸러지고 정상 범위는 통과한다 (ja: 0.87 × 0.8 = 0.70).
        def _set_skip_guard(baseline_fmt: float | None) -> None:
            _kw["skip_translation_below"] = (
                0.0 if baseline_fmt is None
                else min(args.skip_translation_below, 0.8 * baseline_fmt))

        # **z 기준선은 분할별로 한 번 정해 끝까지 고정한다.** 평가마다 다시 잡으면
        # 채택 판정이 프롬프트가 아니라 기준선 이동을 재게 된다 (`_zmix` 참고).
        # 디스크에 남겨 재개(resume) 런도 같은 자를 쓰게 한다.
        zbase_path = run_dir / "z_baseline.json"
        zbase_all: dict = (json.loads(zbase_path.read_text(encoding="utf-8"))
                           if zbase_path.exists() else {})

        def run_eval(prompt_: str, split_sentences, grid, split: str = "train"):
            zb = zbase_all.setdefault(split, {}) if len(targets) > 1 else None
            norm: list[dict] = []
            rows_, m_, viol_, per_, per_rows_ = evaluate_multi(
                targets, make_ctx, prompt_, split_sentences, grid, zbase=zb,
                norm_sink=norm, **_kw)
            run_eval.last_normalizations = norm
            if zb is not None:
                zbase_path.write_text(json.dumps(zbase_all, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
            run_eval.last_per_target = per_
            run_eval.last_per_rows = per_rows_
            return rows_, m_, viol_

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
                min_t = t_floor
                need = (lambda t: coverage_need(t, min_t, spaced, args.min_gap)
                        ) if not args.no_coverage_rule else (lambda t: None)
                segment_batch(
                    gw, pr, texts, cache=seg_cache, workers=args.workers,
                    validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct,
                                                        True, need(t)),
                    normalize_fn=lambda t, o: normalize_tags(o, spaced, trailing_punct,
                                                             min_gap=args.min_gap),
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

        def select_prompt(cands: list[str], select_n: int, tag_: str) -> list[str]:
            """후보 여러 개 중 하나를 고른다. **train 으로 고른다 — dev 는 안 쓴다.**

            **왜 train 인가 — dev 를 두 번 쓰면 안 된다.** 종전에는 `dev[:40]` 로 상위 2개를
            추리고 그 둘을 **dev 전체**로 재채점해 argmax 를 골랐다. 그리고 바로 그 dev 로
            채택 판정을 했다 — 고르는 데 쓴 자로 그 선택이 옳았는지 판정한 셈이다.

            그 결과 기준선과 새 개정본이 **양쪽 다 부풀어 있었다.** 후보 2개 중 최대값은
            잡음 σ 일 때 평균 `0.564σ` 만큼 위로 뜨는데, dev 60 에서 σ≈0.011 이라 편향이
            **≈0.006** 이다. 프롬프트 간 실제 차이가 0.005 규모이므로 **재려는 효과와 같은
            크기**이고, 두 편향이 상쇄되는지 겹치는지 알 수 없어 쌍체 Δ 의 기대값이 정의되지
            않았다. 실측이 이 그림과 맞는다 — 쌍체 Δ 14회 중 음수 9회, 런 18개 중 13개가
            v0 이후 아무것도 채택 못 함.

            train 으로 옮기면 역할이 갈린다: **train = 고르는 곳(게이트·Critic 사례·후보 선별),
            dev = 판정하는 곳.** train 에 과적합되지만 그게 train 의 일이고, 판정은 깨끗한
            dev 가 한다.

            **2단계를 1단계로 합친다.** 2단계는 "dev 가 비싸니 아껴 쓰자"는 절충이었는데
            dev 를 안 쓰면 필요 없다. 채점 비용도 같다 — 종전 `3후보×probe40 + 2후보×dev60
            = 240문장` 이고 지금은 `3후보×train80 = 240문장` 이다.

            선별 정확도는 **문장 수만 따른다** (test 100문장·변종 6종 실측: 20문장 1위 적중
            36%, 40문장 58%, 60문장 76%). 같은 예산으로 한 단계에 몰면 표본이 커져 정확도가
            오른다.

            **쌍체로 바꿔도 순위는 안 바뀐다** — 후보들이 같은 문장 집합을 쓰면 기준값이
            공통 상수라 argmax 에서 소거된다 (실측 소수점까지 동일). 쌍체의 이득은
            오차막대이지 순위가 아니다.
            """
            if len(cands) <= 1:
                return list(cands)
            pool = splits["train"]
            sel = pool[:select_n] if select_n and select_n < len(pool) else pool
            # **분절을 먼저 한 풀에 몰아 캐시를 채운다.** 후보를 순차로 `run_eval` 하면
            # 후보마다 select/batch_size 개의 콜만 던지게 되어 워커를 못 채운다
            # — run04 실측 평균 동시 실행 3.03 / 최대 7 (워커 8). 분절 1콜이 112초라
            # 그 직렬화가 곧 경과 시간이다. 후보 전체의 분절을 한 번에 던지면 동시 폭이
            # 후보 수만큼 늘고, 이후 `run_eval` 은 전부 캐시 히트로 지나간다.
            prewarm(cands, sel)
            scored = []
            for i, c in enumerate(cands):
                _r, _m, _v = run_eval(c, sel, t_grid, "train")
                sc_i = metrics.score(_m)
                scored.append((sc_i, i, c))
                log(f"[{tag_}] 후보 {i}: {len(c)}자 train({len(sel)}) "
                    f"score={sc_i:.4f} fmt={_m.format_pass_rate:.2f}")
            scored.sort(key=lambda x: -x[0])
            log(f"[{tag_}] 후보 {scored[0][1]} 채택 (score={scored[0][0]:.4f}) — dev 미사용")
            # **순위 전체를 돌려준다.** 1위가 분량 관문에 걸리면 2·3위를 시도해야 한다 —
            # 이미 채점한 후보를 버리고 이터레이션을 통째로 헛돌리는 것보다 낫다.
            return [c for _, _, c in scored]

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
            # 좌우된다.** 후보를 만들어 고르면 그 분산을 산다 — 고르는 자는 **train** 이다
            # (`select_prompt` 참조: dev 는 채택 판정 전용).
            candidates: list[str] = []
            for attempt in range(3 * max(1, args.v0_candidates)):
                if len(candidates) >= args.v0_candidates:
                    break
                cand = profiler.initial_prompt(profile, None, spaced, t_floor,
                                               args.min_gap, measured=measured)
                missing = agents.check_skeleton(cand)
                if missing:
                    log(f"[profiler] 골격 누락 {missing} — 재시도 {attempt + 1}")
                    continue
                # 분절은 소스 쪽 문제다 — 프롬프트가 타깃 언어에 기대면 다른 타깃에
                # 재사용할 수 없고, 측정되지 않은 언어 지식이 섞인다.
                tl = agents.check_target_agnostic(cand, args.src_lang, targets)
                if tl:
                    log(f"[profiler] 타깃 종속 — 재시도 {attempt + 1}: {'; '.join(tl)}")
                    continue
                candidates.append(cand)
            if not candidates:
                log(f"[stop] prompt_v0 가 골격 미달만 반복 — 모델/max_tokens 확인")
                return 2

            for k, cand in enumerate(candidates):
                (run_dir / f"prompt_v0_cand{k}.txt").write_text(cand, encoding="utf-8")
            prompt = select_prompt(candidates, args.select_n, "profiler")[0]
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
        best = {"version": 0, "prompt": prompt, "train_score": None,
                "dev_score": None, "fmt": None}
        best_ctx: dict = {}
        start_it = 0
        best_critique: dict | None = None
        # 고착 방지 핸들 — 직전 개정이 **어느 섹션을 고쳤는지**. focus 라벨을 쓰던
        # 것을 바꿨다: 라벨이 달라도 같은 섹션을 고치는 경우가 8/30(27%)이라
        # 라벨로는 반복을 못 잡았다.
        last_sections: str | None = None
        stale = 0
        floor_fn = None          # contradiction 잡음 바닥 — 첫 평가 후 1회 측정

        # ── `--resume` — 중단 지점부터 이어간다 ─────────────────────────
        # 복원해야 하는 것은 넷이다: `history`(PE 가 읽는 시도 이력), `best`(비교 기준),
        # `best_ctx["dev_rows"]`(쌍체 Δ 의 기준선 — **이게 없으면 채택 판정이 무너진다**),
        # 그리고 다음 이터레이션 번호. 앞의 셋은 전부 디스크에 있다.
        #
        # `best_ctx` 의 나머지(rows/metrics/violations/judgements)는 Critic 입력이라
        # 없어도 죽지 않는다 — 첫 이터레이션이 새로 만든 것을 쓴다. 그래서 저장된 것만
        # 채우고 없으면 비운다.
        if args.resume:
            hist_path = run_dir / "history.json"
            best_path = run_dir / "best_prompt.txt"
            if not hist_path.exists() or not best_path.exists():
                log(f"[stop] --resume 인데 {run_dir} 에 history.json / best_prompt.txt 가 없다")
                return 2
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            start_it = len(history)
            best["prompt"] = best_path.read_text(encoding="utf-8")
            # **평가할 프롬프트는 best 가 아니라 "대기 중인 개정본"이다.** 정상 흐름에서
            # iter N 이 재는 것은 iter N-1 이 만든 개정본이고, best 는 비교 기준일 뿐이다.
            # 종전 재개는 `prompt = best` 로 놓아 그 개정본을 버렸다 — run07 iter2 가
            # v0 를 v0 와 비교해 Δ 정확히 0 / 변경 0문장이 됐고, 근거 없이 오른 stale 이
            # patience 조기 종료를 앞당겼다 (6이터 예정 → 4이터).
            #
            # 두 곳을 본다. 이터 **도중** 크래시면 `iter_NN/prompt.txt` 가 이미 있고,
            # 이터 **사이** 크래시면 `next_prompt.txt` 만 있다.
            pending = run_dir / f"iter_{start_it:02d}" / "prompt.txt"
            nxt = run_dir / "next_prompt.txt"
            if pending.exists():
                prompt = pending.read_text(encoding="utf-8")
                src_ = f"iter_{start_it:02d}/prompt.txt"
            elif nxt.exists():
                prompt = nxt.read_text(encoding="utf-8")
                src_ = "next_prompt.txt"
            else:
                prompt = best["prompt"]
                src_ = "best_prompt.txt (대기 개정본 없음)"
            done = [h for h in history if h.get("adopted")]
            if done:
                last = done[-1]
                best["version"] = last["version"]
                best["train_score"] = last.get("score_train")
                best["dev_score"] = last.get("score_dev")
                best["fmt"] = (last.get("train") or {}).get("format_pass_rate")
                d = run_dir / f"iter_{last['version']:02d}"
                for k, f in (("rows", "train_rows.json"), ("dev_rows", "dev_rows.json"),
                             ("violations", "violations.json"),
                             ("judgements", "judgements.json"),
                             ("priority_audit", "priority_audit.json")):
                    if (d / f).exists():
                        best_ctx[k] = json.loads((d / f).read_text(encoding="utf-8"))
                mj = d / "metrics.json"
                if mj.exists():
                    best_ctx["metrics"] = json.loads(mj.read_text(encoding="utf-8")).get("train")
                stale = len(history) - 1 - history.index(last)
            else:
                stale = len(history)
            if "dev_rows" not in best_ctx:
                log("[resume] 경고: best 의 dev_rows 를 못 찾았다 — 다음 이터레이션은 "
                    "쌍체 Δ 없이 점수 비교로 판정한다")
            log(f"[resume] iter {start_it} 부터 재개. best=v{best['version']} "
                f"dev={best['dev_score']} stale={stale} 이력 {len(history)}건")
            log(f"[resume] 평가할 프롬프트 {len(prompt)}자 <- {src_}"
                + ("  (best 와 동일 — 개정본을 잃었다)"
                   if prompt == best["prompt"] and pending.exists() else ""))
            if start_it >= args.iterations:
                log(f"[resume] 이미 {start_it}회 돌았다 (--iterations {args.iterations}) — "
                    f"최종 평가로 넘어간다")

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


        def end_timer(timer, it: int) -> None:
            """열린 단계를 닫고 한 줄로 요약한다. 이터레이션의 모든 출구에서 부른다."""
            timer.mark(None)
            if timer.rows:
                tot = timer.rows[-1]["at_sec"] + timer.rows[-1]["sec"]
                log(f"[iter {it}] 소요 {tot:.0f}초 = "
                    + " + ".join(f"{r['stage']} {r['sec']:.0f}" for r in timer.rows))

        for it in range(start_it, args.iterations):
            it_dir = run_dir / f"iter_{it:02d}"
            it_dir.mkdir(parents=True, exist_ok=True)
            (it_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            timer = StageTimer(it_dir / "timing.json")
            timer.mark("train_eval")

            _set_skip_guard(best.get("fmt") if best["train_score"] is not None else None)
            batch = splits["train"]
            if len(batch) > args.train:
                batch = random.Random(20260808 + it).sample(batch, args.train)
            rows, m, viol = run_eval(prompt, batch, t_grid, "train")
            sc = metrics.score(m)

            # ── A7 Judge — 주 작동점에서만 (비용) ────────────────────────
            # **점수를 안 낸다.** 모순이 가장 큰 경계에 "왜·어디로" 를 붙일 뿐이고,
            # 그 설명은 Critic 케이스에만 실린다 (`agents.judge_top_contra` 참조).
            timer.mark("judge")
            judgements: list[dict] = []
            if judge is not None and m.by_T:
                judgements = judge_distributed(
                    judge, getattr(run_eval, 'last_per_rows', {'_': rows}),
                    main_t, args.judge_frac, workers=args.workers)
                (it_dir / "judgements.json").write_text(
                    json.dumps(judgements, ensure_ascii=False, indent=2), encoding="utf-8")

            timer.mark("rank_audit")
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
            # **결정론적 수정은 위반이 아니라 기록이다.** 검증기를 통과시키려고 고친
            # 것이므로 `violations.json` 에 넣으면 "고쳐졌는데 결함"이 되고, 안 남기면
            # 규칙이 틀렸을 때 아무 흔적이 없다 (zh 여는 따옴표 13건이 그렇게 묻혔다).
            norm = getattr(run_eval, "last_normalizations", None)
            if norm:
                (it_dir / "normalizations.json").write_text(
                    json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
                import collections as _c2
                log(f"[iter {it}] 결정론 수정 {len(norm)}건 "
                    f"{dict(_c2.Counter(x['kind'] for x in norm))}")
            log(f"[iter {it}] {fmt_metrics(m, 'train')}")

            # 쌍체 차이. dev 가 고정 집합이라 `mean(new) − mean(old) = mean(new − old)` 가
            # 항등이고 점추정은 절대값 비교와 동일하다. 얻는 것은 **오차막대와 유효 표본**:
            # 분절이 안 바뀐 문장은 차이가 정확히 0 이라 분산에 기여하지 않으므로
            # se 가 훨씬 작게 나오고, `n_changed` 가 실제로 판정에 참여한 문장 수를 알려준다.
            # 채택 판정이 이 오차막대를 쓴다 (`Δ > adopt_se_mult·se`) — run01~03 실측에서
            # 점 비교는 오차막대 안 잡음(예: −0.013±0.014)까지 채택 후보로 만들었다.
            # **dev 실행 게이트를 없앴다 — 아무것도 안 막고 있었다.**
            #
            # 게이트는 "train 에서 확실히 나쁘면 dev 를 아끼자" 는 장치였다. 기록된
            # train Δ 19건에 문턱을 걸어 보면 **차단 0/19 (0%)** 다 (참 차단율 95% 상한
            # ~15%). 즉 기대 절감은 $0 ~ $0.45/런인데, 게이트를 굴리려면 매 이터레이션
            # best 를 같은 배치에 다시 평가해야 했다 — ~$0.10 × 6이터 = **$0.60/런**.
            # 아끼는 것보다 쓰는 것이 크거나 비슷하다.
            #
            # 그리고 게이트에는 **순환**이 있었다. `select_prompt` 가 후보 K개 중
            # train 최고를 고르는데, `--train-pool` 을 안 주면 그 train 이 게이트 배치와
            # **같은 문장**이다 (설정 26개 중 14개가 겹침 100%). 고른 자로 그 선택을
            # 재검사한 셈 — A9 에서 dev 를 두 번 쓰던 것과 같은 모양이다. 승자 이득을
            # 실측하면 후보 그룹 18개(K=3)에서 최대값−평균 중앙 **0.0118**, 이론값
            # 0.846·sd = 0.0109 와 일치한다. 게이트 여유(1×se) 중앙이 0.0156 이므로
            # 편향이 여유의 70% 를 먹어 실효 문턱이 `−1.7·se` 였다. 다만 두 문턱 어느
            # 쪽으로도 차단이 0/19 라 **바뀌는 결정은 없었다.**
            #
            # 게이트를 지우면 순환도 함께 사라진다 — 선별 결과를 재검사하는 구조가
            # 없어지기 때문이다. 파국적 개정(포맷 붕괴)은 `--skip-translation-below` 와
            # 골격 검사가 이미 잡는다. 가르는 일은 **dev 가 한다.**
            #
            # **채택 판정은 z 축으로 한다 — 다만 근거가 약하다.** 타깃별 분산이 다르다는
            # 것(문장별 sd 0.104~0.195)은 맞지만, 원값 평균에서 **신호에 대한 가중치는
            # 모든 타깃이 정확히 1/K 로 같다.** sd 는 잡음에만 영향을 준다. z 는 오히려
            # 가중치를 **분산 작은 타깃 쪽으로** 옮긴다 (δ 하나가 z 를 δ/(K·sd) 만큼
            # 움직이므로). 표준화 효과크기를 재겠다는 선택이지 "지배를 없앤다"가 아니다.
            #
            # 검출력 차이를 공분산으로 계산하면 (다타깃 dev 2런):
            #   균일 효과 δ      t(z)/t(raw) = 1.02~1.05
            #   sd 비례 효과     t(z)/t(raw) = 0.99
            # **±3%, 사실상 무승부다.** sd 비가 1.9배밖에 안 돼 두 가중치가 거의 같다.
            # 종전 주석이 근거로 든 run05 비교는 raw t=−0.10 / z t≈+0.12 인데, 애초에
            # 효과가 없는(t≈0) 쌍이라 잡음끼리 비교한 값이다.
            #
            # **양쪽 다 행에 기록되므로**(`effective`, `effective_z`) 다음 런의 Δ 분포를
            # 보고 정하면 된다. 보고값(`effective`)은 원값 평균이다.
            # z 기준선은 분할별로 고정돼 있어야 한다 — `_zmix` 참고.
            delta_key = "effective_z" if len(targets) > 1 else "effective"
            timer.mark("dev_eval")
            dev_rows, dev_m, dev_viol = run_eval(prompt, splits["dev"], t_grid, "dev")
            dev_score = metrics.score(dev_m)
            dev_delta = (metrics.paired_delta(dev_rows, best_ctx["dev_rows"],
                                              t_grid, delta_key)
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
                        "train_score": sc, "dev_score": dev_score,
                        "fmt": m.format_pass_rate}
                (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
                best_ctx = {"rows": rows, "dev_rows": dev_rows, "metrics": m.to_dict(),
                            "violations": viol, "judgements": judgements,
                            "priority_audit": audit}
                best_critique = None      # 새 best — 비평을 다시 받아야 한다
                adopted = True
                stale = 0
            elif dev_delta and dev_delta.get("n_changed") == 0:
                # **아무것도 안 잰 이터레이션은 실패로 세지 않는다.** 직전 개정이
                # 거부돼 프롬프트가 그대로면 분절이 한 문장도 안 바뀌고 Δ 가 정확히
                # 0 이 된다 (run08 iter3: Δ 0.00000, 변경 0/265). 그건 "개선이 없었다"
                # 는 증거가 아니라 **측정이 없었다**는 뜻이므로 patience 를 깎으면 안
                # 된다 — run07 에서 이 종류의 가짜 증가가 조기 종료를 앞당겼다.
                log(f"[iter {it}] 분절이 한 문장도 안 바뀌었다 — stale 유지 ({stale})")
            else:
                stale += 1

            (it_dir / "metrics.json").write_text(json.dumps({
                "train": m.to_dict(),
                "dev": dev_m.to_dict() if dev_m else None,
                "score_train": round(sc, 4),
                "score_dev": round(dev_score, 4) if dev_score is not None else None,
                "paired_dev": dev_delta,
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
                end_timer(timer, it)
                break
            if stale >= args.patience:
                end_timer(timer, it)
                log(f"[stop] dev 개선 없음 {stale}회 연속 — 조기 종료")
                break

            # 에이전트 호출이 실패해도 런 전체를 버리지 않는다.
            ctx = best_ctx or {"rows": rows, "metrics": m.to_dict(),
                               "violations": viol, "judgements": judgements}
            try:
                # ── A8 Critic ───────────────────────────────────────────
                # 비평 대상과 개정 대상은 반드시 같은 프롬프트여야 한다.
                timer.mark("critic")
                if best_critique is None:
                    cases = agents.select_cases(ctx["rows"], main_t, ctx.get("judgements"))
                    best_critique = critic.review(
                        cases, ctx["metrics"], ctx["violations"],
                        avoid=last_sections if stale >= 2 else None,
                        priority_audit=ctx.get("priority_audit"),
                        judgements=ctx.get("judgements"))
                critique = best_critique
                # 캐시가 고착 방지를 우회하지 않도록 aggregate 만 다시 계산한다 (LLM 없음).
                if stale >= 2 and last_sections:
                    critique = {**critique, "aggregate": agents.summarize_critique(
                        critique.get("cases") or [], ctx["metrics"],
                        critique.get("summary"), avoid=last_sections,
                        priority_audit=ctx.get("priority_audit"),
                        judgements=ctx.get("judgements"))}
                (it_dir / "critique.json").write_text(
                    json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")
                agg = critique.get("aggregate", {})
                log(f"[iter {it}] critic dominant={agg.get('dominant_error')}"
                    + (f" | {agg['stuck_hint'][:60]}" if agg.get("stuck_hint") else ""))

                # ── A9 Prompt Engineer ──────────────────────────────────
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
                # **후보는 규칙 부분집합이 아니라 표현 차이로 나뉜다.** 종전에는 규칙을
                # 하나씩 나눠 실어 "어느 규칙이 도움됐나" 를 가리려 했는데, 한 규칙짜리
                # 개정의 |Δ| 중앙이 0.00505 이고 dev 225 검출 한계가 0.0104 라 **전체가
                # 도움됐는지조차 못 잰다.** 그 상태에서 규칙별 신용 배분은 환상이다.
                # 첫 후보는 자유 개정(PE 가 critique 을 보고 판단), 나머지는 제안 전부를
                # 각자 구현한다. 크기가 넘치면 압축기가 받는다.
                jobs = [None] + [hints] * max(0, args.revision_candidates - 1) if hints \
                    else [None] * max(1, args.revision_candidates)
                jobs = jobs[:max(1, args.revision_candidates)]

                # 개정 한 번의 **분량 예산** — 런 전체 천장 하나뿐이다.
                # 걸음 크기 상한(직전 best 대비)도 따로 뒀다가 뺐다: 적용된 개정 42건의
                # 이터레이션 간 배수가 중앙 1.03 / p90 1.12 / 최대 1.29 라, 어떤 상수를
                # 놓아도 v0 천장보다 먼저 걸릴 일이 없다.
                size_budget = int(prompt_v0_len * args.max_prompt_growth)

                def make(hint):
                    try:
                        rv = engineer.revise(best["prompt"], critique, history, profile,
                                             t_grid, only_rules=hint,
                                             size_budget=size_budget,
                                             measured=measured)
                    except BudgetExceeded:
                        raise
                    except Exception as e:                      # 후보 하나 실패로 안 죽는다
                        log(f"[iter {it}] 개정 후보 실패: {e}")
                        return None
                    return rv

                timer.mark("prompt_engineer")
                raw_cands = []
                for hint in jobs:
                    rv = make(hint)
                    if not rv: continue
                    pr = rv.get("prompt", "")
                    if pr and not agents.check_skeleton(pr):
                        raw_cands.append((pr, rv))

                # 남은 하드 게이트는 **구조** 하나다 — 섹션 추가/삭제 금지.
                # 분량 위반은 여기서 거르지 않는다: 종전에는 후보 전원이 분량으로
                # 탈락해 "0/3 통과 -> 측정 불가 -> 채택 불가" 교착이 생겼는데,
                # 분량은 거부할 일이 아니라 압축기가 줄일 일이라서 그렇다. 선택이
                # 끝난 뒤 한 번만 줄인다 (압축 호출 K번 -> 1번).
                cands = [(pr, rv) for pr, rv in raw_cands
                         if not agents.check_revision(best["prompt"], pr)]

                _cands2 = []
                for pr, rv in cands:
                    tl = agents.check_target_agnostic(pr, args.src_lang, targets)
                    if tl:
                        log(f"[iter {it}] 개정 후보 타깃 종속 — 거부: {'; '.join(tl)}")
                        continue
                    _cands2.append((pr, rv))
                cands = _cands2
                timer.mark("select_prompt")
                # 점수 순으로 줄 세운다 — 1위가 분량 관문에 걸리면 아래에서 차례로 시도한다.
                ranked: list[dict] = []
                if len(cands) > 1:
                    order = select_prompt([c[0] for c in cands], args.select_n,
                                          f"iter {it} 개정")
                    ranked = [next(rv for pr, rv in cands if pr == pr_) for pr_ in order]
                    revised = ranked[0]
                elif cands:
                    ranked = [cands[0][1]]
                    revised = cands[0][1]
                else:
                    # 여기까지 전멸하려면 후보 전원이 섹션을 추가/삭제했거나 타깃 언어를
                    # 박아 넣었다는 뜻이다 (실측 로그 26개에서 각각 0건/드묾). 분량 때문에
                    # 비는 경우는 이제 없으므로, 재호출로 돈을 더 쓰지 않고 넘어간다.
                    revised = {"prompt": ""}
                log(f"[iter {it}] 개정 후보 {len(cands)}/{len(jobs)} 통과")

                # **분량 관문에 걸리면 다음 후보를 시도한다.** 종전에는 1위가 걸리면
                # 그 이터레이션의 개정이 통째로 없어졌고, 프롬프트가 안 바뀌니 **다음
                # 이터가 같은 것을 다시 쟀다** — run08 에서 Δ 정확히 0 / 변경 0/265 로
                # 이터레이션 두 개가 헛돌았다. 후보는 이미 채점해 두었으므로 아래로
                # 내려가는 데 드는 추가 비용은 압축 호출 한 번뿐이다.
                #
                # **바뀐 섹션은 diff 로 센다 — PE 자기신고를 쓰지 않는다**
                # (`agents.changed_sections` 참조: 저장분 36건 중 14건 불일치).
                # 압축 *전* 프롬프트로 재는 것이 맞다 — 압축기가 보호해야 하는 것은
                # 이번 이터레이션이 넣은 변경이고, 압축은 그 뒤에 일어난다.
                new_prompt, sections_changed, over = "", [], False
                for _rank, _rv in enumerate(ranked or [revised]):
                    cand_p = _rv.get("prompt", "")
                    if not cand_p:
                        continue
                    cand_sec = agents.changed_sections(best["prompt"], cand_p)
                    if len(cand_p) > size_budget:
                        over = True
                        log(f"[iter {it}] 후보 {_rank}: 개정본 {len(cand_p)}자 > 예산 "
                            f"{size_budget}자 — 압축")
                        packed = compressor.compress(cand_p, size_budget, cand_sec)
                        if (packed and not agents.check_skeleton(packed)
                                and len(packed) <= size_budget
                                and not agents.check_revision(best["prompt"], packed)):
                            log(f"[iter {it}] 압축 성공 {len(cand_p)} -> {len(packed)}자")
                            cand_p = packed
                            cand_sec = agents.changed_sections(best["prompt"], cand_p)
                        else:
                            log(f"[iter {it}] 후보 {_rank} 압축 실패"
                                f"({len(packed or '')}자 / 예산 {size_budget}자)"
                                + ("  — 다음 후보 시도" if _rank + 1 < len(ranked) else ""))
                            continue
                    new_prompt, sections_changed, revised = cand_p, cand_sec, _rv
                    break
                missing = agents.check_skeleton(new_prompt) if new_prompt else []
                if not new_prompt:
                    log(f"[iter {it}] 개정 없음 — 이전 프롬프트 유지")
                elif missing or len(new_prompt) < 500:
                    log(f"[iter {it}] 개정 프롬프트 골격 누락 {missing} — 이전 프롬프트 유지")
                else:
                    prompt = new_prompt
                (it_dir / "changelog.json").write_text(json.dumps({
                    "sections_changed": sections_changed,
                    "sections_reported": revised.get("sections_changed"),
                    "changelog": revised.get("changelog"),
                    "size_budget": size_budget,
                    "over_budget": bool(over),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                history[-1]["next_changelog"] = revised.get("changelog")
                # 고착 방지 핸들 갱신 — **다음 이터레이션이 이 섹션을 다시 고치지 않도록.**
                # 채택 여부와 무관하게 "직전에 무엇을 건드렸나" 를 기억한다. 채택되면
                # stale 이 0 으로 돌아가 avoid 가 안 걸리므로, 실제로 쓰이는 건 실패했을
                # 때뿐이다.
                # **다음 이터가 평가할 프롬프트를 디스크에 남긴다.** `--resume` 은 이걸
                # 읽어야 한다 — 안 그러면 iter N-1 이 만든 개정본을 잃고 best 를 다시
                # 평가하게 된다 (run07 iter2 실측: v0 를 v0 와 비교해 Δ 정확히 0,
                # 변경 0/265, stale 만 근거 없이 올라 조기 종료를 앞당겼다).
                (run_dir / "next_prompt.txt").write_text(prompt, encoding="utf-8")
                last_sections = ", ".join(sections_changed) if sections_changed else None
                _rep = revised.get("sections_changed") or []
                log(f"[iter {it}] 개정: {sections_changed}"
                    + (f"  (PE 신고 {_rep} — 불일치)" if sorted(_rep) != sorted(sections_changed)
                       else ""))
            except BudgetExceeded:
                raise
            except Exception as e:
                log(f"[iter {it}] 에이전트 실패 — 루프 중단하고 최종 평가로 넘어간다: {e}")
                end_timer(timer, it)
                break
            end_timer(timer, it)

        # ── 최종 test 평가 (전체 격자) ───────────────────────────────────
        log(f"[final] test 평가 — 격자 {final_grid} (루프가 한 번도 보지 않은 데이터)")
        final_timer = StageTimer(run_dir / "timing_final.json")
        final_timer.mark("test_eval")
        test_rows, test_m, test_viol = run_eval(best["prompt"], splits["test"],
                                                final_grid, "test")
        final_timer.mark(None)
        log(f"[final] test 평가 {final_timer.rows[0]['sec']:.0f}초")
        test_viol, test_viol_1st = split_first_pass(test_viol)
        if test_viol_1st:
            (run_dir / "test_violations_first_pass.json").write_text(
                json.dumps(test_viol_1st, ensure_ascii=False, indent=2), encoding="utf-8")
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
                              final_grid, main_t, translator_id,
                              per_target=getattr(run_eval, "last_per_target", None),
                              targets=targets)
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


# ── 비교군 ──────────────────────────────────────────────────────────────

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
            sp.laal_words, sp.k, [0] * len(texts),
            effective_ent_scores=sp.effective_ent,
            contradiction_ent_scores=sp.contradiction_ent).to_dict()
    return out


def split_first_pass(viol: list[dict]) -> tuple[list[dict], list[dict]]:
    """재시도 후 최종 위반과 1차(재시도 전) 위반을 가른다."""
    final = [v for v in viol if not v.get("first_pass")]
    first = [v for v in viol if v.get("first_pass")]
    return final, first


def build_report(args, run_dir, profile, measured, history, best, test_m, test_viol,
                 usage, baselines, t_grid, final_grid, main_t, translator_id,
                 per_target=None, targets=None) -> str:
    lines = [
        f"# 자동 분절 프롬프트 루프 결과 (v2) — {args.src_lang} → "
        + (", ".join(targets) if targets and len(targets) > 1 else args.tgt_lang),
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
           else f"**xlmr-anli** (`{metrics.NLI_MODEL}`)"),
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
        "| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | eff p10 ↑ | eff min ↑ | eff (ent) | adequacy | contradiction ↓ | contra (ent) | consistency | k | 부족 경계 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(test_m.by_T, key=int):
        s = test_m.by_T[k]
        lines.append(f"| {k} | {s.laal_words:.2f} | **{_cell(s.effective, '.4f')}** | "
                     f"{_cell(s.effective_p10, '.4f')} | {_cell(s.effective_min, '.4f')} | "
                     f"{_cell(s.effective_ent, '.4f')} | "
                     f"{s.adequacy:.4f} | {_cell(s.contradiction, '.4f')} | "
                     f"{_cell(s.contradiction_ent, '.4f')} | "
                     f"{s.consistency:.4f} | "
                     f"{s.chunks_per_sentence:.2f} | {s.missing_boundaries:.2f} |")
    for name, b in baselines.items():
        lines.append(f"| {name} (노브 없음) | {b['laal_words']:.2f} | "
                     f"**{_cell(b['effective'], '.4f')}** | "
                     f"{_cell(b.get('effective_p10'), '.4f')} | {_cell(b.get('effective_min'), '.4f')} | "
                     f"{_cell(b.get('effective_ent'), '.4f')} | "
                     f"{b['adequacy']:.4f} | {_cell(b['contradiction'], '.4f')} | "
                     f"{_cell(b.get('contradiction_ent'), '.4f')} | "
                     f"{b['consistency']:.4f} | {b['chunks_per_sentence']:.2f} | — |")
    lines += [
        "",
        "> `eff p10` 은 하위 10% 지점, `eff min` 은 최악 문장이다. **평균과 같이 읽어야 한다** —",
        "> `effective` 평균은 소수의 나쁜 문장에 크게 끌린다 (실측: 100문장에서 최악 1문장을",
        "> 빼면 평균이 +0.0034 오르는데, 프롬프트 간 실제 차이가 0.003~0.007 이라 같은 크기다).",
        "> 평균과 p10 이 같이 오르면 진짜 개선이고, 평균만 오르고 p10 이 내리면 그 배치에 나쁜",
        "> 문장이 덜 걸린 것일 수 있다.",
        ">",
        "> `eff (ent)` / `contra (ent)` 는 **병기 지표**다 — 조기 방출을 모순 확률이 아니라",
        "> `1 − 함의 확률` 로 잰 값이고 **목적함수·채택 판정에는 안 들어간다**. 같은 NLI 호출에서",
        "> 세 라벨 확률이 함께 나오므로 추가 비용이 0 이다. 교체 여부를 이 로그로 판단한다",
        "> (오프라인 실측은 관문 여유 14배·검출력 1.6배로 그쪽이 유리하지만 표본이 얇다).",
    ]

    # 타깃별 곡선. 평균만으로는 어느 언어에서 무너지는지 안 보인다 — mdeberta-xnli 가
    # ko/zh/ja 에서만 곡선을 뒤집었던 것처럼 백엔드 결함이 언어별로 나타난다.
    if per_target and len(per_target) > 1:
        lines += ["", "### 타깃별 곡선 (effective)", "",
                  "| 타깃 | " + " | ".join(f"T={k}" for k in sorted(test_m.by_T, key=int)) + " |",
                  "|---" * (len(test_m.by_T) + 1) + "|"]
        for tgt, tm in per_target.items():
            cells = [f"{_cell(tm.by_T[k].effective, '.4f')}" if k in tm.by_T else "—"
                     for k in sorted(test_m.by_T, key=int)]
            lines.append(f"| {tgt} | " + " | ".join(cells) + " |")
        lines += ["",
                  "위 본표의 `effective` 는 이 타깃들의 **원값 평균**이다. 채택 판정은 "
                  "타깃별 z-정규화 후 평균(`effective_z`)으로 하는데, 타깃별 분산이 달라 "
                  "(문장별 sd 실측 0.147~0.196) 원값 차이는 분산 큰 타깃이 지배하기 "
                  "때문이다. z 기준선은 분할별로 고정돼 있다(`z_baseline.json`) — 그래야 "
                  "쌍체 Δ 가 기준선 이동이 아니라 실제 점수 차이를 잰다. "
                  "**언어 간 절대값 비교는 하지 말 것** — 지표 스케일이 다르다. "
                  "같은 언어 안에서 T 방향만 읽는다.",
                  "",
                  "본표의 `adequacy` / `contradiction` / `consistency` / `laal` 은 "
                  f"**대표 타깃({targets[0]}) 값**이다 — 조각·지연 정보는 타깃과 무관하고 "
                  "품질 하위 지표는 대표 타깃만 저장한다. 다른 타깃의 하위 지표가 "
                  "필요하면 `eval_prompt --tgt-lang` 으로 재채점한다 (번역 캐시가 "
                  "남아 있어 API 비용 없음)."]

    low_s = test_m.by_T.get(str(min(final_grid)))
    sp = low_s.rank_contra_spearman if low_s else None
    gap = low_s.rank_contra_gap if low_s else None
    lines += [
        "",
        f"- 포맷 통과율 {test_m.format_pass_rate:.4f} (재시도 없이 "
        f"{test_m.format_pass_rate_no_retry:.4f}), 위반 {len(test_viol)}건",
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
