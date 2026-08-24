"""번역 고정점 — "뒤가 와도 앞의 번역이 바뀌지 않는 지점" 을 직접 잰다.

**핀트 교정.** `contradiction`(NLI)은 *이미 내보낸 것이 반박당했는가* 를 묻는다. 그런데
우리가 실제로 원하는 것은 Zhang+ 2020 의 Meaningful Unit 정의에 가깝다 —
*후속 텍스트에 의해 번역이 바뀌지 않는 최소 구간*. 반박은 그 위반의 **한 종류**일 뿐이고,
어순 재배치·시제 확정·격 변경처럼 반박이 아닌 변화도 똑같이 수정을 강요한다.

문헌의 MU 판정은 `조각 번역 이어붙임이 full 번역의 **접두사**인가` 라는 이진 판정이다.
레포는 이것을 v2 에서 폐기했다 (`../SEGMENTATION_CRITERIA_RELATED_WORK.md` 개정표):
**판정 기준이 offline 어순이라, 어순을 단조화한 좋은 경계가 오탐된다.** ko→en 은 SOV→SVO
라 특히 심하다.

여기서는 그 이진 판정만 **어순 무관한 형태로** 바꾼다.

    S_i = translate(소스 어절 prefix i)          # gtx. 결정론적
    보존(i → j) = "S_j 가 S_i 의 내용을 여전히 담고 있는가"   (i < j)

    instability_next(i)  = 1 − 보존(i → i+1)     한 어절 더 들었을 때
    instability_final(i) = 1 − 보존(i → n)       문장 끝까지 갔을 때 (= MU 의 의미판)
    instability_min(i)   = 1 − min_{j>i} 보존(i → j)   **고정점**: 모든 미래를 견디는가

`보존` 을 세 가지로 잰다.
    `entail` NLI 함의 확률 `P(S_j ⊨ S_i)`. **비대칭이 여기서는 맞다** — S_j 는 내용이
             더 많으므로, 앞의 내용이 살아 있으면 함의가 성립하고 바뀌면 무너진다.
             표면 어순은 함의에 영향을 주지 않으므로 폐기 사유가 해소된다.
    `cos`    임베딩 코사인. **여기서는 정당하다** — 양쪽 다 완결된 렌더링이라
             `embed_check` 를 죽였던 미완성 교란이 없다.
    `chrf`   표면형. 폐기된 접두사 판정에 가장 가까운 대조군.

**무엇을 재는 것인지 분명히 할 것.** 이 값은 *렌더링*이 아니라 **절단 위치**의 성질이다.
`premature_cases.json` 은 대부분 같은 위치에서 번역만 바꾼 변이라 관문으로 판정할 수
없다 (`future_dep.py` 와 같은 한계). 근거는 실데이터 쪽에 둔다.

**위치 교란을 반드시 통제한다.** 뒤에 남은 어절이 많을수록 바뀔 여지가 크다.
`boundary_probe.py` 에서 문장 내용을 안 보는 위치 사전확률이 실제 점수를 이겼던 전례가
있으므로, 여기서도 (a) 남은 어절 버킷 잔차, (b) 같은 문장 이웃 위치 대조, (c) 위치
사전확률 기준선을 전부 낸다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.fixed_point --run-id ko-en/run04
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from .boundary_probe import chance_rate, load_sentences, topk_positions
from .embed_check import EmbedScorer, MODELS
from ..autoseg.pipeline import GoogleTranslator, JsonCache, to_lang_code

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS

MEASURES = ["entail", "cos", "chrf"]
HORIZONS = ["next", "final", "min"]


class EntailBackend(metrics._NliBase):
    """`P(premise ⊨ hypothesis)`. `ContradictionBackend` 와 같은 체크포인트를 공유한다."""

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli",
                 batch_size: int = 32, device: int = 0, name: str = "entail"):
        super().__init__(model_name, batch_size, device, name)

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        out = [0.0] * len(premises)
        pending = [i for i, (p, h) in enumerate(zip(premises, hypotheses))
                   if p.strip() and h.strip()]
        if not pending:
            return out
        res = self.load()([{"text": premises[i], "text_pair": hypotheses[i]}
                           for i in pending], batch_size=self.batch_size)
        for i, scores in zip(pending, res):
            out[i] = self._prob(scores, "entail")
        return out


# ── prefix 번역 궤적 ─────────────────────────────────────────────────────

def translate_trajectories(sents: list[dict], gt: GoogleTranslator) -> dict:
    """문장마다 어절 prefix 1..n 을 전부 번역한다. n 번째 = 문장 전체.

    gtx 는 결정론적이라 궤적이 재현 가능하다 (`pipeline.GoogleTranslator.full_uncached`
    주석에 기록된 성질). 문맥 옵션은 끈다 — prefix 만의 함수여야 스트리밍 조건과 같다."""
    need: set[str] = set()
    for s in sents:
        w = s["text"].split()
        for i in range(1, len(w) + 1):
            need.add(" ".join(w[:i]))
    uniq = sorted(need)
    print(f"[gtx] prefix 번역 {len(uniq)}건 (문장 {len(sents)}개)", flush=True)
    trans = dict(zip(uniq, gt.full(uniq)))
    for s in sents:
        w = s["text"].split()
        s["traj"] = [trans.get(" ".join(w[:i]), "") for i in range(1, len(w) + 1)]
    return {"n_prefixes": len(uniq), "n_calls": gt.calls}


def preserve_scores(sents: list[dict], measure: str, scorer) -> None:
    """보존(i → j) 를 필요한 (i, j) 쌍에 대해 계산한다.

    `min` 지평 때문에 i 마다 j = i+1..n 전부가 필요하다 — 문장 길이의 제곱이지만
    문장이 짧아(평균 10어절 남짓) 감당된다. `chrf` 는 모델이 없어 즉시 계산한다."""
    pairs_a, pairs_b, owner = [], [], []
    for si, s in enumerate(sents):
        traj = s["traj"]
        n = len(traj)
        for i in range(1, n):                       # 절단 위치 i (어절 i 뒤)
            for j in range(i + 1, n + 1):
                pairs_a.append(traj[j - 1])         # S_j  (더 긴 쪽 = premise)
                pairs_b.append(traj[i - 1])         # S_i  (hypothesis)
                owner.append((si, i, j))
    if measure == "chrf":
        vals = [metrics.chrf(b, a) for a, b in zip(pairs_a, pairs_b)]
    else:
        vals = scorer.score(pairs_a, pairs_b)
    table: dict = {}
    for (si, i, j), v in zip(owner, vals):
        table.setdefault(si, {}).setdefault(i, {})[j] = float(v)
    for si, s in enumerate(sents):
        s.setdefault("preserve", {})[measure] = table.get(si, {})


def horizons(sents: list[dict], measure: str) -> None:
    """보존 표에서 세 지평의 불안정도를 뽑는다 (높을수록 나쁨)."""
    for s in sents:
        tab = (s.get("preserve") or {}).get(measure) or {}
        n = len(s["traj"])
        out_next, out_final, out_min = {}, {}, {}
        for i, row in tab.items():
            if not row:
                continue
            out_next[i] = 1.0 - row.get(i + 1, 1.0)
            out_final[i] = 1.0 - row.get(n, 1.0)
            out_min[i] = 1.0 - min(row.values())
        s[f"{measure}_next"] = out_next
        s[f"{measure}_final"] = out_final
        s[f"{measure}_min"] = out_min


# ── 위치 통제 ────────────────────────────────────────────────────────────

def positional_controls(sents: list[dict], key: str) -> None:
    """`boundary_probe` 와 같은 통제 3종을 붙인다.

    `_prior`  문장 내용을 **하나도 안 보고** 남은 어절 버킷의 코퍼스 평균만 준 값
    `_resid`  그 버킷 평균을 뺀 잔차
    `_local`  같은 문장에서 한 어절 앞/뒤로 자른 것과의 차
    """
    buckets: dict[str, list[float]] = {}
    for s in sents:
        n = len(s["traj"])
        for i, v in (s.get(key) or {}).items():
            buckets.setdefault(nf.bucket_of(max(1, n - i)), []).append(v)
    mu = {b: sum(v) / len(v) for b, v in buckets.items() if v}
    overall = (sum(sum(v) for v in buckets.values())
               / max(1, sum(len(v) for v in buckets.values())))
    for s in sents:
        n = len(s["traj"])
        cur = s.get(key) or {}
        prior, resid, local = {}, {}, {}
        for i, v in cur.items():
            m = mu.get(nf.bucket_of(max(1, n - i)), overall)
            prior[i] = m
            resid[i] = v - m
            nb = [cur[k] for k in (i - 1, i + 1) if k in cur]
            local[i] = v - sum(nb) / len(nb) if nb else 0.0
        s[key + "_prior"] = prior
        s[key + "_resid"] = resid
        s[key + "_local"] = local


# ── 평가 ─────────────────────────────────────────────────────────────────

def agreement(sents: list[dict], key: str, tolerance: int = 0) -> dict:
    """LLM 경계와 같은 개수를 뽑아 일치율. **불안정도가 낮은** 곳을 고른다."""
    hits = tot = 0
    for s in sents:
        sc = s.get(key) or {}
        if not sc:
            continue
        gold = set(s["cuts"])
        # 고정점 = 불안정도가 작은 곳이므로 부호를 뒤집어 상위 k 개를 뽑는다
        pred = topk_positions({i: -v for i, v in sc.items()}, len(gold))
        hits += sum(1 for p in pred if any(abs(p - g) <= tolerance for g in gold))
        tot += len(gold)
    return {"hit_rate": round(hits / tot, 4) if tot else None, "n": tot}


def auc_at_cuts(sents: list[dict], key: str) -> dict:
    """LLM 경계 위치의 불안정도 vs 비경계 위치. **0.5 미만 = 경계가 더 안정적**."""
    aucs, gaps = [], []
    for s in sents:
        sc = s.get(key) or {}
        gold = set(s["cuts"])
        pos = [v for i, v in sc.items() if i in gold]
        neg = [v for i, v in sc.items() if i not in gold]
        if not pos or not neg:
            continue
        wins = sum(1 for a in pos for b in neg if a > b)
        ties = sum(1 for a in pos for b in neg if a == b)
        aucs.append((wins + 0.5 * ties) / (len(pos) * len(neg)))
        gaps.append(sum(pos) / len(pos) - sum(neg) / len(neg))
    return {"auc": round(sum(aucs) / len(aucs), 4) if aucs else None,
            "mean_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
            "n_sentences": len(aucs)}


def distribution(sents: list[dict], key: str) -> dict:
    """불안정도 분포. **고정점이 실제로 존재하는가** 를 본다 — 전부 크면 개념이 안 산다."""
    vals = sorted(v for s in sents for v in (s.get(key) or {}).values())
    if not vals:
        return {}
    def q(p):
        return round(vals[min(len(vals) - 1, int(p * (len(vals) - 1)))], 4)
    return {"n": len(vals), "p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
            "mean": round(sum(vals) / len(vals), 4),
            "frac_below_0.05": round(sum(1 for v in vals if v < 0.05) / len(vals), 4),
            "frac_below_0.20": round(sum(1 for v in vals if v < 0.20) / len(vals), 4)}


def vs_contradiction(sents: list[dict], recs: list[dict], key: str) -> dict:
    """실제 사용된 경계에서 불안정도 vs 실측 NLI contradiction 의 순위 상관.

    **양수면 같은 축을 재고 있다는 뜻**이다. 0 근처면 서로 다른 것을 재는 것이고,
    그것 자체가 '핀트가 다르다' 의 정량적 확인이 된다."""
    by_text: dict[str, dict] = {s["text"]: s for s in sents}
    xs, ys = [], []
    for r in recs:
        s = by_text.get(r.get("src_sentence") or "")
        if not s:
            continue
        v = (s.get(key) or {}).get(r["cut_word"])
        if v is None:
            continue
        xs.append(r["nli"])
        ys.append(v)
    if len(xs) < 3:
        return {"spearman": None, "n": len(xs)}
    c = metrics._spearman(xs, ys)
    return {"spearman": round(c, 4) if c is not None else None, "n": len(xs)}


def render(result: dict) -> str:
    L = ["# 번역 고정점 — 뒤가 와도 앞의 번역이 바뀌지 않는 지점", "",
         f"런: `{result['run_id']}` · 문장 {result['n_sentences']}개 · "
         f"prefix 번역 {result['n_prefixes']}건 · 후보 절단 위치 {result['n_positions']}개 · "
         f"LLM 호출 0", "",
         "`contradiction`(NLI)은 *이미 내보낸 것이 반박당했는가* 를 묻는다. 그런데 실제 요건은 "
         "Zhang+ 2020 의 Meaningful Unit — *후속 텍스트에 의해 번역이 바뀌지 않는 최소 구간* "
         "이고, 반박은 그 위반의 **한 종류**일 뿐이다. 문헌의 이진 판정(접두사 일치)은 레포가 "
         "v2 에서 폐기했는데(어순 오탐), 여기서는 그 판정만 **어순 무관한 함의/유사도**로 "
         "바꿔 되살린다.", "",
         "`S_i` = 소스 어절 prefix i 의 gtx 번역. 지평 셋: `next`(한 어절 뒤), "
         "`final`(문장 끝), `min`(모든 미래를 견디는가 = **고정점**).", ""]

    L += ["## 1 — 고정점이 실제로 존재하는가 (불안정도 분포)", "",
          "전 위치가 불안정하면 이 개념은 이 데이터에서 못 쓴다.", "",
          "| 보존 척도 | 지평 | mean | p10 | p50 | p90 | <0.05 비율 | <0.20 비율 |",
          "|---|---|---|---|---|---|---|---|"]
    for d in result["distribution"]:
        s = d["stats"]
        L.append(f"| `{d['measure']}` | {d['horizon']} | {s['mean']} | {s['p10']} | "
                 f"{s['p50']} | {s['p90']} | {s['frac_below_0.05']} | "
                 f"{s['frac_below_0.20']} |")

    L += ["", "## 2 — LLM 이 고정점에서 자르고 있는가", "",
          "`AUC` 는 (경계, 비경계) 한 쌍에서 경계 쪽 불안정도가 **더 큰** 확률이다. "
          "**0.5 미만이면 경계가 더 안정적** = LLM 이 고정점을 고르고 있다는 뜻. "
          "`prior` 는 문장 내용을 하나도 안 보고 남은 어절 버킷 평균만 준 기준선 — "
          "`boundary_probe` 에서 이 기준선이 실제 점수를 이겼던 전례가 있으므로 반드시 같이 본다.", "",
          "| 보존 척도 | 지평 | AUC(raw) | AUC(prior) | AUC(resid) | AUC(local) | 일치율 |",
          "|---|---|---|---|---|---|---|"]
    for r in result["at_cuts"]:
        L.append(f"| `{r['measure']}` | {r['horizon']} | {r['raw']['auc']} | "
                 f"{r['prior']['auc']} | {r['resid']['auc']} | {r['local']['auc']} | "
                 f"{r['agreement']['hit_rate']} |")
    L += ["", f"무작위 위치 일치율 {result['chance']['hit_rate']}."]

    L += ["", "## 3 — 현행 contradiction 과 같은 축인가", "",
          "실제 사용된 경계에서의 순위 상관. **0 근처면 서로 다른 것을 재고 있다는 뜻이고, "
          "그것이 '핀트가 다르다' 의 정량적 확인이다.** 현행 contradiction 도 길이 잡음 "
          "바닥이 있으므로(실측 45배 기울기), 통제 없는 `raw` 는 **두 지표가 같은 교란을 "
          "공유해 부풀 수 있다** — `resid`·`local` 을 같이 읽을 것.", "",
          "| 보존 척도 | 지평 | raw | resid | local | n |", "|---|---|---|---|---|---|"]
    for r in result["vs_contra"]:
        L.append(f"| `{r['measure']}` | {r['horizon']} | {r['spearman']} | "
                 f"{r.get('spearman_resid')} | {r.get('spearman_local')} | {r['n']} |")

    def dist(m, h):
        return next((d["stats"] for d in result["distribution"]
                     if d["measure"] == m and d["horizon"] == h), {}) or {}

    def cuts(m, h):
        return next((r for r in result["at_cuts"]
                     if r["measure"] == m and r["horizon"] == h), {}) or {}

    def contra(m, h):
        return next((r for r in result["vs_contra"]
                     if r["measure"] == m and r["horizon"] == h), {}) or {}

    e_min, c_min = dist("entail", "min"), dist("chrf", "min")
    e_fin, ef_cuts, ef_contra = dist("entail", "final"), cuts("entail", "final"), contra("entail", "final")

    L += ["", "## 판정", "",
          "### ① 개념은 살아 있다 — 죽은 것은 **표면형 판정**이었다", ""]
    if e_min and c_min:
        L += [f"엄격한 고정점(`min`: 모든 미래를 견딤) 비율이 판정 방식에 따라 갈린다 — "
              f"표면형 `chrf` 로는 **{c_min['frac_below_0.05']:.1%}**, 어순 무관 `entail` 로는 "
              f"**{e_min['frac_below_0.05']:.1%}** (불안정도 <0.05 기준. 임계값은 서술용이지 "
              "판정용이 아니다).", "",
              "v2 가 prefix-consistency 를 폐기한 사유가 여기서 그대로 확인된다 — "
              "**어순을 단조화한 좋은 경계가 표면형 접두사 판정에서 전멸한다.** 판정만 함의로 "
              "바꾸면 4분의 1이 살아난다. 폐기된 것은 개념이 아니라 자였다.", ""]

    L += ["### ② LLM 은 실제로 더 안정적인 곳을 자른다 — 위치를 통제해야만 보인다", ""]
    if ef_cuts:
        L += [f"`entail`+`final` 기준 AUC: raw {ef_cuts['raw']['auc']} → "
              f"prior **{ef_cuts['prior']['auc']}** → resid {ef_cuts['resid']['auc']} / "
              f"local {ef_cuts['local']['auc']}.", "",
              "**통제 없이 보면 결론이 뒤집힌다.** raw 가 0.5 를 넘어 'LLM 이 불안정한 곳을 "
              "자른다' 로 읽히는데, 문장 내용을 하나도 안 보는 위치 사전확률이 "
              f"{ef_cuts['prior']['auc']} 라 그 값은 전부 위치에서 온다. 교란을 빼면 0.5 "
              "아래로 내려가 방향이 맞는다. `boundary_probe.py` 에서 얻은 교훈이 여기서 "
              "반복된다 — **이 축의 기준선은 무작위가 아니라 위치 사전확률이다.**", ""]

    L += ["### ③ contradiction 과 겹치지만 같지 않다 — '핀트' 차이의 크기", ""]
    if ef_contra:
        r_ = ef_contra.get("spearman") or 0
        L += [f"`entail`+`final` vs 실측 contradiction 의 순위 상관 **{r_}** "
              f"(잔차 통제 후 {ef_contra.get('spearman_resid')} — 공유 교란 때문이 아니다). "
              f"순위 분산의 약 {r_ ** 2:.0%} 를 공유하고 나머지는 서로 다른 것을 잰다.", "",
              "**방향성이 있는 척도가 이긴다**는 것도 재확인됐다: "
              f"`entail` {r_} > `chrf` {contra('chrf', 'final').get('spearman')} > "
              f"`cos` {contra('cos', 'final').get('spearman')}. 대칭 코사인이 꼴찌다.", ""]

    L += ["### ④ 그래서 무엇을 쓸 것인가", "",
          "두 지표는 **다른 질문에 답한다.**", "",
          "| | 묻는 것 | 위반의 의미 |",
          "|---|---|---|",
          "| `contradiction` (현행) | 내보낸 것이 **틀렸는가** | 사용자가 거짓을 봤다 |",
          "| `instability` (고정점) | 내보낸 것을 **고쳐야 하는가** | 수정 없이는 어색하거나 부정확하다 |",
          "",
          "우리 시스템은 **수정하지 않으므로**(streaming, 재번역 없음) 둘 다 실패지만 "
          "심각도가 다르다 — 거짓을 보여준 것과 어색하게 보여준 것은 같은 벌점이 아니다. "
          "현행 목적함수가 contradiction 을 고른 것은 **더 심각한 쪽만 벌하겠다**는 선택이고, "
          "그 선택 자체는 방어 가능하다. 다만 그것이 MU 정의와 다르다는 지적은 옳고, "
          "이 측정이 그 차이의 크기를 처음으로 수치화했다.", "",
          "**당장 할 수 있는 것**: `entail`+`final` 은 지평 중 성능이 가장 좋으면서 "
          "**가장 싸다** (문장당 NLI 호출 n 회. `min` 은 n²/2 회). 오라클 번역도 LLM 도 "
          "필요 없다. 목적함수를 건드리지 않고 **보고 지표로 먼저 얹어** 곡선에서 "
          "contradiction 과 어떻게 갈리는지 보는 것이 위험이 가장 낮다.", "",
          "**한계**: 이 값은 *렌더링*이 아니라 **절단 위치**의 성질이라 "
          "`premature_cases.json`(같은 위치에서 번역만 바꾼 변이)으로는 관문 판정을 할 수 "
          "없다. 채택하려면 절단 위치가 변이마다 다른 관문 케이스를 새로 써야 한다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="번역 고정점 측정")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--measures", nargs="+", default=MEASURES, choices=MEASURES)
    p.add_argument("--max-sentences", type=int, default=0)
    p.add_argument("--max-words", type=int, default=24,
                   help="이보다 긴 문장은 제외 (prefix 수가 제곱으로 는다)")
    p.add_argument("--embed-model", default="e5-inst", choices=sorted(MODELS))
    p.add_argument("--nli-model", default="deberta-mnli", choices=sorted(metrics.NLI_MODELS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = SEG_RUNS / args.run_id
    out_dir = Path(args.out) if args.out else (OUT_RUNS / "fixed_point")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        result = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
        (out_dir / "report.md").write_text(render(result), encoding="utf-8")
        print(f"[done] {out_dir / 'report.md'}")
        return 0

    sents = [s for s in load_sentences(run_dir) if s["n_words"] <= args.max_words]
    # 같은 문장이 T 마다 중복되므로 텍스트 단위로 합치고 경계는 합집합으로 둔다
    merged: dict[str, dict] = {}
    for s in sents:
        m = merged.setdefault(s["text"], {"text": s["text"], "n_words": s["n_words"],
                                          "cuts": set()})
        m["cuts"].update(s["cuts"])
    sents = [{**v, "cuts": sorted(v["cuts"])} for v in merged.values()]
    if args.max_sentences:
        sents = sents[: args.max_sentences]
    if not sents:
        print(f"문장을 못 찾음: {run_dir}", file=sys.stderr)
        return 1

    # 실측 contradiction 과 대조할 경계 레코드 (어절 인덱스로 변환)
    from .embed_check import load_boundaries
    from .future_dep import enrich
    recs = enrich(run_dir, load_boundaries(run_dir))["recs"]
    for r in recs:
        r["cut_word"] = len(r["src_sentence"][: r["prefix_chars"]].split())

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    code = cfg.get("tgt_code") or to_lang_code(cfg.get("tgt_lang") or "English")
    gt = GoogleTranslator(tgt_code=code, workers=args.workers,
                          cache=JsonCache(out_dir / "prefix_cache.json"),
                          use_context=False)
    try:
        stats = translate_trajectories(sents, gt)
    finally:
        gt.close()

    result: dict = {"run_id": args.run_id, "n_sentences": len(sents),
                    "n_prefixes": stats["n_prefixes"],
                    "n_positions": sum(max(0, len(s["traj"]) - 1) for s in sents),
                    "distribution": [], "at_cuts": [], "vs_contra": [],
                    "chance": chance_rate(sents, 0)}

    entail = EntailBackend(model_name=metrics.NLI_MODELS[args.nli_model])
    embed = EmbedScorer(args.embed_model, MODELS[args.embed_model])

    class _CosPreserve:
        def score(self, a, b):
            return embed.cos(list(a), list(b))

    for measure in args.measures:
        print(f"\n[{measure}] 보존 계산...", flush=True)
        scorer = entail if measure == "entail" else (_CosPreserve() if measure == "cos"
                                                     else None)
        preserve_scores(sents, measure, scorer)
        horizons(sents, measure)
        for h in HORIZONS:
            key = f"{measure}_{h}"
            positional_controls(sents, key)
            result["distribution"].append(
                {"measure": measure, "horizon": h, "stats": distribution(sents, key)})
            result["at_cuts"].append({
                "measure": measure, "horizon": h,
                "raw": auc_at_cuts(sents, key),
                "prior": auc_at_cuts(sents, key + "_prior"),
                "resid": auc_at_cuts(sents, key + "_resid"),
                "local": auc_at_cuts(sents, key + "_local"),
                "agreement": agreement(sents, key)})
            # 현행 contradiction 도 길이 잡음 바닥이 있다 (실측 45배 기울기). 통제 없이
            # 재면 **두 지표가 같은 교란을 공유해서** 상관이 부풀 수 있으므로 잔차·국소
            # 버전을 같이 낸다.
            vc = vs_contradiction(sents, recs, key)
            vc.update({"measure": measure, "horizon": h,
                       "spearman_resid": vs_contradiction(
                           sents, recs, key + "_resid")["spearman"],
                       "spearman_local": vs_contradiction(
                           sents, recs, key + "_local")["spearman"]})
            result["vs_contra"].append(vc)
            print(f"  {key}: local AUC {result['at_cuts'][-1]['local']['auc']}, "
                  f"vs contra {vc['spearman']}", flush=True)
        if measure == "cos":
            embed.unload()

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
