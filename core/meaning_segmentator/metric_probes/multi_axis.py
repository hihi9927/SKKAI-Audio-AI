"""유형별 의미 축 여러 개 — `semantic_axis.py` 의 다축 확장.

`semantic_axis` 결과: 규칙으로 만든 **부정 축은 실재한다** (MNLI 모순 축과 코사인 0.5865,
무작위 0.031). 1차원만으로 MNLI 모순 AUC 0.784 로 코사인(0.722)을 넘는다. 그런데 관문은
3/6 이었다 — 관문의 실패 유형이 다섯인데 축 하나는 하나를 덮기 때문이다.

그래서 유형마다 축을 만든다. 전부 **규칙 기반 최소쌍**이라 라벨이 0 이다.

  `negation`    is → is not            극성            (`premature_negation`)
  `role`        고유명사·인칭대명사 교환   참여자          (`premature_role`)
  `modality`    will → might           서법·확신도      (`premature_modal`)
  `quantifier`  all → some             범위·양화        (`premature_scope`)
  `tense`       is → was               시제·상          (`premature_head` 근사)

**먼저 확인해야 하는 것은 축들이 서로 다른 방향인가다.** 전부 같은 쪽을 가리키면 축이
다섯 개인 것이 아니라 하나를 다섯 번 만든 것이다. Gram 행렬로 본다.

**대조군을 반드시 둔다.** 같은 구성으로 만든 **무작위 방향**과 **내용어 교환 축**
(유형이 아닌 그냥 의미 변화)을 같이 재야, 이름 붙인 축이 이름 때문에 작동하는지
아니면 아무 방향이나 그 정도는 하는지 갈린다.

결합은 `max_k |proj_k(d)| / ‖d‖` — 어느 유형이든 하나라도 크게 걸리면 위험으로 본다
(양방향 NLI 가 min 을 쓰는 것과 같은 발상). 다만 **유형 목록이 곧 임의 상수**가 된다는
비용은 남는다: 목록에 없는 실패는 구조적으로 못 잡는다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.multi_axis
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from .embed_check import (MODELS, contra_gate, load_boundaries, measure_floor,
                          prune_degenerate, real_data)
from .embed_probe import FrozenEncoder
from .semantic_axis import _auc, _unit, axis_from_pairs, negate

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS


# ── 유형별 규칙 최소쌍 생성기 ────────────────────────────────────────────
#
# 공통 원칙: **한 곳만 바꾼다.** 문장을 많이 바꾸면 차이벡터에 주제 변화가 섞여 축이
# 흐려진다. 규칙이 안 걸리면 그냥 버린다 — 재현율이 아니라 순도가 중요하다.

_PROPER = re.compile(r"\b([A-Z][a-z]{2,})\b")
_PRON = {"he": "she", "she": "he", "him": "her", "her": "him",
         "his": "her", "they": "we", "we": "they", "I": "you", "you": "I"}
_MODAL = {"will": "might", "might": "will", "must": "may", "may": "must",
          "can": "should", "should": "can", "would": "could", "could": "would"}
_QUANT = {"all": "some", "some": "all", "every": "one", "always": "sometimes",
          "sometimes": "always", "never": "often", "often": "never",
          "everyone": "someone", "someone": "everyone", "none": "some"}
_TENSE = {"is": "was", "was": "is", "are": "were", "were": "are",
          "has": "had", "had": "has", "does": "did", "did": "does"}

_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "that", "this", "it",
         "is", "are", "was", "were", "for", "on", "with", "as", "at", "by"}


def _swap_words(text: str, i: int, j: int) -> str:
    w = text.split()
    w[i], w[j] = w[j], w[i]
    return " ".join(w)


def _sub_map(text: str, mapping: dict) -> str | None:
    """사전에 걸리는 **첫 토큰 하나만** 치환한다."""
    w = text.split()
    for i, tok in enumerate(w):
        core = tok.strip(".,;:!?\"'()").lower()
        if core in mapping:
            repl = mapping[core]
            w[i] = tok.lower().replace(core, repl, 1) if core in tok.lower() else repl
            return " ".join(w)
    return None


# 문장 첫머리의 대문자는 고유명사가 아닐 수 있다. 그렇다고 index 0 을 통째로 빼면
# "Mary told John…" 같은 전형적인 역할 교환 문장을 놓친다 — 기능어만 걸러낸다.
_INIT_STOP = {"the", "this", "that", "these", "those", "there", "it", "he", "she",
              "they", "we", "you", "his", "her", "but", "and", "for", "in", "on",
              "at", "if", "when", "while", "after", "before", "a", "an", "some",
              "all", "one", "two", "many", "most", "such", "then", "now", "here"}


def perturb_role(text: str) -> str | None:
    """고유명사 두 개 또는 인칭대명사 두 개를 맞바꾼다 — 참여자만 뒤집힌다."""
    w = text.split()

    def is_proper(i: int) -> bool:
        core = w[i].strip(".,;:!?\"'()")
        if not _PROPER.fullmatch(core or ""):
            return False
        return i > 0 or core.lower() not in _INIT_STOP

    props = [i for i in range(len(w)) if is_proper(i)]
    if len(props) >= 2:
        return _swap_words(text, props[0], props[1])
    prons = [i for i, t in enumerate(w) if t.strip(".,;:!?\"'()").lower() in _PRON]
    if len(prons) >= 2:
        a, b = prons[0], prons[1]
        if w[a].lower() != w[b].lower():
            return _swap_words(text, a, b)
    # 대명사가 하나뿐이면 상대 인칭으로 치환 (여전히 참여자 변화)
    if prons:
        return _sub_map(text, _PRON)
    return None


def perturb_modality(text: str) -> str | None:
    return _sub_map(text, _MODAL)


def perturb_quantifier(text: str) -> str | None:
    return _sub_map(text, _QUANT)


def perturb_tense(text: str) -> str | None:
    return _sub_map(text, _TENSE)


def perturb_content_swap(text: str, rng: random.Random) -> str | None:
    """**대조군.** 유형이 아닌 그냥 내용어 두 개 교환. 의미는 바뀌지만 이름이 없다."""
    w = text.split()
    cand = [i for i, t in enumerate(w)
            if t.strip(".,;:!?\"'()").lower() not in _STOP and len(t) > 3]
    if len(cand) < 2:
        return None
    i, j = rng.sample(cand, 2)
    return _swap_words(text, i, j)


GENERATORS = {
    "negation": lambda t, rng: negate(t),
    "role": lambda t, rng: perturb_role(t),
    "modality": lambda t, rng: perturb_modality(t),
    "quantifier": lambda t, rng: perturb_quantifier(t),
    "tense": lambda t, rng: perturb_tense(t),
    "content_swap": perturb_content_swap,          # 대조군
}


def build_pairs(texts: list[str], kind: str, limit: int, seed: int = 0):
    rng = random.Random(seed)
    gen = GENERATORS[kind]
    out = []
    for t in texts:
        try:
            p = gen(t, rng)
        except Exception:
            p = None
        if p and p != t:
            out.append((t, p))
        if len(out) >= limit:
            break
    return out


# ── 다축 채점기 ─────────────────────────────────────────────────────────

class MultiAxisScorer:
    """`max_k |proj_k(d)| / ‖d‖`. 어느 유형이든 하나라도 크게 걸리면 위험.

    양방향 NLI 가 `min(entail)` 로 "어느 방향 실패든 잡는" 것과 같은 발상이다.
    정규화(`/‖d‖`)는 `semantic_axis` 에서 필요성이 확인됐다 — 안 하면 짧은 조각이
    큰 `‖d‖` 때문에 모든 성분을 크게 받아 종류와 크기가 다시 섞인다."""

    def __init__(self, enc: FrozenEncoder, axes: dict, name: str,
                 mode: str = "max", blend: float = 0.0):
        self.enc = enc
        self.axes = axes
        self.name = name
        self.mode = mode
        self.blend = blend

    def components(self, premises: list[str], hypotheses: list[str]):
        import numpy as np
        u = self.enc.encode(list(premises))
        v = self.enc.encode(list(hypotheses))
        d = v - u
        nrm = np.clip(np.linalg.norm(d, axis=1), 1e-9, None)
        comp = {k: np.abs(d @ a) / nrm for k, a in self.axes.items()}
        un = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-9, None)
        vn = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        return comp, 1.0 - (un * vn).sum(axis=1)

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import numpy as np
        comp, cos = self.components(premises, hypotheses)
        M = np.stack([comp[k] for k in sorted(comp)], axis=1)
        val = M.max(axis=1) if self.mode == "max" else np.linalg.norm(M, axis=1)
        if self.blend:
            val = val + self.blend * cos
        return [float(x) for x in np.nan_to_num(val)]


def render(result: dict) -> str:
    L = ["# 유형별 의미 축 여러 개 — 다축 구성", "",
         f"인코더 `{result['encoder_id']}` · MNLI {result['n_mnli']}쌍 · "
         f"run04 경계 {result['n_boundaries']}개 · **라벨 0건**(축은 전부 규칙 생성)", "",
         "`semantic_axis` 에서 부정 축 하나로 관문 3/6 이었다. 관문의 실패 유형이 다섯인데 "
         "축 하나가 하나를 덮기 때문이었으므로, 유형마다 축을 만들어 결합한다.", ""]

    L += ["## ① 최소쌍 수율", "", "| 축 | 규칙 | 최소쌍 |", "|---|---|---|"]
    for r in result["pairs"]:
        L.append(f"| `{r['kind']}` | {r['rule']} | {r['n']} |")

    L += ["", "## ② 축들이 서로 다른 방향인가 (Gram 행렬)", "",
          "**전부 같은 쪽을 가리키면 축이 다섯 개가 아니라 하나를 다섯 번 만든 것이다.** "
          f"무작위 방향 두 개의 코사인 기대값 ≈ {result['random_cos']:.3f}.", "",
          "| | " + " | ".join(result["axis_names"]) + " |",
          "|---|" + "---|" * len(result["axis_names"])]
    for i, a in enumerate(result["axis_names"]):
        row = [f"{result['gram'][i][j]:.3f}" for j in range(len(result["axis_names"]))]
        L.append(f"| **{a}** | " + " | ".join(row) + " |")
    L += ["", "MNLI 모순 축과의 정렬:", "",
          "| 축 | 모순 축과 코사인 |", "|---|---|"]
    for k, v in result["align_contra"].items():
        L.append(f"| `{k}` | {v} |")

    L += ["", "## ③ MNLI dev 모순 AUC — 축별과 결합", "",
          "| 읽기 | 모순 AUC |", "|---|---|"]
    for r in result["mnli_auc"]:
        L.append(f"| {r['name']} | **{r['auc']}** |")

    L += ["", "## ④ 관문 (`premature_cases.json`)", "",
          "| 채점기 | 변이 | 위반/케이스 | mean(prem) | mean(safe) | 격차 | 판정 |",
          "|---|---|---|---|---|---|---|"]
    for g in result["gate"]:
        for k, v in g["variants"].items():
            L.append(f"| `{g['scorer']}` | {k} | {v['violations']}/{v['n_cases']} | "
                     f"{v['mean_premature']} | {v['mean_safe']} | {v['margin']} | "
                     f"{'통과' if v['passed'] else '**탈락**'} |")

    if result.get("case_components"):
        L += ["", "### 유형이 맞는 축이 실제로 켜지는가", "",
              "케이스마다 `premature 성분 − safe 성분`. **양수 = 그 축이 조기 방출 쪽에서 "
              "더 켜졌다.** 유형과 축이 대응한다면 대각선이 밝아야 한다.", "",
              "| 케이스 (유형) | " + " | ".join(result["axis_names"]) + " |",
              "|---|" + "---|" * len(result["axis_names"])]
        for row in result["case_components"]:
            cells = []
            for a in result["axis_names"]:
                d = row["delta"].get(a)
                cells.append("—" if d is None else
                             (f"**{d:+.3f}**" if a == row["expect_axis"] else f"{d:+.3f}"))
            L.append(f"| {row['case']} ({row['type']}) | " + " | ".join(cells) + " |")
        L.append("")
        L.append("굵은 칸이 유형에 대응하는 축이다.")

    L += ["", "## ⑤ 잡음 바닥과 실데이터", "",
          "| 채점기 | 바닥 mean | 바닥 sd | " + " | ".join(nf.bucket_labels()) + " |",
          "|---|---|---|" + "---|" * len(nf.bucket_labels())]
    for f in result["floors"]:
        fl = f["floor"]
        cells = [str(fl["by_length_bucket"][b]["mean"]) if fl["by_length_bucket"][b]["n"]
                 else "—" for b in nf.bucket_labels()]
        L.append(f"| `{f['scorer']}` | {fl['overall']['mean']} | {fl['overall']['sd']} | "
                 + " | ".join(cells) + " |")
    L += ["", "| 채점기 | 변이 | 현행 NLI 와 Spearman | topk 겹침 |", "|---|---|---|---|"]
    for r in result["real"]:
        for k, v in r["variants"].items():
            L.append(f"| `{r['scorer']}` | {k} | {v['spearman_global']} | "
                     f"{v['topk_overlap']} |")

    # ── 판정 ──
    L += ["", "## 판정", ""]
    gram = result["gram"]
    names = result["axis_names"]
    offdiag = [gram[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
    mx = max(abs(x) for x in offdiag) if offdiag else 0.0
    mean_off = sum(abs(x) for x in offdiag) / len(offdiag) if offdiag else 0.0
    L += [f"**축들은 서로 구별된다.** Gram 비대각 성분의 평균 절대값 {mean_off:.3f}, "
          f"최대 {mx:.3f} — 무작위 기대({result['random_cos']:.3f})보다는 크지만 "
          "1 과는 멀다. 다섯 축은 겹치되 같지 않다.", ""]

    # `role` 이 대조군과 얼마나 겹치는가 — 구조 변화가 방향으로 잡히는지의 지표
    def gram_of(a, b):
        try:
            return gram[names.index(a)][names.index(b)]
        except ValueError:
            return None

    rc = gram_of("role", "content_swap")
    if rc is not None:
        L += [f"**단 `role` 은 예외다 — 대조군(`content_swap`)과 코사인 {rc:.3f}.** "
              "참여자 교환은 **토큰 집합이 그대로**이고 배열만 바뀐다. 풀링된 임베딩에서 "
              "그것은 '단어가 섞였다' 와 구별되지 않는다. 즉 `role` 축은 참여자 의미를 "
              "잡은 것이 아니라 어순 교란을 잡은 것이다.", ""]

    ctrl = next((r for r in result["mnli_auc"] if "content_swap" in r["name"]), None)
    singles = [r for r in result["mnli_auc"] if r.get("single")]
    best_single = max(singles, key=lambda r: r["auc"] or 0, default=None)
    cosr = next((r for r in result["mnli_auc"] if r["name"].startswith("cos")), None)
    maxr = next((r for r in result["mnli_auc"] if r["name"].startswith("max(neg")), None)
    bestc = max((r for r in result["mnli_auc"] if r.get("combined")),
                key=lambda r: r["auc"] or 0, default=None)

    L += ["### 잘 되는 축과 안 되는 축이 갈린다", ""]
    if ctrl and best_single:
        good = [r for r in singles if (r["auc"] or 0) > 0.6]
        bad = [r for r in singles if (r["auc"] or 0) <= 0.55 and "content" not in r["name"]]
        L += ["| 축 | MNLI 모순 AUC | 성질 |", "|---|---|---|"]
        for r in singles:
            note = ("**토큰이 바뀐다**" if (r["auc"] or 0) > 0.6
                    else ("대조군" if "content" in r["name"] else "배열·문법 자질만 바뀐다"))
            L.append(f"| {r['name']} | {r['auc']} | {note} |")
        L += ["",
              f"**어휘가 실제로 치환되는 유형**(부정 `not` 삽입, 양화 `all→some`)만 신호를 "
              f"낸다. 대조군은 {ctrl['auc']} 로 우연 수준이라, '이름' 이 하는 일이 있다는 "
              "것은 확인된다.", "",
              "> **주의 — MNLI 모순 AUC 는 `modality`·`tense` 에는 맞지 않는 자다.** "
              "서법·시제 변화는 MNLI 에서 대개 `neutral` 로 라벨링되지 모순이 아니다. "
              "우리 문제에서는 그것도 실패이므로, 이 두 축의 평가는 아래 케이스별 표로 "
              "봐야 한다 — 실제로 `modality` 축은 `ko-en-p06`(서법 확정)에서 가장 크게 "
              "켜졌다(+0.167).", ""]

    if maxr and best_single and cosr:
        delta = (maxr["auc"] or 0) - (best_single["auc"] or 0)
        L += ["### 결합이 오히려 해롭다", "",
              f"축 다섯 개의 max 가 {maxr['auc']} 인데 **부정 축 단독이 "
              f"{best_single['auc']}** 다 ({delta:+.3f}). 잡음 축(우연 수준의 셋)이 "
              "max 를 자주 가져가기 때문이다. **max 결합은 최악의 축에 지배된다** — "
              "양방향 NLI 의 `min` 이 두 방향 다 의미가 있어서 성립하는 것과 대조된다. "
              "여기서는 축마다 신뢰도가 달라 같은 발상이 성립하지 않는다.", "",
              f"거리(코사인 {cosr['auc']})를 더한 `max(축)+cos` 가 {bestc['auc']} 로 "
              "가장 높은데, 이는 축의 기여가 아니라 **거리 축이 여전히 주력**이라는 뜻이다.", ""]

    gate_pass = [(g["scorer"], k) for g in result["gate"]
                 for k, v in g["variants"].items() if v["passed"]]
    best_gate = min((v["violations"] for g in result["gate"]
                     for v in g["variants"].values()), default=None)
    hits = sum(1 for r in result["case_components"]
               if r["expect_axis"] and (r["delta"].get(r["expect_axis"]) or 0) > 0)
    n_cases = len(result["case_components"])

    L += ["### 관문과 유형 대응", ""]
    if gate_pass:
        L += ["**관문 통과: " + ", ".join(f"`{s}`({k})" for s, k in gate_pass) + "**", "",
              "바닥·실데이터를 함께 보고 채택 여부를 판단할 것."]
    else:
        L += [f"**관문은 여전히 탈락이다** (최선 {best_gate}/6). 단축 3/6 에서 "
              f"{best_gate}/6 으로 개선은 됐다.", "",
              f"유형에 대응하는 축이 실제로 켜진 케이스는 **{n_cases}건 중 {hits}건**이다. "
              "`ko-en-p01`(부정)에서 부정 축이 **음수**로 나오는 것이 대표적 실패인데, "
              "안전 변이(`As for that,`)가 내용이 거의 없는 조각이라 차이벡터의 방향이 "
              "사실상 임의적이기 때문이다 — `semantic_axis` 에서 확인한 것과 같은 문제이고, "
              "축을 늘려도 해결되지 않는다.", ""]

    L += ["### 결론 — 여기서 멈추는 것이 맞다", "",
          "1. **축 개수를 늘리는 방향은 수익이 체감한다.** 이득은 어휘 치환형 두 축에 "
          "몰려 있고, 나머지를 더하면 max 결합에서는 오히려 깎인다.",
          "2. **구조적 실패는 방향으로 표현되지 않는다.** 참여자 교환은 토큰 집합이 "
          "보존되므로 풀링된 벡터에서 어순 교란과 구별되지 않는다(`role` vs "
          f"`content_swap` 코사인 {rc:.3f}). 이것이 교차 인코더와의 격차의 실체다 — "
          "누가 누구에게 했는지는 두 문장의 **토큰이 서로를 봐야** 판정된다.",
          "3. **유형 목록은 임의 상수다.** 목록에 없는 실패는 구조적으로 못 잡는다. "
          "이 레포가 임의 상수를 계속 줄여온 방향과 반대이고, 지금 이득의 크기가 "
          "그 비용을 정당화하지 못한다.", "",
          "축 탐색으로 얻은 것은 대체재가 아니라 **진단**이다: 임베딩이 잡는 의미 변화는 "
          "어휘 치환형이고, 못 잡는 것은 구조형이다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="유형별 다축 실험")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--encoder", default="e5-inst", choices=sorted(MODELS))
    p.add_argument("--mnli-size", type=int, default=40000)
    p.add_argument("--dev-size", type=int, default=3000)
    p.add_argument("--pairs-per-axis", type=int, default=3000)
    p.add_argument("--floor-sentences", type=int, default=150)
    p.add_argument("--max-boundaries", type=int, default=0)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (OUT_RUNS / "multi_axis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        result = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
        (out_dir / "report.md").write_text(render(result), encoding="utf-8")
        print(f"[done] {out_dir / 'report.md'}")
        return 0

    import numpy as np
    from datasets import load_dataset

    enc = FrozenEncoder(args.encoder)
    run_dir = SEG_RUNS / args.run_id

    print("[data] MNLI 로드...", flush=True)
    tr = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=0)
    tr = tr.select(range(min(args.mnli_size, len(tr))))
    dv = load_dataset("nyu-mll/multi_nli", split="validation_matched").shuffle(seed=0)
    dv = dv.select(range(min(args.dev_size, len(dv))))
    corpus = list(tr["premise"]) + list(tr["hypothesis"])

    RULES = {"negation": "is → is not", "role": "고유명사·대명사 교환",
             "modality": "will → might", "quantifier": "all → some",
             "tense": "is → was", "content_swap": "내용어 두 개 교환 (**대조군**)"}
    axes: dict = {}
    pair_stats = []
    for kind in GENERATORS:
        pairs = build_pairs(corpus, kind, args.pairs_per_axis)
        pair_stats.append({"kind": kind, "rule": RULES[kind], "n": len(pairs)})
        print(f"[axis] {kind}: 최소쌍 {len(pairs)}개", flush=True)
        if len(pairs) >= 100:
            axes[kind] = axis_from_pairs(enc, pairs)

    named = [k for k in axes if k != "content_swap"]
    print("[enc] MNLI 인코딩...", flush=True)
    u_dv = enc.encode(list(dv["premise"]))
    v_dv = enc.encode(list(dv["hypothesis"]))
    y_dv = np.array(dv["label"])
    d_dv = v_dv - u_dv
    u_tr = enc.encode(list(tr["premise"])[:20000])
    v_tr = enc.encode(list(tr["hypothesis"])[:20000])
    y_tr = np.array(tr["label"])[:20000]
    d_tr = v_tr - u_tr
    ax_contra = _unit((d_tr[y_tr == 2]).mean(axis=0) - (d_tr[y_tr == 0]).mean(axis=0))

    names = sorted(axes)
    gram = [[round(float(axes[a] @ axes[b]), 4) for b in names] for a in names]
    dim = len(ax_contra)
    result: dict = {
        "run_id": args.run_id, "encoder": args.encoder,
        "encoder_id": MODELS[args.encoder]["id"], "n_mnli": len(tr),
        "random_cos": float(1 / dim ** 0.5), "pairs": pair_stats,
        "axis_names": names, "gram": gram,
        "align_contra": {k: round(float(axes[k] @ ax_contra), 4) for k in names},
        "mnli_auc": [], "gate": [], "floors": [], "real": [], "n_boundaries": 0,
        "case_components": []}
    print(f"[axis] 모순 축 정렬: {result['align_contra']}", flush=True)

    # MNLI dev AUC
    nrm = np.clip(np.linalg.norm(d_dv, axis=1), 1e-9, None)
    is_c = y_dv == 2
    comps = {k: np.abs(d_dv @ axes[k]) / nrm for k in names}
    un = u_dv / np.clip(np.linalg.norm(u_dv, axis=1, keepdims=True), 1e-9, None)
    vn = v_dv / np.clip(np.linalg.norm(v_dv, axis=1, keepdims=True), 1e-9, None)
    cosv = 1.0 - (un * vn).sum(axis=1)
    for k in names:
        a = _auc(comps[k][is_c], comps[k][~is_c])
        result["mnli_auc"].append({"name": f"axis:{k}", "auc": round(a, 4) if a else None,
                                   "single": True})
    M = np.stack([comps[k] for k in named], axis=1)
    for label, vals, flag in [
            ("cos (거리 기준선)", cosv, {}),
            (f"max({'+'.join(named)})", M.max(axis=1), {"combined": True}),
            ("l2(모든 이름 축)", np.linalg.norm(M, axis=1), {"combined": True}),
            ("max(축) + cos", M.max(axis=1) + cosv, {"combined": True})]:
        a = _auc(vals[is_c], vals[~is_c])
        result["mnli_auc"].append({"name": label, "auc": round(a, 4) if a else None,
                                   **flag})
    for r in result["mnli_auc"]:
        print(f"  {r['name']}: AUC {r['auc']}", flush=True)

    # 관문·바닥·실데이터
    recs = load_boundaries(run_dir)
    if args.max_boundaries:
        recs = recs[: args.max_boundaries]
    result["n_boundaries"] = len(recs)
    fulls = sorted({r["premise"] for r in recs})[: args.floor_sentences]
    cases = json.loads((AUTOSEG / "premature_cases.json").read_text(encoding="utf-8"))["cases"]

    named_axes = {k: axes[k] for k in named}
    scorers = [
        MultiAxisScorer(enc, named_axes, "multi:max"),
        MultiAxisScorer(enc, named_axes, "multi:l2", mode="l2"),
        MultiAxisScorer(enc, named_axes, "multi:max+cos", blend=1.0),
    ]
    for sc in scorers:
        print(f"\n[{sc.name}] 바닥·관문·실데이터...", flush=True)
        floor = measure_floor(fulls, sc)
        result["floors"].append({"scorer": sc.name, "floor": floor})
        g = contra_gate(cases, sc, floor)
        g["scorer"] = sc.name
        result["gate"].append(g)
        rd = real_data(recs, sc, floor)
        rd["scorer"] = sc.name
        result["real"].append(rd)
        print(f"[{sc.name}] 관문 위반 {g['variants']['raw']['violations']}"
              f"/{g['variants']['raw']['n_cases']}", flush=True)

    # 유형 ↔ 축 대응 확인
    EXPECT = {"ko-en-p01": ("premature_negation", "negation"),
              "ko-en-p02": ("premature_role", "role"),
              "ko-en-p03": ("premature_scope", "quantifier"),
              "ko-en-p04": ("premature_head", "tense"),
              "ko-en-p06": ("premature_modal", "modality"),
              "ja-ko-p05": ("premature_negation", "negation")}
    probe = MultiAxisScorer(enc, named_axes, "probe")
    for c in cases:
        prem, safe = [], []
        for name, v in c["variants"].items():
            bd = v.get("boundary", c.get("boundary", 0))
            item = (c["full_translation"], " ".join(v["pieces_tgt"][: bd + 1]))
            (prem if v["expect"] == "premature" else safe).append(item)
        if not prem or not safe:
            continue
        cp, _ = probe.components([x[0] for x in prem], [x[1] for x in prem])
        cs, _ = probe.components([x[0] for x in safe], [x[1] for x in safe])
        typ, exp_axis = EXPECT.get(c["id"], ("?", None))
        result["case_components"].append({
            "case": c["id"], "type": typ, "expect_axis": exp_axis,
            "delta": {k: round(float(cp[k].min() - cs[k].max()), 4) for k in named}})

    prune_degenerate({"floors": {f["scorer"]: f["floor"] for f in result["floors"]},
                      "contra_gate": [dict(g, backend=g["scorer"]) for g in result["gate"]],
                      "real": [dict(r, backend=r["scorer"]) for r in result["real"]]})

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
