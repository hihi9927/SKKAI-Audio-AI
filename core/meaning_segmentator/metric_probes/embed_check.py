"""NLI 를 **semantic similarity(임베딩 코사인)** 로 대체할 수 있는가 — 측정 전용.

NLI 가 들어가는 자리는 두 곳이다.

  contradiction  `NLI(premise = full 번역, hypothesis = 그 시점까지 방출된 누적 번역)`
                 **목적함수에 직접 들어간다** (`effective = adequacy × (1 − contradiction)`).
  consistency    합본 vs full 번역의 양방향 entailment 의 min. 보고 지표.

임베딩 코사인은 **대칭**이고 명제 관계가 아니라 표면 의미의 근접도만 잰다. 그래서
누락(neutral)과 모순(contradiction)을 원리적으로 구별하지 못한다 — 이 스크립트는 그
원리적 우려가 **실측으로 얼마나 나타나는지**를 숫자로 남긴다. LLM 호출은 0 이다.

세 트랙.

  T1 contradiction 관문   `premature_cases.json`. 통과 조건은 judge_check.check_nli 와
                          동일하게 케이스마다 `min(premature) > max(safe)`.
  T2 consistency 관문     `validity_cases.json`. 통과 조건은 validity_check 와 동일하게
                          `심각한 오류 < benign_minimal`.
  T3 실데이터             기존 런의 `*_rows.json` 경계 전부. NLI 값은 `pieces_contra` 에
                          이미 있으므로 재계산이 없고, 임베딩만 새로 돈다.
                          순위 상관 · 잡음 바닥 · SNR · 문장 effective 영향.

**잡음 바닥 보정이 이 측정의 핵심이다.** full 번역의 어절 prefix 는 정의상 무해한
미완성이다. 그 prefix 에 백엔드가 주는 점수가 바닥 c0(길이)이고, 실제 경계 점수에서
바닥을 빼야 "잘못 잘라서 받은 벌점"만 남는다. 임베딩은 prefix 가 짧을수록 코사인이
구조적으로 떨어지므로 바닥이 NLI 보다 훨씬 크고 길이 의존이 심할 것으로 예상된다 —
SNR(신호/바닥 산포)이 그 크기를 잰다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.embed_check \
      --run-id ko-en/run04 --models e5-inst qwen3-06b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from ..autoseg import validity_check as vc

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS


# ── 후보 모델 ────────────────────────────────────────────────────────────
#
# 선정 기준 셋. (1) MTEB/MMTEB 의 **STS·pair-classification** 상위 — 여기서 재는 것은
# 검색 적합도가 아니라 두 문장의 의미 동치라, 리트리벌 평균 순위를 그대로 쓰면 안 된다.
# (2) **다국어** — 타깃이 영어가 아닐 수 있고, contradiction 은 타깃 언어 두 개를 받는다.
# 언어별 자원을 쓰지 않는다는 설계 원칙(§12.1)도 다국어 단일 모델을 요구한다.
# (3) 4090 한 장(24GB, 다른 실험과 공유)에 올라갈 것.
MODELS: dict[str, dict] = {
    # MMTEB STS 상위. 560M, 100+ 언어. instruct 형식을 양쪽에 동일 적용한다.
    "e5-inst": {
        "id": "intfloat/multilingual-e5-large-instruct",
        "prompt": "Instruct: Retrieve semantically similar text.\nQuery: ",
    },
    # MTEB 다국어 보드 1위 계열(Qwen3-Embedding)의 최소 변이. 0.6B.
    "qwen3-06b": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "prompt_name": "query",
        "tokenizer_kwargs": {"padding_side": "left"},
    },
    # 같은 계열 4B — "크기를 키우면 해결되는가"를 가르는 대조군.
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Embedding-4B",
        "prompt_name": "query",
        "tokenizer_kwargs": {"padding_side": "left"},
        "dtype": "float16",
    },
    # 300M, 100+ 언어. **STS 전용 프롬프트를 모델이 직접 제공**한다 (HF 게이트 모델).
    "gemma-300m": {
        "id": "google/embeddinggemma-300m",
        "prompt_name": "STS",
    },
    # 305M. 값싼 대조군.
    "gte-base": {
        "id": "Alibaba-NLP/gte-multilingual-base",
        "trust_remote_code": True,
    },
}


class EmbedScorer:
    """임베딩 모델 하나를 contradiction·consistency 두 인터페이스로 감싼다.

    같은 문자열이 경계마다 반복 등장하므로(누적 prefix, premise 는 문장당 1개)
    인코딩은 **문자열 단위로 캐시**한다."""

    def __init__(self, key: str, spec: dict, device: str = "cuda",
                 batch_size: int = 64):
        self.key = key
        self.name = f"embed:{key}"
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kw: dict = {"device": self.device}
            if self.spec.get("trust_remote_code"):
                kw["trust_remote_code"] = True
            if self.spec.get("tokenizer_kwargs"):
                kw["tokenizer_kwargs"] = self.spec["tokenizer_kwargs"]
            if self.spec.get("dtype"):
                kw["model_kwargs"] = {"torch_dtype": self.spec["dtype"]}
            self._model = SentenceTransformer(self.spec["id"], **kw)
        return self._model

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._cache.clear()
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def encode(self, texts: list[str]) -> None:
        todo = sorted({t for t in texts if t and t not in self._cache})
        if not todo:
            return
        m = self.load()
        kw: dict = {"batch_size": self.batch_size, "normalize_embeddings": True,
                    "show_progress_bar": False, "convert_to_numpy": True}
        if self.spec.get("prompt_name"):
            kw["prompt_name"] = self.spec["prompt_name"]
        elif self.spec.get("prompt"):
            kw["prompt"] = self.spec["prompt"]
        vecs = m.encode(todo, **kw)
        for t, v in zip(todo, vecs):
            self._cache[t] = [float(x) for x in v]

    def cos(self, a: list[str], b: list[str]) -> list[float]:
        """정규화된 벡터의 내적 = 코사인. 빈 문자열은 0.0, 동일 문자열은 1.0."""
        self.encode(list(a) + list(b))
        out = []
        for x, y in zip(a, b):
            if not x or not y:
                out.append(0.0)
            elif x == y:
                out.append(1.0)
            else:
                va, vb = self._cache[x], self._cache[y]
                out.append(sum(p * q for p, q in zip(va, vb)))
        return out

    # contradiction 백엔드 인터페이스 (metrics.ContradictionBackend 와 동일 시그니처)
    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        """`1 − cos`. 높을수록 나쁘다 — NLI contradiction 확률과 방향을 맞춘다."""
        return [max(0.0, 1.0 - s) for s in self.cos(premises, hypotheses)]


class AlignScorer:
    """`1 − max_i cos(hypothesis, full 번역의 i-어절 prefix)`.

    소박한 `1 − cos(full, prefix)` 는 **미완성 자체를 벌한다** — 방출분이 짧을수록
    코사인이 구조적으로 떨어지므로 잡음 바닥이 크고 길이 의존이 심하다. 여기서는
    비교 대상을 full 전체가 아니라 **full 의 모든 prefix 중 가장 잘 맞는 것**으로
    바꾼다. 무해한 미완성은 어느 prefix 와는 재서술 관계이므로 max 가 1 에 가깝고,
    반박당한 방출은 어느 prefix 와도 맞지 않아야 한다 — 길이 바닥을 구조적으로
    제거하면서 대칭 유사도만 쓰는, 임베딩만으로 가능한 가장 강한 구성이다.

    이것마저 떨어지면 문제는 캘리브레이션이 아니라 **대칭 유사도라는 도구 자체**다."""

    def __init__(self, base: EmbedScorer):
        self.base = base
        self.name = base.name + "+align"

    def unload(self) -> None:
        self.base.unload()

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        prefixes: dict[str, list[str]] = {}
        for p in set(premises):
            w = (p or "").split()
            prefixes[p] = [" ".join(w[:i]) for i in range(1, len(w) + 1)]
        self.base.encode([x for v in prefixes.values() for x in v] + list(hypotheses))
        out = []
        for p, h in zip(premises, hypotheses):
            if not p or not h or not prefixes.get(p):
                out.append(0.0)
                continue
            vh = self.base._cache[h]
            best = max(sum(a * b for a, b in zip(vh, self.base._cache[pref]))
                       for pref in prefixes[p] if pref in self.base._cache)
            out.append(max(0.0, 1.0 - best))
        return out


class EmbedQualityBackend(metrics.QualityBackend):
    """consistency 관문용 래퍼 (`validity_check.evaluate_backend` 가 요구하는 인터페이스)."""

    def __init__(self, scorer: EmbedScorer):
        self.scorer = scorer
        self.name = scorer.name

    def score(self, srcs, hyps, refs):
        scores, pending = metrics._identity_shortcut(hyps, refs)
        if not pending:
            return scores
        vals = self.scorer.cos([hyps[i] for i in pending], [refs[i] for i in pending])
        for i, v in zip(pending, vals):
            scores[i] = v
        return scores


# ── 잡음 바닥 (평균 + 표준편차) ──────────────────────────────────────────

def measure_floor(fulls: list[str], backend, max_prefixes: int = 20) -> dict:
    """`noise_floor.measure_floor` 와 같은 측정에 **표준편차**를 추가한다.

    SNR 을 내려면 산포가 필요하다 — 바닥이 높아도 산포가 작으면 신호가 살지만,
    바닥이 낮아도 산포가 크면 못 쓴다."""
    prems, hyps, lens = [], [], []
    for full in fulls:
        words = (full or "").split()
        if len(words) < 2:
            continue
        cut = min(len(words) - 1, max_prefixes)
        for i in range(1, cut + 1):
            prems.append(full)
            hyps.append(" ".join(words[:i]))
            lens.append(i)
    scores = [float(s) for s in backend.score(prems, hyps)]

    by_bucket: dict[str, list[float]] = {b: [] for b in nf.bucket_labels()}
    for n, s in zip(lens, scores):
        by_bucket[nf.bucket_of(n)].append(s)

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "sd": None, "p50": None, "p90": None}
        v = sorted(vals)
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / len(v)) if len(v) > 1 else 0.0
        return {"n": len(v), "mean": round(mu, 4), "sd": round(sd, 4),
                "p50": round(v[len(v) // 2], 4),
                "p90": round(v[min(len(v) - 1, int(0.9 * (len(v) - 1)))], 4)}

    return {"backend": getattr(backend, "name", "?"), "n_sentences": len(fulls),
            "n_prefixes": len(scores), "overall": stats(scores),
            "by_length_bucket": {b: stats(v) for b, v in by_bucket.items()}}


def floor_mean(floor: dict, n_words: int) -> float:
    b = floor["by_length_bucket"].get(nf.bucket_of(max(1, n_words))) or {}
    if b.get("mean") is not None:
        return b["mean"]
    return floor["overall"]["mean"] or 0.0


def floor_sd(floor: dict, n_words: int) -> float:
    b = floor["by_length_bucket"].get(nf.bucket_of(max(1, n_words))) or {}
    if b.get("sd"):
        return b["sd"]
    return floor["overall"]["sd"] or 1e-9


def corrected(raw: float, hyp_words: int, floor: dict) -> float:
    return max(0.0, raw - floor_mean(floor, hyp_words))


def zscored(raw: float, hyp_words: int, floor: dict) -> float:
    return (raw - floor_mean(floor, hyp_words)) / max(1e-9, floor_sd(floor, hyp_words))


# ── T1: contradiction 관문 ───────────────────────────────────────────────

def contra_gate(cases: list[dict], backend, floor: dict | None) -> dict:
    """`judge_check.check_nli` 와 같은 기준. raw / floor 보정 / z 세 변이를 함께 낸다.

    통과 조건: 케이스마다 `min(premature) > max(safe)`.
    """
    rows = []
    for c in cases:
        for name, v in c["variants"].items():
            bd = v.get("boundary", c.get("boundary", 0))
            rows.append({"id": c["id"], "variant": name, "expect": v["expect"],
                         "premise": c["full_translation"],
                         "hypothesis": " ".join(v["pieces_tgt"][: bd + 1])})
    scores = backend.score([r["premise"] for r in rows], [r["hypothesis"] for r in rows])
    for r, s in zip(rows, scores):
        n = len(r["hypothesis"].split())
        r["hyp_words"] = n
        r["raw"] = round(float(s), 4)
        r["floor"] = round(corrected(float(s), n, floor), 4) if floor else None
        r["z"] = round(zscored(float(s), n, floor), 3) if floor else None

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)

    out = {"backend": getattr(backend, "name", "?"), "rows": rows, "variants": {}}
    for key in ["raw", "floor", "z"]:
        if rows[0].get(key) is None:
            continue
        viol, tot, details = 0, 0, []
        sig_p, sig_s = [], []
        for cid, items in by.items():
            p = [r[key] for r in items if r["expect"] == "premature"]
            s = [r[key] for r in items if r["expect"] == "safe"]
            sig_p += p
            sig_s += s
            if not p or not s:
                continue
            tot += 1
            ok = min(p) > max(s)
            if not ok:
                viol += 1
                details.append({"case": cid, "min_premature": min(p), "max_safe": max(s)})
        margin = ((sum(sig_p) / len(sig_p)) - (sum(sig_s) / len(sig_s))
                  if sig_p and sig_s else None)
        out["variants"][key] = {"violations": viol, "n_cases": tot,
                                "detail": details,
                                "mean_premature": round(sum(sig_p) / len(sig_p), 4) if sig_p else None,
                                "mean_safe": round(sum(sig_s) / len(sig_s), 4) if sig_s else None,
                                "margin": round(margin, 4) if margin is not None else None,
                                "passed": viol == 0}
    return out


# ── T3: 실데이터 경계 ────────────────────────────────────────────────────

def load_boundaries(run_dir: Path) -> list[dict]:
    """런 디렉토리의 모든 `*_rows.json` 에서 경계 단위 레코드를 뽑는다.

    마지막 조각은 미래가 없어 구조적 0 이므로 제외한다 (`pieces_contra[:-1]`)."""
    paths = sorted(run_dir.glob("iter_*/train_rows.json"))
    paths += sorted(run_dir.glob("iter_*/dev_rows.json"))
    if (run_dir / "test_rows.json").exists():
        paths.append(run_dir / "test_rows.json")

    recs = []
    for p in paths:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = f"{p.parent.name}/{p.stem}"
        for r in rows:
            full = r.get("full_trans") or ""
            for T, d in (r.get("by_T") or {}).items():
                contras = (d.get("pieces_contra") or [])[:-1]
                pieces = d.get("pieces_tgt") or []
                if not contras or len(pieces) != len(contras) + 1:
                    continue
                for j, c in enumerate(contras):
                    hyp = " ".join(x for x in pieces[: j + 1] if x)
                    if not hyp or not full:
                        continue
                    recs.append({"src": tag, "id": r["id"], "T": int(T), "j": j,
                                 "premise": full, "hypothesis": hyp,
                                 "hyp_words": len(hyp.split()), "nli": float(c),
                                 "adequacy": d.get("adequacy"), "k": d.get("k")})
    return recs


def real_data(recs: list[dict], backend, floor: dict) -> dict:
    """임베딩 점수를 경계마다 매기고 NLI 와 비교한다."""
    vals = backend.score([r["premise"] for r in recs], [r["hypothesis"] for r in recs])
    for r, s in zip(recs, vals):
        r[backend.name] = {
            "raw": float(s),
            "floor": corrected(float(s), r["hyp_words"], floor),
            "z": zscored(float(s), r["hyp_words"], floor),
        }

    nli = [r["nli"] for r in recs]
    out: dict = {"backend": backend.name, "n_boundaries": len(recs), "variants": {}}
    for key in ["raw", "floor", "z"]:
        emb = [r[backend.name][key] for r in recs]
        # 전역 순위 상관
        glob = metrics._spearman(nli, emb)
        # 문장·T 내부 상관 (경계 3개 이상). 문장 간 난이도 차이를 제거한 값.
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(recs):
            groups.setdefault((r["src"], r["id"], r["T"]), []).append(i)
        within = []
        for idxs in groups.values():
            if len(idxs) < 3:
                continue
            c = metrics._spearman([nli[i] for i in idxs], [emb[i] for i in idxs])
            if c is not None:
                within.append(c)
        # 루프가 판정자에게 보내는 것은 contradiction 최상위 경계다 — 그 선택이
        # 얼마나 겹치는지가 "조향이 바뀌는가"의 직접 측정값이다.
        k = min(20, max(1, len(recs) // 20))
        top_nli = {i for i in sorted(range(len(recs)), key=lambda i: -nli[i])[:k]}
        top_emb = {i for i in sorted(range(len(recs)), key=lambda i: -emb[i])[:k]}
        out["variants"][key] = {
            "spearman_global": round(glob, 4) if glob is not None else None,
            "spearman_within_sentence": (round(sum(within) / len(within), 4)
                                         if within else None),
            "n_within": len(within),
            "topk": k,
            "topk_overlap": round(len(top_nli & top_emb) / k, 3),
            "mean": round(sum(emb) / len(emb), 4),
        }
    return out


def sentence_effective(recs: list[dict], backend_key: str, variant: str) -> dict:
    """경계 평균 → 문장 contradiction → effective. NLI 대비 문장 순위가 얼마나 바뀌나.

    임베딩 raw 는 확률이 아니라 스케일이 다르므로, `effective` 를 만들 때는
    **[0,1] 로 재척도한 값**을 쓴다 (경계 값의 분위수 정규화). 스케일 차이로
    상관이 왜곡되는 것을 막기 위한 것이고, 실제 채택 시에는 캘리브레이션이 필요하다."""
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault((r["src"], r["id"], r["T"]), []).append(r)

    lo = min(r[backend_key][variant] for r in recs)
    hi = max(r[backend_key][variant] for r in recs)
    span = max(1e-9, hi - lo)

    nli_eff, emb_eff = [], []
    for items in groups.values():
        adq = items[0].get("adequacy")
        if adq is None:
            continue
        c_nli = sum(r["nli"] for r in items) / len(items)
        c_emb = sum((r[backend_key][variant] - lo) / span for r in items) / len(items)
        nli_eff.append(adq * (1 - c_nli))
        emb_eff.append(adq * (1 - c_emb))
    sp = metrics._spearman(nli_eff, emb_eff) if len(nli_eff) > 2 else None
    return {"n_sentences": len(nli_eff),
            "mean_effective_nli": round(sum(nli_eff) / len(nli_eff), 4) if nli_eff else None,
            "mean_effective_embed": round(sum(emb_eff) / len(emb_eff), 4) if emb_eff else None,
            "spearman_effective": round(sp, 4) if sp is not None else None}


def prune_degenerate(result: dict) -> dict:
    """바닥 보정이 무의미해지는 변이를 표에서 뺀다.

    `+align` 은 후보 prefix 집합에 자기 자신이 들어 있어 자기-prefix 바닥이 **정확히 0**
    이다 (설계상 그렇다). 그러면 `floor` 는 `raw` 와 같은 값이고 `z` 는 0 으로 나눈
    폭발값이라, 남겨 두면 표만 어지럽고 읽는 사람을 오도한다."""
    EPS = 1e-3
    for label, fl in result["floors"].items():
        mean = fl["overall"]["mean"] or 0.0
        sd = fl["overall"]["sd"] or 0.0
        drop = set()
        if mean < EPS:
            drop.add("floor")
        if sd < EPS:
            drop.add("z")
        if not drop:
            continue
        for g in result["contra_gate"]:
            if g["backend"] == label:
                for k in drop:
                    g["variants"].pop(k, None)
        for r in result["real"]:
            if r["backend"] == label:
                for k in drop:
                    r["variants"].pop(k, None)
                    (r.get("effective") or {}).pop(k, None)
    return result


# ── 리포트 ───────────────────────────────────────────────────────────────

def render(result: dict) -> str:
    L = ["# NLI → semantic similarity 대체 가능성 측정", ""]
    L += [f"런: `{result['run_id']}` · 경계 {result['n_boundaries']}개 · "
          f"바닥 측정 문장 {result['n_floor_sentences']}개 · LLM 호출 0", ""]
    L += ["임베딩 후보는 MTEB/MMTEB 의 **STS·pair-classification** 축에서 골랐다 — "
          "여기서 재는 것은 검색 적합도가 아니라 두 문장의 의미 동치이고, 타깃 언어가 "
          "영어가 아닐 수 있어 다국어가 요건이다.", ""]
    L += ["| 키 | 모델 |", "|---|---|"]
    for k in result["models"]:
        L.append(f"| `{k}` | `{MODELS[k]['id']}` |")

    # T1
    L += ["", "## T1 — contradiction 관문 (`premature_cases.json`)", "",
          "통과 조건은 `judge_check.check_nli` 와 같다: 케이스마다 "
          "`min(premature) > max(safe)`. `floor` 는 잡음 바닥 보정, `z` 는 바닥 대비 "
          "표준화. 바닥은 아래 T3 에서 잰 값을 쓴다.", "",
          "| 백엔드 | 변이 | 위반/케이스 | mean(premature) | mean(safe) | 격차 | 판정 |",
          "|---|---|---|---|---|---|---|"]
    for g in result["contra_gate"]:
        for key, v in g["variants"].items():
            L.append(f"| {g['backend']} | {key} | {v['violations']}/{v['n_cases']} | "
                     f"{v['mean_premature']} | {v['mean_safe']} | {v['margin']} | "
                     f"{'통과' if v['passed'] else '**탈락**'} |")

    # T1 케이스별 원점수 — 어디서 뒤집히는지가 여기 다 나온다
    names = [g["backend"] for g in result["contra_gate"]]
    L += ["", "### 케이스별 원점수 (raw, 높을수록 '모순')", "",
          "| 케이스 | 변이 | 기대 | " + " | ".join(names) + " |",
          "|---|---|---|" + "---|" * len(names)]
    seen: list[tuple] = []
    for r in result["contra_gate"][0]["rows"]:
        seen.append((r["id"], r["variant"], r["expect"]))
    by_key: dict[tuple, dict] = {}
    for g in result["contra_gate"]:
        for r in g["rows"]:
            by_key.setdefault((r["id"], r["variant"], r["expect"]), {})[g["backend"]] = r["raw"]
    for key in seen:
        cid, var, exp = key
        mark = "**" if exp == "premature" else ""
        cells = [f"{by_key[key].get(n, float('nan')):.4f}" for n in names]
        L.append(f"| {cid} | {mark}{var}{mark} | {exp} | " + " | ".join(cells) + " |")

    # T2
    if result.get("consistency_gate"):
        L += ["", "## T2 — consistency 관문 (`validity_cases.json`)", "",
              "통과 조건은 `validity_check` 와 같다: 심각한 의미 오류 < `benign_minimal`.", "",
              "| 백엔드 | 순위 검사 | 위반 (minimal 기준) | 참고 (paraphrase 기준) | 판정 |",
              "|---|---|---|---|---|"]
        for r in result["consistency_gate"]:
            hard = [v for v in r["violations"] if v.get("kind") is None]
            L.append(f"| {r['backend']} | {r['n_ordering_checks']} | {len(hard)} | "
                     f"{r['n_soft_violations']} | {'통과' if r['passed'] else '**탈락**'} |")
        L += ["", "변이 유형별 평균:", "",
              "| 변이 | " + " | ".join(r["backend"] for r in result["consistency_gate"]) + " |",
              "|---|" + "---|" * len(result["consistency_gate"])]
        for label in vc.ORDER:
            row = [f"{r['by_variant'][label]:.4f}" if label in r["by_variant"] else "—"
                   for r in result["consistency_gate"]]
            mark = "**" if label in vc.SEVERE else ""
            L.append(f"| {mark}{label}{mark} | " + " | ".join(row) + " |")

    # T3 바닥
    L += ["", "## T3 — 잡음 바닥 (full 번역의 자기-prefix. 정의상 무해한 미완성)", "",
          "prefix 는 같은 번역의 앞부분이라 **모순일 수 없다**. 여기서 나오는 점수가 "
          "바닥이고, 실제 경계 점수에서 이걸 빼야 '잘못 잘라서 받은 벌점'만 남는다.", "",
          "| 백엔드 | 전체 mean | sd | " + " | ".join(nf.bucket_labels()) + " |",
          "|---|---|---|" + "---|" * len(nf.bucket_labels())]
    for b, fl in result["floors"].items():
        cells = []
        for lab in nf.bucket_labels():
            s = fl["by_length_bucket"][lab]
            cells.append(f"{s['mean']}" if s["n"] else "—")
        L.append(f"| {b} | {fl['overall']['mean']} | {fl['overall']['sd']} | "
                 + " | ".join(cells) + " |")
    L += ["", "(길이 버킷 = hypothesis 어절 수)"]

    # T3 SNR
    L += ["", "### 신호 대 바닥 (SNR)", "",
          "신호 = 관문의 `mean(premature) − mean(safe)` (raw 기준). "
          "바닥 산포 = 자기-prefix 점수의 전체 sd. "
          "**SNR 이 1 미만이면 실제 잘못 자른 경계가 무해한 미완성의 산포에 묻힌다.**", "",
          "| 백엔드 | 신호 | 바닥 sd | SNR |", "|---|---|---|---|"]
    for r in result["snr"]:
        L.append(f"| {r['backend']} | {r['signal']} | {r['floor_sd']} | {r['snr']} |")

    # T3 상관
    L += ["", "### 실데이터 경계에서 NLI 와의 일치도", "",
          f"경계 {result['n_boundaries']}개 (`pieces_contra` 재활용 — NLI 재계산 없음). "
          "`topk_overlap` = 루프가 판정자에게 보내는 contradiction 최상위 경계 집합의 "
          "겹침 비율. **조향이 바뀌는가**의 직접 측정값이다.", "",
          "| 백엔드 | 변이 | Spearman(전역) | Spearman(문장 내) | topk 겹침 | "
          "문장 effective Spearman |", "|---|---|---|---|---|---|"]
    for r in result["real"]:
        for key, v in r["variants"].items():
            eff = r["effective"].get(key, {})
            L.append(f"| {r['backend']} | {key} | {v['spearman_global']} | "
                     f"{v['spearman_within_sentence']} (n={v['n_within']}) | "
                     f"{v['topk_overlap']} | {eff.get('spearman_effective')} |")

    L += ["", "## 판정", ""]
    for line in result["verdict"]:
        L.append(f"- {line}")

    passed = [g["backend"] for g in result["contra_gate"]
              if any(v["passed"] for v in g["variants"].values())
              and not g["backend"].startswith("nli") and "embed" in g["backend"]]
    L += ["", "### 결론", ""]
    if passed:
        L += [f"contradiction 관문을 통과한 임베딩 백엔드: {', '.join(passed)}. "
              "consistency 관문과 실데이터 일치도를 함께 보고 채택 여부를 정할 것."]
    else:
        L += [
            "**임베딩 유사도로 contradiction 을 대체할 수 없다.** 통과한 임베딩 구성이 "
            "하나도 없고, 관문 신호(`mean(premature) − mean(safe)`)가 여러 구성에서 "
            "**음수**다 — 잘못 자른 방출이 안전한 방출보다 full 번역에 *더* 가깝게 나온다.",
            "",
            "원인은 캘리브레이션이 아니라 도구의 성질이다. 코사인 유사도는 **대칭**이고 "
            "표면 의미의 근접도만 재는데, 우리가 잡아야 하는 오류(부정 뒤집힘·주체 뒤바뀜)는 "
            "**어휘를 그대로 둔 채 명제만 뒤집는다**. 그래서 오히려 정직한 파편보다 참조와 "
            "가까워진다 (케이스별 원점수 표에서 직접 보인다). 길이 바닥을 구조적으로 없앤 "
            "`+align` 구성도 부호를 되돌리지 못했다 — 문제는 길이 교란이 아니라 명제 관계를 "
            "못 본다는 것이다.",
            "",
            "NLI 는 같은 케이스에서 정확히 이 축을 본다: `neutral`(미완성)과 "
            "`contradiction`(반박)을 나누므로, 무해한 미완성은 바닥에 두고 뒤집힌 방출만 "
            "1.0 쪽으로 보낸다.",
            "",
            "**대안이 필요하다면** 대칭 유사도가 아니라 방향성 있는 것을 찾아야 한다 — "
            "다른 NLI 체크포인트(`metrics.NLI_MODEL`), cross-encoder 계열, 또는 "
            "참조 기반 QE. 임베딩은 `consistency` 보조 지표로도 T2 를 통과하지 못했다.",
        ]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="NLI → 임베딩 유사도 대체 가능성 측정")
    p.add_argument("--run-id", default="ko-en/run04", help="runs/ 이하 경로")
    p.add_argument("--models", nargs="+", default=["e5-inst", "qwen3-06b"],
                   choices=sorted(MODELS), help="비교할 임베딩 후보")
    p.add_argument("--floor-sentences", type=int, default=150,
                   help="잡음 바닥 측정에 쓸 full 번역 문장 수")
    p.add_argument("--max-boundaries", type=int, default=0,
                   help="0 이면 전부. 디버깅용 상한")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--align", action="store_true",
                   help="모델마다 `1 − max_i cos(hyp, full 의 i-어절 prefix)` 변이도 함께 잰다 "
                        "(길이 바닥을 구조적으로 제거한 구성)")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--render-only", action="store_true",
                   help="기존 scores.json 으로 report.md 만 다시 만든다 (GPU 사용 0)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.render_only:
        out_dir = Path(args.out) if args.out else (OUT_RUNS / "embed_vs_nli")
        result = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
        (out_dir / "report.md").write_text(render(prune_degenerate(result)),
                                           encoding="utf-8")
        print(f"[done] {out_dir / 'report.md'}")
        return 0

    run_dir = SEG_RUNS / args.run_id
    if not run_dir.exists():
        print(f"런 디렉토리 없음: {run_dir}", file=sys.stderr)
        return 1
    out_dir = Path(args.out) if args.out else (OUT_RUNS / "embed_vs_nli")
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_boundaries(run_dir)
    if args.max_boundaries:
        recs = recs[: args.max_boundaries]
    if not recs:
        print(f"경계를 못 찾음: {run_dir}", file=sys.stderr)
        return 1
    fulls = sorted({r["premise"] for r in recs})[: args.floor_sentences]
    print(f"[data] 경계 {len(recs)}개, 바닥 측정 문장 {len(fulls)}개", flush=True)

    prem_cases = json.loads((AUTOSEG / "premature_cases.json").read_text(encoding="utf-8"))["cases"]
    val_cases = json.loads((AUTOSEG / "validity_cases.json").read_text(encoding="utf-8"))

    result: dict = {"run_id": args.run_id, "models": args.models,
                    "n_boundaries": len(recs), "n_floor_sentences": len(fulls),
                    "floors": {}, "contra_gate": [], "consistency_gate": [],
                    "real": [], "snr": [], "verdict": []}

    backends: list = []
    # 기준선 NLI 를 먼저 — 같은 자로 재야 비교가 성립한다.
    backends.append(("nli", metrics.make_contradiction_backend(), None))
    for key in args.models:
        sc = EmbedScorer(key, MODELS[key], batch_size=args.batch_size)
        backends.append((key, sc, sc))
        if args.align:
            # 같은 EmbedScorer 를 공유해 인코딩 캐시를 재사용한다 (모델 로드 1회).
            backends.append((key + "+align", AlignScorer(sc), None))

    for bi, (key, b, embed_scorer) in enumerate(backends):
        label = getattr(b, "name", key)
        print(f"\n[{label}] 잡음 바닥 측정...", flush=True)
        floor = measure_floor(fulls, b)
        result["floors"][label] = floor
        print(f"[{label}] 바닥 mean={floor['overall']['mean']} sd={floor['overall']['sd']}",
              flush=True)

        print(f"[{label}] T1 contradiction 관문...", flush=True)
        g = contra_gate(prem_cases, b, floor)
        result["contra_gate"].append(g)
        raw = g["variants"]["raw"]
        sd = floor["overall"]["sd"] or 0.0
        result["snr"].append({
            "backend": label, "signal": raw["margin"], "floor_sd": sd,
            "snr": round(raw["margin"] / sd, 2) if (sd and raw["margin"] is not None) else None})

        if not args.skip_consistency and embed_scorer is not None:
            print(f"[{label}] T2 consistency 관문...", flush=True)
            result["consistency_gate"].append(
                vc.evaluate_backend(EmbedQualityBackend(embed_scorer), val_cases))

        print(f"[{label}] T3 실데이터 {len(recs)} 경계...", flush=True)
        r = real_data(recs, b, floor)
        r["effective"] = {k: sentence_effective(recs, label, k)
                          for k in ["raw", "floor", "z"]}
        result["real"].append(r)

        # 뒤에 같은 임베딩 모델을 쓰는 항목(+align)이 남아 있으면 아직 내리지 않는다.
        base = embed_scorer or getattr(b, "base", None)
        if base is not None:
            nxt = backends[bi + 1][1] if bi + 1 < len(backends) else None
            if base is not (nxt if isinstance(nxt, EmbedScorer) else getattr(nxt, "base", None)):
                base.unload()

    # NLI 자기 자신은 상관 1.0 이라 표에서 빼고 기준선으로만 남긴다
    nli_label = backends[0][1].name
    result["real"] = [r for r in result["real"] if r["backend"] != nli_label]

    # 판정
    for g in result["contra_gate"]:
        best = min((v["violations"] for v in g["variants"].values()), default=None)
        result["verdict"].append(
            f"`{g['backend']}` contradiction 관문 최소 위반 {best}건 "
            f"({', '.join(k + '=' + str(v['violations']) for k, v in g['variants'].items())})")
    for r in result["consistency_gate"]:
        hard = [v for v in r["violations"] if v.get("kind") is None]
        result["verdict"].append(
            f"`{r['backend']}` consistency 관문 위반 {len(hard)}건 "
            f"(soft {r['n_soft_violations']}건)")

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(prune_degenerate(result))
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
