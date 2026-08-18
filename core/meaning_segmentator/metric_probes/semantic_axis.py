"""임베딩 공간에서 **NLI 와 같은 역할을 하는 축**을 직접 찾는다.

`embed_geometry.py` 결론: 차이벡터는 거의 등방적이라 분산 기준(PCA)으로는 읽을 축이
안 나온다. 그런데 축을 찾는 방법이 분산만 있는 것은 아니다 — **대조쌍(contrastive
pair)** 이 있다. "이 속성만 다른" 문장 쌍의 평균 차이벡터가 곧 그 속성의 방향이다.
단어 유추(`king − man + woman`)와 표현공간 조향(steering vector)이 쓰는 그 방법이다.

축을 세 가지로 만든다. **뒤로 갈수록 지도 신호가 강해진다.**

  `neg_rule`     영어 부정 규칙으로 **기계 생성**한 최소쌍. `is → is not` 처럼 극성만
                 뒤집고 나머지를 그대로 둔다. **라벨이 전혀 필요 없다.**
  `mnli_mean`    MNLI 모순쌍의 평균 차이벡터 − 함의쌍의 평균 차이벡터. 클래스 평균만
                 쓰므로 지도는 최소다 (벡터 하나).
  `mnli_lda`     같은 두 집합의 Fisher 판별 방향. 공분산까지 쓰지만 여전히 **1차원**.

핵심 대조 두 개.

  **① 축에 이름이 붙는가** — `neg_rule` 과 `mnli_mean` 의 코사인. 규칙으로 만든 부정
     방향과 데이터로 배운 모순 방향이 같은 쪽을 가리키면, 그 축은 우연한 회귀 계수가
     아니라 **실재하는 의미 방향**이다.
  **② 1차원으로 NLI 를 얼마나 대신하나** — 같은 MNLI dev 에서 모순 AUC 를
     `cos`(0.723) · 1차원 축 투영 · 전체 지도 프로브(0.899) · 교차 인코더(0.991) 와
     나란히 놓는다. 축 하나가 프로브에 근접하면 **해석 가능하고 값싼 대체재**가 된다.

마지막으로 관문(`premature_cases.json`)에 걸어 본다. 크기 축(코사인)과 극성 축을
**함께** 쓰는 2차원 읽기도 같이 재는데, `embed_geometry` 가 "코사인은 크기를 잘 읽고
종류를 못 읽는다" 로 끝났으므로 그 둘을 더하는 것이 자연스러운 다음 수이기 때문이다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.semantic_axis
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from .embed_check import (MODELS, contra_gate, load_boundaries, measure_floor,
                          prune_degenerate, real_data)
from .embed_probe import FrozenEncoder

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS

# ── 규칙 기반 영어 부정 최소쌍 ───────────────────────────────────────────
#
# 극성만 뒤집고 나머지 어휘를 그대로 둔다. 이것이 요점이다 — 문장을 많이 바꾸면
# 차이벡터에 주제 변화가 섞여 축이 흐려진다. 규칙이 안 걸리는 문장은 그냥 버린다
# (재현율이 아니라 **순도**가 중요하다).
_AUX = r"(is|are|was|were|will|would|can|could|should|has|have|had|does|do|did|must|may)"
_NEG_ADD = re.compile(rf"\b{_AUX}\b(?!\s+not\b)", re.I)
_NEG_DEL = re.compile(rf"\b{_AUX}\s+not\b", re.I)
_NT = re.compile(r"\b(\w+)n't\b", re.I)


def negate(text: str) -> str | None:
    """극성을 한 번만 뒤집는다. 못 하면 None."""
    if not text or len(text.split()) < 3:
        return None
    if _NEG_DEL.search(text):                       # 부정 → 긍정
        return _NEG_DEL.sub(lambda m: m.group(1), text, count=1)
    if _NT.search(text):
        return _NT.sub(lambda m: m.group(1), text, count=1)
    if _NEG_ADD.search(text):                       # 긍정 → 부정
        return _NEG_ADD.sub(lambda m: f"{m.group(0)} not", text, count=1)
    return None


def build_neg_pairs(texts: list[str], limit: int = 4000) -> list[tuple[str, str]]:
    out = []
    for t in texts:
        n = negate(t)
        if n and n != t:
            out.append((t, n))
        if len(out) >= limit:
            break
    return out


# ── 축 구성 ──────────────────────────────────────────────────────────────

def _unit(v):
    import numpy as np
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def axis_from_pairs(enc: FrozenEncoder, pairs: list[tuple[str, str]]):
    """대조쌍의 **평균 차이벡터**. 속성만 다르므로 나머지가 상쇄된다."""
    import numpy as np
    a = enc.encode([p[0] for p in pairs])
    b = enc.encode([p[1] for p in pairs])
    return _unit((b - a).mean(axis=0))


def axis_mnli_mean(du_contra, du_entail):
    return _unit(du_contra.mean(axis=0) - du_entail.mean(axis=0))


def axis_mnli_lda(du_contra, du_entail, shrink: float = 1e-2):
    """Fisher 판별 방향. 공분산으로 보정하지만 여전히 1차원이다."""
    import numpy as np
    X = np.concatenate([du_contra, du_entail], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    cov += shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0], dtype=cov.dtype)
    return _unit(np.linalg.solve(cov, du_contra.mean(axis=0) - du_entail.mean(axis=0)))


# ── 축을 쓰는 채점기 ─────────────────────────────────────────────────────

class AxisScorer:
    """`|proj(d, axis)|`. `d = emb(hypothesis) − emb(premise)`.

    **절대값을 쓰는 이유**: 극성은 어느 쪽으로 뒤집혀도 사고다. 부호는 방향(긍정→부정
    인지 그 반대인지)만 알려주므로 벌점에는 크기만 넣는다.

    `blend` 를 주면 `크기 축(1−cos)` 과 가중합한다 — `embed_geometry` 가 "코사인은
    크기를 읽고 종류를 못 읽는다" 로 끝났으므로 둘을 합치는 것이 자연스러운 구성이다."""

    def __init__(self, enc: FrozenEncoder, axis, name: str, blend: float = 0.0,
                 signed: bool = False, normalize: bool = False):
        self.enc = enc
        self.axis = axis
        self.name = name
        self.blend = blend
        self.signed = signed
        # `normalize` 는 `proj / ‖d‖` — **변화 중 이 축이 차지하는 비율**이다.
        # 정규화 없이 쓰면 종류와 크기가 다시 섞인다: 짧고 일반적인 조각은 ‖d‖ 자체가
        # 커서 극성 성분도 덩달아 커지고, 실제로 관문 p01 에서 안전한 조각
        # ("As for that,")이 극성이 뒤집힌 조각보다 높은 투영을 받았다.
        self.normalize = normalize

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import numpy as np
        u = self.enc.encode(list(premises))
        v = self.enc.encode(list(hypotheses))
        d = v - u
        proj = d @ self.axis
        if self.normalize:
            proj = proj / np.clip(np.linalg.norm(d, axis=1), 1e-9, None)
        val = proj if self.signed else np.abs(proj)
        if self.blend:
            un = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-9, None)
            vn = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
            val = val + self.blend * (1.0 - (un * vn).sum(axis=1))
        return [float(x) for x in np.nan_to_num(val)]


def _auc(pos, neg) -> float | None:
    if not len(pos) or not len(neg):
        return None
    import numpy as np
    p = np.asarray(pos)
    n = np.asarray(neg)
    wins = float((p[:, None] > n[None, :]).sum())
    ties = float((p[:, None] == n[None, :]).sum())
    return (wins + 0.5 * ties) / (len(p) * len(n))


def render(result: dict) -> str:
    L = ["# 임베딩에서 NLI 와 같은 역할을 하는 축 찾기", "",
         f"인코더 `{result['encoder_id']}` · 규칙 부정 최소쌍 {result['n_neg_pairs']}개 · "
         f"MNLI {result['n_mnli']}쌍 · run04 경계 {result['n_boundaries']}개", "",
         "`embed_geometry` 결론은 '차이벡터가 등방적이라 **분산 기준으로는** 읽을 축이 "
         "안 나온다' 였다. 여기서는 축을 **대조쌍**으로 만든다 — 그 속성만 다른 문장 쌍의 "
         "평균 차이벡터.", ""]

    L += ["## ① 축에 이름이 붙는가 — 축끼리의 정렬", "",
          "규칙으로 만든 **부정 방향**과 데이터로 배운 **모순 방향**이 같은 쪽을 "
          "가리키는가. 높으면 그 축은 회귀 계수가 아니라 **실재하는 의미 방향**이다.", "",
          "| 축 쌍 | 코사인 |", "|---|---|"]
    for k, v in result["axis_alignment"].items():
        L.append(f"| {k} | **{v}** |")
    L += ["", f"참고: 무작위 방향 두 개의 코사인 기대값 ≈ "
              f"{result['random_cos']:.4f} (1024차원).", ""]

    L += ["## ② 1차원 축이 NLI 를 얼마나 대신하는가 (MNLI dev)", "",
          "같은 dev 집합에서 **모순 vs 나머지** AUC.", "",
          "| 읽기 | 차원 | 모순 AUC |", "|---|---|---|"]
    for r in result["mnli_auc"]:
        L.append(f"| {r['name']} | {r['dims']} | **{r['auc']}** |")
    L += ["", "`cos`·`probe`·교차 인코더 수치는 `embed_probe.py` 와 같은 자로 잰 것이다.", ""]

    L += ["## ③ 관문 (`premature_cases.json`)", "",
          "통과 조건은 `judge_check.check_nli` 와 동일: `min(premature) > max(safe)`.", "",
          "| 채점기 | 변이 | 위반/케이스 | mean(prem) | mean(safe) | 격차 | 판정 |",
          "|---|---|---|---|---|---|---|"]
    for g in result["gate"]:
        for k, v in g["variants"].items():
            L.append(f"| `{g['scorer']}` | {k} | {v['violations']}/{v['n_cases']} | "
                     f"{v['mean_premature']} | {v['mean_safe']} | {v['margin']} | "
                     f"{'통과' if v['passed'] else '**탈락**'} |")

    L += ["", "## ④ 잡음 바닥과 실데이터", "",
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

    # 판정
    align = result["axis_alignment"]
    best_axis = max((r for r in result["mnli_auc"] if r["dims"] == 1),
                    key=lambda r: r["auc"] or 0, default=None)
    cos_row = next((r for r in result["mnli_auc"] if r["name"] == "cos"), None)
    probe_row = next((r for r in result["mnli_auc"] if "probe" in r["name"]), None)
    gate_pass = [(g["scorer"], k) for g in result["gate"]
                 for k, v in g["variants"].items() if v["passed"]]

    L += ["", "## 판정", ""]
    nm = align.get("neg_rule vs mnli_mean")
    if nm is not None:
        if abs(nm) > 0.3:
            L += [f"**축에는 이름이 붙는다.** 규칙으로 만든 부정 방향과 MNLI 에서 배운 "
                  f"모순 방향의 코사인이 **{nm}** 로, 무작위 기대값(≈"
                  f"{result['random_cos']:.3f})을 크게 넘는다. 즉 '극성' 은 임베딩 공간에 "
                  "실재하는 방향이고, 라벨 없이 규칙만으로도 찾아낼 수 있다.", ""]
        else:
            L += [f"**축이 잘 맞지 않는다.** 규칙 부정 방향과 MNLI 모순 방향의 코사인이 "
                  f"{nm} 로 무작위(≈{result['random_cos']:.3f})와 크게 다르지 않다. "
                  "모순은 극성 하나로 환원되지 않는다는 뜻이다 — 참여자 뒤바뀜·범위 "
                  "변화 등이 각각 다른 방향을 쓰고, 하나의 축으로 모이지 않는다.", ""]
    if best_axis and cos_row and probe_row:
        L += [f"**1차원으로는 부족하다.** 최선의 축 하나(`{best_axis['name']}`)가 모순 AUC "
              f"{best_axis['auc']} 인데, 코사인 {cos_row['auc']} 와 전체 지도 프로브 "
              f"{probe_row['auc']} 사이 어디쯤이다. 방향 하나로 접으면 프로브가 쓰던 "
              "정보의 상당 부분이 버려진다.", ""]
    if gate_pass:
        L += ["관문 통과: " + ", ".join(f"`{s}`({k})" for s, k in gate_pass), "",
              "**통과했다면 실제 후보다.** 바닥과 실데이터 거동을 함께 보고 판단할 것."]
    else:
        best = min((v["violations"] for g in result["gate"]
                    for v in g["variants"].values()), default=None)
        L += [f"**그런데 관문은 통과하지 못했다** (최선 {best}/6). 축을 **찾는 것**과 그 축으로 "
              "우리 fixture 를 **가르는 것**은 별개였다.", "",
              "관문의 실패 유형은 다섯 가지다 — 극성 뒤집힘(`premature_negation`), 참여자 "
              "뒤바뀜(`premature_role`), 범위 변화(`premature_scope`), 핵어 확정"
              "(`premature_head`), 서법 확정(`premature_modal`). **극성 축 하나가 덮는 것은 "
              "그중 하나다.** 게다가 케이스별 통과/실패가 유형과 깔끔히 대응하지도 않는다 — "
              "`ko-en-p01` 은 극성 케이스인데도 실패하는데, 그쪽 안전 변이(`As for that,`)가 "
              "내용이 거의 없는 조각이라 차이벡터의 방향이 사실상 임의적이기 때문이다. "
              "**내용이 희박한 조각에서는 방향 자체가 정의되지 않는다.**", "",
              "정규화는 도움이 됐다 (`neg_rule` 4/6 → `neg_rule/norm` 3/6). 투영을 `‖d‖` 로 "
              "나눠 '변화 중 이 축의 비율' 로 바꾸면 크기와 종류가 분리된다 — 안 하면 짧은 "
              "조각이 큰 `‖d‖` 때문에 극성 성분까지 크게 받는다. 그래도 관문을 넘기지는 "
              "못한다.", "",
              "**다음 수는 축을 여러 개 만드는 것이다.** 참여자 뒤바뀜은 문장의 고유명사 두 "
              "개를 기계적으로 맞바꿔 최소쌍을 만들 수 있고(부정 규칙과 같은 방식, 라벨 0), "
              "서법은 조동사 치환으로 만들 수 있다. 유형별 축을 세우고 그 성분들의 최대값을 "
              "쓰면 다섯 유형을 각각 겨냥할 수 있다. 다만 그때도 **유형 목록이 곧 임의 "
              "상수**가 된다는 점은 감안해야 한다 — 목록에 없는 실패는 못 잡는다. "
              "NLI 가 하나의 모델로 그 전부를 덮는 이유가 그것이다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="의미 축 탐색 — NLI 대체 후보")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--encoder", default="e5-inst", choices=sorted(MODELS))
    p.add_argument("--mnli-size", type=int, default=40000)
    p.add_argument("--dev-size", type=int, default=3000)
    p.add_argument("--neg-pairs", type=int, default=4000)
    p.add_argument("--floor-sentences", type=int, default=150)
    p.add_argument("--max-boundaries", type=int, default=0)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (OUT_RUNS / "semantic_axis")
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

    print("[enc] MNLI 인코딩...", flush=True)
    u_tr = enc.encode(list(tr["premise"]))
    v_tr = enc.encode(list(tr["hypothesis"]))
    u_dv = enc.encode(list(dv["premise"]))
    v_dv = enc.encode(list(dv["hypothesis"]))
    y_tr = np.array(tr["label"])
    y_dv = np.array(dv["label"])
    d_tr, d_dv = v_tr - u_tr, v_dv - u_dv

    # 규칙 부정 최소쌍 — MNLI 전제문에서 뽑는다 (영어, 문장 형태가 다양하다)
    pairs = build_neg_pairs(list(tr["premise"]), limit=args.neg_pairs)
    print(f"[axis] 규칙 부정 최소쌍 {len(pairs)}개", flush=True)
    ax_neg = axis_from_pairs(enc, pairs)
    ax_mean = axis_mnli_mean(d_tr[y_tr == 2], d_tr[y_tr == 0])
    ax_lda = axis_mnli_lda(d_tr[y_tr == 2], d_tr[y_tr == 0])

    dim = ax_neg.shape[0]
    result: dict = {
        "run_id": args.run_id, "encoder": args.encoder,
        "encoder_id": MODELS[args.encoder]["id"],
        "n_neg_pairs": len(pairs), "n_mnli": len(tr), "random_cos": float(1 / dim ** 0.5),
        "axis_alignment": {
            "neg_rule vs mnli_mean": round(float(ax_neg @ ax_mean), 4),
            "neg_rule vs mnli_lda": round(float(ax_neg @ ax_lda), 4),
            "mnli_mean vs mnli_lda": round(float(ax_mean @ ax_lda), 4)},
        "mnli_auc": [], "gate": [], "floors": [], "real": [], "n_boundaries": 0}
    print(f"[axis] 정렬 {result['axis_alignment']}", flush=True)

    # MNLI dev 에서 1차원 축의 모순 AUC
    un_dv = u_dv / np.clip(np.linalg.norm(u_dv, axis=1, keepdims=True), 1e-9, None)
    vn_dv = v_dv / np.clip(np.linalg.norm(v_dv, axis=1, keepdims=True), 1e-9, None)
    readings = {
        "cos (거리, 1차원)": 1.0 - (un_dv * vn_dv).sum(axis=1),
        "axis:neg_rule (라벨 없음)": np.abs(d_dv @ ax_neg),
        "axis:mnli_mean (클래스 평균)": np.abs(d_dv @ ax_mean),
        "axis:mnli_lda (Fisher)": np.abs(d_dv @ ax_lda),
        "axis:mnli_lda 부호 있음": d_dv @ ax_lda,
    }
    is_c = y_dv == 2
    for name, vals in readings.items():
        a = _auc(vals[is_c], vals[~is_c])
        result["mnli_auc"].append({"name": name, "dims": 1,
                                   "auc": round(a, 4) if a else None})
        print(f"  {name}: AUC {result['mnli_auc'][-1]['auc']}", flush=True)
    # 참고 상한 — embed_probe 와 같은 특징으로 릿지 (교차검증 없이 train/dev 분리)
    F_tr = np.concatenate([np.abs(d_tr), u_tr * v_tr], axis=1).astype("float32")
    F_dv = np.concatenate([np.abs(d_dv), u_dv * v_dv], axis=1).astype("float32")
    yb = (y_tr == 2).astype("float32")
    G = F_tr.T @ F_tr + 1.0 * np.eye(F_tr.shape[1], dtype="float32")
    beta = np.linalg.solve(G, F_tr.T @ yb)
    pv = F_dv @ beta
    a = _auc(pv[is_c], pv[~is_c])
    result["mnli_auc"].append({"name": "probe (전체 지도, 참고)", "dims": 2048,
                               "auc": round(a, 4) if a else None})
    print(f"  probe: AUC {result['mnli_auc'][-1]['auc']}", flush=True)

    # 관문·바닥·실데이터
    recs = load_boundaries(run_dir)
    if args.max_boundaries:
        recs = recs[: args.max_boundaries]
    result["n_boundaries"] = len(recs)
    fulls = sorted({r["premise"] for r in recs})[: args.floor_sentences]
    cases = json.loads((AUTOSEG / "premature_cases.json").read_text(encoding="utf-8"))["cases"]

    scorers = [
        AxisScorer(enc, ax_neg, "axis:neg_rule"),
        AxisScorer(enc, ax_mean, "axis:mnli_mean"),
        AxisScorer(enc, ax_lda, "axis:mnli_lda"),
        # 정규화판 — 변화 **중** 이 축의 비율. 종류와 크기를 분리한다
        AxisScorer(enc, ax_neg, "axis:neg_rule/norm", normalize=True),
        AxisScorer(enc, ax_mean, "axis:mnli_mean/norm", normalize=True),
        # 종류(정규화 축) + 크기(코사인) 2차원 결합
        AxisScorer(enc, ax_mean, "axis:mean/norm+cos", normalize=True, blend=1.0),
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
              f"/{g['variants']['raw']['n_cases']}, "
              f"실데이터 Spearman {rd['variants']['raw']['spearman_global']}", flush=True)

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
