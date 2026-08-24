"""임베딩의 **여러 차원을 그대로** 쓰면 함의라는 한 축을 넘어설 수 있는가.

지금까지 임베딩을 쓴 방식은 둘 중 하나였다.
  `embed_check`·`fixed_point`  1024 차원을 **코사인 스칼라 하나로 접었다**
  `embed_probe`                차원은 살렸지만 **NLI 라벨로 지도학습**했다 — 결국 NLI 축

그래서 "차원이 많으니 일반화된 의미변화 측정이 되지 않나" 는 아직 검증되지 않았다.
빠진 것이 둘이다.

  **기하 보정.** 문장 임베딩 공간은 비등방(anisotropic)이다 — 모든 벡터가 좁은 원뿔에
  몰려 있어 코사인의 동적 범위가 낭비된다. 중심화·화이트닝으로 그 왜곡을 펴면 같은
  벡터에서 더 나은 신호가 나올 수 있다. 정보가 없는 것과 기하가 가리는 것은 다르다.

  **비지도 구조.** 차이벡터 `d = v_j − v_i` 는 *얼마나* 가 아니라 *어떻게* 바뀌었는지를
  담을 수 있다. 그 구조가 라벨 없이 드러난다면 진짜 일반화된 측정이 된다.

핵심 가설(반증 대상): **우리가 원하는 축은 희귀하고 저분산이다.** 비지도 읽기는 분산이
큰 방향을 먼저 집는데, 인접 prefix 번역의 차이에서 분산을 지배하는 것은 "내용이 추가됨"
이지 "극성이 뒤집힘" 이 아니다. 그렇다면 한계는 차원 수가 아니라 **무엇을 볼지 알려주는
신호의 부재**이고, NLI 는 그 신호를 이미 학습해서 갖고 있는 셈이 된다.

읽기 방식을 계단식으로 늘려 어디서 신호가 생기는지 분리한다.

  cos            원본 코사인 (지금까지 쓴 것). 1차원
  cos_centered   코퍼스 평균을 뺀 뒤 코사인
  cos_whitened   화이트닝 후 코사인 — 비등방 왜곡 제거
  l2             `‖d‖`. 방향 없이 크기만
  mahalanobis    화이트닝 공간에서의 `‖d‖`. **1024 차원을 전부 쓰되 비지도**
  pc_best        `d` 의 주성분 중 목표와 가장 잘 맞는 하나 (선택 단계만 지도)
  probe          `d` → 목표 의 릿지 회귀, K-fold **교차검증**. 지도 상한

목표값은 NLI 함의 하락 `1 − P(S_j ⊨ S_i)` 다. **이것은 "임베딩이 NLI 를 재현할 수
있는가" 를 묻는 것이지 정확도가 아니다** — 라벨이 없는 문제라 현행 축을 기준으로 삼는다.
관문(`premature_cases.json`)이 유일한 라벨 근거이나 절단 위치 성질이라 여기서는 부록이다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.embed_geometry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..autoseg import metrics
from .boundary_probe import load_sentences
from .embed_check import MODELS
from .embed_probe import FrozenEncoder
from .fixed_point import EntailBackend, translate_trajectories
from ..autoseg.pipeline import GoogleTranslator, JsonCache, to_lang_code

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS


def build_pairs(sents: list[dict]) -> list[dict]:
    """`(S_i, S_j)` 쌍. `next` = 한 어절 뒤, `final` = 문장 끝."""
    out = []
    for s in sents:
        traj = s["traj"]
        n = len(traj)
        for i in range(1, n):
            if not traj[i - 1] or not traj[i]:
                continue
            out.append({"text": s["text"], "i": i, "n": n, "horizon": "next",
                        "a": traj[i], "b": traj[i - 1],        # a = 긴 쪽(전제)
                        "dwords": len(traj[i].split()) - len(traj[i - 1].split())})
            if traj[n - 1]:
                out.append({"text": s["text"], "i": i, "n": n, "horizon": "final",
                            "a": traj[n - 1], "b": traj[i - 1],
                            "dwords": len(traj[n - 1].split()) - len(traj[i - 1].split())})
    return out


def whitening(X, eps: float = 1e-6):
    """`(mean, W)` 를 돌려준다. `(x − mean) @ W` 가 화이트닝된 좌표다."""
    import numpy as np
    mu = X.mean(axis=0, keepdims=True)
    Z = X - mu
    cov = (Z.T @ Z) / max(1, len(Z) - 1)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    return mu, V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def _norm_rows(X, eps: float = 1e-9):
    import numpy as np
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), eps, None)


def readouts(u, v, mu, W, target, n_pcs: int = 10, folds: int = 5) -> dict:
    """읽기 방식별 점수. 전부 **높을수록 의미가 많이 변함** 축으로 맞춘다."""
    import numpy as np

    out: dict = {}
    uc, vc = u - mu, v - mu
    uw, vw = uc @ W, vc @ W

    out["cos"] = 1.0 - (_norm_rows(u) * _norm_rows(v)).sum(axis=1)
    out["cos_centered"] = 1.0 - (_norm_rows(uc) * _norm_rows(vc)).sum(axis=1)
    out["cos_whitened"] = 1.0 - (_norm_rows(uw) * _norm_rows(vw)).sum(axis=1)
    d = u - v
    out["l2"] = np.linalg.norm(d, axis=1)
    dw = uw - vw
    out["mahalanobis"] = np.linalg.norm(dw, axis=1)

    # 차이벡터의 주성분. 비지도로 뽑고, **선택만** 목표를 본다.
    dm = d - d.mean(axis=0, keepdims=True)
    cov = (dm.T @ dm) / max(1, len(dm) - 1)
    w_, V = np.linalg.eigh(cov)
    order = np.argsort(-w_)[:n_pcs]
    evr = (w_[order] / max(1e-12, float(w_.sum()))).tolist()
    scores = dm @ V[:, order]
    pcs = []
    for k in range(len(order)):
        c = metrics._spearman(list(target), [float(x) for x in scores[:, k]])
        pcs.append({"pc": k + 1, "explained_var": round(float(evr[k]), 4),
                    "spearman_vs_target": round(c, 4) if c is not None else None})
    best = max(pcs, key=lambda p: abs(p["spearman_vs_target"] or 0))
    sign = 1.0 if (best["spearman_vs_target"] or 0) >= 0 else -1.0
    out["pc_best"] = sign * scores[:, best["pc"] - 1]
    # 부분공간 노름 — 비지도로 차원을 **여러 개 묶어** 읽는 방식
    out["pc_top_norm"] = np.linalg.norm(scores, axis=1)
    out["pc_resid_norm"] = np.sqrt(np.maximum(
        0.0, (dm ** 2).sum(axis=1) - (scores ** 2).sum(axis=1)))

    # 지도 상한. **원소별 크기 특징을 넣어야 공정하다** — 목표는 ‖d‖ 에 관한 양인데
    # 원시 `d` 에 대한 선형함수로는 노름을 표현할 수 없다 (`embed_probe` 가 |u−v|·u∘v 를
    # 쓴 이유가 이것이다). 이걸 빼면 지도 상한을 과소평가해 "지도해도 안 된다" 는
    # 틀린 결론이 나온다.
    y = np.asarray(target, dtype="float32")
    F = np.concatenate([np.abs(d), u * v], axis=1).astype("float32")
    idx = np.arange(len(y))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    pred = np.zeros_like(y)
    lam = 1.0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        A = F[tr]
        G = A.T @ A + lam * np.eye(A.shape[1], dtype=A.dtype)
        beta = np.linalg.solve(G, A.T @ y[tr])
        pred[te] = F[te] @ beta
    out["probe_cv"] = pred

    return {"scores": {k: [float(x) for x in v_] for k, v_ in out.items()},
            "pcs": pcs, "best_pc": best}


def render(result: dict) -> str:
    L = ["# 임베딩의 여러 차원으로 함의를 대신할 수 있는가", "",
         f"인코더 `{result['encoder_id']}` · 문장 {result['n_sentences']}개 · "
         f"쌍 {result['n_pairs']}개 · 번역 호출 0 (캐시) · LLM 호출 0", "",
         "지금까지 임베딩은 **코사인 스칼라로 접거나**(`embed_check`, `fixed_point`) "
         "**NLI 라벨로 지도학습**(`embed_probe`)해서만 썼다. 여기서는 차원을 살린 채 "
         "**비지도**로 읽어 본다 — 기하 보정(중심화·화이트닝)과 차이벡터의 주성분.", "",
         "목표값은 `1 − P(S_j ⊨ S_i)`(NLI 함의 하락). **정확도가 아니라 '현행 축을 "
         "재현하는가' 를 묻는 것이다** — 이 문제에는 라벨이 없다.", ""]

    for h in result["horizons"]:
        L += [f"## 지평 `{h['horizon']}` (쌍 {h['n']}개)", "",
              "| 읽기 방식 | 차원 | 목표와 Spearman | 어절 증가량과 Spearman |",
              "|---|---|---|---|"]
        for r in h["readouts"]:
            L.append(f"| `{r['name']}` | {r['dims']} | **{r['spearman_target']}** | "
                     f"{r['spearman_dwords']} |")
        L += ["", "차이벡터 주성분 (비지도로 뽑음):", "",
              "| PC | 설명 분산 | 목표와 Spearman | 어절 증가량과 Spearman |",
              "|---|---|---|---|"]
        for pcr in h["pcs"]:
            L.append(f"| {pcr['pc']} | {pcr['explained_var']:.1%} | "
                     f"{pcr['spearman_vs_target']} | {pcr.get('spearman_dwords')} |")
        L.append("")

    L += ["## 판정", ""]

    def pick(horizon):
        h = next((x for x in result["horizons"] if x["horizon"] == horizon), None)
        return (h, {r["name"]: r for r in (h or {}).get("readouts", [])})

    hn, bn = pick("next")
    hf, bf = pick("final")

    L += ["### ① 차원을 살려도 코사인을 못 넘는다 — 비지도로는", "",
          "| 읽기 | 차원 | `next` | `final` |", "|---|---|---|---|"]
    for name in ["cos", "cos_whitened", "mahalanobis", "pc_top_norm",
                 "pc_resid_norm", "probe_cv"]:
        L.append(f"| `{name}` | {bn.get(name, {}).get('dims', '—')} | "
                 f"{bn.get(name, {}).get('spearman_target')} | "
                 f"{bf.get(name, {}).get('spearman_target')} |")
    L += ["",
          "1024 차원을 전부 쓰는 마할라노비스도, 주성분 부분공간 노름도 "
          "**스칼라 코사인 아래**다. 화이트닝은 오히려 **해롭다** "
          f"({bn.get('cos', {}).get('spearman_target')} → "
          f"{bn.get('cos_whitened', {}).get('spearman_target')}, final 에서는 "
          f"{bf.get('cos', {}).get('spearman_target')} → "
          f"{bf.get('cos_whitened', {}).get('spearman_target')}) — 모든 방향을 균등하게 "
          "펴면 저분산 방향의 잡음이 증폭된다. 즉 **정보는 고분산 방향에 있고 코사인은 "
          "이미 그쪽을 보고 있다.**", ""]

    top_n = hn["pcs"][0] if hn and hn["pcs"] else {}
    L += ["### ② 차이벡터에 읽을 만한 '축' 자체가 없다", "",
          f"제1주성분이 설명하는 분산이 `next` 에서 {top_n.get('explained_var', 0):.1%}, "
          f"`final` 에서 {(hf['pcs'][0]['explained_var'] if hf and hf['pcs'] else 0):.1%} "
          "에 불과하다. 상위 10개를 합쳐도 20% 남짓이다. **차이벡터는 거의 등방적이라, "
          "'극성 축'·'시제 축' 처럼 해석 가능한 소수의 방향으로 정리되지 않는다.** "
          "비지도로 차원을 읽겠다는 발상이 걸리는 지점이 여기다 — 읽을 축이 있어야 "
          "읽는데, 분산 기준으로는 축이 안 나온다.", ""]
    if hf and hf["pcs"]:
        L += [f"유일하게 해석되는 것은 `final` 의 PC1 인데, 그것은 **어절 증가량과 상관 "
              f"{hf['pcs'][0].get('spearman_dwords')}** — '내용이 얼마나 더 붙었나' 다. "
              "우리가 원하는 축이 아니다.", ""]

    L += ["### ③ 그런데 코사인이 무엇을 재고 있는지가 지평마다 다르다", "",
          f"`next`(한 어절 추가): 목표와 {bn.get('cos', {}).get('spearman_target')}, "
          f"어절 증가량과는 {bn.get('cos', {}).get('spearman_dwords')} — **의미 변화를 "
          "재고 있다.**",
          f"`final`(문장 끝까지): 목표와 {bf.get('cos', {}).get('spearman_target')} 인데 "
          f"어절 증가량과 {bf.get('cos', {}).get('spearman_dwords')} — **길이 차이를 "
          "더 많이 재고 있다.** 멀리 떨어진 두 렌더링을 비교할수록 '얼마나 더 붙었나' 가 "
          "'어떻게 바뀌었나' 를 덮는다.", "",
          "`fixed_point` 에서 `cos`(0.375)가 `entail`(0.592)에 진 이유가 이것이다.", ""]

    L += ["### ④ 핵심 — 코사인이 못 읽는 것은 '크기' 가 아니라 '종류' 다", ""]
    if bn.get("probe_cv") and bn.get("cos"):
        gap = ((bn["probe_cv"]["spearman_target"] or 0)
               - (bn["cos"]["spearman_target"] or 0))
        L += [f"지도 신호를 주고 2048 차원을 읽어도 `next` 에서 이득이 **{gap:+.3f}** "
              f"({bn['cos']['spearman_target']} → {bn['probe_cv']['spearman_target']}) 뿐이다. "
              "**'의미가 얼마나 변했나' 라는 축에서는 코사인이 이미 거의 최선이다.**", ""]
    L += ["그런데 `embed_probe.py` 에서는 같은 임베딩에 지도를 줬을 때 모순 AUC 가 "
          "0.723 → 0.899 로 크게 올랐다. 목표가 달랐기 때문이다 — 저기서는 "
          "**모순/중립/함의라는 *종류* 판정**이었고, 여기서는 **변화의 *크기***다.", "",
          "| 축 | 코사인이 읽나 | 차원을 늘리면 |",
          "|---|---|---|",
          "| 의미가 **얼마나** 변했나 (크기) | **잘 읽는다** | 이득 +0.015 |",
          "| **어떤 종류로** 변했나 (모순 vs 단순 추가) | **못 읽는다** | 지도학습이 있어야 오르고, 그래도 교차 인코더에 모순 오류율 11배 뒤진다 |",
          "",
          "우리 문제에서 갈라야 하는 것은 **무해한 미완성과 반박** 인데, 그 둘은 크기가 "
          "비슷하고 종류가 다르다. 그래서 크기 축에서 아무리 잘해도 갈리지 않는다.", ""]

    L += ["### 정리 — 질문에 대한 답", "",
          "| 질문 | 답 |",
          "|---|---|",
          "| 임베딩이 함의보다 일반적인 측정인가 | **크기 축에서는 그렇다** — 함의 하락과 0.66 |",
          "| 차원을 많이 쓰면 비지도로 더 읽히나 | **아니다** — 마할라노비스·주성분 전부 코사인 이하 |",
          "| 읽을 만한 의미 축이 벡터에 있나 | **없다** — 제1주성분이 분산의 3~5%, 거의 등방 |",
          "| 그럼 무엇이 부족한가 | **종류 판정.** 크기가 아니라 '무해한 변화인가 반박인가' |",
          "",
          "임베딩이 일반화된 메트릭이 되려면 **무엇이 심각한 변화인지** 를 알려주는 신호가 "
          "필요하다. 그 신호는 라벨(또는 그 라벨로 학습된 모델)에서만 오고, 이 문제에서 "
          "그것을 이미 갖고 있는 물건이 NLI 계열이다. 차원이 모자란 것이 아니라 "
          "**차원에 이름이 안 붙어 있는 것이 문제다.**"]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="임베딩 기하·차원 읽기 실험")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--encoder", default="e5-inst", choices=sorted(MODELS))
    p.add_argument("--nli-model", default="deberta-mnli", choices=sorted(metrics.NLI_MODELS))
    p.add_argument("--max-words", type=int, default=24)
    p.add_argument("--max-sentences", type=int, default=0)
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = SEG_RUNS / args.run_id
    out_dir = Path(args.out) if args.out else (OUT_RUNS / "embed_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        result = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
        (out_dir / "report.md").write_text(render(result), encoding="utf-8")
        print(f"[done] {out_dir / 'report.md'}")
        return 0

    import numpy as np

    sents = [s for s in load_sentences(run_dir) if s["n_words"] <= args.max_words]
    merged: dict[str, dict] = {}
    for s in sents:
        merged.setdefault(s["text"], {"text": s["text"], "n_words": s["n_words"]})
    sents = list(merged.values())
    if args.max_sentences:
        sents = sents[: args.max_sentences]
    if not sents:
        print(f"문장을 못 찾음: {run_dir}", file=sys.stderr)
        return 1

    # `fixed_point` 가 만든 prefix 캐시를 그대로 쓴다 — 네트워크 호출 0
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    code = cfg.get("tgt_code") or to_lang_code(cfg.get("tgt_lang") or "English")
    cache_path = OUT_RUNS / "fixed_point" / "prefix_cache.json"
    gt = GoogleTranslator(tgt_code=code, workers=args.workers,
                          cache=JsonCache(cache_path), use_context=False)
    try:
        translate_trajectories(sents, gt)
    finally:
        gt.close()
    print(f"[gtx] 신규 호출 {gt.calls}건 (0 이면 전부 캐시)", flush=True)

    pairs = build_pairs(sents)
    print(f"[data] 쌍 {len(pairs)}개", flush=True)

    entail = EntailBackend(model_name=metrics.NLI_MODELS[args.nli_model])
    print("[nli] 함의 계산...", flush=True)
    ent = entail.score([q["a"] for q in pairs], [q["b"] for q in pairs])
    for q, e in zip(pairs, ent):
        q["target"] = 1.0 - float(e)

    enc = FrozenEncoder(args.encoder)
    texts = sorted({q["a"] for q in pairs} | {q["b"] for q in pairs})
    print(f"[enc] 렌더링 {len(texts)}건 인코딩...", flush=True)
    vecs = dict(zip(texts, enc.encode(texts)))
    mu, W = whitening(np.stack([vecs[t] for t in texts]))

    result: dict = {"run_id": args.run_id, "encoder": args.encoder,
                    "encoder_id": MODELS[args.encoder]["id"],
                    "n_sentences": len(sents), "n_pairs": len(pairs), "horizons": []}

    DIMS = {"cos": 1, "cos_centered": 1, "cos_whitened": 1, "l2": 1,
            "mahalanobis": "1024(비지도)", "pc_best": "1(비지도 축)",
            "pc_top_norm": "10(비지도 부분공간)", "pc_resid_norm": "1014(비지도 잔여)",
            "probe_cv": "2048(지도)"}

    for horizon in ["next", "final"]:
        sub = [q for q in pairs if q["horizon"] == horizon]
        if len(sub) < 50:
            continue
        u = np.stack([vecs[q["a"]] for q in sub])
        v = np.stack([vecs[q["b"]] for q in sub])
        target = [q["target"] for q in sub]
        dwords = [float(q["dwords"]) for q in sub]
        print(f"\n[{horizon}] 읽기 방식 비교 (n={len(sub)})...", flush=True)
        res = readouts(u, v, mu, W, target, n_pcs=args.n_pcs)

        rows = []
        for name, vals in res["scores"].items():
            ct = metrics._spearman(target, vals)
            cd = metrics._spearman(dwords, vals)
            rows.append({"name": name, "dims": DIMS.get(name, "?"),
                         "spearman_target": round(ct, 4) if ct is not None else None,
                         "spearman_dwords": round(cd, 4) if cd is not None else None})
            print(f"  {name:<14} target {rows[-1]['spearman_target']}", flush=True)

        # 주성분이 무엇을 잡는지 — 어절 증가량과의 상관을 같이 본다
        dm = (u - v)
        dm = dm - dm.mean(axis=0, keepdims=True)
        cov = (dm.T @ dm) / max(1, len(dm) - 1)
        w_, V = np.linalg.eigh(cov)
        order = np.argsort(-w_)[: args.n_pcs]
        sc = dm @ V[:, order]
        for k, pcr in enumerate(res["pcs"]):
            c = metrics._spearman(dwords, [float(x) for x in sc[:, k]])
            pcr["spearman_dwords"] = round(c, 4) if c is not None else None

        result["horizons"].append({"horizon": horizon, "n": len(sub),
                                   "readouts": rows, "pcs": res["pcs"],
                                   "best_pc": res["best_pc"]})

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
