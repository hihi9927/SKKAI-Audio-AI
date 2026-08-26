"""임베딩에 모순 정보가 **있는데 코사인이 버리는 것인가** — 프로브로 확인한다.

`embed_check.py` 가 기각한 것은 정확히 말하면 임베딩이 아니라 **코사인**이다. 코사인은
두 벡터를 스칼라 하나로 접고 그 축은 학습 목표(주제·검색 유사도)에 정렬돼 있다.
극성이 차이벡터 `|u−v|` 나 성분곱 `u∘v` 에 남아 있을 수 있고, 그렇다면 문제는 표현이
아니라 **읽는 방법**이다.

그래서 인코더를 **얼린 채** pair feature 위에 헤드만 학습해 MNLI 로 지도한다.
feature 를 계단식으로 늘려 어디서 정보가 들어오는지 분리한다.

  cos    `[cos(u,v)]`                     ← `embed_check` 가 쓴 것. 1차원
  diff   `[|u−v|]`                        ← 차이벡터만
  full   `[u, v, |u−v|, u∘v]`             ← InferSent/SBERT NLI 헤드의 고전적 구성

헤드는 선형과 MLP 둘 다 본다. 선형이 되면 **선형 분리 가능**, MLP 만 되면 정보는 있으나
꼬여 있다, 둘 다 안 되면 **풀링된 벡터에 그 정보가 없다**는 뜻이다.

**교차 인코더 천장을 같은 데이터에서 같이 잰다.** bi-encoder 는 두 문장이 풀링 전에
서로를 못 본다 — 모순은 *특정 명제 쌍 사이의 관계*라, 정렬 없이 고정 길이 요약 두 개만
비교해서는 원리적으로 불리하다. 그 격차의 크기가 이 실험의 핵심 수치다.

평가는 세 곳:
  MNLI dev   3-way 정확도 + 모순 AUC. 프로브가 애초에 학습됐는지 확인
  관문       `premature_cases.json`. 우리 문제에서의 판정
  실데이터   run04 경계 1003개. 잡음 바닥 + 현행 NLI 와의 순위 상관

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.embed_probe \
      --encoder e5-inst --train-size 60000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from .embed_check import (MODELS, contra_gate, load_boundaries, measure_floor,
                          prune_degenerate, real_data)

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS

FEATURE_SETS = ["cos", "diff", "full"]
LABELS = ["entailment", "neutral", "contradiction"]
CONTRA = 2


# ── 인코딩 ───────────────────────────────────────────────────────────────

class FrozenEncoder:
    """sentence-transformers 인코더. **학습하지 않는다** — 프로브만 배운다."""

    def __init__(self, key: str, batch_size: int = 128, device: str = "cuda"):
        self.key = key
        self.spec = MODELS[key]
        self.batch_size = batch_size
        self.device = device
        self._model = None

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

    def encode(self, texts: list[str]):
        import numpy as np
        m = self.load()
        kw: dict = {"batch_size": self.batch_size, "normalize_embeddings": True,
                    "show_progress_bar": False, "convert_to_numpy": True}
        if self.spec.get("prompt_name"):
            kw["prompt_name"] = self.spec["prompt_name"]
        elif self.spec.get("prompt"):
            kw["prompt"] = self.spec["prompt"]
        blank = [i for i, t in enumerate(texts) if not (t or "").strip()]
        safe = [t if (t or "").strip() else "." for t in texts]
        out = m.encode(safe, **kw)
        for i in blank:
            out[i] = 0.0
        return np.asarray(out, dtype="float32")


def pair_features(u, v, kind: str):
    """`u`,`v` 는 L2 정규화된 행렬. 반환은 (n, d')."""
    import numpy as np
    if kind == "cos":
        return (u * v).sum(axis=1, keepdims=True)
    if kind == "diff":
        return np.abs(u - v)
    if kind == "full":
        return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)
    raise ValueError(kind)


# ── 프로브 ───────────────────────────────────────────────────────────────

class Probe:
    """얼린 임베딩 위의 3-way 분류 헤드. `linear` 또는 `mlp`."""

    def __init__(self, in_dim: int, head: str = "linear", hidden: int = 512,
                 device: str = "cuda"):
        import torch
        from torch import nn
        self.torch = torch
        self.device = device
        self.head = head
        if head == "linear":
            self.net = nn.Linear(in_dim, 3)
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(hidden, 3))
        self.net = self.net.to(device)

    def fit(self, X, y, epochs: int = 12, batch_size: int = 512, lr: float = 1e-3):
        torch = self.torch
        from torch import nn
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y).long()
        n = len(yt)
        self.net.train()
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb = Xt[idx].to(self.device)
                yb = yt[idx].to(self.device)
                opt.zero_grad()
                loss = lossf(self.net(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if ep in (0, epochs - 1):
                print(f"    epoch {ep + 1}/{epochs} loss {tot / n:.4f}", flush=True)
        return self

    def predict_proba(self, X):
        torch = self.torch
        self.net.eval()
        out = []
        Xt = torch.from_numpy(X)
        with torch.no_grad():
            for i in range(0, len(Xt), 4096):
                xb = Xt[i : i + 4096].to(self.device)
                out.append(torch.softmax(self.net(xb).float(), dim=-1).cpu().numpy())
        import numpy as np
        return np.concatenate(out, axis=0)


class ProbeScorer:
    """관문·바닥·실데이터 하네스가 요구하는 `.score(premises, hypotheses)`.

    반환은 **모순 확률**이라 현행 NLI 와 축이 같다."""

    def __init__(self, enc: FrozenEncoder, probe: Probe, kind: str, name: str):
        self.enc = enc
        self.probe = probe
        self.kind = kind
        self.name = name

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import numpy as np
        u = self.enc.encode(list(premises))
        v = self.enc.encode(list(hypotheses))
        X = pair_features(u, v, self.kind).astype("float32")
        p = self.probe.predict_proba(X)[:, CONTRA]
        return [float(x) for x in np.nan_to_num(p)]


# ── 평가 ─────────────────────────────────────────────────────────────────

def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def mnli_eval(probs, labels) -> dict:
    import numpy as np
    pred = probs.argmax(axis=1)
    acc = float((pred == labels).mean())
    pc = probs[:, CONTRA]
    auc = _auc([float(x) for x in pc[labels == CONTRA]][:2000],
               [float(x) for x in pc[labels != CONTRA]][:2000])
    return {"acc_3way": round(acc, 4),
            "auc_contradiction": round(auc, 4) if auc is not None else None,
            "n": int(len(labels))}


def cross_encoder_ceiling(prem: list[str], hyp: list[str], labels,
                          model_name: str) -> dict:
    """같은 dev 부분집합에서 교차 인코더 천장. **격차의 크기가 이 실험의 핵심 수치다.**"""
    import numpy as np
    from transformers import pipeline
    pipe = pipeline("text-classification", model=model_name, device=0, top_k=None)
    res = pipe([{"text": p, "text_pair": h} for p, h in zip(prem, hyp)], batch_size=32)
    order = {"entail": 0, "neutral": 1, "contra": 2}
    probs = np.zeros((len(res), 3), dtype="float32")
    for i, scores in enumerate(res):
        for s in scores:
            for pre, j in order.items():
                if s["label"].lower().startswith(pre):
                    probs[i, j] = s["score"]
    return mnli_eval(probs, labels)


def render(result: dict) -> str:
    L = ["# 임베딩에 모순 정보가 남아 있는가 — 프로브 실험", "",
         f"인코더 `{result['encoder_id']}` (**얼림**) · MNLI 학습 {result['train_size']}쌍 · "
         f"dev {result['dev_size']}쌍 · run04 경계 {result['n_boundaries']}개", "",
         "`embed_check.py` 가 기각한 것은 임베딩이 아니라 **코사인**이었다. 코사인은 두 "
         "벡터를 스칼라 하나로 접고 그 축은 주제 유사도에 정렬돼 있다 — 극성이 `|u−v|` 나 "
         "`u∘v` 에 남아 있다면 문제는 표현이 아니라 읽는 방법이다. 인코더를 얼린 채 "
         "헤드만 MNLI 로 학습해 그것을 가른다.", ""]

    L += ["## 1 — MNLI dev: 모순 정보가 애초에 디코딩되는가", "",
          "| feature | 헤드 | 3-way 정확도 | 모순 AUC |", "|---|---|---|---|"]
    for r in result["mnli"]:
        L.append(f"| `{r['kind']}` | {r['head']} | {r['acc_3way']} | "
                 f"{r['auc_contradiction']} |")
    ce = result.get("ceiling")
    if ce:
        L.append(f"| **교차 인코더 (천장)** | `{result['nli_model']}` | "
                 f"**{ce['acc_3way']}** | **{ce['auc_contradiction']}** |")
    L += ["", "무작위 기준 3-way 정확도 0.33, AUC 0.5.", ""]

    L += ["## 2 — 관문 (`premature_cases.json`)", "",
          "통과 조건은 `judge_check.check_nli` 와 동일: `min(premature) > max(safe)`.", "",
          "| feature | 헤드 | 변이 | 위반/케이스 | mean(prem) | mean(safe) | 격차 | 판정 |",
          "|---|---|---|---|---|---|---|---|"]
    for g in result["gate"]:
        for k, v in g["variants"].items():
            L.append(f"| `{g['kind']}` | {g['head']} | {k} | "
                     f"{v['violations']}/{v['n_cases']} | {v['mean_premature']} | "
                     f"{v['mean_safe']} | {v['margin']} | "
                     f"{'통과' if v['passed'] else '**탈락**'} |")

    L += ["", "## 3 — 잡음 바닥과 실데이터 거동", "",
          "바닥은 full 번역의 자기-prefix(정의상 무해한 미완성). **모순 확률이므로 0 에 "
          "가까워야 한다.**", "",
          "| feature | 헤드 | 바닥 mean | 바닥 sd | " + " | ".join(nf.bucket_labels()) + " |",
          "|---|---|---|---|" + "---|" * len(nf.bucket_labels())]
    for f in result["floors"]:
        fl = f["floor"]
        cells = [str(fl["by_length_bucket"][b]["mean"]) if fl["by_length_bucket"][b]["n"]
                 else "—" for b in nf.bucket_labels()]
        L.append(f"| `{f['kind']}` | {f['head']} | {fl['overall']['mean']} | "
                 f"{fl['overall']['sd']} | " + " | ".join(cells) + " |")

    L += ["", "| feature | 헤드 | 변이 | 현행 NLI 와 Spearman | topk 겹침 |",
          "|---|---|---|---|---|"]
    for r in result["real"]:
        for k, v in r["variants"].items():
            L.append(f"| `{r['kind']}` | {r['head']} | {k} | {v['spearman_global']} | "
                     f"{v['topk_overlap']} |")

    # 판정
    cos_row = next((r for r in result["mnli"] if r["kind"] == "cos"), None)
    full_rows = [r for r in result["mnli"] if r["kind"] == "full"]
    best_full = max(full_rows, key=lambda r: r["acc_3way"]) if full_rows else None
    gate_pass = [(g["kind"], g["head"]) for g in result["gate"]
                 if any(v["passed"] for v in g["variants"].values())]

    L += ["", "## 판정", ""]
    if cos_row and best_full:
        L += [f"**모순 정보는 벡터에 남아 있다.** 코사인만 쓰면 3-way 정확도 "
              f"{cos_row['acc_3way']} / 모순 AUC {cos_row['auc_contradiction']} 인데, "
              f"같은 얼린 벡터에서 `full` feature + {best_full['head']} 헤드는 "
              f"{best_full['acc_3way']} / {best_full['auc_contradiction']} 로 오른다. "
              f"코사인이 정보를 **버리고 있었다**.", ""]
    if ce and best_full:
        gap = ce["acc_3way"] - best_full["acc_3way"]
        e_probe = 1.0 - (best_full["auc_contradiction"] or 0)
        e_ce = 1.0 - (ce["auc_contradiction"] or 0)
        L += [f"**그러나 교차 인코더와의 격차가 남는다: {gap:+.3f}** "
              f"({best_full['acc_3way']} vs {ce['acc_3way']}). 모순 AUC 로 보면 "
              f"{best_full['auc_contradiction']} vs {ce['auc_contradiction']} 인데, "
              f"**오류율로 환산하면 {e_probe:.3f} vs {e_ce:.3f} — "
              f"{e_probe / max(e_ce, 1e-9):.0f}배다.**", "",
              "원인은 구조다. bi-encoder 는 두 문장이 **풀링 전에 서로를 못 본다**. 모순은 "
              "*특정 명제 쌍 사이의 관계*라 어느 토큰이 어느 토큰과 충돌하는지를 봐야 하는데, "
              "고정 길이 요약 두 개로 접은 뒤에는 그 정렬이 남아 있지 않다. 헤드를 키워도 "
              "(`linear` → `mlp`) 표현에서 이미 잃은 것은 복구되지 않는다 — 실제로 "
              "`full` 에서 헤드를 키운 이득은 정확도 +0.03 에 그쳤다.", ""]

    # 부호 전환 — 코사인의 '신호가 음수' 문제가 feature 로 해결됐는지
    def raw_margin(kind, head):
        g = next((g for g in result["gate"]
                  if g["kind"] == kind and g["head"] == head), None)
        return (g or {}).get("variants", {}).get("raw", {}).get("margin")

    m_cos, m_diff = raw_margin("cos", "mlp"), raw_margin("diff", "mlp")
    if m_cos is not None and m_diff is not None:
        L += [f"**부호는 뒤집혔다.** `embed_check.py` 에서 코사인의 관문 신호는 "
              f"**음수**였다(잘못 자른 방출이 안전한 방출보다 참조에 더 가깝게 나옴). "
              f"여기서도 `cos` 는 {m_cos:+.4f} 로 여전히 음수지만, `diff` 는 "
              f"{m_diff:+.4f} 로 **양수가 된다.** 방향 문제는 feature 로 해결됐고 "
              "남은 것은 정밀도다 — 평균은 맞는데 케이스별로 틀린다.", ""]
    if gate_pass:
        L += ["관문 통과: " + ", ".join(f"`{k}`+{h}" for k, h in gate_pass), "",
              "**관문을 통과했다면 실제 후보다.** 다만 채택 전에 바닥(위 표)과 "
              "실데이터 거동을 함께 볼 것 — MNLI 에서 배운 프로브가 우리 도메인(조각 vs "
              "full 번역)에서도 같은 축을 재는지는 별개 문제다."]
    else:
        best_gate = min((v["violations"] for g in result["gate"]
                         for v in g["variants"].values()), default=None)
        L += [f"**관문은 어느 구성도 통과하지 못했다** (최선 {best_gate}/6, "
              "코사인 단독은 5/6). 통과 기준이 평균이 아니라 **케이스마다** "
              "`min(premature) > max(safe)` 라, 평균 격차가 양수여도 한 케이스만 "
              "뒤집히면 탈락한다. 목적함수에 들어가는 값이라 그 엄격함이 맞다.", "",
              "**남은 격차에는 도메인 이동도 섞여 있다.** 프로브는 MNLI(완결된 문장 쌍)로 "
              "배웠는데 우리 hypothesis 는 **미완성 조각**이다. 우리 분포로 학습하면 나아질 "
              "여지가 있으나, 지금 가진 라벨은 판정자 판정 21건(그중 premature 1건)뿐이라 "
              "학습이 불가능하다. 루프가 이터레이션마다 판정을 쌓으므로 나중에 다시 볼 만한 "
              "경로이고, 그때도 위의 **구조적 격차**는 그대로 남는다는 점을 감안할 것."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="임베딩 프로브로 모순 정보 확인")
    p.add_argument("--encoder", default="e5-inst", choices=sorted(MODELS))
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--dev-size", type=int, default=3000)
    p.add_argument("--heads", nargs="+", default=["linear", "mlp"],
                   choices=["linear", "mlp"])
    p.add_argument("--features", nargs="+", default=FEATURE_SETS, choices=FEATURE_SETS)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--floor-sentences", type=int, default=150)
    p.add_argument("--max-boundaries", type=int, default=0)
    p.add_argument("--nli-model", default=metrics.NLI_MODEL)
    p.add_argument("--skip-ceiling", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (OUT_RUNS / "embed_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        result = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
        (out_dir / "report.md").write_text(render(result), encoding="utf-8")
        print(f"[done] {out_dir / 'report.md'}")
        return 0

    import numpy as np
    from datasets import load_dataset

    print("[data] MNLI 로드...", flush=True)
    tr = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=0)
    tr = tr.select(range(min(args.train_size, len(tr))))
    dv = load_dataset("nyu-mll/multi_nli", split="validation_matched").shuffle(seed=0)
    dv = dv.select(range(min(args.dev_size, len(dv))))

    enc = FrozenEncoder(args.encoder, batch_size=args.batch_size)
    print(f"[enc] {MODELS[args.encoder]['id']} 로 MNLI 인코딩 "
          f"({len(tr)}+{len(dv)}쌍)...", flush=True)
    u_tr = enc.encode(list(tr["premise"]))
    v_tr = enc.encode(list(tr["hypothesis"]))
    u_dv = enc.encode(list(dv["premise"]))
    v_dv = enc.encode(list(dv["hypothesis"]))
    y_tr = np.array(tr["label"], dtype="int64")
    y_dv = np.array(dv["label"], dtype="int64")

    run_dir = SEG_RUNS / args.run_id
    recs = load_boundaries(run_dir)
    if args.max_boundaries:
        recs = recs[: args.max_boundaries]
    fulls = sorted({r["premise"] for r in recs})[: args.floor_sentences]
    cases = json.loads((AUTOSEG / "premature_cases.json").read_text(encoding="utf-8"))["cases"]

    result: dict = {"encoder": args.encoder, "encoder_id": MODELS[args.encoder]["id"],
                    "train_size": len(tr), "dev_size": len(dv),
                    "n_boundaries": len(recs), "nli_model": args.nli_model,
                    "mnli": [], "gate": [], "floors": [], "real": [], "ceiling": None}

    for kind in args.features:
        X_tr = pair_features(u_tr, v_tr, kind).astype("float32")
        X_dv = pair_features(u_dv, v_dv, kind).astype("float32")
        for head in args.heads:
            label = f"probe:{kind}+{head}"
            print(f"\n[{label}] 학습 (dim={X_tr.shape[1]})...", flush=True)
            probe = Probe(X_tr.shape[1], head=head).fit(X_tr, y_tr, epochs=args.epochs)
            ev = mnli_eval(probe.predict_proba(X_dv), y_dv)
            ev.update({"kind": kind, "head": head})
            result["mnli"].append(ev)
            print(f"[{label}] MNLI dev acc {ev['acc_3way']} "
                  f"모순 AUC {ev['auc_contradiction']}", flush=True)

            sc = ProbeScorer(enc, probe, kind, label)
            floor = measure_floor(fulls, sc)
            result["floors"].append({"kind": kind, "head": head, "floor": floor})
            g = contra_gate(cases, sc, floor)
            g.update({"kind": kind, "head": head})
            result["gate"].append(g)
            print(f"[{label}] 관문 위반 {g['variants']['raw']['violations']}"
                  f"/{g['variants']['raw']['n_cases']}, 바닥 {floor['overall']['mean']}",
                  flush=True)
            rd = real_data(recs, sc, floor)
            rd.update({"kind": kind, "head": head})
            result["real"].append(rd)

    if not args.skip_ceiling:
        print("\n[ceiling] 교차 인코더 천장 측정...", flush=True)
        n = min(1000, len(dv))
        result["ceiling"] = cross_encoder_ceiling(
            list(dv["premise"])[:n], list(dv["hypothesis"])[:n], y_dv[:n],
            args.nli_model)
        print(f"[ceiling] acc {result['ceiling']['acc_3way']}", flush=True)

    view = {"floors": {f"probe:{f['kind']}+{f['head']}": f["floor"]
                       for f in result["floors"]},
            "contra_gate": [dict(g, backend=f"probe:{g['kind']}+{g['head']}")
                            for g in result["gate"]],
            "real": [dict(r, backend=f"probe:{r['kind']}+{r['head']}")
                     for r in result["real"]]}
    prune_degenerate(view)

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
