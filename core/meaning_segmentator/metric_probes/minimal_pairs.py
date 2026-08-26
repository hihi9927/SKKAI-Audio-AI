"""통제된 최소쌍 진단 — 임베딩이 **무엇을 잡고 무엇을 못 잡는지** 를 직접 본다.

세션 초반에 "임베딩은 부정을 못 잡는다" 로 정리했는데 **그것이 틀렸다.** 통제된
최소쌍을 직접 넣어 보면 부정은 재서술보다 확실히 낮게 나온다. 못 잡는 것은 따로 있다 —
**참여자 뒤바뀜**이다. 토큰 집합이 그대로라 풀링된 벡터가 거의 안 움직인다.

이 파일은 그 진단을 재현 가능하게 고정한다. 관문이 아니라 **진단**이다 — 통과·탈락을
매기지 않고, 어떤 변이가 어떤 값을 받는지 표로만 남긴다.

각 세트는 기준문 하나와 변이들로 구성되고, 변이는 **한 곳만** 바꾼다. 기대 순서는

    identical > paraphrase > (그 외 의미 변이) > unrelated

이고, 어떤 의미 변이가 `paraphrase` 위로 올라오면 그 유형은 **그 백엔드의 맹점**이다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.minimal_pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..autoseg import metrics
from .embed_check import MODELS
from .embed_probe import FrozenEncoder

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS

# 변이 이름은 `validity_cases.json` 의 어휘를 따른다 — 같은 축을 말할 때 같은 말을 쓴다.
SETS = [
    {
        "id": "polarity-en",
        "base": "That's a problem",
        "variants": {
            "identical": "That's a problem",
            "paraphrase": "That's an issue",
            "negation_flip": "That's not a problem",
            "negation_flip2": "That isn't a problem",
            "antonym": "That's a blessing",
            "unrelated": "The weather is nice today",
        },
    },
    {
        "id": "participant-en",
        "base": "Kim handed the report to Manager Park",
        "variants": {
            "identical": "Kim handed the report to Manager Park",
            "paraphrase": "Kim gave the report to Manager Park",
            "role_swap": "Manager Park handed the report to Kim",
            "negation_flip": "Kim didn't hand the report to Manager Park",
            "unrelated": "The weather is nice today",
        },
    },
    {
        "id": "quantifier-en",
        "base": "All the students passed the exam",
        "variants": {
            "identical": "All the students passed the exam",
            "paraphrase": "Every student passed the exam",
            "scope_change": "Some of the students passed the exam",
            "negation_flip": "None of the students passed the exam",
            "unrelated": "The weather is nice today",
        },
    },
    {
        "id": "modality-en",
        "base": "The meeting will be held tomorrow",
        "variants": {
            "identical": "The meeting will be held tomorrow",
            "paraphrase": "The meeting is going to take place tomorrow",
            "modal_change": "The meeting might be held tomorrow",
            "negation_flip": "The meeting will not be held tomorrow",
            "unrelated": "The weather is nice today",
        },
    },
    {
        "id": "number-en",
        "base": "We hired three engineers last quarter",
        "variants": {
            "identical": "We hired three engineers last quarter",
            "paraphrase": "We took on three engineers last quarter",
            "number_change": "We hired thirty engineers last quarter",
            "negation_flip": "We didn't hire three engineers last quarter",
            "unrelated": "The weather is nice today",
        },
    },
    {
        # 타깃이 영어가 아닐 때를 위한 대조. 같은 맹점이 언어를 넘어 남는지 본다.
        "id": "participant-ko",
        "base": "김 대리가 박 과장에게 보고서를 넘겼다",
        "variants": {
            "identical": "김 대리가 박 과장에게 보고서를 넘겼다",
            "paraphrase": "김 대리가 박 과장에게 보고서를 전달했다",
            "role_swap": "박 과장이 김 대리에게 보고서를 넘겼다",
            "negation_flip": "김 대리가 박 과장에게 보고서를 넘기지 않았다",
            "unrelated": "오늘 날씨가 좋아서 산책을 했다",
        },
    },
]

ORDER = ["identical", "paraphrase", "negation_flip", "negation_flip2", "antonym",
         "role_swap", "scope_change", "modal_change", "number_change", "unrelated"]


# ── 조각 vs 전체 문장, **어절 수를 맞춘** 대조 ──────────────────────────
#
# 위 SETS 는 완결 문장끼리 비교한다. 그런데 실제 관문·목적함수는 **조각 vs 전체 문장**
# 이라 길이가 어긋난다 — 그 조건에서도 극성이 읽히는지는 따로 물어야 한다.
#
# 여기서는 전체 문장 F(부정문) 하나에 대해 **어절 수가 같은** 조각 두 개를 둔다.
#   aligned  F 의 극성을 지킨 부분 렌더링
#   flipped  극성만 뒤집은 것
# 길이가 같으므로 두 값의 차이는 **오직 극성**에서 온다. 길이 교란을 통제한 셈이다.
FRAGMENT_TRIPLES = [
    ("I don't think that will be a problem.",  "I don't think that",           "I do think that"),
    ("She didn't finish the report on time.",  "She didn't finish the report", "She did finish the report"),
    ("The meeting will not be held tomorrow.", "The meeting won't be held",    "The meeting will be held"),
    ("He never agreed to the terms.",          "He never agreed",              "He fully agreed"),
    ("We can't accept this proposal.",         "We can't accept this",         "We can accept this"),
    ("They have not left the building.",       "They have not left",           "They have now left"),
]


def fragment_check(encoders: list[str]) -> dict:
    """길이를 맞춘 조각 대조. **aligned 가 flipped 보다 가까워야 정답.**"""
    import numpy as np
    out: dict = {"n": len(FRAGMENT_TRIPLES), "rows": [], "backends": {}}
    texts = [t for tr in FRAGMENT_TRIPLES for t in tr]

    for key in encoders:
        enc = FrozenEncoder(key, batch_size=16)
        V = enc.encode(texts)
        V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
        vals = [(float(V[3 * i] @ V[3 * i + 1]), float(V[3 * i] @ V[3 * i + 2]))
                for i in range(len(FRAGMENT_TRIPLES))]
        out["backends"][f"cos:{key}"] = {
            "correct": sum(1 for a, b in vals if a > b),
            "mean_gap": round(float(np.mean([a - b for a, b in vals])), 4),
            "pairs": [[round(a, 4), round(b, 4)] for a, b in vals]}
        enc._model = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    b = metrics.make_contradiction_backend()
    F = [t[0] for t in FRAGMENT_TRIPLES]
    ca = b.score(F, [t[1] for t in FRAGMENT_TRIPLES])
    cb = b.score(F, [t[2] for t in FRAGMENT_TRIPLES])
    out["backends"]["contra:xlmr-anli"] = {
        "correct": sum(1 for a, c in zip(ca, cb) if a < c),
        "mean_gap": round(float(sum(c - a for a, c in zip(ca, cb)) / len(ca)), 4),
        "pairs": [[round(a, 4), round(c, 4)] for a, c in zip(ca, cb)]}
    out["rows"] = [{"full": t[0], "aligned": t[1], "flipped": t[2]}
                   for t in FRAGMENT_TRIPLES]
    return out


def embed_scores(key: str, batch_size: int = 32) -> dict:
    """세트별 `cos(기준, 변이)`."""
    import numpy as np
    enc = FrozenEncoder(key, batch_size=batch_size)
    out = {}
    for s in SETS:
        labels = [k for k in ORDER if k in s["variants"]]
        texts = [s["base"]] + [s["variants"][k] for k in labels]
        V = enc.encode(texts)
        V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
        out[s["id"]] = {k: round(float(V[0] @ v), 4) for k, v in zip(labels, V[1:])}
    enc._model = None
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return out


def nli_scores() -> dict:
    """세트별 `1 − P(기준 ⊨ 변이)`. 임베딩 거리와 **방향을 맞춘다** (낮을수록 가깝다).

    `1 − contradiction` 이 아니라 함의를 쓰는 이유는, 여기서 재는 것이 '모순인가' 가
    아니라 '의미가 보존됐는가' 라서다 — 표의 다른 열과 같은 질문이어야 비교가 된다."""
    b = metrics.ContradictionBackend(model_name=metrics.NLI_MODEL)
    pipe = b.load()
    out = {}
    for s in SETS:
        labels = [k for k in ORDER if k in s["variants"]]
        items = [{"text": s["base"], "text_pair": s["variants"][k]} for k in labels]
        res = pipe(items, batch_size=8)
        out[s["id"]] = {k: round(float(metrics._NliBase._prob(sc, "entail")), 4)
                        for k, sc in zip(labels, res)}
    return out


def render(result: dict) -> str:
    L = ["# 통제된 최소쌍 진단 — 임베딩은 무엇을 잡고 무엇을 못 잡는가", "",
         "관문이 아니라 **진단**이다. 통과·탈락을 매기지 않고 어떤 변이가 어떤 값을 "
         "받는지만 본다. 각 변이는 기준문에서 **한 곳만** 바꾼다.", "",
         "읽는 법: `paraphrase`(무해한 재서술)가 기준선이다. **어떤 의미 변이가 "
         "`paraphrase` 보다 높으면 그 유형은 그 백엔드의 맹점이다.**", ""]

    cols = result["backends"]
    for s in SETS:
        L += [f"## `{s['id']}`", "", f"기준문: `{s['base']}`", "",
              "| 변이 | 문장 | " + " | ".join(cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        para = {c: result["scores"][c][s["id"]].get("paraphrase") for c in cols}
        for k in ORDER:
            if k not in s["variants"]:
                continue
            cells = []
            for c in cols:
                v = result["scores"][c][s["id"]].get(k)
                if v is None:
                    cells.append("—")
                    continue
                p = para.get(c)
                blind = (k not in ("identical", "paraphrase") and p is not None
                         and v >= p)
                cells.append(f"**{v}** ⚠" if blind else f"{v}")
            L.append(f"| {k} | `{s['variants'][k]}` | " + " | ".join(cells) + " |")
        L.append("")
    L += ["⚠ = 그 변이가 `paraphrase` 이상으로 유사하게 나온 칸 (= 맹점).", ""]

    # 요약 — 유형별로 몇 개 백엔드에서 맹점인가
    L += ["## 요약 — 유형별 맹점", "",
          "| 변이 유형 | 맹점인 백엔드 |", "|---|---|"]
    for k in ORDER:
        if k in ("identical", "paraphrase"):
            continue
        hit = []
        for c in cols:
            for s in SETS:
                v = result["scores"][c][s["id"]].get(k)
                p = result["scores"][c][s["id"]].get("paraphrase")
                if v is not None and p is not None and v >= p:
                    hit.append(f"{c}({s['id']})")
        L.append(f"| `{k}` | {', '.join(hit) if hit else '—'} |")

    fc = result.get("fragment_check")
    if fc:
        L += ["", "## 조각 vs 전체 문장 — 어절 수를 맞춘 대조", "",
              "위 세트는 완결 문장끼리 비교한다. 그런데 실제 관문·목적함수는 **조각 vs "
              "전체 문장**이라 길이가 어긋난다. 여기서는 전체 문장 하나에 대해 **어절 수가 "
              "같은** 조각 두 개(극성만 다름)를 두어, 길이 교란을 통제한 채 극성만 본다.", "",
              "| 백엔드 | 정답 | 평균 격차 |", "|---|---|---|"]
        for k, v in fc["backends"].items():
            L.append(f"| `{k}` | {v['correct']}/{fc['n']} | {v['mean_gap']:+.4f} |")
        L += ["", "| 전체 문장 | aligned | flipped |", "|---|---|---|"]
        for r in fc["rows"]:
            L.append(f"| `{r['full']}` | `{r['aligned']}` | `{r['flipped']}` |")
        L += ["", "**길이만 맞추면 코사인도 극성을 읽는다.** 그러면 실제 관문에서 극성이 "
              "묻히는 이유는 폴라리티 맹목이 아니라 **후보들의 길이가 서로 다르다는 것**이다. "
              "자기-prefix 바닥 실측에서 길이가 만드는 코사인 거리 변동은 1–2어절 0.229 → "
              "15+ 0.044 로 약 0.185 인데, 위 극성 신호는 0.10 규모다 — **길이 교란이 극성 "
              "신호보다 크다.**", ""]

    L += ["", "## 판정", ""]
    for line in result["verdict"]:
        L.append(f"- {line}")
    if fc:
        L += ["",
              "**구조적 긴장.** 길이를 맞추면 극성이 읽히지만, 조기 방출을 검출하려면 "
              "**미래를 아는 무언가**와 비교해야 하고 그것은 정의상 방출분보다 길다. "
              "즉 *미래 지식*과 *길이 정합*은 직접 충돌한다. 그래서 필요한 것은 길이 불일치에 "
              "**본래 관대한** 자다 — 함의는 비대칭이라 '긴 전제가 짧은 가설을 함의한다' 가 "
              "자연스러운 관계지만, 코사인은 대칭이라 길이 차를 그냥 거리로 읽는다. "
              "NLI 가 이 자리를 지키는 진짜 이유가 부정 감지력이 아니라 **비대칭성**이다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="통제된 최소쌍 진단")
    p.add_argument("--encoders", nargs="+", default=["e5-inst", "gte-base"],
                   choices=sorted(MODELS))
    p.add_argument("--skip-nli", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (OUT_RUNS / "minimal_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict = {}
    for key in args.encoders:
        print(f"[{key}] 인코딩...", flush=True)
        scores[f"cos:{key}"] = embed_scores(key)
    if not args.skip_nli:
        print("[xlmr-anli] 함의...", flush=True)
        scores["entail:xlmr-anli"] = nli_scores()

    result = {"backends": list(scores), "scores": scores, "verdict": []}
    print("[fragment] 길이 맞춘 조각 대조...", flush=True)
    result["fragment_check"] = fragment_check(args.encoders)

    # 판정 — 자동 집계
    def blind(col, variant):
        return [s["id"] for s in SETS
                if (scores[col][s["id"]].get(variant) is not None
                    and scores[col][s["id"]].get("paraphrase") is not None
                    and scores[col][s["id"]][variant]
                    >= scores[col][s["id"]]["paraphrase"])]

    for col in scores:
        neg = blind(col, "negation_flip")
        role = blind(col, "role_swap")
        result["verdict"].append(
            f"`{col}` — 부정 맹점 {len(neg)}세트{f' {neg}' if neg else ''}, "
            f"참여자 맹점 {len(role)}세트{f' {role}' if role else ''}")

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render(result), encoding="utf-8")
    print("\n" + render(result))
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
