"""x→en 세 트랙의 제안 정책 결과를 표 하나로 모은다 (비교군 없음).

세 트랙 모두 **타깃이 영어**라 토크나이저가 `13a` 로 같다 — en→X 표와 달리 패널을
가로지르는 BLEU 판독이 성립한다. 대신 **소스 단위가 다르다**: de 는 어절, ja·zh 는
문자라 `T` 격자 값 자체를 트랙 간에 맞댈 수 없다. 그래서 가로축은 T 가 아니라
**강제정렬 실측 지연(ms)** 이고, 트랙 비교는 그 위에서만 한다.

    python core/meaning_segmentator/tools/covost2_chain/x2en_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "core/meaning_segmentator/experiment/artifacts/covost2"

TRACKS = [                      # (디렉토리, 소스, 소스 단위)
    ("de-en_n3000", "de", "어절"),
    ("ja-en_n678", "ja", "문자"),
    ("zh-en_n3000", "zh", "문자"),
]
BANDS = [1600, 1900, 2200]      # 트랙 비교용 등지연 지점 (ms)


def order_key(name: str) -> tuple:
    """무분절 → auto → auto_greedy → mechanical 순, 그 안에서는 T 오름차순."""
    rank = {"unsegmented": 0}.get(name, 3)
    if name.startswith("auto_greedy_T"):
        rank, t = 2, int(name.split("_T")[1])
    elif name.startswith("auto_T"):
        rank, t = 1, int(name.split("_T")[1])
    else:
        t = 0
    return (rank, t)


def interp(pts: list[tuple[float, float]], x: float) -> float | None:
    """측정점 사이 선형 보간. 범위 밖은 None (외삽하지 않는다)."""
    pts = sorted(pts)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def main() -> int:
    L: list[str] = []
    L.append("# x→en — 제안 정책 단독 (CoVoST2 test, min_gap=1 라벨 `auto_run02`)\n")
    L.append("비교군(punct/syntax/causal_align/alignatt/mu_prefix)은 **이 트랙에서 아직 "
             "안 돌렸다.** 여기 있는 건 제안 정책과 그 자체 기준선(무분절 상한 / "
             "기계분절 하한 / 순위를 뺀 `auto_greedy`)뿐이다.\n")
    L.append("- 세 트랙 모두 타깃이 영어 → 토크나이저 `13a` 동일 → **트랙 간 BLEU 판독이 "
             "성립한다.** en→X 표와 다른 점이다.")
    L.append("- 소스 단위는 다르다 (de 어절 / ja·zh 문자). **`T` 값 자체를 트랙 간에 "
             "맞대면 안 된다** — 가로축은 강제정렬 실측 지연(ms)이다.")
    L.append("- `auto_greedy_T*` = 같은 마킹에서 **LLM 순위를 빼고** 등간격 절단한 것. "
             "`auto_T*` 와의 차이가 순위 이득이다.")
    L.append("- Δ 는 무분절 대비 corpus BLEU 차, 대괄호는 paired bootstrap 95% CI "
             "(sacrebleu, n=1000).")
    L.append("- COMET 은 `wmt22-comet-da`. 값이 없는 트랙은 열이 `—` 로 나온다.\n")

    curves: dict[str, list[tuple[float, float]]] = {}
    ret_curves: dict[str, list[tuple[float, float]]] = {}
    com_curves: dict[str, list[tuple[float, float]]] = {}
    for d, src, unit in TRACKS:
        path = RUNS / d / "bleu" / "en.json"
        if not path.exists():
            L.append(f"## {src}→en — 산출 없음 ({path})\n")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        C = blob["conditions"]
        L.append(f"## {src}→en (n={blob['n']}, 소스 단위 {unit}, tok:{blob['tokenize']}, "
                 f"번역기 `{blob.get('translator', '?')}`)\n")
        L.append("| 조건 | k | LAAL ms ↓ | BLEU ↑ | COMET ↑ | chrF2 | retention(BLEU) | "
                 "Δ vs unseg [95% CI] |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name in sorted(C, key=order_key):
            c = C[name]
            pv = c.get("paired_vs_unseg")
            dv = ("—" if not pv else
                  f"{pv['delta']:+.2f} [{pv['ci95'][0]:+.2f}, {pv['ci95'][1]:+.2f}]")
            ms = c.get("laal_ms")
            cm = c.get("comet")
            L.append(f"| `{name}` | {c['k']:.2f} | {'—' if ms is None else f'{ms:.0f}'} | "
                     f"{c['bleu']:.2f} | {'—' if cm is None else f'{cm:.4f}'} | "
                     f"{c['chrf2']:.2f} | "
                     f"{c.get('retention_bleu', float('nan')):.4f} | {dv} |")
        L.append("")
        curves[src] = [(C[n]["laal_ms"], C[n]["bleu"]) for n in C
                       if n.startswith("auto_T") and C[n].get("laal_ms")]
        com_curves[src] = [(C[n]["laal_ms"], C[n]["comet"]) for n in C
                           if n.startswith("auto_T") and C[n].get("laal_ms")
                           and C[n].get("comet") is not None]
        ret_curves[src] = [(C[n]["laal_ms"], C[n]["retention_bleu"]) for n in C
                           if n.startswith("auto_T") and C[n].get("laal_ms")
                           and C[n].get("retention_bleu") is not None]

    L.append("## 트랙 비교 — 같은 지연에서 읽는다\n")
    L.append("측정점 사이 **선형 보간**이고 범위 밖은 외삽하지 않는다(`—`). 트랙마다 T "
             "격자가 다르므로 이 방식 외에는 맞댈 방법이 없다.\n")
    srcs = [s for _, s, _ in TRACKS if s in curves]
    L.append("| 지연 | " + " | ".join(f"{s}→en BLEU" for s in srcs) + " | "
             + " | ".join(f"{s} COMET" for s in srcs) + " | "
             + " | ".join(f"{s} retention" for s in srcs) + " |")
    L.append("|---" * (1 + 3 * len(srcs)) + "|")
    for x in BANDS:
        cells = []
        for s in srcs:
            v = interp(curves[s], x)
            cells.append("—" if v is None else f"{v:.2f}")
        for s in srcs:
            v = interp(com_curves.get(s, []), x)
            cells.append("—" if v is None else f"{v:.4f}")
        for s in srcs:
            v = interp(ret_curves[s], x)
            cells.append("—" if v is None else f"{v:.4f}")
        L.append(f"| {x} ms | " + " | ".join(cells) + " |")
    L.append("")
    L.append("측정 지연 범위: " + ", ".join(
        f"{s} {min(x for x, _ in curves[s]):.0f}–{max(x for x, _ in curves[s]):.0f}ms"
        for s in srcs) + ".")

    out = RUNS / "X2EN_OURS.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
