"""COMET 기준 그림 두 장.

A. `comet_tradeoff.png` — 품질–지연 곡선. `plot_tradeoff.py` 의 COMET 판이다.
   BLEU 판과 결정적으로 다른 점: **세 패널의 y축을 공유할 수 있다.** COMET 은 다국어
   인코더 하나를 쓰므로 언어를 가로질러 읽어도 되고, 그게 이 그림의 존재 이유다.

B. `crosslang_retention.png` — 이 실험이 실제로 물은 것. 프롬프트 하나가 세 언어에서
   유지되는가를 T 축 위에서 본다. 왼쪽 BLEU, 오른쪽 COMET. 같은 척도(retention)를 두 지표로
   그린 small multiples 이지 이중축이 아니다. 음영은 세 언어의 폭(max−min)이다.

색은 `plot_tradeoff.py` 의 all-pairs 검증 통과본 4종을 그대로 쓴다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

SERIES = [
    ("auto",         BLUE,   "-",  "o", "Multi-agent loop (ours)"),
    ("auto_greedy",  BLUE,   "--", "s", "…same boundaries, no ranking"),
    ("causal_align", AQUA,   "-",  "o", "Causal align (TransLLaMa)"),
    ("alignatt",     ORANGE, "-",  "o", "AlignAtt (Papi 2023)"),
    ("syntax",       VIOLET, "-",  "o", "Syntactic chunks (SASST-style)"),
]
OUT_OF_BAND = [
    ("mu_prefix", "s", "Prefix-match MU (Zhang 2020)"),
    ("punct",     "^", "Punctuation"),
]
LANG = [("de", BLUE, "o"), ("ja", ORANGE, "s"), ("zh", AQUA, "^")]
T_GRID = [4, 6, 8, 12]
YLO, YHI = 0.55, 0.95

D = Path(__file__).resolve().parents[2] / "runs" / "en-multi" / "clean500" / "bleu"
B = {t: json.loads((D / f"{t}.json").read_text(encoding="utf-8"))
     for t, _, _ in LANG}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
})


def frame(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── A. 품질–지연 곡선 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)

for ax, (tgt, _, _) in zip(axes, LANG):
    C = B[tgt]["conditions"]
    frame(ax)
    unseg = C["unsegmented"]
    ax.axhline(unseg["comet"], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.015, unseg["comet"] + 0.004,
            f"full-sentence ceiling {unseg['comet']:.3f}",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            color=INK2, fontsize=7.5)

    for prefix, color, ls, mk, label in SERIES:
        pts = sorted((C[f"{prefix}_T{T}"]["laal_ms"], C[f"{prefix}_T{T}"]["comet"])
                     for T in T_GRID if f"{prefix}_T{T}" in C
                     and C[f"{prefix}_T{T}"].get("laal_ms") is not None)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=color, lw=2, ms=8, mec=SURFACE, mew=2, zorder=5,
                label=label, alpha=1.0 if ls == "-" else 0.75)

    for prefix, mk, label in OUT_OF_BAND:
        pts = sorted((C[f"{prefix}_T{T}"]["laal_ms"], C[f"{prefix}_T{T}"]["comet"])
                     for T in T_GRID if f"{prefix}_T{T}" in C
                     and C[f"{prefix}_T{T}"].get("laal_ms") is not None)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", marker=mk,
                color=INK2, lw=1.5, ms=6, mec=SURFACE, mew=1.5, alpha=0.55,
                zorder=4, label=label + "  (latency band does not overlap)")

    # 기계분절은 축 아래로 한참 벗어난다 — 잘라내되 어디 있는지는 밝힌다.
    m = C.get("mechanical_8")
    if m and m.get("laal_ms") is not None:
        ax.annotate(f"mechanical_8\n{m['comet']:.3f} (off scale)",
                    xy=(m["laal_ms"], YLO), xytext=(m["laal_ms"] + 40, YLO + 0.055),
                    color=INK2, fontsize=7.5, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1))

    ax.set_xlabel("LAAL (ms of source audio)  ←  lower is better")
    ax.set_title(f"en→{tgt}", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(YLO, YHI)
    xmax = max(c["laal_ms"] for n, c in C.items()
               if n != "unsegmented" and c.get("laal_ms"))
    ax.set_xlim(0, xmax + 600)

axes[0].set_ylabel("COMET  (wmt22-comet-da)")
h, l = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
for hh, ll in zip(h2, l2):
    if ll not in l:
        h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8,
           handletextpad=0.5, columnspacing=1.8, bbox_to_anchor=(0.5, 0.005))
fig.text(0.008, 0.965, "Quality–latency trade-off on unseen FLEURS 500, scored by COMET "
         "(same translator, gtx; T = 4/6/8/12 per curve)",
         ha="left", va="top", fontsize=12.5, fontweight="bold")
fig.text(0.008, 0.915,
         "Upper-left is better. Unlike BLEU, COMET shares one multilingual encoder — so the "
         "y-axis IS comparable across the three panels (per-pair bias remains).",
         ha="left", va="top", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.10, 1, 0.885))
fig.savefig(D / "comet_tradeoff.png", dpi=200, facecolor=SURFACE)
fig.savefig(D / "comet_tradeoff.pdf", facecolor=SURFACE)
print("saved", D / "comet_tradeoff.png")


# ── B. 언어 간 안정성 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)

for ax, (key, name) in zip(axes, (("retention_bleu", "BLEU"),
                                  ("retention_comet", "COMET"))):
    frame(ax)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    # 같은 T 라도 타깃마다 실측 laal 이 조금씩 다르다(최대 ~280ms). 폭은 **같은 T** 끼리의
    # 값 차이이고, 그것을 그리는 x 위치로는 세 타깃의 평균 laal 을 쓴다.
    lo = [min(B[t]["conditions"][f"auto_T{T}"][key] for t, _, _ in LANG)
          for T in T_GRID]
    hi = [max(B[t]["conditions"][f"auto_T{T}"][key] for t, _, _ in LANG)
          for T in T_GRID]
    xm = [sum(B[t]["conditions"][f"auto_T{T}"]["laal_ms"] for t, _, _ in LANG) / 3
          for T in T_GRID]
    ax.fill_between(xm, lo, hi, color=INK2, alpha=0.10, zorder=2, lw=0)
    for tgt, color, mk in LANG:
        C = B[tgt]["conditions"]
        xs = [C[f"auto_T{T}"]["laal_ms"] for T in T_GRID]
        ys = [C[f"auto_T{T}"][key] for T in T_GRID]
        ax.plot(xs, ys, "-", marker=mk, color=color, lw=2, ms=8,
                mec=SURFACE, mew=2, zorder=5, label=f"en→{tgt}")
    for T, x, a, b in zip(T_GRID, xm, lo, hi):
        ax.annotate(f"{b - a:.3f}", xy=(x, a - 0.014), ha="center", va="top",
                    fontsize=7.5, color=INK2, zorder=6)
        ax.annotate(f"T={T}", xy=(x, b + 0.012), ha="center", va="bottom",
                    fontsize=7.5, color=INK2, zorder=6)
    ax.set_xlabel("LAAL (ms of source audio)  ←  lower latency is better")
    ax.set_title(f"retention by {name}", loc="left", fontsize=11,
                 fontweight="bold", pad=8)
    ax.set_ylim(0.44, 1.06)

axes[0].set_ylabel("retention  =  segmented / full-sentence")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h + [plt.Line2D([], [], color=INK2, alpha=0.25, lw=8)],
           l + ["spread across the three targets (max − min)"],
           loc="lower center", ncol=4, frameon=False, fontsize=8,
           handletextpad=0.6, columnspacing=1.8, bbox_to_anchor=(0.5, 0.005))
fig.text(0.008, 0.965, "Does one prompt hold across three targets?",
         ha="left", va="top", fontsize=12.5, fontweight="bold")
fig.text(0.008, 0.912,
         "Same segmentations, same translator; each curve runs T = 4/6/8/12. Under BLEU the three "
         "targets fan out (spread 0.178 at T=4); under COMET they nearly coincide (0.050 → 0.008).",
         ha="left", va="top", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.10, 1, 0.885))
fig.savefig(D / "crosslang_retention.png", dpi=200, facecolor=SURFACE)
fig.savefig(D / "crosslang_retention.pdf", facecolor=SURFACE)
print("saved", D / "crosslang_retention.png")
