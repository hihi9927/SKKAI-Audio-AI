"""품질–지연 트레이드오프 — 정책별 곡선. x축 ms LAAL, y축은 분할(broken) 축.

정책들이 좁은 BLEU 구간에 몰려 있어 0부터 그리면 곡선이 서로 겹쳐 안 보인다. 위 칸을
관심 구간으로 확대하고, 아래 칸에 하한(기계분절)만 따로 둔 뒤 사이를 `⋯` 로 축약한다.

`auto_greedy` 는 경쟁 정책이 아니라 `auto` 의 대조군(같은 경계, 순위 없이 절단)이므로
색을 나누지 않고 같은 파랑 + 파선으로 둔다. 색상 4종은 all-pairs 검증 통과본이다.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
# 다른 정책과 지연대가 겹치지 않아 맞대결이 성립하지 않는다 — 회색 참조.
OUT_OF_BAND = [
    ("mu_prefix", "s", "Prefix-match MU (Zhang 2020)"),
    ("punct",     "^", "Punctuation"),
]
T_GRID = [4, 6, 8, 12]

_ap = argparse.ArgumentParser(description="품질–지연 곡선")
_ap.add_argument("--metric", default="bleu", choices=["bleu", "comet"],
                 help="y축 품질 지표. comet 은 `comet_score.py` 를 먼저 돌려야 한다")
_ap.add_argument("--out", default=None, help="출력 파일 stem (기본: tradeoff[_comet])")
_ap.add_argument("--run-id", default="en-multi/clean500")
_ap.add_argument("--targets", nargs="+", default=["de", "ja"])
_ap.add_argument("--title", default="unseen FLEURS 500")
ARGS = _ap.parse_args()
M = ARGS.metric
STEM = ARGS.out or ("tradeoff" if M == "bleu" else f"tradeoff_{M}")

d = Path("core/meaning_segmentator/runs") / ARGS.run_id / "bleu"
TARGETS = ARGS.targets
blobs = {t: json.loads((d / f"{t}.json").read_text(encoding="utf-8"))
         for t in TARGETS}
missing = [t for t, b in blobs.items()
           if any(M not in c for c in b["conditions"].values())]
if missing:
    raise SystemExit(f"{missing} 에 `{M}` 값이 없다 — comet_score.py 를 먼저 돌릴 것")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
})
fig = plt.figure(figsize=(5.5 * len(TARGETS) if len(TARGETS) > 1 else 6.4, 6.0))
# COMET 은 눈금 라벨이 넓고(0.850) 부제가 한 줄 길어 여백을 더 준다.
gs = GridSpec(2, len(TARGETS), figure=fig, height_ratios=[7, 1], hspace=0.10, wspace=0.22,
              left=0.095 if M == "comet" else 0.075, right=0.985,
              top=0.815 if M == "comet" else 0.845, bottom=0.20)


def curve(C, prefix):
    pts = [(C[f"{prefix}_T{T}"]["laal_ms"], C[f"{prefix}_T{T}"][M])
           for T in T_GRID
           if f"{prefix}_T{T}" in C and C[f"{prefix}_T{T}"].get("laal_ms") is not None]
    return sorted(pts)


for col, tgt in enumerate(TARGETS):
    C = blobs[tgt]["conditions"]
    hi = fig.add_subplot(gs[0, col])       # 관심 구간
    lo = fig.add_subplot(gs[1, col], sharex=hi)   # 하한만

    unseg = C["unsegmented"]
    mech = C.get("mechanical_8")
    # **축은 맞대결 가능한 정책만으로 잡는다.** 지연대가 안 겹치는 둘과 천장은 축을
    # 늘리기만 하고 비교에 쓰이지 않으므로 밖으로 빼고 값만 주석으로 남긴다.
    ys = [y for p, *_ in SERIES for _, y in curve(C, p)]
    pad = 1.3 if M == "bleu" else 0.012
    hi.set_ylim(min(ys) - pad, max(ys) + pad * 1.25)
    floor_v = mech[M] if mech else 0.0
    lo.set_ylim(min(0.0, floor_v - pad), floor_v + pad * 2)

    for ax in (hi, lo):
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    hi.spines["bottom"].set_visible(False)
    hi.tick_params(labelbottom=False, bottom=False)

    # 축 분할 표시 — 두 칸 사이가 축약됐음을 알린다.
    for ax, y in ((hi, 0.0), (lo, 1.0)):
        ax.plot([-0.012, 0.012], [y - 0.012, y + 0.012], transform=ax.transAxes,
                color=INK2, lw=1, clip_on=False, zorder=10)
    lo.text(-0.055, 1.35, "⋯", transform=lo.transAxes, ha="center", va="center",
            color=INK2, fontsize=13, clip_on=False)

    for prefix, color, ls, mk, label in SERIES:
        pts = curve(C, prefix)
        if not pts:
            continue
        hi.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=color, lw=2, ms=8, mec=SURFACE, mew=2, zorder=5,
                label=label, alpha=1.0 if ls == "-" else 0.75)

    fmt = (lambda v: f"{v:.1f}") if M == "bleu" else (lambda v: f"{v:.3f}")
    off = [f"ceiling {fmt(unseg[M])} @ {unseg['laal_ms'] / 1000:.1f}s"]
    for prefix, _mk, label in OUT_OF_BAND:
        pts = curve(C, prefix)
        if pts:
            off.append(f"{label.split(' (')[0]} {fmt(min(y for _, y in pts))}–"
                       f"{fmt(max(y for _, y in pts))} @ "
                       f"{min(x for x, _ in pts) / 1000:.1f}–"
                       f"{max(x for x, _ in pts) / 1000:.1f}s")
    hi.text(0.985, 0.035, "off scale (latency bands do not overlap):\n" + "\n".join(off),
            transform=hi.transAxes, ha="right", va="bottom", color=INK2,
            fontsize=7, linespacing=1.5, zorder=8)

    if mech and mech.get("laal_ms") is not None:
        lo.plot([mech["laal_ms"]], [mech[M]], "o", color=INK2, ms=7,
                mec=SURFACE, mew=2, zorder=5, label="Mechanical 8-char (floor)")
        lo.annotate(f"mechanical {fmt(mech[M])}",
                    (mech["laal_ms"], mech[M]), textcoords="offset points",
                    xytext=(9, -3), color=INK2, fontsize=7.5)

    xs = [x for p, *_ in SERIES for x, _ in curve(C, p)]
    if mech and mech.get("laal_ms"):
        xs.append(mech["laal_ms"])
    span = max(xs) - min(xs)
    hi.set_xlim(min(xs) - span * 0.06, max(xs) + span * 0.08)
    lo.set_xlabel("LAAL (ms of source audio)  ←  lower latency is better",
                  labelpad=6)
    ylab = (f"BLEU  (en→{tgt}, {blobs[tgt]['tokenize']})" if M == "bleu"
            else f"COMET  (en→{tgt}, wmt22-comet-da)")
    hi.set_ylabel(ylab)
    hi.yaxis.set_label_coords(-0.135 if M == "comet" else -0.085, 0.42)
    hi.set_title(f"en→{tgt}", loc="left", fontsize=11, fontweight="bold", pad=8)

h, l = fig.axes[0].get_legend_handles_labels()
h2, l2 = fig.axes[1].get_legend_handles_labels()
fig.legend(h + h2, l + l2, loc="lower center", ncol=3, frameon=False, fontsize=8,
           handletextpad=0.5, columnspacing=1.8, bbox_to_anchor=(0.5, 0.012))
fig.text(0.008, 0.985, f"{'BLEU' if M == 'bleu' else 'COMET'}–latency trade-off on "
         f"{ARGS.title} (same translator, gtx; T = 4/6/8/12 per curve)",
         ha="left", va="top", fontsize=12.5, fontweight="bold")
fig.text(0.008, 0.945,
         "Upper-left is better. LAAL is forced-aligned (Qwen3-ForcedAligner; wav2vec2 CTC "
         "agrees within 22 ms).\nAxes are zoomed to the policies whose latency bands overlap; "
         "y is broken (⋯) for the floor. Off-scale values are listed per panel.\n"
         + ("BLEU is NOT comparable across panels (de 13a, ja ja-mecab)."
            if M == "bleu" else
            "COMET uses one multilingual encoder, so panels are far more comparable "
            "than under BLEU — but it stays reference-based.")
         if len(TARGETS) > 1 else
         ("BLEU tokenisation: " + blobs[TARGETS[0]]["tokenize"] if M == "bleu"
          else "COMET: wmt22-comet-da (reference-based)."),
         ha="left", va="top", fontsize=7.5, color=INK2, linespacing=1.6)
fig.savefig(d / f"{STEM}.png", dpi=200, facecolor=SURFACE)
fig.savefig(d / f"{STEM}.pdf", facecolor=SURFACE)
print("saved", d / f"{STEM}.png")
