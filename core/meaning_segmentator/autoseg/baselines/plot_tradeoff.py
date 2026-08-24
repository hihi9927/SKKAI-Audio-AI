"""품질–지연 트레이드오프 — 정책별 곡선. x축 ms LAAL, y축 단일(비축약) 축.

정책 6종을 모두 한 축에 그린다. `punct` 와 `mu_prefix` 는 다른 정책보다 훨씬 느린
지연대에 있어 맞대결(같은 지연에서의 품질 비교)은 성립하지 않지만, 곡선이 어디로
향하는지 보여주므로 그림에는 남긴다 — 지연대가 겹치지 않는다는 사실 자체가 결과다.

offline 상한(무분절 통번역)은 가로 파선으로만 표시한다. 상한의 지연은 정책들보다
2~5배 커서 x 범위에 넣으면 관심 구간이 짓눌리므로, 값과 지연은 주석으로 적는다.

`punct` 는 T 격자에 반응하지 않아 곡선이 아니라 점 하나로 그린다 (아래 `SINGLE` 주석).

`punct`/`mu_prefix` 가 멀리 떨어져 있어 x축 가운데에 점이 하나도 없는 빈 구간이
생긴다. 그 구간은 `FuncScale` 조각선형 변환으로 축약하고 `⋯` 로 표시한다 — 눈금은
축약 구간 안쪽을 빼고 다시 잡는다. 점의 x값 자체는 건드리지 않는다.

색 6종은 all-pairs CIEDE2000 × {정상, 적/녹/청색맹} 검증본이다. 최소 ΔE 11.1
(auto↔causal, 청색맹)로, 기존 4색본과 동일한 하한을 유지한다.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
MAGENTA, BROWN = "#c9268f", "#8a4b1a"

SERIES = [
    ("auto",         BLUE,    "-",  "o", "Multi-agent loop (ours)"),
    ("causal_align", AQUA,    "-",  "s", "Causal align (TransLLaMa)"),
    ("alignatt",     ORANGE,  "-",  "^", "AlignAtt (Papi 2023)"),
    ("syntax",       VIOLET,  "-",  "D", "Syntactic chunks (SASST-style)"),
    ("mu_prefix",    MAGENTA, "-",  "v", "Prefix-match MU (Zhang 2020)"),
]
# **`punct` 는 곡선이 아니라 점 하나다.** `coarsen` 은 경계를 *지우기만* 하므로 정책이
# 예산보다 적게 찍으면 T 를 바꿔도 산출이 그대로다. 구두점은 원래 성기게 찍어서
# (k=1.6~2.1) T 격자가 지연을 만들지 못하고, T 를 키우면 남은 경계마저 지워져
# 무분절 쪽으로 끌려갈 뿐이다 (FLEURS de: k 2.07→1.67, laal 4889→5093ms).
# 그 점들을 이어 그리면 없는 노브가 있는 것처럼 보인다.
SINGLE = [("punct", BROWN, "X", "Punctuation (no latency knob)")]
T_GRID = [4, 6, 8, 12]
GAP_MIN = 0.20   # 이보다 넓은 빈 구간만 축약 (전체 x 폭 대비).
                 # 0.12 로 내리면 경쟁 정책 사이의 정상적인 간격까지 잘린다.
GAP_KEEP = 0.045  # 축약 후 남길 폭
GAP_PAD = 0.015   # 축약 구간 양끝에 남길 여유 (마커가 잘리지 않게)


def find_gaps(xs, lo, hi):
    """점이 하나도 없는 넓은 x 구간을 찾는다."""
    span = hi - lo
    out, pts = [], sorted(set(xs))
    for a, b in zip(pts, pts[1:]):
        if b - a > GAP_MIN * span:
            out.append((a + GAP_PAD * span, b - GAP_PAD * span,
                        GAP_KEEP * span))
    return out


def gap_scale(gaps):
    """축약 구간을 좁히는 조각선형 정변환/역변환."""
    def fwd(x):
        x = np.asarray(x, dtype=float)
        y = np.array(x, dtype=float)
        for a, b, w in gaps:
            f = w / (b - a)
            y = y - np.where(x >= b, (b - a) - w,
                             np.where(x > a, (x - a) * (1 - f), 0.0))
        return y

    tb = [(float(fwd(a)), float(fwd(a)) + w) for a, b, w in gaps]

    def inv(y):
        y = np.asarray(y, dtype=float)
        x = np.array(y, dtype=float)
        for (a, b, w), (A, B) in zip(gaps, tb):
            k = (b - a) / w
            x = x + np.where(y >= B, (b - a) - w,
                             np.where(y > A, (y - A) * (k - 1), 0.0))
        return x

    return fwd, inv

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
fig, axes = plt.subplots(
    1, len(TARGETS), squeeze=False,
    figsize=(5.5 * len(TARGETS) if len(TARGETS) > 1 else 7.2, 5.9))
axes = axes[0]
_L = (0.095 if M == "comet" else 0.075) if len(TARGETS) > 1 else \
     (0.085 if M == "comet" else 0.070)
fig.subplots_adjust(left=_L, right=0.985,
                    top=0.815 if M == "comet" else 0.845, bottom=0.255,
                    wspace=0.22)


def curve(C, prefix):
    pts = [(C[f"{prefix}_T{T}"]["laal_ms"], C[f"{prefix}_T{T}"][M])
           for T in T_GRID
           if f"{prefix}_T{T}" in C and C[f"{prefix}_T{T}"].get("laal_ms") is not None]
    return sorted(pts)


for ax, tgt in zip(axes, TARGETS):
    C = blobs[tgt]["conditions"]
    unseg = C["unsegmented"]
    fmt = (lambda v: f"{v:.1f}") if M == "bleu" else (lambda v: f"{v:.3f}")
    pad = 1.3 if M == "bleu" else 0.012

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)

    # offline 상한 — gtx 통번역을 데이터셋 정답 번역으로 채점한 값.
    # 상한의 지연(x)은 축 밖이라 선으로만 긋고 값·지연은 주석으로 적는다.
    ax.axhline(unseg[M], color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=3,
               label="Full-sentence offline (ceiling)")
    ax.annotate(f"offline ceiling {fmt(unseg[M])} @ {unseg['laal_ms'] / 1000:.1f}s "
                f"(no segmentation)",
                (0.015, unseg[M]), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, -12),
                color=INK2, fontsize=7.5, zorder=6)

    for prefix, color, ls, mk, label in SERIES:
        pts = curve(C, prefix)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=color, lw=2, ms=7.5, mec=SURFACE, mew=1.8, zorder=5,
                label=label)

    for prefix, color, mk, label in SINGLE:
        c = C.get(prefix)
        if not c or c.get("laal_ms") is None:
            continue
        ax.plot([c["laal_ms"]], [c[M]], marker=mk, ls="none", color=color,
                ms=9, mec=SURFACE, mew=1.8, zorder=5, label=label)

    single = [C[p] for p, *_ in SINGLE
              if p in C and C[p].get("laal_ms") is not None]
    ys = ([y for p, *_ in SERIES for _, y in curve(C, p)]
          + [c[M] for c in single] + [unseg[M]])
    xs = ([x for p, *_ in SERIES for x, _ in curve(C, p)]
          + [c["laal_ms"] for c in single])
    ax.set_ylim(min(ys) - pad, max(ys) + pad * 1.6)
    span = max(xs) - min(xs)
    xlo, xhi = min(xs) - span * 0.06, max(xs) + span * 0.11

    # 점이 없는 넓은 구간을 축약한다 — 관심 구간(경쟁 정책들)이 짓눌리지 않도록.
    gaps = find_gaps(xs, xlo, xhi)
    if gaps:
        fwd, inv = gap_scale(gaps)
        ax.set_xscale("function", functions=(fwd, inv))
        # 눈금은 축약 구간 안쪽을 빼고 다시 잡는다.
        ticks = [t for t in MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10])
                 .tick_values(xlo, xhi)
                 if xlo - span * 0.02 <= t <= xhi + span * 0.02
                 and not any(a < t < b for a, b, _ in gaps)]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlim(xlo, xhi)

    if gaps:
        f0, f1 = (fwd(xlo), fwd(xhi))
        for a, b, _w in gaps:
            fr = (fwd((a + b) / 2) - f0) / (f1 - f0)
            for e in (a, b):   # 축약 구간의 양 끝 — 점선으로 표시
                ax.axvline(e, color=INK2, lw=0.8, ls=(0, (2, 2)),
                           alpha=0.55, zorder=2)
            for dx in (-0.006, 0.006):   # 축 위의 절단 표시
                ax.plot([fr + dx - 0.007, fr + dx + 0.007], [-0.013, 0.013],
                        transform=ax.transAxes, color=INK2, lw=1.1,
                        clip_on=False, zorder=10)
    ax.set_xlabel("LAAL (ms of source audio)  ←  lower latency is better",
                  labelpad=6)
    ylab = (f"BLEU  (en→{tgt}, {blobs[tgt]['tokenize']})" if M == "bleu"
            else f"COMET  (en→{tgt}, wmt22-comet-da)")
    ax.set_ylabel(ylab)
    ax.yaxis.set_label_coords((-0.135 if M == "comet" else -0.085)
                              * (1.0 if len(TARGETS) > 1 else 0.72), 0.5)
    ax.set_title(f"en→{tgt}", loc="left", fontsize=11, fontweight="bold", pad=6)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.8,
           handletextpad=0.5, columnspacing=1.4, labelspacing=0.55,
           bbox_to_anchor=(0.5, 0.005))
fig.text(0.008, 0.985, f"{'BLEU' if M == 'bleu' else 'COMET'}–latency trade-off on "
         f"{ARGS.title}"
         + (" (same translator, gtx; T = 4/6/8/12 per curve)"
            if len(TARGETS) > 1 else "  ·  gtx, T = 4/6/8/12"),
         ha="left", va="top", fontweight="bold",
         fontsize=12.5 if len(TARGETS) > 1 else 12.0)
fig.text(0.008, 0.945,
         "Upper-left is better. LAAL is forced-aligned (Qwen3-ForcedAligner; wav2vec2 CTC "
         "agrees within 22 ms). Empty x ranges are compressed (break marks on the axis)."
         "\nPunctuation has no latency knob (it segments below the T budget), so it is one "
         "point; it and prefix-match MU sit in a slower band — shown, not matched.\n"
         + ("BLEU is NOT comparable across panels (de 13a, ja ja-mecab)."
            if M == "bleu" else
            "COMET uses one multilingual encoder, so panels are far more comparable "
            "than under BLEU — but it stays reference-based.")
         if len(TARGETS) > 1 else
         ("Punctuation has no latency knob (it segments below the T budget), so it is one "
          "point, not a curve.\nIt and prefix-match MU sit in a slower latency band — shown, "
          "but not matched head-to-head.\nEmpty x ranges are compressed (break marks on the "
          "axis). "
          + ("BLEU tokenisation: " + blobs[TARGETS[0]]["tokenize"] if M == "bleu"
             else "COMET: wmt22-comet-da (reference-based).")),
         ha="left", va="top", fontsize=7.5, color=INK2, linespacing=1.6)
fig.savefig(d / f"{STEM}.png", dpi=200, facecolor=SURFACE)
fig.savefig(d / f"{STEM}.pdf", facecolor=SURFACE)
print("saved", d / f"{STEM}.png")
