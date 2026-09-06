#!/usr/bin/env python3
"""CoVoST2 축 비교 그림. 저장된 metric.json / comet_*.json 만 읽는다.

    .venv/bin/python evaluation/ast/plot_covost2.py

라벨은 영문이다 — 논문 그림에 그대로 들어가고 CJK 폰트 의존을 없애기 위해서.
색은 dataviz 기준 팔레트의 categorical 1~3번(all-pairs 검증 통과). static 계열은
한 계열이므로 파랑 하나를 공유하고, seg/punct 가 각각 주황/청록이다. 청록은 밝은
표면 대비가 3:1 미만이라 **직접 라벨이 필수**다 — 모든 점에 라벨을 붙인다.

**static 은 점이 아니라 곡선이다.** 청크 크기를 2/4/6초로 훑어 실측했다. 어제까지
static@2s 와 punct 를 잇는 직선을 기준선으로 썼는데, 실제 곡선은 위로 크게 볼록해서
그 직선이 static 을 심하게 과소평가하고 있었다(그래서 seg 가 "위에 있다"고 잘못 읽혔다).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# ── dataviz 기준 팔레트 (light) ─────────────────────────────────────────────
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
C_STATIC, C_SEG, C_PUNCT = "#2a78d6", "#eb6834", "#1baf7a"

# 축 → (결과 태그, 표시 이름)
# 2026-08-31 재측정(수정 3종 + MADLAD-400-3B) 기준이다. 그 이전 Google 번역 런은
# 절대 점수를 비교할 수 없어 결과째 지웠다.
TAGS = {
    "static":    ("20260831_110647", "static 2 s"),
    "static-c4": ("20260831_133542", "static 4 s"),
    "static-c6": ("20260831_142513", "static 6 s"),
    "seg":       ("20260831_110647", "seg (ours)"),
    "punct":     ("20260831_110647", "punctuation"),
}
STATIC_CURVE = ["static", "static-c4", "static-c6"]      # 지연 오름차순
LANGS = ["de", "ja", "zh"]
LANG_NAME = {"de": "English→German", "ja": "English→Japanese", "zh": "English→Chinese"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def load(root: Path):
    lat, qual, extra = {}, {"BLEU": {}, "COMET": {}}, {}
    comet = {}
    for tag in {t for t, _ in TAGS.values()}:
        f = root / f"comet_{tag}.json"
        if f.exists():
            for ax, d in json.loads(f.read_text())["system"].items():
                for lg, v in d.items():
                    comet[(ax, lg)] = v
    for ax, (tag, _) in TAGS.items():
        for lg in LANGS:
            s = json.loads((root / ax / f"n3000-{lg}" / tag / "metric.json")
                           .read_text())["summary"]
            lat[(ax, lg)] = s["laal_ms"] / 1000.0
            qual["BLEU"][(ax, lg)] = s["bleu"]
            qual["COMET"][(ax, lg)] = comet.get((ax, lg))
            extra[(ax, lg)] = {"ftl": s["first_token_latency_ms"],
                               "laal_ca": s["laal_ca_ms"],
                               "commits": s["mean_segments_per_utt"]}
    return lat, qual, extra


def interp_on_curve(xs, ys, x):
    """실측 static 곡선 위 x 지점의 값. 구간 선형 보간."""
    return float(np.interp(x, xs, ys))


# ── 그림 1: 지연–품질, static 은 실측 곡선 ──────────────────────────────────
def fig_tradeoff(lat, qual, out_stem: Path):
    fig, axs = plt.subplots(2, 3, figsize=(11.8, 6.9))
    for row, metric in enumerate(["BLEU", "COMET"]):
        for col, lg in enumerate(LANGS):
            ax = axs[row, col]
            style(ax)
            cx = [lat[(a, lg)] for a in STATIC_CURVE]
            cy = [qual[metric][(a, lg)] for a in STATIC_CURVE]
            ax.plot(cx, cy, color=C_STATIC, linewidth=2, zorder=3,
                    marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2)
            # static 점 라벨은 선 **위**, seg 라벨은 아래로 갈라둔다. 둘 다 아래에
            # 두면 seg 가 static 4s 바로 옆이라 ja/zh 에서 '4 s seg' 로 겹친다(실측).
            for a, x, y in zip(STATIC_CURVE, cx, cy):
                ax.text(x, y + (max(cy) - min(cy)) * 0.055,
                        TAGS[a][1].replace("static ", ""),
                        color=C_STATIC, fontsize=7.5, ha="center", va="bottom")

            sx, sy = lat[("seg", lg)], qual[metric][("seg", lg)]
            px, py = lat[("punct", lg)], qual[metric][("punct", lg)]
            on = interp_on_curve(cx, cy, sx)

            # seg 가 실측 곡선에서 얼마나 떨어져 있는지 (아래면 음수)
            ax.annotate("", xy=(sx, sy), xytext=(sx, on),
                        arrowprops=dict(arrowstyle="-|>", color=C_SEG,
                                        linewidth=1.4, shrinkA=0, shrinkB=2))
            gap = sy - on
            gtxt = f"{gap:+.2f}" if metric == "BLEU" else f"{gap:+.3f}"
            ax.text(sx + 0.16, (sy + on) / 2, gtxt, color=C_SEG, fontsize=8.5,
                    fontweight="bold", va="center", ha="left")

            for x, y, c, lab, dy in [(sx, sy, C_SEG, "seg", -0.125),
                                     (px, py, C_PUNCT, "punct", 0.075)]:
                ax.plot(x, y, marker="o", markersize=9, color=c,
                        markeredgecolor=SURFACE, markeredgewidth=2,
                        linestyle="none", zorder=5)
                span = max(cy + [sy, py]) - min(cy + [sy, py])
                ax.text(x, y + dy * span, lab, color=INK_2, fontsize=8.5,
                        ha="center", va="center")

            allv = cy + [sy, py]
            span = max(allv) - min(allv)
            ax.set_xlim(0.8, 4.6)
            ax.set_ylim(min(allv) - span * 0.26, max(allv) + span * 0.22)
            if row == 0:
                ax.set_title(LANG_NAME[lg], fontsize=10, pad=8)
            if row == 1:
                ax.set_xlabel("LAAL (s)  →  higher latency", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{metric}  →  better quality", fontsize=9)

    handles = [
        plt.Line2D([], [], color=C_STATIC, linewidth=2, marker="o", markersize=7,
                   markeredgecolor=SURFACE, markeredgewidth=1.5,
                   label="static, measured curve (2 / 4 / 6 s chunk)"),
        plt.Line2D([], [], marker="o", markersize=8, linestyle="none", color=C_SEG,
                   markeredgecolor=SURFACE, markeredgewidth=1.5, label="seg (ours)"),
        plt.Line2D([], [], marker="o", markersize=8, linestyle="none", color=C_PUNCT,
                   markeredgecolor=SURFACE, markeredgewidth=1.5,
                   label="punctuation (≈ offline bound)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2)
    fig.suptitle("Latency–quality trade-off on CoVoST2 (3,000 utterances per pair)",
                 fontsize=11.5, color=INK, y=0.985)
    fig.text(0.5, 0.925,
             "Same ASR weights throughout; only the commit policy differs. "
             "Against the measured static curve, seg sits BELOW it in all three languages.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.915))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 그림 2: 커밋 빈도 ↔ ASR, 그리고 얼마나 오프라인인가 ─────────────────────
def fig_regime(wer, extra, d1t, out_stem: Path):
    axes_all = ["static", "static-c4", "seg", "static-c6", "punct"]
    color = {"static": C_STATIC, "static-c4": C_STATIC, "static-c6": C_STATIC,
             "seg": C_SEG, "punct": C_PUNCT}
    fig, (ax2, ax) = plt.subplots(1, 2, figsize=(10.8, 4.3))

    style(ax2)
    ys2 = [d1t[a] * 100 for a in axes_all]
    ax2.barh(range(len(axes_all)), ys2, height=0.55,
             color=[color[a] for a in axes_all], zorder=3)
    ax2.axvline(100, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax2.text(99, len(axes_all) - 0.42, "utterance ends (= offline)", color=MUTED,
             fontsize=8, ha="right", va="bottom")
    for i, v in enumerate(ys2):
        ax2.text(v - 2.5, i, f"{v:.0f}%", color=SURFACE, fontsize=9,
                 fontweight="bold", ha="right", va="center", zorder=5)
    ax2.set_yticks(range(len(axes_all)))
    ax2.set_yticklabels([TAGS[a][1] for a in axes_all], fontsize=9, color=INK_2)
    ax2.set_ylim(-0.6, len(axes_all) - 0.15)
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("first commit, as % of utterance duration", fontsize=9)
    ax2.set_title("how much of the utterance each policy waits for",
                  fontsize=10.5, pad=10)
    ax2.grid(False, axis="y")

    style(ax)
    xs = [extra[(a, "de")]["commits"] for a in axes_all]
    ys = [wer[a] for a in axes_all]
    o = np.argsort(xs)
    ax.plot(np.array(xs)[o], np.array(ys)[o], color=AXIS, linewidth=1.6, zorder=2)
    for a, x, y in zip(axes_all, xs, ys):
        ax.plot(x, y, marker="o", markersize=9, color=color[a],
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=4)
        ax.text(x, y + 1.3, f"{TAGS[a][1]}\n{y:.2f}%", color=INK_2, fontsize=7.5,
                ha="center", va="bottom", linespacing=1.35)
    ax.set_xlabel("commits per utterance", fontsize=9)
    ax.set_ylabel("ASR WER (%)  →  worse", fontsize=9)
    ax.set_xlim(0.85, 2.8)
    ax.set_ylim(0, max(ys) + 9)
    ax.set_title("frequent commits freeze errors into the transcript",
                 fontsize=10.5, pad=10)
    fig.text(0.5, -0.045,
             "static 4 s and seg land on the same ASR quality (13.78% vs 13.80% WER), "
             "which makes their comparison a clean test of segmentation alone.\n"
             "Punctuation commits at 99% of the utterance — on single-clip CoVoST2 it is "
             "effectively offline, an upper bound rather than a streaming baseline.",
             ha="center", fontsize=8, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="CoVoST2 축 비교 그림")
    p.add_argument("--results-root", default=str(HERE / "results" / "CoVoST2"))
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    root = Path(a.results_root).expanduser().resolve()
    out = Path(a.out_dir) if a.out_dir else root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    lat, qual, extra = load(root)

    import jiwer
    def clean(t):
        return re.sub(r"\s+", " ",
                      re.sub(r"[^\w\s']", " ", (t or "").replace("<SEG>", " "))).strip().lower()
    wer, d1t = {}, {}
    for ax_name, (tag, _) in TAGS.items():
        rows = json.loads((root / ax_name / "n3000-de" / tag / "metric.json")
                          .read_text())["rows"]
        pr = [(clean(r["asr_text"]), clean(r["src_text"])) for r in rows]
        pr = [(h, f) for h, f in pr if f]
        wer[ax_name] = jiwer.wer([f for _, f in pr], [h for h, _ in pr]) * 100
        vals = []
        for r in rows:
            segs = sorted([s for s in r["segments"] if s["translation"]],
                          key=lambda s: s["segment_id"] or 0)
            if not segs:
                continue
            T = r["src_duration_ms"] / 1000
            vals.append(min(segs[0]["decision_audio_sec"], T) / T)
        d1t[ax_name] = float(np.mean(vals))

    fig_tradeoff(lat, qual, out / "tradeoff_covost2")
    fig_regime(wer, extra, d1t, out / "regime_covost2")
    print(f"저장: {out}")
    for f in sorted(out.glob("*covost2*")):
        print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
