#!/usr/bin/env python3
"""ACL 60/60 장문 지연–품질 그림. 저장된 streamlaal_*.json / comet_*.json 만 읽는다.

    .venv/bin/python evaluation/ast/plot_acl6060.py

라벨은 영문이다 — 논문 그림에 그대로 들어가고 CJK 폰트 의존을 없애기 위해서.
색·팔레트는 `plot_covost2.py` 와 같다(dataviz 기준 categorical 1~3번). static 계열은
한 계열이라 파랑을 공유하고 seg/punct 가 각각 주황/청록이다.

**static 은 점이 아니라 곡선이다.** 청크를 6/10/12초로 훑어 실측했다. seg 지연
(5.1~6.1초)이 c10 과 c12 사이에 들어오므로 보간으로 "같은 지연이면 static 은 얼마인가"
를 직접 말할 수 있다. 화살표가 그 차이다.

지연은 StreamLAAL(NCA) — 커밋을 결정한 순간까지 읽은 소스 오디오. 계산 시간은 빠져
있으므로 GPU 가 달라도 재현된다. COMET 은 재분절된 문장 단위 wmt22-comet-da.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
C_STATIC, C_SEG, C_PUNCT = "#2a78d6", "#eb6834", "#1baf7a"

# 축 → (태그, 표시 이름). 축마다 런이 다르므로 태그도 다르다.
TAGS = {
    "static-c6":  ("20260830_180201", "6 s"),
    "static-c10": ("20260831_102723", "10 s"),
    "static-c12": ("20260830_183928", "12 s"),
    "seg":        ("20260830_172429", "seg (ours)"),
    "punct":      ("20260831_013045", "punctuation"),
}
STATIC_CURVE = ["static-c6", "static-c10", "static-c12"]   # 지연 오름차순
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


def load(root: Path, split: str):
    """지연은 streamlaal_{split}_{tag}.json, 품질은 comet_{split}_*.json 에서 읽는다."""
    lat, comet = {}, {}
    for f in root.glob(f"comet_{split}_*.json"):
        for axis, d in json.loads(f.read_text())["system"].items():
            for lg, v in d.items():
                comet[(axis, lg)] = v
    for axis, (tag, _) in TAGS.items():
        f = root / f"streamlaal_{split}_{tag}.json"
        if not f.exists():
            raise SystemExit(f"!! 없음: {f}  — score_acl6060.py 를 먼저 돌릴 것")
        for r in json.loads(f.read_text())["results"]:
            if r["axis"] == axis:
                lat[(axis, r["lang"])] = r["stream_laal_sec"]
    missing = [k for k in lat if k not in comet]
    if missing:
        raise SystemExit(f"!! COMET 없음: {missing}  — comet_acl6060.py 를 먼저 돌릴 것")
    return lat, comet


def main() -> int:
    p = argparse.ArgumentParser(description="ACL 60/60 지연–품질 그림")
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    p.add_argument("--split", default="dev")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve()
    out = Path(a.out_dir) if a.out_dir else root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    lat, comet = load(root, a.split)

    fig, axs = plt.subplots(1, 3, figsize=(11.8, 3.9))
    for col, lg in enumerate(LANGS):
        ax = axs[col]
        style(ax)
        cx = [lat[(k, lg)] for k in STATIC_CURVE]
        cy = [comet[(k, lg)] for k in STATIC_CURVE]
        ax.plot(cx, cy, color=C_STATIC, linewidth=2, zorder=3, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label="fixed chunk")

        sx, sy = lat[("seg", lg)], comet[("seg", lg)]
        px, py = lat[("punct", lg)], comet[("punct", lg)]
        span = max(cy + [sy, py]) - min(cy + [sy, py])

        for k, x, y in zip(STATIC_CURVE, cx, cy):
            ax.text(x, y - span * 0.075, TAGS[k][1], color=C_STATIC,
                    fontsize=7.5, ha="center", va="top")

        # seg 가 실측 static 곡선에서 얼마나 떨어져 있는지. 위면 양수.
        on = float(np.interp(sx, cx, cy))
        ax.annotate("", xy=(sx, sy), xytext=(sx, on),
                    arrowprops=dict(arrowstyle="-|>", color=C_SEG,
                                    linewidth=1.4, shrinkA=0, shrinkB=2))
        ax.text(sx + 0.18, (sy + on) / 2, f"{sy - on:+.3f}", color=C_SEG,
                fontsize=8.5, fontweight="bold", va="center", ha="left")

        for x, y, c, lab, dy in [(sx, sy, C_SEG, "seg", 0.085),
                                 (px, py, C_PUNCT, "punct", 0.085)]:
            ax.plot(x, y, marker="o", markersize=9, color=c, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            ax.text(x, y + dy * span, lab, color=INK_2, fontsize=8.5,
                    ha="center", va="bottom")

        ax.set_title(LANG_NAME[lg], fontsize=10, pad=8)
        ax.set_xlabel("StreamLAAL (s, lower is better)")
        if col == 0:
            ax.set_ylabel("COMET (higher is better)")
        ax.set_xlim(min(cx + [sx, px]) - 0.9, max(cx + [sx, px]) + 0.9)
        ax.set_ylim(min(cy + [sy, py]) - span * 0.22,
                    max(cy + [sy, py]) + span * 0.22)

    fig.suptitle("ACL 60/60 dev — latency vs quality, en→{de,ja,zh}  "
                 "(5 talks, 57.6 min, 468 reference sentences)",
                 fontsize=10.5, color=INK, y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tradeoff_acl6060_{a.split}.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"저장: {out / f'tradeoff_acl6060_{a.split}.png'}")
    print(f"      {out / f'tradeoff_acl6060_{a.split}.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
