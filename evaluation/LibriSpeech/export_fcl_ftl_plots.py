"""
FCL / FTL 시각화 → PNG 저장 스크립트

사용법:
  # 전체 집계 플롯만
  python export_fcl_ftl_plots.py --json results/qwen3_test_other_full_fcl_v1.json --aggregate-only

  # 특정 파일 1개
  python export_fcl_ftl_plots.py --json results/qwen3_test_other_full_fcl_v1.json \
      --audio-root /path/to/LibriSpeech/test-other \
      --file-id 1688-142285-0000

  # 처음 N개 파일 (랜덤 샘플 원하면 --sample N)
  python export_fcl_ftl_plots.py --json results/qwen3_test_other_full_fcl_v1.json \
      --audio-root /path/to/LibriSpeech/test-other \
      --max-files 50 --out-dir output_plots

  # 전체 파일 (느림 주의)
  python export_fcl_ftl_plots.py --json results/qwen3_test_other_full_fcl_v1.json \
      --audio-root /path/to/LibriSpeech/test-other \
      --all-files
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")  # GUI 없는 서버 환경
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── 한글/CJK 폰트 자동 설정 ──────────────────────────────────────────────────
def _find_cjk_font() -> str | None:
    """시스템에서 CJK 지원 폰트를 찾아 경로 반환"""
    candidates = [
        # Ubuntu / Debian
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # matplotlib 폰트 캐시에서 검색
    for f in fm.findSystemFonts():
        lower = f.lower()
        if any(k in lower for k in ("noto", "cjk", "gothic", "gulim", "malgun", "nanum")):
            return f
    return None

_cjk_font = _find_cjk_font()
if _cjk_font:
    fm.fontManager.addfont(_cjk_font)
    _prop = fm.FontProperties(fname=_cjk_font)
    plt.rcParams["font.family"] = _prop.get_name()
else:
    import warnings
    warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

# ──────────────────────────────────────────────────────────────────────────────
# 색상 / 스타일
# ──────────────────────────────────────────────────────────────────────────────

STYLE = {
    "bg":        "#1E1E1E",
    "fg":        "#DDDDDD",
    "grid":      "#333333",
    "speech":    "#4C9BE8",   # 오디오 구간
    "commit_delay":   "#F0A500",   # audio_end → translate_start
    "model":     "#E84C4C",   # translate_start → translate_done (FCL)
    "ftl":       "#9B59B6",   # FTL 수직선
    "fcl_dot":   "#2ECC71",   # FCL 마커 점
    "seg":       "#E84C4C",   # commit=seg 색
    "vad":       "#F0A500",   # commit=vad 색
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["bg"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["fg"],
    "axes.titlecolor":   STYLE["fg"],
    "xtick.color":       STYLE["fg"],
    "ytick.color":       STYLE["fg"],
    "text.color":        STYLE["fg"],
    "grid.color":        STYLE["grid"],
    "legend.facecolor":  "#2D2D2D",
    "legend.edgecolor":  STYLE["grid"],
    "legend.labelcolor": STYLE["fg"],
    "font.size":         9,
})

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> tuple[list, dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "raw_results" in raw:
        return raw["raw_results"], raw.get("overall", {})
    first = next(iter(raw.values()))
    return first.get("raw_results", []), first.get("overall", {})


def find_audio(audio_path: str, audio_root: str | None) -> str | None:
    if os.path.exists(audio_path):
        return audio_path
    if audio_root:
        filename = os.path.basename(audio_path)
        parts = filename.replace(".flac", "").split("-")
        if len(parts) == 3:
            candidate = os.path.join(audio_root, parts[0], parts[1], filename)
            if os.path.exists(candidate):
                return candidate
        # fallback: 전체 탐색 (느림)
        for dirpath, _, files in os.walk(audio_root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None


def build_df(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        for seg in r.get("segment_metrics", []):
            # partial_tail 등 타이밍 데이터 없는 세그먼트 제외
            if seg.get("audio_start_sec") is None or seg.get("server_fcl_sec") is None:
                continue
            rows.append({
                "file_id":        r["file_id"],
                "speaker_id":     r["speaker_id"],
                "file_duration":  r["duration"],
                "file_ftl":       r["first_token_latency"],
                "segment_id":     seg["segment_id"],
                "commit_reason":  seg["commit_reason"],
                "audio_start":    seg["audio_start_sec"],
                "audio_end":      seg["audio_end_sec"],
                "seg_duration":   seg["audio_end_sec"] - seg["audio_start_sec"],
                "translate_started": seg["server_translate_started_elapsed_sec"],
                "translate_done":    seg["server_translate_done_elapsed_sec"],
                "server_fcl_sec":    seg["server_fcl_sec"],
                "translation_latency_sec": seg["translation_latency_sec"],
                "text":           seg.get("text", ""),
                "translation":    seg.get("translation", ""),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file 플롯
# ──────────────────────────────────────────────────────────────────────────────

def plot_file(record: dict, audio_root: str | None, out_path: str) -> None:
    # partial_tail (타이밍 None) 제외
    segs      = [s for s in record["segment_metrics"] if s.get("audio_start_sec") is not None]
    duration  = record["duration"]
    ftl       = record["first_token_latency"]
    file_id   = record["file_id"]

    audio_path = find_audio(record["audio_path"], audio_root)
    has_audio  = audio_path is not None

    # 행 구성: waveform(있으면) + gantt + 텍스트 표
    n_rows   = 3 if has_audio else 2
    heights  = [2, 3, 1.5] if has_audio else [3, 1.5]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 4 + 1.2 * len(segs)),
        gridspec_kw={"height_ratios": heights},
        sharex=False,
    )
    if n_rows == 2:
        axes = list(axes)
        ax_gantt, ax_text = axes
        ax_wave = None
    else:
        ax_wave, ax_gantt, ax_text = axes

    fig.suptitle(
        f"FCL / FTL Timeline  ─  {file_id}  "
        f"(duration={duration:.1f}s  FTL={ftl:.3f}s)",
        fontsize=11, y=0.99,
    )

    # ── Row 1: Waveform or Speech Activity ────────────────────────────────
    if has_audio and ax_wave is not None:
        y_wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        t = np.linspace(0, len(y_wav) / sr, num=len(y_wav))
        step = max(1, len(y_wav) // 8000)
        ax_wave.plot(t[::step], y_wav[::step], color=STYLE["speech"], lw=0.6)
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_xlim(0, max(duration, ftl) * 1.02)
        ax_wave.grid(True, axis="x", lw=0.4)
        ax_wave.set_title("Waveform  (VAD segments shaded)", fontsize=9, pad=3)
        # VAD segment 음영
        for seg in segs:
            ax_wave.axvspan(seg["audio_start_sec"], seg["audio_end_sec"],
                            alpha=0.15, color=STYLE["speech"])
        # FTL 수직선
        ax_wave.axvline(ftl, color=STYLE["ftl"], lw=1.2, ls="--", alpha=0.8)
    elif ax_wave is None:
        # Speech activity step function은 gantt 위에 배치 (별도 ax 없음, 생략)
        pass

    # ── Row 2: Gantt ────────────────────────────────────────────────────────
    ax = ax_gantt
    bar_height = 0.55
    x_max = max(duration, ftl, max(s["translate_done_sec"] if "translate_done_sec" in s
                                    else s["server_translate_done_elapsed_sec"]
                                    for s in segs)) * 1.03

    legend_handles = {}

    for seg in segs:
        s_start = seg["audio_start_sec"]
        s_end   = seg["audio_end_sec"]
        ts      = seg["server_translate_started_elapsed_sec"]
        td      = seg["server_translate_done_elapsed_sec"]
        y_pos   = seg["segment_id"]
        reason  = seg["commit_reason"]

        # 오디오 구간
        bar = ax.barh(y_pos, s_end - s_start, left=s_start,
                      height=bar_height, color=STYLE["speech"], alpha=0.9, zorder=3)
        if "audio" not in legend_handles:
            legend_handles["audio"] = mpatches.Patch(color=STYLE["speech"], label="Audio segment")

        # VAD/Commit 대기 (audio_end → translate_started)
        if ts > s_end + 0.001:
            ax.barh(y_pos, ts - s_end, left=s_end,
                    height=bar_height * 0.6, color=STYLE["commit_delay"], alpha=0.85, zorder=3)
            if "commit_delay" not in legend_handles:
                legend_handles["commit_delay"] = mpatches.Patch(color=STYLE["commit_delay"], label="commit delay")

        # 모델 실행 (translate_started → translate_done)
        ax.barh(y_pos, td - ts, left=ts,
                height=bar_height * 0.6, color=STYLE["model"], alpha=0.85, zorder=3)
        if "model" not in legend_handles:
            legend_handles["model"] = mpatches.Patch(color=STYLE["model"], label="Model run (->FCL)")

        # FCL 점 & 레이블
        ax.scatter(td, y_pos, color=STYLE["fcl_dot"], s=60, zorder=5,
                   marker="D")
        ax.text(td + 0.05, y_pos + 0.28,
                f"FCL {seg['server_fcl_sec']:.3f}s",
                fontsize=7.5, color=STYLE["fcl_dot"], va="bottom")
        if "fcl" not in legend_handles:
            legend_handles["fcl"] = mpatches.Patch(color=STYLE["fcl_dot"], label="FCL point")

        # commit_reason 레이블 (왼쪽)
        r_color = STYLE["vad"] if reason == "vad" else STYLE["seg"]
        ax.text(s_start - 0.05, y_pos, reason,
                fontsize=7, color=r_color, ha="right", va="center")

    # FTL 수직선
    ax.axvline(ftl, color=STYLE["ftl"], lw=1.5, ls="--", zorder=6)
    ax.text(ftl + 0.05, len(segs) + 0.6,
            f"FTL {ftl:.3f}s", fontsize=8.5, color=STYLE["ftl"])
    legend_handles["ftl"] = mpatches.Patch(color=STYLE["ftl"], label=f"FTL ({ftl:.3f}s)")

    ax.set_yticks(range(1, len(segs) + 1))
    ax.set_yticklabels([f"seg {s['segment_id']}" for s in segs], fontsize=8)
    ax.set_xlim(-0.3, x_max)
    ax.set_ylim(0.3, len(segs) + 0.9)
    ax.set_xlabel("Time (s)")
    ax.set_title("FCL / FTL Gantt Timeline", fontsize=9, pad=3)
    ax.grid(True, axis="x", lw=0.4)
    ax.legend(handles=list(legend_handles.values()),
              loc="lower right", fontsize=8, ncol=3)

    # speech activity 음영 (waveform 없을 때 gantt 배경에 표시)
    if not has_audio:
        for seg in segs:
            ax.axvspan(seg["audio_start_sec"], seg["audio_end_sec"],
                       ymin=0, ymax=1, alpha=0.06, color=STYLE["speech"], zorder=1)

    # ── Row 3: 텍스트 표 ─────────────────────────────────────────────────────
    ax_text.axis("off")
    col_labels = ["Seg", "Commit", "Audio (s)", "EN Text", "KO Translation", "FCL (s)"]
    rows_data  = [
        [
            str(s["segment_id"]),
            s["commit_reason"],
            f"{s['audio_start_sec']:.2f}–{s['audio_end_sec']:.2f}",
            _truncate(s.get("text", ""), 40),
            _truncate(s.get("translation", ""), 30),
            f"{s['server_fcl_sec']:.3f}",
        ]
        for s in segs
    ]
    col_widths = [0.05, 0.07, 0.12, 0.38, 0.28, 0.10]
    tbl = ax_text.table(
        cellText=rows_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#2D2D2D" if row % 2 == 0 else "#252525")
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["fg"])
        if row == 0:
            cell.set_facecolor("#3A3A3A")
            cell.set_text_props(color="white", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ──────────────────────────────────────────────────────────────────────────────
# 집계 플롯
# ──────────────────────────────────────────────────────────────────────────────

def plot_aggregate(df: pd.DataFrame, overall: dict, out_dir: str) -> None:
    out = Path(out_dir)

    # ── 1. FCL 분포 (commit_reason별) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(0, df["server_fcl_sec"].quantile(0.995) + 0.1, 0.05)
    for reason, color, alpha in [("vad", STYLE["commit_delay"], 0.75),
                                  ("seg", STYLE["seg"], 0.65)]:
        sub = df[df["commit_reason"] == reason]["server_fcl_sec"]
        ax.hist(sub, bins=bins, color=color, alpha=alpha,
                label=f"commit={reason}  (n={len(sub):,})", edgecolor="none")
    avg = df["server_fcl_sec"].mean()
    ax.axvline(avg, color="white", lw=1.2, ls="--")
    ax.text(avg + 0.01, ax.get_ylim()[1] * 0.92,
            f"mean {avg:.3f}s", color="white", fontsize=8.5)
    ax.set_xlabel("server_fcl_sec (s)")
    ax.set_ylabel("Segment count")
    ax.set_title("FCL Distribution by commit_reason")
    ax.legend()
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "aggregate_fcl_distribution.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  saved: aggregate_fcl_distribution.png")

    # ── 2. FTL 분포 (per-file) ────────────────────────────────────────────────
    ftl_df = df.drop_duplicates("file_id")["file_ftl"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(ftl_df, bins=40, color=STYLE["ftl"], alpha=0.8, edgecolor="none")
    ax.axvline(ftl_df.mean(), color="white", lw=1.2, ls="--")
    ax.text(ftl_df.mean() + 0.1, ax.get_ylim()[1] * 0.92,
            f"mean {ftl_df.mean():.3f}s", color="white", fontsize=8.5)
    ax.set_xlabel("first_token_latency (s)")
    ax.set_ylabel("File count")
    ax.set_title("FTL Distribution (per-file)")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "aggregate_ftl_distribution.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  saved: aggregate_ftl_distribution.png")

    # ── 3. FCL vs Segment Duration scatter ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for reason, color in [("vad", STYLE["commit_delay"]), ("seg", STYLE["seg"])]:
        sub = df[df["commit_reason"] == reason]
        ax.scatter(sub["seg_duration"], sub["server_fcl_sec"],
                   color=color, alpha=0.3, s=8, label=f"commit={reason}")
    ax.set_xlabel("Segment duration (s)")
    ax.set_ylabel("server_fcl_sec (s)")
    ax.set_title("FCL vs Segment Duration")
    ax.legend()
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "aggregate_fcl_vs_seg_duration.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  saved: aggregate_fcl_vs_seg_duration.png")

    # ── 4. avg FCL vs FTL (per-file scatter) ─────────────────────────────────
    file_df = (
        df.groupby("file_id")
        .agg(avg_fcl=("server_fcl_sec", "mean"),
             ftl=("file_ftl", "first"),
             duration=("file_duration", "first"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(file_df["avg_fcl"], file_df["ftl"],
                    c=file_df["duration"], cmap="viridis",
                    alpha=0.6, s=12)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Audio Duration (s)", color=STYLE["fg"])
    cbar.ax.yaxis.set_tick_params(color=STYLE["fg"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=STYLE["fg"])
    ax.set_xlabel("avg server_fcl_sec (s)")
    ax.set_ylabel("first_token_latency (s)")
    ax.set_title("File avg FCL vs FTL  (color = audio duration)")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "aggregate_file_fcl_vs_ftl.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  saved: aggregate_file_fcl_vs_ftl.png")

    # ── 5. Speaker별 avg FCL 바 차트 ─────────────────────────────────────────
    spk_df = (
        df.groupby("speaker_id")["server_fcl_sec"]
        .mean()
        .sort_values()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = [STYLE["model"] if v > df["server_fcl_sec"].mean() else STYLE["speech"]
              for v in spk_df["server_fcl_sec"]]
    ax.bar(spk_df["speaker_id"], spk_df["server_fcl_sec"], color=colors, width=0.7)
    ax.axhline(df["server_fcl_sec"].mean(), color="white", lw=1, ls="--",
               label=f"overall mean {df['server_fcl_sec'].mean():.3f}s")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("avg server_fcl_sec (s)")
    ax.set_title("avg FCL by Speaker")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    ax.grid(True, axis="y", lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "aggregate_speaker_fcl.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  saved: aggregate_speaker_fcl.png")

    # ── 6. FCL CDF ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for reason, color in [("vad", STYLE["commit_delay"]), ("seg", STYLE["seg"]),
                           ("all", STYLE["speech"])]:
        sub = df[df["commit_reason"] == reason]["server_fcl_sec"] if reason != "all" \
              else df["server_fcl_sec"]
        x_sorted = np.sort(sub)
        cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        lbl = f"commit={reason}" if reason != "all" else "all"
        ax.plot(x_sorted, cdf * 100, color=color, lw=1.5, label=lbl)
    for pct, ls in [(50, "--"), (90, ":"), (95, "-.")]:
        q = df["server_fcl_sec"].quantile(pct / 100)
        ax.axvline(q, color="white", lw=0.8, ls=ls, alpha=0.6)
        ax.text(q + 0.01, pct - 3, f"P{pct}={q:.3f}s", fontsize=7.5, color="white")
    ax.set_xlabel("server_fcl_sec (s)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("FCL CDF (P50 / P90 / P95)")
    ax.legend()
    ax.grid(True, lw=0.4)
    ax.set_ylim(0, 102)
    fig.tight_layout()
    fig.savefig(out / "aggregate_fcl_cdf.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  saved: aggregate_fcl_cdf.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FCL/FTL PNG 시각화 저장")
    p.add_argument("--json",    required=True, help="결과 JSON 경로")
    p.add_argument("--audio-root", default=None,
                   help="LibriSpeech test-other 루트 (waveform 표시용, 없으면 Speech Activity)")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results" / "fcl" / "fcl_output_plots_v"), help="출력 디렉터리")
    p.add_argument("--file-id", default=None, help="특정 file_id 하나만 처리")
    p.add_argument("--max-files", type=int, default=None,
                   help="앞에서 N개 파일만 per-file 플롯 생성")
    p.add_argument("--sample", type=int, default=None,
                   help="랜덤 N개 파일 샘플링 (--max-files와 배타적)")
    p.add_argument("--all-files", action="store_true",
                   help="전체 파일 per-file 플롯 생성 (느림)")
    p.add_argument("--aggregate-only", action="store_true",
                   help="집계 플롯만 생성 (per-file 생략)")
    p.add_argument("--no-aggregate", action="store_true", default=True,
                   help="집계 플롯 생략 (기본값: True)")
    p.add_argument("--seed", type=int, default=42, help="샘플링 seed")
    return p.parse_args()


def main():
    args = parse_args()

    # 출력 디렉터리
    out_dir = Path(args.out_dir)
    per_file_dir = out_dir
    if not args.aggregate_only:
        per_file_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSON 로드 중: {args.json}")
    results, overall = load_json(args.json)
    print(f"  → {len(results):,}개 파일")

    # ── 집계 플롯 ─────────────────────────────────────────────────────────────
    if not args.no_aggregate and not args.file_id:
        print("\n[집계 플롯 생성]")
        df = build_df(results)
        plot_aggregate(df, overall, str(out_dir))

    if args.aggregate_only:
        print("\n완료!")
        return

    # ── Per-file 플롯 ─────────────────────────────────────────────────────────
    if args.file_id:
        targets = [r for r in results if r["file_id"] == args.file_id]
        if not targets:
            print(f"file_id '{args.file_id}' 를 찾을 수 없습니다.")
            sys.exit(1)
    elif args.all_files:
        targets = results
    elif args.sample:
        random.seed(args.seed)
        targets = random.sample(results, min(args.sample, len(results)))
    elif args.max_files:
        targets = results[: args.max_files]
    else:
        # 기본: 집계 플롯만 (per-file은 명시적 옵션 필요)
        print("\n--max-files, --sample, --all-files, --file-id 중 하나를 지정하면 per-file 플롯도 생성됩니다.")
        print("완료!")
        return

    print(f"\n[Per-file 플롯 생성]  대상 {len(targets)}개 → {per_file_dir}/")
    for record in tqdm(targets, unit="file"):
        out_path = per_file_dir / f"{record['file_id']}.png"
        try:
            plot_file(record, args.audio_root, str(out_path))
        except Exception as e:
            print(f"  ⚠ {record['file_id']}: {e}")

    print(f"\n완료!  {out_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
