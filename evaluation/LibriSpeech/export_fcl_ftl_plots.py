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
import torch
import matplotlib
matplotlib.use("Agg")  # GUI 없는 서버 환경
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
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
    "lookahead": "#7F8C8D",   # refined_end → server audio_end (look-ahead 오디오)
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


VAD_THRESHOLD        = 0.5
VAD_MIN_SILENCE_MS   = 800
VAD_SPEECH_PAD_MS    = 160
VAD_WINDOW_SAMPLES   = 512
VAD_SR               = 16000


def load_vad_model():
    """Silero VAD 모델 로드. 실패 시 None 반환."""
    try:
        from silero_vad import load_silero_vad, VADIterator  # noqa: F401
        model = load_silero_vad()
        return model
    except Exception as e:
        print(f"  [VAD] Silero VAD 로드 실패, RMS fallback 사용: {e}")
        return None


def _make_vad_iterator(model):
    from silero_vad import VADIterator
    return VADIterator(
        model=model,
        threshold=VAD_THRESHOLD,
        sampling_rate=VAD_SR,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )


def detect_speech_bounds(y_wav: np.ndarray, sr: int, start_sec: float, end_sec: float,
                         vad_model=None) -> tuple[float, float]:
    """Silero VAD로 세그먼트 내 실제 발화 시작·끝 시각을 추정.
    end_sec 이후 (VAD_MIN_SILENCE_MS + VAD_SPEECH_PAD_MS)만큼 더 포함해
    Silero가 end 이벤트를 발생시킬 수 있게 함.
    r_end는 end_sec로 clamp하여 look-ahead 막대가 정확히 표시됨.
    vad_model이 None이면 RMS 에너지 기반 fallback 사용."""
    i_start    = int(start_sec * sr)
    extend_sec = (VAD_MIN_SILENCE_MS + VAD_SPEECH_PAD_MS) / 1000.0
    i_end_ext  = min(int((end_sec + extend_sec) * sr), len(y_wav))
    segment    = y_wav[i_start:i_end_ext]
    if len(segment) == 0:
        return start_sec, end_sec

    if vad_model is not None:
        try:
            vad_iter = _make_vad_iterator(vad_model)
            speech_starts, speech_ends = [], []
            audio_float = segment.astype(np.float32)
            offset = 0
            in_speech = False
            seg_start_local = None
            while offset + VAD_WINDOW_SAMPLES <= len(audio_float):
                window = torch.from_numpy(audio_float[offset:offset + VAD_WINDOW_SAMPLES])
                result = vad_iter(window, return_seconds=False)
                if result is not None:
                    if "start" in result and not in_speech:
                        seg_start_local = start_sec + result["start"] / sr
                        in_speech = True
                    if "end" in result and in_speech:
                        speech_starts.append(seg_start_local)
                        speech_ends.append(start_sec + result["end"] / sr)
                        in_speech = False
                offset += VAD_WINDOW_SAMPLES
            if in_speech and seg_start_local is not None:
                # 확장 구간에서 시작된 speech(다음 세그먼트)는 제외
                if seg_start_local < end_sec:
                    speech_starts.append(seg_start_local)
                    speech_ends.append(end_sec)
            # end_sec 이전에 시작된 region만 유효
            valid = [(s, e) for s, e in zip(speech_starts, speech_ends) if s < end_sec]
            if valid:
                r_end = min(valid[-1][1], end_sec)  # end_sec 초과 불가
                return valid[0][0], r_end
            return start_sec, end_sec
        except Exception:
            pass  # fallback

    # RMS fallback
    frame_len = int(0.02 * sr)
    n_frames = len(segment) // frame_len
    if n_frames == 0:
        return start_sec, end_sec
    rms = np.array([
        np.sqrt(np.mean(segment[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    active = np.where(rms >= 0.01)[0]
    if len(active) == 0:
        return start_sec, end_sec
    return (start_sec + active[0] * 0.02,
            min(start_sec + (active[-1] + 1) * 0.02, end_sec))


# ──────────────────────────────────────────────────────────────────────────────
# Per-file 플롯
# ──────────────────────────────────────────────────────────────────────────────

def plot_file(record: dict, audio_root: str | None, out_path: str, vad_model=None) -> None:
    # partial_tail (타이밍 None) 제외
    segs      = [s for s in record["segment_metrics"] if s.get("audio_start_sec") is not None]
    if not segs:
        raise ValueError("no valid segments (record may be incomplete)")
    duration  = record["duration"]
    ftl       = record["first_token_latency"]  # may be None
    file_id   = record["file_id"]

    audio_path = find_audio(record["audio_path"], audio_root)
    has_audio  = audio_path is not None

    # 오디오 로드 (waveform + speech bounds 둘 다 사용)
    y_wav, sr_wav = (librosa.load(audio_path, sr=16000, mono=True) if has_audio
                     else (None, 16000))

    # 행 구성: waveform(있으면) + gantt + 텍스트 표
    n_rows   = 3 if has_audio else 2
    heights  = [2, 3, 1.5] if has_audio else [3, 1.5]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 4 + 1.2 * len(segs)),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    if n_rows == 2:
        axes = list(axes)
        ax_gantt, ax_text = axes
        ax_wave = None
    else:
        ax_wave, ax_gantt, ax_text = axes

    ftl_str = f"{ftl:.3f}s" if ftl is not None else "N/A"
    fig.suptitle(
        f"FCL / FTL Timeline  ─  {file_id}  "
        f"(duration={duration:.1f}s  FTL={ftl_str})",
        fontsize=11, y=0.99,
    )

    seg_td_vals = [s["translate_done_sec"] if "translate_done_sec" in s
                   else s["server_translate_done_elapsed_sec"] for s in segs]
    x_max = max([duration] + ([ftl] if ftl is not None else []) + seg_td_vals) * 1.03

    # ── Row 1: Waveform or Speech Activity ────────────────────────────────
    if has_audio and ax_wave is not None:
        t = np.linspace(0, len(y_wav) / sr_wav, num=len(y_wav))
        step = max(1, len(y_wav) // 8000)
        ax_wave.plot(t[::step], y_wav[::step], color=STYLE["speech"], lw=0.6)
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_xlim(0, x_max)
        ax_wave.tick_params(labelbottom=True)
        ax_wave.grid(True, axis="x", lw=0.4)
        ax_wave.set_title("Waveform  (VAD segments shaded)", fontsize=9, pad=3)
        # VAD segment 음영
        for seg in segs:
            ax_wave.axvspan(seg["audio_start_sec"], seg["audio_end_sec"],
                            alpha=0.15, color=STYLE["speech"])
        # FTL 수직선
        if ftl is not None:
            ax_wave.axvline(ftl, color=STYLE["ftl"], lw=1.2, ls="--", alpha=0.8)
    elif ax_wave is None:
        # Speech activity step function은 gantt 위에 배치 (별도 ax 없음, 생략)
        pass

    # ── Row 2: Gantt ────────────────────────────────────────────────────────
    ax = ax_gantt
    bar_height = 0.55

    legend_handles = {}

    prev_r_end = 0.0  # 이전 세그먼트의 실제 speech end (다음 세그먼트 검색 시작점)

    for seg in segs:
        s_end   = seg["audio_end_sec"]
        ts      = seg["server_translate_started_elapsed_sec"]
        td      = seg["server_translate_done_elapsed_sec"]
        y_pos   = seg["segment_id"]
        reason  = seg["commit_reason"]

        # 오디오 구간: s_start 대신 이전 세그먼트의 실제 speech end를 검색 시작점으로 사용
        if has_audio and y_wav is not None:
            r_start, r_end = detect_speech_bounds(y_wav, sr_wav, prev_r_end, s_end, vad_model=vad_model)
        else:
            r_start, r_end = prev_r_end, s_end
        prev_r_end = r_end

        ax.barh(y_pos, r_end - r_start, left=r_start,
                height=bar_height, color=STYLE["speech"], alpha=0.9, zorder=3)
        if "audio" not in legend_handles:
            legend_handles["audio"] = mpatches.Patch(color=STYLE["speech"], label="Audio segment (energy)")

        # look-ahead 구간 (refined_end → server audio_end_sec)
        if has_audio and y_wav is not None and s_end > r_end + 0.001:
            ax.barh(y_pos, s_end - r_end, left=r_end,
                    height=bar_height, color=STYLE["lookahead"], alpha=0.6, zorder=3)
            if "lookahead" not in legend_handles:
                legend_handles["lookahead"] = mpatches.Patch(color=STYLE["lookahead"], label="Look-ahead (speech→server commit)")

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
            legend_handles["model"] = mpatches.Patch(color=STYLE["model"], label="Translation API")

        # FCL 점 & 레이블
        ax.scatter(td, y_pos, color=STYLE["fcl_dot"], s=60, zorder=5,
                   marker="D")
        ax.text(td + 0.05, y_pos + 0.28,
                f"FCL {seg['server_fcl_sec']:.3f}s",
                fontsize=7.5, color=STYLE["fcl_dot"], va="bottom")
        if "fcl" not in legend_handles:
            legend_handles["fcl"] = mpatches.Patch(color=STYLE["fcl_dot"], label="FCL point")

        # commit_reason 레이블 (바 위쪽)
        r_color = STYLE["vad"] if reason == "vad" else STYLE["seg"]
        ax.text(r_start, y_pos + bar_height / 2 + 0.05, reason,
                fontsize=7, color=r_color, ha="left", va="bottom")

    # FTL 수직선
    if ftl is not None:
        ax.axvline(ftl, color=STYLE["ftl"], lw=1.5, ls="--", zorder=6)
        ax.text(ftl + 0.05, len(segs) + 0.6,
                f"FTL {ftl:.3f}s", fontsize=8.5, color=STYLE["ftl"])
        legend_handles["ftl"] = mpatches.Patch(color=STYLE["ftl"], label=f"FTL ({ftl:.3f}s)")

    ax.set_yticks(range(1, len(segs) + 1))
    ax.set_yticklabels([f"seg {s['segment_id']}" for s in segs], fontsize=8)
    ax.set_xlim(0, x_max)
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
    p.add_argument("--seed", type=int, default=42, help="샘플링 seed")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSON 로드 중: {args.json}")
    results, _ = load_json(args.json)
    print(f"  → {len(results):,}개 파일")

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
        print("\n--max-files, --sample, --all-files, --file-id 중 하나를 지정하세요.")
        sys.exit(1)

    vad_model = load_vad_model() if args.audio_root else None

    print(f"\n[Per-file 플롯 생성]  대상 {len(targets)}개 → {out_dir}/")
    for record in tqdm(targets, unit="file"):
        out_path = out_dir / f"{record['file_id']}.png"
        try:
            plot_file(record, args.audio_root, str(out_path), vad_model=vad_model)
        except Exception as e:
            print(f"  ⚠ {record['file_id']}: {e}")

    print(f"\n완료!  {out_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
