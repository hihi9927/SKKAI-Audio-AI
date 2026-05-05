"""
FSL / Layer 타이밍 시각화 → PNG 저장 스크립트

행 레이아웃 (위/아래 2존):
  위 (초록) : 오디오 intake — 청크별 음성 수신 구간 (encode_sec까지)
  아래 (주황): 디코드 — 청크별 model.generate() 시간 (chunk_decode_sec), 연속 수신 중 겹침 가능
  아래 (빨강): 번역   — SEG 감지 직후 번역 API 호출 (trans_sec)

FSL (SEG commit): SEG 감지 시점 ~ 번역 완료 시점 = fsl_sec
FSL (VAD/finish): final_decode_sec + trans_sec = fsl_sec

마커:
  ◆ (흰색) : SEG 감지 시점 (encode_sec + decode_sec) — 두 존 경계
  | (초록)  : audioEndSec — 추정 발화 끝 (SEG commit: partial 스냅샷 역산, VAD commit: VAD 기준)
  : (회색)  : seg_audio_sec — SEG 감지 당시 청크 경계

사용법:
  python export_fcl_ftl_plots.py --json /home/ubuntu/STiTy/evaluation/LibriSpeech/servers/results/fsl/test/test_other_fsl_test.json \\
      --audio-root /home/ubuntu/STiTy/evaluation/LibriSpeech/LibriSpeech/test-other \\
      --max-files 50 --out-dir output_plots

  python export_fcl_ftl_plots.py --json results/qwen3_test_other_fcl.json \\
      --file-id 1688-142285-0000
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np
from tqdm import tqdm

# ── 한글/CJK 폰트 자동 설정 ──────────────────────────────────────────────────
def _find_cjk_font() -> str | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
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

# ── 색상 / 스타일 ──────────────────────────────────────────────────────────────
STYLE = {
    "bg":           "#1E1E1E",
    "fg":           "#DDDDDD",
    "grid":         "#333333",
    # 레이어 색상
    "encode":       "#4CAF50",   # encode layer (green)
    "encode_tail":  "#81C784",   # encode layer 마지막 gap (light green)
    "decode":       "#FF9800",   # decode layer (orange, 오디오 수신 중)
    "decode_post":  "#E65100",   # decode layer post-est. SEG end (dark orange)
    "trans":        "#E84C4C",   # trans layer (red)
    "pre_trans":    "#888888",   # decode→trans 사이 처리 시간 (gray)
    "final_decode": "#9B59B6",   # VAD/finish commit 전용 final decode (purple)
    # 오디오 / 마커
    "waveform":     "#4C9BE8",
    "seg_span":     "#4C9BE8",   # 세그먼트 배경 음영
    "audio_end":    "#2ECC71",   # audioEndSec 마커 (추정 발화 끝, 초록)
    "seg_audio":    "#7F8C8D",   # seg_audio_sec 마커 (청크 경계, 회색)
    "seg_marker":   "#FFFFFF",   # SEG 감지 다이아몬드 (흰색)
    # commit reason
    "reason_seg":   "#E84C4C",
    "reason_vad":   "#F0A500",
    "reason_other": "#7F8C8D",
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

# ── 필드 추출 헬퍼 ────────────────────────────────────────────────────────────

def _f(seg: dict, *keys, default=None):
    """여러 키 중 첫 번째로 존재하는 값 반환."""
    for k in keys:
        v = seg.get(k)
        if v is not None:
            return v
    return default

def _audio_start(seg): return _f(seg, "audio_start_sec", "audioStartSec", default=0.0)
def _audio_end(seg):   return _f(seg, "audio_end_sec",   "audioEndSec",   default=0.0)
def _seg_id(seg):      return _f(seg, "segment_id",      "segmentId",     default=0)
def _reason(seg):      return _f(seg, "commit_reason",   "commitReason",  default="?")
def _has_new_format(seg): return "fsl_sec" in seg

# ── 데이터 로드 ───────────────────────────────────────────────────────────────

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
        for dirpath, _, files in os.walk(audio_root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None

# ── VAD (waveform용) ──────────────────────────────────────────────────────────

VAD_THRESHOLD      = 0.5
VAD_MIN_SILENCE_MS = 800
VAD_SPEECH_PAD_MS  = 160
VAD_WINDOW_SAMPLES = 512
VAD_SR             = 16000


def load_vad_model():
    try:
        from silero_vad import load_silero_vad
        return load_silero_vad()
    except Exception as e:
        print(f"  [VAD] Silero VAD 로드 실패, RMS fallback 사용: {e}")
        return None


def detect_speech_bounds(y_wav: np.ndarray, sr: int, start_sec: float, end_sec: float,
                         vad_model=None) -> tuple[float, float]:
    i_start   = int(start_sec * sr)
    extend_sec = (VAD_MIN_SILENCE_MS + VAD_SPEECH_PAD_MS) / 1000.0
    i_end_ext  = min(int((end_sec + extend_sec) * sr), len(y_wav))
    segment    = y_wav[i_start:i_end_ext]
    if len(segment) == 0:
        return start_sec, end_sec

    if vad_model is not None:
        try:
            from silero_vad import VADIterator
            vad_iter = VADIterator(vad_model, threshold=VAD_THRESHOLD,
                                   sampling_rate=VAD_SR,
                                   min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                                   speech_pad_ms=VAD_SPEECH_PAD_MS)
            audio_float = segment.astype(np.float32)
            speech_starts, speech_ends = [], []
            offset, in_speech, seg_start_local = 0, False, None
            while offset + VAD_WINDOW_SAMPLES <= len(audio_float):
                window = torch.from_numpy(audio_float[offset:offset + VAD_WINDOW_SAMPLES])
                result = vad_iter(window, return_seconds=False)
                if result:
                    if "start" in result and not in_speech:
                        seg_start_local = start_sec + result["start"] / sr
                        in_speech = True
                    if "end" in result and in_speech:
                        speech_starts.append(seg_start_local)
                        speech_ends.append(start_sec + result["end"] / sr)
                        in_speech = False
                offset += VAD_WINDOW_SAMPLES
            if in_speech and seg_start_local is not None and seg_start_local < end_sec:
                speech_starts.append(seg_start_local)
                speech_ends.append(end_sec)
            valid = [(s, e) for s, e in zip(speech_starts, speech_ends) if s < end_sec]
            if valid:
                return valid[0][0], min(valid[-1][1], end_sec)
        except Exception:
            pass

    # RMS fallback
    frame_len = int(0.02 * sr)
    n_frames  = len(segment) // frame_len
    if n_frames == 0:
        return start_sec, end_sec
    rms    = np.array([np.sqrt(np.mean(segment[i*frame_len:(i+1)*frame_len]**2))
                       for i in range(n_frames)])
    active = np.where(rms >= 0.01)[0]
    if len(active) == 0:
        return start_sec, end_sec
    return (start_sec + active[0] * 0.02,
            min(start_sec + (active[-1] + 1) * 0.02, end_sec))

# ── Per-file 플롯 ─────────────────────────────────────────────────────────────

def _reason_color(reason: str) -> str:
    if reason == "seg":   return STYLE["reason_seg"]
    if reason == "vad":   return STYLE["reason_vad"]
    return STYLE["reason_other"]


def _vlines_seg(ax, x: float, y: float, half_h: float, color: str, lw=1.5, ls="-", zorder=6):
    """단일 세그먼트 행에만 수직 마커."""
    ax.vlines(x, y - half_h, y + half_h, colors=color, lw=lw, linestyles=ls, zorder=zorder)


def plot_file(record: dict, audio_root: str | None, out_path: str, vad_model=None) -> None:
    segs = [s for s in record.get("segment_metrics", [])
            if s.get("audio_start_sec") is not None or s.get("audioStartSec") is not None]
    if not segs:
        raise ValueError("유효한 세그먼트 없음")

    # 새 포맷 여부 확인
    new_fmt = any(_has_new_format(s) for s in segs)

    duration  = record.get("duration", 0)
    file_id   = record.get("file_id", "unknown")
    audio_path = find_audio(record.get("audio_path", ""), audio_root)
    has_audio  = audio_path is not None
    y_wav, sr_wav = (librosa.load(audio_path, sr=16000, mono=True)
                     if has_audio else (None, 16000))

    n_segs  = len(segs)
    n_rows  = 3 if has_audio else 2
    heights = [1.8, 3.5 + 0.4 * n_segs, 1.5] if has_audio else [3.5 + 0.4 * n_segs, 1.5]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(15, sum(heights) + 1),
        gridspec_kw={"height_ratios": heights},
        sharex=False,  # sharex 비활성 — gantt가 중간에 있을 때 tick label 숨김 방지
    )
    if n_rows == 2:
        ax_gantt, ax_text = axes
        ax_wave = None
    else:
        ax_wave, ax_gantt, ax_text = axes

    # x 범위 계산
    # encode bar는 audio-time 기준(audio_end까지), decode+trans는 wall-clock(seg_det_x + fsl)
    x_vals = [duration]
    for s in segs:
        enc = s.get("encode_sec")
        dec = s.get("decode_sec") or 0
        fsl_v = s.get("fsl_sec") or 0
        a_end_s = _audio_end(s) or 0
        if enc is not None:
            seg_det = enc + dec
            right = seg_det + fsl_v if fsl_v else seg_det + (s.get("trans_sec", 0) or 0)
            x_vals.append(max(a_end_s, right))  # audio bar와 decode bar 중 더 넓은 쪽
        elif a_end_s:
            fd = s.get("final_decode_sec", 0) or 0
            ts = s.get("trans_sec", 0) or 0
            x_vals.append(a_end_s + fd + ts)
    x_max = max(x_vals) * 1.06

    fig.suptitle(
        f"레이어 타이밍  ─  {file_id}  (duration={duration:.1f}s)",
        fontsize=11, y=0.995,
    )

    # ── Row 1: Waveform ──────────────────────────────────────────────────────
    if has_audio and ax_wave is not None and y_wav is not None:
        t    = np.linspace(0, len(y_wav) / sr_wav, num=len(y_wav))
        step = max(1, len(y_wav) // 8000)
        ax_wave.plot(t[::step], y_wav[::step], color=STYLE["waveform"], lw=0.5)
        ax_wave.set_ylabel("Amp", fontsize=8)
        ax_wave.set_xlim(0, x_max)
        ax_wave.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax_wave.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
        ax_wave.tick_params(axis="x", which="major", labelsize=7)
        ax_wave.tick_params(axis="x", which="minor", length=2)
        ax_wave.grid(True, axis="x", which="major", lw=0.4)
        ax_wave.grid(True, axis="x", which="minor", lw=0.15, alpha=0.4)
        ax_wave.set_title("Waveform", fontsize=8, pad=2)

        prev_end = 0.0
        for s in segs:
            a_start = _audio_start(s)
            a_end   = _audio_end(s)
            ax_wave.axvspan(a_start, a_end, alpha=0.12, color=STYLE["seg_span"])
            audio_end_marker = a_end
            seg_audio_marker = s.get("seg_audio_sec")
            if audio_end_marker:
                ax_wave.axvline(audio_end_marker, color=STYLE["audio_end"],
                                lw=0.9, ls="--", alpha=0.7)
            if seg_audio_marker:
                ax_wave.axvline(seg_audio_marker, color=STYLE["seg_audio"],
                                lw=0.9, ls=":", alpha=0.7)
            prev_end = a_end

    # ── Row 2: Layer Gantt ───────────────────────────────────────────────────
    ax  = ax_gantt
    bh  = 0.5   # row half-height (±0.25 from y-center)

    # 행을 위/아래 두 존으로 분리: 위=오디오(초록), 아래=디코드/번역(주황/빨강)
    bh_enc  = bh * 0.44           # 오디오 intake bar 높이
    y_enc   = lambda y: y + bh * 0.26  # 오디오 바 y-center (위쪽 존)
    bh_dec  = bh * 0.44           # 디코드/번역 바 높이
    y_dec   = lambda y: y - bh * 0.32  # 디코드 바 y-center (아래쪽 존)

    legend_handles = {}
    prev_end = 0.0

    for i, seg in enumerate(segs):
        y      = n_segs - i   # seg[0] → top (y=n_segs), seg[-1] → bottom (y=1)
        a_start = _audio_start(seg)
        a_end   = _audio_end(seg)
        reason  = _reason(seg)
        ye      = y_enc(y)
        yd      = y_dec(y)

        if new_fmt and _has_new_format(seg):
            fsl        = seg.get("fsl_sec")
            encode_sec = seg.get("encode_sec")
            decode_sec = seg.get("decode_sec") or 0
            trans_sec  = seg.get("trans_sec", 0) or 0
            fd_sec     = seg.get("final_decode_sec", 0) or 0
            seg_audio  = seg.get("seg_audio_sec")

            if encode_sec is not None:
                # ── SEG commit: 위=encode(초록), 아래=decode(주황)+pre_trans(회색)+trans(빨강) ──
                # x축 = wall-clock elapsed (stream_start_perf 기준)
                # 청크별 bar는 chunk_transcribe_start_elapsed / chunk_decode_start_elapsed 사용
                _CHUNK_COLORS = ["#4CAF50", "#81C784"]
                chunk_log = seg.get("chunk_encode_log", [])
                pre_trans_sec = seg.get("pre_trans_sec", 0) or 0

                if chunk_log:
                    sorted_chunks = sorted(chunk_log, key=lambda c: c.get("chunk_id", 0))
                    prev_pos = a_start
                    for k, chunk in enumerate(sorted_chunks):
                        audio_pos = chunk.get("audio_pos_sec", prev_pos)
                        audio_w   = audio_pos - prev_pos
                        cdec      = chunk.get("chunk_decode_sec", 0) or 0
                        ds_el     = chunk.get("chunk_decode_start_elapsed")

                        # Encode bar (위쪽 존): audio-time 기준 — 청크가 커버하는 음성 구간 표시
                        # wall-clock이 아닌 audio time으로 그려야 "2초짜리 청크" 등 직관적 크기가 보임
                        if audio_w > 0:
                            ax.barh(ye, audio_w, left=prev_pos, height=bh_enc,
                                    color=_CHUNK_COLORS[k % 2], alpha=0.9, zorder=3)

                        # Decode bar (아래쪽 존): wall-clock 기준 — 실제 model.generate() 실행 시점
                        # ds_el(= chunk_decode_start_elapsed) 이 있으면 정확한 wall-clock 위치 사용
                        # 없으면 audio_pos fallback (구형 데이터 호환)
                        dec_left = ds_el if ds_el is not None else audio_pos
                        if cdec > 0:
                            ax.barh(yd, cdec, left=dec_left, height=bh_dec,
                                    color=STYLE["decode"], alpha=0.85, zorder=3)

                        prev_pos = audio_pos

                    # 마지막 chunk audio_pos ~ encode_sec gap (연한 초록)
                    gap = encode_sec - prev_pos
                    if gap > 0.005:
                        ax.barh(ye, gap, left=prev_pos, height=bh_enc,
                                color=STYLE["encode_tail"], alpha=0.7, zorder=3)
                else:
                    # chunk_encode_log 없음: audio duration으로 단일 encode 바 + decode 바
                    audio_dur = max((_audio_end(seg) or encode_sec) - a_start, 0)
                    if audio_dur > 0:
                        ax.barh(ye, audio_dur, left=a_start, height=bh_enc,
                                color=STYLE["encode"], alpha=0.9, zorder=3)
                    if decode_sec > 0:
                        ax.barh(yd, decode_sec, left=encode_sec, height=bh_dec,
                                color=STYLE["decode"], alpha=0.9, zorder=3)

                if "encode" not in legend_handles:
                    legend_handles["encode"] = mpatches.Patch(
                        color=STYLE["encode"], label="Encode (audio coverage, audio-time)")
                if "decode" not in legend_handles:
                    legend_handles["decode"] = mpatches.Patch(
                        color=STYLE["decode"], label="Decode (model.generate(), wall-clock)")

                # SEG 감지 x 좌표 = encode_sec + decode_sec (wall-clock)
                seg_det_x = encode_sec + decode_sec

                # pre_trans bar (회색): SEG 감지 ~ 번역 API 호출 시작 (텍스트 처리/correction)
                trans_left = seg_det_x + pre_trans_sec
                if pre_trans_sec > 0.005:
                    ax.barh(yd, pre_trans_sec, left=seg_det_x, height=bh_dec,
                            color=STYLE["pre_trans"], alpha=0.75, zorder=3)
                    if "pre_trans" not in legend_handles:
                        legend_handles["pre_trans"] = mpatches.Patch(
                            color=STYLE["pre_trans"], label="Pre-trans (text proc / correction)")

                # trans bar (빨강): 번역 API 호출
                if trans_sec > 0:
                    ax.barh(yd, trans_sec, left=trans_left, height=bh_dec,
                            color=STYLE["trans"], alpha=0.9, zorder=3)
                if "trans" not in legend_handles:
                    legend_handles["trans"] = mpatches.Patch(
                        color=STYLE["trans"], label="Trans layer")

                # ◆ SEG 감지 마커
                ax.scatter(seg_det_x, y, marker="D", color=STYLE["seg_marker"],
                           s=55, zorder=7, linewidths=0.5, edgecolors="#888888")
                if "seg_det" not in legend_handles:
                    legend_handles["seg_det"] = plt.Line2D(
                        [], [], marker="D", color=STYLE["seg_marker"],
                        markersize=6, linestyle="None", label="SEG detected")

                # FSL 레이블: seg_det_x + fsl_sec = 번역 완료 시점
                if fsl is not None:
                    ax.text(seg_det_x + fsl + 0.05, y + bh / 2 + 0.06,
                            f"FSL {fsl:.2f}s", fontsize=7, color=STYLE["seg_marker"],
                            va="bottom", zorder=8)

                # audioEndSec 마커 (추정 발화 끝, 초록 실선)
                if a_end:
                    _vlines_seg(ax, a_end, y, bh / 2 + 0.05,
                                STYLE["audio_end"], lw=2.0, ls="-", zorder=6)
                    if "audio_end" not in legend_handles:
                        legend_handles["audio_end"] = plt.Line2D(
                            [], [], color=STYLE["audio_end"], lw=2,
                            label="audioEndSec (추정 발화 끝)")

                # seg_audio_sec 마커 (SEG 감지 청크 경계, 회색 점선)
                if seg_audio:
                    _vlines_seg(ax, seg_audio, y, bh / 2,
                                STYLE["seg_audio"], lw=1.5, ls=":", zorder=6)
                    if "seg_audio" not in legend_handles:
                        legend_handles["seg_audio"] = plt.Line2D(
                            [], [], color=STYLE["seg_audio"], lw=1.5, ls=":",
                            label="seg_audio_sec (SEG 감지 청크 경계)")

            elif fd_sec > 0 or trans_sec > 0:
                # ── VAD/finish commit: final_decode + trans (audioEndSec 기준) ──
                bar_start = a_end or a_start
                if fd_sec > 0:
                    ax.barh(yd, fd_sec, left=bar_start, height=bh_dec,
                            color=STYLE["final_decode"], alpha=0.8, zorder=3)
                    if "final_decode" not in legend_handles:
                        legend_handles["final_decode"] = mpatches.Patch(
                            color=STYLE["final_decode"], label="Final decode (VAD/finish)")
                if trans_sec > 0:
                    ax.barh(yd, trans_sec, left=bar_start + fd_sec, height=bh_dec,
                            color=STYLE["trans"], alpha=0.9, zorder=3)
                    if "trans" not in legend_handles:
                        legend_handles["trans"] = mpatches.Patch(
                            color=STYLE["trans"], label="Trans layer")
                if fsl is not None:
                    ax.text(bar_start + fd_sec + trans_sec + 0.05, y + bh / 2 + 0.06,
                            f"FSL {fsl:.2f}s", fontsize=7, color=STYLE["seg_marker"],
                            va="bottom", zorder=8)

            else:
                ax.barh(y, a_end - a_start, left=a_start, height=bh,
                        color=STYLE["waveform"], alpha=0.5, zorder=3)

        else:
            # 구 포맷 fallback: 오디오 구간만 표시
            ax.barh(y, a_end - a_start, left=a_start, height=bh,
                    color=STYLE["waveform"], alpha=0.5, zorder=3)

        # commit reason 레이블
        ax.text(a_start, y + bh / 2 + 0.05, reason,
                fontsize=6.5, color=_reason_color(reason), ha="left", va="bottom", zorder=8)

        prev_end = a_end

    ax.set_yticks(range(1, n_segs + 1))
    ax.set_yticklabels([f"seg {_seg_id(s)}" for s in reversed(segs)], fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0.25, n_segs + 0.9)
    ax.set_xlabel("Stream elapsed time (s)", fontsize=9)
    ax.set_title("encode / decode / trans  레이어 타이밍", fontsize=9, pad=3)
    # 정수 초 단위 major tick, 0.5초 minor tick
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(axis="x", which="major", labelsize=8, labelbottom=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(True, axis="x", which="major", lw=0.4)
    ax.grid(True, axis="x", which="minor", lw=0.2, alpha=0.4)
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()),
                  loc="lower right", fontsize=8, ncol=3)

    # ── Row 3: 텍스트 표 ─────────────────────────────────────────────────────
    ax_text.axis("off")
    if new_fmt:
        col_labels = ["Seg", "Commit", "Audio (s)", "FSL(SEG→Trans)", "Encode", "Decode", "Trans", "Text", "Translation"]
        rows_data  = [
            [
                str(_seg_id(s)),
                _reason(s),
                f"{_audio_start(s):.2f}–{_audio_end(s):.2f}",
                f"{s.get('fsl_sec', 0):.3f}s"   if s.get("fsl_sec")   is not None else "—",
                f"{s.get('encode_sec', 0):.3f}s" if s.get("encode_sec") is not None else "—",
                f"{s.get('decode_sec', 0):.3f}s" if s.get("decode_sec") is not None else "—",
                f"{s.get('trans_sec', 0):.3f}s"  if s.get("trans_sec")  is not None else "—",
                _truncate(s.get("text", s.get("original", "")), 38),
                _truncate(s.get("translation", ""), 28),
            ]
            for s in segs
        ]
        col_widths = [0.04, 0.06, 0.10, 0.07, 0.07, 0.07, 0.07, 0.30, 0.22]
    else:
        col_labels = ["Seg", "Commit", "Audio (s)", "Text", "Translation"]
        rows_data  = [
            [
                str(_seg_id(s)), _reason(s),
                f"{_audio_start(s):.2f}–{_audio_end(s):.2f}",
                _truncate(s.get("text", ""), 50),
                _truncate(s.get("translation", ""), 40),
            ]
            for s in segs
        ]
        col_widths = [0.05, 0.07, 0.12, 0.42, 0.34]

    tbl = ax_text.table(
        cellText=rows_data, colLabels=col_labels,
        colWidths=col_widths, loc="center", cellLoc="left",
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

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="레이어 타이밍 PNG 시각화")
    p.add_argument("--json",       required=True, help="결과 JSON 경로")
    p.add_argument("--audio-root", default=None,  help="LibriSpeech test-other 루트 (waveform용)")
    p.add_argument("--out-dir",    default=str(Path(__file__).resolve().parent.parent / "servers" / "results" / "fsl" / "test"),
                   help="출력 디렉터리")
    p.add_argument("--file-id",    default=None,  help="특정 file_id 하나만 처리")
    p.add_argument("--max-files",  type=int, default=None, help="앞에서 N개 파일만 처리")
    p.add_argument("--sample",     type=int, default=None, help="랜덤 N개 파일 샘플링")
    p.add_argument("--all-files",  action="store_true",    help="전체 파일 처리 (느림)")
    p.add_argument("--seed",       type=int, default=42,   help="샘플링 seed")
    return p.parse_args()


def main():
    args   = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSON 로드 중: {args.json}")
    results, overall = load_json(args.json)
    print(f"  → {len(results):,}개 파일")
    if overall:
        print(f"  overall: {overall}")

    if args.file_id:
        targets = [r for r in results if r.get("file_id") == args.file_id]
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

    print(f"\n[플롯 생성]  대상 {len(targets)}개 → {out_dir}/")
    for record in tqdm(targets, unit="file"):
        out_path = out_dir / f"{record.get('file_id', 'unknown')}.png"
        try:
            plot_file(record, args.audio_root, str(out_path), vad_model=vad_model)
        except Exception as e:
            print(f"  ⚠ {record.get('file_id')}: {e}")

    print(f"\n완료!  {out_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
