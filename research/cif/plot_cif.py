"""
CIF Weight Predictor 추론 테스트 스크립트.

경계 검출 방식 (--detect):
  cif   : 원래 CIF 방식 — 누적합이 정수를 넘는 시점을 경계로 판단
            acc[t] = Σ w[0..t],  fire at t ← floor(acc[t]) > floor(acc[t-1])
  kpeak : qty_head 기반 k-peak — qty_pred를 k로 반올림 후 weight 곡선 상위 k개 극대점 선택
  both  : 두 방식 동시 실행 후 시각화에서 비교 (기본값)

패널 구성:
  [상단] 파형 + 경계 수직선
  [중단] CIF weight 곡선
  [하단] 누적 곡선 + 정수 격자 + 경계 마커

실행 예시:
  python core/cif/test_cif.py --ckpt checkpoints/encoder/cif_best.pt --audio sample.wav
  python core/cif/test_cif.py --ckpt checkpoints/encoder/cif_best.pt --audio sample.wav --detect kpeak
  python core/cif/test_cif.py --ckpt checkpoints/mel/cif_best.pt     --audio sample.wav --no-show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from peft import PeftModel
from train import (
    CIFWeightPredictor,
    CIFMelPredictor,
    load_audio_16k,
    mel_to_encoder_frames,
    load_encoder,
    _forward,
)


def load_encoder_with_lora(model_path: str, lora_dir: str, device: str, dtype: torch.dtype):
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
    from transformers import AutoConfig
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map=device,
    )
    encoder = model.thinker.audio_tower
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = PeftModel.from_pretrained(encoder, lora_dir)
    encoder.eval()
    del model.thinker.model
    del model.thinker.lm_head
    torch.cuda.empty_cache()
    return encoder


# ---------------------------------------------------------------------------
# 경계 검출
# ---------------------------------------------------------------------------

def detect_by_kpeak(weights_np: np.ndarray, k: int, min_distance: int = 3) -> np.ndarray:
    """qty_pred를 k로 받아 weight 곡선 상위 k개 극대점 반환."""
    if k <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(weights_np, distance=min_distance)
    if len(peaks) == 0:
        peaks = np.arange(len(weights_np))
    if len(peaks) <= k:
        return np.sort(peaks)
    top_k = np.argsort(weights_np[peaks])[-k:]
    return np.sort(peaks[top_k])


def detect_by_accumulation(weights_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    CIF 원래 방식: 누적합이 정수를 넘는 프레임을 경계로 검출.
    반환: (fire_frames, acc)
    """
    acc = np.cumsum(weights_np)
    fire_frames = np.where(np.floor(acc[1:]) > np.floor(acc[:-1]))[0] + 1
    return fire_frames, acc


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------

def infer(args) -> dict:
    device = torch.device(args.device)
    dtype = torch.bfloat16

    ckpt = torch.load(args.ckpt, map_location=device)
    mode = ckpt.get("mode", args.mode)
    print(f"checkpoint mode: {mode}  (epoch {ckpt.get('epoch', '?')},"
          f"  val_loss={ckpt.get('val_loss', '?'):.4f})")

    lora_dir = Path(args.ckpt).parent / (Path(args.ckpt).stem + "_lora")
    is_lora = lora_dir.exists()

    if mode == "encoder":
        if is_lora:
            print(f"Loading encoder + LoRA adapter from {lora_dir} …")
            encoder = load_encoder_with_lora(args.model, str(lora_dir), args.device, dtype)
        else:
            print(f"Loading encoder (frozen) from {args.model} …")
            encoder = load_encoder(args.model, args.device, dtype)
        predictor = CIFWeightPredictor(
            d_model=2048,
            hidden=ckpt.get("hidden", 128),
            kernel_size=ckpt.get("kernel_size", 7),
        ).to(device)
    else:
        encoder = None
        predictor = CIFMelPredictor(n_mel=128, hidden=args.mel_hidden).to(device)

    predictor.load_state_dict(ckpt.get("state_dict") or ckpt["predictor"], strict=False)
    predictor.eval()

    raw_audio = load_audio_16k(args.audio)
    duration = len(raw_audio) / 16000.0
    print(f"audio: {args.audio}  ({duration:.2f}s)")

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=True,
    )

    out = feature_extractor(raw_audio, sampling_rate=16000,
                            return_tensors="pt", return_attention_mask=True)
    mel = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel = mel[:, :mel_len]
    n_enc = mel_to_encoder_frames(mel_len)

    with torch.no_grad():
        mel_dev = mel.to(device=device, dtype=dtype)
        weights, qty_pred = _forward(encoder, predictor, mel_dev, mel_len, device, dtype)
    weights_np = weights.float().cpu().numpy()

    qty_val = qty_pred.item() if qty_pred is not None else None
    if qty_val is not None:
        print(f"qty_head 예측: {qty_val:.2f}  (반올림 → {round(qty_val)}개)")

    fire_frames, acc = detect_by_accumulation(weights_np)
    enc_frame_duration = duration / n_enc
    weight_times = np.arange(len(weights_np)) * enc_frame_duration
    fire_times = fire_frames * enc_frame_duration

    print(f"\n[CIF]   Σw={acc[-1]:.3f}  경계 {len(fire_frames)}개")
    for i, (f, t) in enumerate(zip(fire_frames, fire_times)):
        print(f"  [{i+1}] frame={f}  t={t:.3f}s  w={weights_np[f]:.4f}")

    kpeak_frames = kpeak_times = None
    if qty_val is not None:
        k = round(qty_val)
        kpeak_frames = detect_by_kpeak(weights_np, k, args.peak_min_dist)
        kpeak_times = kpeak_frames * enc_frame_duration
        print(f"\n[kpeak] k={k}  경계 {len(kpeak_frames)}개")
        for i, (f, t) in enumerate(zip(kpeak_frames, kpeak_times)):
            print(f"  [{i+1}] frame={f}  t={t:.3f}s  w={weights_np[f]:.4f}")

    return dict(
        raw_audio=raw_audio,
        duration=duration,
        weights_np=weights_np,
        weight_times=weight_times,
        acc=acc,
        fire_frames=fire_frames,
        fire_times=fire_times,
        kpeak_frames=kpeak_frames,
        kpeak_times=kpeak_times,
        mode=mode,
        ckpt_path=args.ckpt,
    )


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot(result: dict, out_path: str | None, show: bool):
    raw_audio    = result["raw_audio"]
    duration     = result["duration"]
    weights_np   = result["weights_np"]
    weight_times = result["weight_times"]
    acc          = result["acc"]
    fire_frames  = result["fire_frames"]
    fire_times   = result["fire_times"]
    kpeak_frames = result.get("kpeak_frames")
    kpeak_times  = result.get("kpeak_times")
    mode         = result["mode"]
    ckpt_path    = result["ckpt_path"]

    has_kpeak = kpeak_frames is not None and len(kpeak_frames) > 0

    audio_times = np.linspace(0, duration, len(raw_audio))
    acc_max = max(acc[-1] + 0.5, 2.0)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 2]},
    )
    kpeak_info = f"  kpeak={len(kpeak_times)}개" if kpeak_times is not None else ""
    fig.suptitle(
        f"CIF Inference  |  mode={mode}  |  {Path(ckpt_path).name}"
        f"  |  CIF={len(fire_times)}개{kpeak_info}",
        fontsize=12, y=1.01,
    )

    # ── 상단: 파형 ──
    ax_wav = axes[0]
    ax_wav.plot(audio_times, raw_audio, color="#94A3B8", linewidth=0.4, label="waveform")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    for t in fire_times:
        ax_wav.axvline(t, color="#EF4444", alpha=0.7, linewidth=1.2, linestyle="-")
    if kpeak_times is not None:
        for t in kpeak_times:
            ax_wav.axvline(t, color="#F97316", alpha=0.7, linewidth=1.2, linestyle="--")
    ax_wav.legend(loc="upper right", fontsize=8)

    # ── 중단: weight 곡선 + 두 방식 마커 ──
    ax_w = axes[1]
    ax_w.fill_between(weight_times, weights_np, alpha=0.25, color="#6366F1")
    ax_w.plot(weight_times, weights_np, color="#6366F1", linewidth=0.9, label="weight")
    ax_w.set_ylabel("Weight")
    w_max = max(weights_np.max() * 1.2, 0.3)
    ax_w.set_ylim(-0.01, w_max)
    for t in fire_times:
        ax_w.axvline(t, color="#EF4444", alpha=0.35, linewidth=1.0)
    if len(fire_frames) > 0:
        ax_w.scatter(fire_times, weights_np[fire_frames],
                     color="#EF4444", zorder=5, s=40, marker="v",
                     label=f"CIF ({len(fire_times)})")
    if has_kpeak:
        for t in kpeak_times:
            ax_w.axvline(t, color="#F97316", alpha=0.35, linewidth=1.0)
        ax_w.scatter(kpeak_times, weights_np[kpeak_frames],
                     color="#F97316", zorder=6, s=40, marker="^",
                     label=f"kpeak ({len(kpeak_times)})")
    ax_w.legend(loc="upper right", fontsize=8)

    # ── 하단: 누적 곡선 ──
    ax_acc = axes[2]
    ax_acc.plot(weight_times, acc, color="#10B981", linewidth=1.2, label="accumulation")

    for k in range(1, int(acc_max) + 1):
        ax_acc.axhline(k, color="#6B7280", linewidth=0.6, linestyle="--", alpha=0.6)
        ax_acc.text(weight_times[-1] * 1.002, k, f" {k}", va="center",
                    fontsize=7, color="#6B7280", clip_on=False)

    for t in fire_times:
        ax_acc.axvline(t, color="#EF4444", alpha=0.35, linewidth=1.0)
    if len(fire_frames) > 0:
        ax_acc.scatter(fire_times, acc[fire_frames],
                       color="#EF4444", zorder=5, s=40, marker="v",
                       label=f"CIF ({len(fire_times)})")
    if has_kpeak:
        for t in kpeak_times:
            ax_acc.axvline(t, color="#F97316", alpha=0.35, linewidth=1.0)
        ax_acc.scatter(kpeak_times, acc[kpeak_frames],
                       color="#F97316", zorder=6, s=40, marker="^",
                       label=f"kpeak ({len(kpeak_times)})")

    ax_acc.set_ylabel("Accumulated weight")
    ax_acc.set_xlabel("Time (s)")
    ax_acc.set_ylim(-0.05, acc_max)
    ax_acc.legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n그래프 저장: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# 학습 데이터 시각화
# ---------------------------------------------------------------------------

def plot_data_samples(jsonl_path: str, n: int, seed: int, out_path: str | None, show: bool):
    import json, random
    random.seed(seed)

    with open(jsonl_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    samples = random.sample(records, min(n, len(records)))

    cols = 2
    rows = (len(samples) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes = np.array(axes).flatten()

    for ax, rec in zip(axes, samples):
        audio = load_audio_16k(rec["audio"])
        duration = len(audio) / 16000.0
        t = np.linspace(0, duration, len(audio))

        ax.plot(t, audio, color="#94A3B8", linewidth=0.4)
        for ts in rec["seg_timestamps"]:
            ax.axvline(ts, color="#EF4444", linewidth=1.5, alpha=0.85)

        text = rec.get("text_clean", "")[:60]
        ax.set_title(
            f"n_segs={rec['n_segs']}  {text}",
            fontsize=8, pad=3,
        )
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("s", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[len(samples):]:
        ax.set_visible(False)

    fig.suptitle(f"Train data samples  ({jsonl_path})", fontsize=10, y=1.01)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"저장: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CIF 추론/데이터 시각화")
    parser.add_argument("--data", default=None, metavar="JSONL",
                        help="학습 데이터 시각화 모드: jsonl 경로 지정 시 --ckpt 불필요")
    parser.add_argument("--n", type=int, default=6, help="--data 모드에서 표시할 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default=None, help="checkpoint 경로")
    parser.add_argument("--audio", default=None, help="테스트할 wav 파일 경로")
    parser.add_argument("--mode", choices=["encoder", "mel"], default="encoder",
                        help="checkpoint에 mode가 없을 경우 fallback")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B",
                        help="encoder mode용 ASR 모델 경로")
    parser.add_argument("--mel-hidden", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="그래프 저장 경로")
    parser.add_argument("--peak-min-dist", type=int, default=3,
                        help="kpeak 인접 극대점 최소 간격 (encoder frames)")
    parser.add_argument("--no-show", action="store_true",
                        help="plt.show() 생략 (서버 환경)")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    # ── 학습 데이터 시각화 모드 ──
    if args.data:
        out = args.out or str(Path(args.data).parent / "data_samples.png")
        plot_data_samples(args.data, args.n, args.seed, out, show=not args.no_show)
        return

    # ── 추론 시각화 모드 ──
    if not args.ckpt or not args.audio:
        parser.error("추론 모드는 --ckpt 와 --audio 가 필요합니다.")

    if args.out is None:
        audio_stem = Path(args.audio).stem
        args.out = str(Path(args.ckpt).parent / f"cif_plot_{audio_stem}.png")

    result = infer(args)
    plot(result, out_path=args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
