"""
probe_encoder.py

가설: Qwen3-ASR encoder는 SEG 예측 학습 과정에서
  SEG 경계 근방 프레임의 특정 차원 값이 비경계 프레임과 다름.

분석 흐름:
  1. 각 샘플 encoder output (T, D) 추출
  2. seg_frame 근방(±pos_window) → pos / 나머지 → neg 레이블
  3. 차원별 discriminability = |mean_pos - mean_neg| / pooled_std
  4. 상위 K 차원에 대해:
     - threshold 분류 정확도
     - 시계열 plot (SEG 위치 표시)
     - pos/neg 분포 비교

데이터: one_seg_data_tail 권장 (SEG가 클립 중간 → temporal 편향 없음)
        one_seg_data (트림)는 SEG가 항상 마지막 → temporal confound 주의

실행 예시:
  python research/cif/probe_encoder.py
  python research/cif/probe_encoder.py --max-samples 500 --pos-window 3
  python research/cif/probe_encoder.py --test-data research/cif/one_seg_data/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[3] / "Qwen3-ASR"
_CIF_TRAIN = Path(__file__).resolve().parents[1] / "utils" / "cif_train"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CIF_TRAIN))

from train_one_seg import load_audio_16k, mel_to_encoder_frames, load_encoder


# ---------------------------------------------------------------------------
# Encoder inference
# ---------------------------------------------------------------------------

def run_encoder(audio: np.ndarray, encoder, fe, device, dtype) -> np.ndarray:
    """오디오 → encoder output (T_enc, D), float32 numpy."""
    dur = len(audio) / 16000.0
    out = fe(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]

    mel_dev = mel.to(device=device, dtype=dtype)
    with torch.no_grad():
        enc_out = encoder(
            mel_dev,
            feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
        )
    return enc_out.last_hidden_state.squeeze(0).float().cpu().numpy(), dur


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_activations(records, encoder, fe, device, dtype,
                        pos_window: int = 2, max_samples: int | None = None):
    """
    각 샘플에서 pos (SEG 근방) / neg (비경계) 프레임 activation 수집.

    반환:
      pos_acts  (N_pos, D)
      neg_acts  (N_neg, D)
      per_sample: 시각화용 dict 리스트
    """
    if max_samples:
        records = records[:max_samples]

    pos_acts_list = []
    neg_acts_list = []
    per_sample    = []

    bar = tqdm(records, desc="collecting", unit="sample")
    for rec in bar:
        try:
            start = rec.get("start", 0.0)
            end   = rec.get("end")
            audio = load_audio_16k(rec["audio"], start=start, end=end)
            feats, dur = run_encoder(audio, encoder, fe, device, dtype)
            T = feats.shape[0]

            seg_t     = rec["seg_t"]
            seg_frame = min(max(round(seg_t * T / dur), 0), T - 1)

            pos_mask = np.zeros(T, dtype=bool)
            for df in range(-pos_window, pos_window + 1):
                f = seg_frame + df
                if 0 <= f < T:
                    pos_mask[f] = True
            neg_mask = ~pos_mask

            pos_acts_list.append(feats[pos_mask])
            neg_acts_list.append(feats[neg_mask])
            per_sample.append({
                "seg_frame": seg_frame, "T": T,
                "feats": feats, "pos_mask": pos_mask,
            })
        except Exception as e:
            tqdm.write(f"  SKIP: {e}")

    pos_acts = np.concatenate(pos_acts_list, axis=0)
    neg_acts = np.concatenate(neg_acts_list, axis=0)
    return pos_acts, neg_acts, per_sample


# ---------------------------------------------------------------------------
# Discriminability
# ---------------------------------------------------------------------------

def compute_discriminability(pos_acts: np.ndarray, neg_acts: np.ndarray):
    """차원별 판별력: |mean_pos - mean_neg| / pooled_std."""
    mean_pos = pos_acts.mean(axis=0)
    mean_neg = neg_acts.mean(axis=0)
    std_pos  = pos_acts.std(axis=0)
    std_neg  = neg_acts.std(axis=0)
    pooled   = np.sqrt((std_pos ** 2 + std_neg ** 2) / 2) + 1e-8
    disc     = np.abs(mean_pos - mean_neg) / pooled
    return disc, mean_pos, mean_neg, std_pos, std_neg


def threshold_accuracy(pos_acts: np.ndarray, neg_acts: np.ndarray, dims):
    """각 차원의 midpoint threshold 분류 정확도."""
    results = []
    for d in dims:
        p = pos_acts[:, d]
        n = neg_acts[:, d]
        thresh    = (p.mean() + n.mean()) / 2.0
        direction = 1 if p.mean() > n.mean() else -1
        if direction > 0:
            correct = (p > thresh).sum() + (n <= thresh).sum()
        else:
            correct = (p < thresh).sum() + (n >= thresh).sum()
        acc = correct / (len(p) + len(n))
        results.append({
            "dim": int(d), "acc": float(acc), "thresh": float(thresh),
            "mean_pos": float(p.mean()), "mean_neg": float(n.mean()),
            "disc": float(np.abs(p.mean() - n.mean()) /
                         (np.sqrt((p.std()**2 + n.std()**2)/2) + 1e-8)),
        })
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timeseries(per_sample, top_dims, top_disc, n_samples: int = 4,
                   out_path=None):
    """샘플별 상위 차원 시계열 + SEG 수직선."""
    samples = per_sample[:n_samples]
    n_dims  = min(len(top_dims), 4)
    fig, axes = plt.subplots(len(samples), n_dims,
                             figsize=(4 * n_dims, 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]
    if n_dims == 1:
        axes = axes[:, np.newaxis]

    for i, s in enumerate(samples):
        feats     = s["feats"]
        seg_frame = s["seg_frame"]
        for j, d in enumerate(top_dims[:n_dims]):
            ax = axes[i, j]
            ax.plot(feats[:, d], lw=0.8, color="steelblue")
            ax.axvline(seg_frame, color="red", lw=1.5,
                       label=f"SEG f={seg_frame}")
            ax.set_title(f"dim {d}  disc={top_disc[j]:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"sample {i}", fontsize=8)
            if i == 0:
                ax.legend(fontsize=6)

    plt.suptitle("Encoder activation time series — top discriminative dims",
                 fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close()


def plot_distributions(pos_acts, neg_acts, top_dims, top_disc, out_path=None):
    """상위 차원의 pos/neg 분포 비교 히스토그램."""
    n_dims = min(len(top_dims), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, d in enumerate(top_dims[:n_dims]):
        ax = axes[i]
        ax.hist(neg_acts[:, d], bins=60, alpha=0.5, label="non-SEG", density=True,
                color="steelblue")
        ax.hist(pos_acts[:, d], bins=60, alpha=0.6, label="SEG±w", density=True,
                color="tomato")
        ax.set_title(f"dim {d}  disc={top_disc[i]:.2f}", fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle("Encoder activation distribution: SEG vs non-SEG frames",
                 fontsize=11)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close()


def plot_disc_curve(disc: np.ndarray, top_k: int, out_path=None):
    """전체 차원 discriminability 분포 (랭크 순)."""
    sorted_disc = np.sort(disc)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(sorted_disc, lw=0.8)
    axes[0].axvline(top_k, color="red", ls="--", label=f"top-{top_k} cutoff")
    axes[0].set_xlabel("dimension rank")
    axes[0].set_ylabel("discriminability")
    axes[0].set_title("Discriminability by rank (all dims)")
    axes[0].legend()

    axes[1].plot(sorted_disc[:200], lw=1.0)
    axes[1].axvline(top_k, color="red", ls="--")
    axes[1].set_xlabel("dimension rank")
    axes[1].set_title("Discriminability top-200 zoom")

    plt.suptitle("Per-dimension discriminability: |Δmean| / pooled_std",
                 fontsize=11)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close()


def plot_heatmap(per_sample, top_dims, n_samples: int = 30, out_path=None):
    """샘플 × 상위 차원 heatmap. SEG 위치를 x축 정규화해서 정렬."""
    n_show   = min(n_samples, len(per_sample))
    n_dims   = min(len(top_dims), 20)
    # 각 샘플에서 SEG 근방 vs 비근방의 차원별 평균 차이
    diff_mat = np.zeros((n_show, n_dims))

    for i, s in enumerate(per_sample[:n_show]):
        feats    = s["feats"]
        pos_mask = s["pos_mask"]
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        diff_mat[i] = feats[pos_mask].mean(0)[top_dims[:n_dims]] - \
                      feats[neg_mask].mean(0)[top_dims[:n_dims]]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(diff_mat.T, aspect="auto", cmap="RdBu_r",
                   vmin=-diff_mat.std() * 2, vmax=diff_mat.std() * 2)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels([str(d) for d in top_dims[:n_dims]], fontsize=7)
    ax.set_xlabel("sample index")
    ax.set_ylabel("dimension")
    ax.set_title("mean_pos - mean_neg per sample (top dims, red=pos↑)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _tail = Path(__file__).resolve().parent / "one_seg_data_tail"
    _one  = Path(__file__).resolve().parent / "one_seg_data"

    parser = argparse.ArgumentParser(description="Encoder SEG boundary probe")
    parser.add_argument("--test-data", default=str(_tail / "test.jsonl"),
                        help="one_seg_data_tail 권장 (SEG가 중간 → temporal 편향 없음)")
    parser.add_argument("--model",      default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples",type=int, default=300,
                        help="사용할 최대 샘플 수 (속도/메모리)")
    parser.add_argument("--pos-window", type=int, default=2,
                        help="SEG 근방 ±N 프레임을 pos로 간주 (1f≈80ms)")
    parser.add_argument("--top-k",      type=int, default=20)
    parser.add_argument("--n-plot",     type=int, default=6,
                        help="시계열 plot할 샘플 수")
    parser.add_argument("--out-dir",    default="research/cif/probe_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    print(f"Loading encoder from {args.model} …")
    encoder = load_encoder(args.model, args.device, dtype)

    fe = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )

    with open(args.test_data) as f:
        records = [json.loads(l) for l in f if l.strip()]
    n_use = min(args.max_samples, len(records))
    print(f"  data: {args.test_data}  ({len(records)} → using {n_use})\n")

    # ── activation 수집 ──
    pos_acts, neg_acts, per_sample = collect_activations(
        records, encoder, fe, args.device, dtype,
        pos_window=args.pos_window, max_samples=n_use,
    )
    print(f"\n  pos frames: {pos_acts.shape[0]}  neg frames: {neg_acts.shape[0]}")
    print(f"  D = {pos_acts.shape[1]}")

    # ── discriminability ──
    disc, mean_pos, mean_neg, std_pos, std_neg = compute_discriminability(
        pos_acts, neg_acts)

    top_idx  = np.argsort(disc)[::-1][:args.top_k]
    top_disc = disc[top_idx]

    thresh_res = threshold_accuracy(pos_acts, neg_acts, top_idx)

    # ── 결과 출력 ──
    print(f"\n{'='*60}")
    print(f"  SEG probe  pos_window=±{args.pos_window}f  top-{args.top_k}")
    print(f"{'─'*60}")
    print(f"  {'dim':>5}  {'disc':>6}  {'acc':>6}  {'mean_pos':>9}  {'mean_neg':>9}")
    print(f"{'─'*60}")
    for r in thresh_res:
        print(f"  {r['dim']:5d}  {r['disc']:6.3f}  {r['acc']:6.3f}"
              f"  {r['mean_pos']:+9.4f}  {r['mean_neg']:+9.4f}")
    print(f"{'─'*60}")
    avg_acc = np.mean([r["acc"] for r in thresh_res])
    print(f"  mean acc (top-{len(thresh_res)} dims): {avg_acc:.3f}")
    disc_stats = disc[top_idx]
    print(f"  disc  max={disc_stats[0]:.3f}  median={np.median(disc_stats):.3f}"
          f"  min(top-{args.top_k})={disc_stats[-1]:.3f}")
    print(f"{'='*60}")

    # ── 저장 ──
    np.save(out_dir / "disc.npy",     disc)
    np.save(out_dir / "top_dims.npy", top_idx)
    with open(out_dir / "thresh_results.json", "w") as f:
        json.dump(thresh_res, f, indent=2)
    print(f"\n  arrays → {out_dir}/disc.npy, top_dims.npy")
    print(f"  table  → {out_dir}/thresh_results.json")

    # ── plots ──
    print("\nPlotting …")
    plot_disc_curve(disc, args.top_k,
                    out_path=out_dir / "disc_curve.png")
    plot_timeseries(per_sample, top_idx, top_disc,
                    n_samples=args.n_plot,
                    out_path=out_dir / "timeseries.png")
    plot_distributions(pos_acts, neg_acts, top_idx, top_disc,
                       out_path=out_dir / "distributions.png")
    plot_heatmap(per_sample, top_idx,
                 n_samples=min(50, len(per_sample)),
                 out_path=out_dir / "heatmap.png")
    print("Done.")


if __name__ == "__main__":
    main()
