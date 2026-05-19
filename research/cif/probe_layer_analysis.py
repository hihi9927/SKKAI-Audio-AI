"""
probe_layer_analysis.py  v2

[Pass 1]  24레이어 hook → disc 행렬 (24, 1024) + 통계 캐시
[시각화]  violin / heatmap(global+normalized) / dim profiles / top-layer 분포
[일관성]  consistent vs specific 차원 scatter + 프로파일
[Pass 2]  선택 (layer, dim) → LR 단일/혼합 분류기 비교

실행:
  python research/cif/probe_layer_analysis.py
  python research/cif/probe_layer_analysis.py --skip-pass1        # disc 캐시 재사용
  python research/cif/probe_layer_analysis.py --no-classifier     # 시각화/일관성만
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                              precision_recall_curve, auc as sklearn_auc)
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
_CIF_TRAIN = Path(__file__).resolve().parent / "utils" / "cif_train"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CIF_TRAIN))

from train_one_seg import load_audio_16k, load_encoder


# ---------------------------------------------------------------------------
# Hook collector
# ---------------------------------------------------------------------------

class LayerHookCollector:
    """지정 레이어에만 hook을 걸어 hidden_states (T, d_model)를 수집."""

    def __init__(self, encoder, layer_indices=None):
        self.encoder = encoder
        self.layer_indices = (set(layer_indices) if layer_indices is not None
                              else set(range(len(encoder.layers))))
        self._acts: dict[int, np.ndarray] = {}
        self._hooks = []

    def __enter__(self):
        for i, layer in enumerate(self.encoder.layers):
            if i not in self.layer_indices:
                continue
            def _hook(module, inp, output, idx=i):
                act = output[0] if isinstance(output, (tuple, list)) else output
                self._acts[idx] = act.detach().float().cpu().numpy()
            self._hooks.append(layer.register_forward_hook(_hook))
        return self

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def pop(self) -> dict[int, np.ndarray]:
        acts, self._acts = dict(self._acts), {}
        return acts


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def seg_frames_from_rec(rec: dict, T: int, dur: float) -> list[int]:
    if "seg_timestamps" in rec:
        return [min(max(round(t * T / dur), 0), T - 1)
                for t in rec["seg_timestamps"]]
    return [min(max(round(rec["seg_t"] * T / dur), 0), T - 1)]


def make_labels(seg_frames: list[int], T: int, pos_window: int) -> np.ndarray:
    labels = np.zeros(T, np.int32)
    for sf in seg_frames:
        for df in range(-pos_window, pos_window + 1):
            f = sf + df
            if 0 <= f < T:
                labels[f] = 1
    return labels


def load_records(path: str, max_n=None) -> list:
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    return recs[:max_n] if max_n else recs


def make_fe():
    return WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )


def encode_one(rec, encoder, fe, device, dtype):
    """단일 레코드 → mel forward. 반환: dur (float). hook이 activation 수집."""
    start = rec.get("start", 0.0)
    end   = rec.get("end")
    audio = load_audio_16k(rec["audio"], start=start, end=end)
    dur   = len(audio) / 16000.0
    out   = fe(audio, sampling_rate=16000, return_tensors="pt",
               return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]
    with torch.no_grad():
        encoder(mel.to(device=device, dtype=dtype),
                feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long))
    return dur


# ---------------------------------------------------------------------------
# Running stats accumulator
# ---------------------------------------------------------------------------

class DiscAccumulator:
    """레이어 × 차원별 SEG/non-SEG 통계를 온라인 누적. (sum, sum², count)만 저장."""

    def __init__(self, n_layers: int, d_model: int):
        s = (n_layers, d_model)
        self.pos_sum = np.zeros(s, np.float64)
        self.pos_sq  = np.zeros(s, np.float64)
        self.pos_n   = np.zeros(n_layers, np.int64)
        self.neg_sum = np.zeros(s, np.float64)
        self.neg_sq  = np.zeros(s, np.float64)
        self.neg_n   = np.zeros(n_layers, np.int64)
        self.n_layers, self.d_model = n_layers, d_model

    def update(self, layer_acts: dict, labels: np.ndarray):
        pm, nm = labels == 1, labels == 0
        for k, feats in layer_acts.items():
            f64 = feats.astype(np.float64)
            if pm.any():
                pf = f64[pm]
                self.pos_sum[k] += pf.sum(0)
                self.pos_sq[k]  += (pf ** 2).sum(0)
                self.pos_n[k]   += len(pf)
            if nm.any():
                nf = f64[nm]
                self.neg_sum[k] += nf.sum(0)
                self.neg_sq[k]  += (nf ** 2).sum(0)
                self.neg_n[k]   += len(nf)

    def _mean_var(self):
        np_ = np.maximum(self.pos_n[:, None], 1).astype(np.float64)
        nn_ = np.maximum(self.neg_n[:, None], 1).astype(np.float64)
        mp  = self.pos_sum / np_
        mn  = self.neg_sum / nn_
        vp  = np.maximum(self.pos_sq / np_ - mp ** 2, 1e-8)
        vn  = np.maximum(self.neg_sq / nn_ - mn ** 2, 1e-8)
        return mp, mn, vp, vn

    def disc_matrix(self) -> np.ndarray:
        mp, mn, vp, vn = self._mean_var()
        pooled = np.sqrt((self.pos_n[:, None] * vp + self.neg_n[:, None] * vn) /
                         np.maximum(self.pos_n[:, None] + self.neg_n[:, None], 1) + 1e-8)
        return np.abs(mp - mn) / pooled

    def kl_matrix(self) -> np.ndarray:
        """Symmetric KL divergence (Gaussian approx): 0.5*(KL(P||Q) + KL(Q||P)).
        P=SEG, Q=non-SEG. 값이 클수록 두 분포가 멀다."""
        mp, mn, vp, vn = self._mean_var()
        delta_sq = (mp - mn) ** 2
        kl_pq = 0.5 * (np.log(vn / vp) - 1.0 + vp / vn + delta_sq / vn)
        kl_qp = 0.5 * (np.log(vp / vn) - 1.0 + vn / vp + delta_sq / vp)
        return 0.5 * (kl_pq + kl_qp)

    def gaussian_params(self, layer: int, dim: int):
        np_ = max(self.pos_n[layer], 1)
        nn_ = max(self.neg_n[layer], 1)
        mp = self.pos_sum[layer, dim] / np_
        mn = self.neg_sum[layer, dim] / nn_
        sp = max((self.pos_sq[layer, dim] / np_ - mp ** 2) ** 0.5, 1e-6)
        sn = max((self.neg_sq[layer, dim] / nn_ - mn ** 2) ** 0.5, 1e-6)
        return mp, sp, mn, sn

    def save(self, path):
        np.savez(path, pos_sum=self.pos_sum, pos_sq=self.pos_sq, pos_n=self.pos_n,
                       neg_sum=self.neg_sum, neg_sq=self.neg_sq, neg_n=self.neg_n)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        acc = cls(*d["pos_sum"].shape)
        for k in ("pos_sum", "pos_sq", "pos_n", "neg_sum", "neg_sq", "neg_n"):
            setattr(acc, k, d[k])
        return acc


# ---------------------------------------------------------------------------
# Pass 1: disc matrix  (preencode — layer hooks)
# ---------------------------------------------------------------------------

def run_pass1(encoder, fe, records, pos_window, device, dtype) -> DiscAccumulator:
    acc = None
    n_layers = len(encoder.layers)

    with LayerHookCollector(encoder) as coll:
        for rec in tqdm(records, desc="pass1", unit="clip"):
            try:
                dur = encode_one(rec, encoder, fe, device, dtype)
                layer_acts = coll.pop()
                T = layer_acts[0].shape[0]

                if acc is None:
                    d_model = layer_acts[0].shape[1]
                    print(f"  detected: n_layers={n_layers}  d_model={d_model}")
                    acc = DiscAccumulator(n_layers, d_model)

                labels = make_labels(seg_frames_from_rec(rec, T, dur), T, pos_window)
                acc.update(layer_acts, labels)
            except Exception as e:
                tqdm.write(f"  SKIP: {e}")

    return acc


# ---------------------------------------------------------------------------
# Pass 1: disc matrix  (postencode — last_hidden_state, no hooks)
# ---------------------------------------------------------------------------

def encode_one_postencode(rec, encoder, fe, device, dtype):
    """proj2 이후 last_hidden_state (T, 2048) 직접 반환 — hook 없음."""
    start = rec.get("start", 0.0)
    end   = rec.get("end")
    audio = load_audio_16k(rec["audio"], start=start, end=end)
    dur   = len(audio) / 16000.0
    out   = fe(audio, sampling_rate=16000, return_tensors="pt",
               return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]
    with torch.no_grad():
        enc_out = encoder(mel.to(device=device, dtype=dtype),
                          feature_lens=torch.tensor([mel_len], device=device,
                                                    dtype=torch.long))
    feats = enc_out.last_hidden_state.squeeze(0).float().cpu().numpy()
    return feats, dur


def run_pass1_postencode(encoder, fe, records, pos_window, device, dtype) -> DiscAccumulator:
    """postencode: last_hidden_state (T, 2048)를 단일 레이어(idx=0)로 누적."""
    acc = None
    for rec in tqdm(records, desc="postencode", unit="clip"):
        try:
            feats, dur = encode_one_postencode(rec, encoder, fe, device, dtype)
            T = feats.shape[0]
            if acc is None:
                d_model = feats.shape[1]
                print(f"  d_model={d_model}")
                acc = DiscAccumulator(1, d_model)
            labels = make_labels(seg_frames_from_rec(rec, T, dur), T, pos_window)
            acc.update({0: feats}, labels)
        except Exception as e:
            tqdm.write(f"  SKIP: {e}")
    return acc


# ---------------------------------------------------------------------------
# Pass 2: feature collection for classifier
# ---------------------------------------------------------------------------

def run_pass2(encoder, fe, records, configs: dict, pos_window, device, dtype, desc="pass2"):
    """
    configs: {name: {"layers": [int], "dims": {layer_idx: ndarray | None}}}
    반환: {name: (X, y)}  X: (N_frames, n_features)
    """
    all_layers = set()
    for cfg in configs.values():
        all_layers.update(cfg["layers"])

    X_lists = {n: [] for n in configs}
    y_list  = []

    with LayerHookCollector(encoder, list(all_layers)) as coll:
        for rec in tqdm(records, desc=desc, unit="clip"):
            try:
                dur = encode_one(rec, encoder, fe, device, dtype)
                layer_acts = coll.pop()
                T = layer_acts[min(layer_acts)].shape[0]

                labels = make_labels(seg_frames_from_rec(rec, T, dur), T, pos_window)
                y_list.append(labels)

                for name, cfg in configs.items():
                    parts = []
                    for li in sorted(cfg["layers"]):
                        act  = layer_acts[li]
                        dims = cfg["dims"].get(li)
                        parts.append(act[:, dims] if dims is not None else act)
                    X_lists[name].append(np.concatenate(parts, axis=1))

            except Exception as e:
                tqdm.write(f"  SKIP: {e}")

    y_all = np.concatenate(y_list)
    return {name: (np.concatenate(X_lists[name], axis=0), y_all) for name in configs}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def train_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=300, class_weight="balanced",
                              solver="saga", C=1.0, n_jobs=-1, verbose=0)
    clf.fit(X, y)
    return clf


def eval_clf(clf, X: np.ndarray, y: np.ndarray) -> dict:
    proba   = clf.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    pr_auc  = sklearn_auc(rec, prec)
    bal_acc = balanced_accuracy_score(y, (proba >= 0.5).astype(int))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "bal_acc": bal_acc}


# ---------------------------------------------------------------------------
# Plots — Pass 1 시각화
# ---------------------------------------------------------------------------

def plot_violin(disc_mat: np.ndarray, out_path):
    """레이어별 disc 전체 분포 + max disc 오버레이."""
    fig, ax = plt.subplots(figsize=(14, 5))
    parts = ax.violinplot([disc_mat[i] for i in range(disc_mat.shape[0])],
                          positions=range(disc_mat.shape[0]),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue"); pc.set_alpha(0.55)
    ax.plot(range(disc_mat.shape[0]), disc_mat.max(1),
            "o-", ms=4, lw=1.2, color="crimson", label="max disc")
    ax.set_xlabel("Encoder layer"); ax.set_ylabel("Discriminability")
    ax.set_xticks(range(disc_mat.shape[0]))
    ax.set_xticklabels([f"L{i}" for i in range(disc_mat.shape[0])], fontsize=7, rotation=45)
    ax.set_title("Per-layer disc distribution (violin)  +  max disc (red)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_heatmap_global(disc_mat: np.ndarray, out_path, top_k: int = 128):
    """전역 max 기준 상위 dim 히트맵."""
    top_idx = np.argsort(disc_mat.max(0))[::-1][:top_k]
    data    = disc_mat[:, top_idx]
    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="hot")
    ax.set_xlabel(f"Top-{top_k} dims (sorted by max disc across layers)")
    ax.set_ylabel("Encoder layer")
    ax.set_yticks(range(disc_mat.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(disc_mat.shape[0])], fontsize=8)
    ax.set_title("Layer × Dim  discriminability  —  global top dims")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_heatmap_normalized(disc_mat: np.ndarray, out_path, top_k: int = 48):
    """레이어별 row-normalized 히트맵 — 각 레이어의 top dim을 독립적으로 표시."""
    per_top = [set(np.argsort(disc_mat[i])[::-1][:top_k]) for i in range(disc_mat.shape[0])]
    all_dims = sorted(set().union(*per_top))
    sub      = disc_mat[:, all_dims]
    normed   = sub / (sub.max(1, keepdims=True) + 1e-8)   # row-normalize

    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(normed, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Dimensions (union of per-layer top dims)")
    ax.set_ylabel("Encoder layer")
    ax.set_yticks(range(disc_mat.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(disc_mat.shape[0])], fontsize=8)
    ax.set_title("Layer × Dim  —  row-normalized  (relative to each layer's own max)")
    plt.colorbar(im, ax=ax, label="relative disc", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_dim_profiles(disc_mat: np.ndarray, out_path,
                      n_consistent: int = 6, n_specific: int = 6):
    """상위 consistent / specific 차원의 레이어별 disc 프로파일."""
    mean_d = disc_mat.mean(0)
    std_d  = disc_mat.std(0)
    max_d  = disc_mat.max(0)

    top_consist  = np.argsort(mean_d / (std_d + 1e-6))[::-1][:n_consistent]
    top_specific = np.argsort(max_d * std_d)[::-1][:n_specific]

    layers = range(disc_mat.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for dim in top_consist:
        axes[0].plot(layers, disc_mat[:, dim], lw=1.2,
                     label=f"dim{dim}  μ={mean_d[dim]:.3f} σ={std_d[dim]:.3f}")
    axes[0].set_title(f"Top-{n_consistent} consistent dims  (high μ/σ)")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Discriminability")
    axes[0].set_xticks(list(layers))
    axes[0].set_xticklabels([f"L{i}" for i in layers], fontsize=7, rotation=45)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    for dim in top_specific:
        axes[1].plot(layers, disc_mat[:, dim], lw=1.2,
                     label=f"dim{dim}  max={max_d[dim]:.3f} σ={std_d[dim]:.3f}")
    axes[1].set_title(f"Top-{n_specific} specific dims  (high max×σ)")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Discriminability")
    axes[1].set_xticks(list(layers))
    axes[1].set_xticklabels([f"L{i}" for i in layers], fontsize=7, rotation=45)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.suptitle("Dimension profiles across layers", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_top_layer_distributions(acc: DiscAccumulator, disc_mat: np.ndarray,
                                  out_path, n_layers_show: int = 4, n_dims_show: int = 4):
    """disc 상위 레이어 × 상위 차원 SEG vs non-SEG Gaussian 분포."""
    best_layers = np.argsort(disc_mat.max(1))[::-1][:n_layers_show]
    fig, axes   = plt.subplots(n_layers_show, n_dims_show,
                                figsize=(4 * n_dims_show, 3 * n_layers_show))
    if n_layers_show == 1:
        axes = axes[np.newaxis, :]

    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * np.pi) ** 0.5)

    for row, li in enumerate(best_layers):
        top_dims = np.argsort(disc_mat[li])[::-1][:n_dims_show]
        for col, dim in enumerate(top_dims):
            mp, sp, mn, sn = acc.gaussian_params(li, dim)
            lo = min(mp - 3.5 * sp, mn - 3.5 * sn)
            hi = max(mp + 3.5 * sp, mn + 3.5 * sn)
            x  = np.linspace(lo, hi, 300)
            ax = axes[row, col]
            ax.fill_between(x, gauss(x, mn, sn), alpha=0.45, color="steelblue", label="non-SEG")
            ax.fill_between(x, gauss(x, mp, sp), alpha=0.45, color="salmon",    label="SEG")
            ax.set_title(f"L{li}  dim{dim}  disc={disc_mat[li, dim]:.3f}", fontsize=8)
            ax.set_xlabel("activation", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"Layer {li}", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    plt.suptitle("SEG vs non-SEG  (Gaussian approx)  —  top layers × top dims", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Plots — postencode 전용
# ---------------------------------------------------------------------------

def plot_rank_curves(disc_vec: np.ndarray, kl_vec: np.ndarray, out_path,
                     top_k: int = 200):
    """disc/KL 순위 곡선 — postencode 2048 차원 분석용."""
    disc_sorted = np.sort(disc_vec)[::-1]
    kl_sorted   = np.sort(kl_vec)[::-1]
    D = len(disc_sorted)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전체 순위 곡선
    axes[0].plot(range(D), disc_sorted, lw=1.0, color="steelblue", label="disc")
    axes[0].axvline(top_k, color="red", ls="--", lw=0.8,
                    label=f"top-{top_k} cutoff")
    axes[0].set_xlabel("Dimension rank"); axes[0].set_ylabel("Discriminability (disc)")
    axes[0].set_title(f"Disc by rank  (all {D} dims)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # top-K zoom: disc vs KL 비교
    ax1 = axes[1]
    ax2 = ax1.twinx()
    top_by_disc = np.argsort(disc_vec)[::-1][:top_k]
    l1, = ax1.plot(range(top_k), disc_vec[top_by_disc],
                   lw=1.2, color="steelblue", label="disc")
    l2, = ax2.plot(range(top_k), kl_vec[top_by_disc],
                   lw=1.2, color="crimson", ls="--", label="sym KL")
    ax1.set_xlabel("Rank (by disc)"); ax1.set_ylabel("disc", color="steelblue")
    ax2.set_ylabel("sym KL", color="crimson")
    ax1.set_title(f"Top-{top_k} dims: disc vs sym KL")
    ax1.legend(handles=[l1, l2], loc="upper right"); ax1.grid(alpha=0.3)

    plt.suptitle("Postencode (projected 2048-dim)  discriminability", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_disc_kl_scatter(disc_vec: np.ndarray, kl_vec: np.ndarray, out_path,
                          top_n: int = 10):
    """차원별 disc vs KL 산점도 — 두 지표가 잡는 신호 비교."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(disc_vec, kl_vec, s=6, alpha=0.4, color="steelblue")

    # 두 지표 모두 높은 상위 N개 annotate
    score = disc_vec * kl_vec
    top_both = np.argsort(score)[::-1][:top_n]
    for d in top_both:
        ax.annotate(f"d{d}", (disc_vec[d], kl_vec[d]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Disc  (|Δmean| / pooled_std)")
    ax.set_ylabel("Symmetric KL divergence")
    ax.set_title("Per-dim disc vs KL  (postencode 2048-dim)\n"
                 "top-right = both metrics high")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Plots — KL divergence
# ---------------------------------------------------------------------------

def plot_kl_vs_disc(disc_mat: np.ndarray, kl_mat: np.ndarray, out_path,
                    top_k: int = 10):
    """레이어별 max disc / max KL 비교 + 상위 dim 프로파일."""
    layers = list(range(disc_mat.shape[0]))

    disc_best = disc_mat.max(1)
    kl_best   = kl_mat.max(1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # [0] per-layer max 비교
    ax = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(layers, disc_best, "o-", ms=4, lw=1.2, color="steelblue", label="max disc")
    l2, = ax2.plot(layers, kl_best,  "s-", ms=4, lw=1.2, color="crimson",   label="max KL")
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("disc", color="steelblue")
    ax2.set_ylabel("sym KL", color="crimson")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    ax.set_title("Per-layer max disc vs max KL")
    ax.legend(handles=[l1, l2], loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # [1] disc 상위 K dim 프로파일
    top_disc_dims = np.argsort(disc_mat.max(0))[::-1][:top_k]
    for d in top_disc_dims:
        axes[1].plot(layers, disc_mat[:, d], lw=1.0, alpha=0.8, label=f"d{d}")
    axes[1].set_title(f"Top-{top_k} dims by disc  (layer profiles)")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("disc")
    axes[1].set_xticks(layers)
    axes[1].set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    axes[1].legend(fontsize=6, ncol=2); axes[1].grid(alpha=0.3)

    # [2] KL 상위 K dim 프로파일
    top_kl_dims = np.argsort(kl_mat.max(0))[::-1][:top_k]
    for d in top_kl_dims:
        axes[2].plot(layers, kl_mat[:, d], lw=1.0, alpha=0.8, label=f"d{d}")
    axes[2].set_title(f"Top-{top_k} dims by KL  (layer profiles)")
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("sym KL")
    axes[2].set_xticks(layers)
    axes[2].set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    axes[2].legend(fontsize=6, ncol=2); axes[2].grid(alpha=0.3)

    plt.suptitle("Disc vs Symmetric KL divergence  (Gaussian approx)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_kl_heatmap(kl_mat: np.ndarray, out_path, top_k: int = 128):
    """KL divergence 히트맵."""
    top_idx = np.argsort(kl_mat.max(0))[::-1][:top_k]
    data    = kl_mat[:, top_idx]
    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set_xlabel(f"Top-{top_k} dims (sorted by max KL across layers)")
    ax.set_ylabel("Encoder layer")
    ax.set_yticks(range(kl_mat.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(kl_mat.shape[0])], fontsize=8)
    ax.set_title("Layer x Dim  Symmetric KL divergence  (SEG vs non-SEG, Gaussian approx)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Plots — 일관성 분석
# ---------------------------------------------------------------------------

def plot_consistency_scatter(disc_mat: np.ndarray, out_path):
    """차원별 mean disc (일관성) vs std disc (특이성) 산점도."""
    mean_d = disc_mat.mean(0)
    std_d  = disc_mat.std(0)
    max_d  = disc_mat.max(0)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(mean_d, std_d, c=max_d, cmap="hot", s=8, alpha=0.55)
    plt.colorbar(sc, ax=ax, label="max disc")

    top10 = np.argsort(max_d)[::-1][:10]
    for d in top10:
        ax.annotate(f"d{d}", (mean_d[d], std_d[d]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Mean disc (across layers)  ->  consistent")
    ax.set_ylabel("Std disc (across layers)  ->  layer-specific")
    ax.set_title("Dimension consistency analysis\n"
                 "bottom-right=consistent (stable across layers)   "
                 "top-right=specific (peaks at few layers)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_consistency_profiles(disc_mat: np.ndarray, out_path,
                               n_show: int = 5):
    """consistent 상위 dims vs specific 상위 dims 프로파일 + 히트맵 2종."""
    mean_d = disc_mat.mean(0)
    std_d  = disc_mat.std(0)
    max_d  = disc_mat.max(0)

    top_consist  = np.argsort(mean_d / (std_d + 1e-6))[::-1][:n_show]
    top_specific = np.argsort(max_d * std_d)[::-1][:n_show]

    layers = list(range(disc_mat.shape[0]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # [0,0] consistent 프로파일
    for dim in top_consist:
        axes[0, 0].plot(layers, disc_mat[:, dim], lw=1.3,
                        label=f"dim{dim}  μ={mean_d[dim]:.3f}")
    axes[0, 0].set_title(f"Consistent dims (top-{n_show})")
    axes[0, 0].set_xlabel("Layer"); axes[0, 0].set_ylabel("Disc")
    axes[0, 0].set_xticks(layers)
    axes[0, 0].set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    # [0,1] specific 프로파일
    for dim in top_specific:
        axes[0, 1].plot(layers, disc_mat[:, dim], lw=1.3,
                        label=f"dim{dim}  max={max_d[dim]:.3f}@L{disc_mat[:, dim].argmax()}")
    axes[0, 1].set_title(f"Specific dims (top-{n_show})")
    axes[0, 1].set_xlabel("Layer"); axes[0, 1].set_ylabel("Disc")
    axes[0, 1].set_xticks(layers)
    axes[0, 1].set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    # [1,0] consistent dims 히트맵
    data_c = disc_mat[:, top_consist].T   # (n_show, n_layers)
    im0 = axes[1, 0].imshow(data_c, aspect="auto", cmap="YlOrRd")
    axes[1, 0].set_yticks(range(n_show))
    axes[1, 0].set_yticklabels([f"dim{d}" for d in top_consist], fontsize=8)
    axes[1, 0].set_xticks(layers)
    axes[1, 0].set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    axes[1, 0].set_title("Consistent dims  ×  layers"); plt.colorbar(im0, ax=axes[1, 0])

    # [1,1] specific dims 히트맵
    data_s = disc_mat[:, top_specific].T   # (n_show, n_layers)
    im1 = axes[1, 1].imshow(data_s, aspect="auto", cmap="hot")
    axes[1, 1].set_yticks(range(n_show))
    axes[1, 1].set_yticklabels([f"dim{d}" for d in top_specific], fontsize=8)
    axes[1, 1].set_xticks(layers)
    axes[1, 1].set_xticklabels([f"L{i}" for i in layers], fontsize=6, rotation=45)
    axes[1, 1].set_title("Specific dims  ×  layers"); plt.colorbar(im1, ax=axes[1, 1])

    plt.suptitle("Consistent vs Specific dimension analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Plots — 분류기 비교
# ---------------------------------------------------------------------------

def plot_classifier_comparison(results: dict, out_path):
    names    = list(results.keys())
    roc_aucs = [results[n]["roc_auc"] for n in names]
    pr_aucs  = [results[n]["pr_auc"]  for n in names]
    bal_accs = [results[n]["bal_acc"] for n in names]

    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 2), 5))
    ax.bar(x - w, roc_aucs, w, label="ROC-AUC",  color="steelblue",  alpha=0.85)
    ax.bar(x,     pr_aucs,  w, label="PR-AUC",   color="darkorange", alpha=0.85)
    ax.bar(x + w, bal_accs, w, label="Bal-Acc",  color="seagreen",   alpha=0.85)

    for i, n in enumerate(names):
        ax.text(i - w, roc_aucs[i] + 0.003, f"{roc_aucs[i]:.3f}", ha="center", fontsize=7)
        ax.text(i,     pr_aucs[i]  + 0.003, f"{pr_aucs[i]:.3f}",  ha="center", fontsize=7)
        ax.text(i + w, bal_accs[i] + 0.003, f"{bal_accs[i]:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
    ax.set_title("Single vs combined layer  LR classifier comparison")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(disc_mat: np.ndarray):
    best  = disc_mat.max(1)
    order = np.argsort(best)[::-1]
    print(f"\n{'='*54}")
    print(f"  레이어별 max disc  (상위 5)")
    print(f"{'─'*54}")
    for i in order[:5]:
        print(f"  L{i:2d}  max_disc={best[i]:.4f}  "
              f"best_dim={disc_mat[i].argmax()}")
    print()
    mean_d = disc_mat.mean(0)
    std_d  = disc_mat.std(0)
    top5_consist  = np.argsort(mean_d / (std_d + 1e-6))[::-1][:5]
    top5_specific = np.argsort(disc_mat.max(0) * std_d)[::-1][:5]
    print(f"  consistent 상위 dim: {top5_consist.tolist()}")
    print(f"  specific   상위 dim: {top5_specific.tolist()}")
    print(f"{'='*54}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _ds          = Path(__file__).resolve().parent / "dataset" / "data"
    _default_out = Path(__file__).resolve().parent / "preencode" / "layer_analysis"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",          default="preencode",
                        choices=["preencode", "postencode"],
                        help="preencode=24 레이어 hook / postencode=last_hidden_state(2048)")
    parser.add_argument("--data",          default=str(_ds / "test.jsonl"),
                        help="Pass 1 disc 계산용 데이터")
    parser.add_argument("--train-data",    default=str(_ds / "train.jsonl"))
    parser.add_argument("--test-data",     default=str(_ds / "test.jsonl"))
    parser.add_argument("--model",         default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device",        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples",   type=int, default=None)
    parser.add_argument("--max-train",     type=int, default=None)
    parser.add_argument("--max-test",      type=int, default=None)
    parser.add_argument("--pos-window",    type=int, default=2)
    parser.add_argument("--top-k-dims",    type=int, default=32,
                        help="분류기 config당 레이어별 top dim 수")
    parser.add_argument("--top-n-layers",  type=int, default=5,
                        help="혼합 분류기에 쓸 top layer 수")
    parser.add_argument("--out-dir",       default=None,
                        help="미지정 시 mode에 따라 자동 결정")
    parser.add_argument("--skip-pass1",    action="store_true",
                        help="disc_matrix.npy + acc.npz 캐시 있으면 Pass 1 건너뜀")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Pass 2 분류기 생략 (시각화/일관성만)")
    args = parser.parse_args()

    _base = Path(__file__).resolve().parent
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.mode == "postencode":
        out_dir = _base / "postencode"
    else:
        out_dir = _base / "preencode" / "layer_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    disc_path = out_dir / "layer_disc_matrix.npy"
    acc_path  = out_dir / "layer_disc_acc.npz"

    dtype   = torch.bfloat16
    encoder = fe = None

    def ensure_encoder():
        nonlocal encoder, fe
        if encoder is None:
            print("Loading encoder …")
            encoder = load_encoder(args.model, args.device, dtype)
            fe = make_fe()

    # ── Pass 1 ──────────────────────────────────────────────────────────
    if args.skip_pass1 and disc_path.exists() and acc_path.exists():
        print("Pass 1 건너뜀 (캐시 로드)")
        disc_mat = np.load(disc_path)
        acc      = DiscAccumulator.load(acc_path)
    else:
        ensure_encoder()
        records = load_records(args.data, args.max_samples)
        print(f"\n[Pass 1 / {args.mode}]  samples={len(records)}")
        if args.mode == "postencode":
            acc = run_pass1_postencode(encoder, fe, records,
                                       args.pos_window, args.device, dtype)
        else:
            acc = run_pass1(encoder, fe, records, args.pos_window, args.device, dtype)
        disc_mat = acc.disc_matrix()
        np.save(disc_path, disc_mat)
        acc.save(acc_path)
        print(f"  disc_matrix: {disc_mat.shape}  saved.")

    kl_path = out_dir / "layer_kl_matrix.npy"
    kl_mat  = acc.kl_matrix()
    np.save(kl_path, kl_mat)
    print(f"  kl_matrix:   {kl_mat.shape}  saved.")

    print_summary(disc_mat)

    # ── 시각화 ──────────────────────────────────────────────────────────
    print("\n[시각화]")
    if args.mode == "postencode":
        # postencode: 단일 레이어(2048 dim) 전용 플롯
        disc_vec = disc_mat[0]
        kl_vec   = kl_mat[0]
        plot_rank_curves(disc_vec, kl_vec,    out_dir / "p1_rank_curves.png")
        plot_disc_kl_scatter(disc_vec, kl_vec, out_dir / "p1_disc_kl_scatter.png")
        plot_top_layer_distributions(acc, disc_mat, out_dir / "p1_top_dist.png",
                                     n_layers_show=1, n_dims_show=6)
    else:
        # preencode: 24 레이어 다층 플롯
        plot_violin(disc_mat,             out_dir / "p1_violin.png")
        plot_heatmap_global(disc_mat,     out_dir / "p1_heatmap_global.png")
        plot_heatmap_normalized(disc_mat, out_dir / "p1_heatmap_normalized.png")
        plot_dim_profiles(disc_mat,       out_dir / "p1_dim_profiles.png")
        plot_top_layer_distributions(acc, disc_mat, out_dir / "p1_top_dist.png")

        # ── KL divergence ────────────────────────────────────────────────
        print("\n[KL divergence]")
        plot_kl_heatmap(kl_mat,           out_dir / "p1_kl_heatmap.png")
        plot_kl_vs_disc(disc_mat, kl_mat, out_dir / "p1_kl_vs_disc.png")

    # ── 일관성 분석 (공통) ───────────────────────────────────────────────
    print("\n[일관성 분석]")
    plot_consistency_scatter(disc_mat,  out_dir / "p2_consistency_scatter.png")
    if args.mode == "preencode":
        plot_consistency_profiles(disc_mat, out_dir / "p2_consistency_profiles.png")

    # ── Pass 2: 분류기 ───────────────────────────────────────────────────
    if not args.no_classifier:
        print("\n[Pass 2: 분류기]")
        ensure_encoder()

        best_per_layer = disc_mat.max(1)
        best_single    = int(np.argmax(best_per_layer))
        top_layers     = np.argsort(best_per_layer)[::-1][:args.top_n_layers].tolist()
        K              = args.top_k_dims

        print(f"  best_single=L{best_single}  top_layers={sorted(top_layers)}  K={K}")

        configs = {
            f"L{best_single}_all": {
                "layers": [best_single],
                "dims":   {best_single: None},
            },
            f"L{best_single}_top{K}": {
                "layers": [best_single],
                "dims":   {best_single: np.argsort(disc_mat[best_single])[::-1][:K]},
            },
            f"top{args.top_n_layers}L_top{K}": {
                "layers": top_layers,
                "dims":   {li: np.argsort(disc_mat[li])[::-1][:K] for li in top_layers},
            },
            f"all24L_top{K // 2}": {
                "layers": list(range(disc_mat.shape[0])),
                "dims":   {li: np.argsort(disc_mat[li])[::-1][:K // 2]
                           for li in range(disc_mat.shape[0])},
            },
        }

        train_recs = load_records(args.train_data, args.max_train)
        test_recs  = load_records(args.test_data,  args.max_test)
        print(f"  train={len(train_recs)}  test={len(test_recs)}")

        tr_data = run_pass2(encoder, fe, train_recs, configs,
                             args.pos_window, args.device, dtype, "pass2 train")
        te_data = run_pass2(encoder, fe, test_recs,  configs,
                             args.pos_window, args.device, dtype, "pass2 test")

        clf_results = {}
        print(f"\n  {'config':<22} {'feat':>5}  {'ROC':>5}  {'PR':>5}  {'BalAcc':>6}")
        print(f"  {'─'*50}")
        for name, cfg in configs.items():
            X_tr, y_tr = tr_data[name]
            X_te, y_te = te_data[name]
            clf = train_lr(X_tr, y_tr)
            res = eval_clf(clf, X_te, y_te)
            clf_results[name] = res
            print(f"  {name:<22} {X_tr.shape[1]:>5}  "
                  f"{res['roc_auc']:.3f}  {res['pr_auc']:.3f}  {res['bal_acc']:.3f}")

        plot_classifier_comparison(clf_results, out_dir / "p3_classifier_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
