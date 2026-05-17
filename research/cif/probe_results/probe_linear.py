"""
probe_linear.py

인코더 output (T, 2048)에서 분류기로 SEG 경계 검출.

두 가지 데이터 포맷 자동 감지:
  one_seg  : {"audio":..., "seg_t": 1.44, "start":0.0, "end":2.8}
  multi_seg: {"audio":..., "seg_timestamps": [1.44, 2.8, 4.1]}

평가:
  1. 프레임 단위 이진 분류 — balanced_acc, F1, AUC
  2. 클립 단위 검출
     - one_seg : argmax → pos_mae, pos_acc@2f/@5f
     - multi_seg: peak detection → count_acc, count_mae, pos_mae

실행:
  python research/cif/probe_linear.py                      # logistic (기본)
  python research/cif/probe_linear.py --arch cnn1d         # 1D CNN temporal
  python research/cif/probe_linear.py --arch gru           # bi-GRU temporal
  python research/cif/probe_linear.py --arch cnn1d --epochs 20 --lr 5e-4
  python research/cif/probe_linear.py --one-seg
  python research/cif/probe_linear.py --top-k 128
"""

import argparse
import json
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              roc_auc_score, precision_recall_curve)
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[3] / "Qwen3-ASR"
_CIF_TRAIN = Path(__file__).resolve().parents[1] / "utils" / "cif_train"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CIF_TRAIN))

from train_one_seg import load_audio_16k, load_encoder

_TEMPORAL_ARCHS = {"cnn1d", "gru"}


# ---------------------------------------------------------------------------
# Temporal probe models
# ---------------------------------------------------------------------------

class CNN1DProbe(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 64,
                 kernel: int = 7, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, ch = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Conv1d(ch, hidden, kernel, padding=kernel // 2),
                       nn.ReLU(), nn.Dropout(dropout)]
            ch = hidden
        layers.append(nn.Conv1d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, D) → (B, T)
        return self.net(x.transpose(1, 2)).squeeze(1)


class GRUProbe(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, n_layers,
                           bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, D) → (B, T)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


class TorchProbeWrapper:
    """sklearn predict_proba 인터페이스를 흉내내는 PyTorch 모델 래퍼."""
    def __init__(self, model: nn.Module, device: str):
        self.model, self.device = model, device

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:  # (T, D) → (T, 2)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
            p = torch.sigmoid(self.model(x)).squeeze(0).cpu().numpy()
        return np.stack([1 - p, p], axis=1)


def train_temporal_probe(clips_tr: list, arch: str, in_dim: int,
                          pos_window: int, device: str,
                          n_epochs: int = 15, lr: float = 1e-3) -> TorchProbeWrapper:
    model = (CNN1DProbe(in_dim=in_dim) if arch == "cnn1d"
             else GRUProbe(in_dim=in_dim)).to(device)

    n_pos = n_neg = 0
    for c in clips_tr:
        T = c["T"]
        lbl = np.zeros(T, dtype=np.int32)
        for sf in c["seg_frames"]:
            for df in range(-pos_window, pos_window + 1):
                f = sf + df
                if 0 <= f < T:
                    lbl[f] = 1
        n_pos += int(lbl.sum())
        n_neg += T - int(lbl.sum())

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt        = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  training {arch}  epochs={n_epochs}  lr={lr}")
    print(f"    pos={n_pos}  neg={n_neg}  ratio=1:{n_neg // max(n_pos, 1)}")

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for c in clips_tr:
            T   = c["T"]
            lbl = np.zeros(T, dtype=np.float32)
            for sf in c["seg_frames"]:
                for df in range(-pos_window, pos_window + 1):
                    f = sf + df
                    if 0 <= f < T:
                        lbl[f] = 1.0
            x = torch.tensor(c["feats"], dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(lbl).unsqueeze(0).to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}/{n_epochs}  loss={total / len(clips_tr):.4f}")

    return TorchProbeWrapper(model, device)


# ---------------------------------------------------------------------------
# Encoder inference
# ---------------------------------------------------------------------------

def run_encoder(audio: np.ndarray, encoder, fe, device, dtype):
    out = fe(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]
    with torch.no_grad():
        enc_out = encoder(
            mel.to(device=device, dtype=dtype),
            feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
        )
    feats = enc_out.last_hidden_state.squeeze(0).float().cpu().numpy()
    dur   = len(audio) / 16000.0
    return feats, dur


# ---------------------------------------------------------------------------
# Activation cache
# ---------------------------------------------------------------------------

def _cache_paths(cache_dir: Path, data_path: str, pos_window: int):
    key = Path(data_path).stem + f"_pw{pos_window}"
    return (
        cache_dir / f"{key}_X.npy",
        cache_dir / f"{key}_y.npy",
        cache_dir / f"{key}_clips.pkl",
    )


def save_cache(cache_dir: Path, data_path: str, pos_window: int,
               X: np.ndarray, y: np.ndarray, clips: list):
    cache_dir.mkdir(parents=True, exist_ok=True)
    px, py, pc = _cache_paths(cache_dir, data_path, pos_window)
    np.save(px, X)
    np.save(py, y)
    with open(pc, "wb") as f:
        pickle.dump(clips, f)
    print(f"  cache saved: {px.name}, {py.name}, {pc.name}")


def load_cache(cache_dir: Path, data_path: str, pos_window: int):
    """캐시가 있으면 (X, y, clips) 반환, 없으면 None."""
    px, py, pc = _cache_paths(cache_dir, data_path, pos_window)
    if not (px.exists() and py.exists() and pc.exists()):
        return None
    print(f"  loading cache: {px.name} …")
    X = np.load(px)
    y = np.load(py)
    with open(pc, "rb") as f:
        clips = pickle.load(f)
    print(f"  X={X.shape}  clips={len(clips)}")
    return X, y, clips


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def _seg_frames(rec: dict, T: int, dur: float) -> list[int]:
    """레코드에서 seg_frame 목록 추출 (one_seg / multi_seg 자동 감지)."""
    if "seg_timestamps" in rec:
        return [min(max(round(t * T / dur), 0), T - 1)
                for t in rec["seg_timestamps"]]
    else:
        seg_t = rec["seg_t"]
        return [min(max(round(seg_t * T / dur), 0), T - 1)]


def collect(records, encoder, fe, device, dtype,
            pos_window: int, max_samples: int | None,
            desc: str = "collect"):
    """
    프레임별 (feature, label) 쌍 + 클립 단위 정보 반환.
    one_seg / multi_seg 포맷 자동 감지.

    반환:
      X      (N_frames, D)  float32
      y      (N_frames,)    int  (1=pos, 0=neg)
      clips  list[dict]     클립별 feats / seg_frames / dur
    """
    if max_samples:
        records = records[:max_samples]

    X_list, y_list, clips = [], [], []

    for rec in tqdm(records, desc=desc, unit="sample"):
        try:
            start = rec.get("start", 0.0)
            end   = rec.get("end")
            audio = load_audio_16k(rec["audio"], start=start, end=end)
            feats, dur = run_encoder(audio, encoder, fe, device, dtype)
            T = feats.shape[0]

            seg_frames = _seg_frames(rec, T, dur)

            labels = np.zeros(T, dtype=np.int32)
            for sf in seg_frames:
                for df in range(-pos_window, pos_window + 1):
                    f = sf + df
                    if 0 <= f < T:
                        labels[f] = 1

            X_list.append(feats)
            y_list.append(labels)
            clips.append({"feats": feats, "seg_frames": seg_frames,
                          "dur": dur, "T": T})
        except Exception as e:
            tqdm.write(f"  SKIP: {e}")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)
    return X, y, clips


# ---------------------------------------------------------------------------
# Probe training & evaluation
# ---------------------------------------------------------------------------

def train_probe(X_tr: np.ndarray, y_tr: np.ndarray) -> LogisticRegression:
    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    print(f"  training LogisticRegression")
    print(f"    X={X_tr.shape}  pos={n_pos}  neg={n_neg}  ratio=1:{n_neg//max(n_pos,1)}")
    clf = LogisticRegression(
        max_iter=100,
        class_weight="balanced",
        solver="saga",   # epoch별 loss 출력 지원
        C=1.0,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_tr, y_tr)
    print(f"  converged  n_iter={clf.n_iter_[0]}")
    return clf


def eval_frame_level(clf, X_te, y_te: np.ndarray,
                     proba: np.ndarray | None = None):
    """프레임 단위 이진 분류 지표.
    temporal 모델은 proba를 미리 concat해서 넘겨야 함 (X_te 무시)."""
    if proba is None:
        proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    bal_acc = balanced_accuracy_score(y_te, pred)
    f1      = f1_score(y_te, pred, zero_division=0)
    auc     = roc_auc_score(y_te, proba)

    # precision-recall curve에서 F1 최대 threshold
    prec, rec, thr = precision_recall_curve(y_te, proba)
    f1_curve = 2 * prec * rec / (prec + rec + 1e-8)
    best_i   = np.argmax(f1_curve)
    best_thr = thr[best_i] if best_i < len(thr) else 0.5

    return {
        "balanced_acc": bal_acc,
        "f1@0.5": f1,
        "auc": auc,
        "best_thr": best_thr,
        "best_f1": f1_curve[best_i],
        "best_prec": prec[best_i],
        "best_rec": rec[best_i],
    }


def _detect_peaks(proba: np.ndarray, thr: float, method: str) -> np.ndarray:
    """proba 시계열에서 SEG 후보 프레임 인덱스 반환."""
    if method == "run":
        from scipy.ndimage import label as ndlabel
        binary         = (proba >= thr).astype(np.int32)
        labeled, n_run = ndlabel(binary)
        if n_run == 0:
            return np.array([], dtype=np.int64)
        return np.array([int(np.argmax(proba * (labeled == i)))
                         for i in range(1, n_run + 1)], dtype=np.int64)
    else:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(proba, distance=3, height=thr)
        return peaks


def eval_clip_level(clf, clips: list, pos_window: int, peak_thr: float = 0.5,
                    peak_method: str = "peaks"):
    """
    클립 단위 검출 평가.
    - one_seg  : argmax → pos_mae, pos_acc@2f/@5f
    - multi_seg: peak_method 방식으로 경계 검출 → count_acc, count_mae, pos_mae
        peaks : scipy.find_peaks(distance=3, height=thr)
        run   : proba >= thr 연속 구간별 argmax
    """
    from scipy.signal import find_peaks

    is_multi = any(len(c["seg_frames"]) > 1 for c in clips)

    pos_mae_sum = 0.0
    pos_acc_2f = pos_acc_5f = 0
    count_acc = count_mae_sum = 0
    over_n = under_n = 0   # 과검출 / 미검출 카운트
    pos_n = n = 0
    enc_frame_dur_last = 0.08

    by_nsegs: dict[int, dict] = {}

    for clip in clips:
        feats      = clip["feats"]
        seg_frames = clip["seg_frames"]
        dur        = clip["dur"]
        T          = clip["T"]
        n_gt       = len(seg_frames)

        proba = clf.predict_proba(feats)[:, 1]
        enc_frame_dur_last = dur / max(T, 1)

        if not is_multi:
            pred_frame = int(np.argmax(proba))
            err = abs(pred_frame - seg_frames[0])
            pos_mae_sum += err
            pos_acc_2f  += int(err <= 2)
            pos_acc_5f  += int(err <= 5)
        else:
            peaks  = _detect_peaks(proba, peak_thr, peak_method)
            n_pred = len(peaks)

            diff = n_pred - n_gt
            count_acc     += int(diff == 0)
            count_mae_sum += abs(diff)
            if diff > 0:
                over_n  += 1
            elif diff < 0:
                under_n += 1

            bucket = by_nsegs.setdefault(n_gt, {"n": 0, "acc": 0, "mae": 0,
                                                  "over": 0, "under": 0})
            bucket["n"]     += 1
            bucket["acc"]   += int(diff == 0)
            bucket["mae"]   += abs(diff)
            bucket["over"]  += int(diff > 0)
            bucket["under"] += int(diff < 0)

            if n_pred == n_gt and n_gt > 0:
                errs = [abs(p - g) for p, g in
                        zip(sorted(peaks.tolist()), sorted(seg_frames))]
                pos_mae_sum += float(np.mean(errs))
                pos_n += 1

        n += 1

    enc_ms = enc_frame_dur_last * 1000
    result = {"n": n, "pos_mae_f": pos_mae_sum / max(pos_n if is_multi else n, 1),
              "pos_mae_ms": pos_mae_sum / max(pos_n if is_multi else n, 1) * enc_ms}

    if is_multi:
        result["count_acc"]  = count_acc  / n
        result["count_mae"]  = count_mae_sum / n
        result["over_rate"]  = over_n  / n   # 과검출 비율
        result["under_rate"] = under_n / n   # 미검출 비율
        result["by_nsegs"]   = by_nsegs
    else:
        result["pos_acc_2f"] = pos_acc_2f / n
        result["pos_acc_5f"] = pos_acc_5f / n

    return result


# ---------------------------------------------------------------------------
# Plots
def sweep_peak_thr(clf, clips: list, thresholds=None, out_path=None,
                   peak_method: str = "peaks"):
    """threshold 범위를 sweep해서 count_acc / pos_mae 변화를 시각화."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    count_accs, pos_maes, count_maes = [], [], []

    for thr in thresholds:
        r = eval_clip_level(clf, clips, pos_window=2, peak_thr=float(thr),
                            peak_method=peak_method)
        count_accs.append(r.get("count_acc", 0))
        count_maes.append(r.get("count_mae", 0))
        pos_maes.append(r["pos_mae_f"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(thresholds, count_accs, marker="o", ms=4, lw=1.2, label="count_acc")
    axes[0].plot(thresholds, [1 - m for m in count_maes], marker="s", ms=4,
                 lw=1.0, ls="--", label="1 - count_mae (ref)")
    best_i = int(np.argmax(count_accs))
    axes[0].axvline(thresholds[best_i], color="red", ls="--",
                    label=f"best thr={thresholds[best_i]:.2f}  acc={count_accs[best_i]:.3f}")
    axes[0].set_xlabel("peak threshold")
    axes[0].set_ylabel("count_acc")
    axes[0].set_title(f"count_acc vs peak threshold  [{peak_method}]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(thresholds, pos_maes, marker="o", ms=4, lw=1.2, color="steelblue")
    axes[1].set_xlabel("peak threshold")
    axes[1].set_ylabel("pos_mae (frames)")
    axes[1].set_title("pos_mae vs peak threshold  (count 일치 샘플)")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Peak threshold sweep", fontsize=11)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path}")

    best_thr = float(thresholds[best_i])
    print(f"  best peak_thr={best_thr:.2f}  "
          f"count_acc={count_accs[best_i]:.3f}  "
          f"pos_mae={pos_maes[best_i]:.2f}f")
    return best_thr


# ---------------------------------------------------------------------------

def plot_pr_curve(clf, X_te_or_clips, y_te, out_path, is_temporal: bool = False):
    if is_temporal:
        proba = np.concatenate([clf.predict_proba(c["feats"])[:, 1]
                                for c in X_te_or_clips])
    else:
        proba = clf.predict_proba(X_te_or_clips)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, proba)
    f1_curve = 2 * prec * rec / (prec + rec + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rec, prec, lw=1.2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall curve")
    axes[0].grid(alpha=0.3)

    axes[1].plot(thr, f1_curve[:-1], lw=1.2)
    best_i = np.argmax(f1_curve[:-1])
    axes[1].axvline(thr[best_i], color="red", ls="--",
                    label=f"best thr={thr[best_i]:.3f}  F1={f1_curve[best_i]:.3f}")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs threshold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def plot_clip_examples(clf, clips, n_show: int = 6, out_path=None):
    """클립별 P(SEG|frame) 시계열 + GT/예측 위치 (one_seg / multi_seg 자동 대응)."""
    from scipy.signal import find_peaks
    is_multi = any(len(c["seg_frames"]) > 1 for c in clips)
    samples  = clips[:n_show]

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show))
    if n_show == 1:
        axes = [axes]

    for i, clip in enumerate(samples):
        feats      = clip["feats"]
        seg_frames = clip["seg_frames"]
        proba      = clf.predict_proba(feats)[:, 1]
        n_gt       = len(seg_frames)

        if is_multi:
            peaks, _ = find_peaks(proba, distance=3)
            if len(peaks) == 0:
                peaks = np.array([int(np.argmax(proba))])
            if len(peaks) > n_gt:
                top_idx = np.argsort(proba[peaks])[::-1][:n_gt]
                peaks = np.sort(peaks[top_idx])
            pred_frames = peaks.tolist()
        else:
            pred_frames = [int(np.argmax(proba))]

        ax = axes[i]
        ax.plot(proba, lw=0.9, color="steelblue", label="P(SEG|frame)")
        for sf in seg_frames:
            ax.axvline(sf, color="red", lw=1.5, alpha=0.8)
        for j, pf in enumerate(pred_frames):
            nearest_err = min(abs(pf - sf) for sf in seg_frames)
            ax.axvline(pf, color="orange", lw=1.5, ls="--",
                       label=f"pred={pf} err={nearest_err}" if j == 0 else "")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"s{i}", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

    plt.suptitle("Linear probe: P(SEG|frame)  red=GT  orange=pred", fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path}")


def plot_error_hist(clf, clips, out_path=None):
    """nearest-GT 오차 분포 (one_seg / multi_seg 공용)."""
    from scipy.signal import find_peaks
    is_multi = any(len(c["seg_frames"]) > 1 for c in clips)
    errors = []

    for clip in clips:
        proba      = clf.predict_proba(clip["feats"])[:, 1]
        seg_frames = clip["seg_frames"]
        n_gt       = len(seg_frames)

        if is_multi:
            peaks, _ = find_peaks(proba, distance=3)
            if len(peaks) == 0:
                peaks = np.array([int(np.argmax(proba))])
            if len(peaks) > n_gt:
                top_idx = np.argsort(proba[peaks])[::-1][:n_gt]
                peaks = np.sort(peaks[top_idx])
            if len(peaks) == n_gt:
                for p, g in zip(sorted(peaks), sorted(seg_frames)):
                    errors.append(abs(p - g))
        else:
            pred = int(np.argmax(proba))
            errors.append(abs(pred - seg_frames[0]))

    if not errors:
        return
    errors = np.array(errors)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=range(0, min(int(errors.max()) + 2, 50)),
            edgecolor="white", color="steelblue", alpha=0.85)
    ax.axvline(errors.mean(), color="red", ls="--",
               label=f"mean={errors.mean():.2f}f  ≈{errors.mean()*80:.0f}ms")
    ax.set_xlabel("|pred - GT|  (frames)")
    ax.set_ylabel("count")
    ax.set_title("Detection error distribution (nearest GT per pred)")
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _data = Path(__file__).resolve().parent / "data"
    _tail = Path(__file__).resolve().parent / "one_seg_data_tail"

    parser = argparse.ArgumentParser()
    parser.add_argument("--one-seg",    action="store_true",
                        help="one_seg_data_tail/ 사용 (기본: data/ multi-SEG)")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--test-data",  default=None)
    parser.add_argument("--model",      default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train",  type=int, default=None)
    parser.add_argument("--max-test",   type=int, default=None)
    parser.add_argument("--pos-window", type=int, default=2,
                        help="SEG 근방 ±N 프레임을 pos로 간주")
    parser.add_argument("--top-k",      type=int, default=None,
                        help="discriminability 상위 K 차원만 사용 (None=전체 2048)")
    parser.add_argument("--disc-path",  default="research/cif/probe_results/disc.npy",
                        help="--top-k 사용 시 disc.npy 경로")
    parser.add_argument("--out-dir",    default="research/cif/probe_results")
    parser.add_argument("--peak-thr",    type=float, default=None,
                        help="peak threshold (None=sweep해서 자동 결정)")
    parser.add_argument("--peak-method", default="peaks",
                        choices=["peaks", "run"],
                        help="경계 검출 방식: peaks=find_peaks, run=연속구간 argmax")
    parser.add_argument("--no-save-probe", action="store_true",
                        help="probe 저장 안 함 (기본: 저장)")
    parser.add_argument("--load-probe",  default=None,
                        help="저장된 probe pkl 경로 — 학습 건너뜀 (eval/plot만 실행)")
    parser.add_argument("--cache-dir",  default=None,
                        help="activation 캐시 디렉토리 (None=캐시 사용 안 함)")
    parser.add_argument("--no-cache",   action="store_true",
                        help="캐시 무시하고 encoder 재실행")
    parser.add_argument("--arch",       default="logistic",
                        choices=["logistic", "cnn1d", "gru"],
                        help="분류기 아키텍처 (기본: logistic)")
    parser.add_argument("--epochs",     type=int, default=15,
                        help="temporal 모델 학습 epoch 수 (기본: 15)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="temporal 모델 learning rate (기본: 1e-3)")
    args = parser.parse_args()

    # 데이터 경로 결정
    if args.train_data is None:
        args.train_data = str(_tail / "train.jsonl") if args.one_seg \
                          else str(_data / "train.jsonl")
    if args.test_data is None:
        args.test_data  = str(_tail / "test.jsonl") if args.one_seg \
                          else str(_data / "test.jsonl")

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir \
                else out_dir / "cache"

    # ── collect (캐시 우선) ──
    with open(args.train_data) as f:
        train_recs = [json.loads(l) for l in f if l.strip()]
    with open(args.test_data) as f:
        test_recs  = [json.loads(l) for l in f if l.strip()]
    print(f"  train: {args.train_data}  ({len(train_recs)} samples)")
    print(f"  test:  {args.test_data}  ({len(test_recs)} samples)\n")

    tr_cached = None if args.no_cache else load_cache(cache_dir, args.train_data, args.pos_window)
    te_cached = None if args.no_cache else load_cache(cache_dir, args.test_data,  args.pos_window)

    encoder = fe = None  # encoder는 캐시 miss 시에만 로드

    if tr_cached is None or te_cached is None:
        dtype = torch.bfloat16
        print(f"Loading encoder …")
        encoder = load_encoder(args.model, args.device, dtype)
        fe = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=16000, hop_length=160,
            chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
        )

    if tr_cached is None:
        X_tr, y_tr, train_clips = collect(train_recs, encoder, fe, args.device, dtype,
                                           args.pos_window, args.max_train, "train")
        save_cache(cache_dir, args.train_data, args.pos_window, X_tr, y_tr, train_clips)
    else:
        X_tr, y_tr, train_clips = tr_cached
        if args.arch in _TEMPORAL_ARCHS and len(train_clips) == 0:
            print("  [ERROR] 캐시된 train clips가 없습니다 (구버전 캐시).")
            print("  --no-cache 옵션으로 재실행하세요.")
            sys.exit(1)

    if te_cached is None:
        X_te, y_te, test_clips = collect(test_recs, encoder, fe, args.device, dtype,
                                          args.pos_window, args.max_test, "test")
        save_cache(cache_dir, args.test_data, args.pos_window, X_te, y_te, test_clips)
    else:
        X_te, y_te, test_clips = te_cached

    # encoder 메모리 해제
    if encoder is not None:
        del encoder
        torch.cuda.empty_cache()

    # ── dim selection ──
    dim_mask = None
    if args.top_k is not None:
        disc    = np.load(args.disc_path)
        top_idx = np.argsort(disc)[::-1][:args.top_k]
        dim_mask = top_idx
        print(f"  using top-{args.top_k} dims from {args.disc_path}")

    if dim_mask is not None:
        X_tr = X_tr[:, dim_mask]
        X_te = X_te[:, dim_mask]
        for c in test_clips:
            c["feats"] = c["feats"][:, dim_mask]
        for c in train_clips:
            c["feats"] = c["feats"][:, dim_mask]

    # ── train (or load) ──
    if args.load_probe:
        with open(args.load_probe, "rb") as f:
            saved = pickle.load(f)
        clf      = saved["clf"]
        dim_mask = saved.get("dim_mask", dim_mask)
        print(f"  probe loaded: {args.load_probe}  arch={saved.get('arch', '?')}")
    elif args.arch in _TEMPORAL_ARCHS:
        clf = train_temporal_probe(
            train_clips, args.arch, X_tr.shape[1],
            args.pos_window, args.device, args.epochs, args.lr,
        )
    else:
        clf = train_probe(X_tr, y_tr)

    is_temporal = args.arch in _TEMPORAL_ARCHS

    # ── eval ──
    if is_temporal:
        proba_te = np.concatenate([clf.predict_proba(c["feats"])[:, 1]
                                   for c in test_clips])
        frame_res = eval_frame_level(clf, None, y_te, proba=proba_te)
    else:
        frame_res = eval_frame_level(clf, X_te, y_te)
    is_multi  = any(len(c["seg_frames"]) > 1 for c in test_clips)

    # threshold sweep → best_thr 결정
    best_peak_thr = args.peak_thr
    if is_multi and args.peak_thr is None:
        print("\nSweeping peak threshold …")
        best_peak_thr = sweep_peak_thr(
            clf, test_clips,
            out_path=out_dir / "probe_thr_sweep.png",
            peak_method=args.peak_method,
        )
    elif best_peak_thr is None:
        best_peak_thr = 0.5

    clip_res = eval_clip_level(clf, test_clips, args.pos_window,
                               peak_thr=best_peak_thr,
                               peak_method=args.peak_method)

    dims_used = args.top_k if args.top_k else X_tr.shape[1]
    print(f"\n{'='*56}")
    print(f"  [{args.arch}]  dims={dims_used}  pos_window=±{args.pos_window}f"
          + (f"  peak_thr={best_peak_thr:.2f}" if is_multi else ""))
    print(f"{'─'*56}")
    print(f"  [Frame-level]")
    print(f"    balanced_acc : {frame_res['balanced_acc']:.3f}")
    print(f"    AUC          : {frame_res['auc']:.3f}")
    print(f"    F1 @0.5      : {frame_res['f1@0.5']:.3f}")
    print(f"    best F1      : {frame_res['best_f1']:.3f}  "
          f"(thr={frame_res['best_thr']:.3f}  "
          f"P={frame_res['best_prec']:.3f}  R={frame_res['best_rec']:.3f})")
    print(f"{'─'*56}")
    if is_multi:
        print(f"  [Clip-level {args.peak_method}  (multi-SEG, thr={best_peak_thr:.2f})]")
        print(f"    count_acc    : {clip_res['count_acc']:.3f}")
        print(f"    count_mae    : {clip_res['count_mae']:.3f}")
        print(f"    over_rate    : {clip_res['over_rate']:.3f}  (과검출)")
        print(f"    under_rate   : {clip_res['under_rate']:.3f}  (미검출)")
        print(f"    pos_mae      : {clip_res['pos_mae_f']:.2f} f  "
              f"≈ {clip_res['pos_mae_ms']:.0f} ms  (count 일치 샘플)")
        print(f"    n_clips      : {clip_res['n']}")
        if clip_res.get("by_nsegs"):
            print(f"\n  [ n_segs별 count_acc / over / under ]")
            for k in sorted(clip_res["by_nsegs"]):
                b = clip_res["by_nsegs"][k]
                print(f"    n_segs={k}  n={b['n']:4d}  "
                      f"acc={b['acc']/b['n']:.3f}  "
                      f"over={b['over']/b['n']:.3f}  "
                      f"under={b['under']/b['n']:.3f}  "
                      f"mae={b['mae']/b['n']:.2f}")
    else:
        print(f"  [Clip-level argmax detection  (one-SEG)]")
        print(f"    pos_mae      : {clip_res['pos_mae_f']:.2f} f  "
              f"≈ {clip_res['pos_mae_ms']:.0f} ms")
        print(f"    pos_acc @2f  : {clip_res['pos_acc_2f']:.3f}")
        print(f"    pos_acc @5f  : {clip_res['pos_acc_5f']:.3f}")
        print(f"    n_clips      : {clip_res['n']}")
    print(f"{'='*56}")

    # ── plots ──
    print("\nPlotting …")
    plot_pr_curve(clf, X_te if not is_temporal else test_clips, y_te,
                  out_path=out_dir / "probe_pr_curve.png",
                  is_temporal=is_temporal)
    plot_clip_examples(clf, test_clips, n_show=6,
                       out_path=out_dir / "probe_clip_examples.png")
    plot_error_hist(clf, test_clips,
                    out_path=out_dir / "probe_error_hist.png")

    if not args.no_save_probe:
        probe_path = out_dir / f"probe_{args.arch}.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump({"clf": clf, "dim_mask": dim_mask,
                         "peak_thr": best_peak_thr, "arch": args.arch}, f)
        print(f"  probe saved: {probe_path}")

    print("Done.")


if __name__ == "__main__":
    main()
