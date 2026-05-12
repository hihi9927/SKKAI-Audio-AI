"""
train_causal.py

Dialog 단위 스트리밍 causal SEG 검출기 학습.

각 dialog를 하나의 연속 스트리밍 시퀀스로 구성하고,
단방향 (causal) 모델을 학습해 SEG 경계를 프레임별로 trigger.

실행:
  python research/cif/train_causal.py --arch gru
  python research/cif/train_causal.py --arch cnn1d --epochs 20 --lr 5e-4
  python research/cif/train_causal.py --load-model research/cif/causal_results/causal_gru.pkl
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndlabel
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              roc_auc_score, precision_recall_curve)
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from train_one_seg import load_audio_16k, load_encoder


# ---------------------------------------------------------------------------
# Dialog data construction
# ---------------------------------------------------------------------------

def parse_dialog_info(audio_path: str) -> tuple[int, str]:
    """(turn_idx, dialog_id) 반환. 파일명: '{turn}_{spk}_d{did}.wav'"""
    stem  = Path(audio_path).stem                   # e.g. '3_0_d5'
    parts = stem.rsplit("_d", maxsplit=1)            # ['3_0', '5']
    return int(stem.split("_")[0]), parts[-1]


def group_by_dialog(records: list) -> dict[str, list]:
    dialogs: dict[str, list] = defaultdict(list)
    for r in records:
        _, did = parse_dialog_info(r["audio"])
        dialogs[did].append(r)
    for did in dialogs:
        dialogs[did].sort(key=lambda r: parse_dialog_info(r["audio"])[0])
    return dict(dialogs)


def build_dialog_feats(turns: list, encoder, fe, device, dtype,
                       pos_window: int, max_sec: float = 28.0) -> dict:
    """
    Turn 목록을 이어붙여 하나의 스트리밍 시퀀스로 인코딩.
    30초 제한으로 잘리는 걸 방지하기 위해 max_sec 단위로 chunk하여 인코딩 후 concat.
    """
    audio_chunks, seg_times = [], []
    offset = 0.0
    for r in turns:
        audio = load_audio_16k(r["audio"])
        dur   = len(audio) / 16000.0
        for t in r["seg_timestamps"]:
            seg_times.append(offset + t)
        audio_chunks.append(audio)
        offset += dur

    full_audio = np.concatenate(audio_chunks)
    total_dur  = offset

    # 30초 초과 시 chunk 단위로 인코딩 후 concat
    max_samples = int(max_sec * 16000)
    if len(full_audio) <= max_samples:
        feats = _encode_chunk(full_audio, encoder, fe, device, dtype)
    else:
        parts = []
        for i in range(0, len(full_audio), max_samples):
            chunk = full_audio[i: i + max_samples]
            parts.append(_encode_chunk(chunk, encoder, fe, device, dtype))
        feats = np.concatenate(parts, axis=0)

    T = feats.shape[0]

    seg_frames = [min(max(round(t * T / total_dur), 0), T - 1)
                  for t in seg_times]

    labels = np.zeros(T, dtype=np.float32)
    for sf in seg_frames:
        for df in range(-pos_window, pos_window + 1):
            f = sf + df
            if 0 <= f < T:
                labels[f] = 1.0

    return {"feats": feats, "labels": labels,
            "seg_frames": seg_frames, "dur": total_dur, "T": T}


def _encode_chunk(audio: np.ndarray, encoder, fe, device, dtype) -> np.ndarray:
    out     = fe(audio, sampling_rate=16000,
                 return_tensors="pt", return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]
    with torch.no_grad():
        enc_out = encoder(
            mel.to(device=device, dtype=dtype),
            feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
        )
    return enc_out.last_hidden_state.squeeze(0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, data_path: str, pos_window: int) -> Path:
    key = Path(data_path).stem + f"_pw{pos_window}_causal"
    return cache_dir / f"{key}.pkl"


def save_cache(cache_dir: Path, data_path: str, pos_window: int, dialogs: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, data_path, pos_window)
    with open(path, "wb") as f:
        pickle.dump(dialogs, f)
    print(f"  cache saved: {path.name}  ({len(dialogs)} dialogs)")


def load_cache(cache_dir: Path, data_path: str, pos_window: int) -> dict | None:
    path = _cache_path(cache_dir, data_path, pos_window)
    if not path.exists():
        return None
    print(f"  loading cache: {path.name} …")
    with open(path, "rb") as f:
        dialogs = pickle.load(f)
    print(f"  {len(dialogs)} dialogs loaded")
    return dialogs


# ---------------------------------------------------------------------------
# Causal models
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.pad  = kernel - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalCNN1DProbe(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 64,
                 kernel: int = 7, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, ch = [], in_dim
        for _ in range(n_layers):
            layers += [CausalConv1d(ch, hidden, kernel),
                       nn.ReLU(), nn.Dropout(dropout)]
            ch = hidden
        layers.append(nn.Conv1d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h=None):   # (B, T, D) → (B, T), None
        return self.net(x.transpose(1, 2)).squeeze(1), None


class CausalGRUProbe(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 128,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, h=None):   # (B, T, D) → (B, T), h_n
        out, h_n = self.gru(x, h)
        return self.head(out).squeeze(-1), h_n


class CausalWrapper:
    """predict_proba(feats) → (T, 2) — 평가·시각화용 stateless 인터페이스."""
    def __init__(self, model: nn.Module, device: str):
        self.model, self.device = model, device

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:  # (T, D) → (T, 2)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, _ = self.model(x)
            p = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return np.stack([1 - p, p], axis=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_causal(dialogs_tr: list, arch: str, in_dim: int, pos_window: int,
                 device: str, n_epochs: int = 20, lr: float = 1e-3,
                 init_model: nn.Module | None = None,
                 dialogs_val: list | None = None,
                 patience: int = 10) -> CausalWrapper:
    if init_model is not None:
        model = init_model.to(device)
        print(f"  resuming from existing weights")
    else:
        model = (CausalGRUProbe(in_dim=in_dim) if arch == "gru"
                 else CausalCNN1DProbe(in_dim=in_dim)).to(device)

    n_pos = sum(int(d["labels"].sum()) for d in dialogs_tr)
    n_neg = sum(d["T"] - int(d["labels"].sum()) for d in dialogs_tr)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                              dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt        = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  training causal-{arch}  epochs={n_epochs}  lr={lr}  patience={patience}")
    print(f"    dialogs={len(dialogs_tr)}  pos={n_pos}  neg={n_neg}  "
          f"ratio=1:{n_neg // max(n_pos, 1)}")

    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0

    for epoch in range(n_epochs):
        model.train()
        tr_loss = 0.0
        for d in dialogs_tr:
            x = torch.tensor(d["feats"],  dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(d["labels"], dtype=torch.float32).unsqueeze(0).to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(dialogs_tr)

        # ── validation loss & early stopping ──
        if dialogs_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for d in dialogs_val:
                    x = torch.tensor(d["feats"],  dtype=torch.float32).unsqueeze(0).to(device)
                    y = torch.tensor(d["labels"], dtype=torch.float32).unsqueeze(0).to(device)
                    logits, _ = model(x)
                    val_loss += criterion(logits, y).item()
            val_loss /= len(dialogs_val)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1:3d}/{n_epochs}  "
                      f"tr={tr_loss:.4f}  val={val_loss:.4f}"
                      + ("  *" if val_loss < best_val_loss else ""))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    early stopping at epoch {epoch + 1}  "
                          f"best_val={best_val_loss:.4f}")
                    break
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1:3d}/{n_epochs}  loss={tr_loss:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  best weights restored  val_loss={best_val_loss:.4f}")

    return CausalWrapper(model, device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _detect_run(proba: np.ndarray, thr: float) -> np.ndarray:
    binary         = (proba >= thr).astype(np.int32)
    labeled, n_run = ndlabel(binary)
    if n_run == 0:
        return np.array([], dtype=np.int64)
    return np.array([int(np.argmax(proba * (labeled == i)))
                     for i in range(1, n_run + 1)], dtype=np.int64)


def eval_frame_level(proba: np.ndarray, y: np.ndarray) -> dict:
    pred    = (proba >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y, pred)
    f1      = f1_score(y, pred, zero_division=0)
    auc     = roc_auc_score(y, proba)
    prec, rec, thr = precision_recall_curve(y, proba)
    f1c = 2 * prec * rec / (prec + rec + 1e-8)
    bi  = np.argmax(f1c)
    return {"balanced_acc": bal_acc, "f1@0.5": f1, "auc": auc,
            "best_thr": float(thr[bi]) if bi < len(thr) else 0.5,
            "best_f1": f1c[bi], "best_prec": prec[bi], "best_rec": rec[bi]}


def eval_streaming(clf: CausalWrapper, dialogs: list,
                   pos_window: int, peak_thr: float) -> dict:
    """
    각 dialog 전체를 causal 추론 후 run detection으로 trigger 찾고
    GT SEG frames와 비교.
    """
    count_acc = over_n = under_n = 0
    count_mae_sum = pos_mae_sum = 0.0
    total_dur = total_T = pos_n = n = 0

    by_nsegs: dict[int, dict] = {}

    for d in dialogs:
        proba      = clf.predict_proba(d["feats"])[:, 1]
        seg_frames = d["seg_frames"]
        T          = d["T"]
        n_gt       = len(seg_frames)
        total_dur += d["dur"]
        total_T   += T

        peaks  = _detect_run(proba, peak_thr)
        n_pred = len(peaks)
        diff   = n_pred - n_gt

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

    avg_frame_ms = (total_dur / max(total_T, 1)) * 1000
    return {
        "n":           n,
        "count_acc":   count_acc    / n,
        "count_mae":   count_mae_sum / n,
        "over_rate":   over_n  / n,
        "under_rate":  under_n / n,
        "pos_mae_f":   pos_mae_sum / max(pos_n, 1),
        "pos_mae_ms":  pos_mae_sum / max(pos_n, 1) * avg_frame_ms,
        "by_nsegs":    by_nsegs,
    }


def sweep_threshold(clf: CausalWrapper, dialogs: list,
                    pos_window: int, out_path=None) -> float:
    thresholds = np.arange(0.1, 0.95, 0.05)
    count_accs, pos_maes = [], []

    for thr in thresholds:
        r = eval_streaming(clf, dialogs, pos_window, float(thr))
        count_accs.append(r["count_acc"])
        pos_maes.append(r["pos_mae_f"])

    best_i   = int(np.argmax(count_accs))
    best_thr = float(thresholds[best_i])

    if out_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(thresholds, count_accs, marker="o", ms=4, lw=1.2)
        axes[0].axvline(best_thr, color="red", ls="--",
                        label=f"best={best_thr:.2f}  acc={count_accs[best_i]:.3f}")
        axes[0].set_xlabel("threshold"); axes[0].set_ylabel("count_acc")
        axes[0].set_title("count_acc vs threshold  [causal / run]")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        axes[1].plot(thresholds, pos_maes, marker="o", ms=4, lw=1.2, color="steelblue")
        axes[1].set_xlabel("threshold"); axes[1].set_ylabel("pos_mae (frames)")
        axes[1].set_title("pos_mae vs threshold  (count 일치 dialog)")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  saved: {out_path}")

    print(f"  best thr={best_thr:.2f}  count_acc={count_accs[best_i]:.3f}  "
          f"pos_mae={pos_maes[best_i]:.2f}f")
    return best_thr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pr_curve(proba_all: np.ndarray, y_all: np.ndarray, out_path):
    prec, rec, thr = precision_recall_curve(y_all, proba_all)
    f1c = 2 * prec * rec / (prec + rec + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rec, prec, lw=1.2)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall  [causal]"); axes[0].grid(alpha=0.3)

    axes[1].plot(thr, f1c[:-1], lw=1.2)
    bi = np.argmax(f1c[:-1])
    axes[1].axvline(thr[bi], color="red", ls="--",
                    label=f"best thr={thr[bi]:.3f}  F1={f1c[bi]:.3f}")
    axes[1].set_xlabel("threshold"); axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs threshold"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


def plot_dialog_examples(clf: CausalWrapper, dialogs: list,
                         peak_thr: float, n_show: int = 4, out_path=None):
    n_show = min(n_show, len(dialogs))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    for i, d in enumerate(dialogs[:n_show]):
        proba      = clf.predict_proba(d["feats"])[:, 1]
        seg_frames = d["seg_frames"]
        peaks      = _detect_run(proba, peak_thr)

        ax = axes[i]
        ax.plot(proba, lw=0.8, color="steelblue", label="P(SEG|frame)")
        for sf in seg_frames:
            ax.axvline(sf, color="red",    lw=1.2, alpha=0.8, label="GT" if sf == seg_frames[0] else "")
        for j, pf in enumerate(peaks):
            ax.axvline(pf, color="orange", lw=1.2, ls="--", label="pred" if j == 0 else "")
        ax.axhline(peak_thr, color="gray", lw=0.7, ls=":")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"dialog {i}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Causal streaming probe  red=GT  orange=pred", fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _default_data = Path(__file__).resolve().parent / "data" / "dailytalk_10000.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default=str(_default_data))
    parser.add_argument("--model",       default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device",      default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch",        default="gru", choices=["gru", "cnn1d"])
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--pos-window",  type=int,   default=1,
                        help="SEG 근방 ±N 프레임을 pos로 간주 (기본 1)")
    parser.add_argument("--n-val",       type=int,   default=50,
                        help="validation dialog 수 (기본 50)")
    parser.add_argument("--n-test",      type=int,   default=50,
                        help="test dialog 수 (기본 50)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--peak-thr",    type=float, default=None,
                        help="trigger threshold (None=val sweep 자동 결정)")
    parser.add_argument("--out-dir",     default="research/cif/causal_results")
    parser.add_argument("--cache-dir",   default=None)
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--max-dialogs", type=int, default=None,
                        help="디버그용: 최대 dialog 수 제한")
    parser.add_argument("--load-model",  default=None,
                        help="저장된 모델 pkl — 학습 건너뜀")
    parser.add_argument("--resume",      default=None,
                        help="저장된 모델 pkl — 가중치 이어받아 추가 학습")
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    out_dir   = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    # ── load / build dialog sequences ──
    cached = None if args.no_cache else load_cache(cache_dir, args.data, args.pos_window)

    if cached is None:
        with open(args.data) as f:
            records = [json.loads(l) for l in f if l.strip()]
        print(f"  {len(records)} records → grouping by dialog …")
        by_dialog = group_by_dialog(records)
        if args.max_dialogs:
            by_dialog = dict(list(by_dialog.items())[:args.max_dialogs])

        dtype = torch.bfloat16
        print("  Loading encoder …")
        encoder = load_encoder(args.model, args.device, dtype)
        fe = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=16000, hop_length=160,
            chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
        )

        all_dialogs = {}
        for did, turns in tqdm(by_dialog.items(), desc="encoding dialogs"):
            try:
                all_dialogs[did] = build_dialog_feats(
                    turns, encoder, fe, args.device, dtype, args.pos_window)
            except Exception as e:
                tqdm.write(f"  SKIP dialog {did}: {e}")

        del encoder; torch.cuda.empty_cache()
        save_cache(cache_dir, args.data, args.pos_window, all_dialogs)
    else:
        all_dialogs = cached

    # ── train / val / test split (dialog 단위) ──
    rng      = np.random.default_rng(args.seed)
    all_dids = list(all_dialogs.keys())
    rng.shuffle(all_dids)

    n_val  = args.n_val
    n_test = args.n_test
    te_dids  = set(all_dids[-n_test:])
    val_dids = set(all_dids[-(n_val + n_test):-n_test])
    tr_dids  = set(all_dids[:-(n_val + n_test)])

    dialogs_tr  = [all_dialogs[d] for d in all_dids if d in tr_dids]
    dialogs_val = [all_dialogs[d] for d in all_dids if d in val_dids]
    dialogs_te  = [all_dialogs[d] for d in all_dids if d in te_dids]
    print(f"\n  train={len(dialogs_tr)}  val={len(dialogs_val)}  test={len(dialogs_te)}")

    # ── train (or load / resume) ──
    if args.load_model:
        with open(args.load_model, "rb") as f:
            saved = pickle.load(f)
        clf = saved["clf"]
        print(f"  model loaded: {args.load_model}  arch={saved.get('arch','?')}")
    else:
        in_dim = dialogs_tr[0]["feats"].shape[1]
        init_model = None
        if args.resume:
            with open(args.resume, "rb") as f:
                saved = pickle.load(f)
            init_model = saved["clf"].model
            args.arch   = saved.get("arch", args.arch)
            print(f"  resume from: {args.resume}  arch={args.arch}")
        clf = train_causal(dialogs_tr, args.arch, in_dim,
                           args.pos_window, args.device, args.epochs, args.lr,
                           init_model=init_model,
                           dialogs_val=dialogs_val,
                           patience=10)

    # ── threshold sweep on val ──
    best_thr = args.peak_thr
    if best_thr is None:
        print("\nSweeping threshold on val set …")
        best_thr = sweep_threshold(clf, dialogs_val, args.pos_window,
                                   out_path=out_dir / "causal_thr_sweep.png")

    # ── eval on test ──
    print("\nEvaluating on test set …")
    proba_all = np.concatenate([clf.predict_proba(d["feats"])[:, 1]
                                for d in dialogs_te])
    y_all     = np.concatenate([d["labels"] for d in dialogs_te])
    frame_res  = eval_frame_level(proba_all, y_all)
    stream_res = eval_streaming(clf, dialogs_te, args.pos_window, best_thr)

    # ── print results ──
    n_te_segs = sum(len(d["seg_frames"]) for d in dialogs_te)
    print(f"\n{'='*56}")
    print(f"  [causal-{args.arch}]  pos_window=±{args.pos_window}f  "
          f"peak_thr={best_thr:.2f}")
    print(f"{'─'*56}")
    print(f"  [Frame-level]")
    print(f"    balanced_acc : {frame_res['balanced_acc']:.3f}")
    print(f"    AUC          : {frame_res['auc']:.3f}")
    print(f"    F1 @0.5      : {frame_res['f1@0.5']:.3f}")
    print(f"    best F1      : {frame_res['best_f1']:.3f}  "
          f"(thr={frame_res['best_thr']:.3f}  "
          f"P={frame_res['best_prec']:.3f}  R={frame_res['best_rec']:.3f})")
    print(f"{'─'*56}")
    print(f"  [Streaming trigger  (run, thr={best_thr:.2f})]")
    print(f"    count_acc    : {stream_res['count_acc']:.3f}")
    print(f"    count_mae    : {stream_res['count_mae']:.3f}")
    print(f"    over_rate    : {stream_res['over_rate']:.3f}  (과검출)")
    print(f"    under_rate   : {stream_res['under_rate']:.3f}  (미검출)")
    print(f"    pos_mae      : {stream_res['pos_mae_f']:.2f} f  "
          f"≈ {stream_res['pos_mae_ms']:.0f} ms")
    print(f"    n_dialogs    : {stream_res['n']}  "
          f"n_segs_total : {n_te_segs}")
    if stream_res.get("by_nsegs"):
        print(f"\n  [ n_segs별 count_acc / over / under ]")
        for k in sorted(stream_res["by_nsegs"]):
            b = stream_res["by_nsegs"][k]
            print(f"    n_segs={k}  n={b['n']:4d}  "
                  f"acc={b['acc']/b['n']:.3f}  "
                  f"over={b['over']/b['n']:.3f}  "
                  f"under={b['under']/b['n']:.3f}  "
                  f"mae={b['mae']/b['n']:.2f}")
    print(f"{'='*56}")

    # ── plots ──
    print("\nPlotting …")
    plot_pr_curve(proba_all, y_all,
                  out_path=out_dir / "causal_pr_curve.png")
    plot_dialog_examples(clf, dialogs_te, best_thr, n_show=4,
                         out_path=out_dir / "causal_examples.png")

    if not args.no_save:
        model_path = out_dir / f"causal_{args.arch}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"clf": clf, "arch": args.arch,
                         "peak_thr": best_thr,
                         "pos_window": args.pos_window}, f)
        print(f"  model saved: {model_path}")

    print("Done.")


if __name__ == "__main__":
    main()
