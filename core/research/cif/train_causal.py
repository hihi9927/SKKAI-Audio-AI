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
import datetime
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
                              roc_auc_score, roc_curve,
                              precision_recall_curve, auc as sklearn_auc,
                              confusion_matrix, ConfusionMatrixDisplay)
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

_REPO_ROOT  = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
_CIF_TRAIN  = Path(__file__).resolve().parent / "utils" / "cif_train"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CIF_TRAIN))

from train_one_seg import load_audio_16k, load_encoder


# ---------------------------------------------------------------------------
# Logging tee
# ---------------------------------------------------------------------------

class _Tee:
    """stdout을 화면과 파일 양쪽에 동시 출력."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


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
                       pos_window: int, max_sec: float = 28.0,
                       label_type: str = "hard", use_delta: bool = False,
                       decay_frames: int = 30,
                       features: str = "projected",
                       raw_layer_indices: list[int] | None = None) -> dict:
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
    def _enc(audio_chunk):
        if features == "raw":
            return _encode_chunk_raw(audio_chunk, encoder, fe, device, dtype,
                                     raw_layer_indices)
        return _encode_chunk(audio_chunk, encoder, fe, device, dtype)

    if len(full_audio) <= max_samples:
        feats = _enc(full_audio)
    else:
        parts = []
        for i in range(0, len(full_audio), max_samples):
            parts.append(_enc(full_audio[i: i + max_samples]))
        feats = np.concatenate(parts, axis=0)

    if use_delta:
        delta = np.concatenate([feats[:1], np.diff(feats, axis=0)], axis=0)
        feats = np.concatenate([feats, delta], axis=1)

    T = feats.shape[0]

    seg_frames = [min(max(round(t * T / total_dur), 0), T - 1)
                  for t in seg_times]

    labels = np.zeros(T, dtype=np.float32)
    if label_type == "distance":
        # proximity = max(0, 1 - dist_to_next_seg / decay_frames)
        frames_idx = np.arange(T, dtype=np.float32)
        dist = np.full(T, decay_frames, dtype=np.float32)
        for sf in seg_frames:
            d = sf - frames_idx                          # negative after sf
            d = np.where(d >= 0, d, decay_frames)        # 이미 지난 SEG 무시
            dist = np.minimum(dist, d)
        labels = np.clip(1.0 - dist / decay_frames, 0.0, 1.0)
    else:
        sigma  = max(pos_window, 1)
        radius = sigma * 3
        for sf in seg_frames:
            for df in range(-radius, radius + 1):
                f = sf + df
                if 0 <= f < T:
                    v = (np.exp(-0.5 * (df / sigma) ** 2) if label_type == "gaussian"
                         else (1.0 if abs(df) <= pos_window else 0.0))
                    labels[f] = max(labels[f], v)

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


def _encode_chunk_raw(audio: np.ndarray, encoder, fe, device, dtype,
                       layer_indices: list[int]) -> np.ndarray:
    """encoder 24개 레이어에 hook을 걸어 (T, n_layers×d_model) raw feature 반환."""
    out     = fe(audio, sampling_rate=16000,
                 return_tensors="pt", return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]

    layer_acts: dict[int, np.ndarray] = {}
    hooks = []
    idx_set = set(layer_indices)
    for i, layer in enumerate(encoder.layers):
        if i not in idx_set:
            continue
        def _hook(module, inp, output, idx=i):
            act = output[0] if isinstance(output, (tuple, list)) else output
            layer_acts[idx] = act.detach().float().cpu().numpy()
        hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        encoder(mel.to(device=device, dtype=dtype),
                feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long))

    for h in hooks:
        h.remove()

    return np.concatenate([layer_acts[i] for i in sorted(layer_indices)], axis=1)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, data_path: str, pos_window: int,
                label_type: str = "hard", use_delta: bool = False,
                decay_frames: int = 30,
                features: str = "projected",
                raw_layer_indices: list[int] | None = None) -> Path:
    label_tag = "" if label_type == "hard" else f"_{label_type}"
    decay_tag  = f"_d{decay_frames}" if label_type == "distance" else ""
    delta_tag  = "_delta" if use_delta else ""
    if features == "raw":
        n = len(raw_layer_indices) if raw_layer_indices else 24
        feat_tag = f"_raw{n}L"
    else:
        feat_tag = ""
    key = Path(data_path).stem + f"_pw{pos_window}_causal{label_tag}{decay_tag}{delta_tag}{feat_tag}"
    return cache_dir / f"{key}.pkl"


def save_cache(cache_dir: Path, data_path: str, pos_window: int, dialogs: dict,
               label_type: str = "hard", use_delta: bool = False,
               decay_frames: int = 30,
               features: str = "projected",
               raw_layer_indices: list[int] | None = None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, data_path, pos_window, label_type, use_delta,
                       decay_frames, features, raw_layer_indices)
    with open(path, "wb") as f:
        pickle.dump(dialogs, f)
    print(f"  cache saved: {path.name}  ({len(dialogs)} dialogs)")


def load_cache(cache_dir: Path, data_path: str, pos_window: int,
               label_type: str = "hard", use_delta: bool = False,
               decay_frames: int = 30,
               features: str = "projected",
               raw_layer_indices: list[int] | None = None) -> dict | None:
    path = _cache_path(cache_dir, data_path, pos_window, label_type, use_delta,
                       decay_frames, features, raw_layer_indices)
    if not path.exists():
        print(f"  cache miss: {path.resolve()}")
        return None
    print(f"  loading cache: {path.resolve()} …")
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


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CausalTransformerProbe(nn.Module):
    def __init__(self, in_dim: int = 2048, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, attn_window: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) must be divisible by n_heads({n_heads})"
        self.proj        = nn.Linear(in_dim, d_model)
        self.pe          = SinusoidalPE(d_model)
        self.attn_window = attn_window
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head        = nn.Linear(d_model, 1)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        if self.attn_window is None:
            return nn.Transformer.generate_square_subsequent_mask(T, device=device)
        mask = torch.full((T, T), float("-inf"), device=device)
        for i in range(T):
            start = max(0, i - self.attn_window + 1)
            mask[i, start:i + 1] = 0.0
        return mask

    def forward(self, x: torch.Tensor, h=None):   # (B, T, D) → (B, T), None
        x   = self.pe(self.proj(x))
        T   = x.size(1)
        out = self.transformer(x, mask=self._causal_mask(T, x.device), is_causal=True)
        return self.head(out).squeeze(-1), None


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

def _focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                pos_weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p  = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    return ((1 - pt) ** gamma * bce).mean()


def _make_hard_labels(seg_frames: list, T: int, pos_window: int,
                      device) -> torch.Tensor:
    labels = torch.zeros(T, dtype=torch.float32, device=device)
    for sf in seg_frames:
        for df in range(-pos_window, pos_window + 1):
            f = sf + df
            if 0 <= f < T:
                labels[f] = 1.0
    return labels


def train_causal(dialogs_tr: list, arch: str, in_dim: int, pos_window: int,
                 device: str, n_epochs: int = 20, lr: float = 1e-3,
                 hidden: int = 128, n_layers: int = 1, n_heads: int = 4,
                 attn_window: int | None = None,
                 loss_type: str = "bce", loss_alpha: float = 0.5,
                 freeze_backbone: bool = False,
                 lr_scheduler: str = "plateau",
                 init_model: nn.Module | None = None,
                 dialogs_val: list | None = None,
                 patience: int = 10) -> CausalWrapper:
    if init_model is not None:
        model = init_model.to(device)
        print(f"  resuming from existing weights")
        if freeze_backbone:
            for name, param in model.named_parameters():
                param.requires_grad_(name.startswith("head"))
            frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            print(f"  [freeze] frozen={len(frozen)} params  trainable(head)={len(trainable)} params")
    else:
        if arch == "gru":
            model = CausalGRUProbe(in_dim=in_dim, hidden=hidden,
                                   n_layers=n_layers).to(device)
        elif arch == "transformer":
            model = CausalTransformerProbe(in_dim=in_dim, d_model=hidden,
                                           n_heads=n_heads, n_layers=n_layers,
                                           attn_window=attn_window).to(device)
        else:
            model = CausalCNN1DProbe(in_dim=in_dim, hidden=hidden,
                                     n_layers=n_layers).to(device)

    n_pos = sum(int((d["labels"] > 0).sum()) for d in dialogs_tr)
    n_neg = sum(int((d["labels"] == 0).sum()) for d in dialogs_tr)

    use_mse      = (loss_type in ("distance", "combined"))
    use_combined = (loss_type == "combined")

    # combined: hard binary pos_weight (±pos_window 기준)
    n_pos_hard = sum(
        sum(1 for sf in d["seg_frames"]
            for df in range(-pos_window, pos_window + 1)
            if 0 <= sf + df < d["T"])
        for d in dialogs_tr
    )
    n_neg_hard  = sum(d["T"] for d in dialogs_tr) - n_pos_hard
    pos_weight  = torch.tensor([n_neg_hard / max(n_pos_hard, 1)],
                               dtype=torch.float32, device=device)

    if use_combined:
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse_criterion = nn.MSELoss()
    elif use_mse:
        mse_criterion = nn.MSELoss()
        bce_criterion = None
    else:
        if loss_type == "focal":
            bce_criterion = lambda logits, y: _focal_loss(logits, y, pos_weight)
        else:
            bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse_criterion = None

    # combined: hard label numpy 미리 계산 (훈련 루프에서 매번 계산 방지)
    if use_combined:
        all_dialogs_for_precomp = dialogs_tr + (dialogs_val or [])
        for d in all_dialogs_for_precomp:
            if "labels_hard" not in d:
                hard = np.zeros(d["T"], dtype=np.float32)
                for sf in d["seg_frames"]:
                    for df in range(-pos_window, pos_window + 1):
                        f = sf + df
                        if 0 <= f < d["T"]:
                            hard[f] = 1.0
                d["labels_hard"] = hard
        print(f"  [combined] hard labels precomputed for "
              f"{len(all_dialogs_for_precomp)} dialogs")

    opt        = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-6)
    elif lr_scheduler == "none":
        scheduler = None
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)

    extra = (f"  n_heads={n_heads}  attn_window={attn_window}"
             if arch == "transformer" else "")
    alpha_tag = f"  alpha={loss_alpha}" if use_combined else ""
    print(f"  training causal-{arch}  hidden={hidden}  n_layers={n_layers}{extra}  "
          f"loss={loss_type}{alpha_tag}  epochs={n_epochs}  lr={lr}  patience={patience}")
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
            if use_combined:
                y_hard = torch.tensor(d["labels_hard"], dtype=torch.float32).unsqueeze(0).to(device)
                loss = mse_criterion(torch.sigmoid(logits), y) + loss_alpha * bce_criterion(logits, y_hard)
            elif use_mse:
                loss = mse_criterion(torch.sigmoid(logits), y)
            else:
                loss = bce_criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(dialogs_tr)

        # ── validation loss, scheduler, early stopping ──
        if dialogs_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for d in dialogs_val:
                    x = torch.tensor(d["feats"],  dtype=torch.float32).unsqueeze(0).to(device)
                    y = torch.tensor(d["labels"], dtype=torch.float32).unsqueeze(0).to(device)
                    logits, _ = model(x)
                    if use_combined:
                        y_hard = torch.tensor(d["labels_hard"], dtype=torch.float32).unsqueeze(0).to(device)
                        vl = mse_criterion(torch.sigmoid(logits), y) + loss_alpha * bce_criterion(logits, y_hard)
                    elif use_mse:
                        vl = mse_criterion(torch.sigmoid(logits), y)
                    else:
                        vl = bce_criterion(logits, y)
                    val_loss += vl.item()
            val_loss /= len(dialogs_val)

            cur_lr  = opt.param_groups[0]["lr"]
            prev_lr = cur_lr
            if scheduler is not None:
                if lr_scheduler == "cosine":
                    scheduler.step()
                else:
                    scheduler.step(val_loss)
            cur_lr  = opt.param_groups[0]["lr"]
            lr_tag  = f"  lr={cur_lr:.2e}" if cur_lr != prev_lr else ""

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1:3d}/{n_epochs}  "
                      f"tr={tr_loss:.4f}  val={val_loss:.4f}"
                      + ("  *" if val_loss < best_val_loss else "")
                      + lr_tag)

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


def make_gt_binary(dialogs: list, pos_window: int) -> np.ndarray:
    """학습 label type과 무관하게 SEG 위치 ±pos_window 기반 binary GT 생성."""
    parts = []
    for d in dialogs:
        gt = np.zeros(d["T"], dtype=np.int32)
        for sf in d["seg_frames"]:
            for df in range(-pos_window, pos_window + 1):
                f = sf + df
                if 0 <= f < d["T"]:
                    gt[f] = 1
        parts.append(gt)
    return np.concatenate(parts)


def eval_frame_level(proba: np.ndarray, y_gt: np.ndarray) -> dict:
    """y_gt: SEG 위치 ±pos_window 기반 binary (make_gt_binary 결과)."""
    pred    = (proba >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_gt, pred)
    f1      = f1_score(y_gt, pred, zero_division=0)
    auc     = roc_auc_score(y_gt, proba)
    prec, rec, thr = precision_recall_curve(y_gt, proba)
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

def plot_pr_curve(proba_all: np.ndarray, y_gt: np.ndarray, out_path):
    """y_gt: make_gt_binary 결과 (SEG ±pos_window binary, label type 무관)."""
    prec, rec, thr = precision_recall_curve(y_gt, proba_all)
    pr_auc = sklearn_auc(rec, prec)
    f1c = 2 * prec * rec / (prec + rec + 1e-8)
    bi  = np.argmax(f1c[:-1])

    fpr, tpr, _ = roc_curve(y_gt, proba_all)
    roc_auc     = roc_auc_score(y_gt, proba_all)

    best_thr_val = thr[bi]
    y_pred = (proba_all >= best_thr_val).astype(int)
    cm = confusion_matrix(y_gt, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_pr, ax_roc, ax_f1, ax_cm = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_pr.plot(rec, prec, lw=1.2)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall  [causal]  PR-AUC={pr_auc:.3f}")
    ax_pr.grid(alpha=0.3)

    ax_roc.plot(fpr, tpr, lw=1.2, color="darkorange")
    ax_roc.plot([0, 1], [0, 1], lw=0.8, ls="--", color="gray")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"ROC  [causal]  AUC={roc_auc:.3f}")
    ax_roc.grid(alpha=0.3)

    ax_f1.plot(thr, f1c[:-1], lw=1.2)
    ax_f1.axvline(best_thr_val, color="red", ls="--",
                  label=f"best thr={best_thr_val:.3f}  F1={f1c[bi]:.3f}")
    ax_f1.set_xlabel("threshold"); ax_f1.set_ylabel("F1")
    ax_f1.set_title("F1 vs threshold"); ax_f1.legend(); ax_f1.grid(alpha=0.3)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["non-SEG", "SEG"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"Confusion Matrix  (thr={best_thr_val:.3f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")
    return pr_auc, roc_auc


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
    _default_data = Path(__file__).resolve().parent / "dataset" / "data" / "dailytalk_10000.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default=str(_default_data))
    parser.add_argument("--model",       default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device",      default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch",         default="gru", choices=["gru", "cnn1d", "transformer"])
    parser.add_argument("--hidden",       type=int,   default=128,
                        help="GRU/CNN hidden dim 또는 Transformer d_model")
    parser.add_argument("--n-layers",     type=int,   default=2)
    parser.add_argument("--n-heads",      type=int,   default=4,
                        help="Transformer attention heads (d_model % n_heads == 0)")
    parser.add_argument("--attn-window",  type=int,   default=None,
                        help="Transformer sliding window 크기 (None=전체 causal)")
    parser.add_argument("--loss",         default="bce",
                        choices=["bce", "focal", "distance", "combined"])
    parser.add_argument("--loss-alpha",   type=float, default=0.5,
                        help="combined loss에서 BCE 비중 (기본 0.5)")
    parser.add_argument("--label-type",   default="hard",
                        choices=["hard", "gaussian", "distance"])
    parser.add_argument("--decay-frames", type=int, default=30,
                        help="distance 타깃: SEG에서 몇 프레임 전부터 ramp-up (기본 30≈0.6초)")
    parser.add_argument("--features",     default="projected",
                        choices=["projected", "raw"],
                        help="projected=last_hidden_state(2048) / raw=encoder 레이어 직접")
    parser.add_argument("--raw-layers",   default=None,
                        help="raw 모드에서 사용할 레이어 인덱스 (쉼표 구분, 예: 11,16,20). "
                             "미지정 시 전체 24개")
    parser.add_argument("--delta",        action="store_true",
                        help="encoder output에 delta feature를 concat (in_dim 2배)")
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
    parser.add_argument("--resume",       default=None,
                        help="저장된 모델 pkl — 가중치 이어받아 추가 학습")
    parser.add_argument("--no-save",      action="store_true")

    # Curriculum learning
    parser.add_argument("--curriculum",      action="store_true",
                        help="2단계 학습: Phase1=distance, Phase2=BCE hard")
    parser.add_argument("--phase2-epochs",   type=int, default=10,
                        help="Phase2 epochs (기본 10)")
    parser.add_argument("--phase2-lr",       type=float, default=None,
                        help="Phase2 LR (미지정 시 --lr / 5 자동 적용)")
    parser.add_argument("--freeze-backbone",  action="store_true",
                        help="Phase2에서 backbone freeze, head만 학습")
    parser.add_argument("--phase1-scheduler", default="plateau",
                        choices=["none", "cosine", "plateau"],
                        help="Phase1 LR scheduler (기본 plateau)")
    parser.add_argument("--phase2-scheduler", default="cosine",
                        choices=["none", "cosine", "plateau"],
                        help="Phase2 LR scheduler (기본 cosine)")
    parser.add_argument("--log-file",    default=None,
                        help="로그 저장 경로 (미지정 시 out_dir/run_YYYYMMDD_HHMMSS.log 자동 생성)")
    args = parser.parse_args()

    out_dir   = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    # raw-layers 파싱
    if args.features == "raw":
        if args.raw_layers:
            raw_layer_indices = [int(x) for x in args.raw_layers.split(",")]
        else:
            raw_layer_indices = list(range(24))   # 전체 24레이어
        print(f"  features=raw  layers={raw_layer_indices}  "
              f"in_dim={len(raw_layer_indices) * 1024}")
    else:
        raw_layer_indices = None

    log_path = Path(args.log_file) if args.log_file else \
               out_dir / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"log → {log_path}")
    print(f"args: {vars(args)}\n")
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    # ── load / build dialog sequences ──
    cached = None if args.no_cache else load_cache(
        cache_dir, args.data, args.pos_window, args.label_type,
        args.delta, args.decay_frames, args.features, raw_layer_indices)

    if cached is None:
        # label만 다른 경우: 동일 delta 설정의 hard 캐시에서 feats/seg_frames 재활용
        hard_cache = (None if args.label_type == "hard" or args.no_cache
                      else load_cache(cache_dir, args.data, args.pos_window,
                                      "hard", args.delta))
        if hard_cache is not None:
            print(f"  hard 캐시 재활용 → label을 {args.label_type}으로 재계산 …")
            all_dialogs = {}
            for did, d in hard_cache.items():
                T          = d["T"]
                seg_frames = d["seg_frames"]
                if args.label_type == "distance":
                    frames_idx = np.arange(T, dtype=np.float32)
                    dist = np.full(T, args.decay_frames, dtype=np.float32)
                    for sf in seg_frames:
                        dv = sf - frames_idx
                        dv = np.where(dv >= 0, dv, args.decay_frames)
                        dist = np.minimum(dist, dv)
                    labels = np.clip(1.0 - dist / args.decay_frames, 0.0, 1.0)
                else:  # gaussian
                    sigma  = max(args.pos_window, 1)
                    radius = sigma * 3
                    labels = np.zeros(T, dtype=np.float32)
                    for sf in seg_frames:
                        for df in range(-radius, radius + 1):
                            f = sf + df
                            if 0 <= f < T:
                                labels[f] = max(labels[f],
                                                np.exp(-0.5 * (df / sigma) ** 2))
                all_dialogs[did] = {**d, "labels": labels}
            save_cache(cache_dir, args.data, args.pos_window, all_dialogs,
                       args.label_type, args.delta, args.decay_frames,
                       args.features, raw_layer_indices)
        else:
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
                        turns, encoder, fe, args.device, dtype, args.pos_window,
                        label_type=args.label_type, use_delta=args.delta,
                        decay_frames=args.decay_frames,
                        features=args.features,
                        raw_layer_indices=raw_layer_indices)
                except Exception as e:
                    tqdm.write(f"  SKIP dialog {did}: {e}")

            del encoder; torch.cuda.empty_cache()
            save_cache(cache_dir, args.data, args.pos_window, all_dialogs,
                       args.label_type, args.delta, args.decay_frames,
                       args.features, raw_layer_indices)
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
        # distance label이면 loss도 자동 맞춤
        loss_type = args.loss if args.loss != "bce" else (
            "distance" if args.label_type == "distance" else "bce"
        )

        if args.curriculum:
            # ── Phase 1: distance loss ──────────────────────────────────────
            print("\n[curriculum] Phase 1: distance loss")
            clf = train_causal(dialogs_tr, args.arch, in_dim,
                               args.pos_window, args.device, args.epochs, args.lr,
                               hidden=args.hidden, n_layers=args.n_layers,
                               n_heads=args.n_heads, attn_window=args.attn_window,
                               loss_type="distance",
                               lr_scheduler=args.phase1_scheduler,
                               init_model=init_model,
                               dialogs_val=dialogs_val, patience=10)

            # ── Phase 2: BCE hard binary fine-tune ─────────────────────────
            print("\n[curriculum] Phase 2: BCE fine-tune")
            phase2_lr = args.phase2_lr if args.phase2_lr else args.lr / 5

            # hard binary labels로 교체한 shallow copy
            def _to_hard(d):
                hard = np.zeros(d["T"], dtype=np.float32)
                for sf in d["seg_frames"]:
                    for df in range(-args.pos_window, args.pos_window + 1):
                        f = sf + df
                        if 0 <= f < d["T"]:
                            hard[f] = 1.0
                return {**d, "labels": hard}

            dialogs_tr_hard  = [_to_hard(d) for d in dialogs_tr]
            dialogs_val_hard = [_to_hard(d) for d in dialogs_val]

            clf = train_causal(dialogs_tr_hard, args.arch, in_dim,
                               args.pos_window, args.device, args.phase2_epochs,
                               phase2_lr,
                               hidden=args.hidden, n_layers=args.n_layers,
                               n_heads=args.n_heads, attn_window=args.attn_window,
                               loss_type="bce",
                               freeze_backbone=args.freeze_backbone,
                               lr_scheduler=args.phase2_scheduler,
                               init_model=clf.model,
                               dialogs_val=dialogs_val_hard, patience=10)
        else:
            clf = train_causal(dialogs_tr, args.arch, in_dim,
                               args.pos_window, args.device, args.epochs, args.lr,
                               hidden=args.hidden,
                               n_layers=args.n_layers,
                               n_heads=args.n_heads,
                               attn_window=args.attn_window,
                               loss_type=loss_type,
                               loss_alpha=args.loss_alpha,
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
    y_gt      = make_gt_binary(dialogs_te, args.pos_window)   # 통일 기준
    frame_res  = eval_frame_level(proba_all, y_gt)
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
    pr_auc, roc_auc = plot_pr_curve(proba_all, y_gt,
                                     out_path=out_dir / "causal_pr_curve.png")
    print(f"    PR-AUC       : {pr_auc:.3f}")
    print(f"    ROC-AUC      : {roc_auc:.3f}")
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
    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
