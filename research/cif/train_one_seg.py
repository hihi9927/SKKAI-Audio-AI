"""
CIF Weight Predictor 학습 스크립트 — one-seg 버전.

원본 train.py와 달리 모든 샘플이 SEG 1개인 태스크로 단순화됨.
  - qty_head 없음 (항상 n_segs=1이므로 불필요)
  - Loss: quantity_loss + boundary_loss (count_loss 없음)
  - Dataset: start/end/seg_t 필드로 오디오 구간 트리밍

데이터 형식 (one_seg_data/*.jsonl):
  {"audio": "path.wav", "start": 0.0, "end": 1.44, "seg_t": 1.44, "text_clean": "..."}
  start/end: 오디오 트림 구간(초). end=null이면 끝까지.
  seg_t: SEG 위치 (start 기준 상대 좌표, 초)

Loss:
  quantity_loss = (Σw_t - 1)²
  boundary_loss = MSE(w_t, label_t)
  label: [0, seg_frame) 구간에 1/seg_frame 균등 분배 → sum(label)=1

실행 예시:
  python core/cif/train_one_seg.py
  python core/cif/train_one_seg.py --epochs 30 --lr 3e-4
  python core/cif/train_one_seg.py --resume checkpoints/one_seg/cif_best.pt --epochs 40
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
sys.path.insert(0, str(_REPO_ROOT))

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)
from transformers import AutoConfig


# ---------------------------------------------------------------------------
# Encoder frame length
# ---------------------------------------------------------------------------

def mel_to_encoder_frames(mel_len: int) -> int:
    r = mel_len % 100
    if r == 0:
        return (mel_len // 100) * 13
    feat = (r - 1) // 2 + 1
    tail = ((feat - 1) // 2 + 1 - 1) // 2 + 1
    return tail + (mel_len // 100) * 13


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_16k(path: str, start: float = 0.0, end: float | None = None) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        g = math.gcd(sr, 16000)
        data = signal.resample_poly(data, 16000 // g, sr // g).astype(np.float32)

    start_sample = int(start * 16000)
    end_sample = int(end * 16000) if end is not None else len(data)
    return data[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def make_uniform_label(n_frames: int, seg_frame: int) -> torch.Tensor:
    """[0, seg_frame) 구간에 1/seg_frame 균등 분배. sum(label)=1."""
    label = torch.zeros(n_frames)
    n = max(seg_frame, 1)
    label[:n] = 1.0 / n
    return label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CIFWeightPredictor(nn.Module):
    """encoder output (T, 2048) → Conv1d → weight (T,).
    qty_head 없음: 항상 n_segs=1이므로 Σw→1 만 학습."""
    def __init__(self, d_model: int = 2048, hidden: int = 128, kernel_size: int = 7):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.out  = nn.Linear(hidden, 1)
        self.act  = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj(encoder_output))         # (T, hidden)
        x = x.T.unsqueeze(0)                            # (1, hidden, T)
        x = self.act(self.conv(x)).squeeze(0).T         # (T, hidden)
        return self.sigmoid(self.out(x)).squeeze(-1)    # (T,)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cif_loss(
    weights: torch.Tensor,
    labels: torch.Tensor,
    lambda_qty: float = 1.0,
    lambda_bnd: float = 1.0,
):
    qty = (weights.sum() - 1.0) ** 2
    bnd = nn.functional.mse_loss(weights, labels)
    loss = lambda_qty * qty + lambda_bnd * bnd
    return loss, qty.detach().item(), bnd.detach().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OneSegDataset(Dataset):
    def __init__(self, jsonl_path: str, feature_extractor: WhisperFeatureExtractor):
        with open(jsonl_path) as f:
            self.records = [json.loads(l) for l in f if l.strip()]
        self.fe = feature_extractor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        start = rec.get("start", 0.0)
        end   = rec.get("end")
        audio = load_audio_16k(rec["audio"], start=start, end=end)

        out = self.fe(audio, sampling_rate=16000, return_tensors="pt",
                      return_attention_mask=True)
        mel     = out["input_features"].squeeze(0)
        mel_len = int(out["attention_mask"].squeeze(0).sum().item())
        mel     = mel[:, :mel_len]

        n_enc = mel_to_encoder_frames(mel_len)
        dur   = len(audio) / 16000.0

        seg_t     = rec["seg_t"]
        seg_frame = min(max(round(seg_t * n_enc / dur), 0), n_enc - 1)
        label     = make_uniform_label(n_enc, seg_frame)

        return mel, mel_len, label


def collate_fn(batch):
    mel, mel_len, label = batch[0]
    return mel, mel_len, label


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(model_path: str, device: str, dtype: torch.dtype):
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map=device,
    )
    encoder = model.thinker.audio_tower
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    del model
    torch.cuda.empty_cache()
    return encoder


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def _forward(encoder, predictor, mel, mel_len, device, dtype) -> torch.Tensor:
    with torch.no_grad():
        enc_out = encoder(
            mel,
            feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
        )
    features = enc_out.last_hidden_state.squeeze(0).float()  # (T, 2048)
    return predictor(features)  # (T,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    dtype  = torch.bfloat16

    print(f"Loading encoder from {args.model} …")
    encoder   = load_encoder(args.model, args.device, dtype)
    predictor = CIFWeightPredictor(d_model=2048, hidden=args.hidden,
                                   kernel_size=args.kernel_size).to(device)
    print(f"  predictor params: {sum(p.numel() for p in predictor.parameters())}")

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )

    train_dataset = OneSegDataset(args.train_data, feature_extractor)
    train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)

    val_loader = None
    if args.val_data and Path(args.val_data).exists():
        val_dataset = OneSegDataset(args.val_data, feature_extractor)
        val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
        print(f"  train={len(train_dataset)}  val={len(val_dataset)}")
    else:
        print(f"  train={len(train_dataset)}  (val 없음)")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")
    print(f"  로그: {log_path}")

    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, _ = predictor.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  [resume] new params: {missing}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss") or float("inf")
        global_step   = ckpt.get("global_step", 0)
        print(f"  resumed epoch {start_epoch - 1}, best_val={best_val_loss:.4f}")

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epoch", unit="ep")
    for epoch in epoch_bar:

        # ── train ──
        predictor.train()
        ep_loss = ep_qty = ep_bnd = 0.0
        ep_gnorm = 0.0
        n_samples = n_steps = 0
        last_gnorm = 0.0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"  Train {epoch}/{args.epochs}",
                         leave=False, unit="sample")
        for step, (mel, mel_len, labels) in enumerate(train_bar):
            mel    = mel.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.float32)

            weights = _forward(encoder, predictor, mel, mel_len, device, dtype)

            T = weights.shape[0]
            if labels.shape[0] != T:
                labels = labels[:T] if labels.shape[0] > T else \
                    torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

            loss, qty, bnd = cif_loss(weights, labels, args.lambda_qty, args.lambda_bnd)
            (loss / args.accum_steps).backward()

            ep_loss += loss.item()
            ep_qty  += qty
            ep_bnd  += bnd
            n_samples += 1

            if (step + 1) % args.accum_steps == 0:
                last_gnorm = nn.utils.clip_grad_norm_(predictor.parameters(), float("inf")).item()
                ep_gnorm += last_gnorm
                n_steps  += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            train_bar.set_postfix(
                loss=f"{ep_loss/n_samples:.4f}",
                qty=f"{ep_qty/n_samples:.4f}",
                bnd=f"{ep_bnd/n_samples:.4f}",
                gnorm=f"{last_gnorm:.3f}",
            )

        last_gnorm = nn.utils.clip_grad_norm_(predictor.parameters(), float("inf")).item()
        ep_gnorm += last_gnorm
        n_steps  += 1
        optimizer.step()
        optimizer.zero_grad()

        train_loss = ep_loss / max(n_samples, 1)

        # ── validation ──
        val_loss = val_qty = val_bnd = 0.0
        if val_loader is not None:
            predictor.eval()
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"  Val   {epoch}/{args.epochs}",
                               leave=False, unit="sample")
                for mel, mel_len, labels in val_bar:
                    mel    = mel.to(device=device, dtype=dtype)
                    labels = labels.to(device=device, dtype=torch.float32)

                    weights = _forward(encoder, predictor, mel, mel_len, device, dtype)

                    T = weights.shape[0]
                    if labels.shape[0] != T:
                        labels = labels[:T] if labels.shape[0] > T else \
                            torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

                    _, qty, bnd = cif_loss(weights, labels, args.lambda_qty, args.lambda_bnd)
                    val_qty  += qty
                    val_bnd  += bnd
                    val_loss += args.lambda_qty * qty + args.lambda_bnd * bnd

            n_val     = len(val_loader)
            val_loss /= n_val
            val_qty  /= n_val
            val_bnd  /= n_val
            val_tag   = f"  val={val_loss:.4f} (qty={val_qty:.4f} bnd={val_bnd:.4f})"
        else:
            val_tag = ""

        avg_gnorm = ep_gnorm / max(n_steps, 1)
        epoch_bar.write(
            f"Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f} "
            f"(qty={ep_qty/n_samples:.4f} bnd={ep_bnd/n_samples:.4f})  "
            f"gnorm={avg_gnorm:.3f}"
            f"{val_tag}"
        )

        log_entry = {
            "epoch": epoch, "global_step": global_step,
            "train_loss": train_loss,
            "train_qty": ep_qty / n_samples,
            "train_bnd": ep_bnd / n_samples,
            "gnorm": avg_gnorm,
            "val_loss": val_loss if val_loader else None,
            "val_qty":  val_qty  if val_loader else None,
            "val_bnd":  val_bnd  if val_loader else None,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        monitor = val_loss if val_loader is not None else train_loss

        def _save_ckpt(path):
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "hidden": args.hidden,
                "kernel_size": args.kernel_size,
                "state_dict": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, path)

        if monitor < best_val_loss:
            best_val_loss = monitor
            _save_ckpt(ckpt_dir / "cif_best.pt")
            print(f"  → best saved (val_loss={monitor:.4f})")

        if epoch % args.save_every == 0:
            _save_ckpt(ckpt_dir / f"cif_epoch{epoch:03d}.pt")

    log_file.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _data = Path(__file__).resolve().parent / "one_seg_data"
    parser = argparse.ArgumentParser(description="CIF one-seg 학습")
    parser.add_argument("--train-data", default=str(_data / "train.jsonl"))
    parser.add_argument("--val-data",   default=str(_data / "val.jsonl"))
    parser.add_argument("--model",      default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--kernel-size",type=int,   default=7)
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lambda-qty", type=float, default=1.0)
    parser.add_argument("--lambda-bnd", type=float, default=1.0)
    parser.add_argument("--accum-steps",type=int,   default=1)
    parser.add_argument("--save-every", type=int,   default=10)
    parser.add_argument("--ckpt-dir",   default="checkpoints/encoder/one_seg")
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
