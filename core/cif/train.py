"""
CIF Weight Predictor 학습 스크립트.

두 가지 입력 모드 (--mode):

  encoder (기본):
    audio → mel → frozen Qwen3-ASR encoder (317M) → (T, 2048) → CIFWeightPredictor
    인코더 표현에서 SEG 경계를 학습. 추론 파이프라인과 동일한 입력 공간.

  mel:
    audio → mel (128, T_mel) → CIFMelPredictor [3× Conv1d stride-2]
    인코더 없이 acoustic 피처만으로 경계 예측 가능한지 검증하는 baseline.
    encoder mode와 val_loss를 비교해 인코더 표현이 얼마나 기여하는지 확인.

Loss:
  quantity_loss = (Σw_t - N_seg)²
  boundary_loss = MSE(w_t, gaussian_label_t)   [Gaussian normalized to sum=1.0 per SEG]
  total = λ_qty * quantity_loss + λ_bnd * boundary_loss

실행 예시:
  python core/cif/train.py                          # encoder mode
  python core/cif/train.py --mode mel               # mel baseline
  python core/cif/train.py --epochs 30 --lr 3e-4
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import scipy.signal as signal
import torch
import torch.nn as nn
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
# Encoder frame length (mirrors _get_feat_extract_output_lengths in modeling)
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

def load_audio_16k(path: str) -> np.ndarray:
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

    return data


# ---------------------------------------------------------------------------
# Gaussian soft label
# ---------------------------------------------------------------------------

def make_gaussian_label(n_frames: int, boundary_frames: list, sigma: float = 5.0) -> torch.Tensor:
    """Gaussian bumps at boundary_frames, each normalized to integrate to 1.0."""
    t = torch.arange(n_frames, dtype=torch.float32)
    label = torch.zeros(n_frames)
    for b in boundary_frames:
        g = torch.exp(-0.5 * ((t - b) / sigma) ** 2)
        label += g / (g.sum() + 1e-8)
    return label  # sum ≈ n_segs


# ---------------------------------------------------------------------------
# CIF predictors
# ---------------------------------------------------------------------------

class CIFWeightPredictor(nn.Module):
    """encoder mode: Linear on top of frozen encoder output (T, 2048) → (T,)"""
    def __init__(self, d_model: int = 2048):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # encoder_output: (T, d_model)
        return self.sigmoid(self.linear(encoder_output)).squeeze(-1)  # (T,)


class CIFMelPredictor(nn.Module):
    """mel mode: 3× Conv1d stride-2 (8x downsample) directly on mel → (T_enc,).
    인코더 없이 acoustic 피처만으로 SEG 경계를 예측할 수 있는지 검증하는 baseline.
    temporal resolution이 encoder mode와 동일하게 맞춰짐."""
    def __init__(self, n_mel: int = 128, hidden: int = 256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(n_mel, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (n_mel, T_mel)
        x = self.convs(mel.unsqueeze(0)).squeeze(0)  # (hidden, T_enc)
        return self.sigmoid(self.linear(x.T)).squeeze(-1)  # (T_enc,)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cif_loss(
    weights: torch.Tensor,
    labels: torch.Tensor,
    n_segs: int,
    lambda_qty: float = 1.0,
    lambda_bnd: float = 1.0,
):
    qty = (weights.sum() - n_segs) ** 2
    bnd = nn.functional.mse_loss(weights, labels)
    return lambda_qty * qty + lambda_bnd * bnd, qty.detach().item(), bnd.detach().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CIFDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        feature_extractor: WhisperFeatureExtractor,
        sigma: float = 5.0,
    ):
        with open(jsonl_path) as f:
            self.records = [json.loads(l) for l in f if l.strip()]
        self.fe = feature_extractor
        self.sigma = sigma

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        audio = load_audio_16k(rec["audio"])

        # mel features: (1, 128, 3000) → squeeze → (128, 3000)
        out = self.fe(audio, sampling_rate=16000, return_tensors="pt",
                      return_attention_mask=True)
        mel = out["input_features"].squeeze(0)          # (128, 3000)
        mel_len = int(out["attention_mask"].squeeze(0).sum().item())
        mel = mel[:, :mel_len]                          # (128, mel_len)

        n_enc = mel_to_encoder_frames(mel_len)
        seg_frames = rec["seg_encoder_frames"]

        # clamp boundary frames into valid range
        seg_frames = [min(max(f, 0), n_enc - 1) for f in seg_frames]
        labels = make_gaussian_label(n_enc, seg_frames, self.sigma)

        # mel과 mel_len 모두 반환: encoder mode는 mel_len으로 encoder 호출,
        # mel mode는 mel을 직접 predictor에 넣음
        return mel, mel_len, labels, len(seg_frames)


# ---------------------------------------------------------------------------
# Collate: batch_size=1 (encoder processes one sample at a time)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    mel, mel_len, labels, n_segs = batch[0]
    return mel, mel_len, labels, n_segs


# ---------------------------------------------------------------------------
# Forward helper (mode 분기를 한 곳에서 처리)
# ---------------------------------------------------------------------------

def _forward(encoder, predictor, mel, mel_len, device, dtype) -> torch.Tensor:
    """encoder mode / mel mode 공통 forward. weights (T,) 반환."""
    if encoder is not None:
        # encoder mode: mel → frozen encoder → (T, 2048) → CIFWeightPredictor
        with torch.no_grad():
            enc_out = encoder(
                mel,
                feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
            )
        features = enc_out.last_hidden_state.squeeze(0).float()  # (T, 2048)
        return predictor(features)
    else:
        # mel mode: mel (128, T_mel) → CIFMelPredictor
        return predictor(mel.float())


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(model_path: str, device: str, dtype: torch.dtype):
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
    )
    encoder = model.thinker.audio_tower
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    # free the rest of the model
    del model
    torch.cuda.empty_cache()
    return encoder


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    if args.mode == "encoder":
        print(f"Loading encoder from {args.model} …")
        encoder = load_encoder(args.model, args.device, dtype)
        predictor = CIFWeightPredictor(d_model=2048).to(device)
        print(f"  mode=encoder  predictor params: {sum(p.numel() for p in predictor.parameters())}")
    else:
        encoder = None
        predictor = CIFMelPredictor(n_mel=128, hidden=args.mel_hidden).to(device)
        print(f"  mode=mel  predictor params: {sum(p.numel() for p in predictor.parameters())}")
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=True,
    )

    train_dataset = CIFDataset(args.train_data, feature_extractor, sigma=args.sigma)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    val_loader = None
    if args.val_data and Path(args.val_data).exists():
        val_dataset = CIFDataset(args.val_data, feature_extractor, sigma=args.sigma)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)
        print(f"  train={len(train_dataset)}  val={len(val_dataset)}")
    else:
        print(f"  train={len(train_dataset)}  (val 없음)")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epoch", unit="ep")
    for epoch in epoch_bar:
        # ── train ──
        predictor.train()
        epoch_loss = epoch_qty = epoch_bnd = 0.0
        n_samples = 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"  Train {epoch}/{args.epochs}",
                         leave=False, unit="sample")
        for step, (mel, mel_len, labels, n_segs) in enumerate(train_bar):
            mel = mel.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.float32)

            weights = _forward(encoder, predictor, mel, mel_len, device, dtype)

            T = weights.shape[0]
            if labels.shape[0] != T:
                labels = labels[:T] if labels.shape[0] > T else \
                    torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

            loss, qty, bnd = cif_loss(weights, labels, n_segs,
                                      args.lambda_qty, args.lambda_bnd)
            (loss / args.accum_steps).backward()

            epoch_loss += loss.item()
            epoch_qty += qty
            epoch_bnd += bnd
            n_samples += 1

            train_bar.set_postfix(
                loss=f"{epoch_loss/n_samples:.4f}",
                qty=f"{epoch_qty/n_samples:.4f}",
                bnd=f"{epoch_bnd/n_samples:.4f}",
            )

            if (step + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        optimizer.step()
        optimizer.zero_grad()

        train_loss = epoch_loss / max(n_samples, 1)

        # ── validation ──
        val_loss = val_qty = val_bnd = 0.0
        if val_loader is not None:
            predictor.eval()
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"  Val   {epoch}/{args.epochs}",
                               leave=False, unit="sample")
                for mel, mel_len, labels, n_segs in val_bar:
                    mel = mel.to(device=device, dtype=dtype)
                    labels = labels.to(device=device, dtype=torch.float32)

                    weights = _forward(encoder, predictor, mel, mel_len, device, dtype)

                    T = weights.shape[0]
                    if labels.shape[0] != T:
                        labels = labels[:T] if labels.shape[0] > T else \
                            torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

                    _, qty, bnd = cif_loss(weights, labels, n_segs,
                                           args.lambda_qty, args.lambda_bnd)
                    val_qty += qty
                    val_bnd += bnd
                    val_loss += args.lambda_qty * qty + args.lambda_bnd * bnd

            n_val = len(val_loader)
            val_loss /= n_val
            val_qty /= n_val
            val_bnd /= n_val
            val_tag = f"  val={val_loss:.4f} (qty={val_qty:.4f} bnd={val_bnd:.4f})"
        else:
            val_tag = ""

        epoch_bar.write(
            f"Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f} (qty={epoch_qty/n_samples:.4f} bnd={epoch_bnd/n_samples:.4f})"
            f"{val_tag}"
        )

        # best checkpoint은 val loss 기준 (val 없으면 train loss)
        monitor = val_loss if val_loader is not None else train_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            ckpt_path = ckpt_dir / "cif_best.pt"
            torch.save({"epoch": epoch, "mode": args.mode,
                        "state_dict": predictor.state_dict(),
                        "train_loss": train_loss, "val_loss": val_loss}, ckpt_path)
            print(f"  → best checkpoint saved (val_loss={monitor:.4f})")

        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"cif_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "mode": args.mode,
                        "state_dict": predictor.state_dict(),
                        "train_loss": train_loss, "val_loss": val_loss}, ckpt_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CIF Weight Predictor 학습")
    parser.add_argument(
        "--train-data",
        default=str(Path(__file__).resolve().parent / "data" / "train.jsonl"),
    )
    parser.add_argument(
        "--val-data",
        default=str(Path(__file__).resolve().parent / "data" / "val.jsonl"),
    )
    parser.add_argument("--mode", choices=["encoder", "mel"], default="encoder",
                        help="encoder: frozen ASR encoder + Linear / mel: Conv1d baseline")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B",
                        help="encoder mode에서만 사용")
    parser.add_argument("--mel-hidden", type=int, default=256,
                        help="mel mode Conv1d hidden dim")
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="기본값: checkpoints/{mode}",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="Gaussian soft label sigma (encoder frames)")
    parser.add_argument("--lambda-qty", type=float, default=1.0,
                        help="quantity loss weight")
    parser.add_argument("--lambda-bnd", type=float, default=1.0,
                        help="boundary loss weight")
    parser.add_argument("--accum-steps", type=int, default=32,
                        help="gradient accumulation (effective batch size)")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = str(
            Path(__file__).resolve().parent / "checkpoints" / args.mode
        )
    train(args)


if __name__ == "__main__":
    main()
