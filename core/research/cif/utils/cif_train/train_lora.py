"""
CIF + Encoder LoRA 공동 학습 스크립트.

frozen encoder로는 경계 정보를 충분히 뽑기 어려워서
LoRA로 encoder를 CIF loss에 맞게 fine-tuning한다.

학습 대상:
  - Qwen3-ASR audio_tower: LoRA adapter (q_proj, v_proj)
  - CIFWeightPredictor: 전체 파라미터

Checkpoint 저장:
  {ckpt_dir}/lora_best/     ← PEFT adapter (encoder)
  {ckpt_dir}/lora_best.pt   ← CIF head + 메타 정보

실행 예시:
  python core/cif/train_lora.py
  python core/cif/train_lora.py --lora-r 16 --epochs 30
  python core/cif/train_lora.py --resume core/cif/checkpoints/lora/lora_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import WhisperFeatureExtractor, AutoConfig

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
sys.path.insert(0, str(_REPO_ROOT))

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)
from train import (
    CIFWeightPredictor,
    CIFDataset,
    collate_fn,
    cif_loss,
    mel_to_encoder_frames,
    make_uniform_label,
)


# ---------------------------------------------------------------------------
# Encoder 로드 + LoRA 적용
# ---------------------------------------------------------------------------

def load_encoder_lora(model_path: str, device: str, dtype: torch.dtype,
                      lora_r: int, lora_alpha: int,
                      lora_target: list[str], lora_dropout: float):
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map=device,
    )

    encoder = model.thinker.audio_tower

    # 전체 파라미터 freeze
    for p in encoder.parameters():
        p.requires_grad_(False)

    # LoRA 적용
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target,
        lora_dropout=lora_dropout,
        bias="none",
    )
    encoder = get_peft_model(encoder, lora_cfg)
    encoder.to(device)

    # 나머지 모델 해제
    del model.thinker.model
    del model.thinker.lm_head
    torch.cuda.empty_cache()

    lora_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder LoRA params: {lora_params:,} / {total_params:,} "
          f"({lora_params/total_params*100:.2f}%)")

    return encoder


# ---------------------------------------------------------------------------
# Forward (LoRA encoder는 grad 흐름 필요)
# ---------------------------------------------------------------------------

def forward_lora(encoder, predictor, mel, mel_len, device, dtype):
    enc_out = encoder(
        mel,
        feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
    )
    features = enc_out.last_hidden_state.squeeze(0).float()  # (T, 2048)
    return predictor(features)  # (weights, qty_pred)


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"Loading encoder + LoRA (r={args.lora_r}) from {args.model} …")
    encoder = load_encoder_lora(
        args.model, args.device, dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
        lora_dropout=args.lora_dropout,
    )

    predictor = CIFWeightPredictor(d_model=2048, hidden=args.hidden,
                                   kernel_size=args.kernel_size).to(device)
    print(f"  predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # 별도 learning rate: LoRA << predictor
    optimizer = torch.optim.Adam([
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.lora_lr},
        {"params": predictor.parameters(),
         "lr": args.lr},
    ])

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )

    train_dataset = CIFDataset(args.train_data, feature_extractor)
    if args.balanced:
        from collections import Counter
        nsegs_list = [len(r["seg_timestamps"]) for r in train_dataset.records]
        counts = Counter(nsegs_list)
        sample_weights = [1.0 / counts[n] for n in nsegs_list]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler,
                                  collate_fn=collate_fn, num_workers=0)
        print(f"  WeightedRandomSampler: {dict(sorted(counts.items()))}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0)

    val_loader = None
    if args.val_data and Path(args.val_data).exists():
        val_dataset = CIFDataset(args.val_data, feature_extractor)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)
        print(f"  train={len(train_dataset)}  val={len(val_dataset)}")
    else:
        print(f"  train={len(train_dataset)}  (val 없음)")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")
    print(f"  학습 로그: {log_path}\n")

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, _ = predictor.load_state_dict(ckpt["predictor"], strict=False)
        if missing:
            print(f"  [resume] new params (random init): {missing}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        # LoRA adapter는 같은 디렉토리에 저장됨
        lora_dir = Path(args.resume).parent / (Path(args.resume).stem + "_lora")
        if lora_dir.exists():
            encoder.load_adapter(str(lora_dir))
            print(f"  LoRA adapter loaded from {lora_dir}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss") or float("inf")
        global_step = ckpt.get("global_step", 0)
        print(f"  resumed (epoch {start_epoch-1}, best_val={best_val_loss:.4f})")

    def _save_ckpt(name: str, epoch: int, train_loss: float, val_loss: float):
        # LoRA adapter 저장
        lora_dir = ckpt_dir / f"{name}_lora"
        encoder.save_pretrained(str(lora_dir))
        # CIF head + 메타 저장
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hidden": args.hidden,
            "kernel_size": args.kernel_size,
            "lora_r": args.lora_r,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, ckpt_dir / f"{name}.pt")

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epoch", unit="ep")
    for epoch in epoch_bar:

        # ── train ──
        encoder.train()
        predictor.train()
        epoch_loss = epoch_qty = epoch_bnd = epoch_cnt = epoch_gnorm = 0.0
        n_samples = n_steps = 0
        last_gnorm = 0.0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"  Train {epoch}/{args.epochs}",
                         leave=False, unit="sample")
        for step, (mel, mel_len, labels, n_segs) in enumerate(train_bar):
            mel = mel.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.float32)

            weights, qty_pred = forward_lora(encoder, predictor, mel, mel_len, device, dtype)

            T = weights.shape[0]
            if labels.shape[0] != T:
                labels = labels[:T] if labels.shape[0] > T else \
                    torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

            loss, qty, bnd, cnt = cif_loss(weights, labels, n_segs, qty_pred,
                                            args.lambda_qty, args.lambda_bnd, args.lambda_count)
            (loss / args.accum_steps).backward()

            epoch_loss += loss.item()
            epoch_qty += qty
            epoch_bnd += bnd
            epoch_cnt += cnt
            n_samples += 1

            if (step + 1) % args.accum_steps == 0:
                last_gnorm = nn.utils.clip_grad_norm_(
                    list(p for p in encoder.parameters() if p.requires_grad) +
                    list(predictor.parameters()),
                    float("inf"),
                ).item()
                epoch_gnorm += last_gnorm
                n_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            train_bar.set_postfix(
                loss=f"{epoch_loss/n_samples:.4f}",
                qty=f"{epoch_qty/n_samples:.4f}",
                cnt=f"{epoch_cnt/n_samples:.4f}",
                bnd=f"{epoch_bnd/n_samples:.4f}",
                gnorm=f"{last_gnorm:.3f}",
            )

        last_gnorm = nn.utils.clip_grad_norm_(
            list(p for p in encoder.parameters() if p.requires_grad) +
            list(predictor.parameters()),
            float("inf"),
        ).item()
        epoch_gnorm += last_gnorm
        n_steps += 1
        optimizer.step()
        optimizer.zero_grad()

        train_loss = epoch_loss / max(n_samples, 1)
        avg_gnorm = epoch_gnorm / max(n_steps, 1)

        # ── validation ──
        val_loss = val_qty = val_bnd = val_cnt = 0.0
        if val_loader is not None:
            encoder.eval()
            predictor.eval()
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"  Val   {epoch}/{args.epochs}",
                               leave=False, unit="sample")
                for mel, mel_len, labels, n_segs in val_bar:
                    mel = mel.to(device=device, dtype=dtype)
                    labels = labels.to(device=device, dtype=torch.float32)

                    weights, qty_pred = forward_lora(encoder, predictor, mel, mel_len, device, dtype)

                    T = weights.shape[0]
                    if labels.shape[0] != T:
                        labels = labels[:T] if labels.shape[0] > T else \
                            torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

                    _, qty, bnd, cnt = cif_loss(weights, labels, n_segs, qty_pred,
                                                args.lambda_qty, args.lambda_bnd, args.lambda_count)
                    val_qty += qty
                    val_bnd += bnd
                    val_cnt += cnt
                    val_loss += args.lambda_qty * qty + args.lambda_bnd * bnd + args.lambda_count * cnt

            n_val = len(val_loader)
            val_loss /= n_val
            val_qty /= n_val
            val_bnd /= n_val
            val_cnt /= n_val
            val_tag = f"  val={val_loss:.4f} (qty={val_qty:.4f} cnt={val_cnt:.4f} bnd={val_bnd:.4f})"
        else:
            val_tag = ""

        epoch_bar.write(
            f"Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f} "
            f"(qty={epoch_qty/n_samples:.4f} cnt={epoch_cnt/n_samples:.4f} bnd={epoch_bnd/n_samples:.4f})  "
            f"gnorm={avg_gnorm:.3f}"
            f"{val_tag}"
        )

        log_file.write(json.dumps({
            "epoch": epoch, "global_step": global_step,
            "train_loss": train_loss,
            "train_qty": epoch_qty / n_samples,
            "train_cnt": epoch_cnt / n_samples,
            "train_bnd": epoch_bnd / n_samples,
            "gnorm": avg_gnorm,
            "val_loss": val_loss if val_loader else None,
            "val_qty": val_qty if val_loader else None,
            "val_cnt": val_cnt if val_loader else None,
            "val_bnd": val_bnd if val_loader else None,
        }) + "\n")
        log_file.flush()

        monitor = val_loss if val_loader else train_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            _save_ckpt("lora_best", epoch, train_loss, val_loss)
            print(f"  → best checkpoint saved (val_loss={monitor:.4f})")

        if epoch % args.save_every == 0:
            _save_ckpt(f"lora_epoch{epoch:03d}", epoch, train_loss, val_loss)

    log_file.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CIF + Encoder LoRA 학습")
    parser.add_argument("--train-data",
                        default=str(Path(__file__).resolve().parent / "data" / "train.jsonl"))
    parser.add_argument("--val-data",
                        default=str(Path(__file__).resolve().parent / "data" / "val.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--ckpt-dir",
                        default=str(Path(__file__).resolve().parent / "checkpoints" / "lora"))
    parser.add_argument("--resume", default=None, metavar="CKPT")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA 적용할 모듈 이름")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=1e-4,
                        help="LoRA adapter learning rate")

    # CIF head
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="CIF predictor learning rate")

    # Loss
    parser.add_argument("--lambda-qty", type=float, default=1.0)
    parser.add_argument("--lambda-bnd", type=float, default=1.0)
    parser.add_argument("--lambda-count", type=float, default=1.0,
                        help="global count head loss weight (qty_pred - n_segs)²")

    # 학습
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--accum-steps", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--balanced", action="store_true",
                        help="n_segs 별 균등 샘플링 (WeightedRandomSampler)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
