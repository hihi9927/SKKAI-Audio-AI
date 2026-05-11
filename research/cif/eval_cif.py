"""
CIF Weight Predictor 테스트셋 평가 스크립트.

Loss 지표:
  quantity_loss : (Σw_t - n_segs)²
  boundary_loss : MSE(w_t, uniform_label_t)
  total_loss    : λ_qty * qty + λ_bnd * bnd

검출 지표 (누적 기반):
  n_pred        : 예측된 경계 수 (floor(Σw_t) 기준)
  n_gt          : 정답 경계 수
  count_acc     : n_pred == n_gt 인 샘플 비율
  count_mae     : |n_pred - n_gt| 평균

실행 예시:
  python core/cif/eval_cif.py --ckpt checkpoints/mel/cif_best.pt
  python core/cif/eval_cif.py --ckpt checkpoints/encoder/cif_best.pt --test-data core/cif/data/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from tqdm import tqdm
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
    make_uniform_label,
)


def detect_by_kpeak(weights_np: np.ndarray, k: int, min_distance: int = 3) -> np.ndarray:
    """qty_pred를 k로 받아 weight 곡선 상위 k개 극대점 반환."""
    if k <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(weights_np, distance=min_distance)
    if len(peaks) == 0:
        # 극대점 없으면 전체 인덱스에서 top-k
        peaks = np.arange(len(weights_np))
    if len(peaks) <= k:
        return np.sort(peaks)
    top_k = np.argsort(weights_np[peaks])[-k:]
    return np.sort(peaks[top_k])


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


def evaluate(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ── checkpoint ──
    ckpt = torch.load(args.ckpt, map_location=device)
    mode = ckpt.get("mode", args.mode)
    epoch = ckpt.get("epoch", "?")
    print(f"checkpoint : {args.ckpt}")
    print(f"mode={mode}  epoch={epoch}  val_loss={ckpt.get('val_loss', '?'):.4f}")

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

    # train.py는 "state_dict", train_lora.py는 "predictor" 키 사용
    predictor.load_state_dict(ckpt.get("state_dict") or ckpt["predictor"], strict=False)
    predictor.eval()

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )

    with open(args.test_data) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"test samples: {len(records)}\n")

    # 누적 지표
    total_loss = qty_loss_sum = bnd_loss_sum = 0.0
    count_acc = 0
    count_mae_sum = 0
    qty_head_acc = 0
    qty_head_mae = 0
    kpeak_count_acc = 0
    kpeak_mae_sum = 0
    kpeak_pos_mae_sum = 0.0
    kpeak_pos_n = 0
    n_valid = 0

    # n_segs 별 분류
    by_nsegs: dict[int, list] = {}
    by_nsegs_kpeak: dict[int, list] = {}

    bar = tqdm(records, desc="Eval", unit="sample")
    for rec in bar:
        try:
            audio = load_audio_16k(rec["audio"])
            dur = len(audio) / 16000.0

            out = feature_extractor(audio, sampling_rate=16000,
                                    return_tensors="pt", return_attention_mask=True)
            mel = out["input_features"].squeeze(0)
            mel_len = int(out["attention_mask"].squeeze(0).sum().item())
            mel = mel[:, :mel_len]
            n_enc = mel_to_encoder_frames(mel_len)

            seg_ts = rec["seg_timestamps"]
            n_segs = len(seg_ts)
            seg_frames = [min(max(round(ts * n_enc / dur), 0), n_enc - 1) for ts in seg_ts]

            labels = make_uniform_label(n_enc, seg_frames).to(device)

            with torch.no_grad():
                mel_dev = mel.to(device=device, dtype=dtype)
                weights, qty_pred = _forward(encoder, predictor, mel_dev, mel_len, device, dtype)

            T = weights.shape[0]
            if labels.shape[0] != T:
                labels = labels[:T] if labels.shape[0] > T else \
                    torch.cat([labels, torch.zeros(T - labels.shape[0], device=device)])

            qty  = (weights.sum() - n_segs) ** 2
            bnd  = nn.functional.mse_loss(weights, labels)
            loss = args.lambda_qty * qty + args.lambda_bnd * bnd
            if qty_pred is not None:
                loss = loss + args.lambda_count * (qty_pred - float(n_segs)) ** 2

            # 누적 기반 경계 수 예측 (weights.sum() 기준)
            weights_np_arr = weights.float().cpu().numpy()
            acc_np = weights_np_arr.cumsum()
            n_pred = int(np.floor(acc_np[-1]))

            # qty_head 예측값 (별도 집계)
            qty_head_pred = round(qty_pred.item()) if qty_pred is not None else None

            # k-peak 기반 경계 수/위치 예측
            kpeak_k = qty_head_pred if qty_head_pred is not None else n_pred
            kpeak_frames = detect_by_kpeak(weights_np_arr, kpeak_k, args.peak_min_dist)
            kpeak_pred = len(kpeak_frames)

            total_loss    += loss.item()
            qty_loss_sum  += qty.item()
            bnd_loss_sum  += bnd.item()
            count_acc     += int(n_pred == n_segs)
            count_mae_sum += abs(n_pred - n_segs)
            if qty_head_pred is not None:
                qty_head_acc  += int(qty_head_pred == n_segs)
                qty_head_mae  += abs(qty_head_pred - n_segs)
            kpeak_count_acc += int(kpeak_pred == n_segs)
            kpeak_mae_sum   += abs(kpeak_pred - n_segs)
            # 위치 오차: count 맞았을 때만 (정렬 후 프레임 단위 MAE)
            if kpeak_pred == n_segs and n_segs > 0:
                pos_err = float(np.mean(np.abs(kpeak_frames - np.sort(seg_frames))))
                kpeak_pos_mae_sum += pos_err
                kpeak_pos_n += 1
            n_valid       += 1

            # n_segs 별 오차 기록
            by_nsegs.setdefault(n_segs, []).append(abs(n_pred - n_segs))
            by_nsegs_kpeak.setdefault(n_segs, []).append(abs(kpeak_pred - n_segs))

            bar.set_postfix(
                loss=f"{total_loss/n_valid:.4f}",
                acc=f"{count_acc/n_valid:.3f}",
            )

        except Exception as e:
            tqdm.write(f"  SKIP {rec['audio']}: {e}")

    if n_valid == 0:
        print("유효한 샘플 없음")
        return

    kpeak_pos_str = (f"{kpeak_pos_mae_sum/kpeak_pos_n:.2f}"
                     if kpeak_pos_n > 0 else "N/A")

    print(f"\n{'='*56}")
    print(f"  samples        : {n_valid}")
    print(f"  total_loss     : {total_loss/n_valid:.4f}")
    print(f"  qty_loss       : {qty_loss_sum/n_valid:.4f}  (Σw - n_segs)²")
    print(f"  bnd_loss       : {bnd_loss_sum/n_valid:.4f}  MSE(w, label)")
    print(f"{'─'*56}")
    print(f"  count_acc [CIF]    : {count_acc/n_valid:.3f}  ({count_acc}/{n_valid})")
    print(f"  count_mae [CIF]    : {count_mae_sum/n_valid:.3f}")
    print(f"  count_acc [head]   : {qty_head_acc/n_valid:.3f}  ({qty_head_acc}/{n_valid})")
    print(f"  count_mae [head]   : {qty_head_mae/n_valid:.3f}")
    print(f"  count_acc [kpeak]  : {kpeak_count_acc/n_valid:.3f}  ({kpeak_count_acc}/{n_valid})")
    print(f"  count_mae [kpeak]  : {kpeak_mae_sum/n_valid:.3f}")
    print(f"  pos_mae   [kpeak]  : {kpeak_pos_str}  (count 일치 샘플 {kpeak_pos_n}개, 단위: enc frames)")
    print(f"{'='*56}")

    print("\n[ n_segs별 count_acc  /  CIF vs kpeak ]")
    for k in sorted(by_nsegs):
        errs_cif   = by_nsegs[k]
        errs_kpeak = by_nsegs_kpeak.get(k, [])
        acc_cif   = sum(e == 0 for e in errs_cif) / len(errs_cif)
        acc_kpeak = sum(e == 0 for e in errs_kpeak) / len(errs_kpeak) if errs_kpeak else float("nan")
        print(f"  n_segs={k}  n={len(errs_cif):4d}"
              f"  cif={acc_cif:.3f}(mae={np.mean(errs_cif):.2f})"
              f"  kpeak={acc_kpeak:.3f}(mae={np.mean(errs_kpeak):.2f})")


def main():
    parser = argparse.ArgumentParser(description="CIF 테스트셋 평가")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test-data",
                        default=str(Path(__file__).resolve().parent / "data" / "test.jsonl"))
    parser.add_argument("--mode", choices=["encoder", "mel"], default="encoder")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--mel-hidden", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-qty", type=float, default=1.0)
    parser.add_argument("--lambda-bnd", type=float, default=1.0)
    parser.add_argument("--lambda-count", type=float, default=1.0)
    parser.add_argument("--peak-min-dist", type=int, default=3,
                        help="k-peak 검출 시 인접 극대점 최소 간격 (encoder frames)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
