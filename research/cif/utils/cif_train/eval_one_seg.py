"""
CIF one-seg 모델 평가 스크립트.

두 가지 평가 모드 (--mode):

  one_seg (기본):
    one_seg_data/test.jsonl 기준. 트림된 청크에서 모델이 처음으로
    Σw ≥ 1이 되는 시점과 실제 seg_t를 비교.
    지표: fire_acc, pos_mae, pos_acc@2f / @5f

  streaming:
    data/test.jsonl 기준 (원본 멀티-SEG 오디오).
    스트리밍 시뮬레이션: Σw≥1 → commit → 버퍼 리셋을 반복해
    감지된 경계 목록과 정답 seg_timestamps를 비교.
    지표: count_acc, count_mae, pos_mae, per-n_segs 분류

실행 예시:
  python core/cif/eval_one_seg.py --ckpt checkpoints/one_seg/cif_best.pt
  python core/cif/eval_one_seg.py --ckpt checkpoints/one_seg/cif_best.pt --mode streaming
  python core/cif/eval_one_seg.py --ckpt checkpoints/one_seg/cif_best.pt --mode streaming \\
      --test-data core/cif/data/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-ASR"
sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from train_one_seg import (
    CIFWeightPredictor,
    CIFMelPredictor,
    load_audio_16k,
    mel_to_encoder_frames,
    load_encoder,
    make_uniform_label,
)


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str, model_path: str, dtype: torch.dtype):
    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt.get("mode", "encoder")
    print(f"checkpoint: {ckpt_path}  mode={mode}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss','?'):.4f}")

    if mode == "mel":
        encoder = None
        predictor = CIFMelPredictor(
            n_mel=128,
            hidden=ckpt.get("mel_hidden", 256),
        ).to(device)
    else:
        encoder = load_encoder(model_path, device, dtype)
        predictor = CIFWeightPredictor(
            d_model=2048,
            hidden=ckpt.get("hidden", 128),
            kernel_size=ckpt.get("kernel_size", 7),
        ).to(device)
    predictor.load_state_dict(ckpt["state_dict"], strict=False)
    predictor.eval()
    return encoder, predictor


# ---------------------------------------------------------------------------
# 단일 오디오 → weight 추론
# ---------------------------------------------------------------------------

def run_model(audio: np.ndarray, encoder, predictor, fe, device, dtype) -> tuple[np.ndarray, float]:
    """오디오 배열을 받아 weight 배열과 frame_duration(초)을 반환."""
    dur = len(audio) / 16000.0
    out = fe(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    mel     = out["input_features"].squeeze(0)
    mel_len = int(out["attention_mask"].squeeze(0).sum().item())
    mel     = mel[:, :mel_len]

    mel_dev = mel.to(device=device, dtype=dtype)

    if encoder is None:
        # mel mode: Conv1d on mel directly
        with torch.no_grad():
            weights = predictor(mel_dev.float()).cpu().numpy()
        n_frames = len(weights)
    else:
        n_enc = mel_to_encoder_frames(mel_len)
        with torch.no_grad():
            from qwen_asr.core.transformers_backend import Qwen3ASRConfig
            enc_out = encoder(
                mel_dev,
                feature_lens=torch.tensor([mel_len], device=device, dtype=torch.long),
            )
        features = enc_out.last_hidden_state.squeeze(0).float()
        with torch.no_grad():
            weights = predictor(features).cpu().numpy()
        n_frames = n_enc

    frame_dur = dur / max(n_frames, 1)
    return weights, frame_dur


def first_fire(weights: np.ndarray) -> int | None:
    """누적합이 처음으로 1.0을 넘는 프레임 인덱스. 못 넘으면 None."""
    acc = np.cumsum(weights)
    idx = np.where(acc >= 1.0)[0]
    return int(idx[0]) if len(idx) > 0 else None


# ---------------------------------------------------------------------------
# Mode 1: one_seg eval
# ---------------------------------------------------------------------------

def eval_one_seg(args, encoder, predictor, fe, device, dtype):
    with open(args.test_data) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"one_seg eval  samples={len(records)}\n")

    fire_acc = 0          # 모델이 실제로 fire한 비율
    pos_mae_sum = 0.0
    pos_acc_2f = 0        # |fire_frame - seg_frame| <= 2
    pos_acc_5f = 0        # |fire_frame - seg_frame| <= 5
    bnd_loss_sum = 0.0
    qty_loss_sum = 0.0
    n_valid = n_fired = 0

    bar = tqdm(records, desc="one_seg eval", unit="sample")
    for rec in bar:
        try:
            start = rec.get("start", 0.0)
            end   = rec.get("end")
            audio = load_audio_16k(rec["audio"], start=start, end=end)
            dur   = len(audio) / 16000.0

            weights, enc_frame_dur = run_model(audio, encoder, predictor, fe, device, dtype)
            n_enc = len(weights)

            seg_t     = rec["seg_t"]
            seg_frame = min(max(round(seg_t * n_enc / dur), 0), n_enc - 1)
            label     = make_uniform_label(n_enc, seg_frame)
            label_np  = label.numpy()

            bnd_loss_sum += float(np.mean((weights - label_np) ** 2))
            qty_loss_sum += float((weights.sum() - 1.0) ** 2)

            fire_frame = first_fire(weights)
            n_valid += 1

            if fire_frame is not None:
                n_fired += 1
                err = abs(fire_frame - seg_frame)
                pos_mae_sum  += err
                pos_acc_2f   += int(err <= 2)
                pos_acc_5f   += int(err <= 5)

            bar.set_postfix(
                fire=f"{n_fired/n_valid:.3f}",
                pos_mae=f"{pos_mae_sum/max(n_fired,1):.2f}",
            )

        except Exception as e:
            tqdm.write(f"  SKIP {rec['audio']}: {e}")

    if n_valid == 0:
        print("유효한 샘플 없음")
        return

    enc_frame_ms = enc_frame_dur * 1000  # 마지막 샘플 기준 근사값

    print(f"\n{'='*52}")
    print(f"  samples       : {n_valid}")
    print(f"  qty_loss      : {qty_loss_sum/n_valid:.4f}  (Σw - 1)²")
    print(f"  bnd_loss      : {bnd_loss_sum/n_valid:.4f}  MSE(w, label)")
    print(f"{'─'*52}")
    print(f"  fire_acc      : {n_fired/n_valid:.3f}  ({n_fired}/{n_valid})  [Σw≥1 도달]")
    print(f"  pos_mae       : {pos_mae_sum/max(n_fired,1):.2f} frames  "
          f"≈ {pos_mae_sum/max(n_fired,1)*enc_frame_ms:.0f} ms")
    print(f"  pos_acc @2f   : {pos_acc_2f/max(n_fired,1):.3f}  ({pos_acc_2f}/{n_fired})")
    print(f"  pos_acc @5f   : {pos_acc_5f/max(n_fired,1):.3f}  ({pos_acc_5f}/{n_fired})")
    print(f"{'='*52}")


# ---------------------------------------------------------------------------
# Mode 2: streaming simulation
# ---------------------------------------------------------------------------

def simulate_streaming(
    audio: np.ndarray,
    encoder, predictor, fe, device, dtype,
    max_commits: int = 10,
    max_window: float | None = None,
) -> list[float]:
    """
    오디오 전체를 스트리밍처럼 처리.
    Σw≥1 → commit_time 기록 → 그 시점부터 다시 시작.

    max_window: 매 iteration마다 cursor 이후 최대 몇 초까지만 모델에 넘길지.
                None이면 파일 끝까지 전부 넘김.
                학습이 trimmed clip 기준이므로 학습 최대 clip 길이와 맞추는 게 좋음.
    반환: 감지된 commit time 목록(초, 오디오 시작 기준 절대값).
    """
    dur = len(audio) / 16000.0
    commits = []
    cursor = 0.0   # 마지막 commit 이후 오디오 시작점 (초)

    for _ in range(max_commits):
        start_sample = int(cursor * 16000)
        if max_window is not None:
            end_sample = min(start_sample + int(max_window * 16000), len(audio))
            chunk = audio[start_sample:end_sample]
        else:
            chunk = audio[start_sample:]
        if len(chunk) < 1600:   # 0.1s 미만이면 종료
            break

        weights, enc_frame_dur = run_model(chunk, encoder, predictor, fe, device, dtype)
        fire_frame = first_fire(weights)

        if fire_frame is None:
            break

        fire_time_abs = cursor + fire_frame * enc_frame_dur
        commits.append(fire_time_abs)
        cursor = fire_time_abs

    return commits


def eval_streaming(args, encoder, predictor, fe, device, dtype):
    with open(args.test_data) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"streaming eval  samples={len(records)}\n")

    count_acc = 0
    count_mae_sum = 0
    pos_mae_sum = 0.0
    pos_n = 0
    n_valid = 0
    by_nsegs: dict[int, dict] = {}

    bar = tqdm(records, desc="streaming eval", unit="sample")
    for rec in bar:
        try:
            audio  = load_audio_16k(rec["audio"])
            dur    = len(audio) / 16000.0
            gt_ts  = rec["seg_timestamps"]
            n_segs = len(gt_ts)

            commits = simulate_streaming(
                audio, encoder, predictor, fe, device, dtype,
                max_commits=n_segs + 2,
                max_window=args.max_window,
            )
            n_pred = len(commits)

            count_acc     += int(n_pred == n_segs)
            count_mae_sum += abs(n_pred - n_segs)

            # count 일치할 때 위치 오차 (정렬 후 MAE)
            if n_pred == n_segs and n_segs > 0:
                errs = [abs(c - g) for c, g in zip(sorted(commits), sorted(gt_ts))]
                pos_mae_sum += float(np.mean(errs))
                pos_n += 1

            n_valid += 1

            bucket = by_nsegs.setdefault(n_segs, {"n": 0, "acc": 0, "mae": 0})
            bucket["n"]   += 1
            bucket["acc"] += int(n_pred == n_segs)
            bucket["mae"] += abs(n_pred - n_segs)

            bar.set_postfix(
                count_acc=f"{count_acc/n_valid:.3f}",
                pos_mae=f"{pos_mae_sum/max(pos_n,1):.3f}s",
            )

        except Exception as e:
            tqdm.write(f"  SKIP {rec['audio']}: {e}")

    if n_valid == 0:
        print("유효한 샘플 없음")
        return

    pos_str = f"{pos_mae_sum/pos_n:.3f}s  ({pos_n}개)" if pos_n > 0 else "N/A"

    print(f"\n{'='*52}")
    print(f"  samples       : {n_valid}")
    print(f"{'─'*52}")
    print(f"  count_acc     : {count_acc/n_valid:.3f}  ({count_acc}/{n_valid})")
    print(f"  count_mae     : {count_mae_sum/n_valid:.3f}")
    print(f"  pos_mae       : {pos_str}  [count 일치 샘플, 초 단위]")
    print(f"{'='*52}")

    print("\n[ n_segs별 count_acc ]")
    for k in sorted(by_nsegs):
        b = by_nsegs[k]
        print(f"  n_segs={k}  n={b['n']:4d}  "
              f"acc={b['acc']/b['n']:.3f}  mae={b['mae']/b['n']:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _one_seg = Path(__file__).resolve().parent / "one_seg_data"
    _data    = Path(__file__).resolve().parent / "data"

    parser = argparse.ArgumentParser(description="CIF one-seg 평가")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", choices=["one_seg", "streaming"], default="one_seg")
    parser.add_argument("--test-data", default=None,
                        help="미지정 시 mode에 따라 자동 선택")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-window", type=float, default=None,
                        help="streaming 모드: 매 iteration 최대 오디오 길이(초). "
                             "None=파일 끝까지. 학습 clip 길이와 맞추면 분포 일치.")
    args = parser.parse_args()

    if args.test_data is None:
        args.test_data = str(
            _one_seg / "test.jsonl" if args.mode == "one_seg"
            else _data / "test.jsonl"
        )

    dtype = torch.bfloat16
    encoder, predictor = load_model(args.ckpt, args.device, args.model, dtype)

    fe = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )

    if args.mode == "one_seg":
        eval_one_seg(args, encoder, predictor, fe, args.device, dtype)
    else:
        eval_streaming(args, encoder, predictor, fe, args.device, dtype)


if __name__ == "__main__":
    main()
