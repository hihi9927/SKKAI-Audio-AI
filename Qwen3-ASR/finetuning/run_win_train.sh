#!/usr/bin/env bash
# DailyTalk 창 데이터(DailyTalk_win) 로 SEG 파인튜닝.
# 창 = 대화 안 연속 발화 5.5개 묶음(평균 18.2초). 발화 하나짜리 샘플과 달리
# 오디오 끝 단서 없이 <SEG> 를 찾아야 한다.
set -euo pipefail
cd /home/mobility/STiTy/Qwen3-ASR/finetuning

export PYTHONPATH=/home/mobility/STiTy/Qwen3-ASR:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

PY=/home/mobility/STiTy/.venv/bin/python
OUT=./finetuning-out-en-win
mkdir -p "$OUT"

# batch_size 1 — 창이 18.2초라 mel 3000프레임. 3.4초짜리 기준의 기본값 8은 OOM.
# grad_acc 24 로 유효 배치를 맞춘다 → 에폭당 157스텝, 3에폭 472스텝.
"$PY" -u qwen3_asr_sft.py \
  --model_path  Qwen/Qwen3-ASR-1.7B \
  --train_file  ./data/DailyTalk_win/train.jsonl \
  --eval_file   ./data/DailyTalk_win/val.jsonl \
  --output_dir  "$OUT" \
  --batch_size 1 \
  --grad_acc 24 \
  --epochs 3 \
  --save_steps 50 \
  2>&1 | tee -a "$OUT/train.log"
