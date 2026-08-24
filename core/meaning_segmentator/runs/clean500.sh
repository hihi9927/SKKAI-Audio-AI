#!/bin/bash
# 미사용 FLEURS 500문장에서 best_prompt 분절 → 3타깃 BLEU
cd /home/mobility/STiTy
set -a && . .env && set +a
export PYTHONPATH=/home/mobility/STiTy
LOG=core/meaning_segmentator/runs/clean500.log

echo "[1/2] 분절 (auto_best, 500문장)" | tee -a $LOG
python -u -m core.meaning_segmentator.autoseg.eval_prompt \
  --prompt core/meaning_segmentator/runs/en-multi/run06/best_prompt.txt \
  --run-id en-multi/clean500 --split test --label auto_best \
  --t-grid 4 6 8 12 --batch-size 6 --seg-reasoning-effort medium \
  --workers 8 --budget 12 2>&1 | grep -v "HTTP Request" | tee -a $LOG
echo "[EXIT seg] ${PIPESTATUS[0]}" | tee -a $LOG

echo "[2/2] BLEU (de/ja/zh)" | tee -a $LOG
python -u -m core.meaning_segmentator.autoseg.bleu_eval \
  --run-id en-multi/clean500 --label auto_best --targets de ja zh \
  --manifest-tag clean500 --t-grid 4 6 8 12 --workers 4 2>&1 | grep -v "HTTP Request" | tee -a $LOG
echo "[EXIT bleu] ${PIPESTATUS[0]}" | tee -a $LOG
