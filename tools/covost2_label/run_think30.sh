#!/bin/bash
cd /home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate speech_ai
export AUTOSEG_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=local PYTHONPATH=.
OUT=core/meaning_segmentator/runs/covost2/sample300
P=core/meaning_segmentator/runs/en-multi/run13/best_prompt.txt
echo "===== qwen30t-40k thinking / 예산16000 / timeout2400 / batch3 / 30문장 ====="; date
curl -s --max-time 1800 localhost:11434/api/generate \
  -d '{"model":"qwen30t-40k","prompt":"hi","stream":false,"keep_alive":"24h"}' >/dev/null
python tools/covost2_label/label_covost2.py --prompt $P \
  --out   $OUT/labels/covost2_30_qwen30t.jsonl \
  --cache $OUT/cache/segment_qwen30t.json \
  --model qwen30t-40k --workers 8 --limit 30 \
  --max-tokens 16000 --timeout 2400 --batch-size 3 --cache-every 1
date; echo THINKDONE
