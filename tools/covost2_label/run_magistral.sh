#!/bin/bash
cd /home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate speech_ai
export AUTOSEG_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=local PYTHONPATH=.
OUT=core/meaning_segmentator/runs/covost2/sample300
P=core/meaning_segmentator/runs/en-multi/run13/best_prompt.txt
echo "===== magistral-16k / effort=none(native) / 예산16000 / timeout2400 / batch6 / 20문장 ====="; date
curl -s --max-time 1800 localhost:11434/api/generate \
  -d '{"model":"magistral-16k","prompt":"hi","stream":false,"keep_alive":"24h"}' >/dev/null
python tools/covost2_label/label_covost2.py --prompt $P \
  --out   $OUT/labels/covost2_20_magistral.jsonl \
  --cache $OUT/cache/segment_magistral.json \
  --model magistral-16k --workers 4 --limit 20 \
  --seg-reasoning-effort none \
  --max-tokens 16000 --timeout 2400 --batch-size 6 --cache-every 1
date; echo OSSDONE
