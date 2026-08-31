#!/bin/bash
cd /home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate speech_ai
export PYTHONPATH=.
OUT=core/meaning_segmentator/runs/covost2/sample300
P=core/meaning_segmentator/runs/en-multi/run13/best_prompt.txt
echo "===== oss120-16k / effort=high / 예산16000 / timeout2400 / batch6 / 20문장 ====="; date
curl -s --max-time 1800 localhost:11434/api/generate \
  -d '{"model":"oss120-16k","prompt":"hi","stream":false,"keep_alive":"24h"}' >/dev/null
python tools/covost2_label/label_covost2.py --provider local --prompt $P \
  --out   $OUT/labels/covost2_20_oss120_high.jsonl \
  --cache $OUT/cache/segment_oss120_high.json \
  --model oss120-16k --workers 4 --limit 20 \
  --seg-reasoning-effort high \
  --max-tokens 16000 --timeout 2400 --batch-size 6 --cache-every 1
date; echo OSSDONE
