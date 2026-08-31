#!/bin/bash
# 검증된 경로 재현: gpt_seg_en.py + prompt_en_ranked_v2 + llama3.3:70b (ollama).
# DailyTalk 랭크 라벨 20,773줄을 만든 조합 그대로다 (commit 4d00d123).
cd /home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate speech_ai
export PYTHONPATH=.
echo "===== llama3.3:70b / prompt_en_ranked_v2 / batch6 / workers4 ====="; date
curl -s --max-time 1800 localhost:11434/api/generate \
  -d '{"model":"llama3.3:70b","prompt":"hi","stream":false,"keep_alive":"24h"}' >/dev/null
python core/meaning_segmentator/utils/gpt_seg_en.py \
  --input covost2_sample300_en.json \
  --output covost2_sample300_llama33_ranked.json \
  --provider ollama --model llama3.3:70b \
  --prompt-file prompt_en_ranked_v2.txt \
  --workers 4 --batch-size 6 --save-every 6 --limit 20
date; echo LLAMA33DONE
