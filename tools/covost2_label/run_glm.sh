#!/bin/bash
# granite4 / qwen3-next 재시도. 둘 다 사고 미지원이라 effort=none 으로 간다.
cd /home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate speech_ai
export PYTHONPATH=.
OUT=core/meaning_segmentator/runs/covost2/sample300
P=core/meaning_segmentator/runs/en-multi/run13/best_prompt.txt
L=$OUT/logs
label () {
  echo "===== $1 / effort=$3 / batch6 / 20문장 ====="; date
  curl -s --max-time 1800 localhost:11434/api/generate \
    -d "{\"model\":\"$1\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"24h\"}" >/dev/null
  python tools/covost2_label/label_covost2.py --provider local --prompt $P \
    --out   $OUT/labels/covost2_20_$2.jsonl \
    --cache $OUT/cache/segment_$2.json \
    --model $1 --workers 4 --limit 20 \
    --seg-reasoning-effort $3 \
    --max-tokens 16000 --timeout 2400 --batch-size 6 --cache-every 1
  date; echo "DONE $2"; ollama stop $1 2>/dev/null
}
label glm-air-16k glm none > $L/glm.log 2>&1
echo GLMDONE; date
