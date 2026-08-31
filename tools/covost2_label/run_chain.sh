#!/bin/bash
# CoVoST2 라벨링 모델 비교 — 20문장 고정, 순차 실행.
# GPU 는 한 번에 한 모델만 쓰게 직렬화한다. 다운로드는 병렬로 따로 돈다.
cd /home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate speech_ai
export PYTHONPATH=.
OUT=core/meaning_segmentator/runs/covost2/sample300
P=core/meaning_segmentator/runs/en-multi/run13/best_prompt.txt
L=$OUT/logs

# $1=모델태그 $2=슬러그 $3=effort
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
  date; echo "DONE $2"
  ollama stop $1 2>/dev/null
}

# 1) magistral 끝나기를 기다린다
while pgrep -f "label_covost2.py .*magistral" >/dev/null; do sleep 30; done

# 2) 다운로드 0 짜리 먼저
label oss120-16k oss120_low low                 > $L/oss120_low.log 2>&1

# 3) granite4 — pull 완료 대기 후 16k 컨텍스트로 등록
while ! ollama list | grep -q "granite4:small-h"; do sleep 60; done
printf 'FROM granite4:small-h\nPARAMETER num_ctx 16384\n' > /tmp/Modelfile.granite
ollama create granite-16k -f /tmp/Modelfile.granite
label granite-16k granite low                   > $L/granite.log 2>&1

# 4) qwen3-next 80B-A3B instruct
while ! ollama list | grep -q "qwen3-next"; do sleep 60; done
QTAG=$(ollama list | awk '/qwen3-next/{print $1; exit}')
printf "FROM $QTAG\nPARAMETER num_ctx 16384\n" > /tmp/Modelfile.qnext
ollama create qnext-16k -f /tmp/Modelfile.qnext
label qnext-16k qnext low                       > $L/qnext.log 2>&1

echo CHAINDONE; date
