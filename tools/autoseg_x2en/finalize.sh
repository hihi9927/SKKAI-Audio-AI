#!/usr/bin/env bash
# 체인이 끝나기를 기다렸다가, 최종 test 곡선이 없는 트랙만 --final-only 로 채운다.
#
# ja-en / zh-en 은 예산 상한($25)에 걸려 이터레이션 도중 죽는다. 그 경로는 최종 평가를
# 안 거치므로 final_report.md / curve.json 이 안 생긴다. best_prompt.txt 는 남아 있으니
# 그걸로 test 100문장만 다시 채점하면 곡선이 나온다.
#
# **--final-only 는 history/best/config 를 안 건드린다** (loop.py:1358, 1727). 그래서
# 여기서 곡선을 뽑아 둬도 나중에 --resume 으로 iter 를 더 도는 데 지장이 없다.
set -u
cd /home/mobility/STiTy || exit 1
export PYTHONPATH=.
PY=.venv-autoseg/bin/python
RUNS=core/meaning_segmentator/runs
LL=$RUNS/x2en_finalize.launch.log

# 체인이 끝날 때까지 대기. 체인 스크립트가 마지막에 "chain done" 을 찍는다.
echo "[$(date '+%F %T')] 체인 종료 대기" >> $LL
while ! grep -q "chain done" $RUNS/x2en_chain.launch.log 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] 체인 종료 확인" >> $LL

for spec in "ja-en Japanese fleurs-ja-en" "zh-en Chinese fleurs-zh-en"; do
  set -- $spec; pair=$1 src=$2 ds=$3
  if [ -f "$RUNS/$pair/run02/final_report.md" ]; then
    echo "[$(date '+%F %T')] SKIP $pair — final_report.md 이미 있다" >> $LL; continue
  fi
  if [ ! -f "$RUNS/$pair/run02/best_prompt.txt" ]; then
    echo "[$(date '+%F %T')] SKIP $pair — best_prompt.txt 가 없다" >> $LL; continue
  fi
  echo "[$(date '+%F %T')] starting $pair/run02 --final-only" >> $LL
  $PY -m core.meaning_segmentator.autoseg.loop \
      --dataset "$ds" --src-lang "$src" --tgt-lang English \
      --pair-id "$pair" --run-id run02 \
      --model gpt-5-mini --provider openai \
      --agent-reasoning-effort none --seg-reasoning-effort none \
      --train 40 --dev 265 --test 100 \
      --budget 10 --workers 24 \
      --translate-backend local \
      --adequacy-backend cometkiwi --consistency-backend nli \
      --final-only >> "$RUNS/${pair}_run02.log" 2>&1
  rc=$?   # $(date) 서브셸이 $? 를 덮어쓰기 전에 받는다
  echo "[$(date '+%F %T')] $pair --final-only exit=$rc" >> $LL
done
echo "[$(date '+%F %T')] finalize done" >> $LL
