#!/usr/bin/env bash
# 체인이 끝나기를 기다렸다가, 예산 상한에 걸려 멈춘 트랙을 --resume 으로 끝까지 돌린다.
#
# ja-en / zh-en 은 트랙당 $25 상한에 걸려 이터레이션 도중 죽었다. 예산 가드는 프로세스마다
# 0 에서 시작하므로 --budget 을 새로 주면 그만큼 더 돈다. --iterations 5 는 그대로라
# 세 트랙 모두 "iter_00 + 실제 개정 4회" 로 맞춰진다 (de-en 은 중복 슬롯 때문에 6이었다).
#
# 이터레이션을 다 채운 트랙이면 loop.py 가 알아서 최종 평가로 넘어가므로
# (`start_it >= args.iterations`), 이 스크립트 하나가 --final-only 까지 겸한다.
#
# --fresh 는 쓰지 않는다. 재개 여부는 history.json 유무로 판단한다.
set -u
cd /home/mobility/STiTy || exit 1
export PYTHONPATH=.
PY=.venv-autoseg/bin/python
RUNS=core/meaning_segmentator/experiment/artifacts
LL=$RUNS/x2en_resume.launch.log

echo "[$(date '+%F %T')] 체인 종료 대기" >> $LL
while ! grep -q "chain done" $RUNS/x2en_chain.launch.log 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] 체인 종료 확인" >> $LL

# 트랙 남은 예산: ja 는 iter 2~4 세 번(이터당 ~$9.8) + 최종 test, zh 는 죽은 지점부터
# + 최종 test. 상한은 폭주 가드이지 목표액이 아니다 — 실측 예상은 ja ~$30, zh ~$14.
for spec in "ja-en Japanese fleurs-ja-en 45" "zh-en Chinese fleurs-zh-en 30"; do
  set -- $spec; pair=$1 src=$2 ds=$3 budget=$4
  rundir=$RUNS/$pair/run02
  if [ -f "$rundir/final_report.md" ]; then
    echo "[$(date '+%F %T')] SKIP $pair — final_report.md 이미 있다" >> $LL; continue
  fi
  if [ ! -f "$rundir/history.json" ]; then
    echo "[$(date '+%F %T')] SKIP $pair — history.json 이 없다(재개할 것이 없다)" >> $LL; continue
  fi
  echo "[$(date '+%F %T')] starting $pair/run02 --resume budget=\$$budget (history $(jq length "$rundir/history.json")건)" >> $LL
  $PY -m core.meaning_segmentator.autoseg.loop \
      --dataset "$ds" --src-lang "$src" --tgt-lang English \
      --pair-id "$pair" --run-id run02 \
      --model gpt-5-mini --provider openai \
      --agent-reasoning-effort none --seg-reasoning-effort none \
      --iterations 5 --train 40 --dev 265 --test 100 \
      --patience 5 --budget "$budget" --workers 24 \
      --translate-backend local \
      --adequacy-backend cometkiwi --consistency-backend nli --adopt-se-mult 0.5 \
      --resume >> "$RUNS/${pair}_run02.log" 2>&1
  rc=$?   # $(date) 서브셸이 $? 를 덮어쓰기 전에 받는다
  echo "[$(date '+%F %T')] $pair --resume exit=$rc" >> $LL
done
echo "[$(date '+%F %T')] resume done" >> $LL
