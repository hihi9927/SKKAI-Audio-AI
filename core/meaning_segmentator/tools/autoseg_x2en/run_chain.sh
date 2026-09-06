#!/usr/bin/env bash
# x2en autoseg 프롬프트 루프 3트랙 순차 실행 — de-en(재개) → ja-en → zh-en.
#
# 설정은 en-multi/run13 을 그대로 복제한다 (experiment/README.md 의 "공통 설정").
# --min-gap / --t-grid / --t-floor 는 **일부러 안 준다** — 강제정렬 산출물
# (evaluation/ast/manifests/*_unittimes.json)에서 언어마다 유도된다.
#
# 재개는 자동이다: runs/<pair>/run02/history.json 이 있으면 --resume 을 붙인다.
# --fresh 는 어디에도 안 쓴다 (rmtree 라 캐시가 날아간다).
#
# 한 트랙이 죽어도 다음 트랙은 계속 간다 — 트랙끼리는 독립이다.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$REPO" || exit 1

RUNS=core/meaning_segmentator/experiment/artifacts
# PYTHONPATH 를 "." 로 **덮어쓴다**. 이 머신은 ROS humble 이 PYTHONPATH 에 끼어 있고
# 그게 venv 의 site-packages 보다 먼저 검색된다. 비우면 core 패키지를 못 찾으므로
# 비우는 게 아니라 "." 하나로 갈아끼운다.
export PYTHONPATH=.
# **.venv 가 아니라 .venv-autoseg 다.** .venv 에는 langcodes 와 unbabel-comet 이 없어
# to_lang_code 에서 ModuleNotFoundError 로 즉사한다 (2026-09-01 실측).
PY=.venv-autoseg/bin/python

run_track() {
  local pair=$1 src=$2 ds=$3 budget=$4 iters=$5
  local rundir=$RUNS/$pair/run02
  local log=$RUNS/${pair}_run02.log
  local llog=$RUNS/${pair}_run02.launch.log
  local extra=()

  # 이어갈 이력이 있으면 재개. 중간에 죽어도 이 스크립트를 다시 돌리면 이어붙는다.
  if [ -f "$rundir/history.json" ]; then
    extra+=(--resume)
    # 대기 중인 개정본이 없으면 loop.py 가 best 를 다시 평가한다 — 이터레이션 하나가
    # 통째로 헛돈다(3시간·$2.6). 그건 사고이지 정상 재개가 아니므로 여기서 멈춘다.
    # 알고도 강행하려면 ALLOW_LOST_REVISION=1 을 주면 된다.
    if [ ! -f "$rundir/next_prompt.txt" ] \
       && [ ! -f "$rundir/iter_$(printf '%02d' "$(jq length "$rundir/history.json")")/prompt.txt" ] \
       && [ "${ALLOW_LOST_REVISION:-0}" != "1" ]; then
      echo "[$(date '+%F %T')] SKIP $pair/run02 — next_prompt.txt 가 없다(개정본 유실)." >> "$llog"
      echo "                    다른 머신에서 복사하거나 ALLOW_LOST_REVISION=1 로 강행할 것" >> "$llog"
      return 0
    fi
  fi

  echo "[$(date '+%F %T')] starting $pair/run02 ($ds) budget=\$$budget iters=$iters ${extra[*]:-fresh}" >> "$llog"
  $PY -m core.meaning_segmentator.autoseg.loop \
      --dataset "$ds" --src-lang "$src" --tgt-lang English \
      --pair-id "$pair" --run-id run02 \
      --model gpt-5-mini --provider openai \
      --agent-reasoning-effort none --seg-reasoning-effort none \
      --iterations "$iters" --train 40 --dev 265 --test 100 \
      --patience 5 --budget "$budget" --workers 24 \
      --translate-backend local \
      --adequacy-backend cometkiwi --consistency-backend nli --adopt-se-mult 0.5 \
      "${extra[@]}" >> "$log" 2>&1
  # **rc 를 먼저 받는다.** `exit=$?` 로 쓰면 같은 줄의 `$(date)` 서브셸이 먼저 돌아
  # `$?` 를 date 의 종료코드(0)로 덮어쓴다 — 죽은 트랙이 exit=0 으로 남았다.
  local rc=$?
  echo "[$(date '+%F %T')] exit=$rc" >> "$llog"
  return 0
}

# de-en 은 iter_00 에서 이미 $2.62 를 썼다. 트랙 총액을 $25 로 고정하려고 남은 예산만 준다.
#
# **de-en 만 6회다.** 크래시로 iter_00 의 개정본(next_prompt.txt)을 잃어 iter_01 이
# v0 를 다시 평가하며 슬롯 하나를 쓴다 — 캐시가 있어 돈은 거의 안 들지만(분절·번역이
# 전부 히트, 판정자 ~$0.13 뿐) 슬롯은 없어진다. 6회로 올려야 실제 개정 평가가 4회가
# 되어 en-multi/run13 과 동수로 읽힌다. ja/zh 는 처음부터 도니 5회 그대로.
run_track de-en German   fleurs-de-en 22.4 6
run_track ja-en Japanese fleurs-ja-en 25   5
run_track zh-en Chinese  fleurs-zh-en 25   5

echo "[$(date '+%F %T')] chain done" >> $RUNS/x2en_chain.launch.log
