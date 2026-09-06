#!/bin/bash
# 평가 감시견 — 죽은 타깃을 낮은 설정으로 되살리고, 끝까지 못 간 체인을 대신 마무리한다.
#
# **사후 수습만 한다.** 오케스트레이터(14/19)가 살아 있는 동안에는 아무것도 안 건드린다.
# 타깃 사이 전환 구간에 끼어들면 같은 타깃을 두 프로세스가 돌려 `bleu/<tgt>.json` 과
# `cache/translate_<tgt>.json` 을 동시에 쓰게 되므로, **둘 다 사라진 뒤에만** 움직인다.
#
# 되살릴 때는 워커·배치를 낮춘다 (OOM 이 사인이었을 가능성이 가장 크므로).
# 번역 캐시는 타깃별로 남아 있어 재시작해도 번역은 재사용되고 채점만 다시 돈다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
GRID="2 3 4 6"
BASE="punct alignatt mu_prefix causal_align syntax"
LOG=$F/logs/watchdog.log
declare -A tries=( [zh]=0 [de]=0 [ja]=0 )

say () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

orchestrators_alive () {
  pgrep -f "autoseg.bleu_eval" >/dev/null && return 0
  pgrep -f "14_full_eval.sh"  >/dev/null && return 0
  pgrep -f "19_full_eval_ja.sh" >/dev/null && return 0
  return 1
}

rerun () {   # <타깃>  — 낮춘 설정으로
  say "재실행 $1 (workers 8 / batch 16, 시도 $(( ${tries[$1]} + 1 ))/3)"
  $PY -u -m core.meaning_segmentator.autoseg.bleu_eval \
    --run-id covost2/full --label auto_run13_mg1 --split test \
    --dataset covost2 --manifest-tag full --targets $1 \
    --t-grid $GRID --src-spaced 1 \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 16 \
    --workers 8 --baselines $BASE --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    >> $F/logs/bleu_eval_$1.log 2>&1
  say "재실행 $1 exit=$?"
}

finish () {
  say "COMET (없는 조건만)"
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id covost2/full --dataset covost2 --manifest-tag full --src en \
    --label auto_run13_mg1 --split test --targets zh de ja --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 32 >> $F/logs/comet_full.log 2>&1
  say "COMET exit=$?"
  $PY core/meaning_segmentator/autoseg/baselines/plot_tradeoff.py \
    --run-id covost2/full --targets zh de ja --metric comet \
    --out tradeoff_covost2_full_comet \
    --title "CoVoST2 en->X test 15430 (min_gap=1)" >> $LOG 2>&1
  say "그래프 exit=$?"
  rm -f $ST/full_eval.failed
  mark full_eval.done "감시견이 마무리"
  say "완료 — 감시견 종료"
}

say "감시견 시작 (2분 주기)"
while true; do
  sleep 120
  [ -f $ST/full_eval.done ] && { say "체인이 정상 종료했다 — 감시견 종료"; exit 0; }
  orchestrators_alive && continue

  miss=""
  for t in zh de ja; do [ -s $F/bleu/$t.json ] || miss="$miss $t"; done

  if [ -z "$miss" ]; then
    say "세 타깃 json 다 있는데 체인이 마무리를 못 했다 — 대신 끝낸다"
    finish; exit 0
  fi

  say "오케스트레이터 없음 / 결손:$miss"
  for t in $miss; do
    if [ ${tries[$t]} -ge 3 ]; then say "$t 3회 실패 — 포기"; continue; fi
    tries[$t]=$(( ${tries[$t]} + 1 ))
    rerun $t
  done

  # 재실행 결과를 다시 센다. **결손이 없어졌으면 성공이다** — 종전에는 `stuck` 을 1 로
  # 두고 "결손이면서 시도가 남은 타깃"에서만 0 으로 내려서, 전부 성공하면 루프가 통째로
  # continue 되어 stuck 이 1 로 남았다. 성공을 실패로 읽었다 (2026-09-04 05:46 실제 발생).
  miss=""
  for t in zh de ja; do [ -s $F/bleu/$t.json ] || miss="$miss $t"; done
  if [ -z "$miss" ]; then
    say "결손 해소 — 마무리로 넘어간다"; finish; exit 0
  fi
  exhausted=1
  for t in $miss; do [ ${tries[$t]} -ge 3 ] || exhausted=0; done
  if [ $exhausted -eq 1 ]; then
    say "복구 불가 — 결손 남김:$miss"; mark full_eval.failed "감시견 재시도 소진:$miss"; exit 1
  fi
done
