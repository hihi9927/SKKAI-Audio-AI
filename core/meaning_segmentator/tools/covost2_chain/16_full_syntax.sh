#!/bin/bash
# syntax(SASST) 비교군 — 격리 venv 로만 돈다.
# build.py 의 무거운 의존(simalign/nltk/fugashi/transformers)은 전부 함수 안에서
# import 되므로, syntax 경로만 타면 spacy 외에는 numpy 정도만 있으면 된다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
V=$HOME/.venvs/spacyenv
mkdir -p $F/logs/baselines

wait_for spacyenv.done
$V/bin/pip install -q numpy 2>/dev/null

echo "===== syntax 스모크 20문장 $(ts) ====="
PYTHONPATH=/home/mobility/STiTy $V/bin/python -m core.meaning_segmentator.autoseg.baselines.build \
  --run-id covost2/full --dataset covost2 --manifest-tag full \
  --label auto_run13_mg1 --split test --policy syntax --targets de --limit 20 \
  > $F/logs/baselines/syntax_smoke.log 2>&1
rc=$?; tail -4 $F/logs/baselines/syntax_smoke.log
[ $rc -eq 0 ] || { echo "!! 스모크 실패 — syntax 건너뜀"; mark syntax_full.failed "smoke exit=$rc"; exit 1; }

echo "===== syntax 본런 15,430문장 $(ts) ====="
PYTHONPATH=/home/mobility/STiTy $V/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build \
  --run-id covost2/full --dataset covost2 --manifest-tag full \
  --label auto_run13_mg1 --split test --policy syntax --targets de \
  > $F/logs/baselines/syntax.log 2>&1
rc=$?; echo "  exit=$rc $(ts)"; tail -4 $F/logs/baselines/syntax.log
[ $rc -eq 0 ] || { mark syntax_full.failed "exit=$rc"; exit 1; }

mark syntax_full.done "ok"
echo "===== 완료 $(ts) ====="
