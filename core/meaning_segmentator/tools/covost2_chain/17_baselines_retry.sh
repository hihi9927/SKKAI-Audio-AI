#!/bin/bash
# 비교군 재실행 — 2026-09-03 14:50 Xid 69 로 5개가 동시에 죽은 뒤 복구.
#
# 원인: CUDA 컨텍스트를 11개까지 올린 상태(비교군 10 + spacy syntax 1, 서로 다른 torch
# 빌드)에서 드라이버가 채널을 죽였다. `torch.AcceleratorError: unspecified launch failure`.
# ECC·하드웨어 이상 징후는 없었다. 그래서 **동시 실행을 3개로 제한**한다.
#
# `build.py` 는 끝에 한 번에 쓰므로 부분 진행분은 못 살린다 — 죽은 5개는 처음부터다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
mkdir -p $F/logs/baselines
BD=$F/baselines

# 살아남은 mu_prefix_ja 가 끝나야 동시 3개 상한이 지켜진다. PID 가 아니라 산출 파일로 센다.
while [ ! -f $BD/mu_prefix_ja_test.json ]; do sleep 60; done
echo "===== mu_prefix_ja 완료 확인 $(ts) — 재실행 시작 ====="

# 긴 작업을 앞에 둔다 (makespan). xargs -P 3 이 큐처럼 돌며 하나 끝나면 다음을 띄운다.
cat > /tmp/retry_jobs.txt <<'JOBS'
mu_prefix de
mu_prefix zh
alignatt de
alignatt ja
alignatt zh
JOBS

run_one () {
  pol=$1; tgt=$2
  echo "  [$(date '+%H:%M:%S')] 시작 $pol/$tgt"
  /home/mobility/STiTy/.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build \
    --run-id covost2/full --dataset covost2 --manifest-tag full \
    --label auto_run13_mg1 --split test --policy $pol --targets $tgt \
    > /home/mobility/STiTy/core/meaning_segmentator/runs/covost2/full/logs/baselines/${pol}_${tgt}.log 2>&1
  rc=$?
  echo "  [$(date '+%H:%M:%S')] 종료 $pol/$tgt exit=$rc"
  return $rc
}
export -f run_one

xargs -a /tmp/retry_jobs.txt -L1 -P 3 bash -c 'run_one $0 $1'
echo "===== 재실행 종료 $(ts) ====="

# 산출 11개가 다 있어야 평가로 넘긴다 (종료코드만 보면 부분 실패를 놓친다).
missing=""
for f in punct_all syntax_all \
         alignatt_de alignatt_ja alignatt_zh \
         mu_prefix_de mu_prefix_ja mu_prefix_zh \
         causal_align_de causal_align_ja causal_align_zh; do
  [ -s $BD/${f}_test.json ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  echo "!! 결손:$missing"; mark full_baselines.failed "결손$missing"; exit 1
fi
for f in $BD/*.json; do echo "  $(basename $f): $($PY -c "import json,sys;print(len(json.load(open('$f'))['rows']),'행')")"; done

rm -f $ST/full_baselines.failed
mark full_baselines.done "재실행 후 11개 전부"
echo "===== 비교군 완료 $(ts) ====="
