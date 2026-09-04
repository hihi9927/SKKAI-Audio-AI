#!/bin/bash
# 전체(15,430) 비교군 라벨 생성. **최대한 병렬로 돌린다.**
#
# 근거: 비교군은 문장마다 접두사를 늘려가며 디코딩하는 구조라 배치가 작고, 단독 실행 시
# GPU 사용률이 17% 밖에 안 된다(실측). NLLB-600M 은 fp16 1.2GB 뿐이라 9개를 동시에
# 올려도 14GB 다 — 24GB 안에 들어간다. 직렬로 돌리면 6시간, 병렬이면 2~3시간이다.
#
# `syntax` 는 여기 없다 — spacy_transformers 가 레포 transformers 핀을 깨므로 격리
# venv 가 필요하고, 그건 15번 스크립트가 따로 만든다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
mkdir -p $F/logs/baselines
B="$PY -u -m core.meaning_segmentator.autoseg.baselines.build \
   --run-id covost2/full --dataset covost2 --manifest-tag full \
   --label auto_run13_mg1 --split test"

echo "===== 비교군 병렬 시작 $(ts) ====="
pids=""
launch () {  # <정책> <타깃들...>
  pol=$1; shift
  for t in "$@"; do
    $B --policy $pol --targets $t > $F/logs/baselines/${pol}_${t}.log 2>&1 &
    pids="$pids $!"
    echo "  [$(ts)] $pol/$t pid=$!"
  done
}

# 타깃 독립 (한 번만)
$B --policy punct --targets de > $F/logs/baselines/punct.log 2>&1 &
pids="$pids $!"; echo "  [$(ts)] punct pid=$!"

launch alignatt     de ja zh
launch mu_prefix    de ja zh
launch causal_align de ja zh

echo "  총 $(echo $pids | wc -w) 개 병렬"
fail=0
for p in $pids; do wait $p || fail=$((fail+1)); done
echo "===== 비교군 종료 $(ts) 실패 $fail 개 ====="

ls -la $F/baselines/ | tail -15
for f in $F/logs/baselines/*.log; do
  echo "--- $(basename $f): $(grep -c . $f)줄, 마지막: $(tail -1 $f | cut -c1-90)"
done

[ $fail -eq 0 ] || { mark full_baselines.failed "$fail 개 실패"; exit 1; }
mark full_baselines.done "punct+alignatt+mu_prefix+causal_align"
echo "===== 완료 $(ts) ====="
