#!/bin/bash
# alignatt de/ja 만 남았다. **동시 실행 1개** — 3-병렬에서도 CUDA 오류로 두 번 죽었고
# (8,125 와 12,175 지점), 단독으로 돈 alignatt_zh 만 완주했다. 교차어텐션을 켜고 돌려서
# 다른 정책보다 커널 압력이 크다.
#
# `--resume` 으로 `<out>.partial.jsonl` 을 이어받는다. 또 죽어도 100건 단위까지만 잃는다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
BD=$F/baselines

run () {  # <타깃>
  for try in 1 2 3; do
    echo "  [$(ts)] alignatt/$1 시도 $try"
    $PY -u -m core.meaning_segmentator.autoseg.baselines.build \
      --run-id covost2/full --dataset covost2 --manifest-tag full \
      --label auto_run13_mg1 --split test --policy alignatt --targets $1 --resume \
      >> $F/logs/baselines/alignatt_$1.log 2>&1
    rc=$?
    echo "  [$(ts)] alignatt/$1 시도 $try exit=$rc"
    [ $rc -eq 0 ] && return 0
    n=$(wc -l < $BD/alignatt_$1.partial.jsonl 2>/dev/null || echo 0)
    echo "    죽었다 — 진행분 $n 건 보존됨, 이어서 재시도"
    sleep 60
  done
  return 1
}

echo "===== alignatt 단독 재실행 $(ts) ====="
run de || { mark full_baselines.failed "alignatt de 3회 실패"; exit 1; }
run ja || { mark full_baselines.failed "alignatt ja 3회 실패"; exit 1; }

missing=""
for f in punct_all syntax_all alignatt_de alignatt_ja alignatt_zh \
         mu_prefix_de mu_prefix_ja mu_prefix_zh \
         causal_align_de causal_align_ja causal_align_zh; do
  [ -s $BD/${f}_test.json ] || missing="$missing $f"
done
[ -z "$missing" ] || { echo "!! 결손:$missing"; mark full_baselines.failed "결손$missing"; exit 1; }
for f in $BD/*_test.json; do
  echo "  $(basename $f): $($PY -c "import json;print(len(json.load(open('$f'))['rows']))")행"
done

rm -f $ST/full_baselines.failed
mark full_baselines.done "11개 전부"
echo "===== 비교군 완료 $(ts) ====="
