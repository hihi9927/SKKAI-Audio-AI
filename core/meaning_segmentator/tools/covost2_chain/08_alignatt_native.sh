#!/bin/bash
# AlignAtt 네이티브 f 스윕. 지금까지 곡선은 f=2 라벨 한 벌을 coarsen 으로 쓸어 만든
# 것이라 "우리 노브 T" 축이었다. 원논문 노브는 f 이고 f 를 바꾸면 강제 디코딩 경로가
# 통째로 달라져 사후 재사용이 안 되므로 f 마다 라벨을 새로 만든다 (f=2 는 기존 파일).
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000

for f in 4 6 8; do
  echo "===== alignatt f=$f 라벨 $(ts) ====="
  $PY -u -m core.meaning_segmentator.autoseg.baselines.build \
    --run-id covost2/n3000 --policy alignatt --f $f \
    --out-name alignatt_f$f \
    --targets de ja zh --split test \
    --dataset covost2 --manifest-tag n3000 --label auto_run13_mg1 \
    >> $N/alignatt_native_build.log 2>&1
  rc=$?
  echo "  f=$f exit=$rc $(ts)"
  if [ $rc -ne 0 ]; then mark alignatt_native.failed "f=$f exit=$rc"; tail -20 $N/alignatt_native_build.log; exit 1; fi
done

mark alignatt_native_build.done "f=4,6,8"
echo "===== 라벨 완료 $(ts) ====="
