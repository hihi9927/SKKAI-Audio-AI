#!/bin/bash
# X->en 세 트랙 COMET. 조건이 트랙당 10개뿐이라 짧다 (n3000 의 70조건이 6분이었다).
# alignatt 라벨 생성(NLLB)과 GPU 를 같이 쓴다 — 24GB 중 1.8GB 만 쓰고 있어 여유가 있다.
. core/meaning_segmentator/tools/covost2_chain/common.sh

run () {   # <소스> <디렉토리> <매니페스트 태그>
  echo "===== $1->en COMET $(ts) ====="
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id covost2/$2 --dataset covost2 --src $1 --manifest-tag $3 \
    --targets en --label auto_run02 --split test \
    --model Unbabel/wmt22-comet-da --batch-size 32 2>&1 | grep -vE "LeafSpec|_pytree"
  echo "  $1 exit=${PIPESTATUS[0]} $(ts)"
}

run de de-en_n3000 n3000
run ja ja-en_n678  n678
run zh zh-en_n3000 n3000

$PY core/meaning_segmentator/tools/covost2_chain/x2en_table.py
mark comet_x2en.done "ok"
echo "===== 완료 $(ts) ====="
