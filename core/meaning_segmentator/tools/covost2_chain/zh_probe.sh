#!/bin/bash
# zh 재라벨링이 왜 한 콜도 안 돌아오는지 가른다. 30문장 = 5콜씩만 쓴다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
O=$R/zh-en_n3000/probe
mkdir -p $O

# arm <이름> <max_tokens> <reasoning_effort|->
arm () {
  NAME=$1; MT=$2; RE=$3
  EXTRA=""; [ "$RE" != "-" ] && EXTRA="--seg-reasoning-effort $RE"
  echo "===== [$NAME] max_tokens=$MT effort=$RE  시작 $(ts) ====="
  S=$(date +%s)
  timeout 600 $PY -u core/meaning_segmentator/tools/covost2_label/label_covost2.py \
    --provider openai --model gpt-5-mini \
    --prompt core/meaning_segmentator/experiment/artifacts/zh-en/run02/best_prompt_covost2_mg5.txt \
    --manifest evaluation/ast/manifests/covost2_zh-en_n3000.jsonl \
    --out $O/$NAME.jsonl --cache $O/$NAME.cache.json \
    --min-gap 5 --t-floor 7 --batch-size 6 --workers 6 --limit 30 \
    --max-tokens $MT --timeout 420 --budget 1.0 --cache-every 1 $EXTRA \
    > $O/$NAME.log 2>&1
  rc=$?; E=$(( $(date +%s) - S ))
  NC=0; [ -f $O/$NAME.cache.json ] && NC=$($PY -c "import json;print(len(json.load(open('$O/$NAME.cache.json'))))" 2>/dev/null)
  echo "  exit=$rc  경과=${E}초  캐시완료=$NC/30"
  [ $rc -eq 124 ] && echo "  !! 600초 timeout — 이 설정으로는 못 쓴다"
  grep -c "출력이 끊겼다" $O/$NAME.log 2>/dev/null | xargs -I{} echo "  '사고 토큰 소진' 경고: {}건"
  tail -14 $O/$NAME.log
  echo
}

arm A_mt16000      16000 -
arm B_mt24000      24000 -
arm C_mt16000_low  16000 low

echo "===== 검증 끝 $(ts) ====="
echo "판단: 경과시간·캐시완료·경고건수를 A(직전 성공설정) 대비로 볼 것"
