#!/bin/bash
# 범인이 동시성인지 가른다. max_tokens 는 24000 고정, workers 만 바꾼다.
# 150문장 = 25콜 이라 workers 24 를 실제로 포화시킬 수 있다 (30문장이면 5콜뿐이라 재현 불가).
. core/meaning_segmentator/tools/covost2_chain/common.sh
O=$R/zh-en_n3000/probe
mkdir -p $O

arm () {
  NAME=$1; W=$2
  echo "===== [$NAME] workers=$W x max_tokens=24000 = $((W*24000/1000))k  시작 $(ts) ====="
  S=$(date +%s)
  timeout 900 $PY -u core/meaning_segmentator/tools/covost2_label/label_covost2.py \
    --provider openai --model gpt-5-mini \
    --prompt core/meaning_segmentator/experiment/artifacts/zh-en/run02/best_prompt_covost2_mg5.txt \
    --manifest evaluation/ast/manifests/covost2_zh-en_n3000.jsonl \
    --out $O/$NAME.jsonl --cache $O/$NAME.cache.json \
    --min-gap 5 --t-floor 7 --batch-size 6 --workers $W --limit 150 \
    --max-tokens 24000 --timeout 420 --budget 2.0 --cache-every 1 \
    > $O/$NAME.log 2>&1
  rc=$?; E=$(( $(date +%s) - S ))
  NC=0; [ -f $O/$NAME.cache.json ] && NC=$($PY -c "import json;print(len(json.load(open('$O/$NAME.cache.json'))))" 2>/dev/null)
  echo "  exit=$rc  경과=${E}초  캐시완료=$NC/150"
  [ $rc -eq 124 ] && echo "  !! 900초 timeout — 재현됨"
  sed -n '/^{/,/^}/p' $O/$NAME.log
  echo
}

arm D_w24 24     # 실패한 설정 그대로 — 재현되나
arm E_w12 12     # 절반 — 풀리나
echo "===== 검증2 끝 $(ts) ====="
