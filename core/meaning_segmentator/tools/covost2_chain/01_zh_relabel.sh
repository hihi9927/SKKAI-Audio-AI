#!/bin/bash
. core/meaning_segmentator/tools/covost2_chain/common.sh
O=$R/zh-en_n3000
mkdir -p $O/labels $O/cache $O/logs
echo "===== zh 재라벨링 시작 $(ts) ====="
# **캐시가 2시간 넘게 0건인 것은 정상이다.** segment_batch 는 2단계다 —
# 1단계에서 500그룹(2999문장/배치6)을 전부 부르고, 그동안 call_group 은 캐시를
# 안 건드린다. cache.put 은 2단계 one() 안에만 있다. 실측 환산 1단계만 약 2.3시간.
# 이걸 모르고 "멈췄다"고 판단해 두 번 죽인 적이 있다 (2026-09-02).
echo "  주의: 1단계(약 2.3시간) 동안 캐시는 0건이다 — 고장이 아니다."
echo "  살아있는지 보려면: ls -l /proc/\$(pgrep -f label_covost2|head -1)/fd | grep -c socket"
# 이전 라벨은 이미 .mismatch68 로 옮겨져 있다 — 다시 mv 하지 않는다.
$PY -u core/meaning_segmentator/tools/covost2_label/label_covost2.py \
  --provider openai --model gpt-5-mini \
  --prompt core/meaning_segmentator/runs/zh-en/run02/best_prompt_covost2_mg5.txt \
  --manifest evaluation/ast/manifests/covost2_zh-en_n3000.jsonl \
  --out $O/labels/covost2_zh-en_n3000_run02.jsonl \
  --cache $O/cache/segment_run02.json \
  --min-gap 5 --t-floor 7 --batch-size 6 --workers 12 \
  --max-tokens 24000 --timeout 2400 --budget 20.0 --cache-every 1 \
  > $O/logs/zh-en_n3000_mg5.log 2>&1
rc=$?
echo "===== zh 재라벨링 종료 $(ts) exit=$rc ====="
tail -22 $O/logs/zh-en_n3000_mg5.log
if [ $rc -eq 0 ] && [ -s $O/labels/covost2_zh-en_n3000_run02.jsonl ]; then
  mark zh_relabel.done "ok"
else
  mark zh_relabel.failed "exit=$rc"
  echo "!! 실패 — 4번 체인이 zh 없이 못 간다. 로그를 볼 것."
fi
