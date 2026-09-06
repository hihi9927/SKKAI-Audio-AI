#!/bin/bash
. core/meaning_segmentator/tools/covost2_chain/common.sh
echo "===== X->en 대기 시작 $(ts) ====="
echo "  기다리는 것: zh_relabel.done (라벨 필요) + gpu_free.done (GPU 필요)"
wait_for zh_relabel.done gpu_free.done
echo "===== 선행 전부 완료 $(ts) ====="

# zh 라벨 -> prompt_eval 변환. de/ja 는 이미 00:18~00:19 에 변환해 두었다.
echo "===== zh 변환 $(ts) ====="
$PY core/meaning_segmentator/tools/covost2_label/to_prompt_eval.py \
  --labels $R/zh-en_n3000/labels/covost2_zh-en_n3000_run02.jsonl \
  --run-id covost2/zh-en_n3000 --label auto_run02 --min-gap 5 --t-grid 7 11 14 21 \
  --prompt-file core/meaning_segmentator/runs/zh-en/run02/best_prompt_covost2_mg5.txt 2>&1 | tail -4
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "!! zh 변환 실패 — 중단"; mark x2en.failed "zh 변환"; exit 1; }

run () {   # <언어> <n> <격자...>
  L=$1; N=$2; shift 2
  SP=$([ "$L" = de ] && echo 1 || echo 0)
  echo "===== $L->en 번역 $(ts) 격자=$* 띄어쓰기=$SP ====="
  $PY -u -m core.meaning_segmentator.autoseg.bleu_eval \
    --run-id covost2/$L-en_n$N --label auto_run02 --split test \
    --dataset covost2 --src $L --manifest-tag n$N --targets en \
    --t-grid $* --src-spaced $SP \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --bootstrap 1000 > $R/$L-en_n$N/bleu_eval.log 2>&1
  echo "  $L exit=$? $(ts)"; tail -12 $R/$L-en_n$N/bleu_eval.log
}
run de 3000 4 6 8 12
run zh 3000 7 11 14 21
run ja 678 9 14 18 27
mark x2en.done "ok"
echo "===== X->en 전체 완료 $(ts) ====="
