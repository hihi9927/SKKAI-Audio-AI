#!/bin/bash
# 마무리 — COMET + 그래프. bleu_eval 은 세 타깃 다 끝났고(31조건 × 15,430) COMET 만 남았다.
# GPU 는 비어 있으므로 단독으로 돈다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/full --dataset covost2 --manifest-tag full --src en \
  --label auto_run13_mg1 --split test --targets zh de ja --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 > $F/logs/comet_full.log 2>&1
rc=$?; echo "  COMET exit=$rc $(ts)"; grep -E "조건 채점" $F/logs/comet_full.log
[ $rc -eq 0 ] || { mark full_eval.failed "comet exit=$rc"; exit 1; }

$PY - <<'PYCHK'
import json
for t in ("zh","de","ja"):
    c=json.load(open(f"/home/mobility/STiTy/core/meaning_segmentator/runs/covost2/full/bleu/{t}.json"))["conditions"]
    n=sum(1 for v in c.values() if v.get("comet") is not None)
    print(f"  {t}: comet 값 있는 조건 {n}/{len(c)}")
PYCHK

echo "===== 그래프 $(ts) ====="
$PY core/meaning_segmentator/autoseg/baselines/plot_tradeoff.py \
  --run-id covost2/full --targets zh de ja --metric comet \
  --out tradeoff_covost2_full_comet \
  --title "CoVoST2 en->X test 15430 (min_gap=1)" 2>&1 | tail -2

mark full_eval.done "31조건 × 3타깃, COMET 포함"
echo "===== 전체 완료 $(ts) ====="
