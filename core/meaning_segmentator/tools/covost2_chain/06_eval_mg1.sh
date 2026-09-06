#!/bin/bash
# mg1 라벨(경계 3.80, T2 에서 k=4.77) 평가. 비교군이 독점하던 628~1290ms 구간에
# auto 점이 처음 들어간다. mg3 촘촘격자 결과는 bleu_mg3_finegrid/ 에 백업돼 있다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000
GRID="2 3 4 5 6 7 8 10 12"
STAMP=$(mktemp); touch $STAMP

echo "===== mg1 bleu_eval 시작 $(ts)  격자=$GRID ====="
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id covost2/n3000 --label auto_run13_mg1 --split test \
  --dataset covost2 --manifest-tag n3000 --targets de ja zh \
  --t-grid $GRID --src-spaced 1 \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
  --baselines punct syntax alignatt mu_prefix causal_align \
  --bootstrap 1000 > $N/bleu_eval_mg1.log 2>&1
rc=$?
echo "===== bleu_eval 종료 $(ts) exit=$rc ====="; tail -25 $N/bleu_eval_mg1.log
if [ $rc -ne 0 ]; then
  echo "!! exit=$rc — COMET·그래프 중단"; mark eval_mg1.failed "bleu exit=$rc"; exit 1
fi
missing=""
for t in de ja zh; do
  [ -f $N/bleu/$t.json ] && [ $N/bleu/$t.json -nt $STAMP ] || missing="$missing $t"
done
[ -z "$missing" ] || { echo "!! 결손/구버전:$missing"; mark eval_mg1.failed "$missing"; exit 1; }

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/n3000 --dataset covost2 --manifest-tag n3000 \
  --label auto_run13_mg1 --split test --targets de ja zh \
  --model Unbabel/wmt22-comet-da --batch-size 64 > $N/comet_mg1.log 2>&1
echo "  COMET exit=$? $(ts)"; tail -10 $N/comet_mg1.log

echo "===== 그래프 $(ts) ====="
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric bleu \
  --out tradeoff_covost2_mg1 --title "CoVoST2 en->X test 3000 (min_gap=1)" 2>&1 | tail -4
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric comet \
  --out tradeoff_covost2_mg1_comet --title "CoVoST2 en->X test 3000 (min_gap=1)" 2>&1 | tail -4

mark eval_mg1.done "ok"
echo "===== mg1 평가 전체 완료 $(ts) ====="
