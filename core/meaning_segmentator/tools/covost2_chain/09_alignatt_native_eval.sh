#!/bin/bash
# AlignAtt 네이티브 f 스윕 평가 + 그래프. 08 이 라벨을 다 만든 뒤에 돈다.
#
# `--baselines-native` 로 coarsen T 격자를 끈다 — f 라벨에 우리 T 를 또 얹으면 노브가
# 두 겹이 되어 무슨 축인지 알 수 없다.
# `bleu_eval` 은 파일을 통째로 덮어쓰므로 백업을 먼저 뜨고 나중에 도로 합친다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000
BK=$N/bleu_backup_prenative

wait_for alignatt_native_build.done

mkdir -p $BK
cp $N/bleu/de.json $N/bleu/ja.json $N/bleu/zh.json $BK/
echo "백업 → $BK"

echo "===== 네이티브 조건 bleu_eval $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id covost2/n3000 --label auto_run13_mg1 --split test \
  --dataset covost2 --manifest-tag n3000 --targets de ja zh \
  --t-grid 2 3 4 5 6 7 8 10 12 --src-spaced 1 \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
  --baselines alignatt_f4 alignatt_f6 alignatt_f8 \
  --baselines-native alignatt_f4 alignatt_f6 alignatt_f8 \
  --conditions alignatt_f4 alignatt_f6 alignatt_f8 unsegmented \
  --bootstrap 1000 > $N/bleu_eval_alignatt_native.log 2>&1
rc=$?
echo "  bleu_eval exit=$rc $(ts)"; tail -12 $N/bleu_eval_alignatt_native.log
if [ $rc -ne 0 ]; then
  echo "!! 실패 — 백업을 되돌린다"; cp $BK/*.json $N/bleu/
  mark alignatt_native.failed "bleu exit=$rc"; exit 1
fi

echo "===== 백업 조건과 병합 $(ts) ====="
$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $BK $N/bleu de ja zh || {
  echo "!! 병합 실패 — 백업 되돌림"; cp $BK/*.json $N/bleu/
  mark alignatt_native.failed "merge"; exit 1; }

echo "===== COMET (없는 조건만) $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/n3000 --dataset covost2 --manifest-tag n3000 \
  --label auto_run13_mg1 --split test --targets de ja zh --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 \
  > $N/comet_alignatt_native.log 2>&1
echo "  COMET exit=$? $(ts)"; grep -E "^  \[|조건 채점" $N/comet_alignatt_native.log | tail -15

echo "===== 그래프 $(ts) ====="
for M in bleu comet; do
  S=""; [ $M = comet ] && S=_comet
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
    --run-id covost2/n3000 --targets zh de ja --metric $M \
    --out tradeoff_covost2_mg1$S --title "CoVoST2 en->X test 3000 (min_gap=1)" 2>&1 | tail -2
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
    --run-id covost2/n3000 --targets zh de ja --metric $M --point-labels all \
    --out tradeoff_covost2_mg1${S}_alllabels --title "CoVoST2 en->X test 3000 (min_gap=1)" 2>&1 | tail -2
done

mark alignatt_native.done "ok"
echo "===== 완료 $(ts) ====="
