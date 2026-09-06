#!/bin/bash
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000
echo "===== en->X bleu_eval 시작 $(ts) ====="
echo "  (translate_de.json 캐시 48,118건이 남아 있어 de 는 79% 지점부터 이어간다)"
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id covost2/n3000 --label auto_run13 --split test \
  --dataset covost2 --manifest-tag n3000 --targets de ja zh \
  --t-grid 4 6 8 12 --src-spaced 1 \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
  --baselines punct syntax alignatt mu_prefix causal_align \
  --bootstrap 1000 > $N/bleu_eval.log 2>&1
rc=$?
echo "===== bleu_eval 종료 $(ts) exit=$rc ====="; tail -25 $N/bleu_eval.log

missing=""; for t in de ja zh; do [ -f $N/bleu/$t.json ] || missing="$missing $t"; done
if [ -n "$missing" ]; then
  echo "!! 중단: bleu/{$missing }.json 이 없다 — COMET·그래프를 돌리지 않는다"
  mark enx_eval.failed "bleu 결손:$missing"
  mark gpu_free.done "실패로 인한 해제"   # 4번이 GPU 를 못 받고 영영 대기하는 걸 막는다
  exit 1
fi
mark enx_bleu.done "ok"

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/n3000 --dataset covost2 --manifest-tag n3000 \
  --label auto_run13 --split test --targets de ja zh \
  --model Unbabel/wmt22-comet-da --batch-size 64 > $N/comet.log 2>&1
echo "  COMET exit=$? $(ts)"; tail -20 $N/comet.log

echo "===== 그래프 $(ts) ====="
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric bleu \
  --out tradeoff_covost2 --title "CoVoST2 en->X test 3000" 2>&1 | tail -5
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric comet \
  --out tradeoff_covost2_comet --title "CoVoST2 en->X test 3000" 2>&1 | tail -5

mark gpu_free.done "en->X 전체 완료"
echo "===== en->X 체인 전체 완료 $(ts) ====="
