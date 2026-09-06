#!/bin/bash
# mg1 COMET 재실행 + 그래프. 이전 런은 DataLoader 워커가 죽어 COMET 만 날아갔다
# (BLEU 는 무사). metrics.py 를 num_workers=0 으로 고쳤고, 그래프는 T 격자를
# 데이터에서 읽도록 고쳤다 — 하드코딩 [4,6,8,12] 탓에 저지연 점이 안 그려졌다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000

echo "===== COMET 재실행 $(ts)  batch=32 ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/n3000 --dataset covost2 --manifest-tag n3000 \
  --label auto_run13_mg1 --split test --targets de ja zh \
  --model Unbabel/wmt22-comet-da --batch-size 32 > $N/comet_mg1_retry.log 2>&1
rc=$?
echo "  COMET exit=$rc $(ts)"; tail -12 $N/comet_mg1_retry.log
if [ $rc -ne 0 ]; then
  echo "!! COMET 실패 — 그래프는 BLEU 판만 다시 그린다"; mark comet_mg1.failed "exit=$rc"
fi

# comet 값이 실제로 박혔는지 확인한다 (종료코드만 보면 부분 실패를 놓친다)
$PY - <<'PYCHK'
import json
for t in ("de","ja","zh"):
    c=json.load(open(f"core/meaning_segmentator/experiment/artifacts/covost2/n3000/bleu/{t}.json"))["conditions"]
    n=sum(1 for v in c.values() if v.get("comet") is not None)
    print(f"  {t}: comet 값 있는 조건 {n}/{len(c)}")
PYCHK

echo "===== 그래프 $(ts) ====="
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric bleu \
  --out tradeoff_covost2_mg1 --title "CoVoST2 en->X test 3000 (min_gap=1)" 2>&1 | tail -5
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric comet \
  --out tradeoff_covost2_mg1_comet --title "CoVoST2 en->X test 3000 (min_gap=1)" 2>&1 | tail -5

mark comet_mg1.done "ok"
echo "===== 완료 $(ts) ====="
