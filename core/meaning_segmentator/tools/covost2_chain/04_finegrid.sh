#!/bin/bash
# 촘촘한 T 격자로 재평가. 라벨은 그대로 쓴다 — T 는 평가 시점 문턱일 뿐이다.
# 목적: 비교군이 몰려 있는 1200~1600ms 구간에 auto 점을 만들어 정면 비교가 되게 한다.
# 기존 4/6/8/12 결과는 bleu_grid_4-6-8-12/ 에 백업돼 있다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000
GRID="4 5 6 7 8 10 12"

STAMP=$(mktemp); touch $STAMP   # 이번 런의 시작 시각 표식
echo "===== 촘촘 격자 bleu_eval 시작 $(ts)  격자=$GRID ====="
echo "  번역 캐시 재사용: de 61,194 / ja 57,253 / zh 60,237 — 새 T 값의 조각만 새로 번역된다"
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id covost2/n3000 --label auto_run13 --split test \
  --dataset covost2 --manifest-tag n3000 --targets de ja zh \
  --t-grid $GRID --src-spaced 1 \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
  --baselines punct syntax alignatt mu_prefix causal_align \
  --bootstrap 1000 > $N/bleu_eval_finegrid.log 2>&1
rc=$?
echo "===== bleu_eval 종료 $(ts) exit=$rc ====="; tail -25 $N/bleu_eval_finegrid.log

# **종료코드를 먼저 본다.** 파일 존재만 보면 옛 결과가 남아 있어 크래시가 통과한다
# — 실제로 17:28 에 KeyError 로 죽었는데 COMET 이 옛 파일 위에서 돌기 시작했다.
if [ $rc -ne 0 ]; then
  echo "!! bleu_eval 이 exit=$rc 로 죽었다 — COMET·그래프를 돌리지 않는다"
  mark finegrid.failed "bleu_eval exit=$rc"; exit 1
fi
# 파일이 이번 런에서 새로 쓰였는지도 본다 (STAMP 보다 새것이어야 한다)
missing=""
for t in de ja zh; do
  [ -f $N/bleu/$t.json ] && [ $N/bleu/$t.json -nt $STAMP ] || missing="$missing $t"
done
if [ -n "$missing" ]; then
  echo "!! 중단: bleu/{$missing }.json 이 없거나 이번 런 결과가 아니다"
  mark finegrid.failed "bleu 결손/구버전:$missing"; exit 1
fi

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/n3000 --dataset covost2 --manifest-tag n3000 \
  --label auto_run13 --split test --targets de ja zh \
  --model Unbabel/wmt22-comet-da --batch-size 64 > $N/comet_finegrid.log 2>&1
echo "  COMET exit=$? $(ts)"; tail -12 $N/comet_finegrid.log

echo "===== 그래프 $(ts) ====="
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric bleu \
  --out tradeoff_covost2 --title "CoVoST2 en->X test 3000" 2>&1 | tail -4
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/n3000 --targets de ja zh --metric comet \
  --out tradeoff_covost2_comet --title "CoVoST2 en->X test 3000" 2>&1 | tail -4

mark finegrid.done "ok"
echo "===== 촘촘 격자 전체 완료 $(ts) ====="
