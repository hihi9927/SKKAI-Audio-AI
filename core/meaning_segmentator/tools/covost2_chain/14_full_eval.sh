#!/bin/bash
# 전체(15,430) 참조 기반 평가. 비교군 라벨(13) 이 끝난 뒤 마커로 이어받는다.
#
# **타깃을 순차로 돌린다.** 종전에는 de+ja 를 병렬로 띄웠는데, 진짜 병목은 프로세스 수가
# 아니라 `--workers 4` 였다 — 그 기본값은 gtx 무료 엔드포인트의 rate limit 에서 나온 값이라
# 로컬 madlad 에서는 배치를 못 채워 GPU 가 작은 배치만 돌린다 (실측 30 조각/초).
# 프로세스 하나에 워커를 올리는 편이 madlad 사본도 하나(6GB)라 활성값 여유가 커진다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
# T 10·12 는 k 가 1.05·1.01 로 사실상 무분절이라 곡선에 정보가 없다. 9개 격자는
# 비교군까지 곱해져 번역량의 72% 를 먹었다 — 콜드 캐시 15,430문장에서 타깃당 19시간.
GRID="2 3 4 6"
BASE="punct alignatt mu_prefix causal_align"
mkdir -p $F/logs

wait_for full_baselines.done

# syntax 는 격리 venv 가 제때 끝났을 때만 조건에 넣는다 (없어도 나머지는 돈다).
if [ -f $ST/syntax_full.done ]; then
  BASE="$BASE syntax"; echo "  syntax 포함"
else
  echo "  syntax 제외 — 라벨이 아직 없다"
fi

run_tgt () {   # <타깃>
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id covost2/full --label auto_run13_mg1 --split test \
    --dataset covost2 --manifest-tag full --targets $1 \
    --t-grid $GRID --src-spaced 1 \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --workers 24 --baselines $BASE --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    > $F/logs/bleu_eval_$1.log 2>&1
}

for t in zh de ja; do
  echo "===== bleu_eval $t $(ts) ====="
  run_tgt $t; echo "  $t exit=$? $(ts)"
done
r1=$(ls -1 $F/bleu/de.json 2>/dev/null | wc -l)
r2=$(ls -1 $F/bleu/ja.json 2>/dev/null | wc -l)
r3=$(ls -1 $F/bleu/zh.json 2>/dev/null | wc -l)

for t in de ja zh; do tail -3 $F/logs/bleu_eval_$t.log; done
[ "$r1$r2$r3" = "111" ] || {
  echo "!! bleu/<tgt>.json 결손 (de=$r1 ja=$r2 zh=$r3)"
  mark full_eval.failed "bleu de=$r1 ja=$r2 zh=$r3"; exit 1; }

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id covost2/full --dataset covost2 --manifest-tag full --src en \
  --label auto_run13_mg1 --split test --targets zh de ja --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 \
  > $F/logs/comet_full.log 2>&1
echo "  COMET exit=$? $(ts)"; grep "조건 채점" $F/logs/comet_full.log

# BLEU 는 안 재기로 했다 (`--bootstrap 0 --no-sentence-bleu`). corpus BLEU 는 한 번만
# 토크나이즈하면 되므로 그냥 나오지만, 그림은 COMET 판만 뽑는다.
echo "===== 그래프 (COMET) $(ts) ====="
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id covost2/full --targets zh de ja --metric comet \
  --out tradeoff_covost2_full_comet \
  --title "CoVoST2 en->X test 15430 (min_gap=1)" 2>&1 | tail -2

mark full_eval.done "ok"
echo "===== 평가 완료 $(ts) ====="
