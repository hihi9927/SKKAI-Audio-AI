#!/bin/bash
# 2번째 번역 프로세스 — `ja` 만 따로 돈다.
#
# **de 가 아니라 ja 를 고른 이유.** 14번 체인이 `zh → de → ja` 순이라, de 를 여기서 돌리면
# 02:15 부터 체인의 de 와 겹쳐 같은 `bleu/de.json` 과 `cache/translate_de.json` 을 두
# 프로세스가 동시에 쓴다. ja 를 맡으면 타깃이 달라 출력·캐시가 완전히 분리되고, 체인이
# 06시경 ja 에 도달할 때는 이미 캐시가 다 차 있어 몇 분에 통과한다.
#
# GPU 사용률이 62% 라 자기회귀 디코딩의 커널 실행 틈이 남아 있다 — 프로세스를 하나 더
# 띄우면 GIL·CUDA 컨텍스트가 따로라 그 틈이 메워진다. 대신 워커·배치를 낮춰 VRAM 을
# 맞춘다 (13.4GB + ~7GB = 21GB / 24.5GB).
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
GRID="2 3 4 6"
BASE="punct alignatt mu_prefix causal_align syntax"

echo "===== bleu_eval ja (2번째 프로세스) $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id covost2/full --label auto_run13_mg1 --split test \
  --dataset covost2 --manifest-tag full --targets ja \
  --t-grid $GRID --src-spaced 1 \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 24 \
  --workers 10 --baselines $BASE --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
  > $F/logs/bleu_eval_ja.log 2>&1
rc=$?
echo "  ja exit=$rc $(ts)"; tail -3 $F/logs/bleu_eval_ja.log
[ $rc -eq 0 ] || { mark full_eval_ja.failed "exit=$rc"; exit 1; }
mark full_eval_ja.done "ok"
echo "===== ja 완료 $(ts) ====="
