#!/bin/bash
# min_gap 3 -> 2 재라벨링. 목적: auto 곡선을 1252ms 아래로 늘려 비교군(628~1186ms)과
# 같은 구간에서 붙게 한다. 지금 라벨은 경계가 1.40개뿐이라 T4 에서 이미 99.4% 소진이다.
# 분절은 영어 한 벌이고 de/ja/zh 가 공유하므로 라벨링은 한 번만 하면 된다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
N=$R/n3000
PROMPT=core/meaning_segmentator/runs/en-multi/run13/best_prompt_mg1.txt
GRID="2 3 4 5 6 7 8 10 12"

label () {  # <이름> <limit인자> <budget>
  NAME=$1; LIM=$2; BUD=$3
  $PY -u core/meaning_segmentator/tools/covost2_label/label_covost2.py \
    --provider openai --model gpt-5-mini --prompt $PROMPT \
    --manifest evaluation/ast/manifests/covost2_en-de_n3000.jsonl \
    --out $N/labels/$NAME.jsonl --cache $N/cache/$NAME.cache.json \
    --min-gap 1 --t-floor 2 --batch-size 6 --workers 12 $LIM \
    --max-tokens 24000 --timeout 420 --budget $BUD --cache-every 1 \
    > $N/logs/$NAME.log 2>&1
  return $?
}

echo "===== 소량 검증 (30문장) $(ts) ====="
label smoke_mg1 "--limit 30" 1.0
echo "  exit=$?"; sed -n '/^{/,/^}/p' $N/logs/smoke_mg1.log

# 게이트: 형식이 깨지거나 경계가 안 늘면 본런에 $10 을 태우지 않는다
$PY - <<'PYGATE' || { echo "!! 게이트 불통과 — 본런 중단"; mark relabel_mg1.failed "smoke gate"; exit 1; }
import json, sys, re
t = open("/home/mobility/STiTy/core/meaning_segmentator/runs/covost2/n3000/logs/smoke_mg1.log").read()
m = re.search(r"\{.*\}", t, re.S)
if not m:
    print("  요약 JSON 없음"); sys.exit(1)
d = json.loads(m.group(0))
ok = (d["format_pass"] >= 0.90 and d["text_preserved"] >= 0.95
      and d["coverage_met"] >= 0.90)
print("  게이트: format=%s preserved=%s boundaries=%s/%s coverage=%s "
      "(mg3=1.40, mg2=1.97) -> %s" % (
      d["format_pass"], d["text_preserved"], d["mean_boundaries"],
      d["mean_required"], d["coverage_met"], "통과" if ok else "불통과"))
sys.exit(0 if ok else 1)
PYGATE

echo "===== 본런 (3000문장) $(ts) ====="
label covost2_n3000_run13_mg1 "" 15.0
rc=$?; echo "  exit=$rc $(ts)"; sed -n '/^{/,/^}/p' $N/logs/covost2_n3000_run13_mg1.log
[ $rc -eq 0 ] || { echo "!! 본런 실패"; mark relabel_mg1.failed "label exit=$rc"; exit 1; }

echo "===== prompt_eval 변환 $(ts)  격자=$GRID ====="
$PY core/meaning_segmentator/tools/covost2_label/to_prompt_eval.py \
  --labels $N/labels/covost2_n3000_run13_mg1.jsonl \
  --run-id covost2/n3000 --label auto_run13_mg1 --min-gap 1 \
  --t-grid $GRID --prompt-file $PROMPT 2>&1 | tail -6
mark relabel_mg1.done "ok"
echo "===== 재라벨링 완료 $(ts) — 평가는 GPU 가 빈 뒤 따로 건다 ====="
