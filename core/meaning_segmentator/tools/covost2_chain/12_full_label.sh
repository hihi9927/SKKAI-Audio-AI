#!/bin/bash
# CoVoST2 en test 전체(15,531) 라벨링. **여기서만 돈이 든다.**
# 3000 실측 $7.425 / 44.7분에서 외삽하면 ~$38 / ~4시간이다.
# 분절은 영어 한 벌이고 de/ja/zh 가 공유하므로 라벨링은 이 한 번뿐이다.
#
# 선행은 PID 가 아니라 마커 파일로 기다린다 — 세션이 죽으면 PID 가 무의미해진다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
F=$R/full
PROMPT=core/meaning_segmentator/experiment/artifacts/en-multi/run13/best_prompt_mg1.txt
MAN=evaluation/ast/manifests/covost2_en-de_full.jsonl
GRID="2 3 4 5 6 7 8 10 12"
mkdir -p $F/labels $F/logs $F/cache

wait_for full_prep.done

label () {  # <이름> <limit인자> <budget>
  NAME=$1; LIM=$2; BUD=$3
  $PY -u core/meaning_segmentator/tools/covost2_label/label_covost2.py \
    --provider openai --model gpt-5-mini --prompt $PROMPT \
    --manifest $MAN \
    --out $F/labels/$NAME.jsonl --cache $F/cache/$NAME.cache.json \
    --min-gap 1 --t-floor 2 --batch-size 6 --workers 12 $LIM \
    --max-tokens 24000 --timeout 420 --budget $BUD --cache-every 1 \
    > $F/logs/$NAME.log 2>&1
  return $?
}

echo "===== 스모크 30문장 $(ts) ====="
label smoke_full "--limit 30" 1.0
echo "  exit=$?"; sed -n '/^{/,/^}/p' $F/logs/smoke_full.log

# 게이트: 형식이 깨지면 $38 을 태우지 않는다. 기준은 3000 런과 같다.
$PY - <<'PYGATE' || { echo "!! 게이트 불통과 — 본런 중단"; mark full_label.failed "smoke gate"; exit 1; }
import json, re, sys
t = open("/home/mobility/STiTy/core/meaning_segmentator/experiment/artifacts/covost2/full/logs/smoke_full.log").read()
m = re.search(r"\{.*\}", t, re.S)
if not m:
    print("  요약 JSON 없음"); sys.exit(1)
d = json.loads(m.group(0))
ok = (d["format_pass"] >= 0.90 and d["text_preserved"] >= 0.95
      and d["coverage_met"] >= 0.90)
print("  게이트: format=%s preserved=%s coverage=%s 경계=%s/%s (3000런 3.80) -> %s"
      % (d["format_pass"], d["text_preserved"], d["coverage_met"],
         d["mean_boundaries"], d["mean_required"], "통과" if ok else "불통과"))
sys.exit(0 if ok else 1)
PYGATE

echo "===== 본런 15,531문장 $(ts) — 추정 \$38 / ~4시간 ====="
label covost2_full_run13_mg1 "" 50.0
rc=$?; echo "  exit=$rc $(ts)"; sed -n '/^{/,/^}/p' $F/logs/covost2_full_run13_mg1.log
grep "누적 비용" $F/logs/covost2_full_run13_mg1.log | tail -3
[ $rc -eq 0 ] || { echo "!! 본런 실패 — 캐시는 남아 있다"; mark full_label.failed "label exit=$rc"; exit 1; }

echo "===== prompt_eval 변환 $(ts)  격자=$GRID ====="
$PY core/meaning_segmentator/tools/covost2_label/to_prompt_eval.py \
  --labels $F/labels/covost2_full_run13_mg1.jsonl \
  --run-id covost2/full --label auto_run13_mg1 --min-gap 1 \
  --t-grid $GRID --prompt-file $PROMPT 2>&1 | tail -6

mark full_label.done "ok"
echo "===== 라벨링 완료 $(ts) ====="
