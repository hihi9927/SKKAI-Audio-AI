#!/bin/bash
# CoVoST2 en test **전체** (15,531 발화) 준비 — wav 추출 + 매니페스트 + 강제정렬.
# 여기까지는 API 비용 0 이다. 라벨링(12번)만 돈이 든다.
# 매니페스트 빌더와 정렬기는 둘 다 .venv 다 — soundfile / qwen_asr 가 .venv-autoseg 에 없다.
#
# en->{de,ja,zh} 는 **같은 영어 클립 15,531개를 공유**한다. wav 캐시는 config 을 안 타고
# `en_test/` 하나이므로 세 매니페스트가 같은 파일을 가리킨다 (build_manifest 헤더 참조).
. core/meaning_segmentator/tools/covost2_chain/common.sh
M=evaluation/ast/manifests
TAG=full

build_man () {   # <config> <출력 타깃 이름>
  echo "===== 매니페스트 $1 $(ts) ====="
  "$REPO/.venv/bin/python" -u evaluation/ast/build_manifest_covost2.py \
    --covost-root ~/datasets/covost2 --config $1 --split test --mode single \
    --audio-cache ~/datasets/covost2_single \
    --out $M/covost2_en-$2_$TAG.jsonl 2>&1 | tail -6
  return ${PIPESTATUS[0]}
}

build_man en_de    de || { mark full_prep.failed "manifest de"; exit 1; }
build_man en_ja    ja || { mark full_prep.failed "manifest ja"; exit 1; }
build_man en_zh-CN zh || { mark full_prep.failed "manifest zh"; exit 1; }

for t in de ja zh; do
  echo "  covost2_en-$t_$TAG.jsonl: $(wc -l < $M/covost2_en-$t_$TAG.jsonl) 줄"
done

# 강제정렬 — 소스가 영어 하나뿐이라 한 번만 돌린다. **.venv 를 써야 한다** (qwen_asr).
echo "===== 강제정렬 $(ts) ====="
"$REPO/.venv/bin/python" -u -m core.meaning_segmentator.autoseg.baselines.build_unittimes \
  --lang en --manifest $M/covost2_en-de_$TAG.jsonl \
  --out $M/covost2_en-de_${TAG}_unittimes.json --batch 8 \
  > $R/logs_full_align.log 2>&1
rc=$?; echo "  정렬 exit=$rc $(ts)"; tail -6 $R/logs_full_align.log
[ $rc -eq 0 ] || { mark full_prep.failed "align exit=$rc"; exit 1; }

# `bleu_eval` 은 `covost2_en_<tag>_wordtimes_qwen.json` 이름으로 찾는다 (datasets.wordtimes_path).
ln -sf covost2_en-de_${TAG}_unittimes.json $M/covost2_en_${TAG}_wordtimes_qwen.json
echo "  심볼릭 링크: covost2_en_${TAG}_wordtimes_qwen.json"

mark full_prep.done "n=$(wc -l < $M/covost2_en-de_$TAG.jsonl)"
echo "===== 준비 완료 $(ts) ====="
