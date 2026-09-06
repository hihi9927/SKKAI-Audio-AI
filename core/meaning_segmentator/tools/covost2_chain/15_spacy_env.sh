#!/bin/bash
# `syntax`(SASST) 비교군 전용 격리 venv.
#
# en_core_web_trf 는 spacy-transformers 를 요구하고, 그게 **transformers<4.50 을 강제해
# 레포 핀(4.57.6)을 깬다.** 한 번 이 사고가 있었다 (baselines/README.md). 그래서
# system-site-packages 없이 완전 격리된 venv 에 따로 깐다.
. core/meaning_segmentator/tools/covost2_chain/common.sh
V=$HOME/.venvs/spacyenv

echo "===== spacy 격리 venv $(ts) ====="
mkdir -p $HOME/.venvs && python3 -m venv $V || { mark spacyenv.failed "venv 생성"; exit 1; }
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q "spacy>=3.8,<3.9" spacy-transformers torch || { mark spacyenv.failed "pip"; exit 1; }
$V/bin/python -m spacy download en_core_web_trf || { mark spacyenv.failed "모델"; exit 1; }
$V/bin/python -c "import spacy; spacy.load('en_core_web_trf'); print('로드 OK')" \
  || { mark spacyenv.failed "로드"; exit 1; }
echo "  transformers: $($V/bin/python -c 'import transformers;print(transformers.__version__)')"
mark spacyenv.done "ok"
echo "===== 완료 $(ts) ====="
