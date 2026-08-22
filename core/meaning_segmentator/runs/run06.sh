#!/bin/bash
cd /home/mobility/STiTy
set -a && . .env && set +a
PYTHONPATH=/home/mobility/STiTy /home/mobility/STiTy/.venv-autoseg/bin/python -u \
  -m core.meaning_segmentator.autoseg.loop \
  --dataset fleurs-en-de --src-lang English --tgt-lang German \
  --pair-id en-multi --run-id run06 --translator google \
  --candidate-t 4 --batch-size 6 --min-gap 3 \
  --v0-candidates 3 --v0-probe 40 --revision-candidates 4 \
  --main-t 6 \
  --seg-reasoning-effort medium --skip-translation-below 0.90 \
  --train 40 --train-pool 80 --dev 60 --test 100 \
  --iterations 3 --min-chars 25 --budget 8 --workers 24
echo "[EXIT] $?" 
