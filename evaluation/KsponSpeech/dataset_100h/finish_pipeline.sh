#!/bin/bash
cd "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy"
PY="/home/skkai/anaconda3/envs/speech_ai/bin/python"
echo "[finish] train splitgen 완료 대기..."
until ! pgrep -f "generate_split_data.py --input evaluation/KsponSpeech/transcribe/split_input.json" >/dev/null; do sleep 60; done
echo "[finish] train split 완료. val/eval split 시작 $(date)"

echo "[finish] === val split (idx 113001~116000, folder) ==="
$PY evaluation/KsponSpeech/utils/generate_split_data.py --input evaluation/KsponSpeech/transcribe/val_split_input.json --data-dir evaluation/KsponSpeech/data   --split-dir Qwen3-ASR/finetuning/data/KSponSpeech/split_audio --output evaluation/KsponSpeech/results/val_split_data.json --skipped-output evaluation/KsponSpeech/results/val_split_data_skipped.json 2>&1 | tee evaluation/KsponSpeech/results/val_split.log

echo "[finish] === eval_clean split (E00001~E01500, flat) ==="
$PY evaluation/KsponSpeech/utils/generate_split_data.py --input evaluation/KsponSpeech/transcribe/eval_clean_split_input.json --data-dir "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/KsponSpeech data/KsponSpeech_eval.zip/eval_clean" --flat-pcm   --split-dir Qwen3-ASR/finetuning/data/KSponSpeech/eval_split_audio --output evaluation/KsponSpeech/results/eval_clean_split_data.json --skipped-output evaluation/KsponSpeech/results/eval_clean_split_data_skipped.json 2>&1 | tee evaluation/KsponSpeech/results/eval_clean_split.log

echo "[finish] === eval_other split (E03001~E04500, flat) ==="
$PY evaluation/KsponSpeech/utils/generate_split_data.py --input evaluation/KsponSpeech/transcribe/eval_other_split_input.json --data-dir "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/KsponSpeech data/KsponSpeech_eval.zip/eval_other" --flat-pcm   --split-dir Qwen3-ASR/finetuning/data/KSponSpeech/eval_split_audio --output evaluation/KsponSpeech/results/eval_other_split_data.json --skipped-output evaluation/KsponSpeech/results/eval_other_split_data_skipped.json 2>&1 | tee evaluation/KsponSpeech/results/eval_other_split.log

echo "[finish] === jsonl 조립 ==="
$PY Qwen3-ASR/finetuning/utils/assemble_dataset.py 2>&1 | tee evaluation/KsponSpeech/results/assemble.log
echo "[finish] 전체 완료 $(date)"
