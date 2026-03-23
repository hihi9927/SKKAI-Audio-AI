#!/bin/bash

# KsponSpeech PCM -> WAV 변환 스크립트
# 스펙: 16kHz, 16-bit, mono, headerless little-endian linear PCM

DATA_DIR="/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/evaluation/KsponSpeech/data"
WAV_DIR="/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/Qwen3-ASR/finetuning/data/audio"

mkdir -p "$WAV_DIR"

mapfile -d '' pcm_files < <(find "$DATA_DIR" -name "*.pcm" -print0 | sort -z)
total=${#pcm_files[@]}
echo "변환할 PCM 파일 수: $total"

count=0
for pcm_file in "${pcm_files[@]}"; do
    filename=$(basename "${pcm_file%.pcm}")
    wav_file="$WAV_DIR/${filename}.wav"
    ffmpeg -y -f s16le -ar 16000 -ac 1 -i "$pcm_file" "$wav_file" -loglevel error
    count=$((count + 1))
    echo -ne "\r진행: $count / $total"
done

echo -e "\n완료"