#!/bin/bash

# KsponSpeech PCM -> WAV 변환 스크립트
# 스펙: 16kHz, 16-bit, mono, headerless little-endian linear PCM

DATA_DIR="/home/skkai/Desktop/STiTy/evaluation/KsponSpeech/pcm_data"
WAV_DIR="/home/skkai/Desktop/STiTy/evaluation/KsponSpeech/wav_data"

mkdir -p "$WAV_DIR"

pcm_files=$(find "$DATA_DIR" -name "*.pcm")
total=$(echo "$pcm_files" | wc -l)
echo "변환할 PCM 파일 수: $total"

count=0
while IFS= read -r pcm_file; do
    filename=$(basename "${pcm_file%.pcm}")
    wav_file="$WAV_DIR/${filename}.wav"
    ffmpeg -y -f s16le -ar 16000 -ac 1 -i "$pcm_file" "$wav_file" -loglevel error
    count=$((count + 1))
    echo -ne "\r진행: $count / $total"
done <<< "$pcm_files"

echo -e "\n완료"