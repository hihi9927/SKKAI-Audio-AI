#!/usr/bin/env python3
"""
Drop-in replacement for Kokoro-Speech-Dataset/extract.py that uses librosa
instead of torchaudio, avoiding the torchcodec/FFmpeg shared-library issue.

Usage (run from the cloned repo directory):
  python3 /path/to/extract_librosa.py --size tiny
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf


def read_params_list(data_dir: str, size: str) -> list[dict]:
    index_file = os.path.join(data_dir, 'index.json')
    if not os.path.exists(index_file):
        print(f"ERROR: '{index_file}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(index_file) as f:
        params_list = json.load(f)
    return [
        p for p in params_list
        if size == 'xlarge' or size in p['sizes'].split()
    ]


def extract_wav_files(data_dir: str, params_list: list[dict], sample_rate: int, output_dir: str):
    wavs_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    for params in params_list:
        book_id = params['id']
        metadata_file = os.path.join(data_dir, f'{book_id}.metadata.txt')
        audio_dir = os.path.join(data_dir, book_id)

        current_file = None
        current_audio = None

        with open(metadata_file, 'rt') as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                clip_id, audio_file, audio_start, audio_end, _, _ = parts
                audio_start, audio_end = int(audio_start), int(audio_end)

                if current_file != audio_file:
                    file_path = os.path.join(audio_dir, audio_file)
                    print(f'\rReading {file_path}', end='', flush=True)
                    # librosa loads mono float32 at the requested sample rate
                    current_audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
                    current_file = audio_file

                clip = current_audio[audio_start:audio_end]

                # Normalize and convert to int16 (matches original extract.py)
                max_val = np.max(np.abs(clip))
                if max_val > 0:
                    clip = clip / max_val
                clip_int16 = (clip * np.iinfo(np.int16).max).astype(np.int16)

                output_file = os.path.join(wavs_dir, f'{clip_id}.wav')
                sf.write(output_file, clip_int16, sample_rate)

    print()  # newline after \r progress


def write_metafile(data_dir: str, params_list: list[dict], output_dir: str):
    metadata_out = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_out, 'wt', encoding='utf-8') as out_f:
        for params in params_list:
            book_id = params['id']
            metadata_file = os.path.join(data_dir, f'{book_id}.metadata.txt')
            with open(metadata_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\r\n').split('|')
                    clip_id, _, _, _, text, voca = parts
                    out_f.write(f'{clip_id}|{text}|{voca}\n')
    print(f'metadata.csv written → {metadata_out}')


def main():
    parser = argparse.ArgumentParser(description='Kokoro clip extractor (librosa backend)')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--size', default='tiny',
                        choices=['tiny', 'small', 'large', 'xlarge'])
    parser.add_argument('--sample-rate', type=int, default=22050)
    args = parser.parse_args()

    params_list = read_params_list(args.data_dir, args.size)
    if not params_list:
        print(f'ERROR: No entries found for size={args.size}', file=sys.stderr)
        sys.exit(1)

    print(f'Extracting {len(params_list)} book(s) for size={args.size}')
    extract_wav_files(args.data_dir, params_list, args.sample_rate, args.output_dir)
    write_metafile(args.data_dir, params_list, args.output_dir)
    print(f'Done. WAV files → {args.output_dir}/wavs/')


if __name__ == '__main__':
    main()
