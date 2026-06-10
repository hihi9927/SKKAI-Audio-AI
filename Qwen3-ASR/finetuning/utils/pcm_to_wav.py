"""
KsponSpeech PCM(s16le, 16kHz, mono) → WAV 변환기.

입력 JSON({"data":[{"file":...}]})의 각 file_id에 대해 PCM을 찾아 WAV로 저장.
이미 존재하는 WAV는 건너뜀(resume).

PCM 위치 모드:
  --mode folder : data_dir/KsponSpeech_{NNNN}/{file}.pcm  (train, 1000개 단위 폴더)
  --mode flat   : data_dir/{file}.pcm                      (eval, 평면 디렉토리)

Usage:
  python pcm_to_wav.py --input orig_final.json \
    --data-dir ../../evaluation/KsponSpeech/data --mode folder \
    --output-dir ../data/KSponSpeech/audio
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 16000


def pcm_path_folder(data_dir: Path, file_id: str) -> Path:
    num = int(file_id.split('_')[-1])
    folder = (num - 1) // 1000 + 1
    return data_dir / f"KsponSpeech_{folder:04d}" / f"{file_id}.pcm"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON ({'data':[{'file':...}]}) 또는 file_id 리스트")
    p.add_argument("--data-dir", required=True, help="PCM 루트 디렉토리")
    p.add_argument("--output-dir", required=True, help="WAV 출력 디렉토리")
    p.add_argument("--mode", choices=["folder", "flat"], default="folder")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    items = raw["data"] if isinstance(raw, dict) else raw
    files = [e["file"] for e in items]

    done = skipped = missing = 0
    miss_list = []
    for i, fid in enumerate(files):
        dst = out_dir / f"{fid}.wav"
        if dst.exists():
            skipped += 1
            continue
        src = pcm_path_folder(data_dir, fid) if args.mode == "folder" else data_dir / f"{fid}.pcm"
        if not src.exists():
            missing += 1
            if len(miss_list) < 10:
                miss_list.append(fid)
            continue
        buf = src.read_bytes()
        if len(buf) % 2:          # 홀수 바이트(eval PCM 등) → 마지막 1바이트 버림
            buf = buf[:-1]
        pcm = np.frombuffer(buf, dtype=np.int16)
        wavfile.write(dst, SR, pcm)
        done += 1
        if (done + skipped) % 5000 == 0:
            print(f"  진행 {i+1}/{len(files)} (변환 {done} / 스킵 {skipped} / 없음 {missing})")

    print(f"완료: 변환 {done} / 이미존재 {skipped} / PCM없음 {missing}")
    if miss_list:
        print(f"  PCM 없음 예시: {miss_list}")


if __name__ == "__main__":
    main()
