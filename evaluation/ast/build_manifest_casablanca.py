#!/usr/bin/env python3
"""Casablanca(방언 아랍어) → 평가용 manifest(JSONL) 생성기.

Casablanca 는 8개 아랍어 방언(Algeria/Egypt/Jordan/Mauritania/Morocco/Palestine/
UAE/Yemen)의 TV 방송 음성을 사람이 전사한 코퍼스다. train 은 미공개고 validation/
test 만 배포된다.

FLEURS 와 결정적으로 다른 점이 둘 있다.

1. **번역 참조가 없다.** `transcription` 한 컬럼뿐이라 BLEU/COMET 은 못 낸다.
   이 매니페스트는 **ASR 평가(WER) 전용**이다. `tgt_text` 는 빈 문자열로 채운다 —
   스키마를 FLEURS 와 맞춰 두어야 하류 도구가 같은 리더를 쓸 수 있어서 남긴다.
2. **오디오가 44.1kHz 스테레오다.** STiTy 파이프라인은 16kHz 모노 s16le 고정이라
   여기서 변환해 둔다. 서버에 44.1k 를 그대로 넣으면 조용히 잘못 인식된다.

레이아웃 (hf download 후):

    {root}/Jordan/test-00000-of-00001.parquet
    {root}/Jordan/validation-00000-of-00001.parquet

parquet 컬럼: audio{bytes,path}, seg_id, transcription, gender, duration

`datasets` 로 열면 오디오 디코딩에 torchcodec 을 요구하므로(datasets 4.x) pyarrow 로
직접 읽는다. speech_ai env 의 vllm 의존성을 건드리지 않으려는 의도다.

사용:

    python evaluation/ast/build_manifest_casablanca.py \
        --casablanca-root ~/datasets/casablanca --dialect Jordan --split test \
        --out evaluation/ast/manifests/casablanca_jordan_test.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

TARGET_SR = 16000

# 방언 디렉토리 → 서버/번역기가 쓰는 2글자 코드. 전부 아랍어라 코드는 같고,
# 구분은 `dialect` 필드로 남긴다.
DIALECT_DIRS = [
    "Algeria", "Egypt", "Jordan", "Mauritania",
    "Morocco", "Palestine", "UAE", "Yemen",
]


def to_16k_mono(raw: bytes) -> tuple[np.ndarray, float]:
    """parquet 에 담긴 wav 바이트를 16kHz 모노 float32 로 변환한다."""
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)                      # 스테레오 → 모노
    if sr != TARGET_SR:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR)
    return mono, len(mono) / TARGET_SR


def talk_id_from_path(path: str) -> str:
    """오디오 파일명에서 방송 회차를 뽑는다.

    파일명 예: `01 - مسلسل الشريكان الحلقة 1_1921.12_1927.82_6539_1.wav`
    뒤쪽 `_시작_끝_id_n` 을 떼면 회차 제목만 남는다. 같은 회차 발화가 한 화자/한
    녹음 조건을 공유하므로 그룹 단위 분석에 쓴다.
    """
    stem = Path(path).stem
    return re.sub(r"_[\d.]+_[\d.]+_\d+_\d+$", "", stem).strip()


def main() -> int:
    args = parse_args()

    root = Path(args.casablanca_root).expanduser().resolve()
    pq_files = sorted((root / args.dialect).glob(f"{args.split}-*.parquet"))
    if not pq_files:
        print(f"[에러] parquet 없음: {root / args.dialect}/{args.split}-*.parquet",
              file=sys.stderr)
        print(f"       받기: hf download UBC-NLP/Casablanca --repo-type dataset \\\n"
              f'              --include "{args.dialect}/*" --local-dir {root}',
              file=sys.stderr)
        return 1

    wav_dir = Path(args.wav_dir).expanduser().resolve() if args.wav_dir else \
        root / args.dialect / f"wav{TARGET_SR // 1000}k" / args.split
    wav_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, total_sec = 0, 0.0
    skipped = {"no_text": 0, "duration": 0, "decode": 0}

    with open(out_path, "w", encoding="utf-8") as out:
        for pq_file in pq_files:
            table = pq.read_table(pq_file)
            cols = table.to_pydict()
            n = table.num_rows
            for i in range(n):
                text = (cols["transcription"][i] or "").strip()
                if not text:
                    skipped["no_text"] += 1
                    continue

                try:
                    wave, duration = to_16k_mono(cols["audio"][i]["bytes"])
                except Exception as exc:                      # noqa: BLE001
                    print(f"  [건너뜀] {cols['seg_id'][i]} 디코딩 실패: {exc}",
                          file=sys.stderr)
                    skipped["decode"] += 1
                    continue

                if duration < args.min_duration or duration > args.max_duration:
                    skipped["duration"] += 1
                    continue

                seg_id = str(cols["seg_id"][i])
                utt_id = f"{args.dialect.lower()}_{args.split}_{seg_id}"
                wav_path = wav_dir / f"{utt_id}.wav"
                if args.overwrite or not wav_path.exists():
                    sf.write(wav_path, wave, TARGET_SR, subtype="PCM_16")

                out.write(json.dumps({
                    "utt_id": utt_id,
                    "wav": str(wav_path),
                    "offset": 0.0,
                    "duration": round(duration, 3),
                    "src_lang": "ar",
                    "tgt_lang": args.tgt_lang,
                    "src_text": text,
                    # Casablanca 에는 번역 참조가 없다. BLEU 를 내려면 별도로 채워야 한다.
                    "tgt_text": "",
                    "speaker_id": "",
                    "talk_id": talk_id_from_path(cols["audio"][i]["path"] or ""),
                    "dialect": args.dialect,
                    "gender": cols["gender"][i] or "",
                }, ensure_ascii=False) + "\n")
                kept += 1
                total_sec += duration
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break

    print(f"manifest: {out_path}")
    print(f"  wav(16k 모노): {wav_dir}")
    print(f"  {args.dialect}/{args.split} — 세그먼트 {kept}개 / "
          f"오디오 {total_sec / 3600:.2f}시간")
    print(f"  제외 — 전사 없음 {skipped['no_text']}, 길이 {skipped['duration']}, "
          f"디코딩 실패 {skipped['decode']}")
    if kept:
        print(f"  평균 길이 {total_sec / kept:.2f}초")
        print("\n  주의: tgt_text 가 비어 있다 — ASR(WER) 전용 매니페스트다.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Casablanca → ASR 평가 manifest(JSONL)")
    p.add_argument("--casablanca-root", default="~/datasets/casablanca",
                   help="hf download 로 받은 루트")
    p.add_argument("--dialect", default="Jordan", choices=DIALECT_DIRS)
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--out", required=True)
    p.add_argument("--wav-dir", default=None,
                   help="추출한 16k wav 를 둘 곳 (기본: {root}/{dialect}/wav16k/{split})")
    p.add_argument("--tgt-lang", default="en",
                   help="하류 도구 호환용 라벨. 참조 번역은 없다")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-duration", type=float, default=0.5)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--overwrite", action="store_true",
                   help="이미 있는 wav 도 다시 쓴다")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
