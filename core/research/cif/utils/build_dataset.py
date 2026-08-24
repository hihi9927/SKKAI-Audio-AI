"""
CIF 학습 데이터셋 생성 스크립트.

입력: DailyTalk train_and_val.jsonl
  {"audio": "/path/to/audio.wav", "text": "language English<asr_text>Hello world <SEG> how are you <SEG>"}

출력: core/cif/train_and_val.jsonl
  {
    "audio": "/path/to/audio.wav",
    "text_clean": "Hello world how are you",
    "seg_timestamps": [0.82, 1.64],        # SEG 앞 단어 end_time (초)
    "seg_encoder_frames": [10, 20],         # int(ts_sec * 12.5) — 8x CNN downsampled encoder frame index
    "n_encoder_frames": 25,                 # 전체 인코더 프레임 수 (audio duration * 12.5)
    "n_segs": 2
  }

Mel 파라미터 (WhisperFeatureExtractor 기본값):
  sample_rate = 16000, hop_length = 160
  → 1 mel frame = 10ms
  → CNN 3x stride-2 = 8x downsampling
  → 1 encoder frame = 80ms = 12.5 encoder frames/sec
"""

import sys
import os
import json
import re
import wave
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

# qwen_asr가 패키지로 설치되어 있지 않을 경우 로컬 경로 추가
_REPO_ROOT = Path(__file__).resolve().parents[3] / "Qwen3-ASR"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen_asr import Qwen3ForcedAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

# 1 encoder frame = hop_length * 8 / sample_rate seconds
# = 160 * 8 / 16000 = 0.08 s → 12.5 frames/sec
ENCODER_FRAMES_PER_SEC = 12.5


def audio_duration_sec(path: str) -> float:
    with wave.open(path, "r") as f:
        return f.getnframes() / f.getframerate()


def parse_seg_and_clean(raw_text: str) -> Tuple[str, List[int]]:
    """
    "<asr_text>" 이후 텍스트에서 <SEG> 위치(앞 단어 인덱스)와 clean text를 추출한다.

    반환:
        clean_text: <SEG> 제거한 순수 전사 텍스트
        seg_word_positions: i번째 <SEG>가 몇 번째 단어 뒤에 오는지 (0-based)
    """
    asr_marker = "<asr_text>"
    if asr_marker in raw_text:
        raw_text = raw_text[raw_text.index(asr_marker) + len(asr_marker):]

    # SEG 바로 앞 단어의 인덱스를 추적
    # "Hello world <SEG> how are you <SEG>" → word 1("world")과 word 4("you") 뒤
    tokens = raw_text.split()
    word_idx = 0
    seg_word_positions = []  # SEG 앞 단어의 인덱스 (clean word list 기준)
    clean_words = []

    for tok in tokens:
        if tok == "<SEG>":
            if clean_words:  # clean_words가 있을 때만 경계로 인정
                seg_word_positions.append(len(clean_words) - 1)
        else:
            # 단어 중간에 붙어 있는 <SEG> 처리 (e.g., "world<SEG>")
            parts = re.split(r"(<SEG>)", tok)
            for part in parts:
                if part == "<SEG>":
                    if clean_words:
                        seg_word_positions.append(len(clean_words) - 1)
                elif part.strip():
                    clean_words.append(part.strip())

    clean_text = " ".join(clean_words)
    return clean_text, seg_word_positions


def seg_timestamps_from_alignment(alignment_items, seg_word_positions: List[int]) -> List[float]:
    """
    forced alignment 결과에서 seg_word_positions에 해당하는 end_time을 가져온다.
    단어 수 불일치 시 nearest-word fallback.
    """
    n_words = len(alignment_items)
    timestamps = []
    for pos in seg_word_positions:
        clamped = min(pos, n_words - 1)
        timestamps.append(alignment_items[clamped].end_time)
    return timestamps


def ts_to_encoder_frame(ts_sec: float) -> int:
    return int(ts_sec * ENCODER_FRAMES_PER_SEC)


def duration_to_n_encoder_frames(dur_sec: float) -> int:
    # CNN은 floor division이므로 int()로 충분
    return int(dur_sec * ENCODER_FRAMES_PER_SEC)


def build(
    input_jsonl: str,
    output_jsonl: str,
    model_path: str,
    batch_size: int,
    device: str,
    limit: int,
):
    log.info(f"Loading ForcedAligner from {model_path}")
    aligner = Qwen3ForcedAligner.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )

    # 입력 데이터 로드
    with open(input_jsonl) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # 상대 audio 경로는 매니페스트(jsonl) 위치 기준으로 푼다 — CWD 에 의존하지 않게.
    _base = Path(input_jsonl).resolve().parent
    for r in records:
        if r.get("audio") and not Path(r["audio"]).is_absolute():
            r["audio"] = str((_base / r["audio"]).resolve())
    if limit > 0:
        records = records[:limit]
    log.info(f"총 {len(records)}개 샘플 처리")

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0

    with open(out_path, "w") as out_f:
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start: batch_start + batch_size]

            audios, clean_texts, seg_positions_batch, durations = [], [], [], []
            for rec in batch:
                audio_path = rec["audio"]
                clean_text, seg_positions = parse_seg_and_clean(rec["text"])
                if not clean_text or not seg_positions:
                    n_skipped += 1
                    continue
                audios.append(audio_path)
                clean_texts.append(clean_text)
                seg_positions_batch.append(seg_positions)
                durations.append(audio_duration_sec(audio_path))

            if not audios:
                continue

            try:
                results = aligner.align(
                    audio=audios,
                    text=clean_texts,
                    language=["English"] * len(audios),
                )
            except Exception as e:
                log.warning(f"배치 정렬 실패 (start={batch_start}): {e}")
                n_skipped += len(audios)
                continue

            for i, (result, seg_positions, dur) in enumerate(
                zip(results, seg_positions_batch, durations)
            ):
                if len(result) == 0:
                    n_skipped += 1
                    continue

                seg_ts = seg_timestamps_from_alignment(result, seg_positions)
                seg_frames = [ts_to_encoder_frame(ts) for ts in seg_ts]
                n_enc_frames = duration_to_n_encoder_frames(dur)

                out_rec = {
                    "audio": audios[i],
                    "text_clean": clean_texts[i],
                    "seg_timestamps": seg_ts,
                    "seg_encoder_frames": seg_frames,
                    "n_encoder_frames": n_enc_frames,
                    "n_segs": len(seg_frames),
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_written += 1

            if (batch_start // batch_size + 1) % 10 == 0:
                log.info(f"  처리 완료: {batch_start + len(batch)}/{len(records)}")

    log.info(f"완료: {n_written}개 저장, {n_skipped}개 스킵 → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DailyTalk → CIF 학습 데이터셋 생성")
    parser.add_argument(
        "--input", default=str(
            Path(__file__).resolve().parents[3]
            / "Qwen3-ASR/finetuning/data/DailyTalk/train_and_val.jsonl"
        ),
    )
    parser.add_argument(
        "--output", default=str(
            Path(__file__).resolve().parents[1] / "data" / "dailytalk_10000.jsonl"
        ),
    )
    parser.add_argument("--model", default=FORCED_ALIGNER_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="디버그용 샘플 수 제한 (0=전체)")
    args = parser.parse_args()

    build(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
