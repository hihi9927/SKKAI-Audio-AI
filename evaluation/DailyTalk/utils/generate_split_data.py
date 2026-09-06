"""
generate_split_data.py  (DailyTalk)

문장 중간에서 잘린(partial) 오디오-텍스트 쌍을 만든다. forced aligner 로 cut 지점의
시각을 찾아 wav 를 자르고, 그에 맞춰 자른 텍스트를 함께 낸다.

**현행(2세대) 사용법.** 기본값은 아래 1세대 경로를 가리키므로 인자를 반드시 넘긴다:

    python evaluation/DailyTalk/utils/generate_split_data.py \
        --input  evaluation/DailyTalk/transcribe/partial_input.json \
        --output evaluation/DailyTalk/transcribe/partial_all.json

  build_splits.py 가 고른 partial 대상만 처리하고, 결과를 assemble_dailytalk.py 가
  train/val/test.jsonl 로 조립한다.

**1세대(2026-08-24) 경로 — 아래 기본값이 이것이다.**

- 입력: evaluation/DailyTalk/results/train2_seg_en.json  (train2 분할 시절의 라벨)
- 출력 JSON: evaluation/DailyTalk/transcribe/split_train2.json
- 그 뒤: Qwen3-ASR/finetuning/utils/convert_split_to_jsonl.py → split_train.jsonl

  인자 없이 돌리면 현행 데이터가 아니라 1세대를 다시 만든다. 기본값을 그대로 두는 것은
  1세대 산출물(split_train2.json 등)이 아직 남아 있어 재현 경로를 보존하기 위해서다.

- 오디오 소스: finetuning/data/DailyTalk/audio/{file}.wav
- 출력 오디오: finetuning/data/DailyTalk/split_audio/split_{file}.wav

cut 규칙:
  - 전체 단어 수 3개 이상인 utterance만 처리
  - cut 후 partial이 반드시 2단어 이상 남아야 함
  - <SEG> 직전 단어(분절 경계)에서는 cut 불가
  - 마지막 단어에서는 cut 불가
  - partial text는 원본 단어 문자열 그대로 사용 (구두점 보존)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from scipy.io import wavfile

# ── 경로 설정 ─────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_EVAL_BASE   = _SCRIPT_DIR.parent.parent
_STiTy_BASE  = _EVAL_BASE.parent
_FT_BASE     = _STiTy_BASE / "Qwen3-ASR" / "finetuning"

# 1세대 기본값이다. 2세대는 --input / --output 으로 partial_input.json →
# partial_all.json 을 넘긴다 (docstring 참조).
INPUT_JSON   = _EVAL_BASE / "DailyTalk" / "results" / "train2_seg_en.json"
AUDIO_DIR    = _FT_BASE / "data" / "DailyTalk" / "audio"
SPLIT_DIR    = _FT_BASE / "data" / "DailyTalk" / "split_audio"
OUTPUT_JSON  = _EVAL_BASE / "DailyTalk" / "transcribe" / "split_train2.json"
MODEL_ID     = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE     = "English"


# ── seg 구조 분석 ─────────────────────────────────────────────────────────

def analyze_seg(seg_text: str, aligner_processor, language: str):
    """
    seg_text를 <SEG>로 분리하고 각 단어의 aligner 토큰 수를 계산.
    영어는 tokenize_space_lang이 whitespace 기준이라 단어 ≈ 토큰 (구두점만 제거).

    Returns:
        seg_parts  : list[str]
        seg_info   : list of (words: list[str], word_token_counts: list[int])
        word_end_indices : dict {token_idx: cumulative_word_count}
        before_seg_set   : set[int]
        n_tokens   : int
    """
    seg_parts = [s.strip() for s in re.split(r'<SEG>', seg_text) if s.strip()]

    seg_info = []
    for part in seg_parts:
        words = part.split()
        w_counts = []
        for w in words:
            wl, _ = aligner_processor.encode_timestamp(w, language)
            w_counts.append(max(len(wl), 1))
        seg_info.append((words, w_counts))

    word_end_indices = {}  # {token_idx: cumulative_word_count}
    before_seg_set = set()
    cumsum = 0
    word_cumsum = 0

    for part_idx, (words, w_counts) in enumerate(seg_info):
        part_total = sum(w_counts)
        w_cumsum = 0
        for count in w_counts:
            w_cumsum += count
            word_cumsum += 1
            word_end_indices[cumsum + w_cumsum - 1] = word_cumsum
        if part_idx < len(seg_info) - 1:
            before_seg_set.add(cumsum + part_total - 1)
        cumsum += part_total

    n_tokens = cumsum
    return seg_parts, seg_info, word_end_indices, before_seg_set, n_tokens


def build_partial(seg_parts: list, seg_info: list, k: int):
    """
    token index k까지의 partial_text와 partial_seg를 원본 단어 문자열로 반환.
    """
    partial_parts = []
    cumsum = 0

    for i, (part, (words, w_counts)) in enumerate(zip(seg_parts, seg_info)):
        if i > 0:
            if cumsum - 1 < k:
                partial_parts.append('<SEG>')
            else:
                break

        part_total = sum(w_counts)
        k_in_part = k - cumsum

        if k_in_part >= part_total:
            partial_parts.append(part)
            cumsum += part_total
        else:
            w_cumsum = 0
            n_words = 0
            for count in w_counts:
                w_cumsum += count
                n_words += 1
                if w_cumsum > k_in_part:
                    break
            partial_parts.append(' '.join(words[:n_words]))
            break

    partial_seg  = ' '.join(partial_parts)
    partial_text = ' '.join(t for t in partial_parts if t != '<SEG>')
    return partial_text, partial_seg


# ── 오디오 trim ───────────────────────────────────────────────────────────

def trim_and_save(src_path: Path, end_time: float, dst_path: Path):
    sr, data = wavfile.read(src_path)
    end_sample = min(int(end_time * sr), len(data))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(dst_path, sr, data[:end_sample])


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=MODEL_ID)
    parser.add_argument("--input",     default=str(INPUT_JSON))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--output",    default=str(OUTPUT_JSON))
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",    action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path  = Path(args.input)
    audio_dir   = Path(args.audio_dir)
    split_dir   = Path(args.split_dir)
    output_path = Path(args.output)

    split_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding='utf-8') as f:
        src = json.load(f)

    if args.resume and output_path.exists():
        with open(output_path, encoding='utf-8') as f:
            out = json.load(f)
        done_files = {e['original_file'] for gk in out for e in out[gk]['data']}
        print(f"Resume: 이미 처리된 {len(done_files)}건 건너뜀")
    else:
        out = {}
        done_files = set()

    print(f"Aligner 로드 중: {args.model}")
    sys.path.insert(0, str(_STiTy_BASE / "Qwen3-ASR"))
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=torch.float16 if args.device != 'cpu' else torch.float32,
    )
    ap = aligner.aligner_processor
    print("Aligner 로드 완료\n")

    def save():
        output_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    all_entries = [(gk, e) for gk in src for e in src[gk]['data']]
    total = len(all_entries)
    processed = skipped = no_valid_cut = 0

    for idx, (gk, entry) in enumerate(all_entries):
        file_id  = entry['file']
        text     = entry.get('text', '')
        seg_text = entry.get('seg_text') or text

        print(f"[{idx+1}/{total}] {file_id}", end='')

        if file_id in done_files:
            print(" → 건너뜀")
            skipped += 1
            continue

        # ── seg 구조 분석 ──────────────────────────────────────────
        seg_parts, seg_info, word_ends, before_seg_set, n_tokens = analyze_seg(
            seg_text, ap, LANGUAGE
        )

        # 전체 단어 수 3개 이상 확인
        n_words_total = sum(len(w) for w, _ in seg_info)
        if n_words_total < 3:
            print(f" → 단어 수 {n_words_total}개, 스킵")
            no_valid_cut += 1
            continue

        # 유효 cut: 단어 경계, 마지막 토큰 제외, before_seg 제외, cut 후 2단어 이상
        valid_cuts = sorted(
            i for i, n_w in word_ends.items()
            if i < n_tokens - 1
            and i not in before_seg_set
            and n_w >= 2
        )

        if not valid_cuts:
            print(f" → 유효 cut 없음, 스킵")
            no_valid_cut += 1
            continue

        # ── Forced Alignment ───────────────────────────────────────
        src_wav = audio_dir / f"{file_id}.wav"
        if not src_wav.exists():
            print(f" → 오디오 없음, 스킵")
            skipped += 1
            continue

        try:
            results = aligner.align(audio=str(src_wav), text=text, language=LANGUAGE)
            aligned_words = results[0]
        except Exception as e:
            print(f" → align 오류: {e}, 스킵")
            skipped += 1
            continue

        n_aligned = len(aligned_words)
        valid_cuts = [k for k in valid_cuts if k < n_aligned]
        if not valid_cuts:
            print(f" → align 토큰 수 불일치 ({n_aligned} vs {n_tokens}), 스킵")
            no_valid_cut += 1
            continue

        # ── 랜덤 cut ───────────────────────────────────────────────
        k = random.choice(valid_cuts)
        end_time = aligned_words[k].end_time

        # ── partial text (원본 단어 문자열 기반) ───────────────────
        partial_text, partial_seg = build_partial(seg_parts, seg_info, k)

        # ── 오디오 trim & 저장 ─────────────────────────────────────
        dst_wav = split_dir / f"split_{file_id}.wav"
        try:
            trim_and_save(src_wav, end_time, dst_wav)
        except Exception as e:
            print(f" → trim 오류: {e}, 스킵")
            skipped += 1
            continue

        if gk not in out:
            out[gk] = {"data": []}
        out[gk]["data"].append({
            "file":          f"split_{file_id}",
            "text":          partial_text,
            "seg_text":      partial_seg,
            "original_file": file_id,
            "cut_word_idx":  k,
            "end_time":      end_time,
        })
        save()

        print(f" → cut@{k} ({end_time:.2f}s) | \"{partial_text[:40]}\"")
        processed += 1

    print(f"\n완료: 처리 {processed} / 건너뜀 {skipped} / 유효 cut 없음 {no_valid_cut}")
    print(f"출력: {output_path}")


if __name__ == "__main__":
    main()
