"""
generate_split_data.py  (KsponSpeech)

KsponSpeech train2_seg.json 기반으로 문장 중간에 잘린(partial) 오디오-텍스트 쌍을 생성합니다.

- 입력: evaluation/KsponSpeech/results/train2_seg.json
- 오디오 소스: evaluation/KsponSpeech/data/KsponSpeech_{XXXX}/{file}.pcm  (16kHz, s16le, mono)
- 출력 오디오: evaluation/KsponSpeech/split_audio/split_{file}.wav
- 출력 JSON: evaluation/KsponSpeech/transcribe/split_train2.json

★ cut 위치는 반드시 어절(eojeol) 경계:
  - LTokenizer가 어절 하나를 여러 토큰으로 분리할 수 있으므로,
    어절의 마지막 토큰 index에서만 cut 허용.
  - partial text는 원본 어절 문자열을 그대로 사용 (띄어쓰기 보존).
  - end_time은 해당 어절 마지막 토큰의 end_time 사용.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

# ── 경로 설정 ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL_BASE  = _SCRIPT_DIR.parent.parent
_STiTy_BASE = _EVAL_BASE.parent

INPUT_JSON  = _EVAL_BASE / "KsponSpeech" / "results" / "train2_seg.json"
DATA_DIR    = _EVAL_BASE / "KsponSpeech" / "data"
SPLIT_DIR   = _EVAL_BASE / "KsponSpeech" / "split_audio"
OUTPUT_JSON = _EVAL_BASE / "KsponSpeech" / "transcribe" / "split_train2.json"
MODEL_ID    = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE    = "Korean"
PCM_SR      = 16000


# ── 오디오 유틸 ───────────────────────────────────────────────────────────

def get_pcm_path(data_dir: Path, file_id: str) -> Path:
    num = int(file_id.split('_')[-1])
    folder_num = (num - 1) // 1000 + 1
    return data_dir / f"KsponSpeech_{folder_num:04d}" / f"{file_id}.pcm"


def load_pcm(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.int16)


def trim_and_save_wav(pcm: np.ndarray, end_time: float, dst: Path, sr: int = PCM_SR):
    end_sample = min(int(end_time * sr), len(pcm))
    dst.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(dst, sr, pcm[:end_sample])


# ── seg 구조 분석 (어절 단위) ────────────────────────────────────────────

def analyze_seg(seg_text: str, aligner_processor, language: str):
    """
    seg_text를 <SEG>로 분리하고, 각 구간의 어절별 aligner 토큰 수를 계산.

    Returns:
        seg_parts  : list[str]       — <SEG>로 분리된 원본 구간 텍스트
        seg_info   : list of (eojeols: list[str], ej_token_counts: list[int])
        eojeol_end_indices : set[int] — 어절 마지막 aligner token의 절대 index
        before_seg_set     : set[int] — <SEG> 직전 어절 마지막 토큰의 index
        n_tokens   : int             — 전체 aligner 토큰 수
    """
    seg_parts = [s.strip() for s in re.split(r'<SEG>', seg_text) if s.strip()]

    seg_info = []
    for part in seg_parts:
        eojeols = part.split()
        ej_counts = []
        for ej in eojeols:
            wl, _ = aligner_processor.encode_timestamp(ej, language)
            ej_counts.append(max(len(wl), 1))  # 최소 1 (punctuation-only 어절 대비)
        seg_info.append((eojeols, ej_counts))

    # 각 어절의 마지막 토큰 절대 index → 해당 시점까지의 누적 어절 수
    eojeol_end_indices = {}  # {token_idx: cumulative_eojeol_count}
    before_seg_set = set()
    cumsum = 0
    eojeol_cumsum = 0

    for part_idx, (eojeols, ej_counts) in enumerate(seg_info):
        part_total = sum(ej_counts)
        ej_cumsum = 0
        for count in ej_counts:
            ej_cumsum += count
            eojeol_cumsum += 1
            eojeol_end_indices[cumsum + ej_cumsum - 1] = eojeol_cumsum
        # 이 구간의 마지막 토큰 = before_seg (마지막 구간 제외)
        if part_idx < len(seg_info) - 1:
            before_seg_set.add(cumsum + part_total - 1)
        cumsum += part_total

    n_tokens = cumsum
    return seg_parts, seg_info, eojeol_end_indices, before_seg_set, n_tokens


def build_partial(seg_parts: list, seg_info: list, k: int):
    """
    어절 경계 cut index k에 대해 partial_text와 partial_seg를 반환.
    원본 어절 문자열을 그대로 사용해 띄어쓰기를 보존.
    """
    partial_parts = []  # 어절 그룹 또는 '<SEG>'
    cumsum = 0

    for i, (part, (eojeols, ej_counts)) in enumerate(zip(seg_parts, seg_info)):
        if i > 0:
            # 이전 구간이 k 이전에 끝났을 때만 <SEG> 삽입
            if cumsum - 1 < k:
                partial_parts.append('<SEG>')
            else:
                break  # k가 이전 구간에서 끝남

        part_total = sum(ej_counts)
        k_in_part = k - cumsum  # 이 구간 안에서의 상대 index

        if k_in_part >= part_total:
            # 이 구간 전체가 k 이전
            partial_parts.append(part)
            cumsum += part_total
        else:
            # k가 이 구간 내부 → 어절 수 결정
            ej_cumsum = 0
            n_eojeols = 0
            for count in ej_counts:
                ej_cumsum += count
                n_eojeols += 1
                if ej_cumsum > k_in_part:
                    break
            partial_parts.append(' '.join(eojeols[:n_eojeols]))
            break

    partial_seg  = ' '.join(partial_parts)
    partial_text = ' '.join(t for t in partial_parts if t != '<SEG>')
    return partial_text, partial_seg


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=MODEL_ID)
    parser.add_argument("--input",     default=str(INPUT_JSON))
    parser.add_argument("--data-dir",  default=str(DATA_DIR))
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--output",    default=str(OUTPUT_JSON))
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",    action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path  = Path(args.input)
    data_dir    = Path(args.data_dir)
    split_dir   = Path(args.split_dir)
    output_path = Path(args.output)

    split_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding='utf-8') as f:
        raw = json.load(f)
    src_data = raw['data'] if isinstance(raw, dict) else raw

    if args.resume and output_path.exists():
        with open(output_path, encoding='utf-8') as f:
            out_raw = json.load(f)
        out_data = out_raw['data'] if isinstance(out_raw, dict) else out_raw
        done_files = {e['original_file'] for e in out_data}
        print(f"Resume: 이미 처리된 {len(done_files)}건 건너뜀")
    else:
        out_data = []
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
            json.dumps({"data": out_data}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    total = len(src_data)
    processed = skipped = no_valid_cut = 0

    for idx, entry in enumerate(src_data):
        file_id  = entry['file']
        text     = entry.get('text', '')
        seg_text = entry.get('seg_text') or text

        print(f"[{idx+1}/{total}] {file_id}", end='')

        if file_id in done_files:
            print(" → 건너뜀")
            skipped += 1
            continue

        # ── seg 구조 분석 ──────────────────────────────────────────
        seg_parts, seg_info, eojeol_ends, before_seg_set, n_tokens = analyze_seg(
            seg_text, ap, LANGUAGE
        )

        # 전체 어절 수 (cut 후 최소 1어절이 남으려면 총 3어절 이상 필요)
        n_eojeols = sum(len(ej_counts) for _, ej_counts in seg_info)
        if n_eojeols < 3:
            print(f" → 어절 수 {n_eojeols}개, 스킵")
            no_valid_cut += 1
            continue

        # 어절 경계 중에서 유효한 cut 후보 선정
        # (마지막 토큰 제외, before_seg 제외, cut 후 최소 2어절 이상 남아야 함)
        valid_cuts = sorted(
            i for i, n_ej in eojeol_ends.items()
            if i < n_tokens - 1
            and i not in before_seg_set
            and n_ej >= 2
        )

        if not valid_cuts:
            print(f" → 유효 cut 없음, 스킵")
            no_valid_cut += 1
            continue

        # ── PCM 로드 ───────────────────────────────────────────────
        pcm_path = get_pcm_path(data_dir, file_id)
        if not pcm_path.exists():
            print(f" → PCM 없음, 스킵")
            skipped += 1
            continue

        try:
            pcm = load_pcm(pcm_path)
        except Exception as e:
            print(f" → PCM 로드 오류: {e}, 스킵")
            skipped += 1
            continue

        audio_f32 = pcm.astype(np.float32) / 32768.0

        # ── Forced Alignment ───────────────────────────────────────
        try:
            results = aligner.align(
                audio=(audio_f32, PCM_SR),
                text=text,
                language=LANGUAGE,
            )
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

        # ── 랜덤 cut (어절 경계) ───────────────────────────────────
        k = random.choice(valid_cuts)
        end_time = aligned_words[k].end_time

        # ── partial text 생성 (원본 어절 문자열 기반) ──────────────
        partial_text, partial_seg = build_partial(seg_parts, seg_info, k)

        # ── 오디오 trim & 저장 ─────────────────────────────────────
        dst_wav = split_dir / f"split_{file_id}.wav"
        try:
            trim_and_save_wav(pcm, end_time, dst_wav)
        except Exception as e:
            print(f" → trim 오류: {e}, 스킵")
            skipped += 1
            continue

        out_data.append({
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
