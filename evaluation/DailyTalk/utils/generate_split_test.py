"""
generate_split_test.py  (DailyTalk)

test.jsonl의 절반을 문장 중간에서 분절합니다.
  - 단어 수 3개 미만 항목은 eligible pool에서 제외
  - eligible 항목 중 랜덤으로 전체의 절반 선택
  - 선택 결과를 split_test_state.json에 저장 → resume 시 재사용 (재계산 없음)
  - 영구 스킵(유효 cut 없음 / 오디오 없음 / align 오류 등) 발생 시
    reserve pool에서 교체 항목을 뽑아 계속 처리
  - forced alignment으로 오디오 trim → split_audio_test/ 저장
  - test.jsonl을 in-place로 수정 (audio 경로 + text를 분절 버전으로 교체)
  - text 형식: "language English<asr_text>{partial_text}"  (<SEG> 없음)

cut 규칙:
  - 전체 단어 수 3개 이상인 utterance만 처리
  - cut 후 partial이 반드시 2단어 이상 남아야 함
  - <SEG> 직전 단어(분절 경계)에서는 cut 불가
  - 마지막 단어에서는 cut 불가
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
_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL_BASE  = _SCRIPT_DIR.parent.parent
_STiTy_BASE = _EVAL_BASE.parent
_FT_BASE    = _STiTy_BASE / "Qwen3-ASR" / "finetuning"

TEST_JSONL  = _FT_BASE / "data" / "DailyTalk" / "test.jsonl"
SPLIT_DIR   = _FT_BASE / "data" / "DailyTalk" / "split_audio_test"
STATE_FILE  = _FT_BASE / "data" / "DailyTalk" / "split_test_state.json"
MODEL_ID    = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE    = "English"
PREFIX      = "language English<asr_text>"


# ── 텍스트 파싱 ───────────────────────────────────────────────────────────

def parse_asr_text(text: str) -> str:
    if "<asr_text>" in text:
        return text.split("<asr_text>", 1)[1].strip()
    return text.strip()


def count_words(seg_text: str) -> int:
    parts = [s.strip() for s in re.split(r'<SEG>', seg_text) if s.strip()]
    return sum(len(p.split()) for p in parts)


def plain_text(seg_text: str) -> str:
    return re.sub(r'\s*<SEG>', '', seg_text).strip()


# ── seg 구조 분석 ─────────────────────────────────────────────────────────

def analyze_seg(seg_text: str, aligner_processor, language: str):
    seg_parts = [s.strip() for s in re.split(r'<SEG>', seg_text) if s.strip()]

    seg_info = []
    for part in seg_parts:
        words = part.split()
        w_counts = []
        for w in words:
            wl, _ = aligner_processor.encode_timestamp(w, language)
            w_counts.append(max(len(wl), 1))
        seg_info.append((words, w_counts))

    word_end_indices = {}
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

    return seg_parts, seg_info, word_end_indices, before_seg_set, cumsum


def build_partial(seg_parts: list, seg_info: list, k: int) -> str:
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

    return ' '.join(t for t in partial_parts if t != '<SEG>')


# ── 오디오 trim ───────────────────────────────────────────────────────────

def trim_and_save(src_path: Path, end_time: float, dst_path: Path):
    sr, data = wavfile.read(src_path)
    end_sample = min(int(end_time * sr), len(data))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(dst_path, sr, data[:end_sample])


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_ID)
    parser.add_argument("--test-jsonl", default=str(TEST_JSONL))
    parser.add_argument("--split-dir",  default=str(SPLIT_DIR))
    parser.add_argument("--state-file", default=str(STATE_FILE))
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",     action="store_true", default=True)
    parser.add_argument("--no-resume",  dest="resume", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)

    test_jsonl = Path(args.test_jsonl)
    split_dir  = Path(args.split_dir)
    state_path = Path(args.state_file)
    split_dir.mkdir(parents=True, exist_ok=True)

    # ── test.jsonl 로드 ────────────────────────────────────────────────────
    with open(test_jsonl, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f if line.strip()]

    n_total  = len(entries)
    n_target = n_total // 2

    # ── state 로드 or 초기화 ───────────────────────────────────────────────
    if args.resume and state_path.exists():
        with open(state_path, encoding='utf-8') as f:
            state = json.load(f)
        targets = state["targets"]   # 처리할 인덱스 목록 (교체분 포함)
        reserve = state["reserve"]   # 교체 후보 pool
        print(f"State 로드: targets {len(targets)}개, reserve {len(reserve)}개")
    else:
        eligible = [
            i for i, e in enumerate(entries)
            if count_words(parse_asr_text(e['text'])) >= 3
        ]
        if len(eligible) < n_target:
            print(f"경고: eligible({len(eligible)}) < 목표({n_target})")
            n_target = len(eligible)

        selected = random.sample(eligible, n_target)
        sel_set  = set(selected)
        reserve  = [i for i in eligible if i not in sel_set]
        random.shuffle(reserve)
        targets  = sorted(selected)

        state = {"targets": targets, "reserve": reserve}
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f"State 초기화: targets {len(targets)}개, reserve {len(reserve)}개")

    print(f"전체: {n_total}개 | 목표: {n_target}개\n")

    def save_state():
        state_path.write_text(
            json.dumps({"targets": targets, "reserve": reserve}, indent=2),
            encoding='utf-8',
        )

    def save_jsonl():
        with open(test_jsonl, 'w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')

    # ── Aligner 로드 ───────────────────────────────────────────────────────
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

    processed = skipped_resume = replaced = 0

    # targets는 실행 도중 reserve에서 추가될 수 있으므로 index로 순회
    idx = 0
    while idx < len(targets):
        i          = targets[idx]
        entry      = entries[i]
        audio_path = Path(entry['audio'])
        file_id    = audio_path.stem
        seg_text   = parse_asr_text(entry['text'])

        print(f"[{idx+1}/{len(targets)}] {file_id}", end='')

        # ── resume: 이미 처리된 항목 건너뜀 ──────────────────────────────
        if "split_audio_test" in entry['audio']:
            print(" → 건너뜀 (이미 처리)")
            skipped_resume += 1
            idx += 1
            continue

        # ── seg 구조 분석 ──────────────────────────────────────────────────
        seg_parts, seg_info, word_ends, before_seg_set, n_tokens = analyze_seg(
            seg_text, ap, LANGUAGE
        )
        valid_cuts = sorted(
            j for j, n_w in word_ends.items()
            if j < n_tokens - 1
            and j not in before_seg_set
            and n_w >= 2
        )

        if not valid_cuts:
            print(f" → 유효 cut 없음", end='')
            _add_replacement(targets, reserve, save_state)
            print(f" (reserve 남은: {len(reserve)}개)")
            replaced += 1
            idx += 1
            continue

        # ── 오디오 확인 ────────────────────────────────────────────────────
        if not audio_path.exists():
            print(f" → 오디오 없음", end='')
            _add_replacement(targets, reserve, save_state)
            print(f" (reserve 남은: {len(reserve)}개)")
            replaced += 1
            idx += 1
            continue

        # ── Forced Alignment ───────────────────────────────────────────────
        try:
            results = aligner.align(
                audio=str(audio_path),
                text=plain_text(seg_text),
                language=LANGUAGE,
            )
            aligned_words = results[0]
        except Exception as e:
            print(f" → align 오류: {e}", end='')
            _add_replacement(targets, reserve, save_state)
            print(f" (reserve 남은: {len(reserve)}개)")
            replaced += 1
            idx += 1
            continue

        n_aligned  = len(aligned_words)
        valid_cuts = [c for c in valid_cuts if c < n_aligned]
        if not valid_cuts:
            print(f" → align 토큰 수 불일치 ({n_aligned} vs {n_tokens})", end='')
            _add_replacement(targets, reserve, save_state)
            print(f" (reserve 남은: {len(reserve)}개)")
            replaced += 1
            idx += 1
            continue

        # ── 랜덤 cut & trim ────────────────────────────────────────────────
        k        = random.choice(valid_cuts)
        end_time = aligned_words[k].end_time
        partial  = build_partial(seg_parts, seg_info, k)

        dst_wav = split_dir / f"split_{file_id}.wav"
        try:
            trim_and_save(audio_path, end_time, dst_wav)
        except Exception as e:
            print(f" → trim 오류: {e}", end='')
            _add_replacement(targets, reserve, save_state)
            print(f" (reserve 남은: {len(reserve)}개)")
            replaced += 1
            idx += 1
            continue

        # ── test.jsonl 항목 교체 ───────────────────────────────────────────
        entries[i] = {
            "audio": str(dst_wav),
            "text":  f"{PREFIX}{partial}",
        }
        save_jsonl()

        print(f" → cut@{k} ({end_time:.2f}s) | \"{partial[:50]}\"")
        processed += 1
        idx += 1

    print(f"\n완료: 처리 {processed} / resume 건너뜀 {skipped_resume} / 교체 발생 {replaced}")
    print(f"수정된 test.jsonl: {test_jsonl}")
    print(f"분절 오디오: {split_dir}")


def _add_replacement(targets: list, reserve: list, save_state_fn):
    """reserve pool에서 교체 항목을 꺼내 targets에 추가."""
    if reserve:
        new_idx = reserve.pop(0)
        targets.append(new_idx)
        save_state_fn()
    else:
        print(" (reserve 소진, 교체 불가)", end='')


if __name__ == "__main__":
    main()
