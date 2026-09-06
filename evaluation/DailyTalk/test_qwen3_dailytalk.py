#!/usr/bin/env python3
"""DailyTalk 평가 클라이언트.

공통 로직은 `evaluation/harness/` 에 있다. 여기 남는 것은 DailyTalk 고유의 셋뿐이다 —
학습용 jsonl(`language English<asr_text>…`)에서 참조를 뽑는 일, 대화 id 로 묶는 일,
그리고 WER 채점이다.

같은 대화(`d1061`)의 발화들은 한 연결에서 이어 보낸다. 대화가 바뀌면 하네스가 연결을
새로 맺어 문맥을 끊는다.

    python evaluation/DailyTalk/test_qwen3_dailytalk.py \
        --test-jsonl Qwen3-ASR/finetuning/data/DailyTalk/test.jsonl \
        --model "finetuned(1.0.1)" --scope sample --tag run_01
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR
EVALUATION_ROOT = SCRIPT_DIR.parent
STiTy_ROOT = EVALUATION_ROOT.parent
sys.path.insert(0, str(EVALUATION_ROOT))

from harness import cli                       # noqa: E402
from harness.audio import load_soundfile      # noqa: E402
from harness.scoring import calculate_wer     # noqa: E402

logging.basicConfig(format='%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "baseline":         "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)":  "Qwen/Qwen3-ASR-1.7B",
    "finetuned":         str(STiTy_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"),
    "finetuned(1.0.1)":  str(STiTy_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"),
}
DEFAULT_TEST_JSONL = STiTy_ROOT / "Qwen3-ASR" / "finetuning" / "data" / "DailyTalk" / "test.jsonl"
DEFAULT_SERVER_SCRIPT = EVALUATION_ROOT / "streaming_websocket_server_ast.py"

_ASR_TEXT_RE = re.compile(r'<asr_text>(.*?)(?:<SEG>)?$', re.DOTALL)
_DIALOG_ID_RE = re.compile(r'(d\d+)')


def _parse_text(raw_text):
    """`language English<asr_text>TEXT [<SEG>]` 에서 전사만 뽑는다."""
    m = _ASR_TEXT_RE.search(raw_text)
    if m:
        return m.group(1).strip()
    return re.sub(r'\s*<SEG>', '', raw_text).strip()


def _extract_dialog_id(stem):
    """`split_0_1_d1061` → `d1061`. 못 찾으면 stem 을 그대로 쓴다."""
    m = _DIALOG_ID_RE.search(stem)
    return m.group(1) if m else stem


def find_audio_files(test_jsonl, split_only=True):
    """test.jsonl 을 읽어 목록을 만든다.

    split_only 가 참이면 미리 잘라 둔 평가셋(`split_audio_test`)만 쓴다. 상대 경로는
    CWD 가 아니라 jsonl 자신의 위치를 기준으로 푼다.
    """
    audio_files = []
    base_dir = Path(test_jsonl).resolve().parent
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_path = entry.get('audio', '')
            if audio_path and not Path(audio_path).is_absolute():
                audio_path = str((base_dir / audio_path).resolve())
            if split_only and 'split_audio_test' not in audio_path:
                continue
            transcript = _parse_text(entry.get('text', ''))
            if not transcript:
                continue
            stem = Path(audio_path).stem
            audio_files.append({
                'path': audio_path,
                'file_id': stem,
                'speaker_id': _extract_dialog_id(stem),
                'reference': transcript,
            })
    return sorted(audio_files, key=lambda x: x['file_id'])


def main():
    parser = cli.build_parser(
        'Qwen3 DailyTalk integration test with FSL metrics',
        default_server_script=DEFAULT_SERVER_SCRIPT,
        trailing_silence_ms=1000,
    )
    parser.add_argument('--test-jsonl', type=str, default=str(DEFAULT_TEST_JSONL))
    parser.add_argument('--split-only', action=argparse.BooleanOptionalAction, default=True,
                        help='split_audio_test 항목만 쓴다')
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    if not os.path.exists(args.test_jsonl):
        logger.error('test.jsonl not found: %s', args.test_jsonl)
        sys.exit(1)

    files = find_audio_files(args.test_jsonl, split_only=args.split_only)
    if not files:
        logger.error('No DailyTalk audio files found in %s', args.test_jsonl)
        sys.exit(1)

    cli.run(
        args,
        files=files,
        dataset_root=DATASET_ROOT,
        load_audio=load_soundfile,
        score_fn=calculate_wer,
        # 같은 대화는 한 연결에서 이어 보낸다.
        group_key=lambda file_id: (_extract_dialog_id(file_id), _extract_dialog_id(file_id)),
    )


if __name__ == '__main__':
    main()
