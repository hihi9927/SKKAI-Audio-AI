#!/usr/bin/env python3
"""LibriSpeech 평가 클라이언트.

공통 로직은 `evaluation/harness/` 에 있다. 여기 남는 것은 LibriSpeech 고유의 셋뿐이다 —
`.trans.txt` 로 참조를 모으고 `.flac` 을 짝지어 목록을 만드는 일, 오디오 로더 선택,
그리고 WER 채점이다.

    python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
        --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
        --model "baseline(1.0.0)" --scope sample --tag run_01
"""
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent              # evaluation/LibriSpeech
EVALUATION_ROOT = DATASET_ROOT.parent          # evaluation
STiTy_ROOT = EVALUATION_ROOT.parent
sys.path.insert(0, str(EVALUATION_ROOT))

from harness import cli                       # noqa: E402
from harness.audio import load_soundfile      # noqa: E402
from harness.scoring import calculate_wer     # noqa: E402

logging.basicConfig(format='%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --model 값 → 실제 서버 모델 경로. --server-model 을 주면 무시된다.
MODEL_MAP = {
    "baseline":        "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)": "Qwen/Qwen3-ASR-1.7B",
    "finetuned":        str(STiTy_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"),
    "finetuned(1.0.1)": str(STiTy_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"),
}
DEFAULT_TEST_DIR = DATASET_ROOT / "LibriSpeech" / "test-other"
DEFAULT_SERVER_SCRIPT = EVALUATION_ROOT / "streaming_websocket_server_ast.py"


def find_audio_files(test_dir):
    """`{spk}-{chap}-{utt}.flac` 과 같은 폴더의 `.trans.txt` 를 짝지어 목록을 만든다."""
    audio_files = []
    transcript_map = {}

    for trans_file in Path(test_dir).rglob('*.trans.txt'):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    file_id, transcript = parts
                    transcript_map[file_id] = transcript

    for audio_file in Path(test_dir).rglob('*.flac'):
        file_id = audio_file.stem
        if file_id in transcript_map:
            audio_files.append({'path': str(audio_file), 'file_id': file_id,
                                'reference': transcript_map[file_id]})

    return sorted(audio_files, key=lambda x: x['file_id'])


def main():
    parser = cli.build_parser(
        'Qwen3 LibriSpeech integration test with FSL metrics',
        default_server_script=DEFAULT_SERVER_SCRIPT,
        trailing_silence_ms=8000,
    )
    parser.add_argument('--test-dir', type=str, default=str(DEFAULT_TEST_DIR))
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    if not os.path.isdir(args.test_dir):
        logger.error('Test directory not found: %s', args.test_dir)
        logger.error('Pass the real LibriSpeech test-other path with --test-dir.')
        sys.exit(1)

    files = find_audio_files(args.test_dir)
    if not files:
        logger.error('No LibriSpeech files found in %s', args.test_dir)
        sys.exit(1)

    cli.run(args, files=files, dataset_root=DATASET_ROOT,
            load_audio=load_soundfile, score_fn=calculate_wer)


if __name__ == '__main__':
    main()
