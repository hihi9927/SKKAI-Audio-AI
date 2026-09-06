#!/usr/bin/env python3
"""KsponSpeech 평가 클라이언트. 메트릭은 CER.

공통 로직은 `evaluation/harness/` 에 있다. 여기 남는 것은 KsponSpeech 고유의 셋뿐이다 —
JSON 레이블과 PCM 디렉토리를 짝지어 목록을 만드는 일, raw PCM 로더, CER 채점.

**클립마다 연결을 새로 맺는다.** LibriSpeech 는 같은 챕터 안에서 연결을 유지해 문맥을
잇지만, KsponSpeech 는 클립이 서로 무관해서 앞 발화의 문맥이 넘어오면 오염이다.
`group_key` 가 파일마다 다른 값을 돌려주면 하네스가 연결을 다시 맺는다.

    python evaluation/KsponSpeech/test_qwen3_kspon.py \
        --data-json evaluation/KsponSpeech/transcribe/eval_clean_1000.json \
        --data-dir  evaluation/KsponSpeech/data/eval_clean \
        --model "baseline(1.0.0)" --scope sample --tag run_01
"""
import json
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR
EVALUATION_ROOT = SCRIPT_DIR.parent
STiTy_ROOT = EVALUATION_ROOT.parent
sys.path.insert(0, str(EVALUATION_ROOT))

from harness import cli                       # noqa: E402
from harness.audio import load_raw_pcm        # noqa: E402
from harness.scoring import calculate_cer, compute_corpus_cer  # noqa: E402

logging.basicConfig(format='%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "baseline":         "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)":  "Qwen/Qwen3-ASR-1.7B",
    "finetuned":         str(STiTy_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"),
    "finetuned(1.0.1)":  str(STiTy_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"),
}
DEFAULT_DATA_JSON = SCRIPT_DIR / "transcribe" / "eval_clean_1000.json"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "eval_clean"
DEFAULT_SERVER_SCRIPT = EVALUATION_ROOT / "streaming_websocket_server_ast.py"


def find_audio_files(data_json, data_dir):
    """JSON 레이블과 PCM 디렉토리에서 {file_id, path, reference} 목록을 만든다."""
    with open(data_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    items = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
    data_dir = Path(data_dir)
    audio_files = []
    for item in items:
        file_id = item['file']
        pcm_path = data_dir / f'{file_id}.pcm'
        if not pcm_path.exists():
            logger.warning('PCM not found, skipping: %s', pcm_path)
            continue
        audio_files.append({
            'file_id': file_id,
            'path': str(pcm_path),
            'reference': item['text'],
            'speaker_id': 'kspon',
        })
    return audio_files


def main():
    parser = cli.build_parser(
        'Qwen3 KsponSpeech integration test (CER)',
        default_server_script=DEFAULT_SERVER_SCRIPT,
        trailing_silence_ms=5500,
        target_lang='en',
    )
    parser.add_argument('--data-json', type=str, default=str(DEFAULT_DATA_JSON),
                        help='평가 JSON 파일 경로')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='PCM 오디오 디렉토리')
    parser.add_argument('--src-lang', type=str, default='ko', help='ASR 소스 언어')
    parser.set_defaults(model='baseline(1.0.0)')
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    if not os.path.exists(args.data_json):
        logger.error('Data JSON not found: %s', args.data_json)
        sys.exit(1)
    if not os.path.isdir(args.data_dir):
        logger.error('Data directory not found: %s', args.data_dir)
        sys.exit(1)

    files = find_audio_files(args.data_json, args.data_dir)
    if not files:
        logger.error('No audio files found. JSON: %s  Dir: %s', args.data_json, args.data_dir)
        sys.exit(1)

    cli.run(
        args,
        files=files,
        dataset_root=DATASET_ROOT,
        load_audio=load_raw_pcm,
        score_fn=calculate_cer,
        metric_key='corpus_cer',
        # 클립마다 새 연결 — chapter 를 file_id 로 두면 하네스가 매번 다시 맺는다.
        group_key=lambda file_id: ('kspon', file_id),
        running_metric=compute_corpus_cer,
        running_metric_label='CER',
        smoke_lang=args.src_lang,
    )


if __name__ == '__main__':
    main()
