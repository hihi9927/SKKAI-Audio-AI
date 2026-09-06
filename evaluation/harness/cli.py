"""공통 CLI 와 실행 흐름.

데이터셋 클라이언트는 이 셋만 만들어 `run()` 에 넘긴다.

    files       [{file_id, path, reference, speaker_id?}, ...]
    load_audio  경로 → float32 배열
    score_fn    (results, policy, emit_summary) → (전체값, 화자별 dict)

나머지(서버 기동, 스모크, meta.json, 배치 실행, 결과 저장)는 전부 여기서 한다.
"""
import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from .results import load_common_file_ids, resolve_run_dir
from .scoring import calculate_wer
from .server import ServerManager, fetch_server_config, run_protocol_smoke
from .stream import process_batch

logger = logging.getLogger(__name__)

DEFAULT_POLICY = 3


def build_parser(description, *, default_server_script, trailing_silence_ms=8000,
                 target_lang='ko'):
    """모든 데이터셋이 공유하는 인자. 데이터셋 고유 인자는 어댑터가 더 붙인다."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY],
                   help='호환용 필드. Qwen3 는 policy=3 뿐이다.')
    p.add_argument('--host', type=str, default='localhost')
    p.add_argument('--port', type=int, default=8765)

    # 결과 폴더 구조: results/{model}/{scope}/{tag}/
    p.add_argument('--model', type=str, default='finetuned',
                   help='대분류: 모델 종류 (예: baseline, finetuned)')
    p.add_argument('--scope', type=str, default='sample',
                   help='소분류: full(전체) / sample(일부) 또는 임의 문자열')
    p.add_argument('--tag', type=str, default=None,
                   help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    p.add_argument('--description', type=str, default=None, help='description.txt 에 저장')
    p.add_argument('--results-root', type=str, default=None,
                   help='결과 루트. 미지정 시 데이터셋 디렉토리의 results/')

    p.add_argument('--limit', type=int, default=None, help='처리할 최대 파일 수')
    p.add_argument('--calculate-wer', action=argparse.BooleanOptionalAction, default=True,
                   help='끝나고 채점 요약을 출력한다')
    p.add_argument('--log-level', type=str, default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    p.add_argument('--auto-server', action='store_true')
    p.add_argument('--server-script', type=str, default=str(default_server_script))
    p.add_argument('--server-model', type=str, default=None,
                   help='서버에 로드할 모델 경로. 미지정 시 --model 로 MODEL_MAP 추론')
    p.add_argument('--server-args', type=str, default='')
    p.add_argument('--target-lang', type=str, default=target_lang)
    p.add_argument('--common-files', type=str, default=None)
    p.add_argument('--random-sample', type=int, default=None)
    p.add_argument('--random-seed', type=int, default=42)
    p.add_argument('--skip-protocol-smoke', action='store_true')
    p.add_argument('--chunk-size-ms', type=int, default=200)
    p.add_argument('--send-interval-ms', type=int, default=200,
                   help='청크 전송 간격(ms). 200 이면 실시간 페이싱')
    p.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True,
                   help="로그에 커밋 경계를 'seg1 / seg2 /' 로 표시")
    p.add_argument('--fresh-start', action='store_true', default=False,
                   help='기존 결과를 무시하고 처음부터')
    p.add_argument('--trailing-silence-ms', type=int, default=trailing_silence_ms,
                   help='파일 뒤에 붙이는 묵음(ms). 확정 기회는 누적 오디오가 chunk_size 배수일 '
                        '때만 생기므로 짧으면 마지막 문장이 dot 으로 확정되기 전에 스트림이 끊긴다 '
                        '(실측: 5500ms 에서 0.11초 모자라 finish 로 빠졌다)')

    # 서버로 그대로 넘어가는 LLM 관련 인자
    p.add_argument('--gpt-translation', action='store_true', default=False)
    p.add_argument('--translation-model', type=str, default='gpt-5.4-mini')
    p.add_argument('--context-window', type=int, default=5)
    p.add_argument('--correction', action='store_true', default=False)
    p.add_argument('--correction-model', type=str, default='gpt-5.4-mini')
    p.add_argument('--api-key', type=str, default=None,
                   help='OpenAI API 키. 미지정 시 OPENAI_API_KEY 환경변수')
    return p


def run(args, *, files, dataset_root, load_audio, score_fn=None, group_key=None,
        running_metric=None, running_metric_label='WER', smoke_lang='auto',
        metric_key='wer'):
    """서버를 준비하고 배치를 돌린 뒤 결과를 저장한다."""
    score_fn = score_fn or calculate_wer

    if args.common_files:
        if not os.path.exists(args.common_files):
            logger.error('Common files JSON not found: %s', args.common_files)
            sys.exit(1)
        common_ids = load_common_file_ids(args.common_files)
        files = [f for f in files if f['file_id'] in common_ids]

    if args.random_sample is not None and args.random_sample < len(files):
        random.seed(args.random_seed)
        ordered = sorted(files, key=lambda x: x['file_id'])
        files = sorted(random.sample(ordered, args.random_sample), key=lambda x: x['file_id'])

    run_dir = resolve_run_dir(args, Path(dataset_root) / 'results')
    logger.info('Results → %s', run_dir)
    if args.description:
        (run_dir / 'description.txt').write_text(args.description, encoding='utf-8')

    server = None
    ws_url = f'ws://{args.host}:{args.port}'
    try:
        if args.auto_server:
            if not os.path.exists(args.server_script):
                logger.error('Server script not found: %s', args.server_script)
                sys.exit(1)
            server = ServerManager(args.server_script, args.host, args.port, args.server_model)
            extra = args.server_args.split() if args.server_args else []
            if args.gpt_translation:
                extra += ['--gpt-translation',
                          '--translation-model', args.translation_model,
                          '--context-window', str(args.context_window)]
            if args.correction:
                extra += ['--correction', '--correction-model', args.correction_model]
            if args.api_key:
                extra += ['--api-key', args.api_key]
            if not server.start_server(extra):
                logger.error('Failed to start Qwen3 server.')
                sys.exit(1)
        else:
            logger.info('Manual mode. Ensure server is running at %s', ws_url)

        if not args.skip_protocol_smoke:
            asyncio.run(run_protocol_smoke(ws_url, lang=smoke_lang))

        server_config = asyncio.run(fetch_server_config(ws_url))
        meta = {
            'timestamp': datetime.now().isoformat(),
            'cli_args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'server_config': server_config,
        }
        with open(run_dir / 'meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results = asyncio.run(
            process_batch(
                files,
                ws_url,
                run_dir,
                args.policy,
                args.limit,
                chunk_size_ms=args.chunk_size_ms,
                send_interval_ms=args.send_interval_ms,
                show_commit_slash=args.show_commit_slash,
                resume=not args.fresh_start,
                target_lang=args.target_lang,
                trailing_silence_ms=args.trailing_silence_ms,
                load_audio=load_audio,
                group_key=group_key,
                running_metric=running_metric,
                running_metric_label=running_metric_label,
                score_fn=score_fn,
                metric_key=metric_key,
            )
        )
        if args.calculate_wer and results:
            score_fn(results, policy=args.policy, emit_summary=True)

        logger.info('Completed. Results saved to %s', run_dir)
        return results
    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
        return []
    finally:
        if server:
            server.stop_server()
