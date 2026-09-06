"""결과 저장 위치와 형식.

    results/{model}/{scope}/{tag}/
    ├── metric.json       overall + raw_results (발화 단위)
    ├── meta.json         실행 당시 CLI 인자
    └── description.txt

`--tag` 를 지정하면 같은 폴더에 이어 저장한다(resume). 지정하지 않으면 run_01, run_02 …
로 자동 증가한다.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

logger = logging.getLogger(__name__)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def load_common_file_ids(common_files_json):
    try:
        with open(common_files_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'raw_results' in data:
            return {r['file_id'] for r in data['raw_results']}
    except Exception as e:
        logger.error('Failed to load common files: %s', e)
    return set()

def load_processed_files(run_dir):
    metric_file = Path(run_dir) / 'metric.json'
    if not metric_file.exists():
        return set()
    try:
        with open(metric_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {r['file_id'] for r in data.get('raw_results', [])}
    except Exception:
        pass
    return set()

def resolve_run_dir(args, default_results_root) -> Path:
    """결과 저장 경로를 정한다: results/{model}/{scope}/{tag}/

    `--tag` 미지정 시 run_01, run_02 … 로 자동 증가한다.
    `default_results_root` 는 데이터셋 디렉토리의 `results/` 로, 어댑터가 넘긴다.
    """
    results_root = Path(getattr(args, 'results_root', None) or default_results_root)
    base = results_root / args.model / args.scope
    if args.tag:
        run_dir = base / args.tag
    else:
        n = 1
        while True:
            run_dir = base / f'run_{n:02d}'
            if not run_dir.exists():
                break
            n += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def build_summary_payload(results, policy, score_fn=None, metric_key='wer'):
    """metric.json 의 overall 블록을 만든다.

    score_fn 은 (results, policy, emit_summary) → (전체값, 화자별 dict) 를 돌려주는
    채점 함수다. 기본은 WER — 한국어 데이터셋은 CER 판(`scoring.calculate_cer`)과
    `metric_key='corpus_cer'` 를 넘긴다. metric_key 는 overall 에 찍히는 이름이라
    이전 결과 파일과 키가 어긋나지 않게 한다.
    """
    if score_fn is None:
        from .scoring import calculate_wer as score_fn
    wer_value, folder_wers = score_fn(results, policy=policy, emit_summary=False)
    by_speaker = {}
    for row in results:
        by_speaker.setdefault(row['speaker_id'], []).append(row)

    def _collect_segment_metric(metric_name, rows):
        values = []
        for row in rows:
            for segment in row.get('segment_metrics') or []:
                value = segment.get(metric_name)
                if value is not None:
                    values.append(value)
        return values

    def _collect_commit_stats(rows):
        counts = {'vad': 0, 'seg': 0, 'dot': 0, 'finish': 0, 'always': 0}
        for row in rows:
            for segment in row.get('segment_metrics') or []:
                reason = segment.get('commit_reason', 'seg')
                counts[reason] = counts.get(reason, 0) + 1
        total = sum(counts.values())
        ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
        return {'counts': counts, 'total': total, 'ratios': ratios}

    folder_stats = {}
    for speaker_id, rows in sorted(by_speaker.items()):
        lat = [r['first_token_latency'] for r in rows if r['first_token_latency'] is not None]
        model_runtime = [r['model_runtime'] for r in rows if r.get('model_runtime') is not None]
        speaker_fsl = _collect_segment_metric('fsl_sec', rows)
        speaker_fsl_norm = _collect_segment_metric('fsl_normalized_sec', rows)
        speaker_encode = _collect_segment_metric('encode_sec', rows)
        speaker_decode = _collect_segment_metric('decode_sec', rows)
        speaker_final_decode = _collect_segment_metric('final_decode_sec', rows)
        speaker_trans = _collect_segment_metric('trans_sec', rows)
        speaker_tokens = _collect_segment_metric('output_token_count', rows)
        speaker_commit = _collect_commit_stats(rows)
        folder_stats[speaker_id] = {
            'num_files': len(rows),
            metric_key: folder_wers.get(speaker_id),
            'first_token_latency': mean(lat) if lat else None,
            'model_runtime': mean(model_runtime) if model_runtime else None,
            'avg_fsl_sec': mean(speaker_fsl) if speaker_fsl else None,
            'avg_fsl_normalized_sec': mean(speaker_fsl_norm) if speaker_fsl_norm else None,
            'avg_encode_sec': mean(speaker_encode) if speaker_encode else None,
            'avg_decode_sec': mean(speaker_decode) if speaker_decode else None,
            'avg_final_decode_sec': mean(speaker_final_decode) if speaker_final_decode else None,
            'avg_trans_sec': mean(speaker_trans) if speaker_trans else None,
            'avg_output_tokens_per_commit': mean(speaker_tokens) if speaker_tokens else None,
            'commit_stats': speaker_commit,
        }

    all_lat = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    all_model_runtime = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
    all_fsl = _collect_segment_metric('fsl_sec', results)
    all_fsl_norm = _collect_segment_metric('fsl_normalized_sec', results)
    all_encode = _collect_segment_metric('encode_sec', results)
    all_decode = _collect_segment_metric('decode_sec', results)
    all_final_decode = _collect_segment_metric('final_decode_sec', results)
    all_trans = _collect_segment_metric('trans_sec', results)
    all_tokens = _collect_segment_metric('output_token_count', results)
    all_commit = _collect_commit_stats(results)

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            metric_key: wer_value,
            'first_token_latency': mean(all_lat) if all_lat else None,
            'model_runtime': mean(all_model_runtime) if all_model_runtime else None,
            'avg_fsl_sec': mean(all_fsl) if all_fsl else None,
            'avg_fsl_normalized_sec': mean(all_fsl_norm) if all_fsl_norm else None,
            'avg_encode_sec': mean(all_encode) if all_encode else None,
            'avg_decode_sec': mean(all_decode) if all_decode else None,
            'avg_final_decode_sec': mean(all_final_decode) if all_final_decode else None,
            'avg_trans_sec': mean(all_trans) if all_trans else None,
            'avg_output_tokens_per_commit': mean(all_tokens) if all_tokens else None,
            'commit_stats': all_commit,
        },
        'folders': folder_stats,
    }

def save_results_structured(results, run_dir, policy, score_fn=None, metric_key='wer'):
    run_dir = Path(run_dir)
    (run_dir / 'plots').mkdir(parents=True, exist_ok=True)

    summary = build_summary_payload(results, policy, score_fn=score_fn, metric_key=metric_key)
    metric_data = {
        'overall': summary.get('overall'),
        'folders': summary.get('folders'),
        'raw_results': results,
    }
    with open(run_dir / 'metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric_data, f, indent=2, ensure_ascii=False)

    logger.info('Results saved → %s', run_dir)

def save_summary_file(results, summary_output_file, policy):
    ensure_parent_dir(summary_output_file)
    payload = build_summary_payload(results, policy)
    with open(summary_output_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
