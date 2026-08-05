#!/usr/bin/env python3
"""
Qwen3 + ReazonSpeech integration test runner.

ReazonSpeech format:
  Audio : ReazonSpeech/audio_NNNNN.wav  (16 kHz, 16-bit mono, short independent clips)
  Labels: label/metadata.csv            (CSV: file_name, text)

Each audio file is an independent short clip from a different Japanese speaker/context,
so a fresh WebSocket connection is opened per file to avoid cross-clip context pollution.

Usage examples:
  # All clips (server must be running)
  python test_qwen3_reazonspeech.py

  # Limit to first 50 clips
  python test_qwen3_reazonspeech.py --limit 50

  # Auto-start server
  python test_qwen3_reazonspeech.py --auto-server --model baseline

  # Fast-push mode (no realtime pacing)
  python test_qwen3_reazonspeech.py --send-interval-ms 0
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import websockets

logging.basicConfig(format='%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
DEFAULT_POLICY = 3
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# --model → server model path
MODEL_MAP = {
    'baseline':         'Qwen/Qwen3-ASR-1.7B',
    'baseline(1.0.0)':  'Qwen/Qwen3-ASR-1.7B',
    'finetuned':         str(PROJECT_ROOT / 'Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged'),
    'finetuned(1.0.1)':  str(PROJECT_ROOT / 'Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged'),
}

DEFAULT_DATA_DIR = SCRIPT_DIR
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / 'evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_map(label_csv: Path) -> dict[str, str]:
    label_map = {}
    with open(label_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('file_name', '').strip()
            text = row.get('text', '').strip()
            if fname and text:
                label_map[fname] = text
    return label_map


def find_audio_files(data_dir: str) -> list[dict]:
    data_dir = Path(data_dir)
    audio_dir = data_dir / 'ReazonSpeech'
    label_csv = data_dir / 'label' / 'metadata.csv'

    if not audio_dir.is_dir():
        logger.error('Audio directory not found: %s', audio_dir)
        return []
    if not label_csv.exists():
        logger.error('Label CSV not found: %s', label_csv)
        return []

    label_map = load_label_map(label_csv)

    audio_files = []
    for fname, text in sorted(label_map.items()):
        audio_path = audio_dir / fname
        if not audio_path.exists():
            logger.debug('Audio missing: %s', audio_path)
            continue
        audio_files.append({
            'file_id': audio_path.stem,   # audio_00000
            'path': str(audio_path),
            'reference': text,
        })

    return sorted(audio_files, key=lambda x: x['file_id'])


def load_audio_file(audio_path: str):
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        return audio
    except Exception as e:
        logger.error('Error loading %s: %s', audio_path, e)
        return None


# ---------------------------------------------------------------------------
# Japanese CER
# ---------------------------------------------------------------------------

def _kata_to_hira(text: str) -> str:
    """Convert full-width katakana (U+30A1-U+30F6) to hiragana (U+3041-U+3096).
    Katakana long-vowel mark (U+30FC) has no hiragana equivalent; drop it."""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        elif cp == 0x30FC:
            pass
        else:
            result.append(ch)
    return ''.join(result)


def _normalize_japanese(text: str) -> str:
    """Normalize Japanese text for CER comparison.

    - Normalize unicode (NFKC): full-width -> half-width, etc.
    - Convert katakana to hiragana so script differences do not inflate CER.
    - Strip Japanese punctuation and non-speech characters.
    - Remove spaces (Japanese has no inter-word spaces).
    """
    text = unicodedata.normalize('NFKC', text)
    text = _kata_to_hira(text)
    text = re.sub(
        r'[^぀-ゟ'
        r'一-鿿'
        r'㐀-䶿'
        r'切-﫿'
        r'a-zA-Z0-9]',
        '',
        text,
    )
    return text


def _levenshtein(ref: str, hyp: str) -> int:
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = _normalize_japanese(reference)
    hyp = _normalize_japanese(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def compute_corpus_cer(rows: list[dict]) -> float | None:
    total_edit, total_ref = 0, 0
    for row in rows:
        ref = _normalize_japanese(row.get('reference', ''))
        hyp = _normalize_japanese(row.get('hypothesis', ''))
        if not ref:
            continue
        total_ref += len(ref)
        total_edit += _levenshtein(ref, hyp)
    return total_edit / total_ref if total_ref > 0 else None


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

class ServerManager:
    def __init__(self, server_script, host='localhost', port=8765, model='Qwen/Qwen3-ASR-1.7B'):
        self.server_script = server_script
        self.host = host
        self.port = port
        self.model = model
        self.process = None

    def start_server(self, additional_args=None):
        if self.process is not None:
            self.stop_server()

        cmd = [
            sys.executable,
            str(self.server_script),
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.model,
            '--no-idle-shutdown',
        ]
        if additional_args:
            cmd.extend(additional_args)

        logger.info('Starting Qwen3 server...')
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not self._wait_for_server_ready(timeout=180):
            self.stop_server()
            return False

        logger.info('Server started (PID: %s)', self.process.pid)
        return True

    def _wait_for_server_ready(self, timeout=180):
        ws_url = f'ws://{self.host}:{self.port}'
        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                logger.error('Server exited unexpectedly while waiting for readiness.')
                return False

            async def _probe():
                try:
                    async with websockets.connect(ws_url, ping_interval=None, open_timeout=3) as ws:
                        msg = await asyncio.wait_for(ws.recv(), timeout=4)
                        if isinstance(msg, str):
                            return json.loads(msg).get('type') == 'hello'
                except Exception:
                    return False
                return False

            if asyncio.run(_probe()):
                return True
            time.sleep(1)
        logger.error('Server readiness timeout reached.')
        return False

    def stop_server(self):
        if self.process is None:
            return
        logger.info('Stopping server (PID: %s)', self.process.pid)
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.process = None


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

async def recv_type(ws, expected_types, timeout=8.0, ignore_types=None):
    if isinstance(expected_types, str):
        expected_types = {expected_types}
    else:
        expected_types = set(expected_types)
    ignore_types = set(ignore_types or [])
    end_at = time.time() + timeout

    while time.time() < end_at:
        msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end_at - time.time()))
        if not isinstance(msg, str):
            continue
        data = json.loads(msg)
        msg_type = data.get('type', '')
        if msg_type in ignore_types:
            continue
        if msg_type in expected_types:
            return data
    raise TimeoutError(f'Expected {sorted(expected_types)} not received in {timeout}s')


async def fetch_server_config(ws_url: str):
    try:
        async with websockets.connect(ws_url, ping_interval=None, open_timeout=10) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=8)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get('type') == 'hello':
                    return data.get('serverConfig')
    except Exception as e:
        logger.warning('Server config fetch failed: %s', e)
    return None


# ---------------------------------------------------------------------------
# Commit / segment helpers
# ---------------------------------------------------------------------------

def normalize_commit_reason(raw_reason: str) -> str:
    reason = str(raw_reason or '').lower()
    if reason.startswith('vad'):
        return 'vad'
    if reason == 'dot':
        return 'dot'
    if reason == 'finish':
        return 'finish'
    if reason == 'always':
        return 'always'
    return 'seg'


def parse_hms_timestamp(value) -> float | None:
    if not value:
        return None
    try:
        h, m, s = value.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def format_commit_markers(segment_events: list[dict]) -> str:
    parts = []
    for event in segment_events:
        text = (event.get('text') or '').strip()
        if not text:
            continue
        tag = normalize_commit_reason(event.get('tag'))
        parts.append(f'{text} <{tag}>')
    return ' '.join(parts).strip()


def merge_trailing_partial(finals, segment_events, partial_last):
    tail = (partial_last or '').strip()
    if not tail:
        return finals, segment_events, False
    if not finals:
        return [tail], [{'text': tail, 'tag': 'seg'}], True

    full_text = ' '.join((t or '').strip() for t in finals).strip()
    last = (finals[-1] or '').strip()
    if not last:
        finals[-1] = tail
        segment_events[-1] = {'text': tail, 'tag': 'seg'}
        return finals, segment_events, True

    if tail == last:
        return finals, segment_events, False

    lt, ll, lf = tail.lower(), last.lower(), full_text.lower()
    if lt == lf:
        return finals, segment_events, False
    if lt.startswith(lf):
        extra = tail[len(full_text):].strip()
        if not extra:
            return finals, segment_events, False
        finals.append(extra)
        segment_events.append({'text': extra, 'tag': 'seg'})
        return finals, segment_events, True
    if lt.startswith(ll):
        finals[-1] = tail
        if segment_events:
            segment_events[-1]['text'] = tail
        return finals, segment_events, True
    if lf.endswith(lt) or ll.endswith(lt):
        return finals, segment_events, False

    finals.append(tail)
    segment_events.append({'text': tail, 'tag': 'seg'})
    return finals, segment_events, True


def summarize_segment_metrics(segment_metrics: list[dict]) -> dict:
    if not segment_metrics:
        return {}

    def _vals(key):
        return [s[key] for s in segment_metrics if s.get(key) is not None]

    return {
        'num_segments': len(segment_metrics),
        'avg_server_fsl_sec': mean(_vals('server_fsl_sec')) if _vals('server_fsl_sec') else None,
        'avg_server_fsl_normalized_sec': mean(_vals('server_fsl_normalized_sec')) if _vals('server_fsl_normalized_sec') else None,
        'avg_asr_inference_sec': mean(_vals('asr_inference_sec')) if _vals('asr_inference_sec') else None,
        'avg_trans_sec': mean(_vals('trans_sec')) if _vals('trans_sec') else None,
    }


# ---------------------------------------------------------------------------
# Single-file streaming
# ---------------------------------------------------------------------------

async def process_single_file(
    ws,
    audio_data,
    chunk_size_ms=200,
    send_interval_ms=200,
    target_lang='ko',
    trailing_silence_ms=3000,
    src_lang='ja',
):
    processing_start = time.perf_counter()

    start_msg = {'type': 'start', 'lang': src_lang, 'targetLang': target_lang}
    await ws.send(json.dumps(start_msg))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final'})

    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    send_interval_sec = max(0.0, send_interval_ms / 1000.0)

    finals = []
    segment_events = []
    segment_metrics = []
    partial_last = ''
    first_result_time = None
    last_result_time = None
    send_done = asyncio.Event()

    async def _send():
        stream_origin = time.perf_counter()
        try:
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                if send_interval_sec > 0:
                    target_send_at = stream_origin + (i + len(chunk)) / SAMPLING_RATE
                    while True:
                        remaining = target_send_at - time.perf_counter()
                        if remaining <= 0:
                            break
                        await asyncio.sleep(min(remaining, 0.02))
                await asyncio.wait_for(ws.send(chunk.tobytes()), timeout=30.0)

            if trailing_silence_ms > 0:
                silence = np.zeros(int(SAMPLING_RATE * trailing_silence_ms / 1000), dtype=np.int16)
                silence_origin = time.perf_counter()
                for i in range(0, len(silence), chunk_size):
                    chunk = silence[i:i + chunk_size]
                    if send_interval_sec > 0:
                        target_send_at = silence_origin + (i + len(chunk)) / SAMPLING_RATE
                        while True:
                            remaining = target_send_at - time.perf_counter()
                            if remaining <= 0:
                                break
                            await asyncio.sleep(min(remaining, 0.02))
                    await asyncio.wait_for(ws.send(chunk.tobytes()), timeout=30.0)

            await asyncio.wait_for(ws.send(json.dumps({'type': 'finish'})), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning('_send: ws.send() timed out, aborting send')
            return
        send_done.set()

    async def _recv():
        nonlocal partial_last, first_result_time, last_result_time
        audio_duration = len(audio_data) / SAMPLING_RATE
        absolute_deadline = processing_start + audio_duration + 180
        post_send_idle = None

        while time.perf_counter() < absolute_deadline:
            if post_send_idle is not None and time.perf_counter() > post_send_idle:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if send_done.is_set():
                    if post_send_idle is None:
                        post_send_idle = time.perf_counter() + 15
                continue

            if not isinstance(msg, str):
                continue

            data = json.loads(msg)
            msg_type = data.get('type', '')

            if msg_type == 'final':
                if first_result_time is None:
                    first_result_time = time.perf_counter()
                last_result_time = time.perf_counter()
                text = (data.get('original') or '').strip()
                if text:
                    partial_last = ''
                    receive_elapsed = time.perf_counter() - processing_start
                    audio_start = data.get('audioStartSec') or parse_hms_timestamp(data.get('start'))
                    audio_end = data.get('audioEndSec') or parse_hms_timestamp(data.get('end'))

                    finals.append(text)
                    commit_reason = normalize_commit_reason(
                        data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                    )
                    segment_events.append({'text': text, 'tag': commit_reason})

                    fsl_sec = data.get('fsl_sec')
                    server_fsl_norm = (fsl_sec + 0.8) if fsl_sec is not None and commit_reason == 'vad' else fsl_sec
                    encode_sec = data.get('encode_sec') or 0.0
                    decode_sec = data.get('decode_sec') or 0.0
                    final_decode_sec = data.get('final_decode_sec') or 0.0
                    asr_inference_sec = (
                        round(encode_sec + decode_sec + final_decode_sec, 4)
                        if any(data.get(k) is not None for k in ('encode_sec', 'decode_sec', 'final_decode_sec'))
                        else None
                    )
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': commit_reason,
                        'audio_start_sec': audio_start,
                        'audio_end_sec': audio_end,
                        'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                        'server_fsl_sec': fsl_sec,
                        'server_fsl_normalized_sec': server_fsl_norm,
                        'asr_inference_sec': asr_inference_sec,
                        'trans_sec': data.get('trans_sec'),
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'client_final_received_elapsed_sec': receive_elapsed,
                    })
                    if send_done.is_set():
                        post_send_idle = time.perf_counter() + 10

            elif msg_type == 'partial':
                text = (data.get('original') or '').strip()
                if text:
                    partial_last = text
                    if send_done.is_set():
                        post_send_idle = time.perf_counter() + 5

    audio_duration_sec = len(audio_data) / SAMPLING_RATE
    total_timeout = audio_duration_sec + trailing_silence_ms / 1000.0 + 120.0

    send_task = asyncio.create_task(_send())
    recv_task = asyncio.create_task(_recv())
    done, pending = await asyncio.wait({send_task, recv_task}, timeout=total_timeout)
    if pending:
        logger.warning(
            'process_single_file: hard timeout after %.0fs (audio=%.1fs), cancelling',
            total_timeout, audio_duration_sec,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    for t in done:
        if not t.cancelled():
            exc = t.exception()
            if exc:
                logger.warning('process_single_file: task exception: %s', exc)

    if not finals and partial_last:
        finals = [partial_last]
        segment_events = [{'text': partial_last, 'tag': 'seg'}]
    else:
        finals, segment_events, partial_used = merge_trailing_partial(finals, segment_events, partial_last)
        if partial_used:
            segment_metrics.append({
                'segment_id': None,
                'text': partial_last.strip(),
                'translation': '',
                'commit_reason': 'partial_tail',
                'audio_start_sec': None,
                'audio_end_sec': None,
                'seg_audio_sec': None,
                'server_fsl_sec': None,
                'server_fsl_normalized_sec': None,
                'asr_inference_sec': None,
                'trans_sec': None,
                'encode_sec': None,
                'decode_sec': None,
                'final_decode_sec': None,
                'client_final_received_elapsed_sec': None,
            })

    total_time = (
        (last_result_time - processing_start) if last_result_time
        else (time.perf_counter() - processing_start)
    )
    first_token_latency = (first_result_time - processing_start) if first_result_time else None

    return {
        'transcript': ' '.join(finals).strip(),
        'finals': finals,
        'segment_events': segment_events,
        'segment_metrics': segment_metrics,
        'segment_metrics_summary': summarize_segment_metrics(segment_metrics),
        'total_time': total_time,
        'first_token_latency': first_token_latency,
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_processed_files(run_dir: Path) -> set[str]:
    metric_file = run_dir / 'metric.json'
    if not metric_file.exists():
        return set()
    try:
        with open(metric_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {r['file_id'] for r in data.get('raw_results', [])}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def process_batch(
    audio_files: list[dict],
    ws_url: str,
    run_dir: Path,
    policy: int,
    limit=None,
    chunk_size_ms=200,
    send_interval_ms=200,
    show_commit_slash=True,
    resume=True,
    target_lang='ko',
    src_lang='ja',
    trailing_silence_ms=3000,
) -> list[dict]:
    run_dir = Path(run_dir)
    processed_ids = load_processed_files(run_dir) if resume else set()
    targets = [f for f in audio_files if f['file_id'] not in processed_ids]
    if limit is not None:
        targets = targets[:limit]

    if not targets:
        logger.info('No files to process.')
        return []

    all_results = []
    if resume and processed_ids:
        metric_file = run_dir / 'metric.json'
        if metric_file.exists():
            try:
                with open(metric_file, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                all_results = old.get('raw_results', [])
            except Exception:
                pass

    logger.info('Processing %d clip(s) (already done: %d)', len(targets), len(all_results))
    results = []

    for idx, audio_info in enumerate(targets, start=1):
        file_id = audio_info['file_id']
        logger.info('[%d/%d] %s', idx, len(targets), file_id)

        audio = load_audio_file(audio_info['path'])
        if audio is None:
            continue

        duration = len(audio) / SAMPLING_RATE

        try:
            async with websockets.connect(
                ws_url, ping_interval=None, ping_timeout=None,
                open_timeout=30, max_size=10 * 1024 * 1024,
            ) as ws:
                await recv_type(ws, 'hello', timeout=8)
                out = await process_single_file(
                    ws,
                    audio,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                    target_lang=target_lang,
                    trailing_silence_ms=trailing_silence_ms,
                    src_lang=src_lang,
                )
        except Exception as e:
            logger.error('WebSocket processing failed for %s: %s', file_id, e)
            continue

        if not out['transcript']:
            logger.warning('Empty transcript: %s', file_id)
            continue

        model_runtime = out['total_time'] - duration
        cer = compute_cer(audio_info['reference'], out['transcript'])

        row = {
            'file_id': file_id,
            'audio_path': audio_info['path'],
            'reference': audio_info['reference'],
            'hypothesis': out['transcript'],
            'hyp_commit': format_commit_markers(out.get('segment_events') or []),
            'cer': cer,
            'duration': duration,
            'total_time': out['total_time'],
            'first_token_latency': out['first_token_latency'],
            'model_runtime': model_runtime,
            'target_lang': target_lang,
            'src_lang': src_lang,
            'segment_metrics': out.get('segment_metrics') or [],
            'segment_metrics_summary': out.get('segment_metrics_summary') or {},
        }
        results.append(row)
        save_results_structured(all_results + results, run_dir, policy)

        logger.info('  REF: %s', audio_info['reference'])
        logger.info('  HYP: %s', out['transcript'])
        logger.info('  CER: %.2f%% | DUR: %.2fs | FTL: %s | RUNTIME: %.3fs',
                    cer * 100,
                    duration,
                    f"{out['first_token_latency']:.3f}s" if out['first_token_latency'] is not None else 'N/A',
                    model_runtime)
        if show_commit_slash and out.get('segment_events'):
            logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))
        seg_summary = out.get('segment_metrics_summary') or {}
        if seg_summary.get('avg_server_fsl_sec') is not None:
            logger.info('  FSL(avg): %.3fs', seg_summary['avg_server_fsl_sec'])

    return all_results + results


# ---------------------------------------------------------------------------
# Summary / save
# ---------------------------------------------------------------------------

def _collect_segment_metric(metric_name: str, rows: list[dict]) -> list:
    values = []
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            v = seg.get(metric_name)
            if v is not None:
                values.append(v)
    return values


def _collect_commit_stats(rows: list[dict]) -> dict:
    counts = {'vad': 0, 'seg': 0, 'dot': 0, 'finish': 0, 'always': 0}
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            reason = seg.get('commit_reason', 'seg')
            counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return {'counts': counts, 'total': total, 'ratios': ratios}


def build_summary_payload(results: list[dict], policy: int) -> dict:
    corpus_cer = compute_corpus_cer(results)
    lats = [r['first_token_latency'] for r in results if r.get('first_token_latency') is not None]
    runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
    durations = [r['duration'] for r in results if r.get('duration') is not None]

    def _safe_mean(key):
        vals = _collect_segment_metric(key, results)
        return mean(vals) if vals else None

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            'corpus_cer': corpus_cer,
            'avg_cer': mean([r['cer'] for r in results if r.get('cer') is not None]) if results else None,
            'avg_duration': mean(durations) if durations else None,
            'avg_first_token_latency': mean(lats) if lats else None,
            'avg_model_runtime': mean(runtimes) if runtimes else None,
            'avg_server_fsl_sec': _safe_mean('server_fsl_sec'),
            'avg_server_fsl_normalized_sec': _safe_mean('server_fsl_normalized_sec'),
            'avg_asr_inference_sec': _safe_mean('asr_inference_sec'),
            'avg_trans_sec': _safe_mean('trans_sec'),
            'commit_stats': _collect_commit_stats(results),
        },
    }


def save_results_structured(results: list[dict], run_dir: Path, policy: int):
    run_dir = Path(run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    summary = build_summary_payload(results, policy)
    metric_data = {
        'overall': summary.get('overall'),
        'raw_results': results,
    }
    with open(run_dir / 'metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric_data, f, indent=2, ensure_ascii=False)
    logger.info('Results saved → %s', run_dir)


def resolve_run_dir(args) -> Path:
    results_root = SCRIPT_DIR / 'results'
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


def print_summary(results: list[dict], policy: int | None = None):
    corpus_cer = compute_corpus_cer(results)
    lats = [r['first_token_latency'] for r in results if r.get('first_token_latency') is not None]
    runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]

    policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
    logger.info('\n%s', '=' * 70)
    logger.info('RESULTS SUMMARY%s', policy_label)
    logger.info('%s', '=' * 70)
    logger.info('Total clips processed : %d', len(results))
    logger.info('Corpus CER            : %s',
                f'{corpus_cer * 100:.2f}%' if corpus_cer is not None else 'N/A')
    logger.info('Avg First Token Latency: %.3fs', mean(lats) if lats else 0.0)
    logger.info('Avg Model Runtime      : %.3fs', mean(runtimes) if runtimes else 0.0)
    logger.info('%s\n', '=' * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Qwen3 ReazonSpeech integration test (Japanese, CER)')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Root of ReazonSpeech data (contains ReazonSpeech/ and label/)')
    parser.add_argument('--src-lang', type=str, default='ja',
                        help='ASR source language code sent to server (default: ja)')
    parser.add_argument('--target-lang', type=str, default='ko',
                        help='Translation target language (default: ko)')
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    # ─── 결과 폴더 구조 ────────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='finetuned',
                        help='모델 대분류 (예: baseline, finetuned, baseline(1.0.0))')
    parser.add_argument('--scope', type=str, default='sample', choices=['full', 'sample'],
                        help='테스트 범위 (full=전체, sample=일부)')
    parser.add_argument('--tag', type=str, default=None,
                        help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    parser.add_argument('--description', type=str, default=None,
                        help='테스트 설명 (description.txt에 저장)')
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument('--limit', type=int, default=None,
                        help='처리할 최대 클립 수 (미지정 시 전체)')
    parser.add_argument('--chunk-size-ms', type=int, default=200)
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='실시간 페이싱 간격 (ms). 0이면 최대 속도')
    parser.add_argument('--trailing-silence-ms', type=int, default=3000,
                        help='VAD 트리거용 후행 묵음 길이 (ms, 기본값: 3000)')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-start', action='store_true', default=False,
                        help='기존 결과 무시하고 처음부터 재실행')
    parser.add_argument('--calculate-cer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default=None,
                        help='서버 모델 경로. 미지정 시 --model로 MODEL_MAP 자동 추론')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--gpt-translation', action='store_true', default=False)
    parser.add_argument('--translation-model', type=str, default='gpt-5.4-mini')
    parser.add_argument('--context-window', type=int, default=5)
    parser.add_argument('--correction', action='store_true', default=False)
    parser.add_argument('--correction-model', type=str, default='gpt-5.4-mini')
    parser.add_argument('--api-key', type=str, default=None)

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error('Data directory not found: %s', data_dir)
        sys.exit(1)

    files = find_audio_files(str(data_dir))
    if not files:
        logger.error('No ReazonSpeech audio files found in %s', data_dir)
        sys.exit(1)

    logger.info('Found %d clip(s) in %s', len(files), data_dir)

    run_dir = resolve_run_dir(args)
    logger.info('Results → %s', run_dir)

    meta = {
        'timestamp': datetime.now().isoformat(),
        'cli_args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with open(run_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

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

        # hello 메시지에서 serverConfig 수집
        server_config = asyncio.run(fetch_server_config(ws_url))
        if server_config:
            with open(run_dir / 'meta.json', 'r', encoding='utf-8') as f:
                meta = json.load(f)
            meta['server_config'] = server_config
            with open(run_dir / 'meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        results = asyncio.run(
            process_batch(
                files,
                ws_url,
                run_dir,
                args.policy,
                limit=args.limit,
                chunk_size_ms=args.chunk_size_ms,
                send_interval_ms=args.send_interval_ms,
                show_commit_slash=args.show_commit_slash,
                resume=not args.fresh_start,
                target_lang=args.target_lang,
                src_lang=args.src_lang,
                trailing_silence_ms=args.trailing_silence_ms,
            )
        )

        if args.calculate_cer and results:
            print_summary(results, policy=args.policy)

        logger.info('Completed. Results saved to %s', run_dir)

    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
    finally:
        if server:
            server.stop_server()


if __name__ == '__main__':
    main()
