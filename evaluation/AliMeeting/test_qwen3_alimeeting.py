#!/usr/bin/env python3
"""
Qwen3 + AliMeeting Corpus integration test runner.

AliMeeting dataset format:
  Audio : AliMeeting/{meeting_id}.wav            (mixed multi-speaker, 16 kHz)
  Labels: label/{meeting_id}_N_SPK{id}.TextGrid  (Praat TextGrid, per-speaker)

Ground truth is built by merging all speaker TextGrid intervals sorted by start time.
Uses CER (Character Error Rate) as the primary metric for Chinese.

Usage examples:
  # All meetings (realtime streaming)
  python test_qwen3_alimeeting.py

  # Fast-push mode (no realtime pacing)
  python test_qwen3_alimeeting.py --send-interval-ms 0

  # Specific meetings only
  python test_qwen3_alimeeting.py --meetings R8001_M8004 R8003_M8001
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
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
DEFAULT_SERVER_SCRIPT = (
    SCRIPT_DIR.parent / 'LibriSpeech' / 'servers' / 'streaming_websocket_server_fsl.py'
)

# --model 값 → 실제 서버 모델 경로 자동 매핑
MODEL_MAP = {
    "baseline":        "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)": "Qwen/Qwen3-ASR-1.7B",
    "finetuned":        str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
    "finetuned(1.0.1)": str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
}


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

_INTERVAL_RE = re.compile(
    r'intervals\s*\[\d+\]:\s*\n\s*xmin\s*=\s*([\d.eE+\-]+)\s*\n\s*xmax\s*=\s*([\d.eE+\-]+)\s*\n\s*text\s*=\s*"([^"]*)"',
)


def parse_textgrid(textgrid_path: Path) -> list[tuple[float, float, str]]:
    """Parse a Praat TextGrid file.

    Returns a list of (xmin_sec, xmax_sec, text) tuples for non-empty intervals,
    sorted by start time.
    """
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(textgrid_path, 'r', encoding='utf-16') as f:
            content = f.read()
    except Exception as e:
        logger.warning('Failed to read %s: %s', textgrid_path, e)
        return []

    intervals = []
    for m in _INTERVAL_RE.finditer(content):
        text = m.group(3).strip()
        if not text:
            continue
        try:
            xmin = float(m.group(1))
            xmax = float(m.group(2))
        except ValueError:
            continue
        intervals.append((xmin, xmax, text))

    intervals.sort(key=lambda x: x[0])
    return intervals


def build_reference(meeting_id: str, label_dir: Path) -> str:
    """Build combined reference transcript for one meeting.

    Merges all per-speaker TextGrid files by start time and joins text.
    """
    pattern = f'{meeting_id}_N_SPK*.TextGrid'
    textgrid_files = sorted(label_dir.glob(pattern))
    if not textgrid_files:
        return ''

    all_intervals: list[tuple[float, float, str]] = []
    for tg_path in textgrid_files:
        all_intervals.extend(parse_textgrid(tg_path))

    if not all_intervals:
        return ''

    all_intervals.sort(key=lambda x: x[0])
    return ' '.join(text for _, _, text in all_intervals)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_sessions(data_dir: Path) -> list[dict]:
    """Find all (wav, label) pairs and build reference text."""
    audio_dir = data_dir / 'AliMeeting'
    label_dir = data_dir / 'label'

    if not audio_dir.is_dir():
        logger.error('Audio directory not found: %s', audio_dir)
        return []
    if not label_dir.is_dir():
        logger.error('Label directory not found: %s', label_dir)
        return []

    sessions = []
    for wav_path in sorted(audio_dir.glob('*.wav')):
        meeting_id = wav_path.stem
        reference = build_reference(meeting_id, label_dir)
        if not reference:
            logger.warning('No TextGrid labels for %s — skipping', meeting_id)
            continue

        sessions.append({
            'meeting_id': meeting_id,
            'audio_path': str(wav_path),
            'reference': reference,
        })

    return sessions


# ---------------------------------------------------------------------------
# CER computation (Chinese character-level)
# ---------------------------------------------------------------------------

_ZH_PUNCT_RE = re.compile(
    r'[。，！？、：；「」『』（）【】《》〈〉……—～·　＀-￯]'
)
_ASCII_PUNCT_RE = re.compile(r'[^\w\s一-鿿㐀-䶿豈-﫿]')


def _normalize_zh(text: str) -> str:
    """Normalize Chinese text for CER: remove punctuation and spaces."""
    text = text.lower()
    text = _ZH_PUNCT_RE.sub('', text)
    text = _ASCII_PUNCT_RE.sub('', text)
    # Remove all spaces — Chinese CER operates at character level without spaces
    return re.sub(r'\s+', '', text)


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
    ref = _normalize_zh(reference)
    hyp = _normalize_zh(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def compute_corpus_cer(rows: list[dict]) -> float | None:
    total_edit, total_ref = 0, 0
    for row in rows:
        ref = _normalize_zh(row.get('reference', ''))
        hyp = _normalize_zh(row.get('hypothesis', ''))
        if not ref:
            continue
        total_ref += len(ref)
        total_edit += _levenshtein(ref, hyp)
    return total_edit / total_ref if total_ref > 0 else None


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_file(audio_path: str) -> np.ndarray | None:
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
            self.server_script,
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
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


# ---------------------------------------------------------------------------
# Commit / segment helpers
# ---------------------------------------------------------------------------

def normalize_commit_reason(raw_reason):
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


def parse_hms_timestamp(value):
    if not value:
        return None
    try:
        hours, minutes, seconds = value.split(':')
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    except Exception:
        return None


def format_commit_markers(segment_events):
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

    full_final_text = ''.join((t or '').strip() for t in finals)
    last_final = (finals[-1] or '').strip()
    if not last_final:
        finals[-1] = tail
        segment_events[-1] = {'text': tail, 'tag': 'seg'}
        return finals, segment_events, True

    if tail == last_final:
        return finals, segment_events, False

    # Chinese: compare without spaces
    norm_tail = _normalize_zh(tail)
    norm_full = _normalize_zh(full_final_text)
    norm_last = _normalize_zh(last_final)

    if norm_tail == norm_full:
        return finals, segment_events, False
    if norm_full.endswith(norm_tail) or norm_last.endswith(norm_tail):
        return finals, segment_events, False
    if norm_tail.startswith(norm_last):
        finals[-1] = tail
        if segment_events:
            segment_events[-1]['text'] = tail
        return finals, segment_events, True

    finals.append(tail)
    segment_events.append({'text': tail, 'tag': 'seg'})
    return finals, segment_events, True


# ---------------------------------------------------------------------------
# Segment metric helpers
# ---------------------------------------------------------------------------

def summarize_segment_metrics(segment_metrics):
    if not segment_metrics:
        return {}

    def _vals(key):
        return [s[key] for s in segment_metrics if s.get(key) is not None]

    return {
        'num_segments': len(segment_metrics),
        'avg_fsl_sec': mean(_vals('fsl_sec')) if _vals('fsl_sec') else None,
        'avg_fsl_normalized_sec': mean(_vals('fsl_normalized_sec')) if _vals('fsl_normalized_sec') else None,
        'avg_encode_sec': mean(_vals('encode_sec')) if _vals('encode_sec') else None,
        'avg_decode_sec': mean(_vals('decode_sec')) if _vals('decode_sec') else None,
        'avg_final_decode_sec': mean(_vals('final_decode_sec')) if _vals('final_decode_sec') else None,
        'avg_trans_sec': mean(_vals('trans_sec')) if _vals('trans_sec') else None,
    }


# ---------------------------------------------------------------------------
# Single-file streaming
# ---------------------------------------------------------------------------

async def process_single_file(ws, audio_data, chunk_size_ms=200, send_interval_ms=200,
                               src_lang='zh', target_lang='ko', trailing_silence_ms=3000):
    processing_start = time.perf_counter()

    await ws.send(json.dumps({'type': 'start', 'lang': src_lang, 'targetLang': target_lang}))
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
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            chunk_end_sec = (i + len(chunk)) / SAMPLING_RATE

            if send_interval_sec > 0:
                target_send_at = stream_origin + chunk_end_sec
                while True:
                    remaining = target_send_at - time.perf_counter()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(remaining, 0.02))

            await ws.send(chunk.tobytes())

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
                await ws.send(chunk.tobytes())

        await ws.send(json.dumps({'type': 'finish'}))
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
                    receive_elapsed_sec = time.perf_counter() - processing_start
                    audio_start_sec = data.get('audioStartSec') or parse_hms_timestamp(data.get('start'))
                    audio_end_sec = data.get('audioEndSec') or parse_hms_timestamp(data.get('end'))

                    finals.append(text)
                    segment_events.append({
                        'text': text,
                        'tag': normalize_commit_reason(
                            data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                        ),
                    })
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': normalize_commit_reason(
                            data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                        ),
                        'output_token_count': len(text),
                        'audio_start_sec': audio_start_sec,
                        'audio_end_sec': audio_end_sec,
                        'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                        'fsl_sec': data.get('fsl_sec'),
                        'fsl_normalized_sec': (
                            (data.get('fsl_sec') + 0.8)
                            if data.get('fsl_sec') is not None
                            and normalize_commit_reason(
                                data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                            ) == 'vad'
                            else data.get('fsl_sec')
                        ),
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'trans_sec': data.get('trans_sec'),
                        'chunk_encode_log': data.get('chunk_encode_log'),
                        'slotAudioStartSec': data.get('slotAudioStartSec'),
                        'vad_trigger_sec': data.get('vad_trigger_sec'),
                        'client_final_received_elapsed_sec': receive_elapsed_sec,
                    })
                    if send_done.is_set():
                        post_send_idle = time.perf_counter() + 10

            elif msg_type == 'partial':
                text = (data.get('original') or '').strip()
                if text:
                    partial_last = text
                    if send_done.is_set():
                        post_send_idle = time.perf_counter() + 5

            elif msg_type == 'ready':
                pass

    await asyncio.gather(_send(), _recv())

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
                'output_token_count': len(partial_last.strip()),
                'audio_start_sec': None,
                'audio_end_sec': None,
                'seg_audio_sec': None,
                'fsl_sec': None,
                'encode_sec': None,
                'decode_sec': None,
                'final_decode_sec': None,
                'trans_sec': None,
                'chunk_encode_log': None,
                'slotAudioStartSec': None,
                'vad_trigger_sec': None,
                'client_final_received_elapsed_sec': None,
            })

    total_time = (last_result_time - processing_start) if last_result_time else (time.perf_counter() - processing_start)
    first_token_latency = (first_result_time - processing_start) if first_result_time else None

    return {
        'transcript': ' '.join(finals).strip(),
        'segments': finals,
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
        return {r['meeting_id'] for r in data.get('raw_results', [])}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def process_batch(
    sessions: list[dict],
    ws_url: str,
    run_dir: Path,
    policy: int,
    limit: int | None = None,
    chunk_size_ms: int = 200,
    send_interval_ms: int = 200,
    show_commit_slash: bool = True,
    resume: bool = True,
    src_lang: str = 'zh',
    target_lang: str = 'ko',
    trailing_silence_ms: int = 3000,
) -> list[dict]:
    run_dir = Path(run_dir)
    processed_ids = load_processed_files(run_dir) if resume else set()

    targets = [s for s in sessions if s['meeting_id'] not in processed_ids]
    if limit is not None:
        targets = targets[:limit]

    if not targets:
        logger.info('No sessions to process.')
        return []

    all_results: list[dict] = []
    if resume and processed_ids:
        metric_file = run_dir / 'metric.json'
        if metric_file.exists():
            try:
                with open(metric_file, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                all_results = old.get('raw_results', [])
            except Exception:
                pass

    logger.info('Processing %d session(s) (already done: %d)', len(targets), len(all_results))
    results: list[dict] = []

    for idx, session in enumerate(targets, start=1):
        meeting_id = session['meeting_id']
        logger.info('[%d/%d] %s', idx, len(targets), meeting_id)

        audio = load_audio_file(session['audio_path'])
        if audio is None:
            continue

        duration = len(audio) / SAMPLING_RATE
        logger.info('  Duration: %.1fs (%.1f min)', duration, duration / 60)

        try:
            async with websockets.connect(
                ws_url, ping_interval=None, ping_timeout=None, max_size=10 * 1024 * 1024
            ) as ws:
                await recv_type(ws, 'hello', timeout=8)
                out = await process_single_file(
                    ws,
                    audio,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                    src_lang=src_lang,
                    target_lang=target_lang,
                    trailing_silence_ms=trailing_silence_ms,
                )
        except Exception as e:
            logger.error('WebSocket processing failed for %s: %s', meeting_id, e)
            continue

        if not out['transcript']:
            logger.warning('Empty transcript: %s', meeting_id)
            continue

        model_runtime = out['total_time'] - duration
        cer = compute_cer(session['reference'], out['transcript'])

        row = {
            'meeting_id': meeting_id,
            'audio_path': session['audio_path'],
            'reference': session['reference'],
            'hypothesis': out['transcript'],
            'hyp_commit': format_commit_markers(out.get('segment_events') or []),
            'cer': cer,
            'duration': duration,
            'total_time': out['total_time'],
            'first_token_latency': out['first_token_latency'],
            'model_runtime': model_runtime,
            'src_lang': src_lang,
            'target_lang': target_lang,
            'segment_metrics': out.get('segment_metrics') or [],
            'segment_metrics_summary': out.get('segment_metrics_summary') or {},
        }
        results.append(row)
        save_results_structured(all_results + results, run_dir, policy)

        logger.info('  REF: %s', session['reference'][:120])
        logger.info('  HYP: %s', out['transcript'][:120])
        logger.info('  CER: %.2f%%  segments=%d  audio=%.1fs  runtime=%.3fs',
                    cer * 100, len(out['segments']), duration, model_runtime)
        logger.info('  FIRST_TOKEN_LATENCY: %s',
                    f"{out['first_token_latency']:.3f}s" if out['first_token_latency'] is not None else 'N/A')
        logger.info('  MODEL_RUNTIME(total-audio): %.3fs', model_runtime)
        seg_summary = out.get('segment_metrics_summary') or {}
        if seg_summary:
            logger.info('  FSL(avg): %s',
                        f"{seg_summary['avg_fsl_sec']:.3f}s" if seg_summary.get('avg_fsl_sec') is not None else 'N/A')
            logger.info('  ENCODE(avg): %s  DECODE(avg): %s  FINAL_DECODE(avg): %s',
                        f"{seg_summary['avg_encode_sec']:.3f}s" if seg_summary.get('avg_encode_sec') is not None else 'N/A',
                        f"{seg_summary['avg_decode_sec']:.3f}s" if seg_summary.get('avg_decode_sec') is not None else 'N/A',
                        f"{seg_summary['avg_final_decode_sec']:.3f}s" if seg_summary.get('avg_final_decode_sec') is not None else 'N/A')
            logger.info('  TRANS(avg): %s',
                        f"{seg_summary['avg_trans_sec']:.3f}s" if seg_summary.get('avg_trans_sec') is not None else 'N/A')
        if show_commit_slash and out.get('segments'):
            logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    return all_results + results


# ---------------------------------------------------------------------------
# Summary / save
# ---------------------------------------------------------------------------

def _collect_segment_metric(metric_name, rows):
    values = []
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            v = seg.get(metric_name)
            if v is not None:
                values.append(v)
    return values


def _collect_commit_stats(rows):
    counts = {'vad': 0, 'seg': 0, 'dot': 0, 'finish': 0, 'always': 0}
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            reason = seg.get('commit_reason', 'seg')
            counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return {'counts': counts, 'total': total, 'ratios': ratios}


def build_summary_payload(results, policy):
    corpus_cer = compute_corpus_cer(results)
    by_meeting = {}
    for row in results:
        by_meeting.setdefault(row['meeting_id'], []).append(row)

    def _safe_mean(key, r):
        vals = _collect_segment_metric(key, r)
        return mean(vals) if vals else None

    meeting_stats = {}
    for meeting_id, rows in sorted(by_meeting.items()):
        lat = [r['first_token_latency'] for r in rows if r['first_token_latency'] is not None]
        mr = [r['model_runtime'] for r in rows if r.get('model_runtime') is not None]
        meeting_cer = compute_corpus_cer(rows)
        meeting_stats[meeting_id] = {
            'num_files': len(rows),
            'cer': meeting_cer,
            'first_token_latency': mean(lat) if lat else None,
            'model_runtime': mean(mr) if mr else None,
            'avg_fsl_sec': _safe_mean('fsl_sec', rows),
            'avg_fsl_normalized_sec': _safe_mean('fsl_normalized_sec', rows),
            'avg_encode_sec': _safe_mean('encode_sec', rows),
            'avg_decode_sec': _safe_mean('decode_sec', rows),
            'avg_final_decode_sec': _safe_mean('final_decode_sec', rows),
            'avg_trans_sec': _safe_mean('trans_sec', rows),
            'avg_output_tokens_per_commit': _safe_mean('output_token_count', rows),
            'commit_stats': _collect_commit_stats(rows),
        }

    all_lat = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    all_mr = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            'cer': corpus_cer,
            'first_token_latency': mean(all_lat) if all_lat else None,
            'model_runtime': mean(all_mr) if all_mr else None,
            'avg_fsl_sec': _safe_mean('fsl_sec', results),
            'avg_fsl_normalized_sec': _safe_mean('fsl_normalized_sec', results),
            'avg_encode_sec': _safe_mean('encode_sec', results),
            'avg_decode_sec': _safe_mean('decode_sec', results),
            'avg_final_decode_sec': _safe_mean('final_decode_sec', results),
            'avg_trans_sec': _safe_mean('trans_sec', results),
            'avg_output_tokens_per_commit': _safe_mean('output_token_count', results),
            'commit_stats': _collect_commit_stats(results),
        },
        'meetings': meeting_stats,
    }


def save_results_structured(results, run_dir, policy):
    run_dir = Path(run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    summary = build_summary_payload(results, policy)
    metric_data = {
        'overall': summary.get('overall'),
        'meetings': summary.get('meetings'),
        'raw_results': results,
    }
    with open(run_dir / 'metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric_data, f, indent=2, ensure_ascii=False)

    logger.info('Results saved → %s', run_dir)


def resolve_run_dir(args) -> Path:
    """
    결과 저장 경로를 결정합니다.
      results/{model}/{scope}/{tag}/
    --tag 미지정 시 run_01, run_02 ... 순으로 자동 생성.
    """
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


def calculate_cer(results, policy=None, emit_summary=True):
    corpus_cer = compute_corpus_cer(results)
    if corpus_cer is None:
        logger.warning('No valid rows for CER.')
        return None, {}

    meeting_cers = {}
    by_meeting = {}
    for r in results:
        by_meeting.setdefault(r['meeting_id'], []).append(r)
    for meeting_id, rows in by_meeting.items():
        meeting_cers[meeting_id] = compute_corpus_cer(rows)

    if emit_summary:
        ftl = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        mr = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY — AliMeeting%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total sessions processed : %d', len(results))
        logger.info('Corpus CER               : %.2f%%', corpus_cer * 100)
        if ftl:
            logger.info('Avg first token latency  : %.3fs', mean(ftl))
        if mr:
            logger.info('Avg model runtime        : %.3fs', mean(mr))
        logger.info('Meetings: %d', len(by_meeting))
        for meeting_id, cer in sorted(meeting_cers.items()):
            logger.info('  %s CER: %s', meeting_id, f'{cer * 100:.2f}%' if cer is not None else 'N/A')
        logger.info('%s\n', '=' * 70)

    return corpus_cer, meeting_cers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Qwen3 AliMeeting Chinese ASR evaluation (CER)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, default=str(SCRIPT_DIR),
                        help='Dataset root (contains AliMeeting/ and label/ subdirs)')
    parser.add_argument('--meetings', type=str, nargs='+', default=None,
                        metavar='MEETING',
                        help='Meeting IDs to test (default: all found in AliMeeting/)')
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    # ─── 결과 폴더 구조 인자 ────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='finetuned',
                        help='대분류: 모델 종류 (예: baseline, finetuned)')
    parser.add_argument('--scope', type=str, default='sample', choices=['full', 'sample'],
                        help='소분류: 테스트 범위 (full=전체 데이터셋, sample=일부)')
    parser.add_argument('--tag', type=str, default=None,
                        help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    parser.add_argument('--description', type=str, default=None,
                        help='테스트 설명 (description.txt에 저장)')
    # ────────────────────────────────────────────────────────────────────────
    parser.add_argument('--limit', type=int, default=None,
                        help='Max number of meetings to process')
    parser.add_argument('--calculate-cer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default=None,
                        help='서버에 로드할 모델 경로. 미지정 시 --model 값으로 MODEL_MAP에서 자동 추론')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--src-lang', type=str, default='zh',
                        help='ASR source language code sent in start message (default: zh)')
    parser.add_argument('--target-lang', type=str, default='ko',
                        help='Translation target language code (default: ko)')
    parser.add_argument('--chunk-size-ms', type=int, default=200)
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='Realtime pacing (ms per chunk). Set 0 to push as fast as possible.')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-start', action='store_true', default=False)
    parser.add_argument('--trailing-silence-ms', type=int, default=3000)

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error('Data directory not found: %s', data_dir)
        sys.exit(1)

    sessions = find_sessions(data_dir)
    if not sessions:
        logger.error('No sessions found in %s', data_dir)
        sys.exit(1)

    if args.meetings:
        sessions = [s for s in sessions if s['meeting_id'] in args.meetings]
        if not sessions:
            logger.error('No matching meetings found for: %s', args.meetings)
            sys.exit(1)

    logger.info('Found %d session(s):', len(sessions))
    for s in sessions:
        ref_chars = len(_normalize_zh(s['reference']))
        logger.info('  %s  ref_chars=%d', s['meeting_id'], ref_chars)

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
            if not server.start_server(extra):
                logger.error('Failed to start Qwen3 server.')
                sys.exit(1)
        else:
            logger.info('Manual mode. Ensure server is running at %s', ws_url)

        results = asyncio.run(
            process_batch(
                sessions,
                ws_url,
                run_dir,
                args.policy,
                args.limit,
                chunk_size_ms=args.chunk_size_ms,
                send_interval_ms=args.send_interval_ms,
                show_commit_slash=args.show_commit_slash,
                resume=not args.fresh_start,
                src_lang=args.src_lang,
                target_lang=args.target_lang,
                trailing_silence_ms=args.trailing_silence_ms,
            )
        )

        if args.calculate_cer and results:
            calculate_cer(results, policy=args.policy, emit_summary=True)

        logger.info('Completed. Results saved to %s', run_dir)
    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
    finally:
        if server:
            server.stop_server()


if __name__ == '__main__':
    main()
