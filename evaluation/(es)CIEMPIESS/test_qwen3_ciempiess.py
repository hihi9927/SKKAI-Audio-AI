#!/usr/bin/env python3
"""
Qwen3 + (es)CIEMPIESS integration test runner.

CIEMPIESS dataset format:
  Audio : CIEMPIESS/{train,read,fm,description}/{ID}.wav  (16 kHz, 16-bit mono)
  Labels: label/CIEMPIESS_test.fileids         (one 'test/ciempiess/ID' per line)
          label/CIEMPIESS_test.transcription   ('<s> text </s> (ID)' per line)

  Transcription uses uppercase vowels to mark lexical stress (e.g. 'cOn respEcto A').
  These are lowercased during normalization.

Files are grouped by session (train: parts 2+3 of underscore-split ID; fm: prefix before
first underscore; description: first underscore-segment; read: all under 'OSC').
One WebSocket connection is reused per group to maintain context continuity.

Uses WER (Word Error Rate) as the primary metric.

Usage examples:
  # All files (server must be running)
  python test_qwen3_ciempiess.py

  # Limit to first 100 files
  python test_qwen3_ciempiess.py --limit 100

  # Auto-start server
  python test_qwen3_ciempiess.py --auto-server --model baseline

  # Fast-push mode (no realtime pacing)
  python test_qwen3_ciempiess.py --send-interval-ms 0

  # Only 'read' subset
  python test_qwen3_ciempiess.py --subsets read
"""

import argparse
import asyncio
import json
import logging
import os
import random
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
    PROJECT_ROOT / 'LibriSpeech' / 'servers' / 'streaming_websocket_server_fsl.py'
)

# Subdirectory names for audio lookup
AUDIO_SUBDIRS = ['train', 'read', 'fm', 'description']

MODEL_MAP = {
    'baseline':         'Qwen/Qwen3-ASR-1.7B',
    'baseline(1.0.0)':  'Qwen/Qwen3-ASR-1.7B',
    'finetuned':         str(PROJECT_ROOT / 'Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-es-merged'),
    'finetuned(1.0.1)':  str(PROJECT_ROOT / 'Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-es-merged'),
}


# ---------------------------------------------------------------------------
# WER (Spanish, word-level)
# ---------------------------------------------------------------------------

_ES_PUNCT_RE = re.compile(r'[^\w\s]')


def _normalize_es(text: str) -> str:
    """Lowercase, strip stress markers, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = _ES_PUNCT_RE.sub(' ', text)
    return ' '.join(text.split())


def _levenshtein_words(ref: list[str], hyp: list[str]) -> int:
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


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_es(reference).split()
    hyp_words = _normalize_es(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein_words(ref_words, hyp_words) / len(ref_words)


def compute_wer_for_rows(rows: list[dict]) -> float | None:
    try:
        import jiwer
    except ImportError:
        return None

    pairs = [
        (_normalize_es(r['reference']), _normalize_es(r['hypothesis']))
        for r in rows
        if r.get('reference') and r.get('hypothesis')
    ]
    pairs = [(ref, hyp) for ref, hyp in pairs if ref.strip() and hyp.strip()]
    if not pairs:
        return None
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_transcription_file(trans_path: Path) -> dict[str, str]:
    """Parse CIEMPIESS_test.transcription → {file_id: text}."""
    label_map: dict[str, str] = {}
    pattern = re.compile(r'^<s>\s*(.*?)\s*</s>\s*\((\w+)\)\s*$')
    with open(trans_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                text, file_id = m.group(1), m.group(2)
                label_map[file_id] = text
    return label_map


def _get_group_id(file_id: str, subset: str) -> str:
    """Return a group/session key for context-continuity grouping.

    - train: '0026M_09ALX_22OCT12' → group by session '09ALX_22OCT12'
    - read:  'OSC_001'             → group 'OSC'
    - fm:    'AB01_1'              → group 'AB01'
    - description: 'M16ABR1237_0001' → group 'M16ABR1237'
    """
    parts = file_id.split('_')
    if subset == 'train' and len(parts) >= 3:
        return '_'.join(parts[1:])
    return parts[0]


def find_audio_files(
    data_dir: Path,
    subsets: list[str] | None = None,
) -> list[dict]:
    """Find WAV files and pair them with reference transcripts.

    Returns a list of dicts with keys:
      file_id, subset, group_id, path, reference
    sorted by (group_id, file_id).
    """
    ciempiess_dir = data_dir / 'CIEMPIESS'
    label_dir = data_dir / 'label'
    fileids_path = label_dir / 'CIEMPIESS_test.fileids'
    trans_path   = label_dir / 'CIEMPIESS_test.transcription'

    if not fileids_path.exists():
        logger.error('fileids not found: %s', fileids_path)
        return []
    if not trans_path.exists():
        logger.error('transcription file not found: %s', trans_path)
        return []

    # Parse labels
    label_map = _parse_transcription_file(trans_path)
    target_ids: list[str] = []
    with open(fileids_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                target_ids.append(line.split('/')[-1])  # strip 'test/ciempiess/' prefix

    # Build lookup: file_id → (path, subset)
    wav_map: dict[str, tuple[Path, str]] = {}
    check_subdirs = subsets if subsets else AUDIO_SUBDIRS
    for sub in check_subdirs:
        sub_dir = ciempiess_dir / sub
        if not sub_dir.is_dir():
            continue
        for wav in sub_dir.glob('*.wav'):
            wav_map[wav.stem] = (wav, sub)

    audio_files = []
    skipped_missing = 0
    skipped_no_label = 0
    for file_id in target_ids:
        if file_id not in wav_map:
            skipped_missing += 1
            continue
        if file_id not in label_map:
            skipped_no_label += 1
            continue
        wav_path, subset = wav_map[file_id]
        audio_files.append({
            'file_id':  file_id,
            'subset':   subset,
            'group_id': _get_group_id(file_id, subset),
            'path':     str(wav_path),
            'reference': label_map[file_id],
        })

    if skipped_missing:
        if subsets:
            logger.debug('Skipped %d files: WAV not in selected subsets %s', skipped_missing, subsets)
        else:
            logger.warning('Skipped %d files: WAV not found', skipped_missing)
    if skipped_no_label:
        logger.warning('Skipped %d files: no reference label', skipped_no_label)

    return sorted(audio_files, key=lambda x: (x['group_id'], x['file_id']))


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
        self._log_fh = None

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

        log_path = f'/tmp/ciempiess_server_{int(time.time())}.log'
        self._log_fh = open(log_path, 'w')
        logger.info('Starting Qwen3 server... (log: %s)', log_path)
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=self._log_fh,
            preexec_fn=os.setsid,
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
                logger.error('Server exited unexpectedly.')
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
        logger.error('Server readiness timeout.')
        return False

    def stop_server(self):
        if self.process is None:
            return
        logger.info('Stopping server (PID: %s)', self.process.pid)
        try:
            import signal as _signal
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, _signal.SIGTERM)
            except (ProcessLookupError, OSError):
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, _signal.SIGKILL)
                except Exception:
                    self.process.kill()
                self.process.wait(timeout=5)
        finally:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None
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
# Commit / segment helpers
# ---------------------------------------------------------------------------

def normalize_commit_reason(raw_reason) -> str:
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
        hours, minutes, seconds = value.split(':')
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
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

    full_final_text = ' '.join((t or '').strip() for t in finals).strip()
    last_final = (finals[-1] or '').strip()
    if not last_final:
        finals[-1] = tail
        segment_events[-1] = {'text': tail, 'tag': 'seg'}
        return finals, segment_events, True

    lower_tail, lower_last = tail.lower(), last_final.lower()
    lower_full = full_final_text.lower()
    if tail == last_final:
        return finals, segment_events, False
    if lower_tail == lower_full:
        return finals, segment_events, False
    if lower_tail.startswith(lower_full):
        extra = tail[len(full_final_text):].strip()
        if not extra:
            return finals, segment_events, False
        finals.append(extra)
        segment_events.append({'text': extra, 'tag': 'seg'})
        return finals, segment_events, True
    if lower_tail.startswith(lower_last):
        finals[-1] = tail
        if segment_events:
            segment_events[-1]['text'] = tail
        return finals, segment_events, True
    if lower_full.endswith(lower_tail) or lower_last.endswith(lower_tail):
        return finals, segment_events, False

    finals.append(tail)
    segment_events.append({'text': tail, 'tag': 'seg'})
    return finals, segment_events, True


# ---------------------------------------------------------------------------
# Single-file streaming
# ---------------------------------------------------------------------------

async def process_single_file(
    ws,
    audio_data: np.ndarray,
    chunk_size_ms: int = 200,
    send_interval_ms: int = 200,
    target_lang: str = 'ko',
    trailing_silence_ms: int = 3000,
) -> dict:
    processing_start = time.perf_counter()

    await ws.send(json.dumps({'type': 'start', 'lang': 'es', 'targetLang': target_lang}))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final'})

    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    send_interval_sec = max(0.0, send_interval_ms / 1000.0)

    finals: list[str] = []
    segment_events: list[dict] = []
    segment_metrics: list[dict] = []
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
                    audio_start_sec = data.get('audioStartSec') or parse_hms_timestamp(data.get('start'))
                    audio_end_sec = data.get('audioEndSec') or parse_hms_timestamp(data.get('end'))
                    fsl_sec = data.get('fsl_sec')
                    commit_reason = normalize_commit_reason(
                        data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                    )
                    encode_sec = data.get('encode_sec') or 0.0
                    decode_sec = data.get('decode_sec') or 0.0
                    final_decode_sec = data.get('final_decode_sec') or 0.0
                    asr_inference_sec = (
                        round(encode_sec + decode_sec + final_decode_sec, 4)
                        if data.get('encode_sec') is not None
                        or data.get('decode_sec') is not None
                        or data.get('final_decode_sec') is not None
                        else None
                    )
                    trans_sec = data.get('trans_sec')
                    fsl_normalized_sec = (
                        (fsl_sec + 0.8) if fsl_sec is not None and commit_reason == 'vad' else fsl_sec
                    )
                    commit_delay_sec = (
                        round(max(0.0, fsl_sec - trans_sec), 4)
                        if fsl_sec is not None and trans_sec is not None
                        else None
                    )

                    finals.append(text)
                    segment_events.append({'text': text, 'tag': commit_reason})
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': commit_reason,
                        'audio_start_sec': audio_start_sec,
                        'audio_end_sec': audio_end_sec,
                        'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                        'server_fsl_sec': fsl_sec,
                        'server_fsl_normalized_sec': fsl_normalized_sec,
                        'commit_delay_sec': commit_delay_sec,
                        'asr_inference_sec': asr_inference_sec,
                        'translation_latency_sec': trans_sec,
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'slotAudioStartSec': data.get('slotAudioStartSec'),
                        'vad_trigger_sec': data.get('vad_trigger_sec'),
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
    total_timeout = audio_duration_sec + trailing_silence_ms / 1000.0 + 300.0

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
                'commit_delay_sec': None,
                'asr_inference_sec': None,
                'translation_latency_sec': None,
                'encode_sec': None,
                'decode_sec': None,
                'final_decode_sec': None,
                'slotAudioStartSec': None,
                'vad_trigger_sec': None,
                'client_final_received_elapsed_sec': None,
            })

    total_time = (
        (last_result_time - processing_start) if last_result_time
        else (time.perf_counter() - processing_start)
    )
    first_token_latency = (first_result_time - processing_start) if first_result_time else None

    return {
        'transcript': ' '.join(finals).strip(),
        'segments': finals,
        'segment_events': segment_events,
        'segment_metrics': segment_metrics,
        'segment_metrics_summary': _summarize_segment_metrics(segment_metrics),
        'total_time': total_time,
        'first_token_latency': first_token_latency,
    }


def _summarize_segment_metrics(segment_metrics: list[dict]) -> dict:
    if not segment_metrics:
        return {}

    def _vals(key):
        return [s[key] for s in segment_metrics if s.get(key) is not None]

    return {
        'num_segments': len(segment_metrics),
        'avg_server_fsl_sec':            mean(_vals('server_fsl_sec'))            if _vals('server_fsl_sec')            else None,
        'avg_server_fsl_normalized_sec': mean(_vals('server_fsl_normalized_sec')) if _vals('server_fsl_normalized_sec') else None,
        'avg_commit_delay_sec':          mean(_vals('commit_delay_sec'))          if _vals('commit_delay_sec')          else None,
        'avg_translation_latency_sec':   mean(_vals('translation_latency_sec'))   if _vals('translation_latency_sec')   else None,
        'avg_asr_inference_sec':         mean(_vals('asr_inference_sec'))         if _vals('asr_inference_sec')         else None,
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
    limit: int | None = None,
    chunk_size_ms: int = 200,
    send_interval_ms: int = 200,
    show_commit_slash: bool = True,
    resume: bool = True,
    target_lang: str = 'ko',
    trailing_silence_ms: int = 3000,
) -> list[dict]:
    run_dir = Path(run_dir)
    processed_ids = load_processed_files(run_dir) if resume else set()
    targets = [f for f in audio_files if f['file_id'] not in processed_ids]
    if limit is not None:
        targets = targets[:limit]

    if not targets:
        logger.info('No files to process.')
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

    logger.info('Processing %d file(s) (already done: %d)', len(targets), len(all_results))

    results: list[dict] = []
    current_group: str | None = None
    ws = None

    async def _open_ws():
        nonlocal ws
        ws = await websockets.connect(
            ws_url, ping_interval=None, ping_timeout=None,
            open_timeout=30, max_size=10 * 1024 * 1024,
        )
        await recv_type(ws, 'hello', timeout=8)

    async def _close_ws():
        nonlocal ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({'type': 'stop'}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        ws = None

    try:
        for idx, audio_info in enumerate(targets, start=1):
            file_id  = audio_info['file_id']
            group_id = audio_info['group_id']
            subset   = audio_info['subset']
            logger.info('[%d/%d] %s  (group=%s, subset=%s)', idx, len(targets), file_id, group_id, subset)

            audio = load_audio_file(audio_info['path'])
            if audio is None:
                continue

            duration = len(audio) / SAMPLING_RATE

            # Rotate connection on group change to reset server context
            if group_id != current_group:
                await _close_ws()
                current_group = group_id
                try:
                    await _open_ws()
                except Exception as e:
                    logger.error('WebSocket connect failed for group %s: %s', group_id, e)
                    ws = None
                    continue

            if ws is None:
                try:
                    await _open_ws()
                except Exception as e:
                    logger.error('WebSocket connect failed for %s: %s', file_id, e)
                    continue

            try:
                out = await process_single_file(
                    ws,
                    audio,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                    target_lang=target_lang,
                    trailing_silence_ms=trailing_silence_ms,
                )
                # Reset server state between files within the same group
                await ws.send(json.dumps({'type': 'finish'}))
            except Exception as e:
                logger.error('WebSocket processing failed for %s: %s', file_id, e)
                await _close_ws()
                continue

            if not out['transcript']:
                logger.warning('Empty transcript: %s', file_id)
                continue

            model_runtime = out['total_time'] - duration
            wer = compute_wer(audio_info['reference'], out['transcript'])

            row = {
                'file_id':  file_id,
                'group_id': group_id,
                'subset':   subset,
                'audio_path': audio_info['path'],
                'reference': audio_info['reference'],
                'hypothesis': out['transcript'],
                'hyp_commit': format_commit_markers(out.get('segment_events') or []),
                'duration': duration,
                'total_time': out['total_time'],
                'first_token_latency': out['first_token_latency'],
                'model_runtime': model_runtime,
                'wer': wer,
                'target_lang': target_lang,
                'segment_metrics': out.get('segment_metrics') or [],
                'segment_metrics_summary': out.get('segment_metrics_summary') or {},
            }
            results.append(row)
            save_results_structured(all_results + results, run_dir, policy)

            logger.info('  REF: %s', audio_info['reference'])
            logger.info('  HYP: %s', out['transcript'])
            logger.info('  WER: %.2f%%', wer * 100)
            logger.info('  FIRST_TOKEN_LATENCY: %s',
                        f"{out['first_token_latency']:.3f}s" if out['first_token_latency'] is not None else 'N/A')
            logger.info('  MODEL_RUNTIME(total-audio): %.3fs', model_runtime)
            seg_summary = out.get('segment_metrics_summary') or {}
            if seg_summary:
                logger.info('  FSL(avg server): %s',
                            f"{seg_summary['avg_server_fsl_sec']:.3f}s" if seg_summary.get('avg_server_fsl_sec') is not None else 'N/A')
            if show_commit_slash and out.get('segments'):
                logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    finally:
        await _close_ws()

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
    counts: dict[str, int] = {'vad': 0, 'seg': 0, 'dot': 0, 'finish': 0, 'always': 0}
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            reason = seg.get('commit_reason', 'seg')
            counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return {'counts': counts, 'total': total, 'ratios': ratios}


def build_summary_payload(results: list[dict], policy: int) -> dict:
    wer_value, subset_wers = calculate_wer(results, policy=policy, emit_summary=False)
    by_subset: dict[str, list] = {}
    for row in results:
        by_subset.setdefault(row['subset'], []).append(row)

    def _safe_mean(key, rows):
        vals = _collect_segment_metric(key, rows)
        return mean(vals) if vals else None

    subset_stats: dict[str, dict] = {}
    for subset_name, rows in sorted(by_subset.items()):
        lat = [r['first_token_latency'] for r in rows if r['first_token_latency'] is not None]
        mr  = [r['model_runtime'] for r in rows if r.get('model_runtime') is not None]
        subset_stats[subset_name] = {
            'num_files': len(rows),
            'wer': subset_wers.get(subset_name),
            'first_token_latency': mean(lat) if lat else None,
            'model_runtime': mean(mr) if mr else None,
            'avg_server_fsl_sec':            _safe_mean('server_fsl_sec', rows),
            'avg_server_fsl_normalized_sec': _safe_mean('server_fsl_normalized_sec', rows),
            'avg_commit_delay_sec':          _safe_mean('commit_delay_sec', rows),
            'avg_translation_latency_sec':   _safe_mean('translation_latency_sec', rows),
            'avg_asr_inference_sec':         _safe_mean('asr_inference_sec', rows),
            'commit_stats': _collect_commit_stats(rows),
        }

    all_lat = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    all_mr  = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            'wer': wer_value,
            'first_token_latency': mean(all_lat) if all_lat else None,
            'model_runtime': mean(all_mr) if all_mr else None,
            'avg_server_fsl_sec':            _safe_mean('server_fsl_sec', results),
            'avg_server_fsl_normalized_sec': _safe_mean('server_fsl_normalized_sec', results),
            'avg_commit_delay_sec':          _safe_mean('commit_delay_sec', results),
            'avg_translation_latency_sec':   _safe_mean('translation_latency_sec', results),
            'avg_asr_inference_sec':         _safe_mean('asr_inference_sec', results),
            'commit_stats': _collect_commit_stats(results),
        },
        'subsets': subset_stats,
    }


def save_results_structured(results: list[dict], run_dir: Path, policy: int):
    run_dir = Path(run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    summary = build_summary_payload(results, policy)
    metric_data = {
        'overall':    summary.get('overall'),
        'subsets':    summary.get('subsets'),
        'raw_results': results,
    }
    with open(run_dir / 'metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric_data, f, indent=2, ensure_ascii=False)

    logger.info('Results saved → %s', run_dir)


def resolve_run_dir(args) -> Path:
    """결과 저장 경로: results/{model}/{scope}/{tag}/"""
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


def calculate_wer(results: list[dict], policy=None, emit_summary=True):
    try:
        import jiwer  # noqa: F401
    except ImportError:
        logger.warning('jiwer not installed. Install with: pip install jiwer')
        return None, {}

    overall_wer = compute_wer_for_rows(results)
    if overall_wer is None:
        logger.warning('No valid rows for WER.')
        return None, {}

    by_subset: dict[str, list] = {}
    for r in results:
        by_subset.setdefault(r['subset'], []).append(r)
    subset_wers = {s: compute_wer_for_rows(rows) for s, rows in by_subset.items()}

    if emit_summary:
        ftl = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        mr  = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %d', len(results))
        logger.info('Overall WER: %.2f%%', overall_wer * 100)
        logger.info('Average First Token Latency: %.3fs', sum(ftl) / len(ftl) if ftl else 0.0)
        logger.info('Average Model Runtime: %.3fs', sum(mr) / len(mr) if mr else 0.0)
        for subset_name, wer in sorted(subset_wers.items()):
            cnt = len(by_subset[subset_name])
            logger.info('  [%s] WER: %s  (%d files)',
                        subset_name,
                        f'{wer * 100:.2f}%' if wer is not None else 'N/A',
                        cnt)
        logger.info('%s\n', '=' * 70)

    return overall_wer, subset_wers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Qwen3 (es)CIEMPIESS integration test — Spanish ASR (WER)'
    )
    parser.add_argument('--data-dir', type=str, default=str(SCRIPT_DIR),
                        help='Root directory (contains CIEMPIESS/ and label/). Default: script dir')
    parser.add_argument('--subsets', type=str, nargs='+',
                        default=None, choices=AUDIO_SUBDIRS,
                        metavar='SUBSET',
                        help='Audio subsets to include (default: all — train read fm description)')
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    # ─── 결과 폴더 구조 인자 ───────────────────────────────────────────
    parser.add_argument('--model', type=str, default='baseline',
                        help='대분류: 모델 종류 (예: baseline, finetuned)')
    parser.add_argument('--scope', type=str, default='sample',
                        help='소분류: 테스트 범위 (full=전체, sample=일부)')
    parser.add_argument('--tag', type=str, default=None,
                        help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    parser.add_argument('--description', type=str, default=None,
                        help='테스트 설명 (description.txt에 저장)')
    # ──────────────────────────────────────────────────────────────────
    parser.add_argument('--limit', type=int, default=None,
                        help='처리할 최대 파일 수 (미지정 시 전체)')
    parser.add_argument('--random-sample', type=int, default=None,
                        help='전체 파일 중 무작위 N개 샘플링')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--calculate-wer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default=None,
                        help='서버에 로드할 모델 경로. 미지정 시 --model로 MODEL_MAP에서 자동 추론')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--target-lang', type=str, default='ko')
    parser.add_argument('--chunk-size-ms', type=int, default=200)
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='청크 전송 간격 (ms). 0이면 최대 속도')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-start', action='store_true', default=False)
    parser.add_argument('--trailing-silence-ms', type=int, default=3000,
                        help='오디오 뒤에 붙일 묵음 길이 (VAD 트리거용, 기본값: 3000)')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error('Data directory not found: %s', data_dir)
        sys.exit(1)

    files = find_audio_files(data_dir, subsets=args.subsets)
    if not files:
        logger.error('No audio files found in %s', data_dir)
        sys.exit(1)

    if args.random_sample is not None and args.random_sample < len(files):
        random.seed(args.random_seed)
        files = sorted(
            random.sample(files, args.random_sample),
            key=lambda x: (x['group_id'], x['file_id']),
        )

    subset_counts: dict[str, int] = {}
    for f in files:
        subset_counts[f['subset']] = subset_counts.get(f['subset'], 0) + 1
    logger.info('Found %d file(s): %s', len(files),
                ', '.join(f'{s}={c}' for s, c in sorted(subset_counts.items())))

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
            )
        )

        if args.calculate_wer and results:
            calculate_wer(results, policy=args.policy, emit_summary=True)

        logger.info('Completed. Results saved to %s', run_dir)
    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
    finally:
        if server:
            server.stop_server()


if __name__ == '__main__':
    main()
