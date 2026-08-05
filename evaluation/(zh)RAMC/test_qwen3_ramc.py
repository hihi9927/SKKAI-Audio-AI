#!/usr/bin/env python3
"""
Qwen3 + RAMC (Real-world mobile ASR corpus) integration test runner.

RAMC dataset format:
  Audio : RAMC/{SpeakerID}/{UtteranceID}.wav  (16 kHz, mono, short utterances)
  Labels: label/TRANS.txt                     (TSV: UtteranceID, SpeakerID, Transcription)

Uses CER (Character Error Rate) as the primary metric for Chinese.
Groups utterances by SpeakerID — one WebSocket connection per speaker for context continuity.

Usage examples:
  # All speakers (realtime streaming)
  python test_qwen3_ramc.py --data-dir evaluation/(zh)RAMC

  # Limit to 200 files (sample run)
  python test_qwen3_ramc.py --data-dir evaluation/(zh)RAMC --scope sample --limit 200

  # Fast-push mode (no realtime pacing)
  python test_qwen3_ramc.py --data-dir evaluation/(zh)RAMC --send-interval-ms 0

  # Specific speakers only
  python test_qwen3_ramc.py --data-dir evaluation/(zh)RAMC --speakers 37_5622 5_2197
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

# --model 값 → 실제 서버 모델 경로 자동 매핑
MODEL_MAP = {
    "baseline":         "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)":  "Qwen/Qwen3-ASR-1.7B",
    "finetuned":         str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
    "finetuned(1.0.1)":  str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
    "finetuned(1.0.3)":  str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
    "finetuned_silence":        str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
    "finetuned_silence(1.0.3)": str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-zh-merged"),
}


# ---------------------------------------------------------------------------
# CER computation (Chinese character-level)
# ---------------------------------------------------------------------------

_ZH_PUNCT_RE = re.compile(
    r'[。，！？、：；「」『』（）【】《》〈〉……—～·　＀-￯]'
)
_ASCII_PUNCT_RE = re.compile(r'[^\w\s一-鿿㐀-䶿豈-﫿]')


def _normalize_zh(text: str) -> str:
    text = text.lower()
    text = _ZH_PUNCT_RE.sub('', text)
    text = _ASCII_PUNCT_RE.sub('', text)
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


def compute_corpus_cer(rows: list) -> float | None:
    total_edit, total_ref = 0, 0
    for row in rows:
        ref = _normalize_zh(row.get('reference', ''))
        hyp = _normalize_zh(row.get('hypothesis', ''))
        if not ref:
            continue
        total_ref += len(ref)
        total_edit += _levenshtein(ref, hyp)
    return total_edit / total_ref if total_ref > 0 else None


def compute_cer_for_rows(rows: list) -> float | None:
    return compute_corpus_cer(rows)


# ---------------------------------------------------------------------------
# RAMC data loading
# ---------------------------------------------------------------------------

def parse_trans_file(trans_path: Path) -> dict:
    """Parse TRANS.txt → {file_stem: (speaker_id, transcription)}."""
    trans_map = {}
    with open(trans_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('UtteranceID'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            utterance_id, speaker_id, transcription = parts[0], parts[1], parts[2]
            file_stem = Path(utterance_id).stem
            trans_map[file_stem] = (speaker_id, transcription)
    return trans_map


def find_audio_files(data_dir: Path, speakers: list | None = None) -> list:
    """Find WAV files and pair with reference transcripts from TRANS.txt."""
    audio_dir = data_dir / 'RAMC'
    label_path = data_dir / 'label' / 'TRANS.txt'

    if not audio_dir.is_dir():
        logger.error('Audio directory not found: %s', audio_dir)
        return []
    if not label_path.exists():
        logger.error('Label file not found: %s', label_path)
        return []

    trans_map = parse_trans_file(label_path)
    logger.info('Loaded %d transcriptions from %s', len(trans_map), label_path)

    audio_files = []
    for wav_path in sorted(audio_dir.rglob('*.wav')):
        file_stem = wav_path.stem
        if file_stem not in trans_map:
            continue
        speaker_id, transcription = trans_map[file_stem]
        if speakers and speaker_id not in speakers:
            continue
        audio_files.append({
            'file_id': file_stem,
            'speaker_id': speaker_id,
            'path': str(wav_path),
            'reference': transcription,
        })

    audio_files.sort(key=lambda x: (x['speaker_id'], x['file_id']))
    return audio_files


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
            self.server_script,
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.model,
            '--no-idle-shutdown',
        ]
        if additional_args:
            cmd.extend(additional_args)

        log_path = f'/tmp/ramc_server_{int(time.time())}.log'
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


async def fetch_server_config(ws_url):
    try:
        async with websockets.connect(ws_url, ping_interval=None, open_timeout=10) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=8)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get('type') == 'hello':
                    return data.get('serverConfig')
    except Exception as e:
        logger.warning('서버 config 수집 실패: %s', e)
    return None


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
                               target_lang='ko', trailing_silence_ms=3000):
    processing_start = time.perf_counter()

    await ws.send(json.dumps({'type': 'start', 'lang': 'zh', 'targetLang': target_lang}))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final'})

    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    send_interval_sec = max(0.0, send_interval_ms / 1000.0)

    finals = []
    segment_events = []
    segment_metrics = []
    first_result_time = None
    last_result_time = None
    send_done = asyncio.Event()
    vad_fired = asyncio.Event()
    real_audio_done = asyncio.Event()

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

        real_audio_done.set()

        if trailing_silence_ms > 0:
            silence = np.zeros(int(SAMPLING_RATE * trailing_silence_ms / 1000), dtype=np.int16)
            silence_origin = time.perf_counter()
            for i in range(0, len(silence), chunk_size):
                if vad_fired.is_set():
                    break
                chunk = silence[i:i + chunk_size]
                if send_interval_sec > 0:
                    target_send_at = silence_origin + (i + len(chunk)) / SAMPLING_RATE
                    while True:
                        remaining = target_send_at - time.perf_counter()
                        if remaining <= 0:
                            break
                        await asyncio.sleep(min(remaining, 0.02))
                if vad_fired.is_set():
                    break
                await ws.send(chunk.tobytes())

        send_done.set()

    async def _recv():
        nonlocal first_result_time, last_result_time
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if send_done.is_set():
                    break
                continue
            except Exception:
                break

            if not isinstance(msg, str):
                continue

            data = json.loads(msg)
            msg_type = data.get('type', '')

            if msg_type == 'vad_done':
                if real_audio_done.is_set():
                    if not data.get('has_remaining', True):
                        vad_fired.set()
                        break
                    continue
                continue

            elif msg_type == 'final':
                if first_result_time is None:
                    first_result_time = time.perf_counter()
                last_result_time = time.perf_counter()
                text = (data.get('original') or '').strip()
                if text:
                    receive_elapsed_sec = time.perf_counter() - processing_start
                    audio_start_sec = data.get('audioStartSec') or parse_hms_timestamp(data.get('start'))
                    audio_end_sec = data.get('audioEndSec') or parse_hms_timestamp(data.get('end'))
                    fsl_sec = data.get('fsl_sec')
                    commit_reason_norm = normalize_commit_reason(
                        data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                    )
                    finals.append(text)
                    segment_events.append({'text': text, 'tag': commit_reason_norm})
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': commit_reason_norm,
                        'audio_start_sec': audio_start_sec,
                        'audio_end_sec': audio_end_sec,
                        'fsl_sec': fsl_sec,
                        'fsl_normalized_sec': (
                            (fsl_sec + 0.8)
                            if fsl_sec is not None and commit_reason_norm == 'vad'
                            else fsl_sec
                        ),
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'trans_sec': data.get('trans_sec'),
                        'output_token_count': len(text),
                        'slotAudioStartSec': data.get('slotAudioStartSec'),
                        'vad_trigger_sec': data.get('vad_trigger_sec'),
                        'client_final_received_elapsed_sec': receive_elapsed_sec,
                    })

    await asyncio.gather(_send(), _recv())

    total_time = (last_result_time - processing_start) if last_result_time else (time.perf_counter() - processing_start)
    first_token_latency = (first_result_time - processing_start) if first_result_time else None

    return {
        'transcript': ''.join(finals).strip(),
        'segments': finals,
        'segment_events': segment_events,
        'segment_metrics': segment_metrics,
        'segment_metrics_summary': summarize_segment_metrics(segment_metrics),
        'total_time': total_time,
        'first_token_latency': first_token_latency,
    }


# ---------------------------------------------------------------------------
# Concatenated speaker streaming
# ---------------------------------------------------------------------------

async def process_speaker_concat(
    ws,
    audio_files_for_speaker: list,
    chunk_size_ms: int = 200,
    send_interval_ms: int = 200,
    target_lang: str = 'ko',
    inter_utterance_silence_ms: int = 800,
    post_finish_timeout: float = 15.0,
):
    """화자의 모든 발화를 이어붙여 단일 스트림으로 처리."""
    processing_start = time.perf_counter()

    await ws.send(json.dumps({'type': 'start', 'lang': 'zh', 'targetLang': target_lang}))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final'})

    # 오디오 이어붙이기 (발화 사이 묵음 삽입)
    silence_samples = int(SAMPLING_RATE * inter_utterance_silence_ms / 1000)
    silence_block = np.zeros(silence_samples, dtype=np.float32)

    segments_audio: list[np.ndarray] = []
    for info in audio_files_for_speaker:
        audio = load_audio_file(info['path'])
        if audio is not None:
            segments_audio.append(audio)
            segments_audio.append(silence_block)

    if not segments_audio:
        return None

    concat_audio = np.concatenate(segments_audio)
    total_duration = len(concat_audio) / SAMPLING_RATE

    audio_int16 = (np.clip(concat_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    send_interval_sec = max(0.0, send_interval_ms / 1000.0)

    finals = []
    segment_events = []
    segment_metrics = []
    first_result_time = None
    last_result_time = None
    send_done = asyncio.Event()

    async def _send():
        stream_origin = time.perf_counter()
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            if send_interval_sec > 0:
                target_send_at = stream_origin + (i + len(chunk)) / SAMPLING_RATE
                while True:
                    remaining = target_send_at - time.perf_counter()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(remaining, 0.02))
            await ws.send(chunk.tobytes())
        await ws.send(json.dumps({'type': 'finish'}))
        send_done.set()

    async def _recv():
        nonlocal first_result_time, last_result_time
        post_send_idle = None
        while True:
            timeout = 1.0 if not send_done.is_set() else max(0.5, post_send_idle - time.perf_counter() if post_send_idle else post_finish_timeout)
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                if send_done.is_set():
                    if post_send_idle is None:
                        post_send_idle = time.perf_counter() + post_finish_timeout
                    if time.perf_counter() > post_send_idle:
                        break
                continue
            except Exception:
                break

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
                    if send_done.is_set():
                        post_send_idle = time.perf_counter() + post_finish_timeout
                    commit_reason_norm = normalize_commit_reason(
                        data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                    )
                    fsl_sec = data.get('fsl_sec')
                    finals.append(text)
                    segment_events.append({'text': text, 'tag': commit_reason_norm})
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': commit_reason_norm,
                        'audio_start_sec': data.get('audioStartSec') or parse_hms_timestamp(data.get('start')),
                        'audio_end_sec': data.get('audioEndSec') or parse_hms_timestamp(data.get('end')),
                        'fsl_sec': fsl_sec,
                        'fsl_normalized_sec': (
                            (fsl_sec + 0.8)
                            if fsl_sec is not None and commit_reason_norm == 'vad'
                            else fsl_sec
                        ),
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'trans_sec': data.get('trans_sec'),
                        'output_token_count': len(text),
                        'slotAudioStartSec': data.get('slotAudioStartSec'),
                        'vad_trigger_sec': data.get('vad_trigger_sec'),
                        'client_final_received_elapsed_sec': time.perf_counter() - processing_start,
                    })
            elif msg_type == 'partial':
                if send_done.is_set() and post_send_idle is None:
                    post_send_idle = time.perf_counter() + post_finish_timeout

    await asyncio.gather(_send(), _recv())

    total_time = (last_result_time - processing_start) if last_result_time else (time.perf_counter() - processing_start)

    return {
        'transcript': ''.join(finals).strip(),
        'segments': finals,
        'segment_events': segment_events,
        'segment_metrics': segment_metrics,
        'segment_metrics_summary': summarize_segment_metrics(segment_metrics),
        'total_time': total_time,
        'total_duration': total_duration,
        'first_token_latency': (first_result_time - processing_start) if first_result_time else None,
        'num_utterances': len(audio_files_for_speaker),
    }


async def process_batch_concat(
    audio_files,
    ws_url,
    run_dir,
    policy,
    limit=None,
    chunk_size_ms=200,
    send_interval_ms=200,
    show_commit_slash=True,
    resume=True,
    target_lang='ko',
    inter_utterance_silence_ms=800,
    post_finish_timeout=15.0,
):
    """화자별로 오디오를 이어붙여 단일 스트림 처리."""
    run_dir = Path(run_dir)
    processed_ids = load_processed_files(run_dir) if resume else set()

    # 화자별로 그룹화
    from collections import defaultdict
    by_speaker: dict[str, list] = defaultdict(list)
    for f in audio_files:
        by_speaker[f['speaker_id']].append(f)

    targets = {spk: files for spk, files in by_speaker.items() if spk not in processed_ids}

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

    logger.info('Processing %d speaker(s) in concat mode (already done: %d)',
                len(targets), len(processed_ids))
    results = []

    for idx, (speaker_id, files) in enumerate(sorted(targets.items()), start=1):
        if limit is not None and idx > limit:
            break

        logger.info('[%d/%d] Speaker %s — %d utterances', idx, len(targets), speaker_id, len(files))

        try:
            async with websockets.connect(
                ws_url, ping_interval=None, ping_timeout=None,
                open_timeout=30, max_size=10 * 1024 * 1024,
            ) as ws:
                await recv_type(ws, 'hello', timeout=8)
                out = await process_speaker_concat(
                    ws, files,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                    target_lang=target_lang,
                    inter_utterance_silence_ms=inter_utterance_silence_ms,
                    post_finish_timeout=post_finish_timeout,
                )
        except Exception as e:
            logger.error('WebSocket processing failed for speaker %s: %s', speaker_id, e)
            continue

        if out is None or not out['transcript']:
            logger.warning('Empty transcript: speaker %s', speaker_id)
            continue

        concat_reference = ''.join(f['reference'] for f in files)
        cer = compute_cer(concat_reference, out['transcript'])

        row = {
            'file_id': speaker_id,
            'speaker_id': speaker_id,
            'audio_path': files[0]['path'],
            'reference': concat_reference,
            'hypothesis': out['transcript'],
            'hyp_commit': format_commit_markers(out.get('segment_events') or []),
            'duration': out['total_duration'],
            'num_utterances': out['num_utterances'],
            'total_time': out['total_time'],
            'first_token_latency': out['first_token_latency'],
            'model_runtime': out['total_time'] - out['total_duration'],
            'target_lang': target_lang,
            'segment_metrics': out.get('segment_metrics') or [],
            'segment_metrics_summary': out.get('segment_metrics_summary') or {},
        }
        results.append(row)
        save_results_structured(all_results + results, run_dir, policy)

        logger.info('  CER: %.2f%%', cer * 100)
        logger.info('  Segments committed: %d', len(out['segments']))
        logger.info('  Duration: %.1fs / Utterances: %d', out['total_duration'], out['num_utterances'])
        if show_commit_slash and out.get('segments'):
            logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    return all_results + results


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_processed_files(run_dir: Path) -> set:
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
    audio_files,
    ws_url,
    run_dir,
    policy,
    limit=None,
    chunk_size_ms=200,
    send_interval_ms=200,
    show_commit_slash=True,
    resume=True,
    target_lang='ko',
    trailing_silence_ms=3000,
):
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

    logger.info('Processing %d file(s) (already done: %d)', len(targets), len(all_results))
    results = []

    current_speaker: str | None = None
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
            file_id = audio_info['file_id']
            speaker_id = audio_info['speaker_id']
            logger.info('[%d/%d] %s', idx, len(targets), file_id)

            audio = load_audio_file(audio_info['path'])
            if audio is None:
                continue

            duration = len(audio) / SAMPLING_RATE

            # speaker가 바뀌면 기존 연결 종료 후 새 연결 (컨텍스트 초기화)
            if speaker_id != current_speaker:
                await _close_ws()
                current_speaker = speaker_id
                try:
                    await _open_ws()
                except Exception as e:
                    logger.error('WebSocket connect failed for speaker %s: %s', speaker_id, e)
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
                await ws.send(json.dumps({'type': 'finish'}))
            except Exception as e:
                logger.error('WebSocket processing failed for %s: %s', file_id, e)
                await _close_ws()
                continue

            if not out['transcript']:
                logger.warning('Empty transcript: %s', file_id)
                continue

            model_runtime = out['total_time'] - duration
            row = {
                'file_id': file_id,
                'speaker_id': speaker_id,
                'audio_path': audio_info['path'],
                'reference': audio_info['reference'],
                'hypothesis': out['transcript'],
                'hyp_commit': format_commit_markers(out.get('segment_events') or []),
                'duration': duration,
                'total_time': out['total_time'],
                'first_token_latency': out['first_token_latency'],
                'model_runtime': model_runtime,
                'target_lang': target_lang,
                'segment_metrics': out.get('segment_metrics') or [],
                'segment_metrics_summary': out.get('segment_metrics_summary') or {},
            }
            results.append(row)
            save_results_structured(all_results + results, run_dir, policy)

            cer = compute_cer(audio_info['reference'], out['transcript'])
            speaker_rows = [r for r in results if r['speaker_id'] == speaker_id]
            speaker_cer = compute_cer_for_rows(speaker_rows)

            logger.info('  REF: %s', audio_info['reference'])
            logger.info('  HYP: %s', out['transcript'])
            logger.info('  CER: %.2f%%', cer * 100)
            logger.info('  FIRST_TOKEN_LATENCY: %s',
                        f"{out['first_token_latency']:.3f}s" if out['first_token_latency'] is not None else 'N/A')
            logger.info('  MODEL_RUNTIME(total-audio): %.3fs', model_runtime)
            seg_summary = out.get('segment_metrics_summary') or {}
            if seg_summary:
                logger.info('  FSL(avg): %s',
                            f"{seg_summary['avg_fsl_sec']:.3f}s" if seg_summary.get('avg_fsl_sec') is not None else 'N/A')
            if speaker_cer is not None:
                logger.info('  SPEAKER_%s_RUNNING_CER: %.2f%% (%d files)',
                            speaker_id, speaker_cer * 100, len(speaker_rows))
            if show_commit_slash and out.get('segments'):
                logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    finally:
        await _close_ws()

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
    cer_value, speaker_cers = calculate_cer(results, policy=policy, emit_summary=False)
    by_speaker = {}
    for row in results:
        by_speaker.setdefault(row['speaker_id'], []).append(row)

    speaker_stats = {}
    for speaker_id, rows in sorted(by_speaker.items()):
        lat = [r['first_token_latency'] for r in rows if r['first_token_latency'] is not None]
        mr = [r['model_runtime'] for r in rows if r.get('model_runtime') is not None]
        speaker_stats[speaker_id] = {
            'num_files': len(rows),
            'cer': speaker_cers.get(speaker_id),
            'first_token_latency': mean(lat) if lat else None,
            'model_runtime': mean(mr) if mr else None,
            'avg_fsl_sec': mean(_collect_segment_metric('fsl_sec', rows)) if _collect_segment_metric('fsl_sec', rows) else None,
            'avg_fsl_normalized_sec': mean(_collect_segment_metric('fsl_normalized_sec', rows)) if _collect_segment_metric('fsl_normalized_sec', rows) else None,
            'avg_encode_sec': mean(_collect_segment_metric('encode_sec', rows)) if _collect_segment_metric('encode_sec', rows) else None,
            'avg_decode_sec': mean(_collect_segment_metric('decode_sec', rows)) if _collect_segment_metric('decode_sec', rows) else None,
            'avg_final_decode_sec': mean(_collect_segment_metric('final_decode_sec', rows)) if _collect_segment_metric('final_decode_sec', rows) else None,
            'avg_trans_sec': mean(_collect_segment_metric('trans_sec', rows)) if _collect_segment_metric('trans_sec', rows) else None,
            'avg_output_tokens_per_commit': mean(_collect_segment_metric('output_token_count', rows)) if _collect_segment_metric('output_token_count', rows) else None,
            'commit_stats': _collect_commit_stats(rows),
        }

    all_lat = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    all_mr = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            'cer': cer_value,
            'first_token_latency': mean(all_lat) if all_lat else None,
            'model_runtime': mean(all_mr) if all_mr else None,
            'avg_fsl_sec': mean(_collect_segment_metric('fsl_sec', results)) if _collect_segment_metric('fsl_sec', results) else None,
            'avg_fsl_normalized_sec': mean(_collect_segment_metric('fsl_normalized_sec', results)) if _collect_segment_metric('fsl_normalized_sec', results) else None,
            'avg_encode_sec': mean(_collect_segment_metric('encode_sec', results)) if _collect_segment_metric('encode_sec', results) else None,
            'avg_decode_sec': mean(_collect_segment_metric('decode_sec', results)) if _collect_segment_metric('decode_sec', results) else None,
            'avg_final_decode_sec': mean(_collect_segment_metric('final_decode_sec', results)) if _collect_segment_metric('final_decode_sec', results) else None,
            'avg_trans_sec': mean(_collect_segment_metric('trans_sec', results)) if _collect_segment_metric('trans_sec', results) else None,
            'avg_output_tokens_per_commit': mean(_collect_segment_metric('output_token_count', results)) if _collect_segment_metric('output_token_count', results) else None,
            'commit_stats': _collect_commit_stats(results),
        },
        'speakers': speaker_stats,
    }


def save_results_structured(results, run_dir, policy):
    run_dir = Path(run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    summary = build_summary_payload(results, policy)
    metric_data = {
        'overall': summary.get('overall'),
        'speakers': summary.get('speakers'),
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


def calculate_cer(results, policy=None, emit_summary=True):
    overall_cer = compute_cer_for_rows(results)
    if overall_cer is None:
        logger.warning('No valid rows for CER.')
        return None, {}

    by_speaker = {}
    for r in results:
        by_speaker.setdefault(r['speaker_id'], []).append(r)

    speaker_cers = {
        speaker_id: compute_cer_for_rows(rows)
        for speaker_id, rows in by_speaker.items()
    }

    if emit_summary:
        ftl = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        mr = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %d', len(results))
        logger.info('Overall CER: %.2f%%', overall_cer * 100)
        logger.info('Average First Token Latency: %.3fs', sum(ftl) / len(ftl) if ftl else 0.0)
        logger.info('Average Model Runtime: %.3fs', sum(mr) / len(mr) if mr else 0.0)
        logger.info('Number of speakers: %d', len(by_speaker))
        for speaker_id, cer in sorted(speaker_cers.items()):
            logger.info('  %s CER: %s', speaker_id, f'{cer * 100:.2f}%' if cer is not None else 'N/A')
        logger.info('%s\n', '=' * 70)

    return overall_cer, speaker_cers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Qwen3 RAMC (Chinese) integration test')
    parser.add_argument('--data-dir', type=str, default=str(SCRIPT_DIR),
                        help='Root directory containing RAMC/ and label/ (default: script dir)')
    parser.add_argument('--speakers', type=str, nargs='+', default=None,
                        metavar='SPEAKER_ID',
                        help='Speaker IDs to test (default: all speakers)')
    parser.add_argument('--num-speakers', type=int, default=None,
                        help='처리할 최대 화자 수 (정렬 후 상위 N명, default: 전체)')
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    # ─── 결과 폴더 구조 인자 ────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='baseline(1.0.0)',
                        help='대분류: 모델 종류 (예: baseline(1.0.0), finetuned_silence(1.0.3))')
    parser.add_argument('--scope', type=str, default='sample', choices=['full', 'sample'],
                        help='소분류: 테스트 범위 (full=전체 데이터셋, sample=일부)')
    parser.add_argument('--tag', type=str, default=None,
                        help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    parser.add_argument('--description', type=str, default=None,
                        help='테스트 설명 (description.txt에 저장)')
    # ────────────────────────────────────────────────────────────────────────
    parser.add_argument('--limit', type=int, default=None,
                        help='처리할 최대 파일 수 (미지정 시 전체)')
    parser.add_argument('--random-sample', type=int, default=None,
                        help='무작위로 N개 파일 선택')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--calculate-cer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default=None,
                        help='서버에 로드할 모델 경로. 미지정 시 --model 값으로 MODEL_MAP에서 자동 추론')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--target-lang', type=str, default='ko',
                        help='번역 대상 언어 (default: ko)')
    parser.add_argument('--chunk-size-ms', type=int, default=200)
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='Realtime pacing (ms per chunk). Set 0 to push as fast as possible.')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-start', action='store_true', default=False)
    parser.add_argument('--trailing-silence-ms', type=int, default=3000,
                        help='오디오 끝에 추가할 묵음 (VAD 트리거용, default: 3000)')
    parser.add_argument('--concat-by-speaker', action='store_true', default=False,
                        help='화자별 오디오 이어붙여 단일 스트림 처리 (dot commit 테스트용)')
    parser.add_argument('--inter-utterance-silence-ms', type=int, default=800,
                        help='발화 간 묵음 길이 ms (concat-by-speaker 모드, default: 800)')
    parser.add_argument('--post-finish-timeout', type=float, default=15.0,
                        help='finish 후 final 대기 시간 초 (default: 15.0)')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error('Data directory not found: %s', data_dir)
        sys.exit(1)

    files = find_audio_files(data_dir, speakers=args.speakers)
    if not files:
        logger.error('No audio files found in %s', data_dir)
        sys.exit(1)

    if args.num_speakers is not None:
        all_speakers = sorted({f['speaker_id'] for f in files})
        keep = set(all_speakers[:args.num_speakers])
        files = [f for f in files if f['speaker_id'] in keep]
        logger.info('--num-speakers %d → %d speaker(s) kept: %s',
                    args.num_speakers, len(keep), sorted(keep))

    if args.random_sample is not None and args.random_sample < len(files):
        random.seed(args.random_seed)
        files = sorted(
            random.sample(files, args.random_sample),
            key=lambda x: (x['speaker_id'], x['file_id']),
        )

    logger.info('Found %d file(s) from %d speaker(s)',
                len(files), len({f['speaker_id'] for f in files}))

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

        # hello 메시지에서 서버 config 수집
        server_config = asyncio.run(fetch_server_config(ws_url))
        if server_config:
            meta['server_config'] = server_config
            with open(run_dir / 'meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        if args.concat_by_speaker:
            results = asyncio.run(
                process_batch_concat(
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
                    inter_utterance_silence_ms=args.inter_utterance_silence_ms,
                    post_finish_timeout=args.post_finish_timeout,
                )
            )
        else:
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
