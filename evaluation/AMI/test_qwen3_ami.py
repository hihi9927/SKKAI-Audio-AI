#!/usr/bin/env python3
"""
Qwen3 + AMI Corpus integration test runner.

AMI Corpus format:
  Audio : ES2004{a,b,c,d}/audio/ES2004{x}.{Headset-N|Mix-Headset}.wav  (16 kHz, 16-bit mono)
  Words : words/ES2004{x}.{A,B,C,D}.words.xml                          (NITE XML, ISO-8859-1)

Ground truth is built by merging all speaker word files sorted by starttime.
By default, tests against Mix-Headset.wav (realistic multi-speaker audio).

Usage examples:
  # Mixed audio, all meetings (realtime streaming)
  python test_qwen3_ami.py

  # Fast-push mode (no realtime pacing)
  python test_qwen3_ami.py --send-interval-ms 0

  # Single meeting, per-speaker headset
  python test_qwen3_ami.py --meetings ES2004a --audio-type Headset-0
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
import xml.etree.ElementTree as ET
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
DEFAULT_WORDS_DIR = SCRIPT_DIR / 'words'
DEFAULT_OUTPUT = SCRIPT_DIR / 'results' / 'qwen3_ami_results.json'
DEFAULT_SERVER_SCRIPT = (
    SCRIPT_DIR.parent / 'LibriSpeech' / 'servers' / 'streaming_websocket_server_fsl.py'
)

# --model 값 → 실제 서버 모델 경로 자동 매핑
# --server-model을 명시하면 이 매핑이 무시됩니다.
MODEL_MAP = {
    "baseline":        "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)": "Qwen/Qwen3-ASR-1.7B",
    "finetuned":        str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged"),
    "finetuned(1.0.1)": str(PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged"),
}

MEETINGS = ['ES2004a', 'ES2004b', 'ES2004c', 'ES2004d']
SPEAKERS = ['A', 'B', 'C', 'D']
# AMI convention: speaker letter → headset number
_SPEAKER_TO_HEADSET = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}
_HEADSET_TO_SPEAKER = {v: k for k, v in _SPEAKER_TO_HEADSET.items()}


# ---------------------------------------------------------------------------
# AMI ground-truth parsing
# ---------------------------------------------------------------------------

def parse_words_xml(xml_path):
    """Parse an AMI words XML file.

    Returns a list of (starttime_sec, word_text) tuples sorted by starttime.
    Skips: punctuation tokens (punc='true'), <vocalsound>, <disfmarker>, <gap>.
    """
    try:
        tree = ET.parse(str(xml_path), parser=ET.XMLParser(encoding='iso-8859-1'))
    except Exception as e:
        logger.warning('Failed to parse %s: %s', xml_path, e)
        return []

    root = tree.getroot()
    words = []

    for elem in root:
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag != 'w':
            continue
        if elem.get('punc') == 'true':
            continue

        text = (elem.text or '').strip()
        if not text:
            continue

        try:
            starttime = float(elem.get('starttime', 0))
        except (ValueError, TypeError):
            starttime = 0.0

        words.append((starttime, text))

    words.sort(key=lambda x: x[0])
    return words


def build_reference(meeting_id, words_dir, speakers=None):
    """Build combined reference transcript for one meeting.

    Interleaves all speaker word files by timestamp and joins into a single string.
    """
    if speakers is None:
        speakers = SPEAKERS

    all_words = []
    words_dir = Path(words_dir)

    for speaker in speakers:
        xml_path = words_dir / f'{meeting_id}.{speaker}.words.xml'
        if not xml_path.exists():
            logger.debug('No words file: %s, skipping.', xml_path)
            continue
        all_words.extend(parse_words_xml(xml_path))

    if not all_words:
        return ''

    all_words.sort(key=lambda x: x[0])
    return ' '.join(word for _, word in all_words)


def find_audio_files(ami_dir, words_dir, meetings=None, audio_type='Mix-Headset'):
    """Find audio files and pair with reference transcripts.

    audio_type: 'Mix-Headset' (all speakers combined) or 'Headset-N' (single speaker).
    For Headset-N, only that speaker's words file is used as reference.
    """
    if meetings is None:
        meetings = MEETINGS

    ami_dir = Path(ami_dir)
    audio_files = []

    for meeting_id in meetings:
        wav_path = ami_dir / meeting_id / 'audio' / f'{meeting_id}.{audio_type}.wav'
        if not wav_path.exists():
            # flat layout: {ami_dir}/{meeting_id}.{audio_type}.wav
            wav_path = ami_dir / f'{meeting_id}.{audio_type}.wav'
        if not wav_path.exists():
            logger.warning('Audio not found: %s', wav_path)
            continue

        if audio_type.startswith('Headset-'):
            headset_num = audio_type.split('-', 1)[1]
            speaker = _HEADSET_TO_SPEAKER.get(headset_num)
            speakers_for_ref = [speaker] if speaker else SPEAKERS
        else:
            speakers_for_ref = SPEAKERS

        reference = build_reference(meeting_id, words_dir, speakers=speakers_for_ref)
        if not reference.strip():
            logger.warning('Empty reference for %s, skipping.', meeting_id)
            continue

        audio_files.append({
            'file_id': f'{meeting_id}.{audio_type}',
            'meeting_id': meeting_id,
            'audio_type': audio_type,
            'path': str(wav_path),
            'reference': reference,
        })

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

        import os, time as _time
        log_path = f'/tmp/ami_server_{int(_time.time())}.log'
        self._server_log_path = log_path
        self._server_log_fh = open(log_path, 'w')
        logger.info('Starting Qwen3 server... (log: %s)', log_path)
        self.process = subprocess.Popen(
            cmd,
            stdout=self._server_log_fh,
            stderr=self._server_log_fh,
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
            import os, signal as _signal
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
            if hasattr(self, '_server_log_fh') and self._server_log_fh:
                self._server_log_fh.close()
                self._server_log_fh = None
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

def load_audio_file(audio_path):
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


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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

    full_final_text = ' '.join((t or '').strip() for t in finals).strip()
    lower_full = full_final_text.lower()
    last_final = (finals[-1] or '').strip()
    if not last_final:
        finals[-1] = tail
        segment_events[-1] = {'text': tail, 'tag': 'seg'}
        return finals, segment_events, True

    if tail == last_final:
        return finals, segment_events, False

    lower_tail = tail.lower()
    lower_last = last_final.lower()
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
# WER
# ---------------------------------------------------------------------------

# AMI-specific normalization: remove filler sounds and backchannel words that
# the AMI reference includes verbatim but ASR models typically suppress.
_AMI_FILLER_RE = re.compile(r'\b(mm-hmm|mm+|hmm+|um+|uh+|ah+|er+|hm+)\b', re.IGNORECASE)
_AMI_BACKCHANNEL_RE = re.compile(r'\b(yeah|yep|okay|ok)\b', re.IGNORECASE)
_AMI_BRITISH_TO_AMERICAN = {
    'programme': 'program', 'programmes': 'programs',
    'colour': 'color', 'behaviour': 'behavior',
    'organisation': 'organization', 'organisations': 'organizations',
    'realise': 'realize', 'realised': 'realized',
    'organise': 'organize', 'organised': 'organized',
    'analyse': 'analyze', 'analysed': 'analyzed',
    'centre': 'center', 'centres': 'centers',
    'defence': 'defense', 'favour': 'favor', 'favourite': 'favorite',
}


def _base_normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def _ami_normalize(text):
    """Normalize for AMI WER: remove fillers/backchannels, unify spelling/abbreviations."""
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", ' ', text)
    # AMI abbreviation format: t_v_ -> tv, l_c_d_ -> lcd
    text = re.sub(
        r'\b([a-z])_([a-z])_(?:([a-z])_)?',
        lambda m: ''.join(c.rstrip('_') for c in m.groups() if c),
        text,
    )
    words = [_AMI_BRITISH_TO_AMERICAN.get(w, w) for w in text.split()]
    text = ' '.join(words)
    text = _AMI_FILLER_RE.sub('', text)
    text = _AMI_BACKCHANNEL_RE.sub('', text)
    text = re.sub(r"[^\w\s]", ' ', text)
    return ' '.join(text.split())


def compute_wer_for_rows(rows, ami_normalize=True):
    try:
        import jiwer
    except ImportError:
        return None

    normalize = _ami_normalize if ami_normalize else _base_normalize

    pairs = [
        (normalize(r['reference']), normalize(r['hypothesis']))
        for r in rows
        if r.get('reference') and r.get('hypothesis')
    ]
    pairs = [(ref, hyp) for ref, hyp in pairs if ref.strip() and hyp.strip()]
    if not pairs:
        return None

    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))


def summarize_segment_metrics(segment_metrics):
    if not segment_metrics:
        return {}

    def _vals(key):
        return [s[key] for s in segment_metrics if s.get(key) is not None]

    return {
        'num_segments': len(segment_metrics),
        'avg_server_fsl_sec': mean(_vals('server_fsl_sec')) if _vals('server_fsl_sec') else None,
        'avg_server_fsl_normalized_sec': mean(_vals('server_fsl_normalized_sec')) if _vals('server_fsl_normalized_sec') else None,
        'avg_commit_delay_sec': mean(_vals('commit_delay_sec')) if _vals('commit_delay_sec') else None,
        'avg_translation_latency_sec': mean(_vals('translation_latency_sec')) if _vals('translation_latency_sec') else None,
        'avg_asr_inference_sec': mean(_vals('asr_inference_sec')) if _vals('asr_inference_sec') else None,
        'avg_output_tokens_per_commit': mean(_vals('output_token_count')) if _vals('output_token_count') else None,
    }


# ---------------------------------------------------------------------------
# Single-file streaming
# ---------------------------------------------------------------------------

async def process_single_file(ws, audio_data, chunk_size_ms=200, send_interval_ms=200,
                               target_lang='ko', trailing_silence_ms=3000, log_path=None):
    processing_start = time.perf_counter()

    start_msg = {'type': 'start', 'lang': 'auto', 'targetLang': target_lang}
    if log_path:
        start_msg['logPath'] = log_path
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
                chunk_end_sec = (i + len(chunk)) / SAMPLING_RATE

                if send_interval_sec > 0:
                    target_send_at = stream_origin + chunk_end_sec
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
            logger.warning('_send: ws.send() timed out — server stopped reading, aborting send')
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
                    fsl_sec = data.get('fsl_sec')
                    commit_reason_norm = normalize_commit_reason(
                        data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                    )
                    server_fsl_normalized_sec = (fsl_sec + 0.8) if fsl_sec is not None and commit_reason_norm == 'vad' else fsl_sec
                    trans_sec = data.get('trans_sec')
                    encode_sec = data.get('encode_sec') or 0.0
                    decode_sec = data.get('decode_sec') or 0.0
                    final_decode_sec = data.get('final_decode_sec') or 0.0
                    asr_inference_sec = (
                        round(encode_sec + decode_sec + final_decode_sec, 4)
                        if data.get('encode_sec') is not None or data.get('decode_sec') is not None or data.get('final_decode_sec') is not None
                        else None
                    )
                    commit_delay_sec = (
                        round(max(0.0, fsl_sec - trans_sec), 4)
                        if fsl_sec is not None and trans_sec is not None
                        else None
                    )
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': normalize_commit_reason(
                            data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                        ),
                        'audio_start_sec': audio_start_sec,
                        'audio_end_sec': audio_end_sec,
                        'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                        'translation_latency_sec': trans_sec,
                        'asr_inference_sec': asr_inference_sec,
                        'server_fsl_sec': fsl_sec,
                        'server_fsl_normalized_sec': server_fsl_normalized_sec,
                        'commit_delay_sec': commit_delay_sec,
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'slotAudioStartSec': data.get('slotAudioStartSec'),
                        'vad_trigger_sec': data.get('vad_trigger_sec'),
                        'client_final_received_elapsed_sec': receive_elapsed_sec,
                        'output_token_count': len(text.split()) if text else 0,
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

    audio_duration_sec = len(audio_data) / SAMPLING_RATE
    total_timeout = audio_duration_sec + trailing_silence_ms / 1000.0 + 300.0

    send_task = asyncio.create_task(_send())
    recv_task = asyncio.create_task(_recv())
    # asyncio.wait() is used instead of wait_for(gather(return_exceptions=True))
    # because in Python 3.12+ gather(return_exceptions=True) absorbs CancelledError
    # and prevents wait_for from timing out, causing indefinite hangs.
    done, pending = await asyncio.wait(
        {send_task, recv_task},
        timeout=total_timeout,
    )
    if pending:
        logger.warning(
            'process_single_file: hard timeout after %.0fs (audio=%.1fs), cancelling tasks',
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
                'translation_latency_sec': None,
                'asr_inference_sec': None,
                'server_fsl_sec': None,
                'commit_delay_sec': None,
                'encode_sec': None,
                'decode_sec': None,
                'final_decode_sec': None,
                'slotAudioStartSec': None,
                'vad_trigger_sec': None,
                'client_final_received_elapsed_sec': None,
                'output_token_count': len(partial_last.split()) if partial_last else 0,
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
    ami_normalize=True,
):
    run_dir = Path(run_dir)
    if resume:
        processed_ids = load_processed_files(run_dir)
    else:
        processed_ids = set()

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

    for idx, audio_info in enumerate(targets, start=1):
        file_id = audio_info['file_id']
        meeting_id = audio_info['meeting_id']
        logger.info('[%d/%d] %s', idx, len(targets), file_id)

        audio = load_audio_file(audio_info['path'])
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
                    target_lang=target_lang,
                    trailing_silence_ms=trailing_silence_ms,
                )
        except Exception as e:
            logger.error('WebSocket processing failed for %s: %s', file_id, e)
            continue

        if not out['transcript']:
            logger.warning('Empty transcript: %s', file_id)
            continue

        model_runtime = out['total_time'] - duration

        row = {
            'file_id': file_id,
            'meeting_id': meeting_id,
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
        save_results_structured(all_results + results, run_dir, policy, ami_normalize)

        wer = compute_wer_for_rows([row], ami_normalize=ami_normalize)
        logger.info('  REF: %s', audio_info['reference'][:120])
        logger.info('  HYP: %s', out['transcript'][:120])
        logger.info('  WER: %s', f'{wer * 100:.2f}%' if wer is not None else 'N/A')
        logger.info('  FIRST_TOKEN_LATENCY: %s',
                    f"{out['first_token_latency']:.3f}s" if out['first_token_latency'] is not None else 'N/A')
        logger.info('  MODEL_RUNTIME(total-audio): %.3fs', model_runtime)
        seg_summary = out.get('segment_metrics_summary') or {}
        if seg_summary:
            logger.info('  FSL(avg server): %s',
                        f"{seg_summary['avg_server_fsl_sec']:.3f}s" if seg_summary.get('avg_server_fsl_sec') is not None else 'N/A')
            logger.info('  TRANSLATION_LATENCY(avg): %s',
                        f"{seg_summary['avg_translation_latency_sec']:.3f}s" if seg_summary.get('avg_translation_latency_sec') is not None else 'N/A')
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


def build_summary_payload(results, policy, ami_normalize=True):
    wer_value, meeting_wers = calculate_wer(results, policy=policy, emit_summary=False, ami_normalize=ami_normalize)
    by_meeting = {}
    for row in results:
        by_meeting.setdefault(row['meeting_id'], []).append(row)

    meeting_stats = {}
    for meeting_id, rows in sorted(by_meeting.items()):
        lat = [r['first_token_latency'] for r in rows if r['first_token_latency'] is not None]
        mr = [r['model_runtime'] for r in rows if r.get('model_runtime') is not None]
        def _safe_mean(key, r):
            vals = _collect_segment_metric(key, r)
            return mean(vals) if vals else None

        meeting_stats[meeting_id] = {
            'num_files': len(rows),
            'wer': meeting_wers.get(meeting_id),
            'first_token_latency': mean(lat) if lat else None,
            'model_runtime': mean(mr) if mr else None,
            'avg_server_fsl_sec': _safe_mean('server_fsl_sec', rows),
            'avg_server_fsl_normalized_sec': _safe_mean('server_fsl_normalized_sec', rows),
            'avg_commit_delay_sec': _safe_mean('commit_delay_sec', rows),
            'avg_translation_latency_sec': _safe_mean('translation_latency_sec', rows),
            'avg_asr_inference_sec': _safe_mean('asr_inference_sec', rows),
            'avg_output_tokens_per_commit': _safe_mean('output_token_count', rows),
            'commit_stats': _collect_commit_stats(rows),
        }

    def _safe_mean(key, r):
        vals = _collect_segment_metric(key, r)
        return mean(vals) if vals else None

    all_lat = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    all_mr = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            'wer': wer_value,
            'first_token_latency': mean(all_lat) if all_lat else None,
            'model_runtime': mean(all_mr) if all_mr else None,
            'avg_server_fsl_sec': _safe_mean('server_fsl_sec', results),
            'avg_server_fsl_normalized_sec': _safe_mean('server_fsl_normalized_sec', results),
            'avg_commit_delay_sec': _safe_mean('commit_delay_sec', results),
            'avg_translation_latency_sec': _safe_mean('translation_latency_sec', results),
            'avg_asr_inference_sec': _safe_mean('asr_inference_sec', results),
            'avg_output_tokens_per_commit': _safe_mean('output_token_count', results),
            'commit_stats': _collect_commit_stats(results),
        },
        'meetings': meeting_stats,
    }


def save_results_structured(results, run_dir, policy, ami_normalize=True):
    run_dir = Path(run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    summary = build_summary_payload(results, policy, ami_normalize=ami_normalize)
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


def calculate_wer(results, policy=None, emit_summary=True, ami_normalize=True):
    try:
        import jiwer  # noqa: F401
    except ImportError:
        logger.warning('jiwer not installed. Install with: pip install jiwer')
        return None, {}

    overall_wer = compute_wer_for_rows(results, ami_normalize=ami_normalize)
    if overall_wer is None:
        logger.warning('No valid rows for WER.')
        return None, {}

    meeting_wers = {}
    by_meeting = {}
    for r in results:
        by_meeting.setdefault(r['meeting_id'], []).append(r)
    for meeting_id, rows in by_meeting.items():
        meeting_wers[meeting_id] = compute_wer_for_rows(rows, ami_normalize=ami_normalize)

    if emit_summary:
        ftl = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        mr = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        normalize_label = ' [AMI-normalized]' if ami_normalize else ' [raw]'
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s%s', policy_label, normalize_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %d', len(results))
        logger.info('Overall WER: %.2f%%', overall_wer * 100)
        logger.info('Average First Token Latency: %.3fs', sum(ftl) / len(ftl) if ftl else 0.0)
        logger.info('Average Model Runtime: %.3fs', sum(mr) / len(mr) if mr else 0.0)
        logger.info('Meetings: %d', len(by_meeting))
        for meeting_id, wer in sorted(meeting_wers.items()):
            logger.info('  %s WER: %s', meeting_id, f'{wer * 100:.2f}%' if wer is not None else 'N/A')
        logger.info('%s\n', '=' * 70)

    return overall_wer, meeting_wers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Qwen3 AMI Corpus integration test')
    parser.add_argument('--ami-dir', type=str, default=str(SCRIPT_DIR),
                        help='Root of AMI dataset (contains ES2004{a,b,c,d}/ and words/)')
    parser.add_argument('--words-dir', type=str, default=None,
                        help='Path to words/ XML directory (default: ami_dir/words/)')
    parser.add_argument('--audio-type', type=str, default='Mix-Headset',
                        choices=['Mix-Headset', 'Headset-0', 'Headset-1', 'Headset-2', 'Headset-3'],
                        help='Audio channel to test (default: Mix-Headset)')
    parser.add_argument('--meetings', type=str, nargs='+', default=None,
                        metavar='MEETING',
                        help='Meeting IDs to test (default: ES2004a ES2004b ES2004c ES2004d)')
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
    parser.add_argument('--calculate-wer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default=None,
                        help='서버에 로드할 모델 경로. 미지정 시 --model 값으로 MODEL_MAP에서 자동 추론')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--target-lang', type=str, default='ko')
    parser.add_argument('--chunk-size-ms', type=int, default=200)
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='Realtime pacing (ms per chunk). Set 0 to push as fast as possible.')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-start', action='store_true', default=False)
    parser.add_argument('--trailing-silence-ms', type=int, default=3000)
    parser.add_argument('--no-ami-normalize', action='store_true', default=False,
                        help='Disable AMI-specific WER normalization (filler/backchannel removal, British spelling)')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.server_model is None:
        args.server_model = MODEL_MAP.get(args.model, 'Qwen/Qwen3-ASR-1.7B')
        logger.info('server_model 자동 추론: --model %s → %s', args.model, args.server_model)

    ami_dir = Path(args.ami_dir)
    words_dir = Path(args.words_dir) if args.words_dir else ami_dir / 'words'
    meetings = args.meetings or MEETINGS

    if not ami_dir.is_dir():
        logger.error('AMI directory not found: %s', ami_dir)
        sys.exit(1)
    if not words_dir.is_dir():
        logger.error('Words directory not found: %s', words_dir)
        sys.exit(1)

    files = find_audio_files(ami_dir, words_dir, meetings=meetings, audio_type=args.audio_type)
    if not files:
        logger.error('No audio files found.')
        sys.exit(1)

    logger.info('Found %d file(s):', len(files))
    for f in files:
        ref_words = len(f['reference'].split())
        logger.info('  %s  ref_words=%d', f['file_id'], ref_words)

    # 결과 폴더 결정 및 meta/description 저장
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

        ami_normalize = not args.no_ami_normalize

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
                ami_normalize=ami_normalize,
            )
        )

        if args.calculate_wer and results:
            calculate_wer(results, policy=args.policy, emit_summary=True, ami_normalize=ami_normalize)

        logger.info('Completed. Results saved to %s', run_dir)
    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
    finally:
        if server:
            server.stop_server()


if __name__ == '__main__':
    main()
