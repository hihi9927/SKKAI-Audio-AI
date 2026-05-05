#!/usr/bin/env python3
"""
Qwen3 + LibriSpeech integration test runner.

This test intentionally exercises the team Qwen3 server protocol end-to-end:
- hello/start/ready/finish/restart/stop
- pair_host/pair_join/pair_leave signaling
- dataset streaming with final transcript collection

Output JSON keeps compatibility with existing whisper test shape (policy_x buckets).
"""

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
from statistics import mean
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

logging.basicConfig(format='%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
DEFAULT_POLICY = 3
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TEST_DIR = PROJECT_ROOT / "LibriSpeech" / "test-other"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "fsl" / "test" / "test_other_fsl_test.json"
DEFAULT_SERVER_SCRIPT = SCRIPT_DIR / "servers" / "streaming_websocket_server_fcl.py"


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
    raise TimeoutError(f'Expected message types {sorted(expected_types)} not received in {timeout}s')


async def run_protocol_smoke(ws_url):
    logger.info('Running protocol smoke test...')

    # start/ready/finish/restart/stop path
    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
        await recv_type(ws, 'hello', timeout=8)

        await ws.send(json.dumps({'type': 'start', 'lang': 'auto', 'targetLang': ''}))
        await recv_type(ws, 'ready', timeout=20)

        await ws.send((b'\x00\x00' * 1600))  # 100 ms silence (int16 mono)
        await ws.send(json.dumps({'type': 'finish'}))

        await ws.send(json.dumps({'type': 'start', 'lang': 'auto', 'targetLang': ''}))
        await recv_type(ws, 'ready', timeout=20)

        await ws.send(json.dumps({'type': 'stop'}))

    # pairing path
    room_id = f'test-room-{int(time.time())}'
    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as host_ws, \
               websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as guest_ws:

        await recv_type(host_ws, 'hello', timeout=8)
        await recv_type(guest_ws, 'hello', timeout=8)

        await host_ws.send(json.dumps({
            'type': 'pair_host',
            'roomId': room_id,
            'myLang': 'ko',
            'targetLang': 'en',
            'mode': 'mode-2',
        }))
        await recv_type(host_ws, 'pair_hosted', timeout=8)

        await guest_ws.send(json.dumps({
            'type': 'pair_join',
            'roomId': room_id,
            'myLang': 'en',
        }))

        guest_connected = await recv_type(guest_ws, 'pair_connected', timeout=8)
        host_connected = await recv_type(host_ws, 'pair_connected', timeout=8)

        if guest_connected.get('role') != 'guest' or host_connected.get('role') != 'host':
            raise AssertionError('pair_connected role mismatch')

        await guest_ws.send(json.dumps({'type': 'pair_leave'}))
        await recv_type(host_ws, 'pair_peer_left', timeout=8)

    logger.info('Protocol smoke test passed.')


def find_audio_files(test_dir):
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
            audio_files.append({'path': str(audio_file), 'file_id': file_id, 'reference': transcript_map[file_id]})

    return sorted(audio_files, key=lambda x: x['file_id'])


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


def load_common_file_ids(common_files_json):
    try:
        with open(common_files_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'raw_results' in data:
            return {r['file_id'] for r in data['raw_results']}
    except Exception as e:
        logger.error('Failed to load common files: %s', e)
    return set()


def load_processed_files(output_file, policy):
    if not os.path.exists(output_file):
        return set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        key = f'policy_{policy}'
        if key in data:
            return {r['file_id'] for r in data[key].get('raw_results', [])}
    except Exception:
        pass
    return set()


def normalize_commit_reason(raw_reason):
    reason = str(raw_reason or '').lower()
    if reason.startswith('vad'):
        return 'vad'
    if reason == 'dot':
        return 'dot'
    if reason == 'finish':
        return 'finish'
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

    full_final_text = ' '.join((text or '').strip() for text in finals).strip()
    lower_full_final = full_final_text.lower()
    last_final = (finals[-1] or '').strip()
    if not last_final:
        finals[-1] = tail
        segment_events[-1] = {'text': tail, 'tag': 'seg'}
        return finals, segment_events, True

    if tail == last_final:
        return finals, segment_events, False

    lower_tail = tail.lower()
    lower_last = last_final.lower()
    if lower_tail == lower_full_final:
        return finals, segment_events, False

    if lower_tail.startswith(lower_full_final):
        extra_tail = tail[len(full_final_text):].strip()
        if not extra_tail:
            return finals, segment_events, False
        finals.append(extra_tail)
        segment_events.append({'text': extra_tail, 'tag': 'seg'})
        return finals, segment_events, True

    if lower_tail.startswith(lower_last):
        finals[-1] = tail
        if segment_events:
            segment_events[-1]['text'] = tail
        return finals, segment_events, True

    if lower_full_final.endswith(lower_tail):
        return finals, segment_events, False

    if lower_last.endswith(lower_tail):
        return finals, segment_events, False

    finals.append(tail)
    segment_events.append({'text': tail, 'tag': 'seg'})
    return finals, segment_events, True


def compute_wer_for_rows(rows):
    try:
        import jiwer
    except ImportError:
        return None

    refs = [r['reference'] for r in rows if r.get('reference') and r.get('hypothesis')]
    hyps = [r['hypothesis'] for r in rows if r.get('reference') and r.get('hypothesis')]
    if not refs:
        return None

    import re

    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    refs = [normalize(x) for x in refs]
    hyps = [normalize(x) for x in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not pairs:
        return None

    refs2, hyps2 = zip(*pairs)
    return jiwer.wer(list(refs2), list(hyps2))


def summarize_segment_metrics(segment_metrics):
    if not segment_metrics:
        return {}

    def _vals(key):
        return [segment[key] for segment in segment_metrics if segment.get(key) is not None]

    summary = {
        'num_segments': len(segment_metrics),
        'avg_fsl_sec': mean(_vals('fsl_sec')) if _vals('fsl_sec') else None,
        'avg_encode_sec': mean(_vals('encode_sec')) if _vals('encode_sec') else None,
        'avg_decode_sec': mean(_vals('decode_sec')) if _vals('decode_sec') else None,
        'avg_final_decode_sec': mean(_vals('final_decode_sec')) if _vals('final_decode_sec') else None,
        'avg_trans_sec': mean(_vals('trans_sec')) if _vals('trans_sec') else None,
    }
    return summary


async def process_single_file(ws, audio_data, chunk_size_ms=200, send_interval_ms=200, target_lang='ko', trailing_silence_ms=5500, log_path=None):
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
    first_result_time = None
    last_result_time = None
    send_done = asyncio.Event()
    real_audio_done = asyncio.Event()
    vad_done_event = asyncio.Event()

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

        # Append trailing silence so VAD can fire naturally
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

        send_done.set()

    async def _recv():
        nonlocal first_result_time, last_result_time

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                if send_done.is_set():
                    break
                continue
            except Exception:
                break  # ConnectionClosed 등

            if not isinstance(msg, str):
                continue

            data = json.loads(msg)
            msg_type = data.get('type', '')

            if msg_type == 'vad_done':
                if real_audio_done.is_set():
                    vad_done_event.set()
                    break
                # 실제 오디오 전송 중 자연 묵음 VAD — 무시
                continue

            elif msg_type == 'final':
                if first_result_time is None:
                    first_result_time = time.perf_counter()
                last_result_time = time.perf_counter()
                text = (data.get('original') or '').strip()
                if text:
                    receive_elapsed_sec = time.perf_counter() - processing_start
                    audio_start_sec = data.get('audioStartSec')
                    audio_end_sec = data.get('audioEndSec')
                    if audio_start_sec is None:
                        audio_start_sec = parse_hms_timestamp(data.get('start'))
                    if audio_end_sec is None:
                        audio_end_sec = parse_hms_timestamp(data.get('end'))

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
                        'output_token_count': len(text.split()),
                        'audio_start_sec': audio_start_sec,
                        'audio_end_sec': audio_end_sec,
                        'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                        'fsl_sec': data.get('fsl_sec'),
                        'encode_sec': data.get('encode_sec'),
                        'decode_sec': data.get('decode_sec'),
                        'final_decode_sec': data.get('final_decode_sec'),
                        'trans_sec': data.get('trans_sec'),
                        'chunk_encode_log': data.get('chunk_encode_log'),
                        'slotAudioStartSec': data.get('slotAudioStartSec'),
                        'vad_trigger_sec': data.get('vad_trigger_sec'),
                        'prevSlotSpeechEndSec': data.get('prevSlotSpeechEndSec'),
                        'client_final_received_elapsed_sec': receive_elapsed_sec,
                    })


    await asyncio.gather(_send(), _recv())

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


async def process_batch(
    audio_files,
    ws_url,
    output_file,
    policy,
    limit=None,
    chunk_size_ms=200,
    send_interval_ms=200,
    show_commit_slash=True,
    resume=True,
    target_lang='ko',
    trailing_silence_ms=5500,
    log_sample_n=10,
    log_sample_seed=42,
    args=None,
):
    if resume:
        processed_ids = load_processed_files(output_file, policy)
    else:
        processed_ids = set()
    targets = [f for f in audio_files if f['file_id'] not in processed_ids]

    if limit is not None:
        targets = targets[:limit]

    if not targets:
        logger.info('No files to process.')
        return []

    # Pre-select files for per-connection log capture
    import random as _random
    _rng = _random.Random(log_sample_seed)
    all_target_ids = [f['file_id'] for f in targets]
    log_sample_ids = set(_rng.sample(all_target_ids, min(log_sample_n, len(all_target_ids))))
    plots_dir = Path(output_file).parent / "plots"
    logger.info('Log capture enabled for %d sampled files → %s', len(log_sample_ids), plots_dir)

    # Load existing results so incremental saves include everything
    all_results = []
    if resume and processed_ids and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                old = json.load(f)
            all_results = old.get(f'policy_{policy}', {}).get('raw_results', [])
        except Exception:
            pass

    logger.info('Processing %s files (already done: %s)', len(targets), len(all_results))
    results = []

    for idx, audio_info in enumerate(targets, start=1):
        file_id = audio_info['file_id']
        speaker_id = file_id.split('-')[0]
        logger.info('[%s/%s] %s', idx, len(targets), file_id)

        audio = load_audio_file(audio_info['path'])
        if audio is None:
            continue

        duration = len(audio) / SAMPLING_RATE

        log_path = str(plots_dir / f"{file_id}.log") if file_id in log_sample_ids else None

        try:
            async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None, max_size=10 * 1024 * 1024) as ws:
                await recv_type(ws, 'hello', timeout=8)
                out = await process_single_file(
                    ws,
                    audio,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                    target_lang=target_lang,
                    trailing_silence_ms=trailing_silence_ms,
                    log_path=log_path,
                )
        except Exception as e:
            logger.error('WebSocket processing failed for %s: %s', file_id, e)
            continue

        if not out['transcript']:
            logger.warning('Empty transcript: %s', file_id)
            continue

        model_runtime = out['total_time'] - duration

        results.append({
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
        })

        save_results_structured(all_results + results, args)

        speaker_rows = [r for r in results if r['speaker_id'] == speaker_id]
        speaker_wer = compute_wer_for_rows(speaker_rows)

        logger.info('  REF: %s', audio_info['reference'])
        logger.info('  HYP: %s', out['transcript'])
        logger.info('  FIRST_TOKEN_LATENCY: %s',
                    f"{out['first_token_latency']:.3f}s" if out['first_token_latency'] is not None else 'N/A')
        logger.info('  MODEL_RUNTIME(total-audio): %.3fs', model_runtime)
        segment_summary = out.get('segment_metrics_summary') or {}
        if segment_summary:
            logger.info('  FSL(avg): %s',
                        f"{segment_summary['avg_fsl_sec']:.3f}s" if segment_summary.get('avg_fsl_sec') is not None else 'N/A')
            logger.info('  ENCODE(avg): %s  DECODE(avg): %s  FINAL_DECODE(avg): %s',
                        f"{segment_summary['avg_encode_sec']:.3f}s" if segment_summary.get('avg_encode_sec') is not None else 'N/A',
                        f"{segment_summary['avg_decode_sec']:.3f}s" if segment_summary.get('avg_decode_sec') is not None else 'N/A',
                        f"{segment_summary['avg_final_decode_sec']:.3f}s" if segment_summary.get('avg_final_decode_sec') is not None else 'N/A')
            logger.info('  TRANS(avg): %s',
                        f"{segment_summary['avg_trans_sec']:.3f}s" if segment_summary.get('avg_trans_sec') is not None else 'N/A')
        if speaker_wer is not None:
            logger.info('  SPEAKER_%s_RUNNING_WER: %.2f%% (%s files)',
                        speaker_id, speaker_wer * 100, len(speaker_rows))
        else:
            logger.info('  SPEAKER_%s_RUNNING_WER: N/A (%s files)', speaker_id, len(speaker_rows))
        if show_commit_slash and out.get('segments'):
            logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    return all_results + results


def build_summary_payload(results, policy):
    wer_value, folder_wers = calculate_wer(results, policy=policy, emit_summary=False)
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
        counts = {'vad': 0, 'seg': 0, 'dot': 0, 'finish': 0}
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
        speaker_encode = _collect_segment_metric('encode_sec', rows)
        speaker_decode = _collect_segment_metric('decode_sec', rows)
        speaker_final_decode = _collect_segment_metric('final_decode_sec', rows)
        speaker_trans = _collect_segment_metric('trans_sec', rows)
        speaker_tokens = _collect_segment_metric('output_token_count', rows)
        speaker_commit = _collect_commit_stats(rows)
        folder_stats[speaker_id] = {
            'num_files': len(rows),
            'wer': folder_wers.get(speaker_id),
            'first_token_latency': mean(lat) if lat else None,
            'model_runtime': mean(model_runtime) if model_runtime else None,
            'avg_fsl_sec': mean(speaker_fsl) if speaker_fsl else None,
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
            'wer': wer_value,
            'first_token_latency': mean(all_lat) if all_lat else None,
            'model_runtime': mean(all_model_runtime) if all_model_runtime else None,
            'avg_fsl_sec': mean(all_fsl) if all_fsl else None,
            'avg_encode_sec': mean(all_encode) if all_encode else None,
            'avg_decode_sec': mean(all_decode) if all_decode else None,
            'avg_final_decode_sec': mean(all_final_decode) if all_final_decode else None,
            'avg_trans_sec': mean(all_trans) if all_trans else None,
            'avg_output_tokens_per_commit': mean(all_tokens) if all_tokens else None,
            'commit_stats': all_commit,
        },
        'folders': folder_stats,
    }


def save_results_structured(results, args):
    base_dir = Path('evaluation/LibriSpeech/results/finetuned(1.0.1)')
    
    run_num = 0
    while True:
        run_dir = base_dir / f'run_{run_num:02d}'
        if not run_dir.exists():
            break
        run_num += 1
    
    logs_path = run_dir / 'logs'
    logs_path.mkdir(parents=True, exist_ok=True)
    
    args_dict = vars(args)
    serializable_args = {}
    for key, value in args_dict.items():
        if isinstance(value, Path):
            serializable_args[key] = str(value)
        else:
            serializable_args[key] = value

    meta_data = {
        'timestamp': datetime.now().isoformat(),
        'cli_args': serializable_args,
    }
    with open(run_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

    summary = build_summary_payload(results, args.policy)
    metric_data = {
        'overall': summary.get('overall'),
        'folders': summary.get('folders'),
        'raw_results': results
    }
    with open(run_dir / 'metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results incrementally saved to {run_dir}")


def save_summary_file(results, summary_output_file, policy):
    ensure_parent_dir(summary_output_file)
    payload = build_summary_payload(results, policy)
    with open(summary_output_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def calculate_wer(results, policy=None, emit_summary=True):
    try:
        import jiwer  # noqa: F401
    except ImportError:
        logger.warning('jiwer not installed. Install with: pip install jiwer')
        return None, {}
    overall_wer = compute_wer_for_rows(results)
    if overall_wer is None:
        logger.warning('No valid rows for WER.')
        return None, {}
    wer = overall_wer
    folder_wers = {}
    by_speaker = {}
    for r in results:
        by_speaker.setdefault(r['speaker_id'], []).append(r)

    for speaker_id, rows in by_speaker.items():
        folder_wers[speaker_id] = compute_wer_for_rows(rows)

    if emit_summary:
        first_token_latencies = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0.0
        model_runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        avg_model_runtime = sum(model_runtimes) / len(model_runtimes) if model_runtimes else 0.0

        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %s', len(results))
        logger.info('Overall WER: %.2f%%', wer * 100)
        logger.info('Average First Token Latency: %.3fs', avg_first_token_latency)
        logger.info('Average Model Runtime: %.3fs', avg_model_runtime)
        logger.info('Number of speakers: %s', len(by_speaker))
        logger.info('%s\n', '=' * 70)

    return wer, folder_wers


def main():
    parser = argparse.ArgumentParser(description='Qwen3 LibriSpeech integration test with FCL metrics')
    parser.add_argument('--test-dir', type=str, default=str(DEFAULT_TEST_DIR))
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY],
                        help='Compatibility-only policy field. Qwen3 uses policy=3.')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of files to process (default: all files)')
    parser.add_argument('--calculate-wer', action=argparse.BooleanOptionalAction, default=True,
                        help='Calculate and print WER summary (default: enabled)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default='Qwen/Qwen3-ASR-1.7B')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--target-lang', type=str, default='ko')
    parser.add_argument('--common-files', type=str, default=None)
    parser.add_argument('--random-sample', type=int, default=None)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--skip-protocol-smoke', action='store_true')
    parser.add_argument('--chunk-size-ms', type=int, default=200,
                        help='Audio chunk size in ms for streaming send (default: 200)')
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='Delay between chunk sends in ms (default: 200 for real-time-like pacing)')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True,
                        help='Show commit boundaries as \"seg1 / seg2 /\" in logs')
    parser.add_argument('--fresh-start', action='store_true', default=False,
                        help='Ignore existing results and process all files from scratch')
    parser.add_argument('--trailing-silence-ms', type=int, default=5500,
                        help='Silence (ms) appended after each audio file so VAD fires before finish (default: 1000)')

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    ensure_parent_dir(args.output)

    if not os.path.isdir(args.test_dir):
        logger.error('Test directory not found: %s', args.test_dir)
        logger.error('Pass the real LibriSpeech test-other path with --test-dir.')
        sys.exit(1)

    files = find_audio_files(args.test_dir)
    if not files:
        logger.error('No LibriSpeech files found in %s', args.test_dir)
        sys.exit(1)

    if args.common_files:
        if not os.path.exists(args.common_files):
            logger.error('Common files JSON not found: %s', args.common_files)
            sys.exit(1)
        common_ids = load_common_file_ids(args.common_files)
        files = [f for f in files if f['file_id'] in common_ids]

    if args.random_sample is not None and args.random_sample < len(files):
        random.seed(args.random_seed)
        files = sorted(random.sample(sorted(files, key=lambda x: x['file_id']), args.random_sample), key=lambda x: x['file_id'])

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

        if not args.skip_protocol_smoke:
            asyncio.run(run_protocol_smoke(ws_url))

        results = asyncio.run(
            process_batch(
                files,
                ws_url,
                args.output,
                args.policy,
                args.limit,
                chunk_size_ms=args.chunk_size_ms,
                send_interval_ms=args.send_interval_ms,
                show_commit_slash=args.show_commit_slash,
                resume=not args.fresh_start,
                target_lang=args.target_lang,
                trailing_silence_ms=args.trailing_silence_ms,
                args=args,
            )
        )
        if args.calculate_wer and results:
            calculate_wer(results, policy=args.policy, emit_summary=True)

        logger.info('Completed. Results saved to %s', args.output)
    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
    finally:
        if server:
            server.stop_server()


if __name__ == '__main__':
    main()
