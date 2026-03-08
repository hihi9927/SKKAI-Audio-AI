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
PROJECT_ROOT = SCRIPT_DIR.parent


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


async def process_single_file(ws, audio_data, chunk_size_ms=100, send_interval_ms=10):
    processing_start = time.time()
    first_result_time = None

    await ws.send(json.dumps({'type': 'start', 'lang': 'auto', 'targetLang': ''}))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final'})

    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    send_interval_sec = max(0.0, send_interval_ms / 1000.0)

    for i in range(0, len(audio_int16), chunk_size):
        await ws.send(audio_int16[i:i + chunk_size].tobytes())
        if send_interval_sec > 0:
            await asyncio.sleep(send_interval_sec)

    await ws.send(json.dumps({'type': 'finish'}))

    finals = []
    partial_last = ''
    absolute_deadline = time.time() + 60
    idle_deadline = time.time() + 5

    while time.time() < absolute_deadline and time.time() < idle_deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if not isinstance(msg, str):
            continue

        data = json.loads(msg)
        msg_type = data.get('type', '')

        if msg_type == 'final':
            if first_result_time is None:
                first_result_time = time.time()
            text = (data.get('original') or '').strip()
            if text:
                finals.append(text)
                idle_deadline = time.time() + 2

        elif msg_type == 'partial':
            text = (data.get('original') or '').strip()
            if text:
                partial_last = text
                if first_result_time is None:
                    first_result_time = time.time()
                idle_deadline = time.time() + 1

        elif msg_type == 'ready':
            # Some servers may emit ready after internal reset.
            idle_deadline = min(idle_deadline, time.time() + 0.5)

    if not finals and partial_last:
        finals = [partial_last]

    total_time = time.time() - processing_start
    first_token_latency = (first_result_time - processing_start) if first_result_time else None

    return {
        'transcript': ' '.join(finals).strip(),
        'segments': finals,
        'total_time': total_time,
        'first_token_latency': first_token_latency,
    }


async def process_batch(
    audio_files,
    ws_url,
    output_file,
    policy,
    limit=None,
    chunk_size_ms=100,
    send_interval_ms=10,
    show_commit_slash=True,
):
    processed_ids = load_processed_files(output_file, policy)
    targets = [f for f in audio_files if f['file_id'] not in processed_ids]

    if limit is not None:
        targets = targets[:limit]

    if not targets:
        logger.info('No files to process.')
        return []

    logger.info('Processing %s files', len(targets))
    results = []

    for idx, audio_info in enumerate(targets, start=1):
        file_id = audio_info['file_id']
        speaker_id = file_id.split('-')[0]
        logger.info('[%s/%s] %s', idx, len(targets), file_id)

        audio = load_audio_file(audio_info['path'])
        if audio is None:
            continue

        duration = len(audio) / SAMPLING_RATE

        try:
            async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None, max_size=10 * 1024 * 1024) as ws:
                await recv_type(ws, 'hello', timeout=8)
                out = await process_single_file(
                    ws,
                    audio,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                )
        except Exception as e:
            logger.error('WebSocket processing failed for %s: %s', file_id, e)
            continue

        if not out['transcript']:
            logger.warning('Empty transcript: %s', file_id)
            continue

        avg_processing = out['total_time'] - duration if out['total_time'] > duration else out['total_time']

        results.append({
            'file_id': file_id,
            'speaker_id': speaker_id,
            'audio_path': audio_info['path'],
            'reference': audio_info['reference'],
            'hypothesis': out['transcript'],
            'duration': duration,
            'total_time': out['total_time'],
            'first_token_latency': out['first_token_latency'],
            'avg_processing_time': avg_processing,
        })

        logger.info('  REF: %s', audio_info['reference'])
        logger.info('  HYP: %s', out['transcript'])
        if show_commit_slash and out.get('segments'):
            logger.info('  HYP_COMMIT: %s /', ' / '.join(out['segments']))

    if processed_ids and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                old = json.load(f)
            key = f'policy_{policy}'
            existing = old.get(key, {}).get('raw_results', [])
            results = existing + results
        except Exception:
            pass

    save_results_structured(results, output_file, policy)
    return results


def save_results_structured(results, output_file, policy):
    existing = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    wer_value, folder_wers = calculate_wer(results, policy=policy, emit_summary=False)

    by_speaker = {}
    for row in results:
        by_speaker.setdefault(row['speaker_id'], []).append(row)

    folder_stats = {}
    for speaker_id, rows in sorted(by_speaker.items()):
        lat = [r['first_token_latency'] for r in rows if r['first_token_latency'] is not None]
        proc = [r['avg_processing_time'] for r in rows]
        folder_stats[speaker_id] = {
            'num_files': len(rows),
            'wer': folder_wers.get(speaker_id),
            'first_token_latency': (sum(lat) / len(lat)) if lat else None,
            'avg_processing_time': (sum(proc) / len(proc)) if proc else None,
        }

    all_lat = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    all_proc = [r['avg_processing_time'] for r in results]

    existing[f'policy_{policy}'] = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'num_files': len(results),
            'wer': wer_value,
            'first_token_latency': (sum(all_lat) / len(all_lat)) if all_lat else None,
            'avg_processing_time': (sum(all_proc) / len(all_proc)) if all_proc else None,
        },
        'folders': folder_stats,
        'raw_results': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def calculate_wer(results, policy=None, emit_summary=True):
    try:
        import jiwer
    except ImportError:
        logger.warning('jiwer not installed. Install with: pip install jiwer')
        return None, {}

    refs = [r['reference'] for r in results if r.get('reference') and r.get('hypothesis')]
    hyps = [r['hypothesis'] for r in results if r.get('reference') and r.get('hypothesis')]
    if not refs:
        logger.warning('No valid rows for WER.')
        return None, {}

    t = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.RemovePunctuation(), jiwer.Strip()])
    refs = [t(x) for x in refs]
    hyps = [t(x) for x in hyps]

    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not pairs:
        logger.warning('No valid normalized rows for WER.')
        return None, {}

    refs2, hyps2 = zip(*pairs)
    wer = jiwer.wer(list(refs2), list(hyps2))
    folder_wers = {}
    by_speaker = {}
    for r in results:
        by_speaker.setdefault(r['speaker_id'], []).append(r)

    for speaker_id, rows in by_speaker.items():
        s_refs = [x['reference'] for x in rows if x.get('reference') and x.get('hypothesis')]
        s_hyps = [x['hypothesis'] for x in rows if x.get('reference') and x.get('hypothesis')]
        if not s_refs:
            folder_wers[speaker_id] = None
            continue
        s_refs = [t(x) for x in s_refs]
        s_hyps = [t(x) for x in s_hyps]
        s_pairs = [(r, h) for r, h in zip(s_refs, s_hyps) if r.strip() and h.strip()]
        if not s_pairs:
            folder_wers[speaker_id] = None
            continue
        rr, hh = zip(*s_pairs)
        folder_wers[speaker_id] = jiwer.wer(list(rr), list(hh))

    if emit_summary:
        first_token_latencies = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0.0
        processing_times = [r['avg_processing_time'] for r in results if r.get('avg_processing_time') is not None]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0

        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %s', len(results))
        logger.info('Overall WER: %.2f%%', wer * 100)
        logger.info('Average First Token Latency: %.3fs', avg_first_token_latency)
        logger.info('Average Processing Time: %.3fs', avg_processing_time)
        logger.info('Number of speakers: %s', len(by_speaker))
        logger.info('%s\n', '=' * 70)

    return wer, folder_wers


def main():
    parser = argparse.ArgumentParser(description='Qwen3 LibriSpeech integration test')
    parser.add_argument('--test-dir', type=str, default=str(SCRIPT_DIR / 'test-clean'))
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY],
                        help='Compatibility-only policy field. Qwen3 uses policy=3.')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', type=str, default=str(SCRIPT_DIR / 'qwen3_librispeech_results.json'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--calculate-wer', action=argparse.BooleanOptionalAction, default=True,
                        help='Calculate and print WER summary (default: enabled)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-script', type=str, default=str(PROJECT_ROOT / 'Qwen3-ASR/examples/streaming_websocket_server.py'))
    parser.add_argument('--server-model', type=str, default='Qwen/Qwen3-ASR-1.7B')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--common-files', type=str, default=None)
    parser.add_argument('--random-sample', type=int, default=None)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--skip-protocol-smoke', action='store_true')
    parser.add_argument('--chunk-size-ms', type=int, default=100,
                        help='Audio chunk size in ms for streaming send (default: 100)')
    parser.add_argument('--send-interval-ms', type=int, default=10,
                        help='Delay between chunk sends in ms (default: 10, use 100 for real-time-like pacing)')
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True,
                        help='Show commit boundaries as \"seg1 / seg2 /\" in logs')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if not os.path.isdir(args.test_dir):
        logger.error('Test directory not found: %s', args.test_dir)
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
