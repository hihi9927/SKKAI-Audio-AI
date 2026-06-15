#!/usr/bin/env python3
"""
Qwen3 + KsponSpeech integration test runner.

Exercises the Qwen3 server end-to-end on Korean free-speech eval_clean clips.
Audio: raw PCM (s16le, 16 kHz, mono).  Labels: JSON {data: [{file, text}]}.
Metric: CER (Character Error Rate).

Each clip gets a fresh WebSocket connection to avoid cross-utterance context pollution.
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
from statistics import mean
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets

logging.basicConfig(format='%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
DEFAULT_POLICY = 3
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STiTy_ROOT = SCRIPT_DIR.parent.parent

MODEL_MAP = {
    "baseline":         "Qwen/Qwen3-ASR-1.7B",
    "baseline(1.0.0)":  "Qwen/Qwen3-ASR-1.7B",
    "finetuned":         str(STiTy_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged"),
    "finetuned(1.0.1)":  str(STiTy_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged"),
}

DEFAULT_DATA_JSON   = SCRIPT_DIR / "transcribe" / "eval_clean_1000.json"
DEFAULT_DATA_DIR    = SCRIPT_DIR / "data" / "eval_clean"
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / "LibriSpeech" / "servers" / "streaming_websocket_server_fsl.py"


# ─── Server lifecycle ─────────────────────────────────────────────────────────

class ServerManager:
    def __init__(self, server_script, host='localhost', port=8765, model='Qwen/Qwen3-ASR-1.7B'):
        self.server_script = server_script
        self.host = host
        self.port = port
        self.model = model
        self.process = None
        self._stderr_fh = None

    def start_server(self, additional_args=None, startup_timeout=360):
        if self.process is not None:
            self.stop_server()
        cmd = [
            sys.executable, self.server_script,
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.model,
            '--no-idle-shutdown',
        ]
        if additional_args:
            cmd.extend(additional_args)
        # log_file이 cmd에 포함돼 있으면 stderr도 같은 파일에 append
        log_file = next((cmd[i + 1] for i, a in enumerate(cmd) if a == '--log-file'), None)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            self._stderr_fh = open(log_file, 'a', encoding='utf-8')
            stderr_target = self._stderr_fh
        else:
            stderr_target = subprocess.DEVNULL

        logger.info('Starting Qwen3 server...')
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_target)
        if not self._wait_for_server_ready(timeout=startup_timeout):
            self.stop_server()
            return False
        logger.info('Server started (PID: %s)', self.process.pid)
        return True

    def _wait_for_server_ready(self, timeout=360):
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
            if self._stderr_fh:
                self._stderr_fh.close()
                self._stderr_fh = None


# ─── WebSocket helpers ────────────────────────────────────────────────────────

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


async def run_protocol_smoke(ws_url):
    logger.info('Running protocol smoke test...')
    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
        await recv_type(ws, 'hello', timeout=8)
        await ws.send(json.dumps({'type': 'start', 'lang': 'ko', 'targetLang': ''}))
        await recv_type(ws, 'ready', timeout=20)
        await ws.send(b'\x00\x00' * 1600)
        await ws.send(json.dumps({'type': 'finish'}))
        await ws.send(json.dumps({'type': 'start', 'lang': 'ko', 'targetLang': ''}))
        await recv_type(ws, 'ready', timeout=20)
        await ws.send(json.dumps({'type': 'stop'}))
    logger.info('Protocol smoke test passed.')


# ─── Data loading ─────────────────────────────────────────────────────────────

def find_audio_files(data_json: str, data_dir: str) -> list[dict]:
    """JSON 레이블과 PCM 디렉토리에서 {file_id, path, reference} 목록 반환."""
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
        })
    return audio_files


def load_audio_file(audio_path: str):
    """Raw PCM (s16le, 16 kHz) → float32 [-1, 1]."""
    try:
        audio_int16 = np.fromfile(audio_path, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32767.0
    except Exception as e:
        logger.error('Error loading %s: %s', audio_path, e)
        return None


def load_processed_files(run_dir: Path) -> set[str]:
    metric_file = Path(run_dir) / 'metric.json'
    if not metric_file.exists():
        return set()
    try:
        with open(metric_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {r['file_id'] for r in data.get('raw_results', [])}
    except Exception:
        return set()


# ─── Text normalization & CER ─────────────────────────────────────────────────

def _strip_for_cer(text: str) -> str:
    """공백·구두점 제거 후 한국어/알파벳/숫자만 유지."""
    return re.sub(r'[^가-힣ᄀ-ᇿ㄰-㆏a-zA-Z0-9]', '', text)


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
    ref = _strip_for_cer(reference)
    hyp = _strip_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def compute_corpus_cer(rows: list[dict]) -> float | None:
    """코퍼스 수준 CER: sum(edit) / sum(ref_chars)."""
    total_edit, total_ref = 0, 0
    for row in rows:
        ref = _strip_for_cer(row.get('reference', ''))
        hyp = _strip_for_cer(row.get('hypothesis', ''))
        if not ref:
            continue
        total_ref += len(ref)
        total_edit += _levenshtein(ref, hyp)
    return total_edit / total_ref if total_ref > 0 else None


# ─── Commit markers ───────────────────────────────────────────────────────────

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
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
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


# ─── Segment metrics ──────────────────────────────────────────────────────────

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


# ─── Core audio processing ────────────────────────────────────────────────────

async def process_single_file(
    ws,
    audio_data,
    chunk_size_ms=200,
    send_interval_ms=200,
    src_lang='ko',
    target_lang='en',
    trailing_silence_ms=5500,
):
    processing_start = time.perf_counter()

    await ws.send(json.dumps({'type': 'start', 'lang': src_lang, 'targetLang': target_lang}))
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
    vad_fired = asyncio.Event()

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
        # VAD가 커밋하지 않은 짧은 발화 등을 서버가 마저 flush(reason=finish)하도록 finish 전송.
        # 연결이 열린 상태에서 보내야 finish-final이 클라에 도달한다(연결 종료로 트리거하면 늦음).
        try:
            await ws.send(json.dumps({'type': 'finish'}))
        except Exception:
            pass
        send_done.set()

    async def _recv():
        nonlocal first_result_time, last_result_time
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
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

            elif msg_type == 'final':
                if first_result_time is None:
                    first_result_time = time.perf_counter()
                last_result_time = time.perf_counter()
                text = (data.get('original') or '').strip()
                if text:
                    receive_elapsed_sec = time.perf_counter() - processing_start
                    audio_start_sec = data.get('audioStartSec') or parse_hms_timestamp(data.get('start'))
                    audio_end_sec = data.get('audioEndSec') or parse_hms_timestamp(data.get('end'))
                    commit_reason = normalize_commit_reason(
                        data.get('commitReason') or data.get('commit_reason') or data.get('reason')
                    )
                    finals.append(text)
                    segment_events.append({'text': text, 'tag': commit_reason})
                    segment_metrics.append({
                        'segment_id': data.get('segmentId'),
                        'text': text,
                        'translation': (data.get('translation') or '').strip(),
                        'commit_reason': commit_reason,
                        'output_token_count': len(text),  # 한국어는 문자 단위
                        'audio_start_sec': audio_start_sec,
                        'audio_end_sec': audio_end_sec,
                        'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                        'fsl_sec': data.get('fsl_sec'),
                        'fsl_normalized_sec': (
                            (data.get('fsl_sec') + 0.8)
                            if data.get('fsl_sec') is not None and commit_reason == 'vad'
                            else data.get('fsl_sec')
                        ),
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
                # finish 커밋(종료 flush)을 받으면 더 받을 게 없으므로 종료
                if commit_reason == 'finish' and send_done.is_set():
                    break

    await asyncio.gather(_send(), _recv())

    total_time = (last_result_time - processing_start) if last_result_time else (time.perf_counter() - processing_start)
    first_token_latency = (first_result_time - processing_start) if first_result_time else None

    # 비동기 커밋(SEG/VAD)으로 final 도착 순서가 오디오 시간순과 어긋날 수 있어 정렬한다.
    # 이 시점에서 finals/segment_events/segment_metrics는 1:1로 정렬되어 있다.
    # 서버가 커밋(오디오) 순서로 segmentId를 부여하므로 segmentId를 1차 키로 사용한다.
    # (audioStartSec는 token-ratio로 재구성돼 인접 SEG/VAD 간 정밀도가 떨어져 재오염을 유발)
    if len(finals) > 1:
        _order = sorted(
            range(len(finals)),
            key=lambda i: (
                segment_metrics[i].get('segment_id')
                if segment_metrics[i].get('segment_id') is not None
                else float('inf'),
                segment_metrics[i].get('audio_start_sec')
                if segment_metrics[i].get('audio_start_sec') is not None
                else float('inf'),
                i,
            ),
        )
        finals = [finals[i] for i in _order]
        segment_events = [segment_events[i] for i in _order]
        segment_metrics = [segment_metrics[i] for i in _order]

    return {
        'transcript': ' '.join(finals).strip(),
        'segments': finals,
        'segment_events': segment_events,
        'segment_metrics': segment_metrics,
        'segment_metrics_summary': summarize_segment_metrics(segment_metrics),
        'total_time': total_time,
        'first_token_latency': first_token_latency,
    }


# ─── Batch processing ─────────────────────────────────────────────────────────

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
    src_lang='ko',
    target_lang='en',
    trailing_silence_ms=5500,
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
                    src_lang=src_lang,
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
            'src_lang': src_lang,
            'target_lang': target_lang,
            'segment_metrics': out.get('segment_metrics') or [],
            'segment_metrics_summary': out.get('segment_metrics_summary') or {},
        }
        results.append(row)

        save_results_structured(all_results + results, run_dir, policy)

        segment_summary = out.get('segment_metrics_summary') or {}
        logger.info('  REF: %s', audio_info['reference'])
        logger.info('  HYP: %s', out['transcript'])
        logger.info('  CER: %.2f%%  duration=%.1fs  runtime=%.3fs',
                    cer * 100, duration, model_runtime)
        if out['first_token_latency'] is not None:
            logger.info('  FIRST_TOKEN_LATENCY: %.3fs', out['first_token_latency'])
        if segment_summary.get('avg_fsl_sec') is not None:
            logger.info('  FSL(avg): %.3fs  ENCODE: %s  DECODE: %s  FINAL_DECODE: %s',
                        segment_summary['avg_fsl_sec'],
                        f"{segment_summary['avg_encode_sec']:.3f}s" if segment_summary.get('avg_encode_sec') is not None else 'N/A',
                        f"{segment_summary['avg_decode_sec']:.3f}s" if segment_summary.get('avg_decode_sec') is not None else 'N/A',
                        f"{segment_summary['avg_final_decode_sec']:.3f}s" if segment_summary.get('avg_final_decode_sec') is not None else 'N/A')
        if show_commit_slash and out.get('segments'):
            logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    return all_results + results


# ─── Persistence ──────────────────────────────────────────────────────────────

def _collect_segment_metric(metric_name, rows):
    values = []
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            v = seg.get(metric_name)
            if v is not None:
                values.append(v)
    return values


def _collect_commit_stats(rows):
    counts = {'vad': 0, 'seg': 0, 'dot': 0, 'finish': 0}
    for row in rows:
        for seg in row.get('segment_metrics') or []:
            reason = seg.get('commit_reason', 'seg')
            counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    return {
        'counts': counts,
        'total': total,
        'ratios': {k: (v / total if total > 0 else 0.0) for k, v in counts.items()},
    }


def build_summary_payload(results: list[dict], policy: int) -> dict:
    corpus_cer = compute_corpus_cer(results)
    lats = [r['first_token_latency'] for r in results if r.get('first_token_latency') is not None]
    runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
    all_fsl = _collect_segment_metric('fsl_sec', results)
    all_fsl_norm = _collect_segment_metric('fsl_normalized_sec', results)
    all_encode = _collect_segment_metric('encode_sec', results)
    all_decode = _collect_segment_metric('decode_sec', results)
    all_final_decode = _collect_segment_metric('final_decode_sec', results)
    all_trans = _collect_segment_metric('trans_sec', results)
    all_tokens = _collect_segment_metric('output_token_count', results)

    return {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'overall': {
            'num_files': len(results),
            'corpus_cer': corpus_cer,
            'first_token_latency': mean(lats) if lats else None,
            'model_runtime': mean(runtimes) if runtimes else None,
            'avg_fsl_sec': mean(all_fsl) if all_fsl else None,
            'avg_fsl_normalized_sec': mean(all_fsl_norm) if all_fsl_norm else None,
            'avg_encode_sec': mean(all_encode) if all_encode else None,
            'avg_decode_sec': mean(all_decode) if all_decode else None,
            'avg_final_decode_sec': mean(all_final_decode) if all_final_decode else None,
            'avg_trans_sec': mean(all_trans) if all_trans else None,
            'avg_output_chars_per_commit': mean(all_tokens) if all_tokens else None,
            'commit_stats': _collect_commit_stats(results),
        },
    }


def save_results_structured(results: list[dict], run_dir: Path, policy: int):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

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


def print_summary(results: list[dict], policy=None):
    if not results:
        return
    corpus_cer = compute_corpus_cer(results)
    lats = [r['first_token_latency'] for r in results if r.get('first_token_latency') is not None]
    runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
    policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
    logger.info('\n%s', '=' * 70)
    logger.info('RESULTS SUMMARY%s', policy_label)
    logger.info('%s', '=' * 70)
    logger.info('Total files processed: %d', len(results))
    if corpus_cer is not None:
        logger.info('Corpus CER: %.2f%%', corpus_cer * 100)
    if lats:
        logger.info('Average First Token Latency: %.3fs', mean(lats))
    if runtimes:
        logger.info('Average Model Runtime: %.3fs', mean(runtimes))
    logger.info('%s\n', '=' * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Qwen3 KsponSpeech integration test (CER)')
    parser.add_argument('--data-json', type=str, default=str(DEFAULT_DATA_JSON),
                        help='평가 JSON 파일 경로 (기본값: eval_clean_1000.json)')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='PCM 오디오 파일 디렉토리 (기본값: data/eval_clean)')
    parser.add_argument('--policy', type=int, default=DEFAULT_POLICY, choices=[DEFAULT_POLICY])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    # ─── 결과 폴더 인자 ──────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='baseline(1.0.0)',
                        help='대분류: 모델 종류 (예: baseline(1.0.0), finetuned(1.0.1))')
    parser.add_argument('--scope', type=str, default='sample',
                        help='소분류: 테스트 범위 (full / sample)')
    parser.add_argument('--tag', type=str, default=None,
                        help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    parser.add_argument('--description', type=str, default=None)
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto-server', action='store_true')
    parser.add_argument('--server-startup-timeout', type=int, default=360,
                        help='서버 준비 대기 최대 시간 (초, 기본값: 360)')
    parser.add_argument('--server-script', type=str, default=str(DEFAULT_SERVER_SCRIPT))
    parser.add_argument('--server-model', type=str, default=None,
                        help='서버에 로드할 모델 경로. 미지정 시 --model 값으로 MODEL_MAP 자동 추론')
    parser.add_argument('--server-args', type=str, default='')
    parser.add_argument('--log-file', type=str, default=None,
                        help='서버 로그 파일 경로. --auto-server 사용 시 미지정이면 logs/server.log 자동 지정')
    parser.add_argument('--src-lang', type=str, default='ko',
                        help='ASR 소스 언어 (기본값: ko)')
    parser.add_argument('--target-lang', type=str, default='en',
                        help='번역 대상 언어 (기본값: en)')
    parser.add_argument('--random-sample', type=int, default=None)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--skip-protocol-smoke', action='store_true')
    parser.add_argument('--chunk-size-ms', type=int, default=200)
    parser.add_argument('--send-interval-ms', type=int, default=200)
    parser.add_argument('--show-commit-slash', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-start', action='store_true', default=False)
    parser.add_argument('--trailing-silence-ms', type=int, default=5500)
    parser.add_argument('--gpt-translation', action='store_true', default=False)
    parser.add_argument('--translation-model', type=str, default='gpt-5.4-mini')
    parser.add_argument('--context-window', type=int, default=5)
    parser.add_argument('--correction', action='store_true', default=False)
    parser.add_argument('--correction-model', type=str, default='gpt-5.4-mini')
    parser.add_argument('--api-key', type=str, default=None)

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

    if args.random_sample is not None and args.random_sample < len(files):
        random.seed(args.random_seed)
        files = sorted(
            random.sample(sorted(files, key=lambda x: x['file_id']), args.random_sample),
            key=lambda x: x['file_id'],
        )

    run_dir = resolve_run_dir(args)
    logger.info('Results → %s', run_dir)
    logger.info('Clips to process: %d', len(files))

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
            log_file = args.log_file or str(run_dir / 'logs' / 'server.log')
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            extra += ['--log-file', log_file]
            if args.gpt_translation:
                extra += ['--gpt-translation',
                          '--translation-model', args.translation_model,
                          '--context-window', str(args.context_window)]
            if args.correction:
                extra += ['--correction', '--correction-model', args.correction_model]
            if args.api_key:
                extra += ['--api-key', args.api_key]
            if not server.start_server(extra, startup_timeout=args.server_startup_timeout):
                logger.error('Failed to start Qwen3 server.')
                sys.exit(1)
        else:
            logger.info('Manual mode. Ensure server is running at %s', ws_url)

        if not args.skip_protocol_smoke:
            asyncio.run(run_protocol_smoke(ws_url))

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
                src_lang=args.src_lang,
                target_lang=args.target_lang,
                trailing_silence_ms=args.trailing_silence_ms,
            )
        )

        if results:
            print_summary(results, policy=args.policy)

        logger.info('Completed. Results saved to %s', run_dir)

    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
    finally:
        if server:
            server.stop_server()


if __name__ == '__main__':
    main()
