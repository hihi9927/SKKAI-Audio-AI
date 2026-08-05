#!/usr/bin/env python3
"""
KtelSpeech ASR evaluation script.

Streams merged 2-speaker conversation WAV files through the WebSocket ASR
server and computes CER against reference transcripts assembled from
per-utterance label files.

Dataset layout expected under --data-dir:
  KtelSpeech/S{ID}_merged.wav   ← full conversation audio
  label/S{ID}/NNNN.txt          ← per-utterance transcripts (dialog order)
  label/S{ID}/S{ID}.json        ← speaker metadata (optional, for info)

Usage:
  # Server must already be running on localhost:8765
  python test_qwen3_ktelspeech.py

  # Test 5 sessions only
  python test_qwen3_ktelspeech.py --limit 5

  # Auto-start server and run all sessions
  python test_qwen3_ktelspeech.py --auto-server
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
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "ktel_results.json"
DEFAULT_SERVER_SCRIPT = SCRIPT_DIR.parent.parent / "Qwen3-ASR" / "examples" / "streaming_websocket_server.py"


# ─── Label normalization ──────────────────────────────────────────────────────

def normalize_label(text: str) -> str:
    """
    Normalize a single KtelSpeech label line for CER evaluation.

    Handles:
    - Noise/channel markers at line start:  o/ n/ b/ l/ u/
    - Foreign word pronunciation:           (ENGLISH)/(한글) → 한글
    - Numeric with reading:                 (20000)/(이 만) → 이 만
    - Speaker name markers:                 @이름 → 이름
    - Punctuation removal
    """
    text = text.strip()

    # Strip noise/channel markers at the very start (single letter + slash)
    text = re.sub(r'^[a-zA-Z]/', '', text).strip()

    # (original)/(pronunciation) → pronunciation
    text = re.sub(r'\([^)]+\)/\(([^)]+)\)', r'\1', text)

    # @ speaker name marker
    text = text.replace('@', '')

    # Remove punctuation (keep Korean, alphanumeric, spaces)
    text = re.sub(r'[^가-힣ᄀ-ᇿ㄰-㆏a-zA-Z0-9\s]', '', text)

    # Remove standalone single-letter noise markers (e.g. bare "n", "b", "u")
    text = re.sub(r'(?<!\S)[a-zA-Z](?!\S)', '', text)

    return ' '.join(text.split())


def load_session_reference(label_dir: Path) -> str:
    """Concatenate all utterance labels for a session in dialog order."""
    txt_files = sorted(label_dir.glob('*.txt'))
    parts = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "1\ttext" (tab-separated, first field is seq number)
                cols = line.split('\t', 1)
                raw = cols[1] if len(cols) == 2 else cols[0]
                normalized = normalize_label(raw)
                if normalized:
                    parts.append(normalized)
    return ' '.join(parts)


# ─── Session discovery ────────────────────────────────────────────────────────

def find_sessions(data_dir: Path) -> list[dict]:
    """Find all (merged_wav, label_dir) pairs and build reference text."""
    audio_dir = data_dir / 'KtelSpeech'
    label_dir = data_dir / 'label'

    if not audio_dir.is_dir():
        logger.error('Audio directory not found: %s', audio_dir)
        return []
    if not label_dir.is_dir():
        logger.error('Label directory not found: %s', label_dir)
        return []

    sessions = []
    for wav_path in sorted(audio_dir.glob('*_merged.wav')):
        session_id = wav_path.stem.replace('_merged', '')
        session_label_dir = label_dir / session_id
        if not session_label_dir.is_dir():
            logger.debug('No label dir for %s — skipping', session_id)
            continue

        reference = load_session_reference(session_label_dir)
        if not reference:
            logger.warning('Empty reference for %s — skipping', session_id)
            continue

        # Optional: load speaker metadata for logging
        meta = {}
        json_path = session_label_dir / f'{session_id}.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    raw_meta = json.load(f)
                speakers = raw_meta.get('dataSet', {}).get('typeInfo', {}).get('speakers', [])
                meta['speakers'] = [
                    {'id': s.get('id'), 'type': s.get('type'), 'gender': s.get('gender')}
                    for s in speakers
                ]
                meta['category'] = raw_meta.get('dataSet', {}).get('typeInfo', {}).get('category', '')
                num_dialogs = len(raw_meta.get('dataSet', {}).get('dialogs', []))
                meta['num_utterances'] = num_dialogs
            except Exception:
                pass

        sessions.append({
            'session_id': session_id,
            'audio_path': str(wav_path),
            'label_dir': str(session_label_dir),
            'reference': reference,
            'meta': meta,
        })

    return sessions


# ─── Audio loading ────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # stereo → mono
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        return audio
    except Exception as e:
        logger.error('Cannot load %s: %s', path, e)
        return None


# ─── WebSocket helpers ────────────────────────────────────────────────────────

async def recv_type(ws, expected_types, timeout: float = 10.0, ignore_types=None):
    if isinstance(expected_types, str):
        expected_types = {expected_types}
    else:
        expected_types = set(expected_types)
    ignore_types = set(ignore_types or [])
    end_at = time.time() + timeout
    while time.time() < end_at:
        remaining = max(0.1, end_at - time.time())
        msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        if not isinstance(msg, str):
            continue
        data = json.loads(msg)
        t = data.get('type', '')
        if t in ignore_types:
            continue
        if t in expected_types:
            return data
    raise TimeoutError(f'Expected {sorted(expected_types)} not received within {timeout}s')


async def _stream_audio(
    ws,
    audio_data: np.ndarray,
    chunk_size_ms: int,
    send_interval_ms: int,
    trailing_silence_ms: int,
    send_done: asyncio.Event,
    real_audio_done: asyncio.Event,
):
    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int(chunk_size_ms / 1000.0 * SAMPLING_RATE)
    interval_sec = send_interval_ms / 1000.0

    origin = time.perf_counter()
    for i in range(0, len(audio_int16), chunk_size):
        chunk = audio_int16[i:i + chunk_size]
        if interval_sec > 0:
            target = origin + (i + len(chunk)) / SAMPLING_RATE
            while True:
                rem = target - time.perf_counter()
                if rem <= 0:
                    break
                await asyncio.sleep(min(rem, 0.02))
        await ws.send(chunk.tobytes())

    real_audio_done.set()

    if trailing_silence_ms > 0:
        silence = np.zeros(int(SAMPLING_RATE * trailing_silence_ms / 1000), dtype=np.int16)
        sil_origin = time.perf_counter()
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i + chunk_size]
            if interval_sec > 0:
                target = sil_origin + (i + len(chunk)) / SAMPLING_RATE
                while True:
                    rem = target - time.perf_counter()
                    if rem <= 0:
                        break
                    await asyncio.sleep(min(rem, 0.02))
            await ws.send(chunk.tobytes())

    try:
        await ws.send(json.dumps({'type': 'finish'}))
    except Exception:
        pass
    send_done.set()


_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|>]*\|>')


def _strip_special_tokens(text: str) -> str:
    """Remove model special tokens like <|fim_prefix|> from ASR output."""
    return _SPECIAL_TOKEN_RE.sub('', text).strip()


def _parse_hms(ts: str | None) -> float | None:
    """Parse 'H:MM:SS.ss' timestamp string to seconds."""
    if not ts:
        return None
    try:
        parts = ts.split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, AttributeError):
        return None


async def _collect_finals(
    ws,
    send_done: asyncio.Event,
    real_audio_done: asyncio.Event,
    proc_start: float,
    post_send_wait: float = 15.0,
) -> tuple[list[dict], float | None]:
    segments = []
    first_result_time = None
    post_send_deadline: float | None = None
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            if send_done.is_set():
                if post_send_deadline is None:
                    post_send_deadline = time.perf_counter() + post_send_wait
                if time.perf_counter() >= post_send_deadline:
                    break
            continue
        except Exception:
            break
        if not isinstance(msg, str):
            continue
        data = json.loads(msg)
        msg_type = data.get('type')
        if msg_type == 'vad_done':
            if real_audio_done.is_set():
                # has_remaining: 서버 VAD 커트 이후 잔여 오디오 존재 여부
                # False → 잔여 없음, 짧은 드레인 윈도우 / True → tail 처리 대기 5s
                has_remaining = data.get('has_remaining', True)
                post_send_deadline = time.perf_counter() + (5.0 if has_remaining else 1.0)
                continue
            continue

        if msg_type == 'final':
            text = _strip_special_tokens(data.get('original') or '')
            if text:
                receive_elapsed_sec = time.perf_counter() - proc_start
                if first_result_time is None:
                    first_result_time = time.perf_counter()
                audio_start_sec = data.get('audioStartSec')
                audio_end_sec = data.get('audioEndSec')
                if audio_start_sec is None:
                    audio_start_sec = _parse_hms(data.get('start'))
                if audio_end_sec is None:
                    audio_end_sec = _parse_hms(data.get('end'))
                segments.append({
                    'original': text,
                    'translation': _strip_special_tokens(data.get('translation') or ''),
                    'commitReason': data.get('commitReason', ''),
                    'audio_start_sec': audio_start_sec,
                    'audio_end_sec': audio_end_sec,
                    'seg_audio_sec': data.get('seg_audio_sec') or data.get('segAudioSec'),
                    'fsl_sec': data.get('fsl_sec'),
                    'encode_sec': data.get('encode_sec'),
                    'decode_sec': data.get('decode_sec'),
                    'final_decode_sec': data.get('final_decode_sec'),
                    'trans_sec': data.get('trans_sec'),
                    'chunk_encode_log': data.get('chunk_encode_log'),
                    'client_final_received_elapsed_sec': receive_elapsed_sec,
                })
    return segments, first_result_time


async def process_session(ws, audio_data: np.ndarray, target_lang: str = '', chunk_size_ms: int = 200, send_interval_ms: int = 200, trailing_silence_ms: int = 5000) -> dict:
    proc_start = time.perf_counter()

    await ws.send(json.dumps({'type': 'start', 'lang': 'ko', 'targetLang': target_lang}))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final'})

    send_done = asyncio.Event()
    real_audio_done = asyncio.Event()

    (segments, first_result_time), _ = await asyncio.gather(
        _collect_finals(ws, send_done, real_audio_done, proc_start),
        _stream_audio(
            ws,
            audio_data,
            chunk_size_ms,
            send_interval_ms,
            trailing_silence_ms,
            send_done,
            real_audio_done,
        ),
    )

    duration = len(audio_data) / SAMPLING_RATE
    total_time = time.perf_counter() - proc_start
    first_token_latency = (first_result_time - proc_start) if first_result_time else None

    commit_counts = {'seg': 0, 'vad': 0, 'dot': 0, 'finish': 0, 'always': 0}
    for seg in segments:
        reason = seg.get('commitReason', '')
        if reason in commit_counts:
            commit_counts[reason] += 1
    total_commits = sum(commit_counts.values())

    segment_metrics = [
        {
            'segment_id': i,
            'text': seg['original'],
            'translation': seg['translation'],
            'commit_reason': seg['commitReason'],
            'output_token_count': len(seg['original']),
            'audio_start_sec': seg.get('audio_start_sec'),
            'audio_end_sec': seg.get('audio_end_sec'),
            'seg_audio_sec': seg.get('seg_audio_sec'),
            'fsl_sec': seg.get('fsl_sec'),
            'fsl_normalized_sec': (
                (seg.get('fsl_sec') + 0.8)
                if seg.get('fsl_sec') is not None and seg.get('commitReason') == 'vad'
                else seg.get('fsl_sec')
            ),
            'encode_sec': seg.get('encode_sec'),
            'decode_sec': seg.get('decode_sec'),
            'final_decode_sec': seg.get('final_decode_sec'),
            'trans_sec': seg.get('trans_sec'),
            'chunk_encode_log': seg.get('chunk_encode_log'),
            'client_final_received_elapsed_sec': seg.get('client_final_received_elapsed_sec'),
        }
        for i, seg in enumerate(segments, 1)
    ]
    hyp_commit = ' '.join(
        f"{s['original']} <{s['commitReason']}>" for s in segments if s['original']
    ).strip()

    return {
        'transcript': ' '.join(s['original'] for s in segments).strip(),
        'translation': ' '.join(s['translation'] for s in segments if s['translation']).strip(),
        'finals': [s['original'] for s in segments],
        'segment_metrics': segment_metrics,
        'hyp_commit': hyp_commit,
        'commit_stats': {
            'counts': commit_counts,
            'total': total_commits,
            'ratios': {k: v / total_commits if total_commits > 0 else 0.0 for k, v in commit_counts.items()},
        },
        'duration': duration,
        'total_time': total_time,
        'first_token_latency': first_token_latency,
        'model_runtime': total_time - duration,
    }


# ─── CER computation ──────────────────────────────────────────────────────────

def _strip_for_cer(text: str) -> str:
    """Remove spaces and non-Korean/non-alphanum chars for character comparison."""
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
    """Per-session CER (edit distance / reference length, character level)."""
    ref = _strip_for_cer(reference)
    hyp = _strip_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def compute_corpus_cer(rows: list[dict]) -> float | None:
    """Corpus-level CER: sum(edits) / sum(ref_chars)."""
    total_edit, total_ref = 0, 0
    for row in rows:
        ref = _strip_for_cer(row.get('reference', ''))
        hyp = _strip_for_cer(row.get('hypothesis', ''))
        if not ref:
            continue
        total_ref += len(ref)
        total_edit += _levenshtein(ref, hyp)
    return total_edit / total_ref if total_ref > 0 else None


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_processed_sessions(run_dir: Path) -> set[str]:
    metric_file = run_dir / 'metric.json'
    if not metric_file.exists():
        return set()
    try:
        with open(metric_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {r['session_id'] for r in data.get('raw_results', [])}
    except Exception:
        return set()


def save_results(results: list[dict], run_dir: Path):
    run_dir = Path(run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    cer = compute_corpus_cer(results)
    lats = [r['first_token_latency_sec'] for r in results if r.get('first_token_latency_sec') is not None]
    runtimes = [r['model_runtime_sec'] for r in results if r.get('model_runtime_sec') is not None]

    agg_counts: dict[str, int] = {'seg': 0, 'vad': 0, 'dot': 0, 'finish': 0, 'always': 0}
    for r in results:
        for k, v in r.get('commit_stats', {}).get('counts', {}).items():
            if k in agg_counts:
                agg_counts[k] += v
    agg_total = sum(agg_counts.values())

    metric_data = {
        'summary': {
            'num_sessions': len(results),
            'corpus_cer': cer,
            'avg_first_token_latency_sec': mean(lats) if lats else None,
            'avg_model_runtime_sec': mean(runtimes) if runtimes else None,
            'commit_stats': {
                'counts': agg_counts,
                'total': agg_total,
                'ratios': {k: v / agg_total if agg_total > 0 else 0.0 for k, v in agg_counts.items()},
            },
        },
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


# ─── Batch processing ─────────────────────────────────────────────────────────

async def process_batch(
    sessions: list[dict],
    ws_url: str,
    run_dir: Path,
    limit: int | None = None,
    chunk_size_ms: int = 200,
    send_interval_ms: int = 200,
    trailing_silence_ms: int = 5000,
    resume: bool = True,
    target_lang: str = 'en',
) -> list[dict]:
    run_dir = Path(run_dir)
    processed = load_processed_sessions(run_dir) if resume else set()
    targets = [s for s in sessions if s['session_id'] not in processed]
    if limit is not None:
        targets = targets[:limit]

    all_results: list[dict] = []
    if resume and processed:
        metric_file = run_dir / 'metric.json'
        if metric_file.exists():
            try:
                with open(metric_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f).get('raw_results', [])
            except Exception:
                pass

    logger.info('Sessions to process: %d  (already done: %d)', len(targets), len(all_results))
    new_results: list[dict] = []

    for idx, session in enumerate(targets, 1):
        sid = session['session_id']
        meta = session.get('meta', {})
        speakers_desc = ', '.join(
            f"{s['type']}({s['gender']})" for s in meta.get('speakers', [])
        )
        logger.info(
            '[%d/%d] %s  utterances=%s  speakers=[%s]',
            idx, len(targets), sid,
            meta.get('num_utterances', '?'),
            speakers_desc or '?',
        )

        audio = load_audio(session['audio_path'])
        if audio is None:
            continue

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=None,
                ping_timeout=None,
                max_size=100 * 1024 * 1024,
            ) as ws:
                await recv_type(ws, 'hello', timeout=10)
                out = await process_session(
                    ws, audio,
                    target_lang=target_lang,
                    chunk_size_ms=chunk_size_ms,
                    send_interval_ms=send_interval_ms,
                    trailing_silence_ms=trailing_silence_ms,
                )
        except Exception as e:
            logger.error('Failed %s: %s', sid, e)
            continue

        if not out['transcript']:
            logger.warning('Empty transcript: %s', sid)
            continue

        cer = compute_cer(session['reference'], out['transcript'])
        result = {
            'session_id': sid,
            'audio_path': session['audio_path'],
            'reference': session['reference'],
            'hypothesis': out['transcript'],
            'hyp_commit': out['hyp_commit'],
            'translation': out['translation'],
            'cer': cer,
            'duration_sec': out['duration'],
            'total_time_sec': out['total_time'],
            'first_token_latency_sec': out['first_token_latency'],
            'model_runtime_sec': out['model_runtime'],
            'num_finals': len(out['finals']),
            'commit_stats': out['commit_stats'],
            'segment_metrics': out['segment_metrics'],
            'num_utterances': meta.get('num_utterances'),
            'speakers': meta.get('speakers', []),
            'category': meta.get('category', ''),
        }
        new_results.append(result)
        save_results(all_results + new_results, run_dir)

        ref_preview = session['reference'][:120]
        hyp_preview = out['transcript'][:120]
        logger.info('  REF: %s%s', ref_preview, '...' if len(session['reference']) > 120 else '')
        logger.info('  HYP: %s%s', hyp_preview, '...' if len(out['transcript']) > 120 else '')
        logger.info('  CER: %.2f%%  finals=%d  audio=%.1fs  runtime=%.3fs',
                    cer * 100, len(out['finals']), out['duration'], out['model_runtime'])
        if out['first_token_latency'] is not None:
            logger.info('  FIRST_TOKEN: %.3fs', out['first_token_latency'])

    return all_results + new_results


# ─── Optional server management ───────────────────────────────────────────────

class ServerManager:
    def __init__(self, script: str, host: str, port: int, model: str):
        self.script = script
        self.host = host
        self.port = port
        self.model = model
        self.process = None

    def start(self, extra_args=None) -> bool:
        cmd = [
            sys.executable, self.script,
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.model,
            '--no-idle-shutdown',
        ]
        if extra_args:
            cmd.extend(extra_args)
        logger.info('Starting server: %s', ' '.join(cmd))
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, universal_newlines=True)
        if not self._wait_ready(timeout=180):
            self.stop()
            return False
        logger.info('Server ready (PID %s)', self.process.pid)
        return True

    def _wait_ready(self, timeout=180) -> bool:
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
                        return isinstance(msg, str) and json.loads(msg).get('type') == 'hello'
                except Exception:
                    return False
            if asyncio.run(_probe()):
                return True
            time.sleep(1)
        logger.error('Server readiness timeout.')
        return False

    def stop(self):
        if not self.process:
            return
        logger.info('Stopping server (PID %s)', self.process.pid)
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.process = None


async def probe_server_model(ws_url: str, timeout: float = 10.0) -> str | None:
    try:
        async with websockets.connect(
            ws_url,
            ping_interval=None,
            ping_timeout=None,
            max_size=100 * 1024 * 1024,
        ) as ws:
            data = await recv_type(ws, 'hello', timeout=timeout)
            return data.get('modelPath') or data.get('modelId') or data.get('model')
    except Exception:
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='KtelSpeech 2-speaker conversation ASR evaluation (CER)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR),
                        help='Dataset root (contains KtelSpeech/ and label/ subdirs)')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    # ─── 결과 폴더 구조 인자 ────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='finetuned',
                        help='결과 저장 경로용 실험 라벨 (예: baseline, finetuned). 실제 서버 모델 경로가 아님')
    parser.add_argument('--scope', type=str, default='sample', choices=['full', 'sample'],
                        help='소분류: 테스트 범위 (full=전체 데이터셋, sample=일부)')
    parser.add_argument('--tag', type=str, default=None,
                        help='결과 폴더명. 미지정 시 run_01, run_02 ... 자동 생성')
    parser.add_argument('--description', type=str, default=None,
                        help='테스트 설명 (description.txt에 저장)')
    # ────────────────────────────────────────────────────────────────────────
    parser.add_argument('--limit', type=int, default=None,
                        help='Max sessions to process (default: all)')
    parser.add_argument('--chunk-size-ms', type=int, default=200,
                        help='Audio chunk size in ms (default: 200)')
    parser.add_argument('--send-interval-ms', type=int, default=200,
                        help='Pacing interval between chunks in ms (default: 200)')
    parser.add_argument('--trailing-silence-ms', type=int, default=5000,
                        help='Silence appended after audio to trigger VAD (default: 5000)')
    parser.add_argument('--target-lang', default='en',
                        help='Translation target language code (empty = ASR-only, default: en)')
    parser.add_argument('--server-max-new-tokens', type=int, default=15,
                        help='max_new_tokens for auto-started server (default: 15, lower than global 32 to suppress hallucination)')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Ignore existing results and reprocess all sessions')
    parser.add_argument('--auto-server', action='store_true',
                        help='Automatically start/stop the ASR server')
    parser.add_argument('--server-script', default=str(DEFAULT_SERVER_SCRIPT),
                        help='Path to streaming_websocket_server.py')
    parser.add_argument('--server-model', default='Qwen/Qwen3-ASR-1.7B',
                        help='auto-server 시 실제 서버에 넘길 모델 id/path. manual mode에서는 메타 기록용으로만 사용됨')
    parser.add_argument('--model-id', type=str, default=None,
                        help='meta.json에 기록할 실제 모델 식별자. manual mode에서 실제 서버 모델을 명시할 때 사용')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    data_dir = Path(args.data_dir)
    sessions = find_sessions(data_dir)
    if not sessions:
        logger.error('No sessions found in %s', data_dir)
        sys.exit(1)
    logger.info('Found %d sessions', len(sessions))

    # 결과 폴더 결정 및 meta/description 저장
    run_dir = resolve_run_dir(args)
    logger.info('Results → %s', run_dir)

    ws_url = f'ws://{args.host}:{args.port}'
    detected_server_model = None
    if not args.auto_server:
        detected_server_model = asyncio.run(probe_server_model(ws_url))
        if detected_server_model:
            logger.info('Detected running server model: %s', detected_server_model)

    resolved_server_model = detected_server_model or args.server_model
    model_id = args.model_id if args.model_id else resolved_server_model
    cli_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cli_args['model_id'] = model_id  # resolved value (--model-id 미지정 시 server_model로 fallback)
    cli_args['resolved_server_model'] = resolved_server_model
    meta = {
        'timestamp': datetime.now().isoformat(),
        'experiment_label': args.model,
        'server_model': resolved_server_model,
        'model_id': model_id,
        'cli_args': cli_args,
    }
    with open(run_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.description:
        (run_dir / 'description.txt').write_text(args.description, encoding='utf-8')

    ws_url = f'ws://{args.host}:{args.port}'
    server = None
    try:
        if args.auto_server:
            if not os.path.exists(args.server_script):
                logger.error('Server script not found: %s', args.server_script)
                sys.exit(1)
            server = ServerManager(args.server_script, args.host, args.port, args.server_model)
            server_extra = []
            if args.server_max_new_tokens is not None:
                server_extra.extend(['--max-new-tokens', str(args.server_max_new_tokens)])
            if not server.start(extra_args=server_extra or None):
                logger.error('Failed to start server.')
                sys.exit(1)
        else:
            logger.info('Manual mode — server must be running at %s', ws_url)

        results = asyncio.run(process_batch(
            sessions,
            ws_url,
            run_dir,
            limit=args.limit,
            chunk_size_ms=args.chunk_size_ms,
            send_interval_ms=args.send_interval_ms,
            trailing_silence_ms=args.trailing_silence_ms,
            resume=not args.fresh_start,
            target_lang=args.target_lang,
        ))

    except KeyboardInterrupt:
        logger.info('Interrupted.')
        results = []
    finally:
        if server:
            server.stop()

    if results:
        cer = compute_corpus_cer(results)
        lats = [r['first_token_latency_sec'] for r in results if r.get('first_token_latency_sec') is not None]
        runtimes = [r['model_runtime_sec'] for r in results if r.get('model_runtime_sec') is not None]

        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY — KtelSpeech')
        logger.info('%s', '=' * 70)
        logger.info('Sessions processed : %d', len(results))
        logger.info('Corpus CER         : %.2f%%', cer * 100 if cer is not None else float('nan'))
        if lats:
            logger.info('Avg first token    : %.3fs', mean(lats))
        if runtimes:
            logger.info('Avg model runtime  : %.3fs', mean(runtimes))
        logger.info('%s', '=' * 70)
        logger.info('Completed. Results saved to %s', run_dir)


if __name__ == '__main__':
    main()
