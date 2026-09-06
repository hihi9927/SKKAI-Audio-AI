"""스트리밍 전송과 final 수집. 데이터셋과 무관한 부분이 전부 여기 있다.

`process_single_file` 은 오디오 하나를 실시간 속도로 밀어 넣고 `final` 을 모은다.
`process_batch` 는 그것을 파일 목록에 돌리면서 화자(챕터)가 바뀔 때 연결을 새로 맺고,
중간 결과를 `metric.json` 에 이어 저장한다(resume).
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from statistics import mean

import numpy as np
import websockets

from .audio import SAMPLING_RATE
from .results import load_processed_files, save_results_structured
from .server import recv_type

logger = logging.getLogger(__name__)


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

def summarize_segment_metrics(segment_metrics):
    if not segment_metrics:
        return {}

    def _vals(key):
        return [segment[key] for segment in segment_metrics if segment.get(key) is not None]

    summary = {
        'num_segments': len(segment_metrics),
        'avg_fsl_sec': mean(_vals('fsl_sec')) if _vals('fsl_sec') else None,
        'avg_fsl_normalized_sec': mean(_vals('fsl_normalized_sec')) if _vals('fsl_normalized_sec') else None,
        'avg_encode_sec': mean(_vals('encode_sec')) if _vals('encode_sec') else None,
        'avg_decode_sec': mean(_vals('decode_sec')) if _vals('decode_sec') else None,
        'avg_final_decode_sec': mean(_vals('final_decode_sec')) if _vals('final_decode_sec') else None,
        'avg_trans_sec': mean(_vals('trans_sec')) if _vals('trans_sec') else None,
    }
    return summary

async def process_single_file(ws, audio_data, chunk_size_ms=200, send_interval_ms=200, target_lang='ko', trailing_silence_ms=8000):
    processing_start = time.perf_counter()

    start_msg = {'type': 'start', 'lang': 'auto', 'targetLang': target_lang}
    await ws.send(json.dumps(start_msg))
    await recv_type(ws, 'ready', timeout=25, ignore_types={'partial', 'final', 'finish_done'})

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
    vad_fired = asyncio.Event()  # VAD 발동 시 trailing silence 전송 중단용

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

        # Append trailing silence so VAD can fire naturally.
        # VAD 발동(vad_fired) 즉시 중단 — 이후 silence는 slot B 할루시네이션만 유발함.
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

        # finish를 여기서 보내야 _recv()가 아직 듣고 있는 동안 finish-트리거 final을 받는다.
        # (예전엔 process_batch가 이 함수 리턴 후에 보내서, dot/SEG/VAD 중 아무 것도 못 잡은
        #  문장은 finish 커밋으로만 나오는데 그땐 이미 recv 루프가 끝나 유실됐다.)
        await ws.send(json.dumps({'type': 'finish'}))
        send_done.set()

    async def _recv():
        nonlocal first_result_time, last_result_time

        # 종료 조건: finish 전송 후 POST_FINISH_GRACE_SEC 동안 아무 메시지도 안 오면 종료.
        #
        # 예전엔 wait_for(timeout=3.0 if send_done else 15.0) 한 방으로 처리했는데,
        # timeout 값은 wait 진입 시점에 확정되고 대기 중에는 재평가되지 않는다. 이 서버는
        # ready 이후 첫 final까지 아무것도 보내지 않으므로 _recv는 15초짜리 wait 하나에
        # 그대로 앉아 있고, 결과적으로 수신 창이 "실제 유휴 시간"이 아니라 오디오 길이의
        # 계단 함수가 됐다:
        #   dur+trailing < 15s  → 만료 시 send_done이 이미 True  → break        (창 15초)
        #   dur+trailing >= 15s → 만료 시 send_done이 아직 False → continue     (창 30초)
        # 서버가 밀려 final이 15초를 넘기면 그 파일의 final이 통째로 유실되고
        # "Empty transcript" 경고 한 줄만 남는다(실측: 1688-142285-0057, final 16.18s).
        # 짧게 폴링하며 send_done을 매 루프 재평가해 창을 실제 유휴 시간에 건다.
        #
        # 다만 유휴 기준만으로는 서버가 크게 밀릴 때 여전히 놓친다(실측: GPU 경합 시
        # final이 send+11~17s에 도착해 3개 파일 유실). 그래서 평가 서버는 finish 처리를
        # 마치면 'finish_done' ack를 보내고, 이 루프는 그걸 받으면 즉시 종료한다.
        # 유휴 타임아웃은 ack를 안 보내는 서버용 fallback으로만 남는다.
        #
        # 그래서 grace를 짧게 잡으면 안 된다 — 밀린 서버는 finish를 처리하기 전에
        # 큐에 쌓인 오디오를 먼저 디코딩하므로 ack 자체가 늦게 온다. 8s로 뒀을 때
        # ack 도입 후에도 같은 3개 파일이 계속 유실됐다(final이 send+11~17s에 도착).
        # 정상 경로는 ack이므로 이 값이 커져도 런타임에 영향이 없다.
        POLL_SEC = 1.0
        POST_FINISH_GRACE_SEC = 60.0
        idle_since_finish = 0.0

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=POLL_SEC)
            except asyncio.TimeoutError:
                if send_done.is_set():
                    idle_since_finish += POLL_SEC
                    if idle_since_finish >= POST_FINISH_GRACE_SEC:
                        break
                continue
            except Exception:
                break  # ConnectionClosed 등

            idle_since_finish = 0.0

            if not isinstance(msg, str):
                continue

            data = json.loads(msg)
            msg_type = data.get('type', '')

            if msg_type == 'finish_done':
                # 서버가 이 스트림 처리를 완전히 끝냈다는 확정 신호 — 더 올 final 없음.
                # 단 이 파일의 finish를 아직 보내지 않았다면 이전 스트림의 잔여 ack이므로
                # 무시한다(그걸로 종료하면 아직 오지 않은 final을 통째로 놓친다).
                if send_done.is_set():
                    break
                continue

            if msg_type == 'vad_done':
                if real_audio_done.is_set():
                    vad_done_event.set()
                    if not data.get('has_remaining', True):
                        # tail 없음 → trailing silence 중단 후 즉시 종료
                        vad_fired.set()
                        break
                    # has_remaining=True: tail 오디오가 다음 슬롯으로 넘어감.
                    # trailing silence 계속 전송해야 다음 슬롯 VAD가 정상 발동함.
                    continue
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
    run_dir,
    policy,
    limit=None,
    chunk_size_ms=200,
    send_interval_ms=200,
    show_commit_slash=True,
    resume=True,
    target_lang='ko',
    trailing_silence_ms=8000,
    load_audio=None,
    group_key=None,
    running_metric=None,
    running_metric_label='WER',
    score_fn=None,
    metric_key='wer',
):
    """파일 목록을 순서대로 서버에 밀어 넣고 결과를 모은다.

    데이터셋이 갈리는 지점 셋만 주입받는다. 넘기지 않으면 LibriSpeech 기본값이다.

    load_audio : 경로 → float32 배열. 기본은 soundfile(flac/wav).
    group_key  : file_id → (speaker_id, chapter_id). 챕터가 바뀌면 연결을 새로 맺어
                 서버 쪽 컨텍스트를 끊는다. 기본은 LibriSpeech 의 `spk-chap-utt` 규칙.
    running_metric : 진행 로그에 찍을 누적 지표. rows → 0~1 값 또는 None. 기본은 WER.
    """
    from .audio import load_soundfile
    from .scoring import compute_wer_for_rows

    load_audio = load_audio or load_soundfile
    running_metric = running_metric or compute_wer_for_rows
    if group_key is None:
        def group_key(file_id):
            parts = file_id.split('-')
            speaker = parts[0]
            chapter = f'{parts[0]}-{parts[1]}' if len(parts) >= 2 else parts[0]
            return speaker, chapter

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

    # Load existing results so incremental saves include everything
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

    logger.info('Processing %s files (already done: %s)', len(targets), len(all_results))
    results = []

    current_chapter: str | None = None
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
        speaker_id, chapter_id = group_key(file_id)
        if audio_info.get('speaker_id'):
            speaker_id = audio_info['speaker_id']
        logger.info('[%s/%s] %s', idx, len(targets), file_id)

        audio = load_audio(audio_info['path'])
        if audio is None:
            continue

        duration = len(audio) / SAMPLING_RATE

        # chapter가 바뀌면 기존 연결 종료(컨텍스트 초기화) 후 새 연결
        if chapter_id != current_chapter:
            await _close_ws()
            current_chapter = chapter_id
            try:
                await _open_ws()
            except Exception as e:
                logger.error('WebSocket connect failed for chapter %s: %s', chapter_id, e)
                ws = None
                continue

        if ws is None:
            try:
                await _open_ws()
            except Exception as e:
                logger.error('WebSocket connect failed for %s: %s', file_id, e)
                continue

        try:
            # chapter 내 연결 유지 — finish로 서버 상태 초기화하되 히스토리는 보존.
            # finish 전송은 process_single_file 내부(_send)에서 처리 — recv 루프가
            # 아직 살아있는 동안 finish-트리거 final을 받기 위함.
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
            await _close_ws()
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

        save_results_structured(all_results + results, run_dir, policy, score_fn=score_fn,
                                metric_key=metric_key)

        speaker_rows = [r for r in results if r['speaker_id'] == speaker_id]
        speaker_score = running_metric(speaker_rows)

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
        if speaker_score is not None:
            logger.info('  SPEAKER_%s_RUNNING_%s: %.2f%% (%s files)',
                        speaker_id, running_metric_label, speaker_score * 100, len(speaker_rows))
        else:
            logger.info('  SPEAKER_%s_RUNNING_%s: N/A (%s files)',
                        speaker_id, running_metric_label, len(speaker_rows))
        if show_commit_slash and out.get('segments'):
            logger.info('  HYP_COMMIT: %s', format_commit_markers(out.get('segment_events') or []))

    finally:
        await _close_ws()

    return all_results + results
