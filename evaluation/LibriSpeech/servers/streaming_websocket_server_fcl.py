#!/usr/bin/env python3
"""Evaluation-only Qwen3 streaming server with FCL instrumentation.

This server reuses the production streaming server implementation but augments
final segment payloads with timing metadata that can be used to compute FCL.
It lives under evaluation/ so app traffic keeps using the production server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
import websockets


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
QWEN3_ROOT = PROJECT_ROOT / "Qwen3-ASR"
LIBRISPEECH_DIR = HERE.parent
sys.path.insert(0, str(QWEN3_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

import examples.streaming_websocket_server as base_server


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_hms_time(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        hours, minutes, seconds = value.split(":")
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    except Exception:
        return None


class FCLStreamingHandler(base_server.Qwen3ASRStreamingHandler):
    def __init__(self, websocket, asr_model, config, pairing_hub, http_session=None, vad_model_bytes=None):
        super().__init__(websocket, asr_model, config, pairing_hub, vad_model_bytes=vad_model_bytes)
        self._shared_http_session = http_session
        self.stream_start_perf = time.perf_counter()
        self.next_segment_id = 1
        self.segment_audio_start_sec = 0.0
        self.pending_audio_end_sec: Optional[float] = None
        self._effective_audio_end_sec: Optional[float] = None
        # (current_time, partial_text) snapshots — used to find the earliest
        # audio position at which a committed sentence first appeared, giving a
        # better audio_end_sec estimate for seg commits than current_time at
        # detection (which includes look-ahead audio needed for punctuation).
        self._partial_snapshots: list[tuple[float, str]] = []
        self._slot_final_decode_sec: dict[str, float] = {}  # _asr_finish_streaming 누적
        self._slot_seg_detected: dict[str, dict] = {}       # 첫 SEG 감지 시점 {elapsed_sec, audio_sec, decode_start_elapsed_sec}
        self._slot_chunk_encode_log: dict[str, list] = {}   # 청크별 타이밍 로그
        self._slot_last_emitted_chunk_id: dict[str, int] = {}  # 마지막 payload에 포함된 chunk_id (중복 방지)
        # _asr_streaming_transcribe 호출 시작 시간 — on_seg 내부에서 조기 로깅에 사용
        self._slot_active_transcribe_t0: dict[str, float] = {}
        # VAD 트리거 시점 (speech_end + 800ms) — VAD 커밋 시 payload에 포함
        self._pending_vad_trigger_sec: Optional[float] = None
        # 슬롯별 실제 오디오 수신 시작 시점 (VAD trigger of the switch that created this slot)
        self._slot_audio_start_sec: dict[str, float] = {"A": 0.0}
        # Empty flush (VAD fired but no segment produced): prev slot's speech_end for gap bar
        self._prev_slot_speech_end_sec: Optional[float] = None
        # Per-connection log buffer (populated when client requests log capture)
        self._connection_log: list[str] = []
        self._log_output_path: Optional[str] = None

    def _clog(self, msg: str) -> None:
        """Append a timestamped line to the per-connection log buffer."""
        if self._log_output_path is not None:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._connection_log.append(f"{ts} {msg}")

    def _flush_connection_log(self) -> None:
        if self._log_output_path and self._connection_log:
            path = Path(self._log_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._connection_log) + "\n")
            logger.info("Connection log saved: %s", path)

    def init_streaming_state(self):
        super().init_streaming_state()
        self.stream_start_perf = time.perf_counter()
        self.next_segment_id = 1
        self.segment_audio_start_sec = 0.0
        self.pending_audio_end_sec = None
        self._effective_audio_end_sec = None
        self._partial_snapshots = []
        self._slot_final_decode_sec = {}
        self._slot_seg_detected = {}
        self._slot_chunk_encode_log = {}
        self._slot_last_emitted_chunk_id = {}
        self._slot_active_transcribe_t0 = {}
        self._pending_vad_trigger_sec = None
        self._slot_audio_start_sec = {"A": 0.0}
        self._prev_slot_speech_end_sec = None
        self._connection_log = []

    def _stream_elapsed_sec(self) -> float:
        return time.perf_counter() - self.stream_start_perf

    async def _asr_streaming_transcribe(self, chunk, slot_key=None):
        key = slot_key if slot_key is not None else self.active_slot
        state = self._slot(slot_key)["state"]
        chunk_id_before = state.chunk_id
        # 슬롯이 리셋되면(chunk_id=0) last_emitted 초기화 → 재사용 슬롯에서 outer path 차단 방지
        if state.chunk_id == 0:
            self._slot_last_emitted_chunk_id.pop(key, None)

        # base server가 "partial" WebSocket 메시지를 전송하지 않아 send_message("partial")
        # 인터셉트로는 스냅샷이 채워지지 않음. 대신 각 청크 추론 직전에 이전 청크까지의
        # 누적 텍스트를 current_time 기준으로 저장해 _seg_audio_end_sec()가 동작하게 함.
        raw_text = (state.text or "").strip()
        if raw_text:
            self._partial_snapshots.append((self.current_time, raw_text))

        t0 = time.perf_counter()
        ts_elapsed = round(t0 - self.stream_start_perf, 4)  # wall-clock elapsed (plot x축 기준)
        # _process_slot_updates(on_seg) 내부에서 조기 로깅이 가능하도록 t0 공유
        self._slot_active_transcribe_t0[key] = t0
        await super()._asr_streaming_transcribe(chunk, slot_key)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        self._slot_active_transcribe_t0.pop(key, None)
        # model.generate()가 실제 호출된 경우만 기록 (chunk_id가 증가한 경우)
        if state.chunk_id > chunk_id_before:
            # on_seg 경로(→ _emit_final_payload)에서 이미 커밋된 chunk_id는 중복 추가 방지:
            # _emit_final_payload가 on_seg 내부(model.generate() 도중)에서 chunk_log를 pop하면
            # 이후 outer path가 빈 log에 같은 chunk를 다시 추가하여 다음 segment를 오염시킴
            last_emitted = self._slot_last_emitted_chunk_id.get(key, -1)
            if chunk_id_before <= last_emitted:
                return
            existing = self._slot_chunk_encode_log.get(key, [])
            audio_pos = round(self.current_time, 3)
            if not existing or existing[-1].get("audio_pos_sec") != audio_pos:
                dp = getattr(state, "_decode_start_perf", None)
                if dp is not None and t0 <= dp <= t1:
                    chunk_decode_sec = round(t1 - dp, 4)
                    chunk_decode_start_elapsed = round(dp - self.stream_start_perf, 4)
                else:
                    chunk_decode_sec = 0.0
                    chunk_decode_start_elapsed = None
                entry = {
                    "chunk_id": chunk_id_before,
                    "audio_pos_sec": audio_pos,
                    "encode_sec": round(elapsed, 4),
                    "chunk_decode_sec": chunk_decode_sec,
                    "chunk_transcribe_start_elapsed": ts_elapsed,
                    "chunk_decode_start_elapsed": chunk_decode_start_elapsed,
                }
                self._slot_chunk_encode_log.setdefault(key, []).append(entry)
                logger.info(
                    "[ENCODE] slot=%s chunk=%d audio_pos=%.3fs ts=%.3fs ds=%.3fs decode=%.3fs",
                    key, chunk_id_before, audio_pos, ts_elapsed,
                    chunk_decode_start_elapsed or -1, chunk_decode_sec,
                )

    async def _asr_finish_streaming(self, slot_key=None):
        key = slot_key if slot_key is not None else self.active_slot
        t0 = time.perf_counter()
        await super()._asr_finish_streaming(slot_key)
        self._slot_final_decode_sec[key] = self._slot_final_decode_sec.get(key, 0.0) + (time.perf_counter() - t0)

    async def _process_slot_updates(self, slot_key=None, force_reason=None):
        key = slot_key if slot_key is not None else self.active_slot
        if key not in self._slot_seg_detected:
            # t0가 None이면 VAD/finish 경로 호출 (streaming_transcribe 밖) → 기록 금지
            # VAD 경로에서 stale elapsed로 _slot_seg_detected를 오염시키면 decode_sec가
            # 실제보다 수 초 이상 크게 계산되어 decode↔trans 사이에 가짜 공백이 발생함
            t0 = self._slot_active_transcribe_t0.get(key)
            if t0 is not None:
                elapsed = self._stream_elapsed_sec()
                slot = self._slot(slot_key)
                decode_start_perf = getattr(slot["state"], "_decode_start_perf", None)
                decode_start_elapsed = (decode_start_perf - self.stream_start_perf) if decode_start_perf else None
                self._slot_seg_detected[key] = {
                    "elapsed_sec": elapsed,
                    "audio_sec": self.current_time,
                    "decode_start_elapsed_sec": decode_start_elapsed,
                }
                logger.info(
                    "[SEG] slot=%s elapsed=%.3fs decode_start=%.3fs audio_pos=%.3fs",
                    key, elapsed,
                    decode_start_elapsed if decode_start_elapsed is not None else -1.0,
                    self.current_time,
                )
                self._clog(
                    f"[SEG] slot={key} elapsed={elapsed:.3f}s "
                    f"decode_start={decode_start_elapsed:.3f}s audio_pos={self.current_time:.3f}s"
                    if decode_start_elapsed is not None else
                    f"[SEG] slot={key} elapsed={elapsed:.3f}s audio_pos={self.current_time:.3f}s"
                )
                # chunk 로그 추가 (on_seg는 generate() loop 내부 → chunk_id += 1 아직 실행 전)
                t_seg = time.perf_counter()
                encode_elapsed = t_seg - t0
                audio_pos = round(self.current_time, 3)
                existing = self._slot_chunk_encode_log.get(key, [])
                if not existing or existing[-1].get("audio_pos_sec") != audio_pos:
                    dp = getattr(slot["state"], "_decode_start_perf", None)
                    if dp is not None and t0 <= dp <= t_seg:
                        chunk_decode_sec = round(t_seg - dp, 4)
                        chunk_decode_start_elapsed = round(dp - self.stream_start_perf, 4)
                    else:
                        chunk_decode_sec = 0.0
                        chunk_decode_start_elapsed = None
                    ts_elapsed = round(t0 - self.stream_start_perf, 4)
                    entry = {
                        "chunk_id": slot["state"].chunk_id,
                        "audio_pos_sec": audio_pos,
                        "encode_sec": round(encode_elapsed, 4),
                        "chunk_decode_sec": chunk_decode_sec,
                        "chunk_transcribe_start_elapsed": ts_elapsed,
                        "chunk_decode_start_elapsed": chunk_decode_start_elapsed,
                    }
                    self._slot_chunk_encode_log.setdefault(key, []).append(entry)
                    logger.info(
                        "[ENCODE-EARLY] slot=%s chunk=%d audio_pos=%.3fs ts=%.3fs ds=%.3fs decode=%.3fs",
                        key, slot["state"].chunk_id, audio_pos, ts_elapsed,
                        chunk_decode_start_elapsed or -1, chunk_decode_sec,
                    )
            elif force_reason in ("vad", "finish"):
                # VAD/finish path: elapsed/decode_start는 stale → encode_sec/decode_sec 기록 금지
                # SEG가 텍스트에 있으면 audio_sec만 기록해 seg_audio_sec 마커 표시에 사용
                raw = (self._slot(slot_key)["state"].text or "").strip()
                if "<SEG>" in raw:
                    audio_sec = self.pending_audio_end_sec \
                        if self.pending_audio_end_sec is not None else self.current_time
                    self._slot_seg_detected[key] = {
                        "elapsed_sec": None,
                        "audio_sec": audio_sec,
                        "decode_start_elapsed_sec": None,
                    }
                    logger.info("[SEG-VAD] slot=%s audio_pos=%.3fs", key, audio_sec)
        await super()._process_slot_updates(slot_key, force_reason=force_reason)

    async def _translate_with_metadata(
        self,
        text: str,
        target_lang: str,
    ) -> tuple[str, str, dict[str, Any]]:
        t0 = self._stream_elapsed_sec()
        translation, detected_lang = await base_server.google_translate_async(
            self.http_session,
            text,
            target_lang,
        )
        t1 = self._stream_elapsed_sec()
        return translation, detected_lang, {
            "trans_sec": round(t1 - t0, 4),
            "_trans_end_elapsed": t1,    # FSL 계산용 (클라이언트 미전송)
            "_trans_start_elapsed": t0,  # pre_trans_sec 계산용 (클라이언트 미전송)
        }

    # ── 훅 오버라이드 ──────────────────────────────────────────────────────────

    async def send_message(self, msg_type: str, **kwargs) -> None:
        """Intercept partial messages to record (current_time, text) snapshots."""
        if msg_type == "partial":
            text = (kwargs.get("original") or "").strip()
            if text:
                self._partial_snapshots.append((self.current_time, text))
                logger.info("[PARTIAL] t=%.3fs | %s", self.current_time, text)
                self._clog(f"[PARTIAL] t={self.current_time:.3f}s | {text}")
        await super().send_message(msg_type, **kwargs)

    def _seg_audio_end_sec(self, sentence: str) -> float:
        """Return the current_time of the last partial snapshot that contains
        the sentence body *without* its final punctuation.

        Two-stage lookahead problem with seg commits:
          1. The model needs extra audio chunks after the sentence ends to
             decide to add the sentence-final punctuation.
          2. The base server only commits once text *after* the punctuation
             appears (line 544 in base server), adding another chunk of delay.

        Using the first snapshot that contains the full sentence (with punct)
        removes stage-2 lookahead but leaves stage-1.  Using the last snapshot
        that contains the sentence body *before* the punctuation appeared
        removes both stages and gives the best approximation of T_end (the
        actual moment the sentence audio finished).

        Falls back to the first-with-punct snapshot, then self.current_time.
        """
        needle = sentence.strip()
        # Strip sentence-final punctuation to match the pre-punct partial text.
        needle_no_punct = needle.rstrip('.,!?;:…。！？').strip()

        last_pre_punct_time: Optional[float] = None
        first_with_punct_time: Optional[float] = None

        for t, text in self._partial_snapshots:
            if t < self.segment_audio_start_sec:
                continue
            if needle in text:
                first_with_punct_time = t
                break
            if needle_no_punct and needle_no_punct in text:
                last_pre_punct_time = t

        if last_pre_punct_time is not None:
            logger.info("[SEG_AUDIO_END] sentence='%s' → last_pre_punct_time=%.3fs (needle_no_punct='%s')", needle, last_pre_punct_time, needle_no_punct)
            return last_pre_punct_time
        if first_with_punct_time is not None:
            logger.info("[SEG_AUDIO_END] sentence='%s' → first_with_punct_time=%.3fs (fallback)", needle, first_with_punct_time)
            return first_with_punct_time
        logger.info("[SEG_AUDIO_END] sentence='%s' → current_time=%.3fs (no snapshot matched)", needle, self.current_time)
        return self.current_time

    async def _translate(
        self, text: str, target_lang: str, audio_end_sec: Optional[float] = None
    ) -> tuple[str, str, dict]:
        if self.pending_audio_end_sec is not None:
            # VAD commit or finish flush: use the accurate VAD/finish timestamp.
            effective_audio_end = self.pending_audio_end_sec
        else:
            # Seg commit: find the earliest snapshot containing this sentence,
            # which is closer to the true end of the sentence audio than
            # current_time (which includes look-ahead chunks for punctuation).
            effective_audio_end = self._seg_audio_end_sec(text)
        self._effective_audio_end_sec = effective_audio_end
        return await self._translate_with_metadata(text, target_lang)

    def _get_flush_audio_end_sec(self) -> float:
        return self.pending_audio_end_sec if self.pending_audio_end_sec is not None else self.current_time

    async def _on_vad_commit(self, audio_end_sec: float) -> None:
        speech_end_sec = max(0.0, audio_end_sec - base_server.VAD_MIN_SILENCE_MS / 1000.0)
        # Detect empty flush: previous VAD trigger was never consumed → no FINAL was emitted
        if self._pending_vad_trigger_sec is not None and self.pending_audio_end_sec is not None:
            self._prev_slot_speech_end_sec = self.pending_audio_end_sec
        logger.info("[VAD_COMMIT] audio_end_sec=%.3fs → speech_end_sec=%.3fs", audio_end_sec, speech_end_sec)
        self._clog(f"[VAD_COMMIT] audio_end_sec={audio_end_sec:.3f}s → speech_end_sec={speech_end_sec:.3f}s")
        self.pending_audio_end_sec = speech_end_sec
        self._pending_vad_trigger_sec = audio_end_sec
        # 슬롯 스위치 후 _on_vad_commit 호출 시점에 self.active_slot은 이미 새 슬롯
        self._slot_audio_start_sec[self.active_slot] = audio_end_sec

    async def _on_vad_done(self, slot_key: str) -> None:
        logger.info("[VAD_DONE] slot=%s", slot_key)
        await self.send_message("vad_done")

    async def _emit_final_payload(
        self,
        *,
        slot_key: str,
        original: str,
        translation: str,
        language: str,
        reason: str,
        audio_end_sec: float,
        extra: Optional[dict] = None,
    ) -> None:
        timing = extra or {}

        # VAD/finish 커밋 전용: _asr_finish_streaming 시간
        final_decode_sec = self._slot_final_decode_sec.pop(slot_key, 0.0)
        timing["final_decode_sec"] = final_decode_sec

        # 청크별 encode 시간 로그
        chunk_log = self._slot_chunk_encode_log.pop(slot_key, [])
        if chunk_log:
            timing["chunk_encode_log"] = chunk_log
            # 커밋된 최대 chunk_id 기록 → outer _asr_streaming_transcribe의 중복 추가 방지
            self._slot_last_emitted_chunk_id[slot_key] = max(
                c.get("chunk_id", -1) for c in chunk_log
            )

        # encode / decode / FSL
        # - encode_sec: stream 시작 ~ model.generate() 직전 (오디오 입력 + prefill)
        # - decode_sec: model.generate() 시작 ~ SEG 토큰 감지 (autoregressive decode)
        # - fsl_sec (SEG commit): SEG 감지 시점 ~ 번역 완료 시점 (post-SEG 처리 레이턴시)
        # - fsl_sec (VAD/finish): final_decode_sec + trans_sec
        seg_info = self._slot_seg_detected.pop(slot_key, None)
        trans_end_elapsed = timing.pop("_trans_end_elapsed", None)
        trans_start_elapsed = timing.pop("_trans_start_elapsed", None)

        # 슬롯 실제 오디오 시작 시점
        slot_audio_start = self._slot_audio_start_sec.pop(slot_key, None)
        if slot_audio_start is not None:
            timing["slotAudioStartSec"] = round(slot_audio_start, 3)

        # VAD 트리거 시점 (0.8초 침묵 후) — VAD 커밋에서만 기록
        vad_trigger_sec = self._pending_vad_trigger_sec
        self._pending_vad_trigger_sec = None
        if reason == "vad" and vad_trigger_sec is not None:
            timing["vad_trigger_sec"] = round(vad_trigger_sec, 3)

        # 이전 슬롯이 empty flush였을 때 gap bar 시각화를 위해 speech_end 전달
        if self._prev_slot_speech_end_sec is not None:
            timing["prevSlotSpeechEndSec"] = round(self._prev_slot_speech_end_sec, 3)
            self._prev_slot_speech_end_sec = None

        if reason not in ("vad", "finish") and seg_info is not None:
            d_start = seg_info.get("decode_start_elapsed_sec")
            if d_start is not None:
                timing["encode_sec"] = d_start
                timing["decode_sec"] = round(max(0.0, seg_info["elapsed_sec"] - d_start), 4)
            timing["seg_audio_sec"] = round(seg_info["audio_sec"], 3)
            # FSL for SEG: translation end - estimated audio end (both stream-elapsed)
            if trans_end_elapsed is not None:
                effective_ae = (
                    self._effective_audio_end_sec
                    if self._effective_audio_end_sec is not None
                    else audio_end_sec
                )
                timing["fsl_sec"] = round(max(0.0, trans_end_elapsed - effective_ae), 4)
        elif seg_info is not None:
            # VAD/finish path에서도 final decode 중 SEG 감지된 경우 위치 기록
            timing["seg_audio_sec"] = round(seg_info["audio_sec"], 3)
            # elapsed_sec is None for VAD-path SEG (no wall-clock timing available)
            seg_elapsed = seg_info.get("elapsed_sec")
            if seg_elapsed is not None:
                if trans_start_elapsed is not None:
                    pre_trans = round(max(0.0, trans_start_elapsed - seg_elapsed), 4)
                    if pre_trans > 0.001:
                        timing["pre_trans_sec"] = pre_trans
                if trans_end_elapsed is not None:
                    timing["fsl_sec"] = round(max(0.0, trans_end_elapsed - seg_elapsed), 4)
            if "fsl_sec" not in timing:
                # seg_elapsed가 없는 VAD-path SEG → final decode + 번역 시간으로 fallback
                timing["fsl_sec"] = round(final_decode_sec + timing.get("trans_sec", 0.0), 4)
        else:
            # VAD/finish: SEG 감지 없음 → final decode + 번역 시간
            timing["fsl_sec"] = round(final_decode_sec + timing.get("trans_sec", 0.0), 4)

        encoding_sec = timing.get("encode_sec", 0.0)

        if self._effective_audio_end_sec is not None:
            audio_end_sec = self._effective_audio_end_sec
            self._effective_audio_end_sec = None
        segment_id = self.next_segment_id
        self.next_segment_id += 1
        audio_start_sec = self.segment_audio_start_sec
        payload = {
            "type": "final",
            "segmentId": segment_id,
            "start": base_server.format_time(audio_start_sec),
            "end": base_server.format_time(audio_end_sec),
            "audioStartSec": audio_start_sec,
            "audioEndSec": audio_end_sec,
            "original": original,
            "translation": translation,
            "language": language,
            "commitReason": reason,
            **timing,
        }
        logger.info(
            "[FINAL] seg=%d reason=%s audio=[%.3f, %.3f] "
            "fsl=%.3fs encode=%.3fs decode=%.3fs final_decode=%.3fs trans=%.3fs | text='%s'",
            segment_id, reason, audio_start_sec, audio_end_sec,
            timing.get("fsl_sec", 0.0), encoding_sec,
            timing.get("decode_sec", 0.0), final_decode_sec,
            timing.get("trans_sec", 0.0), original,
        )
        self._clog(
            f"[FINAL] seg={segment_id} reason={reason} "
            f"audio=[{audio_start_sec:.3f}, {audio_end_sec:.3f}] "
            f"fsl={timing.get('fsl_sec', 0.0):.3f}s "
            f"encode={encoding_sec:.3f}s "
            f"decode={timing.get('decode_sec', 0.0):.3f}s "
            f"final_decode={final_decode_sec:.3f}s "
            f"trans={timing.get('trans_sec', 0.0):.3f}s | "
            f"text='{original}' | translation='{translation}'"
        )
        await self.send_message("final", **{k: v for k, v in payload.items() if k != "type"})
        self.segment_audio_start_sec = audio_end_sec
        self.pending_audio_end_sec = None

    async def finish_streaming(self):
        self.pending_audio_end_sec = self.current_time
        await super().finish_streaming()

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            logger.info("New connection from %s", remote_addr)
            await self.send_message("hello", message="Qwen3-ASR Streaming Server (FCL)")
            if self._shared_http_session is not None:
                self.http_session = self._shared_http_session
            else:
                timeout = aiohttp.ClientTimeout(total=3)
                self.http_session = aiohttp.ClientSession(timeout=timeout)

            async for message in self.websocket:
                if isinstance(message, bytes):
                    if self.running:
                        await self.process_audio_chunk(message)
                else:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "start":
                            self.client_lang = data.get("lang", "auto")
                            self.client_target_lang = data.get("targetLang", "")
                            log_path = data.get("logPath")
                            if log_path:
                                self._log_output_path = log_path
                            logger.info(
                                "Received start: lang=%s, targetLang=%s",
                                self.client_lang,
                                self.client_target_lang,
                            )
                            self.init_streaming_state()
                            if log_path:
                                self._log_output_path = log_path  # init_streaming_state resets log, restore path
                                self._clog(f"[CONNECTION] start lang={self.client_lang} targetLang={self.client_target_lang}")
                            self.running = True
                            await self.send_message("ready", message="Ready to receive audio")

                        elif msg_type in ("stop", "finish"):
                            logger.info("Received %s command", msg_type)
                            self.running = False
                            await self.finish_streaming()

                            if msg_type == "stop":
                                break

                            self.init_streaming_state()
                            self.running = True

                        elif msg_type == "pair_host":
                            room_id = (data.get("roomId") or "").strip()
                            my_lang = (data.get("myLang") or "").strip()
                            target_lang = (data.get("targetLang") or "").strip()
                            mode = (data.get("mode") or "mode-2").strip()
                            if not room_id or not my_lang or not target_lang:
                                await self.send_message(
                                    "pair_error",
                                    roomId=room_id,
                                    message="invalid_pair_host_payload",
                                )
                            else:
                                await self.pairing_hub.register_host(
                                    self.websocket, room_id, my_lang, target_lang, mode
                                )

                        elif msg_type == "pair_join":
                            room_id = (data.get("roomId") or "").strip()
                            guest_my_lang = (data.get("myLang") or "").strip()
                            if not room_id:
                                await self.send_message(
                                    "pair_error",
                                    roomId=room_id,
                                    message="missing_room_id",
                                )
                            elif not guest_my_lang:
                                await self.send_message(
                                    "pair_error",
                                    roomId=room_id,
                                    message="missing_guest_my_lang",
                                )
                            else:
                                await self.pairing_hub.join_room(
                                    self.websocket, room_id, guest_my_lang
                                )

                        elif msg_type == "log":
                            # Evaluation mode does not persist separate log files.
                            pass

                        elif msg_type == "pair_leave":
                            await self.pairing_hub.leave(self.websocket)

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON: %s", message[:100])

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by client")
        except Exception as exc:
            logger.error("Error in handler: %s", exc)
            traceback.print_exc()
        finally:
            was_running = self.running
            self.running = False
            if was_running:
                await self.finish_streaming()
            self._clog("[CONNECTION] closed")
            self._flush_connection_log()
            if self.http_session is not None and self._shared_http_session is None:
                try:
                    await self.http_session.close()
                except Exception:
                    pass
                self.http_session = None
            await self.pairing_hub.leave(self.websocket)
            logger.info("Connection closed")


class FCLStreamingServer(base_server.Qwen3ASRStreamingServer):
    def __init__(self, config):
        super().__init__(config)
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        timeout = aiohttp.ClientTimeout(total=3)
        self._http_session = aiohttp.ClientSession(timeout=timeout)
        try:
            await super().start()
        finally:
            await self._http_session.close()
            self._http_session = None

    async def handle_connection(self, websocket):
        async with self.connection_lock:
            self.active_connections += 1
            if self.idle_task and not self.idle_task.done():
                self.idle_task.cancel()
            logger.info("Client connected (%s)", self.active_connections)

        try:
            handler = FCLStreamingHandler(websocket, self.asr, self.config, self.pairing_hub, http_session=self._http_session, vad_model_bytes=self.vad_model_bytes)
            await handler.handle()
        finally:
            async with self.connection_lock:
                self.active_connections -= 1
                logger.info("Client disconnected (%s)", self.active_connections)
                if self.active_connections == 0:
                    self._restart_idle_timer()


def main():
    import argparse as _argparse
    base_parser = base_server.parse_args.__wrapped__ if hasattr(base_server.parse_args, '__wrapped__') else None

    # base_server.parse_args()를 먼저 처리하되, --log-file 추가
    # base_server 파서를 직접 재구성하지 않고 known_args 방식으로 확장
    extra_parser = _argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="서버 로그를 저장할 파일 경로 (예: results/fcl/fcl_server_log_v5/server.log)",
    )
    extra_args, remaining = extra_parser.parse_known_args()

    import sys as _sys
    _sys.argv = [_sys.argv[0]] + remaining
    args = base_server.parse_args()

    if extra_args.log_file:
        log_path = Path(extra_args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logger.info("Server log file: %s", log_path.resolve())

    config = base_server.StreamingConfig(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        chunk_size_sec=args.chunk_size,
        host=args.host,
        port=args.port,
        no_idle_shutdown=args.no_idle_shutdown,
        idle_shutdown_sec=args.idle_shutdown_sec,
        close_timeout=args.close_timeout,
        beam_size=args.beam_size,
        no_lora=not args.lora,
        adapter_en=args.adapter_en,
        adapter_ko=args.adapter_ko,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=args.enforce_eager,
        enable_dot_commit=args.enable_dot_commit,
        restrict_languages=not args.no_restrict_languages,
        enable_correction=args.correction,
        correction_model=args.correction_model,
        api_key=args.api_key,
        enable_gpt_translation=args.gpt_translation,
        translation_model=args.translation_model,
    )

    server = FCLStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
