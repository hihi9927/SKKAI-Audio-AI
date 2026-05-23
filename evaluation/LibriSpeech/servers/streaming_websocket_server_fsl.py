#!/usr/bin/env python3
"""Evaluation-only Qwen3 streaming server with FSL instrumentation.

This server reuses the production streaming server implementation but augments
final segment payloads with timing metadata that can be used to compute FSL.
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


class FSLStreamingHandler(base_server.Qwen3ASRStreamingHandler):
    def __init__(self, websocket, asr_model, config, pairing_hub, http_session=None, vad_model_bytes=None, corrector=None, gpt_translator=None):
        super().__init__(websocket, asr_model, config, pairing_hub, vad_model_bytes=vad_model_bytes, corrector=corrector, gpt_translator=gpt_translator)
        self._shared_http_session = http_session
        self.stream_start_perf = time.perf_counter()
        self.next_segment_id = 1
        self.segment_audio_start_sec = 0.0
        self.pending_audio_end_sec: Optional[float] = None
        self._effective_audio_end_sec: Optional[float] = None
        self._slot_final_decode_sec: dict[str, float] = {}  # _asr_finish_streaming 누적
        self._slot_seg_detected: dict[str, dict] = {}       # 첫 SEG 감지 시점 {elapsed_sec, audio_sec, decode_start_elapsed_sec}
        self._slot_chunk_encode_log: dict[str, list] = {}   # 청크별 타이밍 로그
        self._slot_last_emitted_chunk_id: dict[str, int] = {}  # 마지막 payload에 포함된 chunk_id (중복 방지)
        # _asr_streaming_transcribe 호출 시작 시간 — on_seg 내부에서 조기 로깅에 사용
        self._slot_active_transcribe_t0: dict[str, float] = {}
        # VAD 트리거 시점 (speech_end + 800ms) — VAD 커밋 시 payload에 포함
        self._pending_vad_trigger_sec: Optional[float] = None
        # VAD 트리거 wall-clock elapsed — FSL = trans_end_elapsed - vad_trigger_elapsed
        self._pending_vad_trigger_elapsed: Optional[float] = None
        # SEG commit deferred emit queue (generate 완료 후 total_tokens 확정 뒤 flush)
        self._deferred_seg_emits: list = []
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
        self._slot_final_decode_sec = {}
        self._slot_seg_detected = {}
        self._slot_chunk_encode_log = {}
        self._slot_last_emitted_chunk_id = {}
        self._slot_active_transcribe_t0 = {}
        self._pending_vad_trigger_sec = None
        self._pending_vad_trigger_elapsed = None
        self._deferred_seg_emits = []
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

        t0 = time.perf_counter()
        ts_elapsed = round(t0 - self.stream_start_perf, 4)  # wall-clock elapsed (plot x축 기준)
        # 이전 슬롯에서 남은 stale _decode_start_perf가 t0 < dp 윈도우 체크를 통과해
        # encode_sec/decode_sec가 null로 기록되는 문제 방지
        state._decode_start_perf = None
        # _process_slot_updates(on_seg) 내부에서 조기 로깅이 가능하도록 t0 공유
        self._slot_active_transcribe_t0[key] = t0
        # generate 완료 후 _last_chunk_new_tokens를 읽기 위해 state 레퍼런스 보존
        # (슬롯이 리셋돼 dict가 바뀌어도 객체는 유지됨)
        state_ref = state
        await super()._asr_streaming_transcribe(chunk, slot_key)
        # generate() 완료 직후 elapsed (base server가 _last_generate_end_time에 저장)
        # _flush_pending_gpt_tasks 이전 시점을 써야 remaining_decode_sec이 정확하게 계산됨
        generate_end_elapsed = self._last_generate_end_time - self.stream_start_perf
        total_tokens = getattr(state_ref, "_last_chunk_new_tokens", None)
        if self._deferred_seg_emits:
            await self._flush_deferred_seg_emits(total_tokens or 0, generate_end_elapsed)
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
                slot = self._slot(slot_key)
                raw_text = (slot["state"].text or "").strip()
                # on_seg는 model.generate() 내부에서 호출 → state.text에 <SEG> 있음.
                # post-transcribe path(base server line 599)는 generate 미실행 시에도 호출되므로
                # <SEG> 없으면 조기 기록을 건너뜀 — decode_start_elapsed_sec=None 오염 방지.
                if "<SEG>" in raw_text:
                    elapsed = self._stream_elapsed_sec()
                    decode_start_perf = getattr(slot["state"], "_decode_start_perf", None)
                    decode_start_elapsed = (decode_start_perf - self.stream_start_perf) if decode_start_perf else None
                    seg_token_idx = getattr(slot["state"], "_seg_token_idx", None)
                    chunk_audio_start = max(0.0, self.current_time - self.config.chunk_size_sec)
                    self._slot_seg_detected[key] = {
                        "elapsed_sec": elapsed,
                        "audio_sec": self.current_time,
                        "decode_start_elapsed_sec": decode_start_elapsed,
                        "seg_token_idx": seg_token_idx,
                        "chunk_audio_start": chunk_audio_start,
                        "chunk_size_sec": self.config.chunk_size_sec,
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
            elif t0 is None:
                # VAD/finish/timeout path: elapsed/decode_start는 stale → encode_sec/decode_sec 기록 금지
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
        if msg_type == "partial":
            text = (kwargs.get("original") or "").strip()
            if text:
                logger.info("[PARTIAL] t=%.3fs | %s", self.current_time, text)
                self._clog(f"[PARTIAL] t={self.current_time:.3f}s | {text}")
        await super().send_message(msg_type, **kwargs)

    async def _translate(
        self, text: str, target_lang: str, audio_end_sec: Optional[float] = None
    ) -> tuple[str, str, dict]:
        if self.pending_audio_end_sec is not None:
            # VAD/finish: pending_audio_end_sec = speech_end (accurate)
            self._effective_audio_end_sec = self.pending_audio_end_sec
        # SEG path: _effective_audio_end_sec는 None으로 유지 → _emit_final_payload에서 current_time
        # 사용. 실제 audio_end는 _flush_deferred_seg_emits에서 token ratio로 확정.
        return await self._translate_with_metadata(text, target_lang)

    async def _correct_and_translate(self, text: str, current_lang: str, audio_end_sec: float):
        if not self.gpt_translator:
            return await super()._correct_and_translate(text, current_lang, audio_end_sec)
        t0 = self._stream_elapsed_sec()
        corrected, translation, lang_code, extra = await super()._correct_and_translate(text, current_lang, audio_end_sec)
        t1 = self._stream_elapsed_sec()
        extra["trans_sec"] = round(t1 - t0, 4)
        extra["_trans_end_elapsed"] = t1
        extra["_trans_start_elapsed"] = t0
        return corrected, translation, lang_code, extra

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
        self._pending_vad_trigger_elapsed = self._stream_elapsed_sec()
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
        # - fsl_sec (SEG): generate 완료 후 _flush_deferred_seg_emits에서 token 기반으로 확정
        # - fsl_sec (VAD): trans_end_elapsed - vad_trigger_elapsed (VAD trigger 기준)
        # - fsl_sec (finish): final_decode_sec + trans_sec
        seg_info = self._slot_seg_detected.pop(slot_key, None)
        trans_end_elapsed = timing.pop("_trans_end_elapsed", None)
        trans_start_elapsed = timing.pop("_trans_start_elapsed", None)

        # 슬롯 실제 오디오 시작 시점
        slot_audio_start = self._slot_audio_start_sec.pop(slot_key, None)
        if slot_audio_start is not None:
            timing["slotAudioStartSec"] = round(slot_audio_start, 3)

        # VAD 트리거 시점 (0.8초 침묵 후) — VAD 커밋에서만 기록
        vad_trigger_sec = self._pending_vad_trigger_sec
        vad_trigger_elapsed = self._pending_vad_trigger_elapsed
        self._pending_vad_trigger_sec = None
        self._pending_vad_trigger_elapsed = None
        if reason == "vad" and vad_trigger_sec is not None:
            timing["vad_trigger_sec"] = round(vad_trigger_sec, 3)

        # 이전 슬롯이 empty flush였을 때 gap bar 시각화를 위해 speech_end 전달
        if self._prev_slot_speech_end_sec is not None:
            timing["prevSlotSpeechEndSec"] = round(self._prev_slot_speech_end_sec, 3)
            self._prev_slot_speech_end_sec = None

        is_seg = reason not in ("vad", "finish")

        if is_seg and seg_info is not None:
            # ── SEG path ─────────────────────────────────────────────────────
            # encode/decode 기록; FSL·audioEndSec는 _flush_deferred_seg_emits에서 확정
            d_start = seg_info.get("decode_start_elapsed_sec")
            if d_start is not None:
                timing["encode_sec"] = d_start
                timing["decode_sec"] = round(max(0.0, seg_info["elapsed_sec"] - d_start), 4)
            timing["seg_audio_sec"] = round(seg_info["audio_sec"], 3)
            # 임시 FSL (deferred flush에서 token-based로 덮어씀)
            effective_ae = (
                self._effective_audio_end_sec
                if self._effective_audio_end_sec is not None
                else audio_end_sec
            )
            if trans_end_elapsed is not None:
                timing["fsl_sec"] = round(max(0.0, trans_end_elapsed - effective_ae), 4)
        elif seg_info is not None:
            # ── VAD/finish path + SEG 감지 ────────────────────────────────────
            timing["seg_audio_sec"] = round(seg_info["audio_sec"], 3)
            seg_elapsed = seg_info.get("elapsed_sec")
            if seg_elapsed is not None:
                if trans_start_elapsed is not None:
                    pre_trans = round(max(0.0, trans_start_elapsed - seg_elapsed), 4)
                    if pre_trans > 0.001:
                        timing["pre_trans_sec"] = pre_trans
                if trans_end_elapsed is not None:
                    timing["fsl_sec"] = round(max(0.0, trans_end_elapsed - seg_elapsed), 4)
            if "fsl_sec" not in timing:
                timing["fsl_sec"] = round(final_decode_sec + timing.get("trans_sec", 0.0), 4)

        # VAD/finish path FSL (seg 있건 없건 VAD는 trigger 기준으로 통일)
        if not is_seg:
            if reason == "vad" and trans_end_elapsed is not None and vad_trigger_elapsed is not None:
                timing["fsl_sec"] = round(max(0.0, trans_end_elapsed - vad_trigger_elapsed), 4)
            elif "fsl_sec" not in timing:
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
        self.segment_audio_start_sec = audio_end_sec
        self.pending_audio_end_sec = None
        self._committed_utterance_count += 1
        if self.gpt_translator and original and translation:
            self._segment_history.append((original, translation))

        if is_seg:
            # SEG: generate() 완료 후 total_tokens로 audio_end/FSL 확정 (defer)
            # DOT: seg_info is None이지만 char ratio로 token position 근사 (공평한 FSL 측정)
            dot_char_ratio = None
            dot_chunk_audio_start = None
            if seg_info is None:  # DOT commit: char ratio 기반 역추적
                slot = self._slot(slot_key)
                full_raw = (slot["state"].text or "").strip() if slot["state"] else ""
                committed_raw_len = slot.get("committed_len", 0)
                if full_raw:
                    dot_char_ratio = min(1.0, committed_raw_len / len(full_raw))
                dot_chunk_audio_start = max(0.0, self.current_time - self.config.chunk_size_sec)
            self._deferred_seg_emits.append({
                "payload": payload,
                "_trans_end_elapsed": trans_end_elapsed,
                "_trans_start_elapsed": trans_start_elapsed,
                "_seg_token_idx": seg_info.get("seg_token_idx") if seg_info else None,
                "_dot_char_ratio": dot_char_ratio,
                "_chunk_audio_start": seg_info.get("chunk_audio_start") if seg_info else dot_chunk_audio_start,
                "_chunk_size_sec": (
                    seg_info.get("chunk_size_sec", self.config.chunk_size_sec)
                    if seg_info else self.config.chunk_size_sec
                ),
                "_encoding_sec": encoding_sec,
                "_final_decode_sec": final_decode_sec,
            })
            return

        # VAD/finish: 즉시 전송
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

    async def _flush_deferred_seg_emits(self, total_tokens: int, generate_end_elapsed: float = None) -> None:
        """generate() 완료 후 SEG deferred emit을 token 비율로 audio_end/FSL 확정 후 전송."""
        emits = self._deferred_seg_emits
        self._deferred_seg_emits = []
        # 연속 SEG commit 간 audio_start 전파용: 이전 emit의 new_audio_end를 다음 emit의
        # audioStartSec / slotAudioStartSec에 기록해 추측 없이 정확한 오디오 구간 복원
        next_audio_start: float | None = None
        for item in emits:
            payload = item["payload"]
            trans_end_elapsed = item["_trans_end_elapsed"]
            trans_start_elapsed = item.get("_trans_start_elapsed")
            seg_token_idx = item["_seg_token_idx"]
            dot_char_ratio = item.get("_dot_char_ratio")
            # DOT commit: char ratio로 approximate token index 계산 (SEG와 동일한 역추적)
            if seg_token_idx is None and dot_char_ratio is not None and total_tokens > 0:
                seg_token_idx = int(dot_char_ratio * total_tokens)
            chunk_audio_start = item["_chunk_audio_start"]
            chunk_size_sec = item["_chunk_size_sec"]
            encoding_sec = item["_encoding_sec"]
            final_decode_sec = item["_final_decode_sec"]

            # generate() 완료 시점 기준으로 trans_sec 재계산:
            # on_seg 콜백 내부에서 기록된 trans_sec은 generate()가 이벤트 루프를 점유한 채
            # HTTP 응답 처리가 밀려 remaining_decode 시간을 포함함. generate_end 이후 실제
            # 번역 응답 처리 시간만 trans_sec에 기록하고, SEG→generate 완료 구간은
            # remaining_decode_sec으로 분리 기록.
            if generate_end_elapsed is not None:
                seg_det_x = encoding_sec + payload.get("decode_sec", 0.0)
                remaining = round(max(0.0, generate_end_elapsed - seg_det_x), 4)
                if remaining > 0:
                    payload["remaining_decode_sec"] = remaining
                if trans_end_elapsed is not None:
                    payload["trans_sec"] = round(max(0.0, trans_end_elapsed - generate_end_elapsed), 4)

            # 연속 SEG commit: 이전 emit에서 확정된 audio_end를 이번 emit의 audioStartSec로 patch
            if next_audio_start is not None:
                payload["audioStartSec"] = round(next_audio_start, 3)
                payload["start"] = base_server.format_time(next_audio_start)
                payload["slotAudioStartSec"] = round(next_audio_start, 3)

            # token-based audio_end 역추적: chunk 시작 + (k/n) * chunk_duration
            if seg_token_idx is not None and chunk_audio_start is not None and total_tokens > 0:
                seg_ratio = min(1.0, seg_token_idx / total_tokens)
                new_audio_end = max(0.0, chunk_audio_start + seg_ratio * chunk_size_sec)
                payload["audioEndSec"] = round(new_audio_end, 3)
                payload["end"] = base_server.format_time(new_audio_end)
                if trans_end_elapsed is not None:
                    payload["fsl_sec"] = round(max(0.0, trans_end_elapsed - new_audio_end), 4)
                # 다음 세그먼트의 audioStartSec / slotAudioStartSec 전파
                self.segment_audio_start_sec = new_audio_end
                next_audio_start = new_audio_end
                logger.info(
                    "[FLUSH-SEG] seg=%d tok=%d/%d ratio=%.3f audio_end=%.3fs fsl=%.3fs",
                    payload["segmentId"], seg_token_idx, total_tokens, seg_ratio,
                    new_audio_end, payload.get("fsl_sec", 0.0),
                )
            else:
                logger.info(
                    "[FLUSH-SEG] seg=%d no-token-info → audio_end=%.3fs fsl=%.3fs",
                    payload["segmentId"], payload.get("audioEndSec", 0.0), payload.get("fsl_sec", 0.0),
                )

            logger.info(
                "[FINAL] seg=%d reason=%s audio=[%.3f, %.3f] "
                "fsl=%.3fs encode=%.3fs decode=%.3fs final_decode=%.3fs trans=%.3fs | text='%s'",
                payload["segmentId"], payload["commitReason"],
                payload["audioStartSec"], payload.get("audioEndSec", 0.0),
                payload.get("fsl_sec", 0.0), encoding_sec,
                payload.get("decode_sec", 0.0), final_decode_sec,
                payload.get("trans_sec", 0.0), payload["original"],
            )
            self._clog(
                f"[FINAL] seg={payload['segmentId']} reason={payload['commitReason']} "
                f"audio=[{payload['audioStartSec']:.3f}, {payload.get('audioEndSec', 0.0):.3f}] "
                f"fsl={payload.get('fsl_sec', 0.0):.3f}s "
                f"encode={encoding_sec:.3f}s "
                f"decode={payload.get('decode_sec', 0.0):.3f}s "
                f"final_decode={final_decode_sec:.3f}s "
                f"trans={payload.get('trans_sec', 0.0):.3f}s | "
                f"text='{payload['original']}' | translation='{payload['translation']}'"
            )
            await self.send_message("final", **{k: v for k, v in payload.items() if k != "type"})

    async def finish_streaming(self):
        # 미전송 SEG deferred emit을 토큰 정보 없이 best-effort로 먼저 전송
        if self._deferred_seg_emits:
            await self._flush_deferred_seg_emits(0)
        self.pending_audio_end_sec = self.current_time
        await super().finish_streaming()

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            logger.info("New connection from %s", remote_addr)
            await self.send_message(
                "hello",
                message="Qwen3-ASR Streaming Server (FSL)",
                serverConfig={
                    "model": self.config.model_path,
                    "chunk_size_sec": self.config.chunk_size_sec,
                    "enforce_eager": self.config.enforce_eager,
                    "enable_gpt_translation": self.config.enable_gpt_translation,
                    "translation_model": self.config.translation_model,
                    "context_window": self.config.context_window,
                    "enable_correction": self.config.enable_correction,
                    "correction_model": self.config.correction_model,
                },
            )
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


class FSLStreamingServer(base_server.Qwen3ASRStreamingServer):
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
            handler = FSLStreamingHandler(websocket, self.asr, self.config, self.pairing_hub, http_session=self._http_session, vad_model_bytes=self.vad_model_bytes, corrector=self.corrector, gpt_translator=self.gpt_translator)
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
        help="서버 로그를 저장할 파일 경로 (예: results/fsl/fsl_server_log_v5/server.log)",
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
        context_window=args.context_window,
    )

    server = FSLStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
