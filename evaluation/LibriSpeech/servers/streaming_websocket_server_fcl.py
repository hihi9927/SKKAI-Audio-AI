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
import re
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
    def __init__(self, websocket, asr_model, config, pairing_hub):
        super().__init__(websocket, asr_model, config, pairing_hub)
        self.stream_start_perf = time.perf_counter()
        self.next_segment_id = 1
        self.segment_audio_start_sec = 0.0
        self.pending_audio_end_sec: Optional[float] = None

    def init_streaming_state(self):
        super().init_streaming_state()
        self.stream_start_perf = time.perf_counter()
        self.next_segment_id = 1
        self.segment_audio_start_sec = 0.0
        self.pending_audio_end_sec = None

    def _stream_elapsed_sec(self) -> float:
        return time.perf_counter() - self.stream_start_perf

    async def _translate_with_metadata(
        self,
        text: str,
        target_lang: str,
        audio_end_sec: float,
    ) -> tuple[str, str, dict[str, Any]]:
        translate_start_elapsed_sec = self._stream_elapsed_sec()
        translate_start_wall = _utc_now_iso()
        translation, detected_lang = await base_server.google_translate_async(
            self.http_session,
            text,
            target_lang,
        )
        translate_done_elapsed_sec = self._stream_elapsed_sec()
        translate_done_wall = _utc_now_iso()

        return translation, detected_lang, {
            "translate_target_lang": target_lang,
            "translate_started_elapsed_sec": translate_start_elapsed_sec,
            "translate_done_elapsed_sec": translate_done_elapsed_sec,
            "translate_started_wall_utc": translate_start_wall,
            "translate_done_wall_utc": translate_done_wall,
            "translation_latency_sec": translate_done_elapsed_sec - translate_start_elapsed_sec,
            "fcl_sec": translate_done_elapsed_sec - audio_end_sec,
        }

    async def _emit_final_segment(
        self,
        *,
        slot_key: str,
        original: str,
        translation: str,
        language: str,
        reason: str,
        timing: dict[str, Any],
        audio_end_sec: float,
    ) -> None:
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
            "final_payload_wall_utc": _utc_now_iso(),
            **timing,
        }
        await self.send_message("final", **{k: v for k, v in payload.items() if k != "type"})
        self.segment_audio_start_sec = audio_end_sec
        self.pending_audio_end_sec = None

    async def flush_uncommitted(self, force=False, reason="flush", slot_key: Optional[str] = None):
        slot = self._slot(slot_key)
        async with self.flush_lock:
            async with self.asr_lock:
                state = slot["state"]
                current_text = (state.text or "").strip() if state else ""
                current_lang = slot["last_text_lang"] or ""
            uncommitted = current_text[slot["committed_len"]:].strip()
            if not uncommitted:
                return
            if not force and len(uncommitted) < 2:
                logger.info("[timeout-skip] reason=%s too short: '%s'", reason, uncommitted)
                return

            audio_end_sec = self.pending_audio_end_sec if self.pending_audio_end_sec is not None else self.current_time
            translation, detected_lang, timing = await self._translate_with_metadata(
                uncommitted,
                self.client_target_lang,
                audio_end_sec,
            )
            logger.info(
                "[translate-flush] reason=%s sentence='%s' tl=%s detected=%s translation='%s'",
                reason,
                uncommitted,
                self.client_target_lang,
                detected_lang,
                translation,
            )
            if detected_lang == self.client_target_lang:
                translation, _, timing = await self._translate_with_metadata(
                    uncommitted,
                    self.client_lang,
                    audio_end_sec,
                )
                logger.info(
                    "[translate-flush-flip] reason=%s tl=%s translation='%s'",
                    reason,
                    self.client_lang,
                    translation,
                )

            final_lang = detected_lang or base_server.lang_to_code(current_lang)
            commit_reason = "vad" if reason.startswith("vad") else "seg"
            await self._emit_final_segment(
                slot_key=slot_key or self.active_slot,
                original=uncommitted,
                translation=translation,
                language=final_lang,
                reason=commit_reason,
                timing=timing,
                audio_end_sec=audio_end_sec,
            )
            slot["committed_len"] = len(current_text)
            slot["committed_prefix"] = current_text

    async def _process_slot_updates(self, slot_key: str, emit_partial: bool = True):
        slot = self._slot(slot_key)
        state = slot["state"]
        current_text = (state.text or "").strip()
        current_lang = state.language or ""
        if not current_text or current_text == slot["last_text"]:
            return

        slot["last_text"] = current_text
        slot["last_text_lang"] = current_lang

        if emit_partial:
            await self.send_message(
                "partial",
                original=current_text,
                last_translation="",
            )
            logger.info("[partial] slot=%s text=%s...", slot_key, current_text[:80])

        uncommitted = current_text[slot["committed_len"]:]
        sentences_to_commit = []
        remaining = uncommitted

        while True:
            match = re.search(r"[.?!\u3002\uff1f\uff01]\s+", remaining)
            if not match:
                break
            after = remaining[match.end():]
            if after.strip():
                sentence = remaining[:match.end()].strip()
                if sentence:
                    sentences_to_commit.append(sentence)
                remaining = after
            else:
                break

        if not sentences_to_commit:
            return

        translated_payloads = []
        for sentence in sentences_to_commit:
            audio_end_sec = self.current_time
            translation, detected_lang, timing = await self._translate_with_metadata(
                sentence,
                self.client_target_lang,
                audio_end_sec,
            )
            logger.info(
                "[translate-sentence] sentence='%s' tl=%s detected=%s translation='%s'",
                sentence,
                self.client_target_lang,
                detected_lang,
                translation,
            )
            if detected_lang == self.client_target_lang:
                translation, _, timing = await self._translate_with_metadata(
                    sentence,
                    self.client_lang,
                    audio_end_sec,
                )
                logger.info(
                    "[translate-sentence-flip] tl=%s translation='%s'",
                    self.client_lang,
                    translation,
                )
            final_lang = detected_lang or base_server.lang_to_code(current_lang)
            translated_payloads.append({
                "original": sentence,
                "translation": translation,
                "language": final_lang,
                "timing": timing,
            })

        ready_to_emit = []
        async with self.flush_lock:
            async with self.asr_lock:
                latest_state = slot["state"]
                latest_text = (latest_state.text or "").strip() if latest_state else ""

            cursor = slot["committed_len"]
            tail = latest_text[cursor:]

            for payload in translated_payloads:
                sentence = payload["original"]
                stripped_tail = tail.lstrip()
                leading_ws = len(tail) - len(stripped_tail)
                if not stripped_tail.startswith(sentence):
                    break
                cursor += leading_ws + len(sentence)
                tail = latest_text[cursor:]
                ready_to_emit.append(payload)

            if ready_to_emit:
                slot["committed_len"] = cursor
                slot["committed_prefix"] = latest_text[:slot["committed_len"]]

        for payload in ready_to_emit:
            await self._emit_final_segment(
                slot_key=slot_key,
                original=payload["original"],
                translation=payload["translation"],
                language=payload["language"],
                reason="seg",
                timing=payload["timing"],
                audio_end_sec=self.current_time,
            )

    async def process_audio_chunk(self, audio_data: bytes):
        if not audio_data:
            return

        chunk = base_server.np.frombuffer(audio_data, dtype=base_server.np.int16).astype(base_server.np.float32) / 32768.0
        if chunk.size == 0:
            return

        chunk_base_sample = self.sample_cursor
        self.sample_cursor += chunk.size
        vad_end_local_indices: list[int] = []

        if self.vad_enabled and self.vad_iterator is not None:
            try:
                offset = 0
                while offset + base_server.VAD_WINDOW_SIZE_SAMPLES <= chunk.size:
                    window = chunk[offset:offset + base_server.VAD_WINDOW_SIZE_SAMPLES]
                    speech_dict = self.vad_iterator(
                        base_server.torch.from_numpy(window),
                        return_seconds=False,
                    )
                    if speech_dict is not None and "end" in speech_dict:
                        end_sample = self.sample_cursor - (chunk.size - offset - base_server.VAD_WINDOW_SIZE_SAMPLES)
                        local_idx = int(end_sample - chunk_base_sample)
                        local_idx = max(0, min(int(chunk.size), local_idx))
                        if not vad_end_local_indices or local_idx > vad_end_local_indices[-1]:
                            vad_end_local_indices.append(local_idx)
                        logger.info("[vad] speech end detected; target_samples=%s", end_sample)
                    offset += base_server.VAD_WINDOW_SIZE_SAMPLES
            except Exception as exc:
                logger.warning("[vad] error, disabling for this session: %s", exc)
                self.vad_enabled = False

        self.current_time += chunk.size / base_server.SAMPLING_RATE
        seg_start = 0
        for local_cut in vad_end_local_indices:
            if local_cut <= seg_start:
                continue

            pre_chunk = chunk[seg_start:local_cut]
            if pre_chunk.size > 0:
                await self._asr_streaming_transcribe(pre_chunk, self.active_slot)
                await self._process_slot_updates(self.active_slot, emit_partial=True)

            target_sample = chunk_base_sample + local_cut
            target_audio_end_sec = target_sample / base_server.SAMPLING_RATE
            self.pending_audio_end_sec = target_audio_end_sec
            logger.info(
                "[vad-commit] target_samples=%s target_audio_end_sec=%.3f processed=%s active=%s",
                target_sample,
                target_audio_end_sec,
                self.asr_processed_cursor,
                self.active_slot,
            )

            old_active = self.active_slot
            self.active_slot, self.standby_slot = self.standby_slot, self.active_slot
            self.state = self.stream_slots[self.active_slot]["state"]
            logger.info(
                "[slot-switch] old_active=%s new_active=%s new_standby=%s",
                old_active,
                self.active_slot,
                self.standby_slot,
            )

            await self.flush_uncommitted(force=True, reason="vad", slot_key=old_active)
            self._reset_stream_slot(self.standby_slot)

            seg_start = local_cut

        tail_chunk = chunk[seg_start:]
        if tail_chunk.size > 0:
            await self._asr_streaming_transcribe(tail_chunk, self.active_slot)
            await self._process_slot_updates(self.active_slot, emit_partial=True)

    async def finish_streaming(self):
        if not self.stream_slots:
            return
        await self._asr_finish_streaming(self.active_slot)
        self.pending_audio_end_sec = self.current_time
        await self.flush_uncommitted(force=True, reason="finish", slot_key=self.active_slot)

    async def handle(self):
        try:
            remote_addr = self.websocket.remote_address
            logger.info("New connection from %s", remote_addr)
            await self.send_message("hello", message="Qwen3-ASR Streaming Server (FCL)")
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
                            logger.info(
                                "Received start: lang=%s, targetLang=%s",
                                self.client_lang,
                                self.client_target_lang,
                            )
                            self.init_streaming_state()
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
            if self.http_session is not None:
                try:
                    await self.http_session.close()
                except Exception:
                    pass
                self.http_session = None
            await self.pairing_hub.leave(self.websocket)
            logger.info("Connection closed")


class FCLStreamingServer(base_server.Qwen3ASRStreamingServer):
    async def handle_connection(self, websocket):
        async with self.connection_lock:
            self.active_connections += 1
            if self.idle_task and not self.idle_task.done():
                self.idle_task.cancel()
            logger.info("Client connected (%s)", self.active_connections)

        try:
            handler = FCLStreamingHandler(websocket, self.asr, self.config, self.pairing_hub)
            await handler.handle()
        finally:
            async with self.connection_lock:
                self.active_connections -= 1
                logger.info("Client disconnected (%s)", self.active_connections)
                if self.active_connections == 0:
                    self._restart_idle_timer()


def main():
    args = base_server.parse_args()
    config = base_server.StreamingConfig(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        chunk_size_sec=args.chunk_size,
        host=args.host,
        port=args.port,
        no_idle_shutdown=args.no_idle_shutdown,
        idle_shutdown_sec=args.idle_shutdown_sec,
    )

    server = FCLStreamingServer(config)
    server.init_model()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
