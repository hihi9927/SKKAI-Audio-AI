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
        self._connection_log = []

    def _stream_elapsed_sec(self) -> float:
        return time.perf_counter() - self.stream_start_perf

    async def _asr_streaming_transcribe(self, chunk, slot_key=None):
        await super()._asr_streaming_transcribe(chunk, slot_key)

    async def _asr_finish_streaming(self, slot_key=None):
        key = slot_key if slot_key is not None else self.active_slot
        t0 = time.perf_counter()
        await super()._asr_finish_streaming(slot_key)
        self._slot_final_decode_sec[key] = self._slot_final_decode_sec.get(key, 0.0) + (time.perf_counter() - t0)

    async def _process_slot_updates(self, slot_key=None):
        key = slot_key if slot_key is not None else self.active_slot
        if key not in self._slot_seg_detected:
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
                "[SEG] slot=%s fsl=%.3fs decode_start=%.3fs audio_pos=%.3fs",
                key, elapsed,
                decode_start_elapsed if decode_start_elapsed is not None else -1.0,
                self.current_time,
            )
            self._clog(
                f"[SEG] slot={key} fsl={elapsed:.3f}s "
                f"decode_start={decode_start_elapsed:.3f}s audio_pos={self.current_time:.3f}s"
                if decode_start_elapsed is not None else
                f"[SEG] slot={key} fsl={elapsed:.3f}s audio_pos={self.current_time:.3f}s"
            )
        await super()._process_slot_updates(slot_key)

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
        trans_sec = self._stream_elapsed_sec() - t0

        return translation, detected_lang, {
            "trans_sec": trans_sec,
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
        logger.info("[VAD_COMMIT] audio_end_sec=%.3fs → speech_end_sec=%.3fs", audio_end_sec, speech_end_sec)
        self._clog(f"[VAD_COMMIT] audio_end_sec={audio_end_sec:.3f}s → speech_end_sec={speech_end_sec:.3f}s")
        self.pending_audio_end_sec = speech_end_sec

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

        # encode / decode / FSL
        seg_info = self._slot_seg_detected.pop(slot_key, None)
        if seg_info:
            fsl_sec = seg_info["elapsed_sec"]
            timing["fsl_sec"] = fsl_sec
            # encode_sec: stream 시작(오디오 입력) ~ model.generate() 호출 직전
            # decode_sec: model.generate() 호출 ~ SEG 감지
            d_start = seg_info.get("decode_start_elapsed_sec")
            if d_start is not None:
                timing["encode_sec"] = d_start
                timing["decode_sec"] = max(0.0, fsl_sec - d_start)
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
        beam_size=args.beam_size,
        no_lora=not args.lora,
        adapter_en=args.adapter_en,
        adapter_ko=args.adapter_ko,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=args.enforce_eager,
        enable_dot_commit=args.enable_dot_commit,
        restrict_languages=not args.no_restrict_languages,
        enable_correction=not args.no_correction,
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
