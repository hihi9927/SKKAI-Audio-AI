#!/usr/bin/env python3
"""
Verify architecture-level VAD behavior using the same Silero VADIterator logic
as the current Qwen3 streaming server.

This script overlays:
1) raw Silero speech probability per 512-sample window
2) architecture "speech end" detection points
3) architecture "vad-commit" / "flush" points
"""

from __future__ import annotations

import asyncio
import argparse
import csv
import json
import subprocess
import re
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

SAMPLING_RATE = 16000
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 800
VAD_SPEECH_PAD_MS = 160
VAD_WINDOW_SIZE_SAMPLES = 512


async def _wait_for_ws_server(
    url: str,
    timeout_sec: float,
    proc: subprocess.Popen | None = None,
) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "websockets package is required for --auto-run-server mode "
            "(pip install websockets)."
        ) from exc

    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited before websocket became ready "
                f"(returncode={proc.returncode})."
            )
        try:
            async with websockets.connect(url, ping_interval=None, ping_timeout=None):
                return
        except Exception as e:  # pragma: no cover - network dependent
            last_error = e
            await asyncio.sleep(0.2)

    raise TimeoutError(
        f"Timed out waiting for server at {url} for {timeout_sec:.1f}s. "
        f"Last error: {last_error}"
    )


async def _stream_audio_to_server(
    audio: np.ndarray,
    server_url: str,
    chunk_size_ms: int,
    start_message: dict | None,
    finish_message: dict | None,
    post_finish_wait_sec: float,
) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "websockets package is required for --auto-run-server mode "
            "(pip install websockets)."
        ) from exc

    chunk_size_samples = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    if chunk_size_samples <= 0:
        raise ValueError("chunk_size_ms must produce at least 1 sample")

    pcm_int16 = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)

    async with websockets.connect(server_url, ping_interval=None, ping_timeout=None) as ws:
        if start_message is not None:
            await ws.send(json.dumps(start_message, ensure_ascii=False))

        for start in range(0, pcm_int16.size, chunk_size_samples):
            chunk = pcm_int16[start : start + chunk_size_samples]
            if chunk.size == 0:
                continue
            await ws.send(chunk.tobytes())
            await asyncio.sleep(chunk_size_ms / 1000.0)

        if finish_message is not None:
            await ws.send(json.dumps(finish_message, ensure_ascii=False))
            if post_finish_wait_sec > 0:
                await asyncio.sleep(post_finish_wait_sec)


def run_server_and_collect_log(
    audio: np.ndarray,
    *,
    server_cmd: str | None,
    server_log_path: Path,
    server_cwd: Path | None,
    server_url: str,
    chunk_size_ms: int,
    startup_timeout_sec: float,
    start_message: dict | None,
    finish_message: dict | None,
    post_finish_wait_sec: float,
    launch_server: bool = True,
) -> None:
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    proc: subprocess.Popen | None = None
    if launch_server:
        if not server_cmd:
            raise ValueError("launch_server=True requires server_cmd")
        server_log_path.write_text("", encoding="utf-8")
        with server_log_path.open("a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                server_cmd,
                shell=True,
                cwd=str(server_cwd) if server_cwd is not None else None,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            try:
                asyncio.run(_wait_for_ws_server(server_url, startup_timeout_sec, proc=proc))
                asyncio.run(
                    _stream_audio_to_server(
                        audio=audio,
                        server_url=server_url,
                        chunk_size_ms=chunk_size_ms,
                        start_message=start_message,
                        finish_message=finish_message,
                        post_finish_wait_sec=post_finish_wait_sec,
                    )
                )
                time.sleep(0.3)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
    else:
        asyncio.run(_wait_for_ws_server(server_url, startup_timeout_sec))
        asyncio.run(
            _stream_audio_to_server(
                audio=audio,
                server_url=server_url,
                chunk_size_ms=chunk_size_ms,
                start_message=start_message,
                finish_message=finish_message,
                post_finish_wait_sec=post_finish_wait_sec,
            )
        )
        time.sleep(0.3)


def run_architecture_vad_trace(audio: np.ndarray, chunk_size_ms: int) -> dict:
    model = load_silero_vad()
    vad_iterator = VADIterator(
        model=model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLING_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )
    vad_iterator.reset_states()

    chunk_size_samples = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    if chunk_size_samples <= 0:
        raise ValueError("chunk_size_ms must produce at least 1 sample")

    sample_cursor = 0
    prob_times_sec: list[float] = []
    probs: list[float] = []
    speech_start_samples: list[int] = []
    speech_end_samples: list[int] = []
    vad_commit_samples: list[int] = []
    vad_flush_samples: list[int] = []

    for start in range(0, audio.size, chunk_size_samples):
        chunk = audio[start : start + chunk_size_samples]
        if chunk.size == 0:
            continue

        chunk_base_sample = sample_cursor
        sample_cursor += int(chunk.size)
        vad_end_local_indices: list[int] = []

        offset = 0
        while offset + VAD_WINDOW_SIZE_SAMPLES <= chunk.size:
            window = chunk[offset : offset + VAD_WINDOW_SIZE_SAMPLES]
            tensor_window = torch.from_numpy(window).float()

            prob = float(model(tensor_window, SAMPLING_RATE).item())
            absolute_window_end = (
                chunk_base_sample + offset + VAD_WINDOW_SIZE_SAMPLES
            )
            prob_times_sec.append(absolute_window_end / SAMPLING_RATE)
            probs.append(prob)

            speech_dict = vad_iterator(tensor_window, return_seconds=False)
            if speech_dict is not None:
                if "start" in speech_dict:
                    speech_start_samples.append(int(speech_dict["start"]))

                if "end" in speech_dict:
                    end_sample = sample_cursor - (
                        chunk.size - offset - VAD_WINDOW_SIZE_SAMPLES
                    )
                    local_idx = int(end_sample - chunk_base_sample)
                    local_idx = max(0, min(int(chunk.size), local_idx))

                    if (
                        not vad_end_local_indices
                        or local_idx > vad_end_local_indices[-1]
                    ):
                        vad_end_local_indices.append(local_idx)
                        speech_end_samples.append(int(end_sample))

            offset += VAD_WINDOW_SIZE_SAMPLES

        for local_cut in vad_end_local_indices:
            target_sample = chunk_base_sample + local_cut
            vad_commit_samples.append(int(target_sample))
            vad_flush_samples.append(int(target_sample))

    return {
        "prob_times_sec": prob_times_sec,
        "probs": probs,
        "speech_start_samples": speech_start_samples,
        "speech_end_samples": speech_end_samples,
        "vad_commit_samples": vad_commit_samples,
        "vad_flush_samples": vad_flush_samples,
    }


def parse_server_log_trace(server_log_path: Path, start_offset: int = 0) -> dict:
    vad_end_samples: list[int] = []
    vad_commit_samples: list[int] = []
    final_vad_count = 0

    end_re = re.compile(r"\[vad\]\s+speech end detected;.*?target_samples=(\d+)")
    commit_re = re.compile(r"\[vad-commit\].*?target_samples=(\d+)")

    with server_log_path.open("r", encoding="utf-8", errors="ignore") as f:
        if start_offset > 0:
            f.seek(start_offset)
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_end = end_re.search(line)
            if m_end:
                vad_end_samples.append(int(m_end.group(1)))
                continue

            m_commit = commit_re.search(line)
            if m_commit:
                vad_commit_samples.append(int(m_commit.group(1)))
                continue

            if "[final-vad]" in line:
                final_vad_count += 1

    return {
        "vad_end_samples": vad_end_samples,
        "vad_commit_samples": vad_commit_samples,
        "final_vad_count": final_vad_count,
    }


def nearest_abs_diffs(reference_samples: list[int], target_samples: list[int]) -> list[int]:
    if not reference_samples or not target_samples:
        return []
    diffs: list[int] = []
    sorted_target = sorted(target_samples)
    for ref in reference_samples:
        diffs.append(min(abs(ref - t) for t in sorted_target))
    return diffs


def draw_plot(
    audio: np.ndarray,
    trace: dict,
    output_png: Path,
    server_trace: dict | None = None,
    hop_length: int = 256,
) -> None:
    duration_sec = audio.size / SAMPLING_RATE
    time_wave = np.linspace(0.0, duration_sec, num=audio.size, endpoint=False)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLING_RATE,
        n_mels=80,
        fmax=8000,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(time_wave, audio, color="gray", linewidth=0.8)
    axes[0].set_title("Audio Waveform")
    axes[0].set_ylabel("Amplitude")

    img = librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=SAMPLING_RATE,
        hop_length=hop_length,
        ax=axes[1],
    )
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    axes[2].plot(
        trace["prob_times_sec"],
        trace["probs"],
        color="red",
        linewidth=1.0,
        label="Raw Probability",
    )
    axes[2].axhline(
        y=VAD_THRESHOLD,
        color="gray",
        linestyle="--",
        label=f"Threshold ({VAD_THRESHOLD})",
    )

    start_secs = [x / SAMPLING_RATE for x in trace["speech_start_samples"]]
    end_secs = [x / SAMPLING_RATE for x in trace["speech_end_samples"]]
    commit_secs = [x / SAMPLING_RATE for x in trace["vad_commit_samples"]]
    flush_secs = [x / SAMPLING_RATE for x in trace["vad_flush_samples"]]

    for i, t in enumerate(start_secs):
        axes[2].axvline(
            x=t,
            color="green",
            linestyle="-",
            linewidth=1.5,
            label="Iterator Start" if i == 0 else None,
        )
    for i, t in enumerate(end_secs):
        axes[2].axvline(
            x=t,
            color="blue",
            linestyle="--",
            linewidth=2.0,
            label="Architecture Speech End" if i == 0 else None,
        )
    for i, t in enumerate(commit_secs):
        axes[2].axvline(
            x=t,
            color="purple",
            linestyle="-.",
            linewidth=1.8,
            label="Architecture VAD Commit" if i == 0 else None,
        )
    for i, t in enumerate(flush_secs):
        axes[2].axvline(
            x=t,
            color="black",
            linestyle=":",
            linewidth=1.8,
            label="Architecture Flush(reason=vad)" if i == 0 else None,
        )

    if server_trace is not None:
        server_end_secs = [x / SAMPLING_RATE for x in server_trace["vad_end_samples"]]
        server_commit_secs = [x / SAMPLING_RATE for x in server_trace["vad_commit_samples"]]

        for i, t in enumerate(server_end_secs):
            axes[2].axvline(
                x=t,
                color="orange",
                linestyle="--",
                linewidth=1.2,
                label="Server Log Speech End" if i == 0 else None,
            )
        for i, t in enumerate(server_commit_secs):
            axes[2].axvline(
                x=t,
                color="cyan",
                linestyle="-.",
                linewidth=1.2,
                label="Server Log VAD Commit" if i == 0 else None,
            )

    axes[2].set_title("Architecture-Level VAD Behavior")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Probability")
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def _sanitize_path_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _stable_file_key(audio_path: Path, dataset_root: Path | None = None) -> str:
    candidate = audio_path
    if dataset_root is not None:
        try:
            candidate = audio_path.resolve().relative_to(dataset_root.resolve())
        except Exception:
            candidate = audio_path
    parts = [_sanitize_path_fragment(p) for p in candidate.with_suffix("").parts]
    return "__".join([p for p in parts if p]) or _sanitize_path_fragment(audio_path.stem)


def _discover_audio_files(
    *,
    audio_path: str | None,
    subset: str | None,
    dataset_root: str,
    audio_glob: str | None,
    max_files: int | None,
) -> list[Path]:
    files: list[Path] = []
    root = Path(dataset_root)

    if audio_path:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"audio file not found: {p}")
        files = [p]
    elif subset:
        subset_dir = root / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"subset directory not found: {subset_dir}")
        files = sorted(subset_dir.rglob("*.flac"))
    elif audio_glob:
        files = sorted(Path().glob(audio_glob))
    else:
        raise ValueError(
            "Provide one input source: audio_path OR --subset OR --audio-glob."
        )

    if max_files is not None and max_files >= 0:
        files = files[:max_files]
    if not files:
        raise ValueError("No audio files found for the selected input options.")
    return files


def _parse_optional_json_arg(raw: str) -> dict | None:
    if raw.strip().lower() == "none":
        return None
    return json.loads(raw)


def _build_comparison(trace: dict, server_trace: dict) -> dict:
    end_diffs = nearest_abs_diffs(
        trace["speech_end_samples"], server_trace["vad_end_samples"]
    )
    commit_diffs = nearest_abs_diffs(
        trace["vad_commit_samples"], server_trace["vad_commit_samples"]
    )
    return {
        "offline_end_count": len(trace["speech_end_samples"]),
        "server_end_count": len(server_trace["vad_end_samples"]),
        "offline_commit_count": len(trace["vad_commit_samples"]),
        "server_commit_count": len(server_trace["vad_commit_samples"]),
        "server_final_vad_count": server_trace["final_vad_count"],
        "end_nearest_abs_diff_samples": end_diffs,
        "commit_nearest_abs_diff_samples": commit_diffs,
        "end_nearest_abs_diff_sec": [x / SAMPLING_RATE for x in end_diffs],
        "commit_nearest_abs_diff_sec": [x / SAMPLING_RATE for x in commit_diffs],
    }


def _write_batch_summary_files(
    entries: list[dict],
    summary_json: Path,
    summary_csv: Path,
    summary_md: Path,
) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"total": len(entries), "entries": entries}, f, indent=2, ensure_ascii=False)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "audio_path",
            "status",
            "output_png",
            "output_json",
            "offline_end_count",
            "offline_commit_count",
            "server_end_count",
            "server_commit_count",
            "end_diff_mean_ms",
            "commit_diff_mean_ms",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in entries:
            w.writerow(row)

    lines = [
        "# VAD Batch Verification Summary",
        "",
        f"- total files: {len(entries)}",
        "",
        "| status | audio | end diff mean(ms) | commit diff mean(ms) | png | json |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in entries:
        lines.append(
            f"| {row['status']} | {row['audio_path']} | "
            f"{row.get('end_diff_mean_ms', '')} | {row.get('commit_diff_mean_ms', '')} | "
            f"{row.get('output_png', '')} | {row.get('output_json', '')} |"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify current architecture VAD behavior with graph overlay"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        default=None,
        help="Single input audio path (wav/flac/...).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset directory to batch-run (example: test-clean, test-other).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=".",
        help="Dataset root used with --subset and for output naming (default: .).",
    )
    parser.add_argument(
        "--audio-glob",
        type=str,
        default=None,
        help="Batch input glob pattern (example: 'test-clean/**/*.flac').",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files when using --subset/--audio-glob.",
    )
    parser.add_argument(
        "--chunk-size-ms",
        type=int,
        default=100,
        help="Simulated websocket chunk size in ms (default: 100)",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="test-results/vad-verify/architecture_vad_verification.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="test-results/vad-verify/architecture_vad_trace.json",
        help="Output trace json path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test-results/vad-verify/batch",
        help="Base directory for batch outputs (default: test-results/vad-verify/batch).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Batch summary JSON path. Defaults to <output-dir>/summary.json.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Batch summary CSV path. Defaults to <output-dir>/summary.csv.",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default=None,
        help="Batch summary markdown path. Defaults to <output-dir>/summary.md.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue batch processing when one file fails.",
    )
    parser.add_argument(
        "--server-log",
        type=str,
        default=None,
        help="Optional server log path to overlay [vad]/[vad-commit] events",
    )
    parser.add_argument(
        "--auto-run-server",
        action="store_true",
        help=(
            "Launch server subprocess, stream this audio over websocket, and "
            "parse generated server log automatically."
        ),
    )
    parser.add_argument(
        "--auto-stream-to-server",
        action="store_true",
        help=(
            "Do not launch a subprocess server. Connect to an already-running "
            "websocket server, stream this audio, then parse server log deltas."
        ),
    )
    parser.add_argument(
        "--server-cmd",
        type=str,
        default=None,
        help=(
            "Server command used by --auto-run-server "
            "(example: 'python Qwen3-ASR/examples/streaming_websocket_server.py "
            "--host 127.0.0.1 --port 8765 --no-idle-shutdown')."
        ),
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="ws://127.0.0.1:8765",
        help="Websocket URL for auto client streaming (default: ws://127.0.0.1:8765).",
    )
    parser.add_argument(
        "--server-cwd",
        type=str,
        default=None,
        help="Optional working directory for --server-cmd.",
    )
    parser.add_argument(
        "--server-startup-timeout-sec",
        type=float,
        default=30.0,
        help="How long to wait for websocket server startup (default: 30).",
    )
    parser.add_argument(
        "--server-post-finish-wait-sec",
        type=float,
        default=1.0,
        help="Extra wait after sending finish message (default: 1.0).",
    )
    parser.add_argument(
        "--client-start-json",
        type=str,
        default='{"type":"start","lang":"auto","targetLang":"en"}',
        help=(
            "JSON sent before audio in --auto-run-server mode. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--client-finish-json",
        type=str,
        default='{"type":"finish"}',
        help=(
            "JSON sent after audio in --auto-run-server mode. "
            "Use 'none' to disable."
        ),
    )
    args = parser.parse_args()

    if args.auto_run_server and args.auto_stream_to_server:
        raise ValueError("Use only one of --auto-run-server or --auto-stream-to-server")
    dataset_root = Path(args.dataset_root)
    files = _discover_audio_files(
        audio_path=args.audio_path,
        subset=args.subset,
        dataset_root=args.dataset_root,
        audio_glob=args.audio_glob,
        max_files=args.max_files,
    )
    is_batch = len(files) > 1 or args.subset is not None or args.audio_glob is not None

    server_log_path = Path(args.server_log) if args.server_log else None
    server_cwd = Path(args.server_cwd) if args.server_cwd else None
    start_json = _parse_optional_json_arg(args.client_start_json)
    finish_json = _parse_optional_json_arg(args.client_finish_json)

    batch_entries: list[dict] = []
    output_dir = Path(args.output_dir)

    for idx, audio_path in enumerate(files, 1):
        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLING_RATE)
            trace = run_architecture_vad_trace(audio, args.chunk_size_ms)
            server_trace = None
            auto_run_meta = None

            if args.auto_run_server:
                if not args.server_cmd:
                    raise ValueError("--auto-run-server requires --server-cmd")
                target_server_log = (
                    server_log_path
                    if server_log_path is not None
                    else Path("test-results/vad-verify/server.auto.log")
                )
                run_server_and_collect_log(
                    audio=audio,
                    server_cmd=args.server_cmd,
                    server_log_path=target_server_log,
                    server_cwd=server_cwd,
                    server_url=args.server_url,
                    chunk_size_ms=args.chunk_size_ms,
                    startup_timeout_sec=args.server_startup_timeout_sec,
                    start_message=start_json,
                    finish_message=finish_json,
                    post_finish_wait_sec=args.server_post_finish_wait_sec,
                    launch_server=True,
                )
                server_trace = parse_server_log_trace(target_server_log)
                auto_run_meta = {
                    "enabled": True,
                    "mode": "auto_run_server",
                    "server_cmd": args.server_cmd,
                    "server_url": args.server_url,
                    "server_cwd": str(server_cwd) if server_cwd is not None else None,
                    "server_log": str(target_server_log),
                    "client_start_json": start_json,
                    "client_finish_json": finish_json,
                    "server_startup_timeout_sec": args.server_startup_timeout_sec,
                    "server_post_finish_wait_sec": args.server_post_finish_wait_sec,
                }
            elif args.auto_stream_to_server:
                target_server_log = (
                    server_log_path if server_log_path is not None else Path("/tmp/asr_server.log")
                )
                if not target_server_log.exists():
                    raise FileNotFoundError(
                        f"server log not found for --auto-stream-to-server: {target_server_log}"
                    )
                log_offset_before = target_server_log.stat().st_size
                run_server_and_collect_log(
                    audio=audio,
                    server_cmd=None,
                    server_log_path=target_server_log,
                    server_cwd=None,
                    server_url=args.server_url,
                    chunk_size_ms=args.chunk_size_ms,
                    startup_timeout_sec=args.server_startup_timeout_sec,
                    start_message=start_json,
                    finish_message=finish_json,
                    post_finish_wait_sec=args.server_post_finish_wait_sec,
                    launch_server=False,
                )
                server_trace = parse_server_log_trace(
                    target_server_log, start_offset=log_offset_before
                )
                auto_run_meta = {
                    "enabled": True,
                    "mode": "auto_stream_to_server",
                    "server_cmd": None,
                    "server_url": args.server_url,
                    "server_cwd": None,
                    "server_log": str(target_server_log),
                    "server_log_start_offset": log_offset_before,
                    "client_start_json": start_json,
                    "client_finish_json": finish_json,
                    "server_startup_timeout_sec": args.server_startup_timeout_sec,
                    "server_post_finish_wait_sec": args.server_post_finish_wait_sec,
                }
            elif args.server_log:
                target_server_log = Path(args.server_log)
                if not target_server_log.exists():
                    raise FileNotFoundError(f"server log not found: {target_server_log}")
                server_trace = parse_server_log_trace(target_server_log)

            if is_batch:
                key = _stable_file_key(audio_path, dataset_root=dataset_root)
                output_png = output_dir / "plots" / f"{key}.png"
                output_json = output_dir / "traces" / f"{key}.json"
            else:
                output_png = Path(args.output_png)
                output_json = Path(args.output_json)

            draw_plot(audio, trace, output_png, server_trace=server_trace)

            output_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "audio_path": str(audio_path),
                "sampling_rate": SAMPLING_RATE,
                "chunk_size_ms": args.chunk_size_ms,
                "vad_config": {
                    "threshold": VAD_THRESHOLD,
                    "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
                    "speech_pad_ms": VAD_SPEECH_PAD_MS,
                    "window_size_samples": VAD_WINDOW_SIZE_SAMPLES,
                },
                "auto_run_server": auto_run_meta,
                **trace,
            }
            if server_trace is not None:
                payload["server_trace"] = server_trace
                payload["comparison"] = _build_comparison(trace, server_trace)

            with output_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            comparison = payload.get("comparison", {})
            end_diff_ms = comparison.get("end_nearest_abs_diff_sec", [])
            commit_diff_ms = comparison.get("commit_nearest_abs_diff_sec", [])
            end_mean_ms = round(float(np.mean(end_diff_ms)) * 1000.0, 3) if end_diff_ms else ""
            commit_mean_ms = (
                round(float(np.mean(commit_diff_ms)) * 1000.0, 3) if commit_diff_ms else ""
            )

            print(
                f"[{idx}/{len(files)}] Saved plot={output_png} json={output_json} "
                f"events(end={len(trace['speech_end_samples'])},commit={len(trace['vad_commit_samples'])})"
            )

            batch_entries.append(
                {
                    "audio_path": str(audio_path),
                    "status": "ok",
                    "output_png": str(output_png),
                    "output_json": str(output_json),
                    "offline_end_count": len(trace["speech_end_samples"]),
                    "offline_commit_count": len(trace["vad_commit_samples"]),
                    "server_end_count": comparison.get("server_end_count", ""),
                    "server_commit_count": comparison.get("server_commit_count", ""),
                    "end_diff_mean_ms": end_mean_ms,
                    "commit_diff_mean_ms": commit_mean_ms,
                    "error": "",
                }
            )
        except Exception as e:
            batch_entries.append(
                {
                    "audio_path": str(audio_path),
                    "status": "error",
                    "output_png": "",
                    "output_json": "",
                    "offline_end_count": "",
                    "offline_commit_count": "",
                    "server_end_count": "",
                    "server_commit_count": "",
                    "end_diff_mean_ms": "",
                    "commit_diff_mean_ms": "",
                    "error": str(e),
                }
            )
            print(f"[{idx}/{len(files)}] ERROR: {audio_path} -> {e}")
            if not is_batch or not args.continue_on_error:
                raise

    if is_batch:
        summary_json = Path(args.summary_json) if args.summary_json else output_dir / "summary.json"
        summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "summary.csv"
        summary_md = Path(args.summary_md) if args.summary_md else output_dir / "summary.md"
        _write_batch_summary_files(batch_entries, summary_json, summary_csv, summary_md)
        ok_count = sum(1 for x in batch_entries if x["status"] == "ok")
        err_count = len(batch_entries) - ok_count
        print(
            f"Batch done: total={len(batch_entries)}, ok={ok_count}, error={err_count} "
            f"summary_json={summary_json} summary_csv={summary_csv} summary_md={summary_md}"
        )


if __name__ == "__main__":
    main()
