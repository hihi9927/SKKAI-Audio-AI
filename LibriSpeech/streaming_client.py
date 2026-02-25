#!/usr/bin/env python3
"""
LibriSpeech Streaming Client for SimulEval WebSocket Server
Simulates real-time audio streaming by sending FLAC audio chunks at regular intervals
"""

import asyncio
import websockets
import argparse
import os
import glob
from pathlib import Path
import time
import soundfile as sf
import numpy as np
import json

try:
    from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
    HAS_JIWER = True
    # Create transformation pipeline for normalization
    JIWER_TRANSFORM = Compose([
        ToLowerCase(),           # Convert to lowercase
        RemovePunctuation(),     # Remove all punctuation
        RemoveMultipleSpaces(),  # Remove multiple spaces
        Strip()                  # Strip leading/trailing whitespace
    ])
except ImportError:
    HAS_JIWER = False
    JIWER_TRANSFORM = None
    print("Warning: jiwer not installed. WER calculation will be skipped.")
    print("Install with: pip install jiwer")


class LibriSpeechStreamingClient:
    def __init__(self, server_url, dataset_path, interval_ms=500, chunk_size=8000):
        """
        Initialize streaming client

        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8765)
            dataset_path: Path to LibriSpeech dataset root
            interval_ms: Interval between chunks in milliseconds
            chunk_size: Number of audio samples per chunk (at 16kHz, 500ms = 8000 samples)
        """
        self.server_url = server_url
        self.dataset_path = Path(dataset_path)
        self.interval_ms = interval_ms
        self.chunk_size = chunk_size
        self.sample_rate = 16000
        self.gt_list = []  # List of all ground truth transcripts
        self.output_lines = []  # Collect Whisper output lines for saving to file
        self.start_time = None  # Track when streaming starts
        self.end_time = None  # Track when streaming ends
        self.total_audio_duration = 0.0  # Total duration of all audio files

    def get_chapter_files(self, subset, speaker_id, chapter_id):
        """
        Get all FLAC files for a specific chapter, sorted by filename

        Args:
            subset: Dataset subset (e.g., 'test-clean', 'test-other')
            speaker_id: Speaker ID
            chapter_id: Chapter ID

        Returns:
            List of FLAC file paths sorted in order
        """
        chapter_path = self.dataset_path / subset / str(speaker_id) / str(chapter_id)

        if not chapter_path.exists():
            raise ValueError(f"Chapter path not found: {chapter_path}")

        # Get all FLAC files and sort them
        flac_files = sorted(glob.glob(str(chapter_path / "*.flac")))

        if not flac_files:
            raise ValueError(f"No FLAC files found in {chapter_path}")

        return flac_files

    def get_transcript(self, subset, speaker_id, chapter_id):
        """
        Read the transcript file for the chapter

        Returns:
            Dictionary mapping utterance IDs to transcripts
        """
        chapter_path = self.dataset_path / subset / str(speaker_id) / str(chapter_id)
        trans_file = chapter_path / f"{speaker_id}-{chapter_id}.trans.txt"

        transcripts = {}
        if trans_file.exists():
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcripts[utt_id] = text

        return transcripts

    async def receive_responses(self, websocket, finish_event=None):
        """Receive and display server responses"""
        try:
            while True:
                response = await websocket.recv()
                try:
                    data = json.loads(response)
                    msg_type = data.get('type', '')

                    if msg_type == 'hello':
                        print(f"\n✓ Server: {data.get('message', '')}\n")
                    elif msg_type == 'final':
                        # Final transcription result
                        original = data.get('original', '')

                        # Print Whisper result to console
                        print(f"{original} / ", end="", flush=True)

                        # Store just the text (without "Whisper: " prefix)
                        self.output_lines.append(original)
                    elif msg_type == 'finish_complete':
                        print(f"✓ Server finished processing all segments")
                        if finish_event:
                            finish_event.set()
                        return  # Exit cleanly after finish_complete
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            # Connection closed - set finish_event to unblock waiting
            if finish_event:
                finish_event.set()
        except Exception as e:
            print(f"\nError receiving response: {e}")
            if finish_event:
                finish_event.set()

    async def stream_audio(self, subset, speaker_id, chapter_id, show_transcript=True, show_recognition=True):
        """
        Stream audio from a chapter to the WebSocket server

        Args:
            subset: Dataset subset (e.g., 'test-clean')
            speaker_id: Speaker ID
            chapter_id: Chapter ID
            show_transcript: Whether to display ground truth transcripts
            show_recognition: Whether to display ASR recognition results
        """
        print(f"\n{'='*70}")
        print(f"Starting stream: {subset}/{speaker_id}/{chapter_id}")
        print(f"Interval: {self.interval_ms}ms | Chunk size: {self.chunk_size} samples")
        print(f"{'='*70}\n")

        # Get files and transcripts
        flac_files = self.get_chapter_files(subset, speaker_id, chapter_id)
        transcripts = self.get_transcript(subset, speaker_id, chapter_id)

        print(f"Found {len(flac_files)} audio files\n")

        try:
            async with websockets.connect(self.server_url) as websocket:
                print(f"Connected to {self.server_url}\n")

                # Create finish event to signal when server completes finish processing
                finish_event = asyncio.Event()

                # Start receiving responses in background
                receive_task = None
                if show_recognition:
                    receive_task = asyncio.create_task(self.receive_responses(websocket, finish_event))

                # Start timing
                self.start_time = time.time()

                total_chunks_sent = 0

                for file_idx, flac_file in enumerate(flac_files, 1):
                    # Extract utterance ID from filename
                    utt_id = Path(flac_file).stem

                    # Store GT for file output
                    if utt_id in transcripts:
                        self.gt_list.append(transcripts[utt_id])

                    # Read audio file
                    audio, sr = sf.read(flac_file)

                    # Ensure audio is mono
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)

                    # Resample if necessary (LibriSpeech is already 16kHz, but just in case)
                    if sr != self.sample_rate:
                        print(f"Warning: Resampling from {sr}Hz to {self.sample_rate}Hz")
                        # Simple resampling (for production, use librosa.resample)
                        audio = np.interp(
                            np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                            np.arange(len(audio)),
                            audio
                        )

                    # Track total audio duration
                    audio_duration = len(audio) / self.sample_rate
                    self.total_audio_duration += audio_duration

                    # Show file info
                    if show_transcript:
                        print(f"\n[File {file_idx}/{len(flac_files)}] {utt_id}")
                        print(f"⏱️  Audio Length: {audio_duration:.2f}s")

                    # Show GT on screen
                    if utt_id in transcripts:
                        gt_text = transcripts[utt_id]
                        if show_transcript:
                            print(f"📝 GT: {gt_text}")

                    # No start message needed - just stream continuously

                    # Stream audio in chunks
                    num_chunks = (len(audio) + self.chunk_size - 1) // self.chunk_size

                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, len(audio))
                        chunk = audio[start_idx:end_idx]

                        # Convert to bytes (Float32 format)
                        chunk_float32 = chunk.astype(np.float32)
                        chunk_bytes = chunk_float32.tobytes()

                        # Send chunk
                        await websocket.send(chunk_bytes)
                        total_chunks_sent += 1

                        # Wait for interval
                        await asyncio.sleep(self.interval_ms / 1000.0)

                        # Print progress (only if not showing recognition)
                        if not show_recognition and chunk_idx % 10 == 0:
                            progress = (chunk_idx + 1) / num_chunks * 100
                            print(f"  Progress: {progress:.1f}% ({chunk_idx+1}/{num_chunks} chunks)", end='\r')

                    if not show_recognition:
                        print(f"  Progress: 100.0% ({num_chunks}/{num_chunks} chunks) - Complete!")

                    # Small gap between files (don't send finish - maintain context)
                    await asyncio.sleep(self.interval_ms / 1000.0)

                # Send finish signal to flush remaining audio after entire chapter
                print(f"\nSending finish signal to server...")
                await websocket.send(json.dumps({'type': 'finish'}))

                # Wait for server to signal finish complete
                print(f"Waiting for server to finish processing...")
                try:
                    # Wait up to 30 seconds for finish to complete
                    await asyncio.wait_for(finish_event.wait(), timeout=30.0)
                    print(f"Server finished processing!")
                except asyncio.TimeoutError:
                    print(f"Warning: Timeout waiting for finish complete (waited 30s)")

                # End timing
                self.end_time = time.time()
                total_processing_time = self.end_time - self.start_time
                delay = total_processing_time - self.total_audio_duration

                # Cancel receive task
                if receive_task:
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass

                print(f"\n{'='*70}")
                print(f"Stream completed!")
                print(f"Total files: {len(flac_files)}")
                print(f"Total chunks sent: {total_chunks_sent}")
                print(f"Total audio duration: {self.total_audio_duration:.2f}s")
                print(f"Total processing time: {total_processing_time:.2f}s")
                print(f"Total delay: {delay:.2f}s")
                print(f"{'='*70}\n")

                # Save results to file
                print(f"\nSaving results...")

                if self.gt_list or self.output_lines:
                    # Create output directory if it doesn't exist
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)

                    output_filename = output_dir / f"streaming_results_{subset}_{speaker_id}_{chapter_id}.txt"
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        # Write header with chapter info
                        f.write(f"[{speaker_id}-{chapter_id}]\n")

                        # Write all GT transcripts first
                        if self.gt_list:
                            gt_combined = " ".join(self.gt_list)
                            f.write(f"GT: {gt_combined}\n")
                            print(f"  GT: {len(self.gt_list)} utterances combined")

                        # Write all Whisper results
                        if self.output_lines:
                            whisper_combined = " ".join(self.output_lines)
                            f.write(f"Whisper: {whisper_combined}\n\n")
                            print(f"  Whisper: {len(self.output_lines)} segments combined")

                        # Calculate and write WER if possible
                        if HAS_JIWER and self.gt_list and self.output_lines:
                            gt_combined = " ".join(self.gt_list)
                            whisper_combined = " ".join(self.output_lines)

                            # Normalize both strings (lowercase, no punctuation)
                            gt_normalized = JIWER_TRANSFORM(gt_combined)
                            whisper_normalized = JIWER_TRANSFORM(whisper_combined)

                            # Calculate WER with normalized text
                            wer_score = wer(gt_normalized, whisper_normalized)

                            f.write(f"WER (normalized): {wer_score:.2%}\n")
                            print(f"  WER (normalized): {wer_score:.2%}")

                        # Write timing information
                        f.write(f"Total audio duration: {self.total_audio_duration:.2f}s\n")
                        f.write(f"Total processing time: {total_processing_time:.2f}s\n")
                        f.write(f"Total delay: {delay:.2f}s\n")

                    print(f"\nResults saved to: {output_filename}\n")
                else:
                    print(f"No results to save!")

        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Stream LibriSpeech audio to SimulEval WebSocket server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream a specific chapter with 500ms intervals
  python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686

  # Stream all chapters in test-clean
  python streaming_client.py --subset test-clean --all

  # Stream with 250ms intervals (faster)
  python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686 --interval 250

  # Stream with custom chunk size
  python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686 --chunk-size 4000
        """
    )

    parser.add_argument(
        '--server',
        type=str,
        default='wss://edra-raspiest-eagerly.ngrok-free.dev/ws',
        help='WebSocket server URL (default: wss://edra-raspiest-eagerly.ngrok-free.dev/ws)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='.',
        help='Path to LibriSpeech dataset root (default: current directory)'
    )

    parser.add_argument(
        '--subset',
        type=str,
        required=True,
        choices=['test-clean', 'test-other', 'dev-clean', 'dev-other',
                 'train-clean-100', 'train-clean-360', 'train-other-500'],
        help='Dataset subset to use'
    )

    parser.add_argument(
        '--speaker',
        type=int,
        help='Speaker ID (e.g., 1089, 121, 237)'
    )

    parser.add_argument(
        '--chapter',
        type=int,
        help='Chapter ID (e.g., 134686, 134691)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all chapters in the subset'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=500,
        help='Interval between chunks in milliseconds (default: 500ms)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8000,
        help='Number of samples per chunk (default: 8000 = 500ms at 16kHz)'
    )

    parser.add_argument(
        '--no-transcript',
        action='store_true',
        help='Do not display ground truth transcripts'
    )

    parser.add_argument(
        '--no-recognition',
        action='store_true',
        help='Do not display ASR recognition results from server'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and (args.speaker is None or args.chapter is None):
        parser.error("Either --all or both --speaker and --chapter must be specified")

    # Create client
    client = LibriSpeechStreamingClient(
        server_url=args.server,
        dataset_path=args.dataset,
        interval_ms=args.interval,
        chunk_size=args.chunk_size
    )

    # Run streaming
    if args.all:
        # Process all chapters in the subset
        subset_path = Path(args.dataset) / args.subset
        if not subset_path.exists():
            print(f"Error: Subset path not found: {subset_path}")
            return

        # Get all speaker directories
        speaker_dirs = sorted([d for d in subset_path.iterdir() if d.is_dir()])

        print(f"\n{'='*70}")
        print(f"Processing all chapters in {args.subset}")
        print(f"Found {len(speaker_dirs)} speakers")
        print(f"{'='*70}\n")

        for speaker_dir in speaker_dirs:
            speaker_id = int(speaker_dir.name)

            # Get all chapter directories for this speaker
            chapter_dirs = sorted([d for d in speaker_dir.iterdir() if d.is_dir()])

            for chapter_dir in chapter_dirs:
                chapter_id = int(chapter_dir.name)

                try:
                    asyncio.run(client.stream_audio(
                        subset=args.subset,
                        speaker_id=speaker_id,
                        chapter_id=chapter_id,
                        show_transcript=not args.no_transcript,
                        show_recognition=not args.no_recognition
                    ))
                except Exception as e:
                    print(f"\nError processing {speaker_id}-{chapter_id}: {e}")
                    print("Continuing to next chapter...\n")
                    continue
    else:
        # Process single chapter
        asyncio.run(client.stream_audio(
            subset=args.subset,
            speaker_id=args.speaker,
            chapter_id=args.chapter,
            show_transcript=not args.no_transcript,
            show_recognition=not args.no_recognition
        ))


if __name__ == '__main__':
    main()
