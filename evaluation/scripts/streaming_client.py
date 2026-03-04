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
            # 1. 전체 타이밍 및 카운트 초기화 (루프 밖에서 한 번만)
            self.start_time = time.time()
            total_chunks_sent = 0

            # 2. 파일마다 반복 (화자/챕터 내의 파일들)
            for file_idx, flac_file in enumerate(flac_files, 1):
                # ⭐ 핵심: 파일마다 새 연결을 맺어 서버 버퍼를 초기화합니다.
                async with websockets.connect(self.server_url) as websocket:
                    utt_id = Path(flac_file).stem
                    finish_event = asyncio.Event()
                    receive_task = None

                    if show_recognition:
                        # 파일별 응답을 받기 위한 백그라운드 태스크 시작
                        receive_task = asyncio.create_task(self.receive_responses(websocket, finish_event))

                    # 정답(GT) 저장 및 오디오 로드
                    if utt_id in transcripts:
                        self.gt_list.append(transcripts[utt_id])

                    audio, sr = sf.read(flac_file)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)

                    audio_duration = len(audio) / self.sample_rate
                    self.total_audio_duration += audio_duration

                    # 파일 정보 및 GT 출력
                    if show_transcript:
                        print(f"\n[File {file_idx}/{len(flac_files)}] {utt_id}")
                        print(f"⏱️  Audio Length: {audio_duration:.2f}s")
                        if utt_id in transcripts:
                            print(f"📝 GT: {transcripts[utt_id]}")

                    # 3. 오디오 청크 스트리밍
                    num_chunks = (len(audio) + self.chunk_size - 1) // self.chunk_size
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, len(audio))
                        chunk_bytes = audio[start_idx:end_idx].astype(np.float32).tobytes()

                        await websocket.send(chunk_bytes)
                        total_chunks_sent += 1
                        await asyncio.sleep(self.interval_ms / 1000.0)

                    # ⭐ 4. 파일 전송 완료 후 서버에 '끝났다'고 신호 보내기
                    await websocket.send(json.dumps({'type': 'finish'}))
                    
                    # 서버가 마지막 인식을 완료할 때까지 대기 (최대 10초)
                    try:
                        await asyncio.wait_for(finish_event.wait(), timeout=10.0)
                    except asyncio.TimeoutError:
                        print("  (Timeout: 서버 응답 대기 시간 초과)")

                    # 5. 이번 파일의 태스크 종료 (with문을 나가면 연결도 닫힘)
                    if receive_task:
                        receive_task.cancel()
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass

            # 6. 모든 파일 완료 후 통계 및 저장
            self.end_time = time.time()
            total_processing_time = self.end_time - self.start_time
            delay = total_processing_time - self.total_audio_duration

            print(f"\n{'='*70}\nStream completed!\nTotal files: {len(flac_files)}\nTotal audio duration: {self.total_audio_duration:.2f}s\n{'='*70}\n")

            # 결과 저장 로직
            if self.gt_list or self.output_lines:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                output_filename = output_dir / f"streaming_results_{subset}_{speaker_id}_{chapter_id}.txt"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(f"[{speaker_id}-{chapter_id}]\n")
                    if self.gt_list: f.write(f"GT: {' '.join(self.gt_list)}\n")
                    if self.output_lines: f.write(f"Whisper: {' '.join(self.output_lines)}\n\n")
                    f.write(f"Total audio duration: {self.total_audio_duration:.2f}s\n")
                    f.write(f"Total processing time: {total_processing_time:.2f}s\n")
                print(f"Results saved to: {output_filename}")
            else:
                print("No results to save!")

        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            raise

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