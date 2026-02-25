#!/usr/bin/env python3
"""
Batch processing script for LibriSpeech test-clean dataset
Processes all audio files in test-clean directory using the selected mode
"""
import sys
import os
import argparse
import asyncio
import json
import websockets
import soundfile as sf
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import logging

logging.basicConfig(
    format='%(levelname)s\t%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


def find_audio_files(test_clean_dir):
    """Find all audio files in test-clean directory"""
    audio_files = []
    transcript_map = {}

    # Parse all transcript files
    for trans_file in Path(test_clean_dir).rglob('*.trans.txt'):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        file_id, transcript = parts
                        transcript_map[file_id] = transcript

    # Find all audio files
    for audio_file in Path(test_clean_dir).rglob('*.flac'):
        file_id = audio_file.stem
        if file_id in transcript_map:
            audio_files.append({
                'path': str(audio_file),
                'file_id': file_id,
                'reference': transcript_map[file_id]
            })

    return sorted(audio_files, key=lambda x: x['file_id'])


def load_audio_file(audio_path):
    """Load audio file and convert to 16kHz mono Float32"""
    try:
        audio, sr = sf.read(audio_path, dtype='float32')

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz if needed
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)

        return audio
    except Exception as e:
        logger.error(f"Error loading {audio_path}: {e}")
        return None


async def process_single_file(ws, audio_data, file_id, mode):
    """Process a single audio file through WebSocket"""
    try:
        processing_start = time.time()
        first_result_time = None

        # Send audio data
        if mode == 'streaming':
            # For streaming, send in chunks
            chunk_size = int(0.1 * SAMPLING_RATE)  # 100ms chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                await ws.send(chunk.tobytes())
                await asyncio.sleep(0.01)  # Small delay to simulate real-time

        elif mode == 'chunked':
            # For chunked, send in larger chunks
            chunk_size = int(2.0 * SAMPLING_RATE)  # 2 second chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                await ws.send(chunk.tobytes())

        elif mode == 'original':
            # For original, send entire file at once
            await ws.send(audio_data.tobytes())

        # Send finish command
        await ws.send(json.dumps({'type': 'finish'}))

        # Collect all results
        results = []
        timeout = 120  # 120 seconds total timeout
        start_time = time.time()
        first_message_timeout = 60.0  # Wait up to 60s for first result
        subsequent_timeout = 15.0  # Wait 15s between subsequent results

        message_count = 0
        while time.time() - start_time < timeout:
            try:
                # Use longer timeout for first message
                current_timeout = first_message_timeout if message_count == 0 else subsequent_timeout
                message = await asyncio.wait_for(ws.recv(), timeout=current_timeout)

                message_count += 1
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get('type') == 'final':
                        # Record first result time
                        if first_result_time is None:
                            first_result_time = time.time()
                        results.append(data.get('original', ''))
                        logger.debug(f"Received result: {data.get('original', '')}")
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.debug(f"Error receiving message: {e}")
                break

        processing_end = time.time()

        # Calculate metrics
        total_time = processing_end - processing_start
        first_token_latency = (first_result_time - processing_start) if first_result_time else None

        # Combine all results
        full_transcript = ' '.join(results).strip()

        return {
            'transcript': full_transcript,
            'total_time': total_time,
            'first_token_latency': first_token_latency
        }

    except Exception as e:
        logger.error(f"Error processing {file_id}: {e}")
        return None


def load_processed_files(output_file, mode):
    """Load already processed files from existing JSON"""
    import os

    if not os.path.exists(output_file):
        return set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if mode not in data:
            return set()

        # Get all file_ids from raw_results
        processed_files = {r['file_id'] for r in data[mode].get('raw_results', [])}

        if processed_files:
            logger.info(f"Found {len(processed_files)} already processed files for {mode} mode")

        return processed_files
    except Exception as e:
        logger.warning(f"Error loading existing results: {e}")
        return set()


async def process_batch(audio_files, ws_url, output_file, mode, limit=None):
    """Process all audio files in batch"""
    # Load already processed files
    processed_file_ids = load_processed_files(output_file, mode)

    # Filter out already processed files
    files_to_process = [f for f in audio_files if f['file_id'] not in processed_file_ids]

    if len(processed_file_ids) > 0:
        logger.info(f"Skipping {len(processed_file_ids)} already processed files")
        logger.info(f"Remaining files to process: {len(files_to_process)}")

    # Apply limit to remaining files
    if limit is not None:
        files_to_process = files_to_process[:limit]

    results = []
    processed = 0
    total = len(files_to_process)

    if total == 0:
        logger.info("No files to process. All files already completed for this mode.")
        return []

    logger.info(f"Processing {total} audio files from test-clean dataset")
    logger.info(f"Mode: {mode}")
    logger.info(f"Server: {ws_url}")

    start_time = time.time()

    for idx, audio_info in enumerate(files_to_process):
        file_id = audio_info['file_id']
        audio_path = audio_info['path']
        reference = audio_info['reference']

        # Extract speaker_id (folder) from file_id (e.g., "1089-134686-0000" -> "1089")
        speaker_id = file_id.split('-')[0]

        logger.info(f"[{processed+1}/{total}] Processing: {file_id}")

        # Load audio
        audio_data = load_audio_file(audio_path)
        if audio_data is None:
            logger.error(f"Failed to load audio: {audio_path}")
            continue

        audio_duration = len(audio_data) / SAMPLING_RATE

        # Process through WebSocket
        try:
            async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
                # Receive hello message
                hello = await ws.recv()
                logger.debug(f"Connected: {json.loads(hello)}")

                # Process file
                result = await process_single_file(ws, audio_data, file_id, mode)

                if result and result['transcript']:
                    hypothesis = result['transcript']
                    total_time = result['total_time']
                    first_token_latency = result['first_token_latency']

                    # Calculate processing time (excluding audio duration for simulated modes)
                    # This is an approximation
                    avg_processing_time = total_time - audio_duration if total_time > audio_duration else total_time

                    results.append({
                        'file_id': file_id,
                        'speaker_id': speaker_id,
                        'audio_path': audio_path,
                        'reference': reference,
                        'hypothesis': hypothesis,
                        'duration': audio_duration,
                        'total_time': total_time,
                        'first_token_latency': first_token_latency,
                        'avg_processing_time': avg_processing_time
                    })
                    logger.info(f"  REF: {reference}")
                    logger.info(f"  HYP: {hypothesis}")
                    logger.info(f"  First token: {first_token_latency:.2f}s, Total: {total_time:.2f}s")
                else:
                    logger.warning(f"No result for {file_id}")

        except Exception as e:
            logger.error(f"WebSocket error for {file_id}: {e}")
            continue

        processed += 1

        # Save intermediate results every 10 files
        if processed % 10 == 0:
            save_results_raw(results, output_file + '.tmp', mode)
            logger.info(f"Saved intermediate results ({processed}/{total})")

    elapsed = time.time() - start_time
    logger.info(f"\nProcessing complete: {processed}/{total} files in {elapsed:.2f}s")
    if processed > 0:
        logger.info(f"Average time per file: {elapsed/processed:.2f}s")

    # Merge with existing results if any
    if len(processed_file_ids) > 0:
        import os
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if mode in existing_data and 'raw_results' in existing_data[mode]:
                    existing_results = existing_data[mode]['raw_results']
                    # Combine existing + new results
                    all_results = existing_results + results
                    logger.info(f"Merged {len(existing_results)} existing + {len(results)} new = {len(all_results)} total results")
                    results = all_results
            except Exception as e:
                logger.warning(f"Could not merge with existing results: {e}")

    # Save final results (structured by folder)
    save_results_structured(results, output_file, mode)
    logger.info(f"Results saved to: {output_file}")

    # Calculate WER if requested
    return results


def save_results_raw(results, output_file, mode):
    """Save raw results to JSON file (for intermediate saves)"""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'total_files': len(results),
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def save_results_structured(results, output_file, mode):
    """Save results structured by mode and folder, merging with existing data"""
    import jiwer
    from collections import defaultdict
    import os

    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            logger.warning(f"Could not load existing {output_file}, creating new file")

    # Group by speaker_id (folder)
    folders = defaultdict(list)
    for r in results:
        speaker_id = r['speaker_id']
        folders[speaker_id].append(r)

    # Calculate metrics for each folder
    folder_stats = {}
    for speaker_id, folder_results in sorted(folders.items()):
        references = [r['reference'] for r in folder_results]
        hypotheses = [r['hypothesis'] for r in folder_results]

        # Calculate WER for this folder
        try:
            folder_wer = jiwer.wer(references, hypotheses)
        except:
            folder_wer = None

        # Calculate average first token latency
        first_token_latencies = [r['first_token_latency'] for r in folder_results if r['first_token_latency'] is not None]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else None

        # Calculate average processing time
        avg_processing_times = [r['avg_processing_time'] for r in folder_results]
        avg_processing_time = sum(avg_processing_times) / len(avg_processing_times) if avg_processing_times else None

        folder_stats[speaker_id] = {
            'num_files': len(folder_results),
            'wer': folder_wer,
            'first_token_latency': avg_first_token_latency,
            'avg_processing_time': avg_processing_time
        }

    # Calculate overall metrics
    all_references = [r['reference'] for r in results]
    all_hypotheses = [r['hypothesis'] for r in results]

    try:
        overall_wer = jiwer.wer(all_references, all_hypotheses)
    except:
        overall_wer = None

    all_first_token_latencies = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
    overall_first_token_latency = sum(all_first_token_latencies) / len(all_first_token_latencies) if all_first_token_latencies else None

    all_processing_times = [r['avg_processing_time'] for r in results]
    overall_avg_processing_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else None

    # Build mode-specific data
    mode_data = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'num_files': len(results),
            'wer': overall_wer,
            'first_token_latency': overall_first_token_latency,
            'avg_processing_time': overall_avg_processing_time
        },
        'folders': folder_stats,
        'raw_results': results  # Include raw data for reference
    }

    # Merge with existing data
    existing_data[mode] = mode_data

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def calculate_wer(results, mode):
    """Calculate and display Word Error Rate with additional metrics"""
    try:
        import jiwer
        from collections import defaultdict

        references = [r['reference'] for r in results]
        hypotheses = [r['hypothesis'] for r in results]

        overall_wer = jiwer.wer(references, hypotheses)

        # Calculate average metrics
        first_token_latencies = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0

        processing_times = [r['avg_processing_time'] for r in results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        # Group by folder
        folders = defaultdict(list)
        for r in results:
            folders[r['speaker_id']].append(r)

        logger.info(f"\n{'='*70}")
        logger.info(f"RESULTS SUMMARY - {mode.upper()} MODE")
        logger.info(f"{'='*70}")
        logger.info(f"Total files processed: {len(results)}")
        logger.info(f"Overall WER: {overall_wer*100:.2f}%")
        logger.info(f"Average First Token Latency: {avg_first_token_latency:.3f}s")
        logger.info(f"Average Processing Time: {avg_processing_time:.3f}s")
        logger.info(f"Number of speakers (folders): {len(folders)}")
        logger.info(f"{'='*70}\n")

        return overall_wer
    except ImportError:
        logger.warning("jiwer not installed. Install with: pip install jiwer")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Batch process LibriSpeech test-clean dataset'
    )

# 현재 스크립트 파일의 부모 폴더(LibriSpeech)를 기준으로 경로 설정
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-clean")
    
    parser.add_argument('--data-dir', type=str, # 변수명을 범용적으로 변경
                        default=default_path,
                        help='Path to dataset directory (e.g., test-clean or test-other)')
    parser.add_argument('--mode', type=str, default='streaming',
                        choices=['streaming', 'chunked', 'original'],
                        help='Processing mode (default: streaming)')
    parser.add_argument('--all-modes', action='store_true',
                        help='Process all three modes sequentially (streaming -> chunked -> original)')
    parser.add_argument('--continue-modes', action='store_true',
                        help='After completing current mode, continue with remaining modes')
    parser.add_argument('--host', type=str, default='localhost',
                        help='WebSocket server host (default: localhost)')
    parser.add_argument('--port', type=int, default=None,
                        help='WebSocket server port (default: auto-select based on mode)')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file (default: results.json - all modes merged)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of NEW files to process per mode (default: all). Already processed files are automatically skipped.')
    parser.add_argument('--calculate-wer', action='store_true',
                        help='Calculate Word Error Rate (requires jiwer)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level (default: INFO)')

    args = parser.parse_args()

    # Set log level
    logger.setLevel(args.log_level)

    # Determine modes to process
    if args.all_modes:
        modes_to_process = ['streaming', 'chunked', 'original']
        logger.info("Processing all modes: streaming -> chunked -> original")
    elif args.continue_modes:
        all_modes = ['streaming', 'chunked', 'original']
        start_idx = all_modes.index(args.mode)
        modes_to_process = all_modes[start_idx:]
        logger.info(f"Continue mode: processing {' -> '.join(modes_to_process)}")
    else:
        modes_to_process = [args.mode]

    # Check dataset directory
    if not os.path.isdir(args.data_dir):
        logger.error(f"Directory not found: {args.data_dir}")
        sys.exit(1)

    # Find all audio files
    logger.info(f"Scanning directory: {args.data_dir}...")
    audio_files = find_audio_files(args.data_dir)
    
    if len(audio_files) == 0:
        logger.error("No audio files found")
        sys.exit(1)

    # Port mapping
    port_map = {'streaming': 8001, 'chunked': 8002, 'original': 8003}

    # Process each mode
    all_results = {}
    try:
        for mode_idx, mode in enumerate(modes_to_process):
            logger.info(f"\n{'='*70}")
            logger.info(f"MODE {mode_idx+1}/{len(modes_to_process)}: {mode.upper()}")
            logger.info(f"{'='*70}\n")

            # Determine port for this mode
            if args.port is not None and len(modes_to_process) == 1:
                # Use specified port only for single mode
                port = args.port
            else:
                # Use default port for each mode
                port = port_map[mode]

            ws_url = f'ws://{args.host}:{port}'

            logger.info(f"Connecting to: {ws_url}")

            # Process batch for this mode
            results = asyncio.run(
                process_batch(audio_files, ws_url, args.output, mode, args.limit)
            )

            # Calculate WER if requested
            if args.calculate_wer and len(results) > 0:
                calculate_wer(results, mode)

            all_results[mode] = results

            # Wait between modes
            if mode_idx < len(modes_to_process) - 1:
                next_mode = modes_to_process[mode_idx + 1]
                logger.info(f"\n{'='*70}")
                logger.info(f"Completed {mode.upper()} mode")
                logger.info(f"Next: {next_mode.upper()} mode (port {port_map[next_mode]})")
                logger.info(f"Make sure the {next_mode} server is running on port {port_map[next_mode]}")
                logger.info(f"{'='*70}\n")

                # Wait a bit before next mode
                import time
                logger.info("Waiting 3 seconds before next mode...")
                time.sleep(3)

        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("ALL MODES COMPLETED")
        logger.info(f"{'='*70}")
        for mode in modes_to_process:
            if mode in all_results:
                num_files = len(all_results[mode])
                logger.info(f"  {mode.upper()}: {num_files} files processed")
        logger.info(f"\nResults saved to: {args.output}")
        logger.info(f"{'='*70}\n")

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        logger.info(f"Partial results saved to: {args.output}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
