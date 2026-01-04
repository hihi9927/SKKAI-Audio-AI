#!/usr/bin/env python3
"""
Batch testing script for WhisperLiveKit with LibriSpeech dataset
Tests with different backend-policy settings (1: SimulStreaming, 2: LocalAgreement)
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
import subprocess
import signal

logging.basicConfig(
    format='%(levelname)s\t%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class ServerManager:
    """Manages WhisperLiveKit server lifecycle"""

    def __init__(self, server_script, host='localhost', port=8001, model='large-v3', lan='en'):
        self.server_script = server_script
        self.host = host
        self.port = port
        self.model = model
        self.lan = lan
        self.process = None

    def start_server(self, policy, additional_args=None):
        """Start WhisperLiveKit server with specified policy"""
        if self.process is not None:
            logger.warning("Server already running, stopping first...")
            self.stop_server()

        # Build command
        cmd = [
            sys.executable,
            self.server_script,
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.model,
            '--lan', self.lan,
            '--backend-policy', str(policy),
        ]

        # Add additional arguments
        if additional_args:
            cmd.extend(additional_args)

        logger.info(f"Starting server with policy {policy}...")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Start server process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Wait for server to be ready
            if not self._wait_for_server_ready(timeout=60):
                logger.error("Server failed to start within timeout")
                self.stop_server()
                return False

            logger.info(f"Server started successfully (PID: {self.process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def _wait_for_server_ready(self, timeout=60):
        """Wait for server to be ready by checking WebSocket connection"""
        start_time = time.time()
        ws_url = f'ws://{self.host}:{self.port}/asr'

        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    logger.error("Server process terminated unexpectedly")
                    return False

                # Try to connect
                async def check_connection():
                    try:
                        async with websockets.connect(ws_url, ping_interval=None, open_timeout=2) as ws:
                            # Wait for hello message
                            await asyncio.wait_for(ws.recv(), timeout=2.0)
                            return True
                    except:
                        return False

                if asyncio.run(check_connection()):
                    logger.info("Server is ready!")
                    return True

            except Exception as e:
                logger.debug(f"Waiting for server... ({e})")

            time.sleep(1)

        return False

    def stop_server(self):
        """Stop the running server"""
        if self.process is None:
            logger.debug("No server process to stop")
            return

        logger.info(f"Stopping server (PID: {self.process.pid})...")

        try:
            # Try graceful termination first
            self.process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                self.process.wait(timeout=10)
                logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
                logger.info("Server force stopped")

        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        finally:
            self.process = None

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_server()


def find_audio_files(test_dir):
    """Find all audio files in test directory"""
    audio_files = []
    transcript_map = {}

    # Parse all transcript files
    for trans_file in Path(test_dir).rglob('*.trans.txt'):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        file_id, transcript = parts
                        transcript_map[file_id] = transcript

    # Find all audio files
    for audio_file in Path(test_dir).rglob('*.flac'):
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


async def process_single_file(ws, audio_data, file_id):
    """Process a single audio file through WebSocket (LibriSpeech style)"""
    try:
        processing_start = time.time()
        first_result_time = None

        # Send audio data in chunks (simulate streaming)
        # Convert float32 to int16 PCM (s16le format) as expected by WhisperLiveKit
        audio_int16 = (audio_data * 32768.0).astype(np.int16)

        chunk_size = int(0.1 * SAMPLING_RATE)  # 100ms chunks
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i+chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.01)  # Small delay to simulate real-time

        # Send finish command to flush buffer (LibriSpeech style)
        logger.debug(f"Sending finish command for {file_id}")
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
                    msg_type = data.get('type', '')

                    # Skip hello and config messages
                    if msg_type in ['hello', 'config']:
                        logger.debug(f"Received {msg_type} message")
                        continue

                    # LibriSpeech style: look for 'final' type
                    if msg_type == 'final':
                        # Record first result time
                        if first_result_time is None:
                            first_result_time = time.time()

                        text = data.get('original', '').strip()
                        if text:
                            results.append(text)
                            logger.debug(f"Received final: {text}")

                    # WhisperLiveKit style: also accept active_transcription
                    elif data.get('status') == 'active_transcription':
                        # Record first result time
                        if first_result_time is None:
                            first_result_time = time.time()

                        # Collect transcription from lines (keep only the longest/latest text)
                        lines = data.get('lines', [])
                        for line in lines:
                            text = line.get('text', '').strip()
                            if text:
                                # Replace with longest text (streaming updates)
                                if not results or len(text) > len(results[-1]):
                                    if results:
                                        results[-1] = text
                                    else:
                                        results.append(text)
                                    logger.debug(f"Received: {text}")

            except asyncio.TimeoutError:
                logger.debug(f"Timeout waiting for results")
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


def load_processed_files(output_file, policy):
    """Load already processed files from existing JSON"""
    import os

    if not os.path.exists(output_file):
        return set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        policy_key = f'policy_{policy}'
        if policy_key not in data:
            return set()

        # Get all file_ids from raw_results
        processed_files = {r['file_id'] for r in data[policy_key].get('raw_results', [])}

        if processed_files:
            logger.info(f"Found {len(processed_files)} already processed files for policy {policy}")

        return processed_files
    except Exception as e:
        logger.warning(f"Error loading existing results: {e}")
        return set()


async def process_batch(audio_files, ws_url, output_file, policy, limit=None):
    """Process all audio files in batch"""
    # Load already processed files
    processed_file_ids = load_processed_files(output_file, policy)

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
        logger.info("No files to process. All files already completed for this policy.")
        return []

    logger.info(f"Processing {total} audio files")
    logger.info(f"Backend Policy: {policy}")
    logger.info(f"Server: {ws_url}")

    start_time = time.time()

    for idx, audio_info in enumerate(files_to_process):
        file_id = audio_info['file_id']
        audio_path = audio_info['path']
        reference = audio_info['reference']

        # Extract speaker_id from file_id
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
            async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None, max_size=10*1024*1024) as ws:
                # Wait for hello/config message (LibriSpeech style)
                try:
                    hello_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    logger.debug(f"Server hello: {hello_msg}")
                except Exception as e:
                    logger.warning(f"No hello message received: {e}")

                # Process file
                result = await process_single_file(ws, audio_data, file_id)

                if result and result['transcript']:
                    hypothesis = result['transcript']
                    total_time = result['total_time']
                    first_token_latency = result['first_token_latency']

                    # Calculate processing time
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
            save_results_raw(results, output_file + '.tmp', policy)
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
                policy_key = f'policy_{policy}'
                if policy_key in existing_data and 'raw_results' in existing_data[policy_key]:
                    existing_results = existing_data[policy_key]['raw_results']
                    all_results = existing_results + results
                    logger.info(f"Merged {len(existing_results)} existing + {len(results)} new = {len(all_results)} total results")
                    results = all_results
            except Exception as e:
                logger.warning(f"Could not merge with existing results: {e}")

    # Save final results
    save_results_structured(results, output_file, policy)
    logger.info(f"Results saved to: {output_file}")

    return results


def save_results_raw(results, output_file, policy):
    """Save raw results to JSON file (for intermediate saves)"""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'policy': policy,
        'total_files': len(results),
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def save_results_structured(results, output_file, policy):
    """Save results structured by policy, merging with existing data"""
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

    # Group by speaker_id
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

        # Calculate average metrics
        first_token_latencies = [r['first_token_latency'] for r in folder_results if r['first_token_latency'] is not None]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else None

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

    # Build policy-specific data
    policy_key = f'policy_{policy}'
    policy_data = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'num_files': len(results),
            'wer': overall_wer,
            'first_token_latency': overall_first_token_latency,
            'avg_processing_time': overall_avg_processing_time
        },
        'folders': folder_stats,
        'raw_results': results
    }

    # Merge with existing data
    existing_data[policy_key] = policy_data

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def calculate_wer(results, policy):
    """Calculate and display Word Error Rate"""
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
        logger.info(f"RESULTS SUMMARY - BACKEND POLICY {policy}")
        logger.info(f"{'='*70}")
        logger.info(f"Total files processed: {len(results)}")
        logger.info(f"Overall WER: {overall_wer*100:.2f}%")
        logger.info(f"Average First Token Latency: {avg_first_token_latency:.3f}s")
        logger.info(f"Average Processing Time: {avg_processing_time:.3f}s")
        logger.info(f"Number of speakers: {len(folders)}")
        logger.info(f"{'='*70}\n")

        return overall_wer
    except ImportError:
        logger.warning("jiwer not installed. Install with: pip install jiwer")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Test WhisperLiveKit with LibriSpeech dataset'
    )

    parser.add_argument('--test-dir', type=str,
                        default=r'c:\Users\VRSTUDIO3\Desktop\STiTy-main1\LibriSpeech\test-other',
                        help='Path to test directory (default: c:\\Users\\VRSTUDIO3\\Desktop\\STiTy-main1\\LibriSpeech\\test-other)')
    parser.add_argument('--policy', type=int, default=None,
                        choices=[1, 2],
                        help='Backend policy (1: SimulStreaming, 2: LocalAgreement)')
    parser.add_argument('--all-policies', action='store_true',
                        help='Test all policies sequentially (1 -> 2)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='WebSocket server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8001,
                        help='WebSocket server port (default: 8001)')
    parser.add_argument('--output', type=str, default='whisperlivekit_results.json',
                        help='Output JSON file (default: whisperlivekit_results.json)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of NEW files to process per policy')
    parser.add_argument('--calculate-wer', action='store_true',
                        help='Calculate Word Error Rate (requires jiwer)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level (default: INFO)')
    parser.add_argument('--auto-server', action='store_true',
                        help='Automatically start/stop server for each policy (requires --server-script)')
    parser.add_argument('--server-script', type=str,
                        default='whisperlivekit/websocket_server.py',
                        help='Path to server script (default: whisperlivekit/websocket_server.py)')
    parser.add_argument('--server-model', type=str, default='large-v3',
                        help='Model to use for server (default: large-v3)')
    parser.add_argument('--server-lan', type=str, default='en',
                        help='Language for server (default: en)')
    parser.add_argument('--server-args', type=str, default='',
                        help='Additional server arguments (space-separated)')

    args = parser.parse_args()

    # Set log level
    logger.setLevel(args.log_level)

    # Determine policies to test
    if args.all_policies:
        policies_to_test = [1, 2]
        logger.info("Testing all policies: 1 (SimulStreaming) -> 2 (LocalAgreement)")
    elif args.policy:
        policies_to_test = [args.policy]
    else:
        logger.error("Must specify either --policy or --all-policies")
        sys.exit(1)

    # Check test directory
    if not os.path.isdir(args.test_dir):
        logger.error(f"Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Find all audio files
    logger.info("Scanning test directory...")
    audio_files = find_audio_files(args.test_dir)
    logger.info(f"Found {len(audio_files)} audio files\n")

    if len(audio_files) == 0:
        logger.error("No audio files found")
        sys.exit(1)

    # Initialize server manager if auto-server is enabled
    server_manager = None
    if args.auto_server:
        if not os.path.exists(args.server_script):
            logger.error(f"Server script not found: {args.server_script}")
            sys.exit(1)

        # Parse additional server arguments
        additional_args = args.server_args.split() if args.server_args else []

        server_manager = ServerManager(
            server_script=args.server_script,
            host=args.host,
            port=args.port,
            model=args.server_model,
            lan=args.server_lan
        )
        logger.info("Auto-server mode enabled")
        logger.info(f"Server script: {args.server_script}")
        logger.info(f"Server model: {args.server_model}, language: {args.server_lan}\n")

    # Process each policy
    all_results = {}
    try:
        for policy_idx, policy in enumerate(policies_to_test):
            logger.info(f"\n{'='*70}")
            logger.info(f"POLICY {policy_idx+1}/{len(policies_to_test)}: Backend Policy {policy}")
            policy_name = "SimulStreaming" if policy == 1 else "LocalAgreement"
            logger.info(f"({policy_name})")
            logger.info(f"{'='*70}\n")

            # Start server if auto-server is enabled
            if server_manager:
                if not server_manager.start_server(policy, additional_args):
                    logger.error(f"Failed to start server for policy {policy}")
                    sys.exit(1)
            else:
                # Manual mode: wait for user
                ws_url = f'ws://{args.host}:{args.port}/asr'
                logger.info(f"Connecting to: {ws_url}")
                logger.info(f"NOTE: Make sure server is running with --backend-policy {policy}\n")

            ws_url = f'ws://{args.host}:{args.port}/asr'

            # Process batch
            results = asyncio.run(
                process_batch(audio_files, ws_url, args.output, policy, args.limit)
            )

            # Calculate WER if requested
            if args.calculate_wer and len(results) > 0:
                calculate_wer(results, policy)

            all_results[f'policy_{policy}'] = results

            # Stop server after processing (if auto-server)
            if server_manager:
                server_manager.stop_server()
                logger.info(f"Server stopped for policy {policy}\n")

            # Wait between policies
            if policy_idx < len(policies_to_test) - 1:
                next_policy = policies_to_test[policy_idx + 1]
                logger.info(f"\n{'='*70}")
                logger.info(f"Completed policy {policy}")
                logger.info(f"Next: policy {next_policy}")

                if not server_manager:
                    # Manual mode: wait for user to restart server
                    logger.info(f"Please restart server with --backend-policy {next_policy}")
                    logger.info(f"{'='*70}\n")
                    input("Press Enter when server is ready with new policy...")
                else:
                    logger.info(f"Auto-starting server with policy {next_policy}")
                    logger.info(f"{'='*70}\n")
                    time.sleep(2)  # Brief pause between restarts

        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("ALL POLICIES COMPLETED")
        logger.info(f"{'='*70}")
        for policy in policies_to_test:
            policy_key = f'policy_{policy}'
            if policy_key in all_results:
                num_files = len(all_results[policy_key])
                policy_name = "SimulStreaming" if policy == 1 else "LocalAgreement"
                logger.info(f"  Policy {policy} ({policy_name}): {num_files} files processed")
        logger.info(f"\nResults saved to: {args.output}")
        logger.info(f"{'='*70}\n")

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        logger.info(f"Partial results saved to: {args.output}")
        if server_manager:
            logger.info("Stopping server...")
            server_manager.stop_server()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if server_manager:
            logger.info("Stopping server...")
            server_manager.stop_server()
        sys.exit(1)
    finally:
        # Ensure server is stopped
        if server_manager:
            server_manager.stop_server()


if __name__ == "__main__":
    main()
