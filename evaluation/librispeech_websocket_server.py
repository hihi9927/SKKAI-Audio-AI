#!/usr/bin/env python3
"""
LibriSpeech WebSocket Server - Multi-mode ASR Server
Supports three modes:
1. streaming (default) - Real-time streaming with immediate processing (port 8001)
2. chunked - Fixed-size chunk processing with real-time simulation (port 8002)
3. original - Full file processing with real-time simulation (port 8003)
"""
import sys
import os
import argparse

# Add parent directory to path to import whisper_streaming
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'SimulStreaming'))

# Add servers directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'servers'))

# Import the factory and args functions for each mode
from librispeech_streaming_factory import librispeech_streaming_args, librispeech_streaming_factory
from librispeech_chunked_factory import librispeech_chunked_args, librispeech_chunked_factory
from librispeech_original_factory import librispeech_original_args, librispeech_original_factory
from server_test import main_websocket_server
from librispeech_chunked_server import main_chunked_server
from librispeech_original_server import main_original_server

if __name__ == "__main__":
    # Parse mode argument first
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument('--mode', type=str, default='streaming',
                             choices=['streaming', 'chunked', 'original'],
                             help='Server mode: streaming (default, port 8001), chunked (port 8002), or original (port 8003)')
    mode_args, remaining_argv = mode_parser.parse_known_args()

    # Update sys.argv to remove mode argument for downstream parsers
    sys.argv = [sys.argv[0]] + remaining_argv

    # Set default warmup file if not specified
    if '--warmup-file' not in sys.argv:
        warmup_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
        if os.path.exists(warmup_path):
            sys.argv.extend(['--warmup-file', warmup_path])

    # Set default port based on mode if not specified
    if '--port' not in sys.argv:
        if mode_args.mode == 'streaming':
            sys.argv.extend(['--port', '8001'])
        elif mode_args.mode == 'chunked':
            sys.argv.extend(['--port', '8002'])
        elif mode_args.mode == 'original':
            sys.argv.extend(['--port', '8003'])

    # Run appropriate server based on mode
    print(f"\n{'='*60}")
    if mode_args.mode == 'streaming':
        print("Starting Simul-Streaming Server")
        print("Mode: Real-time streaming with immediate processing")
        print("Default Port: 8001")
        print("Using Model: large-v2 (SimulStreaming)")
        print(f"{'='*60}\n")
        main_websocket_server(librispeech_streaming_factory, add_args=librispeech_streaming_args)
    elif mode_args.mode == 'chunked':
        # Set default chunk size if not specified
        if '--chunk-size' not in sys.argv:
            sys.argv.extend(['--chunk-size', '2.0'])
        print("Starting Chunked Upload Server")
        print("Mode: Fixed-size chunk processing with real-time simulation")
        print("Default Port: 8002")
        print("Default Chunk Size: 2.0 seconds")
        print("Using Model: large-v2 (Chunked)")
        print(f"{'='*60}\n")
        main_chunked_server(librispeech_chunked_factory, add_args=librispeech_chunked_args)
    elif mode_args.mode == 'original':
        print("Starting Whisper Original Server")
        print("Mode: Full file processing with real-time simulation")
        print("Default Port: 8003")
        print("Using Model: large-v2 (Original)")
        print(f"{'='*60}\n")
        main_original_server(librispeech_original_factory, add_args=librispeech_original_args)
