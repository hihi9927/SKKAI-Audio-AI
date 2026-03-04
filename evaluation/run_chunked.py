#!/usr/bin/env python3
"""
Run Chunked Upload mode server
Processes audio in fixed-size chunks with real-time simulation
Default port: 8002
Default chunk size: 2.0 seconds
"""
import sys
import os

# Add parent directory to path to import whisper_streaming
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'SimulStreaming'))

# Add servers directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'servers'))

# Import the factory and args functions
from librispeech_chunked_factory import librispeech_chunked_args, librispeech_chunked_factory
from librispeech_chunked_server import main_chunked_server

if __name__ == "__main__":
    # Set default warmup file if not specified
    if '--warmup-file' not in sys.argv:
        warmup_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
        if os.path.exists(warmup_path):
            sys.argv.extend(['--warmup-file', warmup_path])

    # Set default port for chunked mode
    if '--port' not in sys.argv:
        sys.argv.extend(['--port', '8002'])

    # Set default chunk size if not specified
    if '--chunk-size' not in sys.argv:
        sys.argv.extend(['--chunk-size', '2.0'])

    print("Starting Chunked Upload Server (port 8002, chunk size: 2.0s)")
    print("Using model: large-v2 (Chunked)")
    main_chunked_server(librispeech_chunked_factory, add_args=librispeech_chunked_args)
