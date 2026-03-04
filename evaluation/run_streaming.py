#!/usr/bin/env python3
"""
Run Simul-Streaming mode server
Real-time streaming ASR with immediate processing
Default port: 8001
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
from librispeech_streaming_factory import librispeech_streaming_args, librispeech_streaming_factory
from server_test import main_websocket_server

if __name__ == "__main__":
    # Set default warmup file if not specified
    if '--warmup-file' not in sys.argv:
        warmup_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
        if os.path.exists(warmup_path):
            sys.argv.extend(['--warmup-file', warmup_path])

    # Set default port for streaming mode
    if '--port' not in sys.argv:
        sys.argv.extend(['--port', '8001'])

    print("Starting Simul-Streaming Server (port 8001)")
    print("Using model: large-v2 (SimulStreaming)")
    main_websocket_server(librispeech_streaming_factory, add_args=librispeech_streaming_args)
