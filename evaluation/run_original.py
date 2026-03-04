#!/usr/bin/env python3
"""
Run Whisper Original mode server
Processes complete audio file after receiving all data with real-time simulation
Default port: 8003
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
from librispeech_original_factory import librispeech_original_args, librispeech_original_factory
from librispeech_original_server import main_original_server

if __name__ == "__main__":
    # Set default warmup file if not specified
    if '--warmup-file' not in sys.argv:
        warmup_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
        if os.path.exists(warmup_path):
            sys.argv.extend(['--warmup-file', warmup_path])

    # Set default port for original mode
    if '--port' not in sys.argv:
        sys.argv.extend(['--port', '8003'])

    print("Starting Whisper Original Server (port 8003)")
    print("Using model: large-v2 (Original)")
    main_original_server(librispeech_original_factory, add_args=librispeech_original_args)
