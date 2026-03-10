#!/usr/bin/env python3
"""
Run Qwen3 Simul-Streaming mode server
"""
import sys
import os

# 1. 현재 파일이 있는 디렉토리 (LibriSpeech 폴더)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 2. 최상위 디렉토리 (STiTy 폴더)
base_path = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, base_path)

# 3. servers 폴더와 SimulStreaming 폴더 추가
sys.path.insert(0, os.path.join(current_dir, 'servers'))
sys.path.insert(0, os.path.join(base_path, 'SimulStreaming'))

# 4. Qwen3-ASR 폴더 추가
qwen_asr_path = os.path.join(base_path, "Qwen3-ASR")
sys.path.insert(0, qwen_asr_path)

# 팀의 공통 웹소켓 서버 실행 함수 임포트
from server_test import main_websocket_server

# 우리가 새로 만들 Qwen용 팩토리 임포트
from librispeech_qwen import qwen_streaming_args, qwen_streaming_factory

if __name__ == "__main__":
    if '--port' not in sys.argv:
        sys.argv.extend(['--port', '8001'])

    print("Starting Qwen3 Streaming Server with Team Pipeline (port 8001)")
    # 팀 표준 함수에 Qwen 팩토리를 주입!
    main_websocket_server(qwen_streaming_factory, add_args=qwen_streaming_args)