import sys
import os

# 현재 파일(run_qwen_streaming.py)의 위치를 기준으로 최상위 폴더(STiTy) 경로를 계산합니다.
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. STiTy 폴더를 경로에 추가 (다른 서버 모듈용)
sys.path.append(base_path)

# 2. Qwen3-ASR 폴더를 경로에 추가 (qwen_asr 패키지를 찾기 위함)
qwen_asr_path = os.path.join(base_path, "Qwen3-ASR")
sys.path.append(qwen_asr_path)

import asyncio
import json
import traceback
import numpy as np
import websockets
from servers.librispeech_qwen import Qwen3SpeechRecognizer, Qwen3CommitPolicy
from pipeline.types import AudioSegment, SpeakerInfo, TimeRange


def _build_audio_segment(audio_array: np.ndarray) -> AudioSegment:
    return AudioSegment(
        speaker=SpeakerInfo(speaker_id=0, speaker_language=""),
        audio=audio_array,
        time_range=TimeRange(audio_start_time_ms=0, audio_end_time_ms=0),
    )

async def handle_client(websocket):
    print("Client connected", file=sys.stderr)
    try:
        global global_recognizer
        global_recognizer.state = global_recognizer.model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )
        recognizer = global_recognizer 
        policy = Qwen3CommitPolicy()
        
        await websocket.send(json.dumps({"type": "hello", "message": "Qwen3 Streaming Server Ready"}))

        async for message in websocket:
            if isinstance(message, bytes):
                audio_array = np.frombuffer(message, dtype=np.float32)
                segment = _build_audio_segment(audio_array)

                token = recognizer.transcribe(segment)
                committed = policy.process_token(token)

                if committed and committed.text:
                    result_msg = {
                        "type": "final",
                        "original": committed.text,
                    }
                    await websocket.send(json.dumps(result_msg))
            else:
                data = json.loads(message)
                if data.get("type") == "finish":
                    recognizer.model.finish_streaming_transcribe(recognizer.state)
                    final_text = recognizer.state.text
                    if final_text:
                        remaining_words = policy._extract_new_words(policy.last_committed_text, final_text)
                        if remaining_words:
                            await websocket.send(json.dumps({"type": "final", "original": remaining_words}))
                    await websocket.send(json.dumps({"type": "finish_complete"}))
                    print("Streaming session finished", file=sys.stderr)
                    break
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

global_recognizer = None

async def main():
    global global_recognizer
    print("Qwen3 모델을 GPU에 올립니다...", file=sys.stderr)
    global_recognizer = Qwen3SpeechRecognizer()  
    
    print("Qwen3 Streaming Server listening on 8001", file=sys.stderr)
    async with websockets.serve(
        handle_client,
        "0.0.0.0",
        8001,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())