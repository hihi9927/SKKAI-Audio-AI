import asyncio
import json
import sys
import traceback

import numpy as np
import websockets

from pipeline.qwen_pipeline import Qwen3SpeechRecognizer, Qwen3CommitPolicy
from pipeline.types import AudioSegment, SpeakerInfo, TimeRange


def _build_audio_segment(audio_array: np.ndarray) -> AudioSegment:
    # Fallback speaker/time metadata for minimal streaming flow.
    return AudioSegment(
        speaker=SpeakerInfo(speaker_id=0, speaker_language=""),
        audio=audio_array,
        time_range=TimeRange(audio_start_time_ms=0, audio_end_time_ms=0),
    )


async def handle_client(websocket):
    print("Client connected", file=sys.stderr)
    try:
        # Load model/session before telling client we're ready.
        recognizer = Qwen3SpeechRecognizer()
        policy = Qwen3CommitPolicy()
        await websocket.send(json.dumps({"type": "hello", "message": "Qwen3 Streaming Server Ready"}))

        async for message in websocket:
            if isinstance(message, bytes):
                # Client sends float32 PCM(16k) raw bytes.
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
                        await websocket.send(json.dumps({"type": "final", "original": final_text}))
                    await websocket.send(json.dumps({"type": "finish_complete"}))
                    print("Streaming session finished", file=sys.stderr)
                    break
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


async def main():
    print("Qwen3 Streaming Server listening on 8001", file=sys.stderr)
    # Preload model before accepting clients to avoid websocket ping timeout
    # during first-request initialization.
    _ = Qwen3SpeechRecognizer()
    print("Qwen3 model preload complete", file=sys.stderr)
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
