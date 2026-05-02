#!/usr/bin/env python3
"""KsponSpeech 오디오 한 파일을 서버로 전송하고 결과 출력."""
import argparse
import asyncio
import json
import numpy as np
import websockets

SAMPLING_RATE = 16000
DEFAULT_AUDIO = "/home/ubuntu/STiTy/evaluation/KsponSpeech/data/eval_clean/KsponSpeech_E00001.pcm"


async def run(uri, audio_path, chunk_ms, target_lang):
    audio = np.fromfile(audio_path, dtype=np.int16)
    duration = len(audio) / SAMPLING_RATE
    print(f"오디오: {audio_path}  ({duration:.2f}s, {len(audio)} samples)")

    chunk_size = int(chunk_ms / 1000 * SAMPLING_RATE)

    async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as ws:
        # hello
        msg = json.loads(await ws.recv())
        print(f"[서버] {msg}")

        # start
        await ws.send(json.dumps({"type": "start", "lang": "ko", "targetLang": target_lang}))
        msg = json.loads(await ws.recv())
        print(f"[서버] {msg}")

        # 오디오 전송 (실시간 페이싱)
        print(f"[전송] chunk={chunk_ms}ms ...")
        origin = asyncio.get_event_loop().time()
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await ws.send(chunk.tobytes())
            sent_sec = (i + len(chunk)) / SAMPLING_RATE
            wait = origin + sent_sec - asyncio.get_event_loop().time()
            if wait > 0:
                await asyncio.sleep(wait)

        # 묵음 1.5초 추가 (VAD 트리거)
        silence = np.zeros(int(SAMPLING_RATE * 1.5), dtype=np.int16)
        await ws.send(silence.tobytes())

        # finish
        await ws.send(json.dumps({"type": "finish"}))
        print("[전송] finish 전송 완료, 결과 대기...\n")

        # 결과 수신 (10초 대기)
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                if not isinstance(raw, str):
                    continue
                data = json.loads(raw)
                t = data.get("type", "")
                if t == "partial":
                    print(f"  [partial] {data.get('original', '')}")
                elif t == "final":
                    print(f"\n  [final] reason={data.get('commitReason')}")
                    print(f"    원문: {data.get('original', '')}")
                    print(f"    번역: {data.get('translation', '')}")
                    print(f"    fsl={data.get('fsl_sec')}  encode={data.get('encode_sec')}  decode={data.get('decode_sec')}  final_decode={data.get('final_decode_sec')}  trans={data.get('trans_sec')}")
                elif t == "ready":
                    break  # finish 후 ready = 재시작 신호
        except asyncio.TimeoutError:
            print("\n[타임아웃] 더 이상 메시지 없음")

        await ws.send(json.dumps({"type": "stop"}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default=DEFAULT_AUDIO)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--chunk-ms", type=int, default=200)
    p.add_argument("--target-lang", default="en")
    args = p.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    asyncio.run(run(uri, args.audio, args.chunk_ms, args.target_lang))


if __name__ == "__main__":
    main()
