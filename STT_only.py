#!/usr/bin/env python3
import whisper
import torch
import gc
import os
import argparse
import librosa
import time

import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr


@contextmanager
def silent_io():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull, \
             redirect_stdout(devnull), \
             redirect_stderr(devnull):
            yield

def find_audio_file(audio_dir, audio_name):
    extensions = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'mp4', 'aac']
    for ext in extensions:
        exact_path = os.path.join(audio_dir, f"{audio_name}.{ext}")
        if os.path.exists(exact_path):
            return exact_path
    return None

def main():
    parser = argparse.ArgumentParser(description='Whisper STT Only (No Translation)')
    parser.add_argument('--model', type=str, default='base', 
                        help='Whisper model (tiny/base/small/medium/large)')
    parser.add_argument('--audio', type=str, required=True, 
                        help='Audio file name (without extension)')
    parser.add_argument('--audio_dir', type=str, default='audio_data', 
                        help='Audio directory path')
    parser.add_argument('--verbose', action='store_true', 
                        help='Show detailed transcription info')
    args = parser.parse_args()

    model = None
    try:
        # 1. 오디오 파일 찾기
        audio_file = find_audio_file(args.audio_dir, args.audio)
        if not audio_file:
            print(f"❌ 파일을 찾을 수 없습니다: '{args.audio}' (dir='{args.audio_dir}')")
            return

        print(f"📁 파일: {audio_file}")

        # 2. 오디오 로드
        print("🔊 오디오 로딩 중...")
        with silent_io():
            audio_data = librosa.load(audio_file, sr=16000, mono=True)[0]
        
        duration = len(audio_data) / 16000
        print(f"⏱️  길이: {duration:.2f}초")

        # 3. 모델 로드
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 모델 로딩 중... (model={args.model}, device={device})")
        
        with silent_io():
            model = whisper.load_model(args.model, device=device)
        
        if device == "cuda":
            model.float()

        # 4. 음성 인식 (STT만 수행)
        print("🎤 음성 인식 중...")
        with silent_io():
            result = model.transcribe(audio_data, fp16=False)
        
        # 5. 결과 출력
        lang = result.get("language", "unknown")
        text = (result.get("text") or "").strip()
        
        print("\n" + "="*60)
        print(f"🌍 감지 언어: {lang}")
        print("="*60)
        print(f"📝 전사 결과:\n{text}")
        print("="*60)
        
        # 6. 상세 정보 출력 (옵션)
        if args.verbose and "segments" in result:
            print(f"\n📊 세그먼트 정보 (총 {len(result['segments'])}개):")
            print("-"*60)
            for i, seg in enumerate(result['segments'], 1):
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                seg_text = seg.get('text', '').strip()
                print(f"{i}. [{start:.2f}s ~ {end:.2f}s] {seg_text}")
            print("-"*60)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 메모리 정리
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"실행 시간: {end_time - start_time:.2f}초")