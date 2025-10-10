#!/usr/bin/env python3
import whisper
import torch
import gc
import os
import argparse
import librosa
import platform
from contextlib import contextmanager

def find_audio_file(audio_dir, audio_name):
    extensions = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'mp4', 'aac']
    for ext in extensions:
        exact_path = os.path.join(audio_dir, f"{audio_name}.{ext}")
        if os.path.exists(exact_path):
            return exact_path
    return None

def main():
    parser = argparse.ArgumentParser(description='Whisper STT Script')
    parser.add_argument('--model', type=str, default='base', help='Model to use')
    parser.add_argument('--audio', type=str, required=True, help='Audio file name')
    parser.add_argument('--audio_dir', type=str, default='audio_data', help='Audio directory')
    args = parser.parse_args()

    print("=" * 50)
    print("🎤 Whisper STT 시작")
    print("=" * 50)

    model = None
    try:
        # 1. 오디오 파일 찾기
        print(f"📁 오디오 파일 검색: {args.audio}")
        audio_file = find_audio_file(args.audio_dir, args.audio)
        if not audio_file:
            print(f"❌ '{args.audio}' 파일을 '{args.audio_dir}' 폴더에서 찾을 수 없습니다.")
            return

        print(f"✅ 파일 발견: {os.path.basename(audio_file)}")

        # 2. 오디오 로드
        print("\n🎵 오디오 로드 중...")
        audio_data = librosa.load(audio_file, sr=16000, mono=True)[0]
        print("✅ 오디오 로드 성공")

        # 3. 모델 로드
        print(f"\n📥 '{args.model}' 모델 로드 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(args.model, device=device)

        # GPU 사용 시 모델을 float32로 변환 (안정성 확보)
        if device == "cuda":
            model.float()
        print("✅ 모델 로드 성공")

        # 4. 음성 인식
        print("\n🗣️ 음성 인식 시작...")
        result = model.transcribe(audio_data, language='ko', fp16=False) # fp16=False로 안정성 확보

        # 5. 결과 출력
        print("\n" + "="*50)
        print("🎉 음성 인식 완료!")
        print(f"📝 인식 결과: {result['text'].strip()}")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 치명적 에러 발생: {e}")

    finally:
        # 메모리 정리
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("\n🧹 스크립트 종료.")

if __name__ == "__main__":
    main()