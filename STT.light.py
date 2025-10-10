#!/usr/bin/env python3
import whisper
import torch
import gc
import os
import argparse
import librosa

import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr

try:
    from googletrans import Translator  # 영어 → 한국어 번역용 (온라인)
    _gt_available = True
except Exception:
    _gt_available = False

# 모든 출력/경고를 잠그는 컨텍스트

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
    parser = argparse.ArgumentParser(description='Whisper STT Script')
    parser.add_argument('--model', type=str, default='base', help='Model to use')
    parser.add_argument('--audio', type=str, required=True, help='Audio file name')
    parser.add_argument('--audio_dir', type=str, default='audio_data', help='Audio directory')
    args = parser.parse_args()

    model = None
    try:
        # 1. 오디오 파일 찾기
        audio_file = find_audio_file(args.audio_dir, args.audio)
        if not audio_file:
            print(f"파일없음: '{args.audio}' (dir='{args.audio_dir}')")
            return

        # 2. 오디오 로드
        with silent_io():
            audio_data = librosa.load(audio_file, sr=16000, mono=True)[0]        

        # 3. 모델 로드
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with silent_io():
            model = whisper.load_model(args.model, device=device)        

        # GPU 사용 시 모델을 float32로 변환 (안정성 확보)
        if device == "cuda":
            model.float()

        # 4. 음성 인식
        with silent_io():
            result = model.transcribe(audio_data, language='ko', fp16=False)  # fp16=False로 안정성 확보
        
        # 🔹 언어 감지 결과
        lang = result.get("language", "unknown")
        text = (result.get("text") or "").strip()

        if lang == "ko":
            kor_text = text
            # 원문이 한국어면 영어 번역은 whisper translate로
            with silent_io():
                translated_en = model.transcribe(audio_data, task="translate", fp16=False)
            eng_text = (translated_en.get("text") or "").strip()
        else:
            # 원문이 영어면: 영어 전사 확보
            if lang == "en":
                eng_text = text
            else:
                # 그 외 언어면 영어 번역 먼저 확보
                with silent_io():
                    to_en = model.transcribe(audio_data, task="translate", fp16=False)
                eng_text = (to_en.get("text") or "").strip()

            # 영어 → 한국어 번역 (googletrans 있으면 사용)
            if _gt_available:
                try:
                    kor_text = Translator().translate(eng_text, src="en", dest="ko").text
                except Exception:
                    kor_text = "(한국어 번역 실패)"
            else:
                kor_text = "(한국어 번역 필요: googletrans 미설치)"

        # 5. 최종 출력: 감지언어 + 한국어/영어 자막
        print(f"[언어 감지] {lang}")
        print(f"[한국어] {kor_text}")
        print(f"[영어] {eng_text}")


    except Exception as e:
        print(f"에러: {e}")

    finally:
        # 메모리 정리
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()