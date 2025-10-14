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

try:
    from deep_translator import GoogleTranslator
    _gt_available = True
except Exception:
    _gt_available = False


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

def translate_with_gt(text, source_lang, target_lang):
    """GoogleTranslator를 사용한 번역"""
    if not _gt_available:
        return f"(번역 필요: pip install deep-translator)"
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        return f"(번역 실패: {str(e)[:50]})"

def main():
    parser = argparse.ArgumentParser(description='Whisper STT Script')
    parser.add_argument('--model', type=str, default='base', help='Model to use')
    parser.add_argument('--audio', type=str, required=True, help='Audio file name')
    parser.add_argument('--audio_dir', type=str, default='audio_data', help='Audio directory')
    parser.add_argument('--use-whisper-translate', action='store_true', 
                        help='Whisper의 번역 기능 사용 (느림, 기본: GoogleTranslator)')
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

        # 3. 모델 로드 (1회만)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with silent_io():
            model = whisper.load_model(args.model, device=device)        

        if device == "cuda":
            model.float()

        # 4. 음성 인식 (Whisper 1회 호출 - 언어 감지 + 전사)
        with silent_io():
            result = model.transcribe(audio_data, fp16=False)
        
        lang = result.get("language", "unknown")
        text = (result.get("text") or "").strip()

        # 5. 번역 전략 선택
        if args.use_whisper_translate:
            # ===== 옵션 A: Whisper 번역 사용 (느림, 정확) =====
            if lang == "ko":
                kor_text = text
                with silent_io():
                    translated_en = model.transcribe(audio_data, task="translate", fp16=False)
                eng_text = (translated_en.get("text") or "").strip()
            elif lang == "en":
                eng_text = text
                # Whisper는 영어→한국어 직접 번역 불가, GoogleTranslator 사용
                kor_text = translate_with_gt(eng_text, 'en', 'ko')
            else:
                with silent_io():
                    to_en = model.transcribe(audio_data, task="translate", fp16=False)
                eng_text = (to_en.get("text") or "").strip()
                kor_text = translate_with_gt(eng_text, 'en', 'ko')
        else:
            # ===== 옵션 B: GoogleTranslator 사용 (빠름, 기본값) =====
            if lang == "ko":
                kor_text = text
                eng_text = translate_with_gt(kor_text, 'ko', 'en')
            elif lang == "en":
                eng_text = text
                kor_text = translate_with_gt(eng_text, 'en', 'ko')
            else:
                # 기타 언어 → 영어 → 한국어
                eng_text = translate_with_gt(text, lang, 'en')
                kor_text = translate_with_gt(eng_text, 'en', 'ko')

        # 6. 최종 출력
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
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"실행 시간: {end_time - start_time:.2f}초")