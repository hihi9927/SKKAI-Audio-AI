#!/usr/bin/env python3
"""
Whisper STT 최종 통합 버전 (크로스 플랫폼)
- 메모리 안전성 최적화
- 오디오 파일 자동 탐지  
- 모델별 안정성 체크
- macOS + Windows 최적화
- 실시간 STT 지원
- 청크 기반 처리
- 종합적 에러 처리
"""

import whisper
import torch
import gc
import os
import argparse
import glob
import signal
import sys
import platform
from contextlib import contextmanager

# ========================================
# 안전성 설정
# ========================================

# 안전한 모델 vs 위험한 모델
SAFE_MODELS = ['tiny', 'base', 'small', 'medium']
RISKY_MODELS = ['large-v1', 'large-v2', 'large-v3']

# 모델별 상세 정보
MODEL_INFO = {
    'tiny': {'size_mb': 39, 'accuracy': '낮음', 'speed': '매우 빠름', 'memory': '100MB', 'stable': True},
    'base': {'size_mb': 144, 'accuracy': '보통', 'speed': '빠름', 'memory': '300MB', 'stable': True},
    'small': {'size_mb': 488, 'accuracy': '좋음', 'speed': '보통', 'memory': '800MB', 'stable': True},
    'medium': {'size_mb': 1550, 'accuracy': '매우 좋음', 'speed': '느림', 'memory': '2GB', 'stable': True},
    'large-v1': {'size_mb': 2950, 'accuracy': '최고', 'speed': '매우 느림', 'memory': '3.5GB', 'stable': False},
    'large-v2': {'size_mb': 2950, 'accuracy': '최고', 'speed': '매우 느림', 'memory': '3.5GB', 'stable': False},
    'large-v3': {'size_mb': 2950, 'accuracy': '최고', 'speed': '매우 느림', 'memory': '3.5GB', 'stable': False},
}

# ========================================
# 메모리 관리 클래스
# ========================================

class WhisperMemoryManager:
    """Whisper 전용 메모리 관리"""
    
    def __init__(self):
        self.setup_environment()
        
    def setup_environment(self):
        """크로스 플랫폼 환경 최적화 (macOS + Windows)"""
        
        # 플랫폼 감지
        current_platform = platform.system().lower()
        is_apple_silicon = platform.machine() == 'arm64'
        is_windows = current_platform == 'windows'
        is_macos = current_platform == 'darwin'
        
        # 공통 환경 설정 (모든 플랫폼)
        common_env = {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 
            'NUMEXPR_NUM_THREADS': '1',
            'TOKENIZERS_PARALLELISM': 'false',
            'PYTORCH_ENABLE_MPS_FALLBACK': '1',
            'OMP_MAX_ACTIVE_LEVELS': '1',
            'OMP_NESTED': 'FALSE',
        }
        
        # macOS 전용 설정
        if is_macos:
            macos_env = {
                'VECLIB_MAXIMUM_THREADS': '1',     # Apple Accelerate 제한
                'OPENBLAS_NUM_THREADS': '1',       # OpenBLAS 제한
                'MKL_THREADING_LAYER': 'SEQUENTIAL',
            }
            common_env.update(macos_env)
            
            # Apple Silicon 추가 최적화
            if is_apple_silicon:
                common_env.update({
                    'MALLOC_CHECK_': '0',
                    'PYTHONMALLOC': 'malloc',
                })
        
        # Windows 전용 설정  
        elif is_windows:
            windows_env = {
                'MKL_THREADING_LAYER': 'INTEL',    # Windows에서 Intel 레이어
                'CUDA_VISIBLE_DEVICES': '',        # GPU 비활성화 (안정성)
            }
            common_env.update(windows_env)
        
        # 환경 변수 적용
        os.environ.update(common_env)
        
        # PyTorch 설정 (플랫폼 공통)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # 플랫폼별 PyTorch 백엔드 설정
        if hasattr(torch.backends, 'openmp'):
            torch.backends.openmp.is_available = lambda: False
            
        # 디버그 정보 출력
        print(f"🖥️  플랫폼: {current_platform.title()}" + 
              (f" (Apple Silicon)" if is_apple_silicon else ""))
        print(f"🔧 환경 최적화: {'macOS' if is_macos else 'Windows' if is_windows else 'Linux'} 모드")
        
    def force_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# ========================================
# 안전한 모델 관리
# ========================================

@contextmanager
def safe_model_context(model_name, force_load=False):
    """안전한 모델 로드/언로드 컨텍스트"""
    model = None
    memory_manager = WhisperMemoryManager()
    
    try:
        # 안정성 체크
        if not force_load and model_name in RISKY_MODELS:
            print(f"⚠️ {model_name}은 현재 시스템에서 불안정할 수 있습니다")
            alt_model = suggest_safe_alternative(model_name)
            print(f"💡 권장 대안: {alt_model}")
            
            confirm = input(f"그래도 {model_name}을 사용하시겠습니까? (y/N): ")
            if confirm.lower() != 'y':
                print(f"🔄 {alt_model} 모델로 변경합니다")
                model_name = alt_model
        
        # 메모리 정리 후 로드
        memory_manager.force_cleanup()
        
        info = MODEL_INFO.get(model_name, {})
        print(f"📥 {model_name} 모델 로드 중...")
        print(f"📊 예상 크기: {info.get('memory', 'Unknown')}, 정확도: {info.get('accuracy', 'Unknown')}")
        
        model = whisper.load_model(model_name)
        
        # 실시간 최적화 적용
        if hasattr(model, 'optimize_for_realtime'):
            model.optimize_for_realtime()
            
        print(f"✅ {model_name} 로드 성공")
        
        yield model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        
        # 에러 유형별 처리
        if "out of memory" in str(e).lower():
            print("💾 메모리 부족 - 더 작은 모델을 사용하세요")
        elif any(word in str(e).lower() for word in ["segmentation", "fault", "crash"]):
            print("🔧 Segmentation Fault - 시스템 불안정")
            print("💡 해결책: 시스템 재시작 또는 더 작은 모델 사용")
        else:
            print(f"🐛 예상치 못한 에러: {type(e).__name__}")
        
        raise
        
    finally:
        if model is not None:
            del model
        memory_manager.force_cleanup()
        print("🧹 모델 메모리 정리 완료")

def suggest_safe_alternative(risky_model):
    """위험한 모델에 대한 안전한 대안 제시"""
    if risky_model.startswith('large'):
        return 'medium'  # 가장 큰 안전한 모델
    return 'base'

# ========================================
# 오디오 파일 처리
# ========================================

def find_audio_file(audio_dir, audio_name):
    """
    오디오 파일을 찾는 함수 (확장자 자동 감지)
    괄호가 포함된 한글 파일명도 처리
    """
    extensions = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'mp4', 'aac']
    
    # 1. 정확한 파일명으로 시도
    for ext in extensions:
        exact_path = os.path.join(audio_dir, f"{audio_name}.{ext}")
        if os.path.exists(exact_path):
            return exact_path
    
    # 2. glob 패턴으로 유사한 파일 찾기
    for ext in extensions:
        pattern = os.path.join(audio_dir, f"*{audio_name}*.{ext}")
        try:
            matches = glob.glob(pattern)
            if matches:
                print(f"💡 유사한 파일 발견: {os.path.basename(matches[0])}")
                return matches[0]
        except:
            continue
    
    # 3. 디렉토리 전체 검색
    try:
        all_files = []
        for ext in extensions:
            pattern = os.path.join(audio_dir, f"*.{ext}")
            all_files.extend(glob.glob(pattern))
        
        # 파일명에 검색어가 포함된 파일 찾기
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if audio_name in filename:
                print(f"💡 부분 일치 파일 발견: {filename}")
                return file_path
                
    except Exception as e:
        print(f"⚠️ 파일 검색 중 오류: {e}")
    
    return None

def load_audio_safe(file_path):
    """안전한 오디오 로드 (여러 방법 시도)"""
    
    # 방법 1: librosa 사용
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        print(f"✅ librosa로 오디오 로드: {len(audio)/16000:.1f}초")
        return audio
    except ImportError:
        print("📦 librosa 미설치 - 다른 방법 시도")
    except Exception as e:
        print(f"⚠️ librosa 실패: {e}")
    
    # 방법 2: whisper 내장 방법 사용
    try:
        from whisper.audio import load_audio
        audio = load_audio(file_path)
        print(f"✅ whisper 내장으로 오디오 로드: {len(audio)/16000:.1f}초")
        return audio
    except Exception as e:
        print(f"❌ whisper 내장 방법도 실패: {e}")
        raise

# ========================================
# 안전한 음성 인식
# ========================================

def transcribe_safe(model, audio_data, language='ko', chunk_duration=30):
    """
    안전한 음성 인식 (Segfault 방지 강화)
    """
    try:
        # Segfault 방지를 위한 사전 설정
        import torch
        torch.set_num_threads(1)  # 재확인
        
        # 메모리 사전 정리
        gc.collect()
        
        print("🔒 안전 모드로 음성 인식 시작...")
        
        # 기본 처리 시도 (최소 옵션)
        result = whisper.transcribe(
            model,
            audio_data,
            language=language,
            fp16=False,              # Apple Silicon 안정성
            verbose=False,           # 출력 최소화
            beam_size=1,             # 메모리 절약
            temperature=0,           # 결정론적 결과
            condition_on_previous_text=False,  # 메모리 절약
            word_timestamps=False,    # 타임스탬프 비활성화 (안정성)
            no_speech_threshold=0.6,  # 무음 구간 처리
            logprob_threshold=-1.0,   # 로그 확률 임계값
        )
        
        return result
        
    except Exception as e:
        print(f"⚠️ 기본 처리 실패: {e}")
        
        # 청크 처리로 재시도
        return transcribe_chunked(model, audio_data, language, chunk_duration)

def transcribe_chunked(model, audio_data, language, chunk_duration=30):
    """청크 기반 안전 처리"""
    
    try:
        SAMPLE_RATE = 16000
        total_duration = len(audio_data) / SAMPLE_RATE
        
        print(f"🔄 청크 처리 모드 ({chunk_duration}초 단위)")
        print(f"🎵 총 길이: {total_duration:.1f}초")
        
        if total_duration <= chunk_duration:
            # 짧은 오디오는 한 번에
            result = whisper.transcribe(
                model, audio_data,
                language=language,
                fp16=False,
                verbose=False
            )
            return result
        
        # 청크로 분할 처리
        chunk_size = chunk_duration * SAMPLE_RATE
        texts = []
        segments = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk_start_sec = i / SAMPLE_RATE
            chunk_audio = audio_data[i:i+chunk_size]
            
            print(f"   📊 처리 중: {chunk_start_sec:.1f}초...")
            
            chunk_result = whisper.transcribe(
                model, chunk_audio,
                language=language,
                fp16=False,
                verbose=False,
                word_timestamps=True
            )
            
            texts.append(chunk_result['text'])
            
            # 타임스탬프 조정
            if 'segments' in chunk_result:
                for segment in chunk_result['segments']:
                    segment['start'] += chunk_start_sec
                    segment['end'] += chunk_start_sec
                segments.extend(chunk_result['segments'])
            
            # 청크마다 메모리 정리
            gc.collect()
        
        return {
            'text': ' '.join(texts),
            'language': language,
            'segments': segments
        }
        
    except Exception as e:
        print(f"❌ 청크 처리도 실패: {e}")
        raise

# ========================================
# 신호 처리
# ========================================

def signal_handler(signum, frame):
    """안전한 종료 처리"""
    print(f"\n 신호 {signum} 받음 - 안전하게 종료 중...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

# ========================================
# 메인 함수
# ========================================

def main():
    # 크로스 플랫폼 신호 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C (모든 플랫폼)
    
    # Windows에서는 SIGTERM이 제한적이므로 조건부 등록
    if platform.system().lower() != 'windows':
        signal.signal(signal.SIGTERM, signal_handler)  # Unix/Linux/macOS만
    
    # 인자 파싱
    parser = argparse.ArgumentParser(
        description='Whisper STT 최종 통합 버전',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시 (크로스 플랫폼):
  # macOS/Linux
  python STT.py --model base --audio "파일명" --language ko
  python STT.py --model tiny --audio "노이즈없는단일화자(한어)2" --language ko --info
  
  # Windows  
  python STT.py --model base --audio "파일명" --language ko
  python STT.py --model tiny --audio "노이즈없는단일화자(한어)2" --language ko --info
  
  # 공통
  python STT.py --list-models
        """
    )
    
    parser.add_argument('--model', type=str, default='base',
                       choices=list(MODEL_INFO.keys()),
                       help='사용할 모델 (기본: base)')
    parser.add_argument('--audio', type=str, required=True,
                       help='오디오 파일명 (확장자 제외)')
    parser.add_argument('--language', type=str, default='ko',
                       help='언어 코드 (기본: ko)')
    parser.add_argument('--audio_dir', type=str, default='audio_data',
                       help='오디오 파일 디렉토리 (기본: audio_data)')
    parser.add_argument('--chunk_duration', type=int, default=30,
                       help='청크 길이(초) (기본: 30)')
    parser.add_argument('--force', action='store_true',
                       help='위험한 모델도 강제 사용')
    parser.add_argument('--info', action='store_true',
                       help='상세 정보 출력')
    parser.add_argument('--list-models', action='store_true',
                       help='사용 가능한 모델 목록 출력')
    
    args = parser.parse_args()
    
    print("🎤 Whisper STT 최종 통합 버전")
    print("=" * 50)
    
    # 모델 목록 출력
    if args.list_models:
        print("📋 사용 가능한 모델:")
        for model, info in MODEL_INFO.items():
            stability = "✅ 안전" if info['stable'] else "⚠️ 불안정"
            print(f"  {model:10s}: {info['memory']:>6s} | {info['accuracy']:>8s} | {stability}")
        return
    
    try:
        # 오디오 파일 찾기
        print(f"📁 오디오 파일 검색: {args.audio}")
        audio_file = find_audio_file(args.audio_dir, args.audio)
        
        if not audio_file:
            print(f"❌ 오디오 파일을 찾을 수 없습니다: {args.audio}")
            print(f"� 검색 디렉토리: {os.path.abspath(args.audio_dir)}")
            print("💡 해결책:")
            print("  1. 파일명 정확히 확인")
            print("  2. 괄호 포함 시 따옴표 사용")
            print(f'     python STT.py --model {args.model} --audio "{args.audio}"')
            return
        
        print(f"✅ 파일 발견: {os.path.basename(audio_file)}")
        print(f"📊 파일 크기: {os.path.getsize(audio_file)/(1024*1024):.1f}MB")
        
        # 상세 정보 출력
        if args.info:
            info = MODEL_INFO[args.model]
            print(f"\n📋 모델 정보:")
            print(f"  🏷️  이름: {args.model}")
            print(f"  💾 메모리: {info['memory']}")
            print(f"  🎯 정확도: {info['accuracy']}")
            print(f"  ⚡ 속도: {info['speed']}")
            print(f"  🛡️  안정성: {'안전' if info['stable'] else '불안정'}")
        
        # 오디오 로드
        print(f"\n🎵 오디오 로드 중...")
        audio_data = load_audio_safe(audio_file)
        
        # 모델 로드 및 음성 인식
        with safe_model_context(args.model, args.force) as model:
            print(f"\n🗣️ 음성 인식 시작 (언어: {args.language})...")
            
            result = transcribe_safe(
                model, 
                audio_data, 
                args.language, 
                args.chunk_duration
            )
            
            # 결과 출력
            print("\n" + "="*50)
            print("🎉 음성 인식 완료!")
            print(f"🌍 감지 언어: {result.get('language', args.language)}")
            print(f"📝 인식 결과:")
            print(f"   {result['text']}")
            
            # 세그먼트 정보
            if args.info and 'segments' in result and result['segments']:
                print(f"\n⏱️ 세그먼트 정보 (총 {len(result['segments'])}개):")
                for i, segment in enumerate(result['segments'][:5]):  # 처음 5개만
                    start, end = segment['start'], segment['end']
                    text = segment['text'].strip()
                    print(f"  {i+1:2d}. [{start:5.1f}s-{end:5.1f}s] {text}")
                if len(result['segments']) > 5:
                    print(f"     ... (나머지 {len(result['segments'])-5}개)")
            
            print("="*50)
            
    except KeyboardInterrupt:
        print("\n🛑 사용자가 중단했습니다")
        
    except Exception as e:
        print(f"\n❌ 치명적 에러: {e}")
        print("\n💡 해결 방법:")
        print("1. 모델 목록 확인: --list-models")
        print("2. 더 작은 모델: --model tiny")
        print("3. 강제 실행: --force")
        print("4. 라이브러리 설치: pip install librosa")
        print("5. 시스템 재시작")
        
        if args.info:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()