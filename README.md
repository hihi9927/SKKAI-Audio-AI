# 🎤 Whisper STT 최종 통합 버전

OpenAI Whisper를 활용한 **오프라인 음성 인식(STT)** 도구입니다.  
**macOS**와 **Windows** 모두에서 안정적으로 작동하도록 최적화되었습니다.

> 🔧 **수정된 버전**: 이 코드는 OpenAI Whisper를 기반으로 오프라인 환경과 실시간 STT용으로 최적화되었습니다.

## ✨ 주요 기능

- 🛡️ **메모리 안전성**: Segmentation Fault 방지 및 자동 메모리 관리
- 🌍 **크로스 플랫폼**: macOS (Intel/Apple Silicon) + Windows 완벽 지원  
- 📁 **스마트 파일 탐지**: 확장자 자동 감지, 한글 파일명 지원
- 🔧 **모델별 안정성**: 안전한 모델 추천 및 위험 모델 경고
- 📊 **청크 처리**: 긴 오디오 자동 분할 처리
- 🎵 **다양한 포맷**: mp3, wav, flac, m4a, ogg, mp4, aac 지원

## 🚀 빠른 시작

### 1. 초기 설정

```bash
# Python 가상환경 생성 (권장)
conda create -n whisper

# 가상환경 활성화
conda activate whisper

# 필수 패키지 설치
pip install -r requirements.txt


# 폴더 준비
'audio_data', 'weights' 폴더 직접 만들기
cd SKKAI_VOICE_AI_whisper
mkdir audio_data
mkdir weights

SKKAI_VOICE_AI_whisper/
├── STT.py              
├── audio_data/         # 오디오 파일 디렉토리. 변환하고 싶은 오디오 파일은 해당 폴더 내부에 위치시킬 것
├── weights/           # 사전학습 가중치 디렉토리
├── whisper/           # 실제 모델 폴더
├── README.md          # 이 파일
└──  ...

# 가중치 준비
python weight_download.py


### 2. 기본 사용법

```bash
# 기본 실행
python STT.py --model base --audio "파일명" --language ko

# 상세 정보와 함께
python STT.py --model medium --audio "파일명" --language ko --info

# 사용 가능한 모델 목록 확인
python STT.py --list-models
```

## 📋 명령어 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `base` | 사용할 모델 (tiny, base, small, medium, large-v1/v2/v3) |
| `--audio` | **필수** | 오디오 파일명 (확장자 제외) |
| `--language` | `ko` | 언어 코드 (ko, en, ja, zh 등) |
| `--audio_dir` | `audio_data` | 오디오 파일 디렉토리 |
| `--chunk_duration` | `30` | 청크 길이(초) |
| `--force` | - | 위험한 모델도 강제 사용 |
| `--info` | - | 상세 정보 출력 |
| `--list-models` | - | 모델 목록 출력 |

## 🎯 모델 선택 가이드

### 📊 모델별 특성

| 모델 | 크기 | 메모리 | 정확도 | 속도 | 안정성 | 권장 용도 |
|------|------|--------|--------|------|--------|----------|
| **tiny** | 100MB | ~150MB | 낮음 | 매우 빠름 | ✅ 안전 | 빠른 테스트, 실시간 |
| **base** | 300MB | ~400MB | 보통 | 빠름 | ✅ 안전 | 일반적 용도 |
| **small** | 800MB | ~1GB | 좋음 | 보통 | ✅ 안전 | 품질 중시 |
| **medium** | 2GB | ~3GB | 매우 좋음 | 느림 | ✅ 안전 | 고품질 필요 |
| **large-v1/v2/v3** | 3.5GB | ~5GB | 최고 | 매우 느림 | ⚠️ 불안정 | 최고 품질 (주의) |

### 💡 권장 사항

- **첫 사용**: `tiny` 또는 `base` 모델로 시작
- **일반 용도**: `base` 또는 `small` 모델 
- **고품질**: `medium` 모델 (메모리 충분한 경우)
- **최고 품질**: `large` 모델 (`--force` 옵션 필요, 불안정 위험)

## 📁 파일 구조

```
whisper/
├── STT.py              # 메인 실행 파일
├── audio_data/         # 오디오 파일 디렉토리
│   ├── 파일1.mp3
│   ├── 파일2.wav
│   └── ...
├── whisper/           # Whisper 라이브러리 (수정됨)
└── README.md          # 이 파일
```

## 🎵 지원 오디오 형식

- **MP3** (.mp3)
- **WAV** (.wav) 
- **FLAC** (.flac)
- **M4A** (.m4a)
- **OGG** (.ogg)
- **MP4** (.mp4)
- **AAC** (.aac)

## 🌍 지원 언어

주요 언어 코드:

| 언어 | 코드 | 언어 | 코드 |
|------|------|------|------|
| 한국어 | `ko` | 영어 | `en` |
| 일본어 | `ja` | 중국어 | `zh` |
| 스페인어 | `es` | 프랑스어 | `fr` |
| 독일어 | `de` | 러시아어 | `ru` |

> 💡 언어를 지정하지 않으면 자동 감지됩니다.

## 📖 사용 예시

### 기본 사용

```bash
# 한국어 음성 인식
python STT.py --model base --audio "회의록_2024" --language ko

# 영어 음성 인식  
python STT.py --model small --audio "interview_english" --language en

# 자동 언어 감지
python STT.py --model base --audio "multilingual_audio"
```

### 고급 사용

```bash
# 상세 정보와 함께 실행
python STT.py --model medium --audio "긴_강의" --language ko --info

# 청크 크기 조정 (긴 오디오용)
python STT.py --model small --audio "장시간_녹음" --chunk_duration 60

# 위험한 대형 모델 강제 사용
python STT.py --model large-v3 --audio "고품질_필요" --language ko --force
```

### 파일명 특수 문자 처리

```bash
# 괄호나 공백이 포함된 파일명
python STT.py --model base --audio "노이즈없는단일화자(한어)2" --language ko

# Windows에서 특수 문자
python STT.py --model base --audio "회의_2024-10-02(최종)" --language ko
```

## 🛠️ 문제 해결

### 자주 발생하는 문제

#### 1. Segmentation Fault
```bash
# 해결 방법: 더 작은 모델 사용
python STT.py --model tiny --audio "파일명" --language ko

# 또는 환경 변수 설정 후 실행 (macOS)
export OMP_NUM_THREADS=1
python STT.py --model base --audio "파일명" --language ko
```

#### 2. 파일을 찾을 수 없음
```bash
# 파일 존재 확인
ls audio_data/

# 확장자 없이 파일명만 입력
python STT.py --model base --audio "파일명만" --language ko  # ✅ 맞음
python STT.py --model base --audio "파일명.mp3" --language ko  # ❌ 틀림
```

#### 3. 메모리 부족
```bash
# 더 작은 모델 사용
python STT.py --model tiny --audio "파일명" --language ko

# 청크 크기 줄이기
python STT.py --model base --audio "파일명" --chunk_duration 15
```

#### 4. 라이브러리 오류
```bash
# 가상환경 재생성
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # macOS
# 또는
.venv\Scripts\activate     # Windows

pip install torch openai-whisper librosa
```

### 플랫폼별 최적화

#### macOS
```bash
# Apple Silicon 최적화가 자동 적용됩니다
# M1/M2 Mac에서 최적 성능을 위해 base 모델 권장
python STT.py --model base --audio "파일명" --language ko
```

#### Windows  
```bash
# GPU 사용 비활성화로 안정성 확보
# medium 모델까지 안정적 사용 가능
python STT.py --model medium --audio "파일명" --language ko
```

## 📈 성능 튜닝

### 속도 우선
```bash
python STT.py --model tiny --audio "파일명" --language ko
```

### 품질 우선  
```bash
python STT.py --model medium --audio "파일명" --language ko --info
```

### 안정성 우선
```bash
python STT.py --model base --audio "파일명" --language ko
```

## 🔧 고급 설정

### 환경 변수 (선택사항)

```bash
# macOS 최적화
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Windows 최적화  
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
```

## 📄 라이선스

이 프로젝트는 OpenAI Whisper의 수정 버전으로, 원본 라이선스를 따릅니다.

---

## 🎉 성공적인 실행 예시

```bash
$ python STT.py --model base --audio "테스트파일" --language ko

🎤 Whisper STT 최종 통합 버전 (크로스 플랫폼)
==================================================
🖥️  플랫폼: Darwin (Apple Silicon)
🔧 환경 최적화: macOS 모드
📁 오디오 파일 검색: 테스트파일
✅ 파일 발견: 테스트파일.mp3
📊 파일 크기: 0.4MB

🎵 오디오 로드 중...
✅ librosa로 오디오 로드: 5.2초
📥 base 모델 로드 중...
📊 예상 크기: 300MB, 정확도: 보통
✅ base 로드 성공

🔒 안전 모드로 음성 인식 시작...

🗣️ 음성 인식 시작 (언어: ko)...

==================================================
🎉 음성 인식 완료!
🌍 감지 언어: ko
📝 인식 결과:
   안녕하세요. 이것은 테스트 음성 파일입니다.
==================================================
🧹 모델 메모리 정리 완료
```

---

## 📋 원본 OpenAI Whisper 정보

이 수정 버전은 다음 원본을 기반으로 합니다:
- **Blog**: [OpenAI Whisper](https://openai.com/blog/whisper)
- **Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Original Repository**: [openai/whisper](https://github.com/openai/whisper)
