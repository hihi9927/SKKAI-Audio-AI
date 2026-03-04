# LibriSpeech WebSocket ASR Server

LibriSpeech test-clean 데이터셋을 위한 다중 모드 ASR (Automatic Speech Recognition) 서버입니다.

## 목차

1. [개요](#개요)
2. [3가지 모드](#3가지-모드)
3. [설치](#설치)
4. [빠른 시작](#빠른-시작)
5. [단일 파일 처리](#단일-파일-처리)
6. [배치 처리](#배치-처리)
7. [고급 사용법](#고급-사용법)

## 개요

이 프로젝트는 SimulWhisper 기반의 ASR 서버로, 3가지 다른 처리 모드를 제공합니다:

- **Simul-Streaming**: 실시간 스트리밍 (최저 지연시간)
- **Chunked Upload**: 청크 단위 처리 (안정성)
- **Whisper Original**: 전체 파일 처리 (최고 정확도)

## 3가지 모드

### 비교표

| 모드 | 포트 | 지연시간 | 처리 단위 | 현실 시간 대기 | 용도 |
|------|------|---------|---------|--------------|------|
| **Streaming** | 8001 | 최저 | 즉시 | 없음 | 실시간 대화, 라이브 자막 |
| **Chunked** | 8002 | 중간 | 2초 청크 | 청크 길이 | 안정적인 배치 처리 |
| **Original** | 8003 | 최고 | 전체 파일 | 파일 길이 | 오프라인 전사, 최고 품질 |

### 1. Simul-Streaming Mode
```bash
python run_streaming.py
# 포트: 8001
# 특징: 오디오를 받는 즉시 처리, 부분 결과 반환
```

### 2. Chunked Upload Mode
```bash
python run_chunked.py
# 포트: 8002
# 특징: 2초 청크 단위로 처리, 현실 시간 시뮬레이션
```

### 3. Whisper Original Mode
```bash
python run_original.py
# 포트: 8003
# 특징: 전체 파일을 받은 후 한 번에 처리
```

## 설치

### 1. 기본 패키지

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
websockets>=12.0
soundfile>=0.12.1
numpy>=1.24.0
jiwer>=3.0.0
librosa>=0.10.0
```

### 2. SimulStreaming 설정

프로젝트 구조:
```
STiTy/
├── SimulStreaming/         # SimulWhisper 코드
└── LibriSpeech/           # 이 프로젝트
    ├── test-clean/        # LibriSpeech 데이터
    └── ...
```

## 빠른 시작

### 서버 실행

```bash
# 방법 1: 통합 스크립트
python librispeech_websocket_server.py --mode streaming

# 방법 2: 개별 스크립트
python run_streaming.py
python run_chunked.py
python run_original.py
```

### 클라이언트 테스트

```python
import asyncio
import websockets
import json
import numpy as np

async def test():
    async with websockets.connect('ws://localhost:8001') as ws:
        # Hello 메시지
        hello = await ws.recv()
        print(json.loads(hello))

        # 오디오 전송 (1초, 16kHz, Float32)
        audio = np.random.randn(16000).astype(np.float32)
        await ws.send(audio.tobytes())

        # 결과 수신
        result = await ws.recv()
        print(json.loads(result))

asyncio.run(test())
```

## 단일 파일 처리

### 기본 사용법

```bash
# 1. 서버 시작
python run_streaming.py

# 2. 다른 터미널에서 파일 처리
python scripts/streaming_client.py path/to/audio.wav
```

### 모드별 실행

```bash
# Streaming 모드
python run_streaming.py
python scripts/streaming_client.py audio.wav

# Chunked 모드
python run_chunked.py --chunk-size 3.0
# 클라이언트를 8002 포트로 연결

# Original 모드
python run_original.py
# 클라이언트를 8003 포트로 연결
```

## 배치 처리

### test-clean 전체 데이터셋 처리

#### 1. 단일 모드 배치 처리

```bash
# 1. 서버 시작
python run_streaming.py

# 2. 다른 터미널에서 배치 처리
python scripts/batch_process.py --mode streaming --calculate-wer
```

**옵션**:
- `--mode`: 처리 모드 (streaming/chunked/original)
- `--limit`: 처리할 파일 개수 제한
- `--calculate-wer`: WER 자동 계산
- `--output`: 출력 파일 경로

**예제**:
```bash
# 처음 10개만 테스트
python scripts/batch_process.py --mode streaming --limit 10

# 전체 처리 + WER 계산
python scripts/batch_process.py --mode chunked --calculate-wer

# 출력 파일 지정
python scripts/batch_process.py --mode original --output my_results.json

# 모든 모드 순차 실행
python scripts/batch_process.py --all-modes --calculate-wer
```

#### 2. 모든 모드 순차 실행

```bash
# 터미널 1-3: 각 모드 서버 시작
python run_streaming.py  # 포트 8001
python run_chunked.py    # 포트 8002
python run_original.py   # 포트 8003

# 터미널 4: 모든 모드 배치 처리
python scripts/batch_process.py --all-modes --calculate-wer
```

**옵션**:
```bash
# 테스트용 (각 모드당 10개씩)
python scripts/batch_process.py --all-modes --limit 10

# 현재 모드부터 이어서 실행
python scripts/batch_process.py --mode chunked --continue-modes --calculate-wer
```

### 배치 처리 결과

결과는 JSON 형식으로 저장됩니다:

```json
{
  "timestamp": "2024-11-26T15:30:00",
  "total_files": 2620,
  "results": [
    {
      "file_id": "1089-134686-0000",
      "audio_path": "test-clean/1089/134686/1089-134686-0000.flac",
      "reference": "HE HOPED THERE WOULD BE STEW FOR DINNER...",
      "hypothesis": "he hoped there would be stew for dinner...",
      "duration": 12.5
    }
  ]
}
```

### WER 계산

```bash
python scripts/batch_process.py --mode streaming --calculate-wer
```

출력:
```
============================================================
Word Error Rate (WER): 5.23%
============================================================
```

## 고급 사용법

### 서버 옵션

#### 모든 모드 공통
```bash
--host 0.0.0.0              # 서버 호스트
--port 9000                 # 포트 변경
--lan ko                    # 언어 (ko, en, etc.)
--task translate            # 번역 모드
--log-level DEBUG           # 로그 레벨
```

#### Chunked 모드 전용
```bash
--chunk-size 3.0            # 청크 크기 (초)
```

### 예제

```bash
# 한국어 + 번역 모드
python librispeech_websocket_server.py --mode streaming --lan ko --task translate

# 포트 변경
python run_streaming.py --port 9001

# 큰 청크 크기
python run_chunked.py --chunk-size 5.0

# 디버그 모드
python run_original.py --log-level DEBUG
```

## 파일 구조

```
LibriSpeech/
├── servers/                         # 서버 구현
│   ├── server_test.py              # Streaming 구현
│   ├── librispeech_chunked_server.py    # Chunked 구현
│   ├── librispeech_original_server.py   # Original 구현
│   └── librispeech_whisper.py           # ASR 백엔드
│
├── scripts/                         # 처리 스크립트
│   ├── batch_process.py            # 배치 처리 (모든 모드 지원)
│   └── streaming_client.py         # WebSocket 클라이언트
│
├── docs/                            # 문서
│   ├── CONTINUOUS_MODE.md          # 연속 모드 실행 가이드
│   ├── OUTPUT_FORMAT.md            # JSON 출력 형식
│   └── README_STREAMING.md         # 스트리밍 상세 가이드
│
├── librispeech_websocket_server.py  # 통합 서버 (모든 모드)
├── run_streaming.py                 # Streaming 모드 실행
├── run_chunked.py                   # Chunked 모드 실행
├── run_original.py                  # Original 모드 실행
│
├── test-clean/                      # LibriSpeech 데이터
│   ├── 1089/
│   ├── 121/
│   └── ...
│
├── requirements.txt                 # 의존성
└── README.md                        # 이 파일
```

## 성능 및 처리 시간

### test-clean 전체 데이터셋
- **파일 수**: 2,620개
- **총 길이**: 약 5.4시간

### 예상 처리 시간 (GPU 기준)

| 모드 | 처리 시간 | RTF* |
|------|----------|------|
| Streaming | 5.5-6시간 | ~1.0 |
| Chunked | 6-7시간 | ~1.1-1.3 |
| Original | 6-8시간 | ~1.1-1.5 |

*RTF (Real-Time Factor): 처리 시간 / 오디오 길이

## 문제 해결

### 포트가 이미 사용 중
```bash
python librispeech_websocket_server.py --mode streaming --port 9001
```

### GPU 메모리 부족
```bash
# 작은 모델 사용
python run_streaming.py --model_path path/to/small-model.pt
```

### 서버 연결 실패
```
WebSocket error: Connection refused
```
→ 서버가 실행 중인지 확인

### 배치 처리 중단
- 중간 결과는 `<output_file>.tmp`에 자동 저장됩니다
- 10개 파일마다 저장되므로 재시작 가능

## 워크플로우 예제

### 1. 빠른 테스트 (10개 파일)

```bash
# 터미널 1
python run_streaming.py

# 터미널 2
python scripts/batch_process.py --mode streaming --limit 10 --calculate-wer
```

### 2. 전체 데이터셋 처리 (단일 모드)

```bash
# 터미널 1
python run_streaming.py

# 터미널 2
python scripts/batch_process.py --mode streaming --calculate-wer
```

### 3. 모든 모드 비교

```bash
# 터미널 1-3: 서버 시작
python run_streaming.py  # 포트 8001
python run_chunked.py    # 포트 8002
python run_original.py   # 포트 8003

# 터미널 4: 모든 모드 배치 처리
python scripts/batch_process.py --all-modes --limit 100 --calculate-wer
```

## 참고 자료

- **[docs/CONTINUOUS_MODE.md](docs/CONTINUOUS_MODE.md)**: 연속 모드 실행 가이드
- **[docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md)**: JSON 출력 형식 설명
- **[docs/README_STREAMING.md](docs/README_STREAMING.md)**: 스트리밍 상세 가이드

## 라이선스

SimulWhisper 프로젝트를 기반으로 합니다.

## 문의

문제가 발생하면 이슈를 등록해주세요.
