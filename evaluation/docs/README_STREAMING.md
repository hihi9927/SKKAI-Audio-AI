# LibriSpeech Streaming Client

실시간으로 LibriSpeech 오디오 데이터를 WebSocket 서버에 스트리밍하는 클라이언트입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```bash
python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686
```

### 옵션 설명

- `--server`: WebSocket 서버 URL (기본값: `wss://edra-raspiest-eagerly.ngrok-free.dev/ws`)
- `--dataset`: LibriSpeech 데이터셋 경로 (기본값: 현재 디렉토리)
- `--subset`: 데이터셋 서브셋 (필수)
  - `test-clean`, `test-other`
  - `dev-clean`, `dev-other`
  - `train-clean-100`, `train-clean-360`, `train-other-500`
- `--speaker`: 화자 ID (필수, 예: 1089, 121, 237)
- `--chapter`: 챕터 ID (필수, 예: 134686, 134691)
- `--interval`: 청크 전송 간격 (ms, 기본값: 500)
- `--chunk-size`: 청크당 샘플 수 (기본값: 8000 = 16kHz에서 500ms)
- `--no-transcript`: 트랜스크립트 표시 안 함

## 사용 예제

### 1. 기본 스트리밍 (500ms 간격)

```bash
python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686
```

### 2. 더 빠른 스트리밍 (250ms 간격)

```bash
python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686 --interval 250
```

### 3. 더 느린 스트리밍 (1초 간격)

```bash
python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686 --interval 1000
```

### 4. 다른 서버로 스트리밍

```bash
python streaming_client.py \
  --server ws://192.168.1.100:8765 \
  --subset test-clean \
  --speaker 1089 \
  --chapter 134686
```

### 5. 트랜스크립트 없이 스트리밍

```bash
python streaming_client.py \
  --subset test-clean \
  --speaker 1089 \
  --chapter 134686 \
  --no-transcript
```

## 화자와 챕터 ID 찾기

### test-clean의 모든 화자 보기

```bash
ls test-clean/
```

### 특정 화자의 모든 챕터 보기

```bash
ls test-clean/1089/
```

### 특정 챕터의 오디오 파일 개수 확인

```bash
ls test-clean/1089/134686/*.flac | wc -l
```

## 동작 원리

1. **파일 로드**: 지정된 챕터의 모든 FLAC 파일을 순서대로 로드
2. **청크 분할**: 각 오디오 파일을 지정된 크기의 청크로 분할
3. **실시간 전송**: 지정된 간격으로 WebSocket을 통해 청크 전송
4. **연속 재생**: 파일 간 끊김 없이 연속적으로 전송

## 주요 기능

- ✅ FLAC 파일 자동 정렬 및 연속 스트리밍
- ✅ 실시간 트랜스크립트 표시
- ✅ 전송 진행률 표시
- ✅ 유연한 간격 및 청크 크기 조정
- ✅ 16kHz, 16-bit PCM 오디오 전송

## 출력 예시

```
======================================================================
Starting stream: test-clean/1089/134686
Interval: 500ms | Chunk size: 8000 samples
======================================================================

Found 25 audio files

Connected to ws://localhost:8765

[File 1/25] 1089-134686-0000
Transcript: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS
Duration: 3.52s
----------------------------------------------------------------------
  Progress: 100.0% (7/7 chunks) - Complete!
[File 2/25] 1089-134686-0001
Transcript: AND POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK SAUCE
Duration: 4.18s
----------------------------------------------------------------------
  Progress: 100.0% (9/9 chunks) - Complete!
...
```

## 트러블슈팅

### WebSocket 연결 실패

서버가 실행 중인지 확인하세요:
```bash
# 서버 실행 예시 (SimulEval)
simuleval --standalone --server-port 8765 ...
```

### 챕터 경로를 찾을 수 없음

데이터셋 경로와 ID를 확인하세요:
```bash
python streaming_client.py \
  --dataset /path/to/LibriSpeech \
  --subset test-clean \
  --speaker 1089 \
  --chapter 134686
```

### 의존성 오류

필요한 패키지를 설치하세요:
```bash
pip install websockets soundfile numpy
```
