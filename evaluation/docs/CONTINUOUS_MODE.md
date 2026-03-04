# 연속 모드 실행 가이드

## 개요

`batch_process.py`는 여러 모드를 자동으로 연속 실행할 수 있는 기능을 제공합니다.
이 기능을 사용하면 각 서버를 미리 실행해 놓고, 스크립트가 자동으로 모든 모드를 순차 처리합니다.

## 사용 방법

### 1. **단일 모드 (기본)**

```bash
python batch_process.py --mode streaming
```

특징:
- streaming 모드만 처리
- 포트 8001 사용

---

### 2. **모든 모드 자동 실행** (`--all-modes`)

```bash
python batch_process.py --all-modes
```

실행 순서:
1. **Streaming** (포트 8001)
2. **Chunked** (포트 8002)
3. **Original** (포트 8003)

특징:
- 세 모드를 자동으로 순차 실행
- 각 모드 완료 후 3초 대기
- 하나의 `results.json`에 모든 결과 저장

---

### 3. **이어서 실행** (`--continue-modes`)

특정 모드부터 시작해서 나머지 모드 모두 실행:

```bash
# Streaming부터 시작 (streaming -> chunked -> original)
python batch_process.py --mode streaming --continue-modes

# Chunked부터 시작 (chunked -> original)
python batch_process.py --mode chunked --continue-modes

# Original만 (original)
python batch_process.py --mode original --continue-modes
```

특징:
- 지정한 모드부터 마지막 모드까지 실행
- 이미 처리된 모드는 건너뛰고 싶을 때 유용

---

## 사전 준비

### 모든 서버 실행

연속 모드를 사용하려면 **먼저 모든 서버를 실행**해야 합니다:

```bash
# 터미널 1: Streaming 서버
python run_streaming.py

# 터미널 2: Chunked 서버
python run_chunked.py

# 터미널 3: Original 서버
python run_original.py
```

또는 백그라운드로 실행:

```bash
# Windows (PowerShell)
Start-Process python -ArgumentList "run_streaming.py" -WindowStyle Hidden
Start-Process python -ArgumentList "run_chunked.py" -WindowStyle Hidden
Start-Process python -ArgumentList "run_original.py" -WindowStyle Hidden

# Linux/Mac
python run_streaming.py &
python run_chunked.py &
python run_original.py &
```

---

## 실행 예시

### 예시 1: 전체 데이터셋, 모든 모드

```bash
# 1. 서버 실행 (3개 터미널)
python run_streaming.py  # 포트 8001
python run_chunked.py    # 포트 8002
python run_original.py   # 포트 8003

# 2. 배치 처리 (자동으로 3개 모드 순차 실행)
python batch_process.py --all-modes --calculate-wer
```

**출력**:
```
======================================================================
MODE 1/3: STREAMING
======================================================================
Connecting to: ws://localhost:8001
Processing 2620 audio files from test-clean dataset
...
======================================================================
RESULTS SUMMARY - STREAMING MODE
======================================================================
Total files processed: 2620
Overall WER: 5.23%
...

======================================================================
Completed STREAMING mode
Next: CHUNKED mode (port 8002)
Make sure the chunked server is running on port 8002
======================================================================

Waiting 3 seconds before next mode...

======================================================================
MODE 2/3: CHUNKED
======================================================================
...

======================================================================
MODE 3/3: ORIGINAL
======================================================================
...

======================================================================
ALL MODES COMPLETED
======================================================================
  STREAMING: 2620 files processed
  CHUNKED: 2620 files processed
  ORIGINAL: 2620 files processed

Results saved to: results.json
======================================================================
```

---

### 예시 2: 테스트 (각 모드당 10개씩)

```bash
python batch_process.py --all-modes --limit 10 --calculate-wer
```

특징:
- 각 모드당 10개 파일만 처리
- 빠른 테스트에 적합
- 총 30개 파일 처리 (10 × 3 모드)

---

### 예시 3: Streaming 완료 후 나머지 모드 실행

```bash
# 1단계: Streaming만 처리
python batch_process.py --mode streaming --calculate-wer

# 2단계: Chunked와 Original 추가 처리
python batch_process.py --mode chunked --continue-modes --calculate-wer
```

이렇게 하면:
1. 첫 실행: Streaming 완료
2. 두 번째 실행: Chunked → Original 순차 실행

---

## 자동 스킵 기능

이미 처리된 파일은 자동으로 건너뜁니다:

```bash
# 첫 실행: 100개 처리
python batch_process.py --all-modes --limit 100

# 두 번째 실행: 나머지 처리
python batch_process.py --all-modes
# → 각 모드에서 이미 처리된 100개는 스킵
```

**출력**:
```
======================================================================
MODE 1/3: STREAMING
======================================================================
Found 100 already processed files for streaming mode
Skipping 100 already processed files
Remaining files to process: 2520
Processing 2520 audio files from test-clean dataset
...
```

---

## 결과 JSON 구조

모든 모드는 하나의 `results.json`에 저장됩니다:

```json
{
  "streaming": {
    "timestamp": "2024-11-26T15:30:00",
    "overall": { "num_files": 2620, "wer": 0.052, ... },
    "folders": {...},
    "raw_results": [...]
  },
  "chunked": {
    "timestamp": "2024-11-26T16:00:00",
    "overall": { "num_files": 2620, "wer": 0.051, ... },
    "folders": {...},
    "raw_results": [...]
  },
  "original": {
    "timestamp": "2024-11-26T16:30:00",
    "overall": { "num_files": 2620, "wer": 0.050, ... },
    "folders": {...},
    "raw_results": [...]
  }
}
```

---

## 중단 및 재개

### 중단 (Ctrl+C)

언제든지 `Ctrl+C`로 중단 가능:

```
^C
Interrupted by user
Partial results saved to: results.json
```

중단하면:
- 완료된 모드의 결과는 저장됨
- 진행 중이던 모드는 저장 안 됨

### 재개

다시 실행하면 자동으로 재개:

```bash
# 중단된 지점부터 재개
python batch_process.py --all-modes
```

스마트 스킵 기능이 자동으로:
- 완료된 모드/파일 건너뜀
- 미완료된 것만 처리

---

## 고급 사용

### 특정 모드 건너뛰기

```bash
# Streaming은 이미 완료, Chunked부터 시작
python batch_process.py --mode chunked --continue-modes
```

### 커스텀 출력 파일

```bash
python batch_process.py --all-modes --output my_results.json
```

### 로그 레벨 조정

```bash
# 간단한 로그
python batch_process.py --all-modes --log-level WARNING

# 상세한 로그
python batch_process.py --all-modes --log-level DEBUG
```

---

## 비교: run_batch_all_modes.py vs --all-modes

### `run_batch_all_modes.py` (기존)
```bash
python run_batch_all_modes.py
```
- 별도 스크립트
- subprocess로 실행
- 각 모드가 독립적인 프로세스

### `--all-modes` (신규)
```bash
python batch_process.py --all-modes
```
- 단일 스크립트
- 동일 프로세스에서 순차 실행
- 더 간단하고 직관적

**추천**: `--all-modes` 사용

---

## 예상 소요 시간

전체 데이터셋 (2620 파일, 약 5.4시간 분량):

| 모드 | 예상 시간 | 특징 |
|------|----------|------|
| Streaming | 5.5-6시간 | 가장 빠름 |
| Chunked | 6-7시간 | 중간 |
| Original | 6-8시간 | 가장 느림 |
| **전체 (3개)** | **18-21시간** | 순차 실행 |

**참고**: 실제 시간은 GPU 성능에 따라 다름

---

## 문제 해결

### 서버 연결 실패
```
WebSocket error: Connection refused
```

**해결**:
1. 해당 포트의 서버가 실행 중인지 확인
2. 포트 번호 확인 (8001/8002/8003)

### 메모리 부족

**해결**:
```bash
# 제한된 개수씩 처리
python batch_process.py --all-modes --limit 500
```

### 특정 모드만 실패

다시 실행하면 자동으로 실패한 모드만 재처리:
```bash
python batch_process.py --all-modes
```

---

## 팁

1. **야간 실행**: 전체 데이터셋은 시간이 오래 걸리므로 야간에 실행
2. **로그 저장**:
   ```bash
   python batch_process.py --all-modes > log.txt 2>&1
   ```
3. **백그라운드 실행**:
   ```bash
   nohup python batch_process.py --all-modes &
   ```
4. **진행 상황 확인**: `results.json` 파일 크기나 timestamp로 확인
