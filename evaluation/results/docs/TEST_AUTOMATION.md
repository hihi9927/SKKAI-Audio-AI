# WhisperLiveKit 자동 테스트 가이드

LibriSpeech 데이터셋으로 SimulStreaming과 LocalAgreement 백엔드를 자동으로 테스트하는 스크립트입니다.

## 기본 사용법

### 자동 모드 (서버 자동 관리) - 권장

```bash
python test_whisperlivekit.py \
    --test-dir "C:\path\to\LibriSpeech\test-other" \
    --all-policies \
    --auto-server \
    --server-model large-v3 \
    --server-lan en \
    --limit 10
```

**동작:**
- Policy 1 (SimulStreaming) 서버 자동 시작 → 테스트 → 자동 종료
- Policy 2 (LocalAgreement) 서버 자동 시작 → 테스트 → 자동 종료
- 결과를 `whisperlivekit_results.json`에 저장

### 수동 모드 (기존 방식)

```bash
# 터미널 1: 서버 수동 실행
python whisperlivekit/websocket_server.py --backend-policy 1 --model large-v3 --lan en

# 터미널 2: 테스트 실행
python test_whisperlivekit.py \
    --test-dir "C:\path\to\LibriSpeech\test-other" \
    --policy 1 \
    --limit 10
```

## 주요 옵션

### 필수 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--test-dir` | LibriSpeech 테스트 디렉토리 | `--test-dir "C:\LibriSpeech\test-other"` |
| `--policy` | 테스트할 백엔드 (1 또는 2) | `--policy 1` |
| `--all-policies` | 모든 백엔드 순차 테스트 | `--all-policies` |

### 자동 서버 관리 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--auto-server` | 서버 자동 시작/종료 | (비활성화) |
| `--server-script` | 서버 스크립트 경로 | `whisperlivekit/websocket_server.py` |
| `--server-model` | 서버 모델 | `large-v3` |
| `--server-lan` | 서버 언어 | `en` |
| `--server-args` | 추가 서버 인자 | (없음) |

### 기타 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--host` | WebSocket 호스트 | `localhost` |
| `--port` | WebSocket 포트 | `8001` |
| `--output` | 결과 JSON 파일 | `whisperlivekit_results.json` |
| `--limit` | 처리할 파일 수 제한 | (제한 없음) |
| `--calculate-wer` | WER 계산 (jiwer 필요) | (비활성화) |
| `--log-level` | 로그 레벨 | `INFO` |

## 사용 예시

### 예시 1: 모든 백엔드 자동 테스트 (10개 파일)

```bash
python test_whisperlivekit.py \
    --test-dir "C:\LibriSpeech\test-other" \
    --all-policies \
    --auto-server \
    --limit 10 \
    --calculate-wer
```

### 예시 2: SimulStreaming만 테스트 (한국어)

```bash
python test_whisperlivekit.py \
    --test-dir "C:\Korean\test-data" \
    --policy 1 \
    --auto-server \
    --server-model large-v3 \
    --server-lan ko \
    --limit 20
```

### 예시 3: 추가 서버 옵션과 함께

```bash
python test_whisperlivekit.py \
    --test-dir "C:\LibriSpeech\test-other" \
    --all-policies \
    --auto-server \
    --server-args "--beams 10 --min-chunk-size 2.0" \
    --limit 5
```

### 예시 4: 수동 모드 (기존 실행 중인 서버 사용)

```bash
# 서버는 별도로 실행해두고
python test_whisperlivekit.py \
    --test-dir "C:\LibriSpeech\test-other" \
    --policy 1
```

## 결과 파일 구조

```json
{
  "policy_1": {
    "timestamp": "2024-01-15T10:30:00",
    "overall": {
      "num_files": 100,
      "wer": 0.05,
      "first_token_latency": 0.234,
      "avg_processing_time": 1.56
    },
    "folders": { ... },
    "raw_results": [ ... ]
  },
  "policy_2": {
    ...
  }
}
```

## 중단 및 재시작

### 중단
- `Ctrl+C`: 안전하게 중단 (서버 자동 종료, 중간 결과 저장)

### 재시작
- 중단된 지점부터 자동으로 재개
- 이미 처리된 파일은 건너뜀

```bash
# 중단 후 다시 실행하면 자동으로 이어서 진행
python test_whisperlivekit.py \
    --test-dir "C:\LibriSpeech\test-other" \
    --all-policies \
    --auto-server
```

## Policy 설명

| Policy | 이름 | 설명 |
|--------|------|------|
| 1 | SimulStreaming | 실시간 스트리밍 ASR (AlignAtt 정책) |
| 2 | LocalAgreement | 로컬 합의 기반 ASR (버퍼 기반) |

## 문제 해결

### 서버가 시작되지 않을 때
```bash
# 로그 레벨을 DEBUG로 설정
python test_whisperlivekit.py ... --log-level DEBUG
```

### 특정 파일만 테스트하고 싶을 때
```bash
# limit으로 개수 제한
python test_whisperlivekit.py ... --limit 5
```

### WER 계산이 안 될 때
```bash
# jiwer 설치
pip install jiwer
```

## 참고

- 자동 모드는 각 policy 테스트 후 서버를 완전히 종료하고 재시작합니다
- 서버 시작 대기 시간: 최대 60초
- 서버 종료 대기 시간: 10초 (이후 강제 종료)
- Policy 간 전환 대기: 2초
