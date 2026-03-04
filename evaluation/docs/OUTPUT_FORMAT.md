# 배치 처리 결과 JSON 출력 형식

## 개요

`batch_process.py`는 모든 모드를 하나의 JSON 파일에 통합하여 저장합니다. 각 모드별, 폴더(화자)별로 구조화된 결과를 제공합니다.

## 주요 특징

1. **단일 JSON 파일**: 모든 모드 (streaming, chunked, original)가 하나의 파일에 저장
2. **자동 병합**: 같은 파일로 다시 실행하면 해당 모드만 업데이트
3. **스마트 스킵**: 이미 처리된 파일은 자동으로 건너뜀
4. **타임스탬프**: 각 모드별 마지막 실행 시간 기록

## JSON 구조

```json
{
  "streaming": {
    "timestamp": "2024-11-26T15:30:00",
    "overall": {
      "num_files": 2620,
      "wer": 0.052,
      "first_token_latency": 1.234,
      "avg_processing_time": 0.876
    },
    "folders": {
      "1089": {
        "num_files": 50,
        "wer": 0.048,
        "first_token_latency": 1.200,
        "avg_processing_time": 0.850
      },
      "121": {
        "num_files": 45,
        "wer": 0.055,
        "first_token_latency": 1.250,
        "avg_processing_time": 0.900
      }
    },
    "raw_results": [...]
  },
  "chunked": {
    "timestamp": "2024-11-26T16:00:00",
    "overall": {...},
    "folders": {...},
    "raw_results": [...]
  },
  "original": {
    "timestamp": "2024-11-26T16:30:00",
    "overall": {...},
    "folders": {...},
    "raw_results": [
      {
        "file_id": "1089-134686-0000",
        "speaker_id": "1089",
        "audio_path": "test-clean/1089/134686/1089-134686-0000.flac",
        "reference": "HE HOPED THERE WOULD BE STEW FOR DINNER...",
        "hypothesis": "he hoped there would be stew for dinner...",
        "duration": 12.5,
        "total_time": 13.2,
        "first_token_latency": 1.5,
        "avg_processing_time": 0.7
      }
    ]
  }
}
```

## 스마트 처리 기능

### 자동 스킵 및 재개

배치 처리는 이미 완료된 파일을 자동으로 건너뜁니다:

```bash
# 첫 번째 실행: 100개 파일 처리
python batch_process.py --mode streaming --limit 100
# → results.json에 streaming 모드 100개 파일 저장

# 두 번째 실행: 나머지 파일 처리
python batch_process.py --mode streaming
# → 이미 처리된 100개는 스킵, 나머지만 처리
# → results.json의 streaming 모드 업데이트 (전체 결과 포함)

# 다른 모드 실행: chunked 모드 추가
python batch_process.py --mode chunked
# → results.json에 chunked 모드 추가 (streaming은 그대로 유지)
```

### 출력 예시

```
Found 100 already processed files for streaming mode
Skipping 100 already processed files
Remaining files to process: 2520
Processing 2520 audio files from test-clean dataset
...
Merged 100 existing + 2520 new = 2620 total results
```

## 필드 설명

### Top Level
- **`<mode>`**: 처리 모드 (streaming, chunked, original)
- **`timestamp`**: 해당 모드가 마지막으로 실행된 시간

### Overall Section
전체 데이터셋에 대한 통계

- **`num_files`**: 처리된 전체 파일 개수
- **`wer`**: 전체 데이터셋의 Word Error Rate (0~1 사이 값, 0.052 = 5.2%)
- **`first_token_latency`**: 첫 번째 응답까지의 평균 시간 (초)
- **`avg_processing_time`**: 평균 순수 처리 시간 (초, 대기 시간 제외)

### Folders Section
화자(폴더)별 통계

각 키는 화자 ID (예: "1089", "121", "237")이며, 값은:
- **`num_files`**: 해당 화자의 파일 개수
- **`wer`**: 해당 화자의 WER
- **`first_token_latency`**: 해당 화자의 평균 첫 응답 시간
- **`avg_processing_time`**: 해당 화자의 평균 처리 시간

### Raw Results Section
개별 파일별 상세 결과

- **`file_id`**: 파일 ID (예: "1089-134686-0000")
- **`speaker_id`**: 화자 ID (예: "1089")
- **`audio_path`**: 오디오 파일 경로
- **`reference`**: 참조 텍스트 (정답)
- **`hypothesis`**: 인식 결과 텍스트
- **`duration`**: 오디오 파일 길이 (초)
- **`total_time`**: 전체 처리 시간 (초, 대기 포함)
- **`first_token_latency`**: 첫 응답까지의 시간 (초)
- **`avg_processing_time`**: 순수 처리 시간 (초, total_time - duration)

## 메트릭 설명

### 1. WER (Word Error Rate)
```
WER = (S + D + I) / N
```
- S: 대체 (Substitutions)
- D: 삭제 (Deletions)
- I: 삽입 (Insertions)
- N: 참조 텍스트의 총 단어 수

**값의 의미**:
- 0.0 (0%): 완벽
- 0.05 (5%): 매우 좋음
- 0.10 (10%): 좋음
- 0.20 (20%): 보통
- 0.50 (50%): 나쁨

### 2. First Token Latency
오디오 전송 시작부터 첫 번째 결과를 받을 때까지의 시간

**의미**:
- **Streaming**: 사용자가 느끼는 반응성 (낮을수록 좋음)
- **Chunked**: 첫 청크 처리 시간 (보통 2초 + α)
- **Original**: 전체 오디오 대기 + 처리 시간 (높음)

**예상 범위**:
- Streaming: 1~3초
- Chunked: 2~5초
- Original: 오디오 길이 + 처리 시간

### 3. Avg Processing Time
순수 ASR 처리에 걸린 시간 (강제 대기 시간 제외)

**계산 방식**:
```python
avg_processing_time = total_time - audio_duration
```

**의미**:
- 실제 모델 추론 시간
- GPU 성능 및 모델 효율성 반영
- 낮을수록 빠른 처리

## 사용 예제

### Python으로 JSON 읽기

```python
import json

# JSON 파일 로드 (모든 모드 포함)
with open('results.json', 'r') as f:
    data = json.load(f)

# Streaming 모드 통계
if 'streaming' in data:
    streaming = data['streaming']
    print(f"Streaming Mode (updated: {streaming['timestamp']})")
    print(f"  Total files: {streaming['overall']['num_files']}")
    print(f"  WER: {streaming['overall']['wer']*100:.2f}%")
    print(f"  First Token Latency: {streaming['overall']['first_token_latency']:.3f}s")
    print(f"  Processing Time: {streaming['overall']['avg_processing_time']:.3f}s")
    print()

# 특정 화자 통계 (streaming 모드)
if 'streaming' in data:
    speaker_id = '1089'
    if speaker_id in data['streaming']['folders']:
        folder_stats = data['streaming']['folders'][speaker_id]
        print(f"화자 {speaker_id} (Streaming):")
        print(f"  파일 수: {folder_stats['num_files']}")
        print(f"  WER: {folder_stats['wer']*100:.2f}%")
        print()
```

### 모드 간 비교

하나의 JSON 파일에 모든 모드가 있으므로 간단히 비교 가능:

```python
import json

# 단일 파일 로드
with open('results.json', 'r') as f:
    data = json.load(f)

# 비교표 출력
print(f"{'Mode':<15} {'Files':<10} {'WER':<10} {'First Token':<15} {'Proc Time':<15}")
print("-" * 65)

for mode in ['streaming', 'chunked', 'original']:
    if mode in data:
        stats = data[mode]['overall']
        print(f"{mode:<15} {stats['num_files']:<10} {stats['wer']*100:<10.2f} "
              f"{stats['first_token_latency']:<15.3f} "
              f"{stats['avg_processing_time']:<15.3f}")
    else:
        print(f"{mode:<15} {'N/A':<10} {'N/A':<10} {'N/A':<15} {'N/A':<15}")
```

출력 예시:
```
Mode            Files      WER        First Token     Proc Time
-----------------------------------------------------------------
streaming       2620       5.20       1.234           0.876
chunked         2620       5.15       2.456           0.845
original        2620       5.05       12.345          0.823
```

## 화자별 WER 분석

```python
import json

with open('results.json', 'r') as f:
    data = json.load(f)

# Streaming 모드의 화자별 WER
if 'streaming' in data:
    folders = data['streaming']['folders']

    # WER이 높은 화자 찾기
    wer_by_speaker = {
        speaker_id: stats['wer']
        for speaker_id, stats in folders.items()
    }

    # 상위 10명
    top_10_worst = sorted(wer_by_speaker.items(), key=lambda x: x[1], reverse=True)[:10]

    print("WER이 가장 높은 화자 10명 (Streaming):")
    for speaker_id, wer in top_10_worst:
        print(f"  {speaker_id}: {wer*100:.2f}%")
```

## 중간 결과 파일

배치 처리 중 10개 파일마다 생성되는 임시 파일:
- 파일명: `<output_file>.tmp`
- 형식: Raw results format (구조화되지 않음)

```json
{
  "timestamp": "2024-11-26T15:30:00",
  "mode": "streaming",
  "total_files": 10,
  "results": [...]
}
```

## 참고사항

1. **WER 계산**: `jiwer` 라이브러리 사용
2. **폴더 ID**: 파일 ID의 첫 번째 부분 (예: "1089-134686-0000" → "1089")
3. **시간 단위**: 모든 시간은 초(seconds) 단위
4. **NULL 값**: 측정 실패 시 `null`로 표시될 수 있음
