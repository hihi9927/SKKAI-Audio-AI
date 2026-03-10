# WhisperLiveKit Common Files Testing Guide

이 가이드는 공통 파일들에 대해서만 테스트를 수행하는 방법을 설명합니다.

## 배경

각 모델(chunked, original, simul 등)로 테스트할 때 일부 파일에서 결과가 나오지 않아 파일 수가 일치하지 않을 수 있습니다. 공정한 비교를 위해 모든 모델에서 공통으로 존재하는 파일들만 테스트해야 합니다.

## 사전 준비

### 1. 공통 파일 추출

먼저 test-results 폴더에서 공통 파일을 추출합니다:

```bash
cd c:\Users\VRSTUDIO3\Desktop\STiTy-main1\test-results
python extract_common_files.py
```

이 스크립트는 다음 파일들을 생성합니다:
- `chunked-common-files.json` (1,619개 공통 파일)
- `original-common-files.json` (1,619개 공통 파일)
- `simul_meaning_policy1-common-files.json` (1,619개 공통 파일)
- `simul_no_meaning_policy1-common-files.json` (1,619개 공통 파일)
- `common-files-summary.json` (요약 통계)

### 2. 공통 파일에서 랜덤 샘플링

전체 1,619개 파일을 모두 테스트하는 대신, 랜덤 시드를 고정하여 200개만 샘플링할 수 있습니다:

## 테스트 실행 방법

### 방법 1: 간편 스크립트 사용

```bash
cd c:\Users\VRSTUDIO3\Desktop\STiTy-main1\WhisperLiveKit
python test_common_200_samples.py
```

### 방법 2: 직접 커맨드 실행

```bash
cd c:\Users\VRSTUDIO3\Desktop\STiTy-main1\WhisperLiveKit

# Policy 1 테스트 (SimulStreaming with meaning segmentation)
python test_whisperlivekit.py \
  --test-dir "c:\Users\VRSTUDIO3\Desktop\STiTy-main1\LibriSpeech\test-other" \
  --policy 1 \
  --host localhost \
  --port 8001 \
  --output whisperlivekit_common_200_results.json \
  --common-files "../test-results/chunked-common-files.json" \
  --random-sample 200 \
  --random-seed 42 \
  --calculate-wer \
  --log-level INFO
```

## 새로운 옵션 설명

### `--common-files <path>`
공통 파일 JSON 경로를 지정합니다. 이 파일에 포함된 file_id만 테스트합니다.

**예시:**
```bash
--common-files "../test-results/chunked-common-files.json"
```

### `--random-sample <N>`
공통 파일 중에서 N개만 랜덤으로 선택합니다. 재현성을 위해 `--random-seed`와 함께 사용하세요.

**예시:**
```bash
--random-sample 200  # 200개 파일만 테스트
```

### `--random-seed <seed>`
랜덤 샘플링에 사용할 시드 값입니다. 기본값은 42입니다.

**예시:**
```bash
--random-seed 42  # 고정된 시드로 매번 동일한 200개 선택
```

## 이어서 테스트하기

테스트 중 중단되었거나 일부만 완료한 경우, 자동으로 이미 완료된 파일은 건너뜁니다:

```bash
# 중단된 테스트를 이어서 진행
python test_whisperlivekit.py \
  --policy 1 \
  --output whisperlivekit_common_200_results.json \
  --common-files "../test-results/chunked-common-files.json" \
  --random-sample 200 \
  --random-seed 42 \
  --calculate-wer
```

스크립트는 자동으로:
1. `whisperlivekit_common_200_results.json`에서 이미 완료된 파일 확인
2. 동일한 시드(42)로 동일한 200개 파일 선택
3. 이미 완료된 파일은 건너뛰고 나머지만 처리
4. 기존 결과와 새 결과를 자동으로 병합

## 작동 원리

1. **파일 스캔**: LibriSpeech test-other 폴더에서 모든 오디오 파일 검색
2. **공통 파일 필터링**: `--common-files`로 지정된 JSON의 file_id만 선택
3. **랜덤 샘플링**:
   - 고정된 시드(기본 42)로 난수 생성기 초기화
   - file_id 기준으로 정렬하여 일관성 보장
   - random.sample()로 N개 선택
   - 처리 순서를 위해 다시 정렬
4. **이미 처리된 파일 스킵**: 출력 파일에서 완료된 file_id 확인하고 제외
5. **테스트 실행**: 남은 파일들만 처리
6. **결과 병합**: 기존 결과 + 새 결과 자동 병합

## 예제 출력

```
INFO    Scanning test directory...
INFO    Found 2620 audio files in test directory

INFO    Loaded 1619 common file IDs from ../test-results/chunked-common-files.json
INFO    Filtered to 1619 common files
INFO    Random sampling 200 files from 1619 (seed=42)
INFO    Selected 200 files for testing
INFO    First file: 1272-128104-0000
INFO    Last file: 8463-294828-0029

INFO    Found 50 already processed files for policy 1
INFO    Skipping 50 already processed files
INFO    Remaining files to process: 150
INFO    Processing 150 audio files
```

## 주의사항

1. **동일한 시드 사용**: 같은 결과를 얻으려면 항상 같은 `--random-seed` 값을 사용하세요.
2. **공통 파일 JSON 일치**: 모든 모델에 대해 동일한 common-files JSON을 사용하세요.
3. **중간 저장**: 10개 파일마다 `.tmp` 파일에 중간 결과가 저장됩니다.

## 결과 비교

모든 모델에 대해 동일한 200개 파일로 테스트 후, WER을 공정하게 비교할 수 있습니다:

```python
# 결과 비교 예시
import json

with open('whisperlivekit_common_200_results.json') as f:
    data = json.load(f)

for policy_key, policy_data in data.items():
    print(f"{policy_key}:")
    print(f"  Files: {policy_data['overall']['num_files']}")
    print(f"  WER: {policy_data['overall']['wer']:.4f}")
    print(f"  FTL: {policy_data['overall']['first_token_latency']:.2f}s")
```
