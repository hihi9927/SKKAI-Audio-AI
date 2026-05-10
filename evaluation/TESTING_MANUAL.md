# STiTy ASR 평가 테스트 매뉴얼

## 1. 개요

테스트 결과는 `--model`, `--scope`, `--tag` 세 인자에 따라 아래 구조로 자동 저장됩니다.

```
evaluation/{Dataset}/results/
└── {model}/          ← --model  (대분류: 모델 종류)
    └── {scope}/      ← --scope  (소분류: 테스트 범위)
        └── {tag}/    ← --tag    (폴더명, 미지정 시 run_01, run_02 ... 자동 생성)
            ├── metric.json       # WER/CER 등 핵심 성능 지표
            ├── meta.json         # 실행에 사용된 모든 CLI 인자
            ├── description.txt   # --description으로 전달한 테스트 설명
            ├── plots/            # 샘플 파일별 타이밍 플롯 (PNG + LOG)
            └── logs/             # 서버/테스트 실행 로그
```

각 실행은 독립된 폴더에 저장되므로 덮어쓸 위험이 없습니다.  
`--tag`를 지정하면 동일 폴더에 이어서 저장(resume)됩니다.

---

## 2. 공통 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | `finetuned(1.0.1)` | 대분류: 모델 종류 (예: `baseline(1.0.0)`, `finetuned(1.0.1)`) |
| `--scope` | `sample` | 소분류: `full`(전체 데이터셋) 또는 `sample`(일부) |
| `--tag` | 자동 생성 | 결과 폴더명. 지정 시 해당 폴더에 이어서 저장 |
| `--description` | — | 테스트 설명. `description.txt`에 저장 |
| `--host` | `localhost` | 서버 호스트 |
| `--port` | `8765` | 서버 포트 |
| `--limit` | — | 처리할 최대 파일 수 (미지정 시 전체) |
| `--chunk-size-ms` | `200` | 오디오 청크 크기 (ms) |
| `--send-interval-ms` | `200` | 청크 전송 간격 (ms, 실시간 페이싱) |
| `--trailing-silence-ms` | 스크립트별 상이 | 오디오 끝에 추가할 묵음 길이 (VAD 트리거용) |
| `--target-lang` | `ko` | 번역 대상 언어 |
| `--fresh-start` | — | 기존 결과 무시하고 처음부터 재실행 |
| `--auto-server` | — | ASR 서버 자동 시작/종료 |
| `--server-model` | `Qwen/Qwen3-ASR-1.7B` | 자동 시작 시 사용할 모델 |

> `baseline` 계열 모델(`Qwen/Qwen3-ASR-1.7B`, `baseline`, `baseline(1.0.0)` 등)은 서버에서 `dot commit`이 기본 활성화됩니다.  
> baseline에서도 끄고 싶으면 `--disable-dot-commit`, finetuned 등에서 강제로 켜고 싶으면 `--enable-dot-commit`을 사용합니다.

---

## 3. 데이터셋별 실행 방법

### 3.1 서버 실행 (공통)

서버 로그를 결과 폴더(`logs/server.log`)에 저장하려면 `--log-file`을 지정해야 합니다.  
run 폴더 이름은 테스트 스크립트 실행 시 결정되므로, **`--tag`로 폴더명을 미리 고정**한 뒤 서버와 테스트 스크립트에 같은 경로를 맞춰줍니다.  

baseline 서버는 `dot commit`이 기본으로 켜지므로 별도 인자를 붙이지 않아도 됩니다.  
반대로 baseline에서 `dot commit`을 끄고 싶다면 서버 실행 시 `--disable-dot-commit`을 추가합니다.

```bash
cd ~/STiTy && git pull

# Baseline 모델
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --no-idle-shutdown \
  --log-file "evaluation/LibriSpeech/results/baseline(1.0.0)/sample/run_01/logs/server.log"

# Finetuned 모델
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --model /home/ubuntu/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged \
  --no-idle-shutdown \
  --enforce-eager \
  --log-file "evaluation/LibriSpeech/results/finetuned(1.0.1)/sample/run_01/logs/server.log"
```

> **`--log-file` 경로 규칙:** `evaluation/{Dataset}/results/{model}/{scope}/{tag}/logs/server.log`  
> 서버를 먼저 시작하고, 테스트 스크립트에서 동일한 `--model`, `--scope`, `--tag` 조합을 사용하면 같은 폴더에 결과가 모입니다.  
> 로그 저장이 필요 없으면 `--log-file`을 생략해도 됩니다.

---

### 3.2 LibriSpeech

```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model <baseline(1.0.0)|finetuned(1.0.1)> \
  --scope <full|sample> \
  [--tag <폴더명>] \
  [--description "테스트 설명"] \
  --target-lang ko \
  --trailing-silence-ms 5500 \
  --chunk-size-ms 200
```

`--auto-server`를 사용할 때도 같은 규칙이 적용됩니다. baseline 계열 모델은 `dot commit`이 자동 활성화되고, 필요하면 `--server-args "--disable-dot-commit"` 또는 `--server-args "--enable-dot-commit"`으로 override할 수 있습니다.

결과 위치: `evaluation/LibriSpeech/results/{model}/{scope}/{tag}/`

**예시 1 — finetuned, 전체 데이터셋**
```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "finetuned(1.0.1)" \
  --scope full \
  --description "finetuned 모델 전체 test-other 기준선 측정"
# → results/finetuned(1.0.1)/full/run_01/
```

**예시 2 — baseline, 일부 샘플**
```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "baseline(1.0.0)" \
  --scope sample \
  --limit 50
# → results/baseline(1.0.0)/sample/run_01/
```

**예시 3 — chunk size 변경 실험 (tag 지정)**
```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "finetuned(1.0.1)" \
  --scope full \
  --tag chunk_1500ms \
  --chunk-size-ms 1500
# → results/finetuned(1.0.1)/full/chunk_1500ms/
```

---

### 3.3 KtelSpeech (2화자 전화 대화, Korean)

서버를 먼저 시작할 때 `--tag`를 미리 고정하고 `--log-file`을 맞춰줍니다.

```bash
# 1) 서버 시작 (별도 터미널)
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --model /home/ubuntu/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged \
  --no-idle-shutdown \
  --enforce-eager \
  --log-file "evaluation/KtelSpeech/results/finetuned(1.0.1)/sample/run_01/logs/server.log"

# 2) 테스트 실행
python evaluation/KtelSpeech/test_qwen3_ktelspeech.py \
  --data-dir evaluation/KtelSpeech \
  --model "finetuned(1.0.1)" \
  --scope sample \
  --tag run_01 \
  --description "테스트 설명" \
  --trailing-silence-ms 5000
```

> **참고:** KtelSpeech는 `--data-dir` 아래 `KtelSpeech/` (merged WAV)와 `label/` (발화별 텍스트)가 있어야 합니다. 평가 지표는 WER 대신 **CER** (Character Error Rate) 입니다.

결과 위치: `evaluation/KtelSpeech/results/{model}/{scope}/{tag}/`

---

### 3.4 AMI

서버를 먼저 시작할 때 `--tag`를 미리 고정하고 `--log-file`을 맞춰줍니다.

```bash
# 1) 서버 시작 (별도 터미널)
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --no-idle-shutdown \
  --log-file "evaluation/AMI/results/baseline(1.0.0)/sample/run_01/logs/server.log"

# 2) 테스트 실행
python evaluation/AMI/test_qwen3_ami.py \
  --ami-dir evaluation/AMI/AMI \
  --words-dir evaluation/AMI/words \
  --model "baseline(1.0.0)" \
  --scope sample \
  --tag run_01 \
  --description "테스트 설명" \
  --trailing-silence-ms 5000
```

> **참고:** 현재 저장소의 AMI 데이터는 오디오가 `evaluation/AMI/AMI/`, 단어 XML이 `evaluation/AMI/words/` 아래에 있습니다.  
> 그래서 `--ami-dir evaluation/AMI`만 주면 `No audio files found.`가 날 수 있고, 위 예시처럼 `--ami-dir`와 `--words-dir`를 각각 지정하는 것이 안전합니다.

결과 위치: `evaluation/AMI/results/{model}/{scope}/{tag}/`

---

## 4. 결과 파일 설명

| 파일 | 설명 |
|---|---|
| `metric.json` | 핵심 성능 지표 (WER/CER, FSL, latency 등) |
| `meta.json` | 실행 당시 모든 CLI 인자 + 타임스탬프. 재현에 사용 |
| `description.txt` | `--description`으로 전달한 이 테스트의 목적 설명 |
| `plots/` | 샘플 파일별 타이밍 분석 플롯 (PNG) 및 원본 로그 (LOG) |
| `logs/` | 서버 stdout/stderr 로그 |

---

## 5. 동시 접속 벤치마크 (multi_speaker_test)

N명이 동시 접속할 때 WER / FSL 변화를 측정하는 전용 벤치마크입니다.  
관련 코드와 결과는 `evaluation/LibriSpeech/multi_speaker_test/` 안에 자체 격리되어 있습니다.  
자세한 내용은 [multi_speaker_test/README.md](LibriSpeech/multi_speaker_test/README.md)를 참조하세요.

```bash
# 기본 (full 모드, 1~10명, finetuned 모델)
python evaluation/LibriSpeech/multi_speaker_test/run_benchmark.py

# 특정 범위 / 모드 지정
python evaluation/LibriSpeech/multi_speaker_test/run_benchmark.py --mode split --start-n 3 --end-n 5
```

결과 위치: `evaluation/LibriSpeech/multi_speaker_test/results/{model}/run_{N:02d}/`

---

## 6. Resume (이어서 실행)

중단된 테스트를 이어서 실행하려면 `--tag`로 동일 폴더를 지정합니다.  
이미 처리된 파일은 `metric.json`을 기준으로 자동으로 건너뜁니다.

```bash
# 처음 실행 (run_01 자동 생성)
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --model "finetuned(1.0.1)" --scope full

# 중단 후 이어서 실행
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --model "finetuned(1.0.1)" --scope full --tag run_01

# 처음부터 다시 실행
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --model "finetuned(1.0.1)" --scope full --tag run_01 --fresh-start
```
