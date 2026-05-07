# multi_speaker_test — 동시 접속 벤치마크

N명이 동시 접속했을 때 WER과 FSL(First Segment Latency)이 어떻게 변하는지 측정합니다.

---

## 폴더 구조

```
evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/
├── scripts/
│   ├── run_benchmark.py   ← 벤치마크 러너
│   └── README.md          ← 이 파일
├── run_01/                ← 동시 1명 결과
├── run_02/                ← 동시 2명 결과
│   ├── metric.json            # WER + latency 종합 지표
│   ├── meta.json              # 실행 정보 (접속자 수, 모드, 모델, 타임스탬프)
│   ├── clients/               # 클라이언트별 세부 결과
│   │   ├── c0.json
│   │   ├── c0_files.json      # 클라이언트에게 할당된 파일 목록
│   │   └── ...
│   └── logs/
│       ├── server.log
│       ├── server_stdout.log
│       └── client_c0.log
└── ...
```

> 새 실행 결과는 `run_{N:02d}/` 형식으로 이 폴더에 직접 저장됩니다.  
> 기존 `run_01`~`run_10`은 구 포맷(`qwen3_test_other_fcl*.json`)이고,  
> 이 스크립트로 생성된 결과는 신 포맷(`metric.json` + `meta.json` + `clients/` + `logs/`)입니다.

---

## 실행

**실행 위치: `/home/ubuntu/STiTy`**

### 기본 (full 모드, 1~10명, finetuned 모델)
```bash
python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py"
```

### 특정 범위만
```bash
python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py" \
  --start-n 3 --end-n 5
```

### split 모드
```bash
python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py" \
  --mode split
```

### baseline 모델
```bash
python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py" \
  --model "baseline(1.0.0)" \
  --server-model Qwen/Qwen3-ASR-1.7B
```

---

## 모드 설명

| 모드 | 파일 분배 | 목적 |
|------|----------|------|
| `full` (기본) | 각 클라이언트가 **전체** 파일 처리 | 동시 접속에 따른 latency 저하 측정 |
| `split` | 파일을 N명에게 **균등 분배** | throughput / 부하 분산 테스트 |

---

## 인자 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `full` | `full` 또는 `split` |
| `--start-n` | `1` | 시작 동시 접속자 수 |
| `--end-n` | `10` | 종료 동시 접속자 수 |
| `--model` | `finetuned(1.0.1)` | 결과 대분류 태그 |
| `--server-model` | finetuned 경로 | ASR 서버에 로드할 모델 경로 |
| `--num-files` | `548` | 사용할 파일 수 |

---

## 결과 파일 설명

### `metric.json`

```json
{
  "overall": {
    "num_files": 548,
    "wer": 0.073,
    "avg_first_token_latency": 4.85,
    "avg_fsl_sec": 1.23,
    "avg_asr_inference_sec": 0.71,
    "avg_translation_latency_sec": 0.21,
    "avg_output_tokens_per_commit": 9.0,
    "commit_stats": {
      "counts": {"vad": 1142, "seg": 4469, "dot": 0, "finish": 0},
      "total": 5611,
      "ratios": {"vad": 0.20, "seg": 0.80, "dot": 0.0, "finish": 0.0}
    }
  },
  "per_client": [
    {"client": 0, "num_files": 548, "wer": 0.071, "avg_first_token_latency": 4.8},
    {"client": 1, "num_files": 548, "wer": 0.075, "avg_first_token_latency": 5.1}
  ],
  "raw_results": [...]
}
```

### `meta.json`

```json
{
  "timestamp": "2026-05-07T12:00:00",
  "n_clients": 2,
  "mode": "full",
  "model_tag": "finetuned(1.0.1)",
  "server_model": "/home/ubuntu/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged",
  "num_files": 548,
  "test_dir": "..."
}
```

### `clients/c{i}.json`

```json
{
  "overall": {"client": 0, "num_files": 548, "wer": 0.071, "avg_first_token_latency": 4.8},
  "raw_results": [...]
}
```

---

## 동작 방식

1. 서버 자동 시작 (VRAM 해제 확인 포함)
2. N개 클라이언트를 동시에 subprocess로 실행
3. 전체 완료 후 결과 병합 → `metric.json`, `meta.json`, `clients/c{i}.json` 저장
4. 서버 종료 → 다음 N으로 반복
