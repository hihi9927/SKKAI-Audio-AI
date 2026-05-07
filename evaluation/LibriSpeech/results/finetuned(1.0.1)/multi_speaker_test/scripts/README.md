# multi_speaker_test — 동시 접속 벤치마크

N명이 동시 접속했을 때 WER과 FSL(First Segment Latency)이 어떻게 변하는지 측정합니다.

**테스트 설계**: N명의 클라이언트가 동일한 548개 파일을 동시에 각자 처리하여 공정한 비교를 보장합니다.

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
│   │   └── ...
│   └── logs/
│       ├── server.log
│       ├── server_stdout.log
│       └── client_c0.log
├── ...
├── run_10/                ← 동시 10명 결과
├── run_20/                ← 동시 20명 결과 (확장 테스트)
├── run_30/                ← 동시 30명 결과 (확장 테스트)
└── ...                    ← run_40, run_50, run_60
```

> 모든 run은 신 포맷(`metric.json` + `meta.json` + `clients/` + `logs/`)을 사용합니다.  
> 구버전 원본 파일(`qwen3_test_other_fcl*.json`)은 각 `before/` 폴더에 보관됩니다.

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
    "num_files": 1096,
    "wer": 0.1079,
    "avg_first_token_latency": 4.680,
    "avg_fsl_sec": 2.482,
    "avg_asr_inference_sec": 0.291,
    "avg_translation_latency_sec": 0.089,
    "avg_output_tokens_per_commit": 8.20,
    "commit_stats": {
      "counts": {"vad": 752, "seg": 1323, "dot": 0, "finish": 0},
      "total": 2075,
      "ratios": {"vad": 0.362, "seg": 0.638, "dot": 0.0, "finish": 0.0}
    }
  },
  "per_client": [
    {"client": 0, "num_files": 548, "wer": 0.1050, "avg_first_token_latency": 4.680},
    {"client": 1, "num_files": 548, "wer": 0.1049, "avg_first_token_latency": 4.692}
  ],
  "raw_results": [...]
}
```

> `overall.num_files`는 전체 클라이언트의 처리 건수 합계 (N × 548).  
> `overall.wer`는 c0 기준 548개 파일의 WER (중복 제거).

### `meta.json`

```json
{
  "timestamp": "2026-05-07T12:00:00",
  "n_clients": 2,
  "mode": "full",
  "model_tag": "finetuned(1.0.1)",
  "server_model": "Qwen3-ASR-1.7B-en-merged",
  "num_files": 548,
  "migrated": false
}
```

### `clients/c{i}.json`

```json
{
  "raw_results": [...]
}
```

---

## 동작 방식

1. 서버 자동 시작 (VRAM 해제 확인 포함)
2. N개 클라이언트를 동시에 subprocess로 실행 (각자 동일한 548개 파일 처리)
3. 전체 완료 후 결과 병합 → `metric.json`, `meta.json`, `clients/c{i}.json` 저장
4. 서버 종료 → 다음 N으로 반복

---

## 주요 결과 요약 (finetuned v1.0.1, LibriSpeech test-other 548파일)

| 동시접속 | WER | FSL | ASR | FTL |
|---------|-----|-----|-----|-----|
| 1명 | 10.76% | 2.475s | 0.293s | 4.669s |
| 2명 | 10.79% | 2.482s | 0.291s | 4.680s |
| 3명 | 10.76% | 2.475s | 0.281s | 4.683s |
| 4명 | 10.77% | 2.467s | 0.280s | 4.653s |
| 5명 | 10.78% | 2.470s | 0.284s | 4.666s |
| 6명 | 10.77% | 2.455s | 0.287s | 4.669s |
| 7명 | 10.75% | 2.457s | 0.288s | 4.716s |
| 8명 | 10.71% | 2.451s | 0.294s | 4.726s |
| 9명 | 10.77% | 2.420s | 0.296s | 4.678s |
| 10명 | 10.70% | 2.455s | 0.296s | 4.742s |

> 10명까지 WER, FSL 모두 유의미한 변화 없음. GPU 메모리 대역폭 기준 이론적 한계 약 50~60명.
