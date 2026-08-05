# evaluation/LibriSpeech/

영어 읽기 음성 ASR 벤치마크. LibriSpeech `test-other` 서브셋 사용. 주요 메트릭: WER와 FSL(First Committed Latency).

## 디렉토리 구조

| 경로 | 역할 |
|---|---|
| `LibriSpeech/test-other/` | 오디오 파일 (git 미포함, 별도 다운로드 필요) |
| `servers/` | 평가 서버(`streaming_websocket_server_fsl.py`) + 테스트 클라이언트(`test_qwen3_librispeech.py`) |
| `utils/` | 후처리·플롯 유틸 (`export_fsl_ftl_plots.py`, `compute_comet.py`, `export_results_xlsx.py` 등) |
| `paper_result/` | 논문용 mode2/3/4 실행 스크립트와 측정 결과 |
| `results/` | 벤치마크 출력 (자동 생성, 수동 수정 금지) |

## 평가 서버 위치

`servers/streaming_websocket_server_fsl.py` — 모든 데이터셋이 이 서버를 공유한다 (다른 데이터셋 디렉토리에는 클라이언트만 있음).

## 빠른 시작

```bash
# 터미널 1
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py --no-idle-shutdown

# 터미널 2
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## 다화자 벤치마크

- `run_concurrent_chapters.py` — 챕터를 N개 클라이언트로 나눠 동시 실행. CLI 인자(`--num-clients`, `--model`, `--scope`, `--tag`) 지원. **현재 주 경로.**
- `run_concurrent_benchmark.py`, `run_multi_speaker_full.py` — 동시 1~10명 스윕. 인자가 없고 파일 상단 상수를 편집해야 하며, `PROJECT_ROOT = Path("/home/ubuntu/STiTy")`가 하드코딩되어 있다.

## FSL 타이밍

FSL = 세그먼트 마지막 오디오 바이트부터 클라이언트가 `final` 메시지를 받기까지의 시간.  
결과의 `fsl_sec` 필드. 플롯 출력: `utils/export_fsl_ftl_plots.py`

## 데이터 다운로드

오디오는 git에 포함되지 않음. 예상 경로: `evaluation/LibriSpeech/LibriSpeech/test-other/`
