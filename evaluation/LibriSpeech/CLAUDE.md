# evaluation/LibriSpeech/

영어 읽기 음성 ASR 벤치마크. LibriSpeech `test-other` 서브셋 사용. 주요 메트릭: WER와 FSL(First Committed Latency).

## 디렉토리 구조

| 경로 | 역할 |
|---|---|
| `LibriSpeech/test-other/` | 오디오 파일 (git 미포함, 별도 다운로드 필요) |
| `servers/` | 평가 서버 + 테스트 클라이언트 |
| `utils/` | 후처리 및 플롯 유틸리티 |
| `results/` | 벤치마크 출력 (자동 생성, 수동 수정 금지) |
| `dot_commit_probe/` | 커밋 정책(naive / 확정 게이트) 비교 검증 하네스. 서버 없이 모델 직접 구동 |

## 평가 서버 위치

`servers/streaming_websocket_server_fsl.py` — 모든 데이터셋(LibriSpeech, AMI, KsponSpeech, KtelSpeech)이 이 서버를 공유함.

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

`run_concurrent_benchmark.py`, `run_multi_speaker_full.py`: N개의 동시 클라이언트로 실행.  
**주의**: `PROJECT_ROOT = Path("/home/ubuntu/STiTy")` 하드코딩됨 — 다른 머신에서 실행 시 수정 필요.

## FSL 타이밍

FSL = 세그먼트 마지막 오디오 바이트부터 클라이언트가 `final` 메시지를 받기까지의 시간.  
결과의 `fsl_sec` 필드. 플롯 출력: `utils/export_fsl_ftl_plots.py`

## 데이터 다운로드

오디오는 git에 포함되지 않음. 예상 경로: `evaluation/LibriSpeech/LibriSpeech/test-other/`
