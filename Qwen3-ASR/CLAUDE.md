# Qwen3-ASR (vendored upstream)

upstream(QwenLM/Qwen3-ASR) 코드를 STiTy 리포가 **직접 추적**한다. git submodule이 아니라서 이 안의 파일 변경은 다른 파일과 똑같이 부모 리포 커밋에 포함된다 (`git add Qwen3-ASR/...`). 일반 사용법은 README.md를 보고, 이 파일은 STiTy 통합 사항만 다룬다.

## STiTy 통합 진입점

- **프로덕션 서버**: `examples/streaming_websocket_server.py` — 모바일 앱이 연결하는 WebSocket 서버
- **평가 서버**: `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py`가 이 파일을 래핑
- **파인튜닝**: `finetuning/qwen3_asr_sft.py` (SFT) + `finetuning/utils/` (병합). 병합된 가중치는 여기 두지 않고 리포 루트 `models/`에 있다 — `models/Qwen3-ASR-1.7B-{en,ko}-silence-*-merged`.
  평가 스크립트의 `finetuned` 별칭은 아직 `finetuning/Qwen3-ASR-1.7B-en-merged`(현재 없음)로 매핑되어 있으므로 서버에는 `--model models/...` 실제 경로를 직접 넘길 것.

## upstream 동기화 규칙

- 여기 있는 파일에는 STiTy 쪽 수정이 섞여 있다(예: `examples/streaming_websocket_server.py`). upstream을 통째로 덮어쓰면 날아간다.
- 자동 동기화 명령은 없다. 필요한 파일만 골라 가져와 수동 병합할 것.

## 설치

```bash
pip install -e ./Qwen3-ASR              # transformers 백엔드
pip install -e "./Qwen3-ASR[vllm]"     # vLLM 포함
```

## 패키지 구조

| 디렉토리 | 역할 |
|---|---|
| `qwen_asr/core/` | 모델 백엔드 (transformers/vllm) |
| `qwen_asr/inference/` | ASR 추론 엔진 |
| `qwen_asr/cli/` | CLI 진입점 (qwen-asr-serve 등) |
| `examples/` | 프로덕션 서버 및 사용 예제 |
| `finetuning/` | SFT 스크립트 및 병합 유틸리티 |
| `app/` | Electron/Gradio 데모 앱 (프로덕션 배포와 무관) |

## 주의사항

`examples/streaming_websocket_server.py` 수정 시 모든 평가 벤치마크가 영향받는다. 변경 후 반드시 LibriSpeech sample 벤치마크 한 번 이상 실행할 것.
