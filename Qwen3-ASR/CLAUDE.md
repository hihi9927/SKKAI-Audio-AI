# Qwen3-ASR (vendored upstream)

이 디렉토리는 upstream(QwenLM/Qwen3-ASR)에서 가져온 코드를 STiTy 리포가 **직접 추적**하는 형태다. git submodule이 아니다 — `.gitmodules`도 없고 gitlink도 아니라서, 이 안의 파일 변경은 다른 파일과 똑같이 부모 리포 커밋에 포함된다. 일반 사용법은 README.md를 참조하고, 이 파일은 STiTy 통합 관련 사항만 다룬다.

## STiTy 통합 진입점

- **프로덕션 서버**: `examples/streaming_websocket_server.py` — 모바일 앱이 연결하는 실제 WebSocket 서버
- **평가 서버**: `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py`가 이 파일을 래핑함
- **파인튜닝 결과물**: `finetuning/Qwen3-ASR-1.7B-en-merged/` — 평가 스크립트의 `--model` 인자가 여기를 가리킴

## upstream 동기화 규칙

- 여기 있는 파일은 STiTy 쪽 수정이 섞여 있다(예: `examples/streaming_websocket_server.py`). upstream 코드를 덮어쓰면 그 수정이 날아간다.
- upstream 업데이트는 자동 명령이 없다. 필요한 파일만 골라 가져오고, 로컬 수정과 수동으로 병합할 것.
- 커밋은 프로젝트 루트에서 평소처럼 하면 된다 (`git add Qwen3-ASR/...`).

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
