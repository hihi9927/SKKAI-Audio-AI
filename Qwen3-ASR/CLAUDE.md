# Qwen3-ASR (vendored upstream)

upstream(QwenLM/Qwen3-ASR) 코드를 STiTy 리포가 **직접 추적**한다. git submodule이 아니라서 이 안의 파일 변경은 다른 파일과 똑같이 부모 리포 커밋에 포함된다 (`git add Qwen3-ASR/...`). 일반 사용법은 README.md를 보고, 이 파일은 STiTy 통합 사항만 다룬다.

## STiTy 통합 진입점

- **프로덕션 서버**: `examples/streaming_websocket_server.py` — 모바일 앱이 연결하는 WebSocket 서버
- **평가 서버**: `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py`가 이 파일을 래핑하고,
  실제로 띄우는 것은 그것을 다시 상속한 `evaluation/streaming_websocket_server_ast.py` 다
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

## 주의사항

`examples/streaming_websocket_server.py` 수정 시 모든 평가 벤치마크가 영향받는다. 변경 후 반드시 LibriSpeech sample 벤치마크 한 번 이상 실행할 것.

그 전에 `tests/` 의 단위 테스트 둘을 먼저 돌린다. 모델도 GPU도 필요 없고 1.2초면
끝나는데, 벤치마크가 몇 분을 태우고 나서야 알려줄 회귀를 그 자리에서 잡는다.

```bash
PYTHONPATH=$PWD/Qwen3-ASR python Qwen3-ASR/tests/test_dot_commit_boundary.py
PYTHONPATH=$PWD/Qwen3-ASR python Qwen3-ASR/tests/test_final_residual_commit.py
```

둘 다 고쳐 놓은 버그를 붙잡아 둔 회귀 테스트다 — 마지막 문장의 마침표를 경계로 잡되
소수점·약어는 제외하는 판정(`41268ba`), 스트림 종료 시 잔여 커밋이 flush 로 새던 두
경로(`e1564a3`). `test_final_residual_commit.py` 는 모델을 띄우지 않으려고 핸들러를
`object.__new__` 로 만들어 필요한 속성만 채우므로, 커밋 경로가 새 속성을 참조하기
시작하면 AttributeError 로 먼저 깨진다.

`tests/test_streaming_architecture.py` 는 성격이 다르다. 실제로 뜬 서버에 붙는 통합
테스트이고(`QWEN3_TEST_AUTO_SERVER=1` 로 직접 띄우게 할 수도 있다), 검사하는 것은
start/ready/finish 흐름과 페어링 프로토콜 둘뿐이다. 페어링은 지금 어느 클라이언트도
쓰지 않는다 — 서버에서 걷어낼 때 이 테스트도 함께 정리해야 한다.
