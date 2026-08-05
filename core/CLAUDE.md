# core/

파이프라인 추상화 + LLM 교정/번역 레이어. 서버 코드는 포함하지 않는다.

| 경로 | 역할 |
|---|---|
| @types.py | 파이프라인 각 단계 데이터클래스 (`AudioSegment` → `TranslationResult`) |
| @modules.py | 추상 베이스 클래스 시그니처 (구현 없음) |
| `correct_and_trans.py` | `GPTTranslator` — 교정 + 번역 단일 GPT 호출. 서버 `--gpt-translation`으로 활성화 |
| `llm_corrector/gpt_corrector.py` | `GPTCorrector` — 교정만. 서버 `--correction`으로 활성화 |
| `meaning_segmentator/utils/` | 의미 분절 연구 스크립트 (GPT `<SEG>` 마킹, 점진적 컨텍스트 번역, COMET 평가) |
| `research/cif`, `research/context_scoring` | CIF·컨텍스트 스코어링 실험. 런타임 경로 아님 |

두 GPT 모듈 모두 기본 모델은 `gpt-5.4-mini`. 두 플래그 모두 꺼져 있으면 서버는 Google Translate로 번역하므로 `core/`의 GPT 경로를 아예 타지 않는다.

## 규칙

- `modules.py`에는 추상 시그니처만 — 구현 추가 금지.
- 학습/실험 코드는 `research/` 아래에만. 런타임 파일 옆에 두지 말 것.
- 런타임 경로(`types.py`, `modules.py`, `correct_and_trans.py`, `llm_corrector/`) 의존성은 stdlib, `numpy`, `openai`만. 연구 스크립트는 각자 `requirements.txt`를 갖는다 (예: `meaning_segmentator/requirements.txt`).
- `types.py`의 단계를 추가·변경하면 `modules.py` 시그니처와 `Qwen3-ASR/examples/streaming_websocket_server.py` 핸들러도 같이 고칠 것.
