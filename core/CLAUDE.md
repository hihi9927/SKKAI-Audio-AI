# core/

런타임 파이프라인 추상화 레이어. 서버 코드와 연구 실험 코드는 포함하지 않는다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| @types.py | 파이프라인 각 단계의 데이터클래스 정의 (`AudioSegment` → `TranslationResult`) |
| @modules.py | 추상 베이스 클래스 시그니처 (구현 없음) |
| `llm_corrector/gpt_corrector.py` | 독립형 GPT 교정기 (ablation/단독 사용) |
| `correct_and_trans.py` | 교정 + 번역 통합 GPT 호출 (프로덕션 사용) |
| `meaning_segmentator/` | 의미 분절 연구 유틸리티 — GPT로 `<SEG>` 태그 마킹, 점진적 컨텍스트 번역, COMET 평가 스크립트 모음 |

## 경계 규칙

- **이 디렉토리 안에 학습 코드 추가 금지** — 연구 코드는 `research/`에 배치
- **`modules.py`에 구현 추가 금지** — 추상 시그니처만 유지
- 의존성: stdlib, `numpy`, `openai`만 허용. 새 최상위 의존성은 `Qwen3-ASR/pyproject.toml` 업데이트 필요

## `correct_and_trans.py` vs `llm_corrector/gpt_corrector.py`

- `correct_and_trans.py` (GPTTranslator): 교정 + 번역을 단일 API 호출로 처리 → **프로덕션 서버 사용**
- `llm_corrector/gpt_corrector.py` (GPTCorrector): 교정만 수행 → ablation/단독 테스트용

## `types.py` 수정 시

파이프라인 단계 추가 또는 기존 단계 필드 변경 시 `modules.py` 시그니처와 프로덕션 서버 핸들러도 함께 업데이트 필요.
