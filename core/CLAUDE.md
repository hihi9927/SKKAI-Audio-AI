# core/

파이프라인 추상화 + LLM 교정/번역 레이어. 서버 코드는 포함하지 않는다.

| 경로 | 역할 |
|---|---|
| @types.py | 파이프라인 각 단계 데이터클래스 (`AudioSegment` → `TranslationResult`) |
| @modules.py | 추상 베이스 클래스 시그니처 (구현 없음) |
| `correct_and_trans.py` | `GPTTranslator` — 교정 + 번역 단일 GPT 호출. 서버 `--gpt-translation`으로 활성화 |
| `llm_corrector/gpt_corrector.py` | `GPTCorrector` — 교정만. 서버 `--correction`으로 활성화 |
| `meaning_segmentator/utils/` | 의미 분절 연구 스크립트 (GPT `<SEG>` 마킹, 점진적 컨텍스트 번역, COMET 평가) |
| `meaning_segmentator/autoseg/` | 분절 프롬프트 자동 생성 에이전트 루프 (v2). 설계 @meaning_segmentator/AUTO_PROMPT_LOOP_DESIGN.md, 사용법 @meaning_segmentator/autoseg/README.md |
| `meaning_segmentator/metric_probes/` | `contradiction`(NLI) 지표 백엔드 대체 탐침. **루프 경로 아님** — 기존 런을 읽어 재활용. 결론 @meaning_segmentator/NLI_ALTERNATIVES.md |
| `meaning_segmentator/docs/` | 진단 기록. 순위 축·비용 구조·루프 채택 실패 원인과 처방 @meaning_segmentator/docs/RANK_METRIC_DIAGNOSIS.md |
| ⤷ 근거·이력 | 문헌 대조 @meaning_segmentator/SEGMENTATION_CRITERIA_RELATED_WORK.md |
| `research/cif`, `research/context_scoring` | CIF·컨텍스트 스코어링 실험. 런타임 경로 아님 |

두 GPT 모듈 모두 기본 모델은 `gpt-5.4-mini` (런타임 경로. `autoseg/`의 모델과 무관하다). 두 플래그 모두 꺼져 있으면 서버는 Google Translate로 번역하므로 `core/`의 GPT 경로를 아예 타지 않는다.

## 규칙

- `modules.py`에는 추상 시그니처만 — 구현 추가 금지.
- 학습/실험 코드는 `research/` 아래에만. 런타임 파일 옆에 두지 말 것.
- 런타임 경로(`types.py`, `modules.py`, `correct_and_trans.py`, `llm_corrector/`) 의존성은 stdlib, `numpy`, `openai`만. 연구 스크립트는 각자 `requirements.txt`를 갖는다 (예: `meaning_segmentator/requirements.txt`).
- `autoseg/`는 OpenAI 또는 Letsur AI Gateway를 쓴다 — 엔드포인트는 키가 온 **환경변수 이름**이 정한다 (`OPENAI_API_KEY` → OpenAI, `LETSUR_API_KEY`/`CLAUDE_API_KEY` → Letsur). 접두사로 고르면 `sk-` 로 시작하는 Letsur 키를 OpenAI 로 보낸다. 기본 모델은 `gpt-5-mini` (실측 근거 @meaning_segmentator/docs/RANK_METRIC_DIAGNOSIS.md 부록). 에이전트 호출은 `httpx`만 있으면 되지만, 지표 백엔드(COMET/CometKiwi)는 `unbabel-comet` + GPU가 필요하다. CometKiwi는 HF 게이트 모델이라 라이선스 동의 + `hf auth login` 선행 (구버전은 `huggingface-cli login`).
- **관문을 먼저 통과시킨다.** consistency 백엔드를 바꾸면 `validity_check.py`, adequacy 백엔드를 바꾸면 `adequacy_check.py`, 판정자 모델이나 `JUDGE_SYSTEM`을 바꾸면 `judge_check.py`. 지표는 틀리면 숫자로 드러나지만 판정자는 조용히 루프를 발산시킨다. NLI contradiction 의 잡음 바닥·순위 정렬 재검은 `noise_floor.py`.
- 언어별 자원(형태소 분석기·의존 파서)을 `autoseg/`에 넣지 말 것. 언어 지식은 `measured_profile.json`(측정)과 `language_profile.json`(LLM)으로만 들어간다.
- `types.py`의 단계를 추가·변경하면 `modules.py` 시그니처와 `Qwen3-ASR/examples/streaming_websocket_server.py` 핸들러도 같이 고칠 것.
