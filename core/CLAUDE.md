# core/

번역 레이어 + 분절 연구 코드. 서버 코드는 포함하지 않는다.

| 경로 | 역할 |
|---|---|
| `translator/correct_and_trans.py` | `GPTTranslator` — 교정 + 번역 단일 GPT 호출. 서버 `--gpt-translation`으로 활성화 |
| `translator/gpt_corrector.py` | `GPTCorrector` — 교정만. 서버 `--correction`으로 활성화 |
| `translator/local_translator.py` | 로컬 번역기 — seq2seq(MADLAD/NLLB)와 `LLMTranslator`(지시형 LLM, **앞 발화를 문맥으로 받을 수 있는 유일한 백엔드**), 그리고 독립 번역 서버를 부르는 HTTP 클라이언트 `RemoteTranslator`. `make_translator` 가 모델 이름으로 고른다 |
| `translator/LOCAL_TRANSLATION.md` | 어떤 로컬 번역 모델을 올릴지, 문맥을 몇 턴 줄지 — 실측표 |
| `meaning_segmentator/utils/` | 의미 분절 연구 스크립트 (GPT `<SEG>` 마킹, 점진적 컨텍스트 번역, COMET 평가) |
| `meaning_segmentator/autoseg/` | 분절 프롬프트 자동 생성 에이전트 루프. 코드가 하는 일 @meaning_segmentator/autoseg/AUTOSEG_SIMPLIFY.md, 사용법 @meaning_segmentator/autoseg/README.md |
| ⤷ 근거·기각 기록 | 왜 이 지표 조합인가, 무엇을 검토하고 버렸나, 순위 축 진단, 참조 기반 평가 프로토콜 @meaning_segmentator/autoseg/AUTOSEG_DETAILS.md |
| `meaning_segmentator/autoseg/baselines/` | Table 1a 타 정책 구현(`punct`/`syntax`/`causal_align`/`alignatt`/`mu_prefix`) + 강제정렬 타임스탬프 빌더 |
| `meaning_segmentator/docs/X2EN_DATASET.md` | {de,ja,zh}→en 트랙 데이터셋 구축 기록. 소스 언어별 단위·`T` 환산과 오염 방지 구간 배분 |
| ⤷ 문헌 대조 | @meaning_segmentator/docs/SEGMENTATION_CRITERIA_RELATED_WORK.md |
| `research/cif`, `research/context_scoring` | CIF·컨텍스트 스코어링 실험. 런타임 경로 아님 |

두 GPT 모듈 모두 기본 모델은 `gpt-5.4-mini` (런타임 경로. `autoseg/`의 모델과 무관하다). 두 플래그 모두 꺼져 있으면 서버는 Google Translate로 번역하므로 `core/`의 GPT 경로를 아예 타지 않는다.

## 규칙

- 학습/실험 코드는 `research/` 아래에만. 런타임 파일 옆에 두지 말 것.
- **문맥은 LLM 백엔드만 받는다.** 서버의 `--local-translation-context N` 이 앞 발화 원문을 넘기고,
  seq2seq 번역기는 그걸 받아서 버린다(경고 1회). 이어붙여 넣는 `--google-context` 방식은 Google 이
  줄바꿈을 보존해 주기 때문에 되는 것이라 로컬 모델에서는 깨진다 — NLLB 는 줄 수를 안 지켜
  문맥 덩어리가 통째로 자막에 나가고, MADLAD 는 번역 대신 잡음을 뱉는다. 실측과 문맥 깊이별
  수치는 [translator/LOCAL_TRANSLATION.md](translator/LOCAL_TRANSLATION.md).
- 런타임 경로(`translator/`) 의존성은 GPT 쪽이 stdlib + `openai`, 로컬 번역기가 `transformers`/`torch` 다. 둘은 서로를 요구하지 않으며, `translator/__init__.py` 가 이름을 쓸 때 가져오는 것도 그래서다 — 로컬 번역 서버는 `openai` 없이 뜬다. `bitsandbytes` 는 `LLMTranslator` 를 4bit/8bit 로 올릴 때만 더 필요하고, 없으면 `--quant none` 으로 뜬다. 연구 스크립트는 각자 `requirements.txt`를 갖는다 (예: `meaning_segmentator/requirements.txt`).
- `autoseg/`는 Letsur AI Gateway / OpenAI / OpenAI 호환 로컬 서버를 쓴다 — 상대는 **`--provider {letsur,openai,local}`** 가 정하고(기본 `letsur`), 키는 그 프로바이더의 환경변수 하나만 본다 (`letsur`→`LETSUR_API_KEY`, `openai`→`OPENAI_API_KEY`, `local`→키 불필요). 환경변수를 순서대로 뒤지거나 `sk-` 접두사로 추측하던 종전 방식은 없앴다 — `.env` 에 키가 둘이면 명령줄만 봐서 어디로 갔는지 알 수 없었고 런 기록에도 안 남았다. 프로바이더와 해석된 `api_base_url` 은 런의 `config.json` 에 기록된다. 기본 모델은 `gpt-5-mini` (실측 근거 @meaning_segmentator/autoseg/AUTOSEG_DETAILS.md '순위 축 진단'). 에이전트 호출은 `httpx`만 있으면 되지만, 지표 백엔드(COMET/CometKiwi)는 `unbabel-comet` + GPU가 필요하다. CometKiwi는 HF 게이트 모델이라 라이선스 동의 + `hf auth login` 선행 (구버전은 `huggingface-cli login`).
- **관문을 먼저 통과시킨다.** consistency 백엔드를 바꾸면 `validity_check.py`, adequacy 백엔드를 바꾸면 `adequacy_check.py`, 판정자 모델이나 `JUDGE_SYSTEM`을 바꾸면 `judge_check.py`. 지표는 틀리면 숫자로 드러나지만 판정자는 조용히 루프를 발산시킨다. NLI contradiction 의 잡음 바닥·순위 정렬 재검은 `noise_floor.py`.
- 언어별 자원(형태소 분석기·의존 파서)을 `autoseg/`에 넣지 말 것. 언어 지식은 `measured_profile.json`(측정)과 `language_profile.json`(LLM)으로만 들어간다.
