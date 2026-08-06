# autoseg 인계 문서 — 실험 기록과 COMET 환경 이관

작성 시점: 2026-08-06. 대상 독자: COMET을 돌릴 수 있는 GPU 환경에서 이 작업을 이어받는 사람.

설계 근거는 [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md), 사용법은 [README.md](README.md).

---

## 1. 한 문단 요약

사람이 언어마다 손으로 쓰던 `<SEG>` 분절 프롬프트를 에이전트 루프로 자동 생성하는 시스템을 구현하고 일본어(ja→ko)로 6회 실험했다. **부트스트랩(사람 입력 0으로 언어 프로파일링 → 초기 프롬프트 생성)은 작동한다.** **개선 루프는 아직 검증되지 않았다** — 채택된 프롬프트가 매번 Profiler의 초기 프롬프트였다. 중단 사유는 루프 결함이 아니라 **품질 지표의 타당도 붕괴**다: 현재 환경에서 쓸 수 있는 백엔드(임베딩 코사인, chrF)가 분절이 실제로 일으키는 오류를 못 잡는다. COMET이 필요하다.

---

## 2. 실행 환경 (기존)

- 모델 게이트웨이: **Letsur AI Gateway** `https://gw.letsur.ai/v1` (OpenAI 호환)
  - 인증: `.env`의 `CLAUDE_API_KEY` (`LETSUR_API_KEY` 환경변수도 인식)
  - 사용 모델: `claude-sonnet-5` (분절·번역·에이전트 전부), `text-embedding-3-large` (Q)
  - 응답에 `estimated_cost` 필드가 있어 런 단위 비용 집계에 사용
- 로컬에 **`torch`·`comet`·`sacrebleu` 없음.** 그래서 COMET을 못 썼다.
- `temperature=0`을 주는데도 동일 입력이 서로 다른 번역을 낸다 (§4 참조). 게이트웨이/모델 어느 쪽 원인인지는 미확인.
- 프롬프트 캐싱이 걸리지 않는다 (`cached_tokens: 0`). 입력 토큰 전액 과금.

---

## 3. 구현 현황

| 파일 | 역할 | LLM |
|---|---|---|
| `gateway.py` | 게이트웨이 클라이언트, 재시도, 비용 집계, 예산 가드, JSON 복구 | — |
| `data.py` | A0 Data Preparer — 정규화, 문장 복원, 층화 분할. 로더 `kokoro`(ja) / `kspon`(ko) | — |
| `pipeline.py` | A2 Segmenter(자가복구 1회) / A3 Format Validator / A4 번역 툴 2종 / 디스크 캐시 | 분절·번역만 |
| `metrics.py` | A5 Scorer — Q(임베딩·chrF), `Q_seg`, LCB, L, 목적함수 | — |
| `agents.py` | A1 Profiler / A6 Critic / A7 Prompt Engineer, 프롬프트 골격 | ● |
| `loop.py` | A8 Loop Controller — 앵커 캘리브레이션, 채택·롤백, 조기 종료, 리포트 | — |
| `eval_prompt.py` | 임의 프롬프트 1개를 루프와 동일 지표로 평가 (사람 프롬프트 비교용) | — |
| `human_prompts/` | `utils/prompt.txt`에서 추출한 사람 작성 한국어 프롬프트 v1/v2 | — |

전부 동작하며 import 검증됨. LLM 판단은 `agents.py` 3곳뿐이고, 포맷 검증·점수·채택 판정·집계·재시도는 결정론적이다.

### 언어 무관성

에이전트 프롬프트 4종(Profiler, Prompt Writer, Critic, Prompt Engineer)에는 특정 언어 지식이 없다. 언어 지식은 **`language_profile.json`을 통해 데이터로만** 들어간다. 검증기의 구두점 목록도 프로파일의 `trailing_punctuation`에서 받고, 없으면 유니코드 범주로 추정한다(es `¿` 는 태그 뒤 허용, hi `।` 인식, th 는 규칙 자동 무력화).

**언어 종속이 남아 있는 곳은 `data.py`의 로더뿐이며 이는 불가피하다** — 파일 포맷과 전처리가 데이터셋 고유다. 새 언어는 `LOADERS`에 로더 하나만 추가하면 되고 프롬프트는 손대지 않는다.

---

## 4. 측정 결과 — 이것이 인계의 핵심

### 4.1 지표 타당도 (가장 중요)

같은 참조 문장에 오류를 주입해 백엔드 반응을 실측했다.

기준문: `곤은 뱀장어를 훔치지 않았습니다. 병십이 그것을 강에 돌려보냈습니다.`

| 가설 | 임베딩 | chrF |
|---|---|---|
| 동일 | 1.0000 | 1.0000 |
| 무해한 표기 차이 (장어/뱀장어, 헤이주/병십) | 0.8401 | 0.4597 |
| **부정 뒤집힘** (훔치지 않았다 → 훔쳤다) | **0.9278** | 0.7840 |
| **주체 뒤바뀜** (곤↔병십) | 0.8460 | 0.8244 |
| **뒷절 누락** | **0.8878** | 0.4920 |
| **지시대상 소실** (그/그것/거기) | 0.4862 | 0.4869 |
| 무관한 문장 | 0.1360 | 0.0411 |

ja 앵커: 하한 0.7341 / 상한 0.9010 / `Q_floor` 0.8509.

- 임베딩은 **의미가 정반대인 번역에 상한 앵커보다 높은 점수**를 준다.
- 절이 통째로 누락돼도 `Q_floor`를 통과한다.
- 반대로 무해한 표기 차이는 탈락시킨다.
- 순위가 `부정뒤집힘 > 절누락 > 무해한변이`로 뒤집혀 있다.

**이 상태에서는 목적함수가 성립하지 않는다.** 앵커 캘리브레이션은 스케일만 맞출 뿐 타당도를 보증하지 않는다.

chrF는 절 누락(0.4920)을 잘 잡지만 무해한 변이(0.4597)를 똑같이 벌하고 주체 뒤바뀜(0.8244)을 놓친다. 단독 사용 불가. BLEU는 chrF보다 표면형에 더 가혹하고 ja(무공백)·ko(교착어)에 구조적으로 부적합해 대안이 아니다.

### 4.2 번역기 비결정성

분절을 전혀 하지 않고 **같은 문장을 두 번 번역**한 결과 (ja→ko, n=25, `temperature=0`):

| | 값 |
|---|---|
| 임베딩 Q 평균 | 0.9273 |
| 중앙값 | 0.9516 |
| 최소 | 0.5866 |
| Q < 0.90 비율 | 24% |
| chrF 평균 | 0.7890 |

최악 사례:
```
1회: 밤새도록 좋은 방법이 없을까 생각한 끝에 떠올린 것은 천둥신에 관한 것이었습니다.
2회: 어디 뭔가 좋은 방법이 없을까 하고, 하룻밤 내내 생각한 끝에 떠올린 것이 우레신 이야기였습니다.
```

같은 런에서 고유명사 `兵十`가 **한자 유지 / 병십 / 헤이주** 세 가지로 번역됐고 인용부호도 `「」` 유지와 `""` 변환이 섞였다.

대응으로 세 번역 프롬프트에 표기 규범(고유명사 음역 강제, 인용부호 유지, 절 순서 유지)을 공통으로 못 박았다. 효과: `Q_seg` 0.8760 → 0.8898, 1차 포맷 통과율 0.90 → 0.93. 잔여 변동은 상한 앵커로 흡수한다.

**COMET에서 이 측정을 반드시 다시 하라.** 상한 앵커가 백엔드마다 다르다.

### 4.3 런 이력

데이터: KokoroSpeech(ja) 낭독 동화 2편. 원본 308행 → 문장 복원 후 168문장(평균 39자).

| 런 | 변경점 | 결과 | 중단 사유 |
|---|---|---|---|
| test01 | 최초 | — | 입력이 문장이 아님(낭독 호흡 단위). Critic JSON 크래시 |
| test02 | 문장 복원(177) | iter0 V=0.95 gain=0.068 / iter1 악화 | Critic `aggregate` 누락 → PE 조향 상실 |
| test03 | Critic 집계 결정론화, Segmenter 자가복구 | iter0 V=0.95(1차 0.90) | 인용부호 미닫힘 강제절단 조각 → 모델 빈 출력 → V=1.0 원천 불가 |
| test04 | 데이터 정형성 검사(168) | **완주.** test V=0.98(1차 0.88), Q=0.9668, **Q_seg=0.8618(n=12)**, gain=0.0577, k=1.32, 분절률 0.24 | 정상 종료(patience 3). 채택=iter_00 |
| — | 노이즈 바닥 측정 | §4.2 | — |
| test05 | `Q_seg` 제약, LCB 전, train 30 | iter0 V=0.97 Q_seg=0.8760(n=11) gain=0.1154 | 상한 1.0 가정이 틀렸음을 발견 |
| test06 | **앵커 2개**, 번역 규범 고정 | iter0 V=1.00(1차 0.93) **Q_seg=0.8898(n=11) gain=0.1116 obj=+0.1116 — 처음으로 제약 만족** | 지표 타당도 붕괴 발견(§4.1). 이관 결정 |

누적 비용 약 7 units.

**test04가 유일한 완주 런이다.** 4 이터레이션, PE 개정 3회 전부 미채택, 최종 채택 프롬프트 = Profiler의 초기 프롬프트. 즉 개선 루프의 효과는 아직 0이다.

test06 마지막 Critic 출력은 `dominant=under_segmentation`, `direction=segment more aggressively`였다. 이전 런들이 영구히 `fix boundary placement`에 고정돼 PE를 단조 보수화시키던 문제(k 1.40→1.30, 분절률 0.35→0.25)는 해소됐다. **개선 루프는 고쳐졌지만 그 효과를 측정할 지표가 없어 검증되지 못했다.**

### 4.4 데이터 한계 (ja)

- 원본 8,319자 중 **6,621자(79.6%)만 채택.** `max_chars` 초과 강제절단분 폐기 정책 때문이며, 버려진 부분이 긴 문장에 편중돼 지연 이득이 과소평가될 수 있다.
- 첫 문장에 제목·저자·장번호가 붙어 있다 (`ごん狐新美南吉一これは、…`).
- 소스가 동화 2편(`ごんぎつね` 119문장, `カウカサスの禿鷹` 49문장)뿐 — **구어 ASR 분포가 아니고 도메인이 사실상 단일.**
- test 50문장 중 38개가 무분절이었다. 모델의 태만이 아니라 head-final 낭독 문어체에서 실제로 자를 데가 적어서일 가능성이 크다.

**ja는 루프 동작 검증용이지 성능 일반화 근거가 아니다.**

---

## 5. COMET 환경에서 할 일

### 5.1 설치와 백엔드 연결

```bash
pip install -r core/meaning_segmentator/requirements.txt   # unbabel-comet 포함
```

`metrics.py`에 COMET 백엔드를 추가한다. 인터페이스는 `embed_similarity(gw, hyps, refs)`와 동일하게 `list[float]`를 반환하면 되고, COMET은 `src`도 쓰므로 시그니처에 원문을 추가한다.

```python
def comet_similarity(srcs, hyps, refs, model_name="Unbabel/wmt22-comet-da", gpus=1):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))   # 프로세스당 1회만
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return model.predict(data, batch_size=8, gpus=gpus).scores
```

주의: `comet_eval.py`(기존 스크립트)는 실행마다 모델을 로드한다. 루프에서는 **반드시 재사용**할 것 — 이터레이션마다 로드하면 런 시간이 배로 든다.

`--quality-backend {embed,chrf,comet}` 플래그를 추가하고 `loop.evaluate`와 `calibrate_q_floor` 양쪽이 같은 백엔드를 쓰게 한다.

### 5.2 백엔드 교체 후 반드시 다시 할 것

이 순서를 지키지 않으면 같은 실패를 반복한다.

1. **타당도 표 재작성 (§4.1).** 오류 유형별 가설 문장에 COMET을 돌려 순위를 확인한다. 부정 뒤집힘이 무해한 변이보다 낮게 나와야 한다. 이 검사에 실패하면 그 백엔드도 쓸 수 없다.
2. **노이즈 바닥 재측정 (§4.2).** `translator.full_uncached()`로 동일 문장 2회 번역 후 COMET 점수를 낸다. 이것이 상한 앵커다.
3. **앵커 간격 확인.** `baseline.json`의 `anchor_gap`. 임베딩은 0.1669였다. COMET이 이보다 넓어야 분해능 개선이다. 0.05 미만이면 분절 품질을 분해하지 못한다는 뜻이다.
4. **`Q_floor` 재산출.** 코드가 자동으로 한다 — 상수를 손댈 필요 없다.

### 5.3 지표가 검증되면

**ko→en을 다음 실험으로 하라.** ja보다 강한 실험이다.

- 데이터: `evaluation/KsponSpeech/transcribe/eval_clean_1000.json` — 실제 자발 발화 1,000문장, 25자 이상 337건. 로더 `kspon` 구현 완료.
- 구어라 분절 여지가 낭독 문어체보다 훨씬 크다.
- **사람이 쓴 프롬프트가 이미 있다.** `human_prompts/ko_human_v1.txt`(2,506자), `ko_human_v2.txt`(3,110자). v2 = v1 + `[판단 절차]` 섹션 + 예시 1개이므로 **사람이 아는 품질 순서(v2 ≥ v1)가 정해져 있다.**

지표 검증과 성능 비교를 한 번에 한다:

```bash
# 1) 루프 실행
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id run01 \
    --iterations 6 --train 30 --dev 60 --test 100 --min-chars 25 --budget 20

# 2) 같은 분할·앵커로 사람 프롬프트 평가
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_v1.txt \
    --run-id ko-en/run01 --split test --label human_v1
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_v2.txt \
    --run-id ko-en/run01 --split test --label human_v2
```

**v2 > v1이 재현되지 않으면 에이전트가 아니라 지표를 더 고쳐야 한다.** 재현되면 그때 자동 생성 프롬프트와 비교한다.

`--dev 60`을 권장하는 이유: ja에서 dev 30일 때 분절된 문장이 4건뿐이라 `Q_seg` 추정이 무의미했다. LCB가 이를 벌하도록 해뒀지만 표본 자체를 늘리는 편이 낫다.

### 5.4 미해결 사항

1. **개선 루프의 효과가 아직 0이다.** 유일한 완주 런에서 PE 개정 3회 전부 미채택. 지표가 유효해진 뒤 다시 판단해야 한다. 그때도 개선이 없으면 PE의 탐색 전략(현재 단순 언덕오르기)을 바꿔야 한다.
2. **`temperature=0`이 안 먹는 원인 미확인.** 게이트웨이인지 모델인지. 결정론적 디코딩이 가능하면 상한 앵커가 1.0에 가까워져 분해능이 크게 개선된다.
3. **프롬프트 캐싱 미작동** (`cached_tokens: 0`). 시스템 프롬프트가 이터레이션 내 고정이라 캐싱이 걸리면 비용이 크게 준다.
4. **타깃 언어별 프롬프트 분리 여부.** 최적 분절은 언어쌍에 의존한다(ko→ja와 ko→en의 좋은 경계가 다르다). `runs/{pair}/` 구조가 양쪽을 수용하나 정책은 미정.
5. **`direction`이 train 지표로만 계산된다.** 위반 목록에는 dev 위반이 합쳐져 들어가므로 미세한 불일치가 있다.
6. **런타임 반영 경로 미정.** 현재 서버 커밋은 ASR의 SEG 토큰·구두점·VAD 기반이지 GPT 분절이 아니다. 이 루프 산출물이 (a) 연구용 데이터 생성, (b) 파인튜닝용 SEG 레이블, (c) 런타임 LLM 분절 중 어디로 가는지에 따라 요구 지연 예산이 달라진다.

---

## 6. 산출물 위치

```
core/meaning_segmentator/
  AUTO_PROMPT_LOOP_DESIGN.md      설계와 근거
  autoseg/
    HANDOFF.md                    이 문서
    README.md                     사용법
    *.py                          구현
    human_prompts/                사람 작성 프롬프트 (비교 기준)
  runs/ja-ko/
    ja-ko-test04/                 유일한 완주 런. final_report.md 있음
    ja-ko-test06/                 앵커 2개 적용 런 (iter 0까지)
```

`runs/**/cache/`와 `runs/**/*_rows.json`은 `.gitignore` 처리했다. 프롬프트·리포트·`baseline.json`·`history.json`은 추적된다.

`ja-ko-test01`~`test03`, `test05`는 폐기된 런이므로 삭제해도 무방하다.
