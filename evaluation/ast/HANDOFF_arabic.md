# 아랍어 지원 조사 — 인수인계

작성 2026-08-31. 브랜치 `autoseg-temp`.

STiTy 를 아랍어(요르단)로 돌릴 수 있는지, dot-commit 경로가 아랍어를 덮는지 확인한 작업이다.

> **2026-08-31 갱신 — 측정 완료. 이 문서의 "미완" 항목 1~4 와 패치 1·2 는 다 끝났다.**
> 결과는 [RESULTS.md](RESULTS.md) 의 "아랍어 ASR — 요르단 구어 vs MSA" 절에 있다.
> 요약: **정규화 WER 요르단 구어 40.96% vs MSA 9.77% — 격차 31.2%p(4.2배)**,
> **AST 지연은 punct 10.41초/BLEU 34.70, static@2s 1.10초/BLEU 8.55**(FLEURS ar→en 283발화),
> dot-commit 은 아랍어에서 **정상 발동**(모델이 848행에 `.` 728회), `؟` 패치는 구어에서
> 경계 +13.3%(69발화가 `؟` 로 끝남)·MSA 에서 0. 남은 것은 아래 "미완" 절 참고.

## 물음

1. 아랍어가 STiTy 에서 돌아가는 언어인가
2. dot-commit(`--enable-dot-commit`) 경로가 아랍어를 덮는가

## 1번 답 — 백엔드는 된다, 클라이언트가 막는다

| 층 | 상태 | 근거 |
|---|---|---|
| Qwen3-ASR | 지원 | `qwen_asr/inference/utils.py` `SUPPORTED_LANGUAGES` 에 `"Arabic"` |
| 서버 코드 매핑 | 지원 | `streaming_websocket_server.py` `LANG_NAME_TO_CODE` 에 `"Arabic": "ar"`, 역매핑 + `lang_to_code` 키워드 폴백 |
| `restrict_languages` 로짓 바이어스 | 동작 | `_make_language_logit_bias` 가 `SUPPORTED_LANGUAGES` 를 순회하므로 Arabic 통과 |
| 번역 (Google gtx) | 동작 | `tl=ar` |
| **모바일 클라이언트** | **막힘** | `STiTy-Mobile/src/constants/languages.ts` `LANGUAGES` 에 ko/ja/zh/es/en 5개뿐 — `ar` 없어 앱에서 선택 불가 |
| **TTS** | **구멍** | `src/utils/tts.ts` / `tts.web.ts` 의 `BCP47` 맵에 `ar` 없음. `toBCP47` 가 raw `'ar'` 을 흘려보내 보이스팩 유무에 따라 조용히 엉뚱한 언어로 폴백 |
| RTL | 없음 | `I18nManager` 사용처 0건 |

## 2번 답 — 반쯤 덮는다. 단 통념과 반대 방향이다

`qwen_asr/inference/sentence_boundary.py` 의 `DOT_COMMIT_BOUNDARY_RE` 는 `.` `?` `!`
+ CJK `。？！` + `<SEG>` 만 잡는다. **아랍어 물음표 `؟`(U+061F)가 빠져 있다.**
코드베이스 전체에 아랍어 문장부호는 0건이다.

`streaming_websocket_server_dualbase.py` 는 같은 판정을 인라인 정규식으로 따로
구현해 두었고(구버전 — 문자열 끝 경계를 인정하지 않는다), 거기에도 `؟` 가 없다.
**고칠 때 두 군데를 같이 고쳐야 한다.**

여기서 함정이 하나 있다. Casablanca 요르단 test 의 **참조 전사**는 이렇다.

```
، 249    ؟ 122    ! 137    . 2    : 1      (848 행)
```

ASCII 마침표를 사실상 안 쓴다. 이것만 보면 "아랍어는 `.` 을 안 쓰니 dot-commit 이
죽는다" 로 읽힌다. **틀린 추론이다.** dot-commit 을 굴리는 건 참조가 아니라 **모델
출력**이다. 16 발화 예비 측정에서 모델은 반대로 나왔다.

```
가설(모델 출력):  . 15    ، 3    ؟ 1      종결부호로 끝난 발화 15/16
참조:            ، 4              ؟ 1      (`.` 없음)
```

즉 Qwen3-ASR 은 아랍어에도 ASCII `.` 을 찍는다. **dot-commit 은 아랍어에서 정상
발동한다.** `؟` 패치는 여전히 필요하지만(16개 중 1회 출현) 우선순위는 처음 생각보다
낮다.

~~**이 숫자는 16 발화짜리다. 848 전체로 재확인하기 전에는 결론으로 쓰지 말 것.**~~
→ **848 전체로 확인됨(2026-08-31).** 모델 출력 `.` 728 / `،` 108 / `؟` 97, 종결부호로 끝난
발화 767/848. 예비 측정의 방향이 그대로 유지됐다. 다만 `؟` 는 97회로 무시할 양이 아니라,
패치 우선순위는 "낮음"이 아니라 **회화체에서 유의미**로 정정한다(경계 +13.3%).

## 방언 — 코드에는 영향 없다

아랍어는 다이글로시아 구조다. 문자·문장부호는 전 지역 동일하고(`؟` `،` `؛` 공통,
마침표는 어디서나 ASCII `.`), 쓰면 MSA·말하면 방언이다. 요르단은 **남부 레반트**
(팔레스타인/시리아남부/레바논과 한 묶음)이고, 암만은 팔레스타인 도시방언 + 베두인이
섞인 코이네다.

따라서 **파이프라인 텍스트 층에 지역 분기를 넣을 필요가 없다.** 방언 문제는 전부 ASR
음향 단계에 몰린다. 자원 순위는 MSA > 이집트 > 레반트 > 걸프 > 마그레브 로, 레반트는
중간이다.

표기 차이가 하나 있긴 하다 — 동부 아랍-인도 숫자(`١٢٣`, 마슈리크/걸프/이집트) vs
ASCII(`123`, 마그레브), 소수점 `٫`(U+066B) vs `.`. 다만 Casablanca 요르단 test 는
숫자 포함 행이 0 이라 이 코퍼스에선 무의미하다.

## 데이터

로컬에 아랍어 데이터는 없었다. 아래 둘을 새로 받았다.

| 데이터 | 경로 | 규모 | 용도 |
|---|---|---|---|
| Casablanca Jordan | `~/datasets/casablanca/Jordan/` | test 848행 0.98h / validation 848행 1.00h | 요르단 구어 |
| FLEURS `ar_eg` | `~/datasets/fleurs/data/ar_eg/` | test 283행 0.88h | MSA 대조군 |

규모가 비슷해 그대로 비교하면 된다. **격차 = 방언 열화 폭**이 이번 측정의 목표다.

Casablanca 특성:
- 44.1kHz **스테레오** WAV PCM_16 — STiTy 는 16k 모노 s16le 고정이라 변환 필수
- 출처는 TV 드라마(`مسلسل الشريكان`). 연기된 구어라 낭독체는 아니다
- 전사는 MSA 정규화 안 된 진짜 레반트다 (`بدك تيجي تسهر عنا` 의 `بدك`/`تيجي` 는 MSA 아님)
- 번역 참조가 없다 — `transcription` 한 컬럼뿐이라 BLEU/COMET 은 못 낸다. **WER 전용**
- train 미공개, validation/test 만 배포

### 라이선스 주의

Casablanca 는 **CC-BY-NC-ND-4.0** 이다.

- **NC(비상업)** — 내부 연구 평가는 통상 허용 범위로 보지만, STiTy 가 제품화 경로면 법무 확인이 안전하다
- **ND(변경 금지)** — 파생물 배포 금지. 읽고 측정만 하면 무관하나, 이걸로 파인튜닝한 모델이나 가공 데이터를 **배포**하면 걸린다

이번 작업은 측정 전용이라 문제 없다. 학습·재배포로 넘어가면 다시 확인할 것.

## 추가한 코드

| 파일 | 역할 |
|---|---|
| `evaluation/ast/build_manifest_casablanca.py` | parquet → 16k 모노 wav 추출 + 매니페스트. FLEURS 빌더와 스키마 동일(`tgt_text` 는 빈 문자열) |
| `evaluation/ast/asr_eval_arabic.py` | WER(raw/정규화) + **모델 출력 문장부호 분포** 측정. 서버 불필요한 오프라인 경로 |

`test_ast.py` 를 안 쓴 이유: WebSocket 서버 + BLEU 참조가 필요한 AST 하네스라
지금 물음(WER, 부호 분포)에는 과하다.

`asr_eval_arabic.py` 는 아랍어 WER 정규화를 함께 낸다 — 타슈킬 제거, 알레프 통일
(`أإآٱ`→`ا`), `ة`→`ه`, `ى`→`ي`. 표기 변이 때문에 raw WER 이 품질을 과소평가해서
**raw 와 정규화 둘 다** 출력한다. 한쪽만 보면 오해한다.

## 재현 절차

```bash
conda activate speech_ai        # STiTy Python 작업은 전부 이 env

# 1. 데이터 (Jordan 만 — 전체 8방언 9.16GB 받지 말 것)
hf download UBC-NLP/Casablanca --repo-type dataset \
    --include "Jordan/*" --local-dir ~/datasets/casablanca
hf download google/fleurs --repo-type dataset \
    --include "data/ar_eg/test.tsv" "data/ar_eg/audio/test.tar.gz" \
    --local-dir ~/datasets/fleurs

# 2. 매니페스트 (.jsonl 은 gitignore 대상 — 매번 재생성한다)
python evaluation/ast/build_manifest_casablanca.py \
    --dialect Jordan --split test \
    --out evaluation/ast/manifests/casablanca_jordan_test.jsonl

python evaluation/ast/build_manifest_fleurs.py \
    --fleurs-root ~/datasets/fleurs --src ar_eg --tgt ar_eg --split test \
    --out evaluation/ast/manifests/fleurs_ar_test.jsonl

# 3. 측정 — 요르단 구어
python evaluation/ast/asr_eval_arabic.py \
    --manifest evaluation/ast/manifests/casablanca_jordan_test.jsonl \
    --out evaluation/ast/results/asr_ar/casablanca_jordan_test.json \
    --batch-size 32 --backend-kwargs '{"gpu_memory_utilization": 0.2}'

# 4. 측정 — MSA 대조군
python evaluation/ast/asr_eval_arabic.py \
    --manifest evaluation/ast/manifests/fleurs_ar_test.jsonl \
    --out evaluation/ast/results/asr_ar/fleurs_ar_test.json \
    --batch-size 32 --backend-kwargs '{"gpu_memory_utilization": 0.2}'
```

### 환경 함정

- **transformers 백엔드는 쓰지 말 것.** GB10 에서 4.2초 오디오에 31.6s/발화가 나왔다.
  848 발화면 7.4시간이다. `--backend vllm`(기본값)을 쓴다.
- **vLLM 기본 `gpu_memory_utilization=0.9` 는 실패한다.** GB10 은 통합메모리
  121.69 GiB 인데 다른 프로세스가 상시 40G 대를 점유해서, 0.9(109.52 GiB) 요구가
  `ValueError: Free memory on device cuda:0 (76.44/121.69 GiB) ... less than desired`
  로 죽는다. **1.7B 모델이라 0.2 로 충분하다.**
- **`datasets` 로 Casablanca 오디오를 열지 말 것.** datasets 4.x 는 디코딩에
  `torchcodec` 을 요구한다. speech_ai env 의 vllm 의존성을 건드리기 싫어
  `build_manifest_casablanca.py` 는 pyarrow 로 직접 읽는다.
- GB10 은 `cuda capability 12.1` 경고가 뜨지만(PyTorch 지원 상한 12.0) 동작에는 지장 없었다.

## 미완 — 여기서 이어가면 된다

1. ~~Casablanca 848 full 추론~~ **완료** — WER raw 46.66% / 정규화 40.96%.
2. ~~FLEURS `ar_eg` MSA 기준선~~ **완료** — WER raw 15.02% / 정규화 9.77%.
3. ~~격차 계산~~ **완료** — 정규화 WER 31.2%p(4.2배). 정규화가 양쪽 다 5~6%p 만 깎으므로
   표기 변이가 아니라 음향·어휘의 실제 열화다.
4. ~~모델 출력 부호 분포 확정~~ **완료** — 위 "2번 답" 절 갱신 참고.
5. ~~요약을 `RESULTS.md` 에~~ **완료**.

6. **AST 지연 측정** **완료** — `run_fleurs_ar.sh`, FLEURS ar_eg→en 283발화.
   dot-commit 이 스트리밍에서 실제 발동함을 확인(커밋 324건 전부 `dot`).
   **단 seg 축은 아랍어에서 불가** — 베이스는 `<SEG>` 미출력, `en-dailytalk-seg` 는 영어 전용.
   그래서 지금 아랍어는 1.1초/BLEU 8.55 와 10.4초/BLEU 34.70 **양극단뿐이고 중간이 없다.**
   중간을 채우려면 **아랍어 SEG 파인튜닝이 선결 과제**다.

**남은 판단(사용자 몫).** 40.96% 를 보고 ① 요르단 구어까지 파인튜닝할지 ② 용도를 MSA·격식
발화로 좁힐지 결정해야 한다. 단 Casablanca 는 TV 드라마 연기 구어라 낭독체 대비 어려운
쪽이므로, 이 값을 "아랍어 회화 일반"의 상한으로 읽으면 안 된다. 더 정하려면 레반트
낭독체/대화 코퍼스가 하나 더 필요하다.

**환경 주의.** 이 문서의 재현 절차는 GB10 기준이다. `skkai`(RTX 4090 24GB)에서는
`speech_ai` conda env 가 없어 `PYTHONPATH= .venv/bin/python` 을 쓰고,
`gpu_memory_utilization` 은 **0.2 가 아니라 0.7** 이다(0.2 는 4.9GB 로 모자란다).

## 패치 대기 목록

방언·측정 결과와 무관하게 필요한 것들이다.

1. ~~`qwen_asr/inference/sentence_boundary.py` — 문자 클래스에 `؟` 추가~~
   **완료(`1bae6fc`)** — `DOT_COMMIT_BOUNDARY_RE`·`SENTENCE_SPLIT_RE` 둘 다,
   `؟`(U+061F)와 `۔`(U+06D4) 함께. 아랍 문자는 RTL 이라 정규식 리터럴로 박으면 소스 줄이
   시각적으로 뒤섞이므로 `\u` 이스케이프로 넣었다.
2. ~~`streaming_websocket_server_dualbase.py` 인라인 정규식에 동일 반영~~ **완료(`1bae6fc`)**.
3. `STiTy-Mobile/src/constants/languages.ts` — `ar` 엔트리 추가 + 기존 5개 언어의
   `translations` 필드에 아랍어 이름 채우기.
4. `tts.ts` / `tts.web.ts` `BCP47` — `ar-JO` 우선, 없으면 `ar` 폴백.
5. RTL 대응 — 별개 UI 작업. 범위 큼.

1·2 번은 완료했다. **3·4 번이 남았고, 아랍어를 실제로 서비스하려면 필수다** — 지금은
백엔드가 아랍어를 처리해도 앱에서 `ar` 을 고를 수 없어 끝까지 흐르지 않는다. 5 번(RTL)은
범위가 큰 별개 UI 작업이다.
