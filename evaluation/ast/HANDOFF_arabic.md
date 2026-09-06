# 아랍어 지원 — 조사 결과와 남은 일

브랜치 `autoseg-temp`. 조사 2026-08-31, 측정 완료 2026-09-01.

STiTy 를 아랍어로 돌릴 수 있는가를 확인한 작업이다. **답은 "된다, 단 앱이 막고 있고
품질은 방언에서 무너진다"** 이다. 아래 1~3장이 현재 상태이고, 4장이 이미 끝낸 일,
5장 이후는 배경·재현 자료다.

상세 수치와 도출 과정은 [RESULTS.md](RESULTS.md) 의 "아랍어 ASR" · "아랍어 AST 지연" 절에 있다.

---

## 1. 결론 — 지금 아는 것

### ① 백엔드는 아랍어를 처리한다. 앱이 막는다

| 층 | 상태 | 근거 |
|---|---|---|
| Qwen3-ASR | ✅ | `SUPPORTED_LANGUAGES` 에 `"Arabic"` |
| 서버 언어 매핑 | ✅ | `LANG_NAME_TO_CODE` 에 `"Arabic": "ar"` + 역매핑 + 키워드 폴백 |
| 로짓 바이어스 | ✅ | `_make_language_logit_bias` 가 `SUPPORTED_LANGUAGES` 순회 |
| 번역 | ✅ | Google `tl=ar`, MADLAD `<2ar>` 둘 다 |
| dot-commit 경계 | ✅ | `؟` 추가 완료 (4장) |
| **모바일 앱** | ❌ **막힘** | `languages.ts` 에 ko/ja/zh/es/en 5개뿐 — `ar` 이 없어 **선택 자체가 불가** |
| **TTS** | ⚠️ **구멍** | `BCP47` 맵에 `ar` 없음. `toBCP47` 가 raw `'ar'` 을 흘려 보이스팩에 따라 조용히 엉뚱한 언어로 폴백 |
| RTL | ❌ 없음 | `I18nManager` 사용처 0건 |

### ② 방언이 품질을 무너뜨린다 — MSA 대비 3~4배

Qwen3-ASR-1.7B 베이스, 언어 `Arabic` 고정, 감지 언어 100% Arabic.

| | 발화 | WER raw | **WER 정규화** | CER raw | **CER 정규화** |
|---|---|---|---|---|---|
| Casablanca 요르단 (구어) | 848 / 0.98h | 46.66% | **40.96%** | 14.88% | **12.85%** |
| FLEURS `ar_eg` (MSA) | 283 / 0.88h | 15.02% | **9.77%** | 5.12% | **3.95%** |

**격차 = 방언 열화 폭: 정규화 WER 31.2%p(4.2배), CER 3.25배.** 정규화(타슈킬 제거·알레프
통일·`ة→ه`·`ى→ي`)가 양쪽 다 5~6%p 만 깎으므로 **표기 변이가 아니라 음향·어휘의 실제 열화**다.

> **WER 단독으로 읽지 말 것.** 아랍어 어절은 평균 3.87글자뿐이라 글자 하나만 틀려도 어절
> 전체가 오답이 된다(`WER ≈ CER × 어절당 글자수`). 실제로 틀린 어절의 **59.9%가 1~2글자
> 차이**이고 완전 오인식은 12.4%뿐이다. 접어(`وعلى` = `و`+`على`)가 공백 없이 붙는 것도
> 같은 방향으로 작용한다. **CER 을 같이 봐야 품질이 제대로 보인다.**

MSA CER 3.95% 는 실사용 가능하다. 요르단 구어 CER 12.85% 는 여덟 글자에 하나 틀리는
수준으로 **아직 부족하다.** 단 Casablanca 는 TV 드라마 연기 구어라 낭독체보다 어려운
쪽이므로, 이 값을 "아랍어 회화 일반"의 상한으로 읽으면 안 된다.

### ③ 실시간 지연 — `static@8s` 가 현재 최선의 운용점

FLEURS `ar_eg`→en 283발화, 베이스 모델, 16클라이언트, 번역 MADLAD 로컬.
전 축 빈 가설 0 / 경계 넘은 final 0 / 번역 실패 0.

| 축 | LAAL | BLEU | 지연 1초당 BLEU |
|---|---|---|---|
| static@2s | 1.10초 | 8.55 | — |
| static@4s | 2.12초 | 18.77 | 9.11 |
| static@6s | 3.33초 | 24.56 | 3.19 |
| **static@8s** | **4.67초** | **28.74** | 2.60 |
| static@9s | 5.42초 | 29.20 | 0.61 ← 평탄 |
| punct (dot-commit) | 10.41초 | **34.70** | 1.10 |

- **8초가 static 의 한계점.** 8→9초에서 지연 1초당 사는 BLEU 가 2.60 → 0.61 로 급락한다.
- **static 은 punct 를 못 따라잡는다** — 29.20 대 34.70, **5.50 BLEU 부족.** 영어 표에서
  static COMET 이 punct 에 못 미친 채 평평해진 것과 같은 형태다.
- **대신 지연이 절반이다.** punct 는 BLEU 5.96 을 더 얻으려고 5.74초를 더 쓴다(초당 1.04).
  평균 발화가 11.14초인데 10.41초를 기다리는 것은 사실상 "문장 다 들을 때까지 대기"라
  실시간 대화용으로 못 쓴다.
- **seg 축은 아랍어에서 찍을 수 없다.** 베이스는 `<SEG>` 를 뱉지 않고 `en-dailytalk-seg` 는
  영어 전용이다. 영어 기준 seg 는 6.075초에 COMET 최고로 static 전 구간을 위에서 눌렀다.

**dot-commit 은 아랍어에서 정상 발동한다 — 스트리밍으로 확인했다.** 커밋 324건이 전부
`dot` 이고 발화당 1.15건이다. 영어 punct(9.044초)와 같은 대역이라 아랍어라서 느린 게 아니다.

---

## 2. 남은 일 — 우선순위 순

### A. 앱이 아랍어를 못 고른다 (서비스하려면 필수)

백엔드가 처리해도 **앱에서 `ar` 을 선택할 수 없어 끝까지 흐르지 않는다.**

1. `STiTy-Mobile/src/constants/languages.ts` — `ar` 엔트리 추가 + 기존 5개 언어의
   `translations` 필드에 아랍어 이름 채우기.
2. `tts.ts` / `tts.web.ts` 의 `BCP47` — `ar-JO` 우선, 없으면 `ar` 폴백.
3. RTL 대응 (`I18nManager`) — 범위가 큰 별개 UI 작업.

### B. 사용자 판단이 필요한 갈림길

요르단 구어 CER 12.85% 를 보고 결정해야 한다.

- ① 요르단 구어까지 **파인튜닝**한다
- ② 용도를 **MSA·격식 발화로 좁힌다** (MSA CER 3.95% 는 지금도 쓸 만하다)

판단을 더 정확히 하려면 **레반트 낭독체/대화 코퍼스가 하나 더** 필요하다. Casablanca 만으로는
"연기된 드라마 구어"라는 편향이 남는다.

### C. 품질 상한을 올리려면 — 아랍어 SEG 파인튜닝

시급하진 않다(`static@8s` 로 쓸 수 있다). 다만 지금 4.67초에 punct 대비 **BLEU 5.50 을
손해보고 있고**, 그 손해를 없애는 것이 seg 의 몫이다. 영어에서 seg 가 static 전 구간을
위에서 누른 형태라면, 아랍어 SEG 도 **더 낮은 지연에 더 높은 품질**을 줄 것으로 본다.

### D. 확증 못 한 것

- **`؟` 패치의 실측 효과.** 지금의 +13.3% 는 모델 출력 텍스트에 정규식을 다시 돌린
  **정적 추정**이지 실제 커밋 타이밍이 아니다. `؟` 가 나오는 것은 Casablanca 구어인데
  거기서는 LAAL 이 성립하지 않아(아래 함정) 검증 경로가 막혀 있다.
  FSL·커밋 사유 분포만 재는 것은 가능하다.
- **채점기 검산의 드리프트.** static 은 정의상 `LAAL ≈ 청크/2` 인데 오차가 청크와 함께
  +5.9% → +20.4% 로 벌어진다. 평균 발화 11.14초라 9초 청크면 발화당 청크가 1.24개뿐이라
  근사가 깨지는 것으로 보이나 **확증은 못 했다.** 청크를 발화 길이 위로 더 밀면 갈린다.

---

## 3. 함정 — 다시 밟지 말 것

- **Casablanca 로는 LAAL 을 못 낸다.** `tgt_text` 가 비어 있어 `metrics_ast.py:141` 이
  `n_ref=None` 을 넘기고 `:117` 에서 분모가 `max(n_hyp, 0)` = `|Y_hyp|` 가 된다 —
  **LAAL 이 조용히 AL 로 바뀌어** 짧게 생성할수록 지연이 작아 보이는 구멍이 열린다.
  지연 측정은 반드시 FLEURS(n-way 병렬)로 한다.
- **소규모 스모크의 CA/FTL 은 읽지 말 것.** 8발화 스모크에서 CA 42.6초가 나왔는데,
  16클라이언트가 동시 출발하면서 MADLAD 첫 로드(~35초)를 전부 뒤집어쓴 artifact 였다.
  283발화에서는 CA−NCA 가 566ms 로 정상이다.
- **Cloud Translation v2 키는 죽어 있다** (403 User Rate Limit Exceeded, 2026-08-31 확인).
  번역은 `--trans-backend local`(MADLAD-400-3B)로 돈다.
- **transformers 백엔드를 쓰지 말 것.** 4.2초 오디오에 31.6s/발화가 나온다(848발화면 7.4시간).
  `--backend vllm`(기본값)을 쓰면 848발화가 **17초**다.
- **`datasets` 로 Casablanca 오디오를 열지 말 것.** datasets 4.x 는 디코딩에 `torchcodec` 을
  요구한다. `build_manifest_casablanca.py` 는 pyarrow 로 직접 읽는다.
- **FLEURS TSV 따옴표 버그는 `ar_eg` 에 해당 없다.** 기본 파싱과 `QUOTE_NONE` 이 둘 다
  428행이고 3행이 리터럴 `"` 유지 여부만 다르다(행 삼킴 없음). 그래서
  `build_manifest_fleurs.py` 는 손대지 않았다 — 고치면 기존 AST 매니페스트가 바뀌므로
  여전히 사용자 판단 대기다.

---

## 4. 이미 끝낸 일 (요약)

| # | 한 일 | 결과 | 커밋 |
|---|---|---|---|
| 1 | Casablanca 요르단 848발화 추론 | WER 정규화 40.96% / CER 12.85% | `67c8e73` |
| 2 | FLEURS `ar_eg` MSA 기준선 283발화 | WER 정규화 9.77% / CER 3.95% | `67c8e73` |
| 3 | 방언 열화 폭 산출 | WER 31.2%p(4.2배), CER 3.25배 | `67c8e73` |
| 4 | 모델 출력 부호 분포 848 확정 | `.` 728 / `،` 108 / `؟` 97, 종결 767/848 | `67c8e73` |
| 5 | dot-commit 경계에 `؟` 추가 | 구어 경계 +13.3%, MSA +0 | `1bae6fc` |
| 6 | AST 지연 측정 (`run_fleurs_ar.sh`) | punct 10.41초/34.70, 커밋 324건 전부 `dot` | `e18191f` |
| 7 | static 청크 스윕 2~9초 | 8초 평탄, static@8s 4.67초/28.74 | `a675edc`, `4ef4c47` |

**5번 상세.** `sentence_boundary.py` 의 `DOT_COMMIT_BOUNDARY_RE`·`SENTENCE_SPLIT_RE` 에
`؟`(U+061F)·`۔`(U+06D4) 를 넣었다. 부호를 쥔 곳은 이 파일 하나뿐이다. 아랍 문자는 RTL 이라
정규식 리터럴로 박으면 소스 줄이 시각적으로 뒤섞이므로 `\u` 이스케이프로 넣었다. 라틴/CJK/약어(`Mr.`)/소수점(3.14)
회귀 없음을 확인했다.

**추가한 코드.**

| 파일 | 역할 |
|---|---|
| `build_manifest_casablanca.py` | parquet → 16k 모노 wav + 매니페스트. FLEURS 와 스키마 동일(`tgt_text` 는 빈 문자열) |
| `asr_eval_arabic.py` | WER/CER(raw·정규화) + 모델 출력 부호 분포. 서버 불필요한 오프라인 경로 |
| `run_fleurs_ar.sh` | 아랍어 AST 지연 측정. 축은 static·punct 둘뿐(seg 불가) |

### 왜 처음에 dot-commit 이 죽는 줄 알았나 (기록)

Casablanca **참조 전사**는 `، 249 / ؟ 122 / ! 137 / . 2 / : 1` 로 ASCII 마침표를 사실상 안 쓴다.
이것만 보면 "아랍어는 `.` 을 안 쓰니 dot-commit 이 죽는다"로 읽힌다. **틀린 추론이다** —
dot-commit 을 굴리는 건 참조가 아니라 **모델 출력**이고, 모델은 848행에 `.` 을 **728회** 찍는다.
같은 함정을 다른 언어에서도 밟지 않도록 남겨 둔다.

---

## 5. 배경 — 방언과 데이터

### 방언: 코드에는 영향 없다

아랍어는 다이글로시아 구조다. 문자·문장부호는 전 지역 동일하고(`؟` `،` `؛` 공통, 마침표는
어디서나 ASCII `.`), **쓰면 MSA·말하면 방언**이다. 요르단은 **남부 레반트**(팔레스타인/
시리아남부/레바논과 한 묶음)이고, 암만은 팔레스타인 도시방언 + 베두인이 섞인 코이네다.

따라서 **파이프라인 텍스트 층에 지역 분기를 넣을 필요가 없다.** 방언 문제는 전부 ASR 음향
단계에 몰린다. 자원 순위는 MSA > 이집트 > 레반트 > 걸프 > 마그레브 로, 레반트는 중간이다.

표기 차이가 하나 있다 — 동부 아랍-인도 숫자(`١٢٣`, 마슈리크/걸프/이집트) vs ASCII(`123`,
마그레브), 소수점 `٫`(U+066B) vs `.`. 다만 Casablanca 요르단 test 는 숫자 포함 행이 0 이라
이 코퍼스에선 무의미하다.

**아랍어는 공백을 쓴다.** 848발화 전부 공백이 있고 평균 10.4어절 / 49.8글자다. 필기체라
한 단어 안에서 글자가 이어 붙고 일부 글자(`ا د ذ ر ز و`)는 뒤와 안 이어져 단어 중간에 틈이
생기는데, 그게 공백처럼 보일 뿐이다. 공백을 안 쓰는 건 중국어·일본어·태국어 쪽이다.

### 데이터

| 데이터 | 경로 | 규모 | 용도 |
|---|---|---|---|
| Casablanca Jordan | `~/datasets/casablanca/Jordan/` | test 848행 0.98h / validation 848행 1.00h | 요르단 구어 (WER 전용) |
| FLEURS `ar_eg` | `~/datasets/fleurs/data/ar_eg/` | test 283행 0.88h | MSA 대조군 + **AST 지연** |

Casablanca 특성:

- 44.1kHz **스테레오** WAV PCM_16 — STiTy 는 16k 모노 s16le 고정이라 변환 필수
- 출처는 TV 드라마(`مسلسل الشريكان`). 연기된 구어라 낭독체가 아니다
- 전사는 MSA 정규화 안 된 진짜 레반트다 (`بدك تيجي تسهر عنا` 의 `بدك`/`تيجي` 는 MSA 아님)
- **번역 참조가 없다** — `transcription` 한 컬럼뿐이라 BLEU/COMET/LAAL 을 못 낸다
- train 미공개, validation/test 만 배포

**라이선스: CC-BY-NC-ND-4.0.** NC(비상업) — 내부 연구 평가는 통상 허용 범위로 보지만 제품화
경로면 법무 확인이 안전하다. ND(변경 금지) — 읽고 측정만 하면 무관하나 **이걸로 파인튜닝한
모델이나 가공 데이터를 배포하면 걸린다.** 이번 작업은 측정 전용이라 문제 없다. 위 2-B 의
①(파인튜닝)을 고르면 **다시 확인할 것.**

---

## 6. 재현 절차

**환경.** 이 머신(`skkai`, RTX 4090 24GB)에는 `speech_ai` conda env 가 없다.
`PYTHONPATH= .venv/bin/python` 으로 실행한다 — ROS 의 `PYTHONPATH` 가 venv 를 가리므로
비우는 것이 필수다.

**`gpu_memory_utilization`.** 24GB 기준이다. 오프라인 WER 측정은 **0.7**, 서버 + MADLAD 를
같이 올리는 AST 측정은 **0.5**(vLLM 12GB + MADLAD 6.75GB). 예전 문서의 `0.2` 는 GB10
(통합메모리 121GB)용이라 여기서는 4.9GB 밖에 안 돼 모자란다.

```bash
# 1. 데이터 (Jordan 만 — 전체 8방언 9.16GB 받지 말 것)
PYTHONPATH= .venv/bin/hf download UBC-NLP/Casablanca --repo-type dataset \
    --include "Jordan/*" --local-dir ~/datasets/casablanca
PYTHONPATH= .venv/bin/hf download google/fleurs --repo-type dataset \
    --include "data/ar_eg/test.tsv" "data/ar_eg/audio/test.tar.gz" \
    --local-dir ~/datasets/fleurs

# 2. 매니페스트 (.jsonl 은 gitignore 대상 — 매번 재생성한다)
PYTHONPATH= .venv/bin/python evaluation/ast/build_manifest_casablanca.py \
    --casablanca-root ~/datasets/casablanca --dialect Jordan --split test \
    --out evaluation/ast/manifests/casablanca_jordan_test.jsonl

PYTHONPATH= .venv/bin/python evaluation/ast/build_manifest_fleurs.py \
    --fleurs-root ~/datasets/fleurs --src ar_eg --tgt ar_eg --split test \
    --out evaluation/ast/manifests/fleurs_ar_test.jsonl        # WER 용 (ar→ar)

PYTHONPATH= .venv/bin/python evaluation/ast/build_manifest_fleurs.py \
    --fleurs-root ~/datasets/fleurs --src ar_eg --tgt en_us --split test \
    --out evaluation/ast/manifests/fleurs_ar-en_test.jsonl     # AST 지연용 (ar→en)

# 3. WER/CER + 부호 분포 (서버 불필요, 각 17초 / 13초)
for m in casablanca_jordan_test fleurs_ar_test; do
  PYTHONPATH= .venv/bin/python evaluation/ast/asr_eval_arabic.py \
      --manifest evaluation/ast/manifests/$m.jsonl \
      --out evaluation/ast/results/asr_ar/$m.json \
      --batch-size 32 --backend-kwargs '{"gpu_memory_utilization": 0.7}'
done

# 4. AST 지연 (축당 약 4분)
bash evaluation/ast/run_fleurs_ar.sh                    # static@2s + punct
AXES=static CHUNK=8.0 bash evaluation/ast/run_fleurs_ar.sh   # 청크 스윕
```

서버 종료는 반드시 `stop_server.sh` 로 한다 — `pkill` 은 vLLM EngineCore 를 남긴다.
`run_fleurs_ar.sh` 는 내부에서 그렇게 한다.
