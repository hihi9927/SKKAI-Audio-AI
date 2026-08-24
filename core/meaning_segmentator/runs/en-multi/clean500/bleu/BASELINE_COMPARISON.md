# Table 1a 비교군 결과 — clean500 (미열람 FLEURS 500문장)

번역기 gtx(컨텍스트 동일) / 부트스트랩 1000회 / **API 비용 $0** (gtx 무료, 라벨은 로컬 GPU).
지연은 **강제정렬 실측** (Qwen3-ForcedAligner-0.6B). 그림: `tradeoff.png` / `.pdf`.

> **개정 이력 — 이 문서는 세 번 틀렸고 세 번 고쳤다.**
>
> | 판 | 주장 | 왜 틀렸나 |
> |---|---|---|
> | 1판 | "제안이 causal_align 을 de +10.71 / ja +12.40 으로 이긴다" | 비교군을 **고유 입도 점 하나**로만 쟀다. causal_align 이 과분절(de 12.91조각)이라 지연축이 안 맞았다 |
> | 2판 | "순위가 +2.4 BLEU" | `auto` 와 `auto_greedy` 를 **같은 T** 에서 비교했다. 등간격 선택은 정의상 지연을 최소화하므로 같은 T 가 같은 지연이 아니다 |
> | 3판 | 보간으로 지연을 맞춰 +0.4~1.2 | 보간값이라 **신뢰구간이 없다** |
> | **현재** | 아래 §2 | **실측 지연이 80ms 이내로 겹치는 쌍만** 쓴다 |
>
> 세 번 모두 같은 실수였다 — 품질과 지연을 스칼라 하나로 합치려 한 것. 선행연구가 곡선을
> 그리고 등지연 비교를 하는 이유가 이것이다.

## 0. 방법

**모든 정책을 같은 T 격자에 올린다.** 정책은 경계 *위치*를 정하고, 노브 `T` 가 *몇 개*를
남길지 정한다. 예산은 절단기와 동일(`k = max(2, round(어절/T))`)이고, 비교군에는 순위가
없으므로 등간격 이상 위치에 가장 가까운 경계를 고른다(결정론적, 새 경계 생성 없음).

**`auto_greedy` 는 제안의 대조군이다.** `auto_T*` 만 LLM 순위로 *어느* 경계를 남길지 고른다.
이 이점을 뺀 조건(같은 경계, 같은 등간격 규칙)을 함께 내야 "경계 위치가 좋은 건지 순위가
좋은 건지"가 갈린다. 비교군과 맞붙일 상대는 `auto` 가 아니라 `auto_greedy` 다.

**지연은 강제정렬 실측이다.** 어절 종료 시각을 음성에서 직접 재고, 조각 경계 시각을 거기서
읽는다. 독립 정렬기 둘이 일치한다:

| | 어절 종료시각 \|차이\| | 조건 LAAL \|차이\| |
|---|---|---|
| Qwen3-ForcedAligner vs wav2vec2 CTC | 중앙 33.9ms / p90 116ms | **최대 22.3ms** |

정렬 성공률은 양쪽 500/500. **기본값을 Qwen 으로 둔 이유는 정확도가 아니라 일관성이다** —
Table 3/4 가 Qwen3-ASR 로 돌므로 지연축을 같은 자로 재야 하고, 비영어 소스로 확장할 때
영어 전용인 wav2vec2 는 못 쓴다. `--wordtimes {qwen,ctc,interp}` 로 재현 가능.

**구 방식(발화 내 균일속도 보간)은 폐기했다.** 지연을 64~131ms 과소평가했고 **정책마다
편차가 67ms** 였다 — 도입부 묵음 길이가 첫 경계 위치에 따라 다르게 잡히기 때문이다.
(실측 예: 17어절 7.38초 문장에서 첫 어절 종료가 실제 1.80s, 보간 0.43s.)

## 1. 격자 — `laal_ms / BLEU`, T = 4/6/8/12

### en→de (tok 13a)

| 조건 | T4 | T6 | T8 | T12 |
|---|---|---|---|---|
| auto (제안) | 1682/24.2 | 2342/31.7 | 2828/35.2 | 3273/37.9 |
| auto_greedy (순위 제거) | 1645/22.9 | 2148/28.5 | 2627/32.5 | 3046/35.6 |
| syntax (SASST 식) | 1679/**25.4** | 2175/28.9 | 2603/33.6 | 3060/36.3 |
| causal_align (TransLLaMa) | 1791/23.3 | 2260/28.1 | 2698/33.0 | 3115/35.2 |
| alignatt (Papi 2023) | 1614/22.0 | 2131/26.8 | 2590/32.1 | 3037/34.9 |
| mu_prefix (Zhang 2020) | 3516/32.2 | 3803/34.5 | 4060/36.6 | 4293/38.0 |
| punct | 4889/40.0 | 4906/40.2 | 4964/40.7 | 5093/41.1 |

무분절 8848ms / 44.10 · 기계분절 2302ms / 2.59

### en→ja (tok ja-mecab)

| 조건 | T4 | T6 | T8 | T12 |
|---|---|---|---|---|
| auto (제안) | 1518/**20.9** | 2089/**28.0** | 2560/**32.8** | 2980/**35.5** |
| auto_greedy | 1539/20.3 | 1974/26.1 | 2408/30.1 | 2825/33.2 |
| syntax | 1527/19.5 | 1996/26.2 | 2403/29.4 | 2842/33.0 |
| causal_align | 2016/24.1 | 2343/29.2 | 2653/32.2 | 3014/35.3 |
| alignatt | 1574/19.8 | 2034/25.3 | 2473/29.2 | 2871/32.8 |
| mu_prefix | 5691/36.1 | 5778/36.9 | 5852/37.5 | 5917/38.2 |
| punct | 4580/37.7 | 4608/38.0 | 4661/38.5 | 4794/38.8 |

무분절 8801ms / 41.61 · 기계분절 2153ms / 3.19

**타깃 간 절대 BLEU 비교 금지** (토크나이저가 다르다).

## 2. 실측 지연이 80ms 이내로 겹치는 쌍 — 이것이 근거의 전부다

de 19쌍 중 유의 11개, ja 23쌍 중 유의 8개. 주요 항목만 옮긴다 (전체는 재현 명령 참조).

### 순위의 값 — `auto` − `auto_greedy`

| 타깃 | 대조 | 지연차 | ΔBLEU [95% CI] |
|---|---|---|---|
| de | auto_T4 vs auto_greedy_T4 | +37ms | **+1.29** [+0.47,+2.11] |
| ja | auto_T4 vs auto_greedy_T4 | −21ms | +0.53 [−0.15,+1.17] n.s. |

### 경계 위치 — `auto_greedy` vs 비교군 (조건 동일)

| 타깃 | 대조 | 지연차 | ΔBLEU [95% CI] |
|---|---|---|---|
| de | vs syntax (T4) | −34ms | **−2.50** [−3.54,−1.46] |
| de | vs syntax (T8) | +24ms | **−1.09** [−1.94,−0.22] |
| de | vs alignatt (T6) | +17ms | **+1.71** [+0.71,+2.75] |
| de | vs causal_align (T8/T12) | −71/−69ms | −0.49 / +0.30 n.s. |
| ja | vs alignatt (T8) | −65ms | **+0.90** [+0.12,+1.68] |
| ja | vs syntax (T6/T8/T12) | −22/+5/−16ms | −0.16 / +0.69 / +0.08 n.s. |

### 비교군끼리

| 타깃 | 대조 | 지연차 | ΔBLEU [95% CI] |
|---|---|---|---|
| de | syntax vs alignatt (T4) | +65ms | **+3.43** [+2.31,+4.50] |
| de | syntax vs alignatt (T6) | +44ms | **+2.06** [+1.02,+2.96] |
| de | syntax vs causal_align (T12) | −55ms | **+1.01** [+0.32,+1.77] |
| ja | syntax vs causal_align (T6↔T4) | −20ms | **+2.07** [+0.97,+3.15] |

## 3. 읽기

**1. 타깃에 따라 승자가 뒤집힌다.** de 저지연에서는 **SASST 식 구문 경계가 제안을 이긴다**
(T4 에서 −1.20 [−2.31,−0.04], `auto_greedy` 상대로는 −2.50). 반면 ja 에서는 제안이 전 구간
최상단이고 syntax 를 T4 에서 +1.34 로 앞선다. **"제안이 항상 낫다"는 주장은 성립하지 않는다.**

**2. 이 루프의 이득은 순위 쪽이고, de 에서만 검출된다.** `auto` − `auto_greedy` = de +1.29
(유의), ja +0.53 (n.s.). 경계 *위치* 는 de 에서 syntax 에 지고 causal_align 과 동률이다.
즉 논문 주장은 "LLM 이 좋은 자리를 찾는다"가 아니라 **"LLM 이 자리에 우선순위를 매길 수
있다"** 로 좁혀야 한다.

**3. AlignAtt 이 가장 약하다.** 두 타깃 모두 최하단이고, de 에서 syntax 에 −2.06~−3.43 으로
일관되게 진다. 어텐션 argmax 기준은 이 설정(텍스트 MT, NLLB)에서 경쟁력이 없다.

**4. `mu_prefix`·`punct` 는 맞대결이 성립하지 않는다.** 어떤 정책과도 지연이 안 겹친다
(de: mu_prefix 3.5~4.3s, punct 4.9~5.1s, 나머지 1.6~3.3s). 결론은 우열이 아니라 **작동 구간이
다르다**는 것이다. 저지연이 요건이면 둘 다 선택지가 아니다 — punct 는 500문장 중 185문장에
자를 자리가 없고, mu_prefix 는 ja 에서 244/500 이 무분절이다.

**5. 미해결**: 제안 곡선이 de 3273ms 에서 끝나 punct 구간(4.9~5.1s)에 비교점이 없다.
T=16/20/24 를 채워야 갈린다.

## 4. 원논문 대조 — 구현이 논문과 다른 지점

**임의 구성 금지 원칙으로 재작성한 결과다.** 초판 구현에서 세 건이 틀렸다.

| 정책 | 초판의 오류 | 논문 근거 |
|---|---|---|
| causal_align | 공백 분리 사용 | "we split each sentence using the **`word_tokenize` function from the nltk package**, treating punctuation marks as \"words\"" |
| syntax | 파서 `en_core_web_sm` | "we parse source sentences using the **`en_core_web_trf`** model from spaCy" |
| syntax | **구두점 규칙 누락**, 의존관계 15종을 임의 확장 | "boundaries derived from **noun phrases (NP), verb phrases (VP), and prepositional phrases (PP), as well as punctuation and dependency transitions (e.g., nsubj → VERB)** ... maximum span of **seven tokens**" |
| syntax | VP 규칙이 조동사를 분리 (`can \| contain`) | 조동사는 헤드 동사의 동사구에 속한다 |
| alignatt | 층을 1문장 관찰로 선택 | 논문은 6층 중 4층 + 헤드 평균. 우리는 50문장 스윕으로 L5 확정 |

### 논문이 정의를 주지 않아 우리가 정한 것 — **표 각주 필수**

| 정책 | 조작화 |
|---|---|
| causal_align | SimAlign matching 방식(`itermax`), 정렬 모델(기본 mBERT), 비정렬 타깃 단어 처리(carry-forward) |
| causal_align | 원논문은 `<WAIT>` 삽입까지만 하고 **소스 경계를 내놓지 않는다** — `g(j)=max(req(1..j))` 에서 경계를 유도하는 단계는 우리가 얹은 것 |
| syntax | **VP 정의** — spaCy 에 동사구 청커가 없고 논문도 정의를 안 준다. 동사+조동사·부정·불변화사로 잡음 |
| alignatt | 층 번호 — 아키텍처가 달라(논문 Conformer+6층 디코더, 우리 NLLB 12층) 직접 이식 불가 |

### 범위를 축소한 것

- **mu_prefix**: basic method 만. MU++(prefix-attention 단조 NMT 파인튜닝)는 범위 밖.
  논문 Fig.2 가 basic 은 재배열이 심한 쌍에서 붕괴한다고 밝히며, 실측도 그렇다
  (무분절 de 71/500, **ja 244/500**. 무분절 문장이 오히려 **더 길다** — 어절 중앙 23 vs 19)
- **syntax**: 경계 규칙만. 논문은 이 청크로 LLM 을 파인튜닝하지만 우리는 라벨 출처만 비교
- **alignatt**: 프레임 대신 소스 어절 단위 (오디오 스트리밍 아님)
- **causal_align**: **참조 번역이 필요한 오라클성 조건**이다. 제안은 소스만 보고 낸다

### 온라인 실행 가능성 — 시스템으로 존재하는가

| 정책 | 온라인 | 이유 |
|---|---|---|
| alignatt | ✅ | 접두사만으로 동작 |
| syntax / punct | ⚠️ | ASR 구두점 품질에 종속. 정답 텍스트 구두점을 쓰는 지금 수치는 낙관적 |
| causal_align | ❌ | 참조 번역 필요 |
| mu_prefix | ❌ | 전체 문장 번역 필요 |

오프라인 라벨 출처로는 정당하지만 **뒤 둘은 시스템으로 존재하지 않는다.** 각주 필수.

## 5. 재현

```bash
# 라벨 (mu_prefix·punct 는 정정 없음, 나머지는 논문 기준 재작성본)
B="python3 -m core.meaning_segmentator.autoseg.baselines.build --run-id en-multi/clean500"
$B --policy punct
$B --policy causal_align --targets de ja      # nltk word_tokenize
$B --policy alignatt     --targets de ja      # NLLB 교차어텐션 L5
$B --policy mu_prefix    --targets de ja
.venv/bin/python -m core.meaning_segmentator.autoseg.baselines.build \
    --run-id en-multi/clean500 --policy syntax      # en_core_web_trf — 격리 venv 필요

# 어절 타임스탬프 (강제정렬)
python3 -m core.meaning_segmentator.autoseg.baselines.build_wordtimes            # wav2vec2 CTC
.venv/bin/python -m core.meaning_segmentator.autoseg.baselines.build_wordtimes_qwen  # Qwen

# 평가 + 그림
python3 -m core.meaning_segmentator.autoseg.bleu_eval --run-id en-multi/clean500 \
    --targets de ja zh --wordtimes qwen \
    --baselines punct syntax causal_align alignatt mu_prefix
python3 core/meaning_segmentator/autoseg/baselines/plot_tradeoff.py
```

**주의**: `syntax` 는 `spacy-transformers` 가 `transformers<4.50` 을 요구해 메인 venv 의
핀(4.57.6)을 깬다. 격리 venv(`scratchpad/spacyenv`)에서만 돌릴 것.
