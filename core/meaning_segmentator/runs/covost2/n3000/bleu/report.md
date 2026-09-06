# 분절 조건별 참조 기반 BLEU — covost2/n3000

- 분절: `auto_run13_mg1` / 번역기 `local:google/madlad400-3b-mt:de:ctx=False / local:google/madlad400-3b-mt:ja:ctx=False / local:google/madlad400-3b-mt:zh:ctx=False` / 부트스트랩 1000회
- **언어 간 절대 BLEU 비교 금지** — 토크나이저가 다르다. 언어를 가로지르는 판독은 `retention`(자기 무분절 대비)과 `chrF2` 로 한다.
- 소스 en → 타깃 de·ja·zh. 이 타깃들은 프롬프트 최적화 시 목적함수에 포함됐던 언어다 (미출현 타깃 아님).
- `k`(조각 수)는 자동·기계 분절에서는 세 타깃이 **같은 분절**을 쓰므로 동일하다. 단 `causal_align`·`mu_prefix` 는 타깃별 자원(정렬 대상·NMT)에 의존하므로 타깃마다 다르다. `laal_words` 는 정의상 타깃 길이가 들어가고 ja·zh 는 타깃 단위가 문자라 값이 커진다 — **타깃 간 laal 비교 금지**, 같은 타깃 안에서 T 방향만 읽을 것.
- `laal_ms` 는 **강제정렬 실측**이다 (`--wordtimes qwen`). 독립 정렬기 둘(wav2vec2 CTC / Qwen3-ForcedAligner)이 조건 수준 LAAL 에서 22ms 이내로 일치한다. 구 방식(발화 내 균일속도 보간)은 지연을 64~131ms 과소평가했고 정책마다 편차가 있었다.
- `*_T*` 비교군 점은 정책의 경계 **부분집합**이다 (좌→우 탐욕). 제안 `auto_T*` 만 LLM 순위로 남길 경계를 고르므로, 순위 이득을 뺀 대조는 `auto_greedy_T*` 다.

## en→de (n=3000, tok:13a, 번역기 `local:google/madlad400-3b-mt:de:ctx=False`)

| 조건 | k | laal_ms ↓ | laal_words ↓ | BLEU ↑ | chrF2 | retention(BLEU) | Δ vs unseg [95% CI] |
|---|---|---|---|---|---|---|---|
| unsegmented | 1.00 | 2651 | 9.06 | 39.87 | 64.64 | 1.0000 | — |
| alignatt_f4 | 3.94 | 1122 | 3.01 | 18.76 | 51.15 | 0.4706 | -21.10 [-22.04, -20.19] |
| alignatt_f6 | 2.82 | 1587 | 4.43 | 26.13 | 55.84 | 0.6554 | -13.73 [-14.56, -12.96] |
| alignatt_f8 | 1.98 | 2016 | 5.93 | 31.38 | 59.45 | 0.7871 | -8.47 [-9.20, -7.78] |

## en→ja (n=3000, tok:ja-mecab, 번역기 `local:google/madlad400-3b-mt:ja:ctx=False`)

| 조건 | k | laal_ms ↓ | laal_words ↓ | BLEU ↑ | chrF2 | retention(BLEU) | Δ vs unseg [95% CI] |
|---|---|---|---|---|---|---|---|
| unsegmented | 1.00 | 2744 | 9.06 | 25.42 | 33.65 | 1.0000 | — |
| alignatt_f4 | 2.93 | 1483 | 5.07 | 13.49 | 23.36 | 0.5306 | -11.94 [-12.62, -11.24] |
| alignatt_f6 | 2.13 | 1919 | 6.27 | 17.56 | 26.45 | 0.6908 | -7.87 [-8.50, -7.25] |
| alignatt_f8 | 1.59 | 2287 | 7.35 | 20.87 | 29.50 | 0.8210 | -4.56 [-5.02, -4.07] |

## en→zh (n=3000, tok:zh, 번역기 `local:google/madlad400-3b-mt:zh:ctx=False`)

| 조건 | k | laal_ms ↓ | laal_words ↓ | BLEU ↑ | chrF2 | retention(BLEU) | Δ vs unseg [95% CI] |
|---|---|---|---|---|---|---|---|
| unsegmented | 1.00 | 2629 | 9.06 | 40.36 | 33.54 | 1.0000 | — |
| alignatt_f4 | 3.56 | 1245 | 4.38 | 21.40 | 20.86 | 0.5303 | -18.93 [-19.82, -18.09] |
| alignatt_f6 | 2.52 | 1683 | 5.65 | 27.34 | 24.81 | 0.6776 | -13.00 [-13.81, -12.23] |
| alignatt_f8 | 1.82 | 2074 | 6.87 | 31.79 | 27.98 | 0.7876 | -8.56 [-9.35, -7.81] |

## 언어 간 안정성 (retention = 조건 / 무분절)

| 조건 | de BLEU | ja BLEU | zh BLEU | de chrF2 | ja chrF2 | zh chrF2 | BLEU 폭 | chrF2 폭 |
|---|---|---|---|---|---|---|---|---|
| alignatt_f4 | 0.4706 | 0.5306 | 0.5303 | 0.7913 | 0.6943 | 0.6219 | 0.0600 | 0.1694 |
| alignatt_f6 | 0.6554 | 0.6908 | 0.6776 | 0.8639 | 0.7860 | 0.7397 | 0.0354 | 0.1242 |
| alignatt_f8 | 0.7871 | 0.8210 | 0.7876 | 0.9198 | 0.8769 | 0.8342 | 0.0339 | 0.0856 |
