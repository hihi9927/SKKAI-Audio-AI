# Table 1a 비교군 — 외부 라벨 출처

제안 루프(멀티에이전트)와 붙일 **타 정책** 구현. 모든 정책이 소스 조각 리스트만 내놓고,
번역·BLEU·chrF2·쌍체 부트스트랩·지연은 `bleu_eval` 이 그대로 쓴다.

**결과와 판독은 [`runs/en-multi/clean500/bleu/BASELINE_COMPARISON.md`](../../runs/en-multi/clean500/bleu/BASELINE_COMPARISON.md) 에 있다.**

| 정책 | 출처 | 경계 기준 | 외부 의존 | 타깃별 | 온라인 |
|---|---|---|---|---|---|
| `punct` | — | 문장 내부 구두점 | 없음 | 아니오 | ⚠️ |
| `syntax` | SASST (Yang+ 2026) | NP·VP·PP 끝 + 구두점 + `nsubj→VERB`, 최대 7토큰 | spaCy `en_core_web_trf` | 아니오 | ⚠️ |
| `causal_align` | TransLLaMa (Koshkin+ 2024) | `g(j)=max(req(1..j))` 인과 스케줄 | SimAlign + **참조 번역** | 예 | ❌ |
| `alignatt` | AlignAtt (Papi+ 2023) | 교차어텐션 argmax 가 최근 `f` 어절 밖 | NLLB 어텐션 (층 5) | 예 | ✅ |
| `mu_prefix` | Zhang+ 2020 | prefix 번역이 full 번역의 접두사 | NLLB 강제 디코딩 | 예 | ❌ |

`⚠️` = ASR 구두점 품질에 종속. `❌` = 미래 정보가 필요해 **시스템으로 존재하지 않는다**
(오프라인 라벨 출처로만 정당).

## 실행

```bash
B="python3 -m core.meaning_segmentator.autoseg.baselines.build --run-id en-multi/clean500"
$B --policy punct
$B --policy causal_align --targets de ja
$B --policy alignatt     --targets de ja
$B --policy mu_prefix    --targets de ja
.venv/bin/python -m core.meaning_segmentator.autoseg.baselines.build \
    --run-id en-multi/clean500 --policy syntax        # ← 격리 venv (아래 주의)

python3 -m core.meaning_segmentator.autoseg.bleu_eval --run-id en-multi/clean500 \
    --targets de ja zh --baselines punct syntax causal_align alignatt mu_prefix
python3 core/meaning_segmentator/autoseg/baselines/plot_tradeoff.py
```

산출: `runs/<run-id>/baselines/<policy>_<tgt>_<split>.json`

## 파일

| 파일 | 역할 |
|---|---|
| `punct.py` / `syntax_sasst.py` | 구두점 / 구문 경계 (소스만 보므로 타깃 독립) |
| `causal_align.py` | SimAlign 정렬 → 인과 스케줄 → 소스 경계 |
| `alignatt.py` | 교차어텐션 방출 스케줄 → 소스 경계 |
| `mu_prefix.py` | Zhang 2020 Algorithm 1 |
| `nmt.py` | NLLB-600M — 강제 디코딩 · beam 후보 · 교차어텐션 |
| `__init__.py` | `coarsen()` — 정책 경계의 부분집합으로 T 격자에 올린다 |
| `build.py` | 라벨 생성 CLI |
| `align_audio.py` / `build_wordtimes.py` | wav2vec2 CTC 강제정렬 (교차검증용) |
| `build_wordtimes_qwen.py` | Qwen3-ForcedAligner (기본) |
| `plot_tradeoff.py` | 품질–지연 곡선 |

## 원논문과 다른 점 — 표 각주로 옮길 것

**임의로 구성하지 않는다.** 초판에서 파서·토크나이저·규칙 집합을 추측으로 채웠다가 원문
대조에서 세 건이 틀린 것으로 드러났다. 지금은 전부 원문 인용에 근거한다.

### 논문이 정의를 안 준 부분 (우리 조작화)

- **`causal_align`** — SimAlign matching 방식(`itermax`), 정렬 모델, 비정렬 타깃 단어 처리.
  그리고 원논문은 `<WAIT>` 삽입까지만 하고 **소스 경계를 내놓지 않는다** — 경계 유도는 우리가 얹었다.
- **`syntax`** — **VP 정의**. spaCy 에 동사구 청커가 없다. 동사 subtree 를 쓰면 목적어까지
  삼키므로 동사+조동사·부정·불변화사로 잡았다.
- **`alignatt`** — 층 번호. 논문은 6층 중 4층이지만 NLLB 는 12층이라 직접 이식이 안 된다.
  50문장 스윕(argmax 정렬 단조성)으로 L5 확정. 층 간 차이는 작다(de 0.72~0.84).

### 범위 축소

- **`mu_prefix`** — basic method 만. MU++(단조 NMT 파인튜닝)는 범위 밖. 논문 Fig.2 대로
  재배열이 심한 쌍에서 붕괴한다 (무분절 de 71/500, **ja 244/500**)
- **`syntax`** — 경계 규칙만. 논문은 이 청크로 LLM 을 파인튜닝한다
- **`alignatt`** — 프레임 대신 소스 어절 (오디오 스트리밍 아님)
- **`causal_align`** — **참조 번역을 쥐는 오라클성 조건**

## 지연 측정

**강제정렬 실측이다.** 어절 종료 시각을 음성에서 재고 조각 경계 시각을 거기서 읽는다.
기본 정렬기는 Qwen3-ForcedAligner-0.6B — 정확도가 아니라 **일관성** 때문이다 (Table 3/4 가
Qwen3-ASR 로 돌고, 비영어 소스로 확장할 때 영어 전용 wav2vec2 는 못 쓴다).

wav2vec2 CTC 와 교차검증했고 **조건 LAAL 이 최대 22.3ms 차이**로 일치한다(어절 중앙 33.9ms).
`--wordtimes {qwen,ctc,interp}` 로 전환 가능. `interp`(발화 내 균일속도 보간)는 폐기됐다 —
지연을 64~131ms 과소평가했고 정책마다 편차가 67ms 였다.

## ⚠️ 환경 주의

`syntax` 가 쓰는 `en_core_web_trf` 는 `spacy-transformers` 를 요구하고, 그것이
**`transformers<4.50` 을 강제해 레포 핀(4.57.6)을 깬다.** 한 번 이 사고가 있었다.
반드시 격리 venv 에서만 돌릴 것 — `scratchpad/spacyenv` (system-site-packages 로 생성).

`bleu_eval` 은 `metrics_ast._bleu_metric` 의 `lru_cache` 에 의존한다. 이걸 되돌리면
`ja-mecab` 이 호출마다 MeCab Tagger 를 만들어 ipadic 4파일을 mmap 하고, 1.6만 회에서
`vm.max_map_count`(65530)를 넘겨 **스레드 생성 실패 → executor 무한 대기**로 죽는다.
