# docs_v1 — 폐기된 v1 설계·지표 문서

현행 설계는 [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md) 다.
여기 있는 두 문서는 **더 이상 코드와 일치하지 않는다.** 남겨 둔 이유는 두 가지다.

1. `runs/ko-en/run01`~`run12`, `runs/ja-ko/*` 는 v1 지표(`Q`·`L`·`gain`·`Q_floor`·달성률)로
   측정됐다. 그 수치를 읽으려면 v1 정의가 필요하다.
2. v2 가 무엇을 왜 버렸는지의 근거가 여기 있다 — 특히 `METRICS.md` 의 실측표
   (지연 프록시의 닫힌 형태, 앵커 캘리브레이션, 백엔드 타당도)는 v2 설계의 논증에
   그대로 인용된다.

| 파일 | 대체된 곳 |
|---|---|
| `AUTO_PROMPT_LOOP_DESIGN.md` | `../AUTO_PROMPT_LOOP_DESIGN.md` |
| `METRICS.md` | `../AUTO_PROMPT_LOOP_DESIGN.md` §5 (지표) + `../autoseg/README.md` |

**v1 수치와 v2 수치를 같은 표에 넣지 말 것.** `gain` 은 Average Proportion, `laal_words` 는
LAAL 이고 축이 다르다. `Q` 는 full 번역을 참조로 쓰지만 `adequacy` 는 참조가 없다.
