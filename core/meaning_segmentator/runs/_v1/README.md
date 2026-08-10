# runs/_v1 — v1 지표로 측정된 런

`gain`(= Average Proportion) · `Q_floor` · `LCB` · 달성률로 채점된 결과다.
**v2 수치와 같은 표에 넣을 수 없다** — 축이 다르다:

| v1 | v2 | 왜 비교 불가 |
|---|---|---|
| `gain` | `laal_words` | AP vs LAAL. 단위도 방향도 다름 |
| `Q` (참조 = full 번역) | `adequacy` (참조 없음) | v1 은 offline 어순 편향을 포함 |
| `objective` | `score` | 가중치·임계값 유무 |

지표 정의는 [../../docs_v1/METRICS.md](../../docs_v1/METRICS.md).

`_v1` 아래에 두는 이유는 `.gitignore` 의 `runs/**/cache/`·`runs/**/*_rows.json` 패턴이
그대로 먹히게 하기 위해서다. 경로를 `runs/` 밖으로 빼면 무시되던 대용량 산출물이
추적 대상이 된다.

읽을 가치가 있는 것:

| 런 | 조건 |
|---|---|
| `ko-en/run03-google-fix/` | Google 번역기 + COMET. 사람 프롬프트 3종 비교 포함 |
| `ko-en/run09-t60/` ~ `run12/` | 달성률 목적함수 실험. direction 고착 사례 |
| `ja-ko/ja-ko-test04/` | `language_profile.json` 의 `trailing_punctuation` 이 null 로 나온 실패 사례 |
