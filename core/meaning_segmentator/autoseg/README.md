# autoseg — 의미 분절 프롬프트 자동 생성 루프

사람이 언어마다 `<SEG>` 삽입 프롬프트를 직접 쓰던 작업을 에이전트 루프로 대체한다.
입력은 평문 문장 데이터와 언어쌍뿐이고, 출력은 **프롬프트**와 그 프롬프트로 분절된 데이터다.

설계 배경과 근거는 [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md) 참조.

## 실행

```bash
# .env 에 CLAUDE_API_KEY (Letsur AI Gateway) 필요
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kokoro --src-lang Japanese --tgt-lang Korean \
    --iterations 4 --train 20 --dev 30 --test 50 --budget 6.0
```

주요 옵션:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--model` | `claude-sonnet-5` | 분절·에이전트 모델 |
| `--translator-model` | `claude-sonnet-5` | 번역 툴 모델. **런 중 변경 금지** — 점수 변화의 원인이 섞인다 |
| `--q-floor` | 자동 | 미지정 시 기계분절 앵커로 캘리브레이션 |
| `--q-floor-ratio` | `0.7` | `Q_floor = Q_bad + ratio·(1−Q_bad)` |
| `--patience` | `3` | dev 무개선 연속 횟수 상한 |
| `--budget` | `5.0` | 게이트웨이 추정 비용 상한. 초과 시 중단 |
| `--fresh` | — | 런 디렉토리를 지우고 처음부터 (캐시도 삭제) |

`--fresh` 없이 같은 `--run-id` 로 다시 실행하면 언어 프로파일·prompt_v0·캘리브레이션·번역 캐시를
재사용해 이어서 돈다.

## 구성

| 파일 | 역할 | LLM |
|---|---|---|
| `gateway.py` | Letsur AI Gateway 클라이언트, 재시도, 비용 집계, 예산 가드 | — |
| `data.py` | A0 Data Preparer — 정규화 + 층화 train/dev/test 분할 | — |
| `pipeline.py` | A2 Segmenter / A3 Format Validator / A4 번역 툴 2종 + 디스크 캐시 | 분절·번역만 |
| `metrics.py` | A5 Scorer — Q(임베딩 코사인·chrF), L(지연 프록시), 목적함수 | — |
| `agents.py` | A1 Profiler / A6 Critic / A7 Prompt Engineer | ● |
| `loop.py` | A8 Loop Controller — 채택·롤백·중단·리포트 | — |

LLM 판단이 들어가는 곳은 `agents.py` 세 곳뿐이다. 포맷 검증, 점수, 채택 판정, 재시도는 전부
결정론적 코드다.

## 목적함수

품질 단일 축으로 최적화하면 최적해가 "`<SEG>`를 하나도 넣지 않는 프롬프트"로 수렴한다
(분절이 없으면 세그 번역 = 전체 번역이라 품질이 자동 만점). 그래서 2축이다.

```
maximize   G = 1 − L                 # 지연 이득
s.t.       V = 1.0                   # 포맷 유효율
           Q ≥ Q_floor               # 품질 하한
```

- `L` = 세그먼트가 나가기까지 통과해야 하는 문장 비율의 평균. `k=1`이면 `1.0`(무분절),
  `k→∞`면 `0.5`. 낮을수록 빠름.
- `Q` = 세그 번역 이어붙임 vs 전체 번역의 의미 유사도. 분절 없는 문장은 제외하지 않고 `1.0`.
- `Q_floor` = 의미를 무시한 기계적 분절(8자마다 절단)의 Q를 실측해 상대 기준으로 산출.

## 산출물

```
runs/{src}-{tgt}/{run_id}/
  config.json  data/{train,dev,test}.json  language_profile.json  baseline.json
  iter_NN/{prompt.txt, train_rows.json, dev_rows.json, violations.json,
           metrics.json, critique.json, changelog.json}
  history.json  best_prompt.txt  test_rows.json  final_report.md  cache/
```

## 새 언어 추가

`data.py` 의 `LOADERS` 에 로더 하나만 추가하면 된다. 프롬프트는 손대지 않는다 —
Language Profiler 가 샘플 문장에서 어순·표기·절 경계 표지를 뽑아 `prompt_v0` 를 만든다.

```python
LOADERS = {
    "kokoro": load_kokoro,
    "my_data": lambda: load_json_entries(Path("..."), text_field="text"),
}
```

## 품질 백엔드 교체

기본은 게이트웨이 임베딩 코사인 유사도다 (로컬 ML 의존성 없음). `unbabel-comet` 이 설치되어
있으면 `metrics.py` 에 COMET 백엔드를 연결해 운영 기준 지표로 바꿀 수 있다. `Q_floor` 는
캘리브레이션으로 자동 재산출되므로 백엔드를 바꿔도 상수를 손댈 필요는 없다.
