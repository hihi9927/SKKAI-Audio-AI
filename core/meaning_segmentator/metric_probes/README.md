# metric_probes — `contradiction` 지표 백엔드 탐침

`autoseg/` 의 목적함수는 `effective = adequacy × (1 − contradiction)` 이고, `contradiction`
자리는 NLI(`microsoft/deberta-large-mnli`)가 맡고 있다. 이 폴더는 **그 자리를 무엇으로
대신할 수 있는지** 를 잰 실험들이다.

- 결론과 근거: [../NLI_ALTERNATIVES.md](../NLI_ALTERNATIVES.md)
- 루프 본체: [../autoseg/](../autoseg/)

**여기 있는 코드는 어느 것도 루프 경로에 들어가지 않는다.** 기존 런(`../runs/`)을
읽어 재활용할 뿐이고, 산출물은 전부 `runs/` 아래로만 나간다. `autoseg/` 를 실험 코드로
어지럽히지 않으려고 분리했다.

## 실행

```bash
# 의존성은 ../requirements.txt (+ sentence-transformers, datasets)
PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.<모듈> [옵션]

# 대부분 --render-only 를 지원한다 — 기존 scores.json 으로 리포트만 다시 만든다 (GPU 0)
PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.contra_alt --render-only
```

## 구성

측정 대상별로 묶었다. 오른쪽은 산출물 디렉토리(`runs/` 이하).

### 대체 후보를 관문에 태우는 것

| 모듈 | 무엇을 재나 | 산출물 |
|---|---|---|
| `embed_check.py` | 임베딩 코사인으로 NLI 를 대신할 수 있나. MTEB 상위 4종 × 관문 2종 + 잡음 바닥 | `embed_vs_nli/` |
| `contra_alt.py` | **(premise 축 `oracle`/`retrans`) × (scorer 축 `nli`/`summac`/`minicheck`/`erasure`)** | `contra_alt/` |

### 임베딩이 무엇을 담고 있는지 파고드는 것

| 모듈 | 무엇을 재나 | 산출물 |
|---|---|---|
| `embed_probe.py` | 모순 정보가 벡터에 남아 있나 — 얼린 인코더 + MNLI 프로브, 교차 인코더 천장 대조 | `embed_probe/` |
| `embed_geometry.py` | 여러 차원을 **비지도**로 읽으면 되나 — 화이트닝·마할라노비스·주성분 | `embed_geometry/` |
| `semantic_axis.py` | 대조쌍으로 **의미 축**(극성)을 찾을 수 있나. 규칙 생성이라 라벨 0 | `semantic_axis/` |
| `multi_axis.py` | 유형별 축 5종 + 대조군. 축을 늘리면 나아지나 | `multi_axis/` |
| `minimal_pairs.py` | **통제된 최소쌍 진단** — 무엇을 잡고 무엇을 못 잡나. 길이 맞춘 조각 대조 포함 | `minimal_pairs/` |

### 문제 정의 자체를 바꿔 보는 것

| 모듈 | 무엇을 재나 | 산출물 |
|---|---|---|
| `fixed_point.py` | **번역 고정점** — 뒤가 와도 앞 번역이 안 바뀌는 지점 (Meaningful Unit 의 의미판) | `fixed_point/` |
| `future_dep.py` | 미래 소스 의존도 — 오라클 없이 소스 쪽에서만 재는 절단 위험 | `future_dep/` |
| `boundary_probe.py` | 어절 점진 추가 시 표현 급변점이 분절 경계인가 (**분절기** 후보) | `boundary_probe/` |

`paths.py` 가 세 경로를 고정한다: `AUTOSEG`(관문 픽스처), `SEG_RUNS`(루프 산출물, 읽기
전용), `OUT_RUNS`(이 폴더의 산출물).

## 이 폴더의 실험이 지킨 규칙

세션 내내 같은 함정에 반복해서 걸렸으므로 규칙으로 남긴다.

1. **기준선은 무작위가 아니라 위치 사전확률이다.** `boundary_probe` 에서 문장 내용을
   전혀 안 보는 위치 사전확률(AUC 0.864)이 실제 점수(0.815)를 이겼다. 경계 관련 지표는
   위치 교란을 통제하기 전에는 부호조차 믿을 수 없다 — `fixed_point` 에서도 raw 0.529 가
   잔차 보정 후 0.447 로 방향이 뒤집혔다.
2. **잡음 바닥부터 잰다.** raw 상관 0.39 가 바닥 보정 후 0.10 으로 무너진 사례가 있다
   (`embed_vs_nli`). 바닥은 full 번역의 자기-prefix — 정의상 무해한 미완성이다.
3. **대조군을 같이 만든다.** 이름 붙인 축이 이름 때문에 작동하는지 보려면 이름 없는 축
   (`content_swap`)이 필요하다.
4. **지도 상한을 공정하게 건다.** 목표가 `‖d‖` 에 관한 양이면 원시 `d` 에 대한 선형
   회귀로는 표현할 수 없다 — `|d|`·`u∘v` 같은 원소별 크기 특징을 넣어야 상한을 과소평가
   하지 않는다 (`embed_geometry` 에서 실제로 한 번 틀렸다).
