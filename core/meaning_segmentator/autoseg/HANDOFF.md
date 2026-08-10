# autoseg 핸드오프 — 2026-08-10

## 현재 상태

- 지표 체계 v2.1 구현·검증 완료, 커밋 `80a27af` (`feat/autoseg-prompt-loop`) 푸시됨
- **run04 는 이 머신에서 GPU 공유 문제(OOM)로 중단** — 다른 환경에서 재개 대기
- run01~03 의 effective/contradiction/consistency 수치는 구 집계라 **새 런과 비교 불가**

## run04 재개 (다른 환경에서)

```bash
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon-train --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id run04 --translator google \
    --iterations 6 --train 60 --train-pool 120 --dev 150 --test 150 --budget 60
```

환경 요건:
- `unbabel-comet`(+`setuptools<81`), `transformers`, GPU ~7GB
- CometKiwi 는 HF 게이트 모델 — 라이선스 동의 + `hf auth login`
- `.env` 에 `CLAUDE_API_KEY` (Letsur AI Gateway)
- 원래 머신에서는 `.venv/bin/python` (기본 python 에 comet 없음)

`runs/ko-en/run04/` 시드(분할·prompt_v0)는 커밋 안 됨 — 없으면 `--fresh` 로 새로 생성
(분할은 시드 고정이라 어차피 동일).

## 오늘(v2.1) 바뀐 것 — 요지

| 변경 | 왜 |
|---|---|
| contradiction 문장 값 = **경계 (k−1)개 평균**, 무분절은 미정의 | 구 집계(조각 가중)는 k 클수록 기계적으로 상승 — run03 재집계에서 effective 기울기 전체가 artifact 로 판명 |
| consistency = **양방향 NLI** (COMET 폐기) | 어순 편향 제거. **논문 주 곡선 y축으로 승격** — run03 에서 유일하게 기울기 살아있는 축 (0.55→0.76) |
| 잡음 바닥 측정 (`noise_floor.py`) | NLI base rate 가 길이 의존 (1-2어절 0.113 / 10어절+ 0.003). 순위 역전 Spearman −0.25 가 보정 후 **+0.14 로 반전** — 역전은 위치 교란이었음 |
| adequacy 조각 관문 (`adequacy_check.py`) | 부정/의미/무관 정상 검출. 관용구 1케이스에서 source_echo 복사 편향 확인. **케이스 문안 사람 확정 필요** |
| 채택 요건 `dev 쌍체 Δ > 1·se` | run03 은 점 비교라 오차막대 안 잡음까지 채택 후보 (iter0 고착) |
| test premature_rate 무작위 표본 / reference_suspect_rate / 순위정렬 Spearman 자동 산출 | 편향 제거 + 오라클 오염 감시 + [Priority Rules] 조향 근거 추적 |
| prompt_v0 max_tokens 16k + 골격 재검사 3회 | thinking 잘림으로 꼬리 섹션 누락 실제 발생 |
| NLI pipeline 전역 공유, judge Gateway 비용 합산, gtx 줄 불일치 카운터, N/3 지침 파라미터화, kspon-train 로더(4,884문장) | 잔결함 정리 |

## run04 에서 볼 것

1. **채택이 iter0 이후에도 일어나는가** (새 임계 + dev 150 검출력)
2. **순위정렬 Spearman 추이** — 이터레이션마다 로그에 찍힘. 오르면 [Priority Rules] 조향이 작동한다는 뜻
3. `reference_suspect_rate` — 높으면 Google 오라클 의심
4. 완료 후 비교군: 
   ```bash
   PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
       --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
       --run-id ko-en/run04 --split test --label human_current --no-priority
   ```
   → 곡선(우리 4점 + 무분절 상한선 + 기계 + 사람) 완성

## 미해결 (우선순위순)

1. **gold 참조 교차검증** — consistency 가 주축이 된 지금 논문 방어 급소. AIHub 한-영 신청 필요 (사람)
2. `adequacy_cases.json` 문안 사람 확정 (10분)
3. 잡음 바닥 c₀ 를 목적함수에서 차감할지 — run04 Spearman 추이 보고 결정
4. use_context=False 재채점 (평가 vs 운영 서버 번역 모드 불일치, 설계 §12.5)
5. ms-LAAL — KsponSpeech 오디오 강제정렬 (설계 §13-3)
6. 비영어 타깃 consistency — NLI 음역 맹점 (설계 §13-8)

## 문서 위치

설계: [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md) (v2.1 반영) /
사용법: [README.md](README.md) / 관문 결과: `../runs/validity_nli/`, `../runs/adequacy_validity/`,
`../runs/ko-en/run03/noise_floor_test.json`
