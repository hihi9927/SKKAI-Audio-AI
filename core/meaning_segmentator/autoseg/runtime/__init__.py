"""루프가 매 반복 쓰는 것 — 분절 요청, 채점, 데이터.

`agents` 는 LLM 이 판단하는 네 자리(Profiler/Judge/Critic/Prompt Engineer)이고,
나머지 셋은 전부 결정론적 코드다. `pipeline` 이 태그 파싱·정규화·검증·절단·번역을,
`metrics` 가 채점 백엔드와 지표를, `data` 가 데이터셋 로딩과 언어 프로파일 측정을 맡는다.
"""
