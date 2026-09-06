"""LLM 호출과 그 비용 — 분절 로직과 무관한 배관.

`gateway` 가 프로바이더별 API 를 한 인터페이스로 덮고 예산·재시도·usage 기록을 하며,
`tracing` 은 호출마다 용도 라벨을 붙이는 LangSmith 연동(선택), `cost_report` 는 런 하나가
실제로 쓴 돈을 집계한다.
"""
