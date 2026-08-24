"""경로 상수 — 실험 코드가 `autoseg/` 와 루프 산출물을 어디서 찾는지 한 곳에 모은다."""

from pathlib import Path

HERE = Path(__file__).resolve().parent          # metric_probes/
AUTOSEG = HERE.parent / "autoseg"               # 관문 픽스처(`*_cases.json`) 위치
SEG_RUNS = HERE.parent / "runs"                 # 루프 산출물 — **읽기 전용**으로 쓴다
OUT_RUNS = HERE / "runs"                        # 이 폴더 실험의 산출물
