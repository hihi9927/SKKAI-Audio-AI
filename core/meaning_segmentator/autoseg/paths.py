"""저장소 안에서의 고정 위치. **파일 깊이로 경로를 세지 않기 위해 있다.**

종전에는 19개 파일이 각자 `Path(__file__).resolve().parents[N]` 으로 저장소 루트를
구했고 N 이 파일마다 2·3·4 로 달랐다. 파일을 한 폴더 아래로 옮기면 전부 틀어지는데,
그게 `ImportError` 로 드러나지 않는 것이 문제다 — 런 산출물이 엉뚱한 데 쌓이거나
매니페스트를 못 찾는 식으로 조용히 어긋난다. 여기 한 곳만 맞으면 나머지는 따라온다.
"""

from __future__ import annotations

from pathlib import Path

# 이 파일은 `<repo>/core/meaning_segmentator/autoseg/paths.py` 다.
PKG_DIR = Path(__file__).resolve().parent                 # .../autoseg
SEGMENTATOR_DIR = PKG_DIR.parent                          # .../core/meaning_segmentator
REPO_ROOT = PKG_DIR.parents[2]                            # .../<repo>

# 루프·평가 산출물이 쌓이는 곳. autoseg 밖(형제 디렉토리)이다.
RUNS_DIR = SEGMENTATOR_DIR / "runs"

# 평가용 매니페스트 (FLEURS n-way, CoVoST2).
MANIFEST_DIR = REPO_ROOT / "evaluation" / "ast" / "manifests"
