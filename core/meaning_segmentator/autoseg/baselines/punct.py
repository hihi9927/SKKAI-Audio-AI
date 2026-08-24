"""문장부호 휴리스틱 — 외부 의존 없음, Table 1a 하한 쪽 기준선."""

from __future__ import annotations

import re

# FLEURS 는 문장 단위라 종결부호는 대개 문장 끝 하나뿐이다. 실질 경계는 문장 **내부**
# 구두점이 만든다. 닫는 따옴표·괄호가 뒤따르면 그것까지 왼쪽 조각에 붙인다.
_BOUNDARY = re.compile(r'(?<=[,;:!?.—–])(?=\s)(?![\'"’”)\]])')


def segment(text: str) -> list[str]:
    parts = [p.strip() for p in _BOUNDARY.split(text)]
    parts = [p for p in parts if p]
    return parts or [text.strip()]
