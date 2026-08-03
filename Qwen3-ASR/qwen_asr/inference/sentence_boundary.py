"""dot-commit 문장 경계 판정 (스트리밍 ASR 텍스트에서 <SEG>급으로 취급 가능한 마침표 탐지).

기존 dot-commit 정규식은 마침표 뒤에 다음 단어가 와야만 매치되도록 룩어헤드
(`(?=\\S)`)를 요구했다 — 아직 안 끝난 hypothesis frontier에 조기 커밋하는 걸
막기 위한 장치였지만, 부작용으로 발화의 마지막 문장(뒤에 더 텍스트가 없음)은
절대 매치되지 않아 dot-commit이 아닌 finish 커밋으로만 나갔다.

이 모듈은 그 룩어헤드를 없애 문자열 끝 마침표도 경계로 인정한다 — 단, 문장
경계가 아닌 마침표(소수점, 알려진 약어)는 기존과 동일하게 계속 제외한다.
"""
import re

# 마침표 뒤에 abbreviation이 오면 문장 경계 아님 (Mr. Smith 같은 경우).
_ABBREV_LOOKBEHIND = r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<!Jr)(?<!Sr)(?<!vs)(?<!No)"

# 종료 문장부호 뒤에 붙는 닫는 따옴표/괄호. 대화체 인용문은 마침표가 인용부호
# 안쪽에 찍혀(`... a shop boy."`) 마침표 바로 뒤가 공백도 문자열 끝도 아니게 된다.
# 이걸 흡수하지 않으면 인용문으로 끝나는 문장은 경계로 인정되지 않아 dot-commit이
# 아예 발동하지 못하고 finish 커밋으로만 빠진다.
_CLOSERS = r"[\"'”’»›\)\]\}]*"

# 소수점(3.14)은 마침표 뒤에 공백이 없어 자연히 제외됨 — 별도 처리 불필요.
DOT_COMMIT_BOUNDARY_RE = re.compile(
    r"(?:"
    rf"{_ABBREV_LOOKBEHIND}\.{_CLOSERS}(?:\s+|$)"
    rf"|[?!]{_CLOSERS}(?:\s+|$)"
    rf"|[。？！]{_CLOSERS}"
    r"|<SEG>"
    r")"
)


def count_dot_commit_boundaries(text: str) -> int:
    """`text` 안에서 dot-commit 트리거로 쓸 수 있는 문장 경계 개수를 센다.

    on_seg의 `txt_p.count("<SEG>")`와 동일한 용도 — 스트리밍 중 텍스트가
    자라날 때마다 이 값이 증가하는 만큼 새 경계가 디코딩된 것으로 취급한다.
    """
    return len(DOT_COMMIT_BOUNDARY_RE.findall(text))
