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
CLOSERS = r"[\"'”’»›\)\]\}]*"

# 소수점(3.14)은 마침표 뒤에 공백이 없어 자연히 제외됨 — 별도 처리 불필요.
#
# \u061f = 아랍어 물음표, \u06d4 = 우르두식 마침표. 아랍어는 CJK 와 달리 부호 뒤에
# 공백을 쓰므로 `[。？！]` 분기가 아니라 공백/문자열끝을 요구하는 라틴 분기에 넣는다.
# 아랍어 마침표는 어느 방언이든 ASCII `.` 이라 별도 추가가 필요 없다.
DOT_COMMIT_BOUNDARY_RE = re.compile(
    r"(?:"
    rf"{_ABBREV_LOOKBEHIND}\.{CLOSERS}(?:\s+|$)"
    rf"|[?!\u061f\u06d4]{CLOSERS}(?:\s+|$)"
    rf"|[。？！]{CLOSERS}"
    r"|<SEG>"
    r")"
)

# 종료 문장부호 기준 단순 분할 — 약어를 구분하지 않으므로 커밋 경계 판정에는 쓰지
# 말고(그건 DOT_COMMIT_BOUNDARY_RE), "텍스트를 문장 단위 조각으로 훑는" 용도로만 쓴다.
# 닫는 따옴표/괄호는 앞 문장에 붙여 흡수한다. 흡수하지 않으면 인용문으로 끝나는
# 문장에서 닫는 따옴표가 독립 조각으로 떨어져 나온다(`... own men!"` →
# `... own men!` + `"`). 앞 조각이 dedup 등으로 폐기되면 그 따옴표만 살아남아
# 한 글자짜리 커밋이 나갔다 (실측: dev-clean 5694-64029-0020이 finish 커밋).
SENTENCE_SPLIT_RE = re.compile(
    rf"[^.!?。！？\u061f\u06d4]*[.!?。！？\u061f\u06d4]+{CLOSERS}\s*"
    rf"|[^.!?。！？\u061f\u06d4]+$"
)


def split_sentences(text: str) -> list[str]:
    """`text`를 종료 문장부호 기준으로 쪼갠다. 조각을 이어붙이면 원문이 복원된다."""
    return SENTENCE_SPLIT_RE.findall(text)


def count_dot_commit_boundaries(text: str) -> int:
    """`text` 안에서 dot-commit 트리거로 쓸 수 있는 문장 경계 개수를 센다.

    on_seg의 `txt_p.count("<SEG>")`와 동일한 용도 — 스트리밍 중 텍스트가
    자라날 때마다 이 값이 증가하는 만큼 새 경계가 디코딩된 것으로 취급한다.
    """
    return len(DOT_COMMIT_BOUNDARY_RE.findall(text))
