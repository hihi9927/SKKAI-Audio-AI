"""LLM 출력의 표기 흔들림을 결정론적으로 되돌린다 — 경계 위치는 유지한다.

`core/meaning_segmentator/utils/gpt_seg_en.py` 의 앵커 복원과 같은 방식이다. 로컬
모델은 원문의 CSV 인용부호를 제 손으로 벗기는 일이 잦은데(CoVoST2 매니페스트에
`\"...\"` 가 그대로 들어 있다), 그러면 `text_modified` 로 전량 폐기된다. 경계의 왼쪽
마지막 1~2어절을 앵커로 원문에서 되찾아 태그만 원문 위에 다시 꽂는다.
"""
from __future__ import annotations

import re

SEG_TAG_RE = re.compile(r"<SEG:(\d+)>")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_seg(s: str) -> str:
    return _norm(SEG_TAG_RE.sub(" ", s))


def is_exact(original: str, output: str) -> bool:
    return _strip_seg(output) == _norm(original)


def _split_seg(seg_text: str) -> tuple[list[str], list[int]]:
    parts = SEG_TAG_RE.split(seg_text)
    return [x.strip() for x in parts[0::2]], [int(n) for n in parts[1::2]]


def renumber(seg_text: str) -> str:
    """랭크를 상대 순서만 유지한 채 1..k 로 재부여 (누락·중복 복구)."""
    _, nums = _split_seg(seg_text)
    if not nums or sorted(nums) == list(range(1, len(nums) + 1)):
        return seg_text
    order = sorted(range(len(nums)), key=lambda i: (nums[i], i))
    new = [0] * len(nums)
    for rank, i in enumerate(order, start=1):
        new[i] = rank
    it = iter(new)
    return SEG_TAG_RE.sub(lambda _m: f" <SEG:{next(it)}> ", seg_text).strip()


def clean_output(text: str, result: str) -> str:
    """코드펜스·`Output:` 라벨·설명 줄을 걷어내고 태그된 한 줄만 남긴다."""
    result = re.sub(r"^```[a-zA-Z]*\n|```$", "", result.strip()).strip()
    result = re.sub(r"^(Output|출력)\s*:\s*", "", result, flags=re.IGNORECASE).strip()
    lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
    if len(lines) > 1:
        tagged = [ln for ln in lines if "<SEG" in ln]
        return tagged[0] if tagged else lines[0]
    return result if lines else text


def recover_positions(original: str, modified: str) -> str | None:
    segments, nums = _split_seg(modified)
    if len(segments) <= 1 or len(nums) != len(segments) - 1:
        return None
    inserts: list[tuple[int, int]] = []
    search_start = 0
    for seg, num in zip(segments[:-1], nums):
        words = seg.split()
        if not words:
            return None
        m = None
        for n_anchor in (2, 1):
            if len(words) < n_anchor:
                continue
            anchors = [re.sub(r"""[?!.,;:'"]+$""", "", w) for w in words[-n_anchor:]]
            if not all(anchors):
                continue
            pat = re.compile(r"\s+".join(re.escape(a) for a in anchors) + r"""[?!.,;:'"]*""")
            m = pat.search(original, search_start)
            if m:
                break
        if m is None:
            return None
        inserts.append((m.end(), num))
        search_start = m.end()
    out = original
    for pos, num in sorted(inserts, reverse=True):
        out = out[:pos] + f" <SEG:{num}>" + out[pos:]
    return renumber(out) if is_exact(original, out) else None


def repair(original: str, raw: str) -> str:
    """(정리된 seg_text). 복원 불가면 원문 그대로 — 무분절로 떨어진다."""
    out = renumber(clean_output(original, raw))
    if is_exact(original, out):
        return out
    return recover_positions(original, out) or original
