"""Zhang et al. 2020 (EMNLP) Algorithm 1 — 접두사 매칭 Meaning Unit.

    prefix x≤t 의 번역 M(x≤t) 가 전체 문장 번역 ỹ 의 **접두사이면** t 를 MU 경계로 삼는다.
    확정된 MU 의 번역은 고정이므로 다음 판정은 그것을 tgt_force 로 깔고 이어 디코딩한다.
    엄격한 접두사 조건은 논문대로 ỹ 를 beam top-N(=10) 후보 집합으로 넓혀 완화한다.

원논문은 이 산출을 BERT 분류기 **학습 데이터**로 쓰지만, Table 1a 는 오프라인
"라벨 출처" 비교이므로 Algorithm 1 자체를 라벨러로 쓴다 — 분류기 학습이 필요 없다.

**구현 범위: basic method 만.** 논문의 MU++(refined)는 prefix-attention 으로 단조 NMT 를
수백만 문장에 파인튜닝해야 해서 범위 밖이다. 논문 스스로 basic 은 재배열이 심한 쌍에서
MU 가 문장 전체 하나로 붕괴한다고 밝힌다(Fig. 2) — en→de(동사후치)·en→ja(SOV)가 정확히
그 경우다. 붕괴하면 그 자체가 결과다. 표에 올릴 때 "basic, top-10 완화" 를 명시할 것.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def is_prefix_of_any(hyp: str, cands: list[str], spaced: bool) -> bool:
    h = _norm(hyp)
    if not h:
        return False
    for c in cands:
        c = _norm(c)
        if not c.startswith(h):
            continue
        # 띄어쓰기 언어는 어절 중간에서 끊긴 매칭을 배제한다 ("I go" vs "I going").
        if spaced and len(c) > len(h) and not c[len(h)].isspace():
            continue
        return True
    return False


def segment(nmt, text: str, tgt_spaced: bool, n_cands: int = 10,
            trace: list | None = None) -> list[str]:
    toks = _WS.split(text.strip())
    if len(toks) < 2:
        return [text.strip()]
    cands = nmt.full_candidates(text, n=n_cands)

    pieces: list[str] = []
    forced: list[int] = []
    k = 0
    for t in range(1, len(toks) + 1):
        prefix_src = " ".join(toks[:t])
        hyp, ids = nmt.translate_prefix(prefix_src, forced=forced or None)
        if trace is not None:
            trace.append({"t": t, "hyp": hyp})
        if t < len(toks) and is_prefix_of_any(hyp, cands, tgt_spaced):
            pieces.append(" ".join(toks[k:t]))
            k = t
            forced = ids
    if k < len(toks):                       # 남은 꼬리는 마지막 MU
        pieces.append(" ".join(toks[k:]))
    return pieces or [text.strip()]
