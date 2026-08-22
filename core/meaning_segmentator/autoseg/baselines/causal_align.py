"""TransLLaMa(Koshkin et al. 2024) 식 인과정렬 → 소스 경계.

원논문은 SimAlign 정렬로 `<WAIT>` 토큰을 끼워 **파인튜닝 시퀀스**를 만든다. 소스 분절
경계를 직접 내놓지 않으므로, 같은 인과 스케줄에서 경계를 유도하는 단계는 우리가 얹는다:

    req(j)  = max{ i : (i, j) ∈ align }      타깃 토큰 j 가 요구하는 마지막 소스 위치
    g(j)    = max(req(1..j))                 단조화 — 이것이 `<WAIT>` 삽입과 동치다
    경계    = { g(j) } 의 서로 다른 값        새 타깃 조각이 방출 가능해지는 지점

정렬이 없는 타깃 토큰은 직전 토큰의 요구치를 물려받는다(carry-forward). 첫 토큰이
비정렬이면 첫 정렬 토큰의 요구치까지 미룬다.

**참조 번역이 있어야 한다** → 오라클성 라벨이다. 제안 루프는 소스만 보고 내는데 이쪽은
정답 타깃을 쥐고 경계를 정한다. 표에 올릴 때 각주 필수.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def tokenize_src(text: str) -> list[str]:
    """논문 그대로 — nltk `word_tokenize`, **구두점도 하나의 "단어"로 센다**.

        "we split each sentence using the word_tokenize function from the nltk
         package, treating punctuation marks as \"words\""

    공백 분리를 쓰면 정렬도 경계 위치도 달라진다.
    """
    from nltk import word_tokenize

    return word_tokenize(text.strip()) or _WS.split(text.strip())


def tokenize_tgt(text: str, tgt: str) -> list[str]:
    """정렬 대상 토큰. 비띄어쓰기 언어는 형태소/문자로 쪼갠다."""
    if tgt in ("de", "en", "es"):
        from nltk import word_tokenize

        return word_tokenize(text.strip()) or _WS.split(text.strip())
    if tgt == "ja":
        from fugashi import Tagger

        global _JA
        try:
            _JA
        except NameError:
            _JA = Tagger()
        return [w.surface for w in _JA(text)] or [text]
    if tgt == "zh":
        # jieba 를 안 쓰고 문자 단위로 둔다. mBERT 서브워드가 어차피 문자에 가깝다.
        return [c for c in text if not c.isspace()] or [text]
    return _WS.split(text.strip())


class CausalAligner:
    def __init__(self, model: str = "bert", device: str = "cuda",
                 matching: str = "itermax"):
        from simalign import SentenceAligner

        self.matching = matching
        self.aligner = SentenceAligner(model=model, token_type="bpe",
                                       matching_methods="i", device=device)

    def boundaries(self, src_toks: list[str], tgt_toks: list[str]) -> list[int]:
        """1-기반 소스 절단 위치(그 위치까지 포함해 조각을 끊는다)."""
        if len(src_toks) < 2 or not tgt_toks:
            return [len(src_toks)]
        aligns = self.aligner.get_word_aligns(src_toks, tgt_toks)[self.matching]

        req: list[int | None] = [None] * len(tgt_toks)
        for i, j in aligns:
            if 0 <= j < len(tgt_toks) and 0 <= i < len(src_toks):
                req[j] = i + 1 if req[j] is None else max(req[j], i + 1)

        # 비정렬 토큰은 앞의 요구치를 물려받는다. 선두 비정렬 구간은 첫 정렬값으로 채운다.
        first = next((r for r in req if r is not None), len(src_toks))
        filled: list[int] = []
        last = first
        for r in req:
            if r is not None:
                last = r
            filled.append(last)

        g: list[int] = []
        run = 0
        for r in filled:
            run = max(run, r)
            g.append(run)

        cuts = sorted(set(g))
        if cuts[-1] != len(src_toks):
            cuts.append(len(src_toks))
        return cuts

    def segment(self, src_text: str, tgt_text: str, tgt: str) -> list[str]:
        """정렬은 nltk 토큰 위에서 하되, 조각은 **원문 어절**로 되돌려 낸다.

        다른 정책과 조각 단위가 어긋나면 지연 격자가 안 맞는다. nltk 는 구두점을
        떼내므로 그대로 join 하면 `word .` 처럼 원문과 다른 문자열이 나온다.
        """
        src_toks = tokenize_src(src_text)
        tgt_toks = tokenize_tgt(tgt_text, tgt)
        cuts = self.boundaries(src_toks, tgt_toks)

        # nltk 토큰 인덱스 → 원문 어절 인덱스 (문자 오프셋으로 되매핑)
        words = _WS.split(src_text.strip())
        w_end, pos = [], 0
        for w in words:
            j = src_text.index(w, pos)
            pos = j + len(w)
            w_end.append(pos)
        t_end, pos = [], 0
        for t in src_toks:
            j = src_text.find(t, pos)
            if j < 0:
                j = pos
            pos = j + len(t)
            t_end.append(pos)

        word_cuts = set()
        for c in cuts:
            ce = t_end[min(c, len(t_end)) - 1]
            for i, e in enumerate(w_end):
                if e >= ce:
                    word_cuts.add(i + 1)
                    break
        word_cuts = sorted(x for x in word_cuts if 0 < x <= len(words))
        if not word_cuts or word_cuts[-1] != len(words):
            word_cuts.append(len(words))

        pieces, prev = [], 0
        for c in word_cuts:
            if c > prev:
                pieces.append(" ".join(words[prev:c]))
                prev = c
        return pieces or [src_text]
