"""SASST(Yang et al., AAAI 2026) 식 구문 경계 분절 — 논문 본문 기준.

    "we parse source sentences using the `en_core_web_trf` model from spaCy...
     Chunk segmentation is guided by syntactic boundaries derived from noun phrases
     (NP), verb phrases (VP), and prepositional phrases (PP), as well as punctuation
     and dependency transitions (e.g., nsubj → VERB). Rule-based constraints ensure
     that each chunk forms a semantically coherent unit and does not exceed a
     maximum span of seven tokens."

경계 근거는 논문이 나열한 **다섯 갈래뿐**이다. 이 목록을 임의로 넓히지 않는다.

    NP      spaCy `noun_chunks` 의 끝
    PP      `dep_ == "prep"` 인 토큰의 subtree 끝
    VP      동사구의 끝 — 아래 주석 참조
    구두점  구두점 토큰 뒤
    의존 전이  nsubj subtree 끝 (주어 → 동사로 넘어가는 지점)

**남은 조작화 하나: VP.** spaCy 에 동사구 청커가 없고 논문도 정의를 안 준다. 동사 subtree 를
그대로 쓰면 목적어까지 삼켜 VP 가 문장 전체가 되므로, **동사 + 조동사·부정·불변화사**까지를
동사구로 보고 그 끝에 경계를 둔다. 이 선택은 논문에 근거가 없으므로 표에 각주로 남길 것.

원 논문과 다른 점(범위): SASST 는 이 청크로 정렬된 데이터를 만들어 LLM 을 파인튜닝한다.
여기서는 **경계 규칙만** 쓴다 — 우리가 비교하는 것은 라벨 출처이지 학습된 모델이 아니다.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")

# 동사구에 함께 묶이는 의존 관계 (동사 자신 + 이것들)
_VERB_GROUP = {"aux", "auxpass", "neg", "prt"}


class SyntaxSegmenter:
    def __init__(self, model: str = "en_core_web_trf", max_chunk: int = 7):
        import spacy

        self.nlp = spacy.load(model, disable=["ner", "lemmatizer"])
        self.max_chunk = max_chunk

    def _word_ends(self, text: str) -> list[int]:
        """어절 i 의 끝 문자 오프셋(배타적)."""
        ends, pos = [], 0
        for w in _WS.split(text.strip()):
            j = text.index(w, pos)
            pos = j + len(w)
            ends.append(pos)
        return ends

    def boundaries(self, text: str) -> list[int]:
        """1-기반 어절 절단 위치. 마지막 원소는 항상 문장 끝."""
        words = _WS.split(text.strip())
        n = len(words)
        if n < 2:
            return [n]
        ends = self._word_ends(text)
        doc = self.nlp(text)
        end_of = lambda t: t.idx + len(t.text)

        chars: set[int] = set()
        for ch in doc.noun_chunks:                       # NP
            chars.add(end_of(ch[-1]))
        for t in doc:
            if t.dep_ == "prep":                          # PP
                chars.add(max(end_of(x) for x in t.subtree))
            # VP (조작화 — 위 주석). 조동사·부정·불변화사는 **헤드 동사의 동사구에
            # 속하므로** 스스로 VP 를 열지 않는다. 안 빼면 `can | contain` 으로 갈린다.
            if t.pos_ in ("VERB", "AUX") and t.dep_ not in _VERB_GROUP:
                grp = [t] + [c for c in t.children if c.dep_ in _VERB_GROUP]
                chars.add(max(end_of(x) for x in grp))
            if t.is_punct:                                # 구두점
                chars.add(end_of(t))
            if t.dep_ in ("nsubj", "nsubjpass"):          # 의존 전이 nsubj → VERB
                chars.add(max(end_of(x) for x in t.subtree))

        cuts = set()
        for c in chars:
            for i, e in enumerate(ends):
                if e >= c:
                    cuts.add(i + 1)
                    break
        cuts = sorted(x for x in cuts if 0 < x < n)

        # 최대 span 7 토큰 강제 — 규칙이 못 자른 긴 구간을 잘라 넣는다.
        out, prev = [], 0
        for c in cuts + [n]:
            while c - prev > self.max_chunk:
                prev += self.max_chunk
                out.append(prev)
            if c != n:
                out.append(c)
            prev = c
        out = sorted(set(x for x in out if 0 < x < n))
        return out + [n]

    def segment(self, text: str) -> list[str]:
        words = _WS.split(text.strip())
        pieces, prev = [], 0
        for c in self.boundaries(text):
            if c > prev:
                pieces.append(" ".join(words[prev:c]))
                prev = c
        return pieces or [text.strip()]
