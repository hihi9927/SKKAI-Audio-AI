"""AlignAtt (Papi et al., Interspeech 2023) → 소스 경계.

원 정책: 지금 내보내려는 타깃 토큰의 **교차어텐션 argmax 가 최근 `f` 프레임 안에 있으면
아직 정보가 부족한 것이므로 내보내지 않고 더 읽는다.** 노브는 `f` 하나다.

텍스트 적응 두 가지를 명시한다:
  1. 프레임 대신 **소스 어절**을 센다 (오디오가 없다).
  2. 원 논문은 방출 스케줄만 정하지 소스 분절을 내놓지 않는다. `causal_align` 과 같은
     방식으로, 접두사 길이 `t` 에서 새 타깃 토큰이 하나라도 나가면 거기에 경계를 찍는다.

마지막 층은 `</s>` 로 쏠려(attention sink 실측 확인) 쓸 수 없다. `attn_layer` 기본 6 은
NLLB-600M 에서 정렬 단조성이 가장 높았던 층이다(0.92) — 원 논문도 층을 고른다.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def segment(nmt, text: str, f: int = 2) -> list[str]:
    words = _WS.split(text.strip())
    n = len(words)
    if n < 2:
        return [text.strip()]

    pieces: list[str] = []
    forced: list[int] = []
    k = 0
    for t in range(1, n):                     # t = 지금까지 읽은 어절 수
        emitted = nmt.emit_with_alignment(" ".join(words[:t]), forced or None)
        commit: list[int] = []
        for tok_id, w in emitted:
            if w >= t - f:                    # 최근 f 어절에 붙어 있다 → 아직 못 낸다
                break
            commit.append(tok_id)
        if commit:
            forced = forced + commit
            pieces.append(" ".join(words[k:t]))
            k = t
    if k < n:                                 # 소스를 다 읽은 뒤 남은 꼬리
        pieces.append(" ".join(words[k:]))
    return pieces or [text.strip()]
