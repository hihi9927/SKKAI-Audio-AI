#!/usr/bin/env python3
"""우리 커밋 스트림 → simulstream(StreamLAAL) 입력 어댑터.

지표 계산은 **공식 구현에 위임**한다(`simulstream.metrics.scorers.latency.stream_laal`).
이 파일이 하는 일은 딱 하나 — 우리 `metric.json` 의 커밋 목록을 그쪽이 기대하는
`OutputWithDelays` / `ReferenceSentenceDefinition` 로 옮기는 것. 그리고 그 옮기는
과정에서 조용히 틀릴 수 있는 자리에 진단을 박는다.

이 어댑터에서만 결정하는 규칙 (공식 구현은 여기까지 관여하지 않는다)
--------------------------------------------------------------
**조각 사이 공백의 지연은 앞 조각의 마지막 지연을 물려받는다.**
`latency_unit="char"` 이면 `text_items` 가 공백도 한 단위로 세므로(`list(text)`),
조각을 이어붙일 때 생기는 공백에도 지연을 하나 배정해야 한다. 뒤 조각의 지연을 주면
그 한 칸이 실제보다 늦게 찍혀 `d_i` 가 부풀고, 문장 안에서 누적되면 `LAAL_s` 가
조용히 올라간다. **개수만 보는 assert 로는 절대 안 잡힌다** — 그래서 규칙을 여기
명시하고 합성 케이스(`check_streamlaal.py`)로 못 박는다.

**빈 조각은 루프 진입 전에 버린다.** `word` 는 `split(" ")` 가 빈 토큰을 걸러 주지만
`char` 는 안 걸러서, 조각이 0단위인데 사이 공백은 세어져 index 가 어긋난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from simulstream.metrics.readers import (
    OutputWithDelays, ReferenceSentenceDefinition, text_items)


@dataclass
class Commit:
    """커밋 하나 = 서버가 `final` 로 내보낸 세그먼트 하나."""
    text: str
    ideal_delay: float            # 비계산인지: 커밋 결정 시점까지 읽은 소스 오디오(초)
    ca_delay: float               # 계산인지: 클라이언트가 받은 실시간 경과(초)


@dataclass
class Diagnostics:
    """어댑터가 새는지 보는 계기판. assert 는 개수만 지키고 의미는 안 지킨다."""
    n_commits: int = 0
    n_empty_commits_dropped: int = 0
    n_units: int = 0
    # 단조성은 **조각 내부**와 **조각 경계**를 나눠 센다. 공백 상속을 하면 조각 내부는
    # 정의상 단조지만, 뒤 조각의 첫 단어가 앞 공백보다 이른 커밋에서 왔으면 경계에서
    # 합법적으로 역전이 난다 — 그건 커밋 타이밍의 정상 특성이지 백트레이스 누수가 아니다.
    # 뭉뚱그리면 정상 신호가 누수 신호를 오염시켜 "단조위반↔음수 동반상승" 판정이 무뎌진다.
    n_violation_intra: int = 0
    n_violation_boundary: int = 0
    notes: List[str] = field(default_factory=list)


def build_output_with_delays(
    commits: Sequence[Commit],
    latency_unit: str,
    diag: Optional[Diagnostics] = None,
) -> OutputWithDelays:
    """커밋 목록 → `OutputWithDelays`. **이어붙이기와 지연 펼치기를 한 루프에서** 만든다.

    따로 만들면 `char` 에서 조각 사이 공백만큼 개수가 어긋난다. 같은 루프에서 조각과
    공백을 같은 방식으로 소비하면 그 어긋남이 원천적으로 생기지 않는다.
    """
    d = diag if diag is not None else Diagnostics()

    # 빈 조각 제거 — char 에서 유령 공백을 만든다(모듈 docstring 참고).
    kept: List[Commit] = []
    for c in commits:
        if c.text and c.text.strip():
            kept.append(c)
        else:
            d.n_empty_commits_dropped += 1
    d.n_commits = len(kept)

    pieces: List[str] = []
    ideal: List[float] = []
    ca: List[float] = []
    # 각 단위가 몇 번째 조각에서 왔는지. 단조 위반을 내부/경계로 가르는 데 쓴다.
    # 조각 사이 공백은 **앞 조각 소속**으로 본다(지연을 물려받았으므로).
    owner: List[int] = []
    sep_units = len(text_items(" ", latency_unit))     # word→0, char→1

    for idx, c in enumerate(kept):
        n = len(text_items(c.text, latency_unit))
        if n == 0:
            # strip 은 통과했는데 단위가 0 — 이론상 안 나와야 한다. 나오면 기록한다.
            d.notes.append(f"단위 0인 조각을 건너뜀: {c.text!r}")
            continue
        if pieces:
            # 조각 사이 공백. **앞 조각의 마지막 지연을 물려받는다** — 그 시점엔 이미
            # 나와 있던 문자이므로. 뒤 조각 지연을 주면 d_i 가 부푼다.
            pieces.append(" ")
            if sep_units:
                ideal.extend([ideal[-1]] * sep_units)
                ca.extend([ca[-1]] * sep_units)
                owner.extend([owner[-1]] * sep_units)
        pieces.append(c.text)
        ideal.extend([c.ideal_delay] * n)
        ca.extend([c.ca_delay] * n)
        owner.extend([idx] * n)

    final_text = "".join(pieces)

    # 여기서 어긋나면 이어붙이기/펼치기가 갈라진 것이다. 공식 구현의
    # `_split_delays_by_segmented_text` 가 뒤에서 같은 검사를 하지만, 원인이 우리 쪽인지
    # 재분절 쪽인지 구분하려면 여기서 먼저 걸어야 한다.
    n_units = len(text_items(final_text, latency_unit))
    if n_units != len(ideal):
        raise AssertionError(
            f"단위 수 불일치: final_text {n_units} vs 지연 {len(ideal)} "
            f"(unit={latency_unit}, 조각 {len(kept)}개)")
    d.n_units = n_units

    # 단조 위반을 **실제로 센다.** 조각 내부는 같은 값이 반복되므로 정의상 0이어야 하고,
    # 0이 아니면 펼치기가 샌 것이다(가정하지 말고 확인한다). 경계 역전은 커밋 타이밍의
    # 정상 특성일 수 있으므로 따로 센다.
    for i in range(1, len(ideal)):
        if ideal[i] < ideal[i - 1]:
            if owner[i] == owner[i - 1]:
                d.n_violation_intra += 1
            else:
                d.n_violation_boundary += 1

    return OutputWithDelays(final_text, ideal, ca)


def build_reference_defs(sentences: Sequence[dict]) -> List[ReferenceSentenceDefinition]:
    """gold 문장 → `ReferenceSentenceDefinition`.

    `start_time` / `duration` 은 **오직 gold 타임스탬프에서만** 온다
    (`recover_acl6060_timings.py` 산출물). delays 의 max 같은 걸로 추론하면 안 된다 —
    음수 지연과 τ 절단 때문에 `max(delay) ≠ |X_s|` 이고, 그러면 `d*` 와 τ 가 함께 틀어진다.

    참조 토큰 수(`|Y_ref|`)는 **여기서 세지 않는다.** 공식 구현이 `_do_score` 안에서
    `text_items(reference.content, latency_unit)` 로 센다. 우리가 따로 세면 규칙이
    갈라질 수 있다.
    """
    return [
        ReferenceSentenceDefinition(
            content=s["tgt"],
            start_time=float(s["offset"]),
            duration=float(s["duration"]),
        )
        for s in sentences
    ]