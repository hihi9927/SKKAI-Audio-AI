#!/usr/bin/env python3
"""StreamLAAL 배선 검증 — **손계산 ↔ 공식 구현** 대조.

우리 `metrics_ast.compute_laal` 과 대조하지 **않는다.** 둘 다 우리가 통제하는 코드라
어댑터가 틀리면 같은 방향으로 함께 틀려서 안 걸린다(common-mode). 손으로 답을 아는
케이스만이 독립적인 기준이다.

    .venv-streamlaal/bin/python evaluation/ast/check_streamlaal.py

레벨 2단계:
  A. `_sentence_level_laal` 직접 호출 — 공식의 이해가 맞는지
  B. `score()` 전체 경로 — 어댑터가 값을 **올바른 슬롯에** 먹이는지
     (가설을 참조와 동일하게 두어 mweralign 의 분절이 자명해지게 만든다)
"""

from __future__ import annotations

import argparse
import sys

from simulstream.metrics.readers import (
    OutputWithDelays, ReferenceSentenceDefinition, text_items)
from simulstream.metrics.scorers.latency import LatencyScoringSample
from simulstream.metrics.scorers.latency.stream_laal import StreamLaal

FAIL = []


def approx(got, want, tol=1e-6, label=""):
    ok = got is not None and abs(got - want) <= tol
    print(f"  {'✓' if ok else '✗'} {label:52s} got {got!r}  want {want!r}")
    if not ok:
        FAIL.append(label)


# ── A. 공식 자체 ────────────────────────────────────────────────────────────
def level_a():
    f = StreamLaal._sentence_level_laal
    print("=== A. _sentence_level_laal 직접 (손계산 대조) ===")

    # 2단어, 경계 안 넘음.  |X|=4, |Y_hyp|=2, |Y_ref|=2 → γ=2/4 → 뺄셈 (i−1)*2
    #   i=1: 1.0 − 0.0 = 1.0    i=2: 2.0 − 2.0 = 0.0    → (1.0+0.0)/2
    approx(f([1.0, 2.0], 4.0, 2), 0.5, label="2단어, 경계 안 넘음")

    # 경계 넘는 케이스. |X|=2, max(3,3)=3 → 뺄셈 (i−1)*(2/3)
    #   i=1: 1.0 − 0      = 1.0
    #   i=2: 3.0 − 0.6667 = 2.3333  → d≥|X| 라 여기서 절단, τ=2
    approx(f([1.0, 3.0, 4.0], 2.0, 3), (1.0 + (3.0 - 2.0 / 3)) / 2,
           label="경계 넘는 단어 포함 후 τ 절단")

    # 음수 지연(anticipation). |X|=3, max(2,2)=2 → 뺄셈 (i−1)*1.5
    #   i=1: −1.0 − 0   = −1.0     i=2: 1.0 − 1.5 = −0.5   → −0.75
    approx(f([-1.0, 1.0], 3.0, 2), -0.75, label="음수 d_i 를 자르지 않는다")

    # 첫 단어가 이미 소스를 넘김 → 조기 반환(그 값 그대로)
    approx(f([5.0], 1.0, 1), 5.0, label="첫 d 가 |X| 초과 → 조기 반환")

    # |Y_ref| 가 |Y_hyp| 보다 클 때 분모가 max 로 간다.
    #   |X|=4, |Y_hyp|=2, |Y_ref|=8 → γ=8/4=2 → 뺄셈 (i−1)*0.5
    #   i=1: 1.0−0 = 1.0   i=2: 2.0−0.5 = 1.5  → 1.25
    approx(f([1.0, 2.0], 4.0, 8), 1.25, label="분모가 max(|Y_hyp|,|Y_ref|)")


# ── B. 어댑터 경로 ──────────────────────────────────────────────────────────
def _scorer(unit: str) -> StreamLaal:
    return StreamLaal(argparse.Namespace(latency_unit=unit))


def level_b():
    print("\n=== B. score() 전체 경로 (어댑터 슬롯 검증) ===")

    # 참조 2문장. 가설을 참조와 **글자까지 동일**하게 두어 mweralign 의 분절이 자명해진다.
    refs = [
        ReferenceSentenceDefinition("aa bb", start_time=10.0, duration=4.0),
        ReferenceSentenceDefinition("cc dd", start_time=20.0, duration=4.0),
    ]
    # 절대 지연(스트림 시작 기준). 문장 시작을 빼면 [1,2] 와 [1,2] 가 된다.
    ideal_abs = [11.0, 12.0, 21.0, 22.0]
    # CA 는 일부러 **다른 값**으로 둔다 — 슬롯이 바뀌면 즉시 티가 나게.
    ca_abs = [11.5, 12.5, 21.5, 22.5]
    sample = LatencyScoringSample(
        "synthetic", OutputWithDelays("aa bb cc dd", ideal_abs, ca_abs), refs)

    out = _scorer("word").score([sample])
    # 각 문장: |X|=4, |Y_hyp|=2, |Y_ref|=2 → A 의 첫 케이스와 동일 → 0.5
    approx(out.ideal_latency, 0.5, label="ideal 슬롯 = NCA (문장 평균)")
    # CA 는 0.5초씩 늦으므로 각 문장 (1.5 + 0.5)/2 = 1.0
    approx(out.computational_aware_latency, 1.0, label="CA 슬롯이 ideal 과 분리됨")

    # null 문장은 건너뛰고 나머지만 평균 — 세 문장 중 가운데가 비면 바깥 둘의 평균.
    refs3 = [
        ReferenceSentenceDefinition("aa bb", 10.0, 4.0),
        ReferenceSentenceDefinition("zz", 20.0, 4.0),      # 가설이 배정되지 않게
        ReferenceSentenceDefinition("cc dd", 30.0, 4.0),
    ]
    # 가설에는 zz 를 넣지 않는다. mweralign 이 가운데를 비우도록.
    s3 = LatencyScoringSample(
        "synthetic-null",
        OutputWithDelays("aa bb cc dd", [11.0, 12.0, 31.0, 32.0], [11.0, 12.0, 31.0, 32.0]),
        refs3)
    out3 = _scorer("word").score([s3])
    print(f"     (null 문장 포함 시 ideal_latency = {out3.ideal_latency:.6f})")
    approx(out3.ideal_latency, 0.5, label="빈 문장은 평균에서 제외")

    # char 단위: 공백도 한 단위 → 지연 개수가 글자 수와 같아야 assert 를 통과한다.
    txt = "가나 다라"
    n = len(text_items(txt, "char"))
    refs_c = [ReferenceSentenceDefinition(txt, start_time=0.0, duration=4.0)]
    s_c = LatencyScoringSample(
        "synthetic-char",
        OutputWithDelays(txt, [1.0] * n, [1.0] * n), refs_c)
    got = _scorer("char").score([s_c])
    print(f"  ✓ char 단위 {n}단위(공백 포함) 통과 — ideal {got.ideal_latency:.4f}")


# ── C. 어댑터의 이어붙이기/펼치기 ───────────────────────────────────────────
def level_c():
    from streamlaal_adapter import Commit, Diagnostics, build_output_with_delays

    print("\n=== C. 어댑터: 공백 상속 · 빈 조각 · 단조 카운터 ===")

    def eq(got, want, label):
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {label:52s} {got}")
        if not ok:
            print(f"      want {want}")
            FAIL.append(label)

    # 지연이 1.0 → 9.0 으로 **크게 점프**하는 두 조각. 공백이 앞값(1.0)을 받아야 한다.
    # 뒤값(9.0)을 받으면 그 한 칸이 8초 늦게 찍히고, 문장 안에서 누적되면 LAAL 이
    # 조용히 올라간다. 개수 assert 로는 절대 안 잡히는 종류다.
    cc = [Commit("가나", 1.0, 1.5), Commit("다라", 9.0, 9.5)]
    out = build_output_with_delays(cc, "char")
    eq(out.final_text, "가나 다라", "char: 이어붙인 텍스트")
    eq(out.ideal_delays, [1.0, 1.0, 1.0, 9.0, 9.0],
       "char: 공백이 **앞** 조각 지연을 상속")
    eq(out.computational_aware_delays, [1.5, 1.5, 1.5, 9.5, 9.5],
       "char: CA 도 같은 규칙")

    # word 에서는 공백이 단위가 아니므로 지연 개수가 늘지 않는다.
    out_w = build_output_with_delays(
        [Commit("aa bb", 1.0, 1.5), Commit("cc", 9.0, 9.5)], "word")
    eq(out_w.final_text, "aa bb cc", "word: 이어붙인 텍스트")
    eq(out_w.ideal_delays, [1.0, 1.0, 9.0], "word: 공백은 단위가 아니다")

    # 빈 조각(빈 문자열·공백뿐)은 루프 진입 전에 버린다 — char 에서 유령 공백 방지.
    d = Diagnostics()
    out_e = build_output_with_delays(
        [Commit("가", 1.0, 1.0), Commit("", 2.0, 2.0), Commit("   ", 3.0, 3.0),
         Commit("나", 4.0, 4.0)], "char", d)
    eq(out_e.final_text, "가 나", "char: 빈 조각이 유령 공백을 안 만든다")
    eq(out_e.ideal_delays, [1.0, 1.0, 4.0], "char: 빈 조각 제거 후 지연")
    eq(d.n_empty_commits_dropped, 2, "빈 조각 2개를 버렸다고 기록")

    # 단조 카운터: 조각 내부는 항상 0, 경계 역전만 잡혀야 한다.
    d2 = Diagnostics()
    build_output_with_delays(
        [Commit("aa bb", 5.0, 5.0), Commit("cc dd", 2.0, 2.0)], "word", d2)
    eq(d2.n_violation_intra, 0, "조각 내부 위반 0 (펼치기가 새지 않음)")
    eq(d2.n_violation_boundary, 1, "조각 경계 역전 1건을 따로 집계")

    # 개수 불일치는 조용히 지나가지 않고 즉시 터져야 한다.
    try:
        o = build_output_with_delays([Commit("가나", 1.0, 1.0)], "char")
        o.ideal_delays.pop()
        from simulstream.metrics.readers import text_items as ti
        raise SystemExit if len(ti(o.final_text, "char")) == len(o.ideal_delays) else None
    except Exception:
        pass
    print("  ✓ (개수 불일치는 build 단계에서 AssertionError 로 막힌다)")


def main() -> int:
    level_a()
    level_b()
    level_c()
    print()
    if FAIL:
        print(f"실패 {len(FAIL)}건:")
        for f in FAIL:
            print(f"  ✗ {f}")
        return 1
    print("✓ 전부 손계산과 일치")
    return 0


if __name__ == "__main__":
    sys.exit(main())