#!/usr/bin/env python3
"""metrics_ast 자체 검증. pytest 없이 단독 실행한다.

    python evaluation/ast/check_metrics_ast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics_ast as M  # noqa: E402


def approx(a, b, tol=1e-6):
    return a is not None and abs(a - b) < tol


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        check.failed += 1


check.failed = 0


# ── LAAL 손계산 대조 ─────────────────────────────────────────────────────────
# T=2000ms, |Y_hyp|=|Y_ref|=4, d=[1000,1000,2000,2000]
#   γ = 2000/4 = 500
#   τ = 3 (d_3 = 2000 ≥ T 인 첫 인덱스)
#   Σ = (1000−0) + (1000−500) + (2000−1000) = 2500 → /3 = 833.333...
check(
    "LAAL 손계산 (등길이)",
    approx(M.compute_laal([1000, 1000, 2000, 2000], 2000, 4), 2500 / 3, 1e-9),
)

# 지연 0 · 즉시 전량 출력 → LAAL 은 γ 만큼의 구조적 하한을 갖는다.
#   d=[0,0,0,0], γ=500 → Σ = 0 −500 −1000 −1500 = −3000 → /4 = −750
check("LAAL 지연 0", approx(M.compute_laal([0, 0, 0, 0], 2000, 4), -750.0, 1e-9))

# ── LAAL 의 존재 이유: 과소생성이 보상받지 않아야 한다 ───────────────────────
# T=2000, hyp 2단위, ref 4단위, d=[500,2000]
#   LAAL γ = 2000/max(2,4) = 500 → (500) + (2000−500) = 2000 → /2 = 1000
#   AL   γ = 2000/2        = 1000 → (500) + (2000−1000) = 1500 → /2 = 750
laal_under = M.compute_laal([500, 2000], 2000, 4)
al_under = M.compute_laal([500, 2000], 2000, None)
check("과소생성 LAAL 손계산", approx(laal_under, 1000.0, 1e-9))
check("과소생성 AL 손계산", approx(al_under, 750.0, 1e-9))
check("과소생성은 AL보다 LAAL이 크다", laal_under > al_under)

# 참조와 길이가 같으면 LAAL == AL
check(
    "등길이면 LAAL == AL",
    approx(
        M.compute_laal([1000, 1000, 2000, 2000], 2000, 4),
        M.compute_laal([1000, 1000, 2000, 2000], 2000, None),
    ),
)

# ── 경계 조건 ────────────────────────────────────────────────────────────────
check("빈 가설 → None", M.compute_laal([], 2000, 4) is None)
check("소스 길이 0 → None", M.compute_laal([1000], 0, 4) is None)
# 소스를 다 읽기 전에 출력이 끝난 경우 → τ = |Y_hyp|
check("τ fallback", approx(M.compute_laal([100, 200], 5000, 2), (100 + (200 - 2500)) / 2))

# ── 세그먼트 확장 ────────────────────────────────────────────────────────────
segs = [("ein zwei", 1000.0), ("drei", 2000.0)]
check("expand_delays", M.expand_delays(segs, "word") == [1000.0, 1000.0, 2000.0])
check("빈 세그먼트는 제외", M.expand_delays([("", 500.0)] + segs, "word") == [1000.0, 1000.0, 2000.0])
check(
    "laal_for_utterance",
    approx(
        M.laal_for_utterance(segs, 2000, "ein zwei drei vier", "word"),
        M.compute_laal([1000, 1000, 2000], 2000, 4),
    ),
)

# ── 단위 세기 ────────────────────────────────────────────────────────────────
check("word 단위", M.count_units("hallo wie geht es", "word") == 4)
check("char 단위(공백 제외)", M.count_units("你 好 吗", "char") == 3)

# ── 비음성 마커 제거 ─────────────────────────────────────────────────────────
check(
    "영어 이벤트 제거",
    M.strip_nonspeech("So I said (Laughter) it was fine") == "So I said it was fine",
)
check(
    "독일어 이벤트 제거",
    M.strip_nonspeech("Also sagte ich (Gelächter) alles gut") == "Also sagte ich alles gut",
)
check(
    "본문 괄호는 보존",
    M.strip_nonspeech("Im Jahr (2005) passierte es") == "Im Jahr (2005) passierte es",
)

# ── BLEU ─────────────────────────────────────────────────────────────────────
score, sig = M.corpus_bleu_score(["das ist ein test"], ["das ist ein test"], "13a")
check("완전 일치 BLEU = 100", approx(score, 100.0, 1e-6))
check("signature 기록됨", bool(sig) and "13a" in str(sig))
score_bad, _ = M.corpus_bleu_score(["völlig anderer satz hier"], ["das ist ein test"], "13a")
check("불일치 BLEU < 완전일치", score_bad is not None and score_bad < score)
check("토크나이저 매핑", M.resolve_tokenize("de") == "13a" and M.resolve_tokenize("zh") == "zh")

print()
if check.failed:
    print(f"{check.failed} 개 실패")
    sys.exit(1)
print("모두 통과")
