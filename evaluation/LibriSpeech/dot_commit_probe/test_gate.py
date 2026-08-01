"""gate.py 단위 테스트 — 실제 probe 로그에서 관찰된 시퀀스를 그대로 사용."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import DotCommitGate

# 대략적 토큰 수 (단어 수 기반) — 단위테스트용
def wc(s):
    return len(s.split())


def run(seqs, final, naive=False, unfixed=3):
    g = DotCommitGate(wc, unfixed_token_num=unfixed, naive=naive)
    for i, h in enumerate(seqs):
        g.feed(h, float(i * 2))
    g.finish(final, float(len(seqs) * 2))
    return g


def test_mechanical_dot_not_committed():
    # 1688-142285-0012 실제 시퀀스
    seqs = [
        "No one came forwards to help the.",
        "No one came forwards to help the mother and this boy.",
        "No one came forwards to help the mother and this boy.",
    ]
    g = run(seqs, seqs[-1])
    texts = [t for _, t, _ in g.commits]
    assert texts == ["No one came forwards to help the mother and this boy."], texts
    assert g.conflicts == 0


def test_real_dot_confirmed_by_next_chunk():
    # 1688-142285-0019: 'Not vicious.' 는 진짜 경계 → c1에서 확정
    seqs = [
        "Not vicious. He never.",
        "Not vicious. He never said that.",
        "Not vicious. He never said that.",
    ]
    g = run(seqs, seqs[-1])
    assert [t for _, t, _ in g.commits] == [
        "Not vicious.",
        "He never said that.",
    ], g.commits
    assert g.commits[0][2] == "dot-stable"


def test_context_confirmed_commits_immediately():
    # 온점 뒤 단어가 충분히 쌓이면(롤백 창 밖) 같은 청크에서 바로 커밋
    seqs = ["Not vicious. He never said that at all, I think."]
    g = run(seqs, seqs[-1])
    assert g.commits[0][1] == "Not vicious."
    assert g.commits[0][2] == "dot-context"


def test_silence_persistence_confirms_final_dot():
    # 발화 종료 후 무음 청크에서 가설 불변 → 다음 청크에 확정 (VAD 대체)
    seqs = ["So they left Milton.", "So they left Milton."]
    g = run(seqs, seqs[-1])
    assert [(t, r) for _, t, r in g.commits] == [("So they left Milton.", "dot-stable")]


def test_naive_mode_commits_every_chunk():
    seqs = [
        "No one came forwards to help the.",
        "No one came forwards to help the mother and this boy.",
    ]
    g = run(seqs, seqs[-1], naive=True)
    assert len(g.commits) == 2, g.commits
    assert g.conflicts >= 1  # 커밋한 텍스트가 뒤에서 수정됨


def test_abbreviation_not_a_boundary():
    seqs = ["Mr. Hale said so, and then he left the room quietly."]
    g = run(seqs, seqs[-1])
    assert [t for _, t, _ in g.commits] == [
        "Mr. Hale said so, and then he left the room quietly."
    ], g.commits


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
