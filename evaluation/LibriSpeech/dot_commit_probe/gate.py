"""dot-commit 확정 게이트(안 A) — 청크별 가설 시퀀스에 대해 오프라인 리플레이 가능한 순수 로직.

규칙:
  1) 문맥 확정 — 온점 뒤에 남은 토큰 수 > unfixed_token_num 이면 즉시 커밋.
     (롤백 창 밖이라 다음 청크에서 모델이 수정하지 않는다)
  2) 합의 확정 — 온점이 롤백 창 안(프론티어)이면 pending. 다음 청크 가설이
     여전히 그 지점까지를 prefix로 유지하면 커밋, 바뀌었으면 후보 폐기.
  3) finish — 스트림 종료 시 남은 미커밋 구간 flush.
"""
import re

# streaming_websocket_server가 쓰는 것과 동일 (sentence_boundary.py)
_ABBREV_LOOKBEHIND = r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<!Jr)(?<!Sr)(?<!vs)(?<!No)"
DOT_COMMIT_BOUNDARY_RE = re.compile(
    r"(?:"
    rf"{_ABBREV_LOOKBEHIND}\.(?:\s+|$)"
    r"|[?!](?:\s+|$)"
    r"|[。？！]"
    r"|<SEG>"
    r")"
)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class DotCommitGate:
    """청크 가설을 순서대로 먹여 커밋 이벤트를 뽑는다.

    count_tokens: 문자열 -> 토큰 수 (모델 토크나이저)
    """

    def __init__(self, count_tokens, unfixed_token_num=5, naive=False):
        self.count_tokens = count_tokens
        self.unfixed_token_num = unfixed_token_num
        self.naive = naive  # True면 현재 브랜치 동작(감지 즉시 커밋) 재현
        self.committed = ""
        self.pending = None  # 확정 대기 중인 후보 prefix
        self.commits = []    # (audio_sec, text, reason)
        self.conflicts = 0   # 이미 커밋한 구간이 나중에 수정된 횟수

    def _emit(self, new_committed: str, audio_sec: float, reason: str):
        text = new_committed[len(self.committed):].strip()
        if text:
            self.commits.append((audio_sec, text, reason))
        self.committed = new_committed

    def feed(self, hypothesis: str, audio_sec: float):
        h = hypothesis
        if self.committed and not h.startswith(self.committed):
            # 이미 커밋한 텍스트가 재디코딩으로 바뀜 = 되돌릴 수 없는 오염
            self.conflicts += 1
            self.committed = h[:_common_prefix_len(h, self.committed)]

        if not self.naive and self.pending is not None:
            if h.startswith(self.pending):
                self._emit(self.pending, audio_sec, "dot-stable")
            self.pending = None

        pos = len(self.committed)
        while True:
            m = DOT_COMMIT_BOUNDARY_RE.search(h, pos)
            if not m:
                break
            end = m.end()
            tail_tokens = self.count_tokens(h[end:])
            if self.naive or tail_tokens > self.unfixed_token_num:
                self._emit(h[:end], audio_sec, "dot" if self.naive else "dot-context")
                pos = len(self.committed)
            else:
                self.pending = h[:end]
                break

    def finish(self, final_hypothesis: str, audio_sec: float):
        h = final_hypothesis
        if self.committed and not h.startswith(self.committed):
            self.conflicts += 1
            self.committed = h[:_common_prefix_len(h, self.committed)]
        if h[len(self.committed):].strip():
            self._emit(h, audio_sec, "finish")
        self.pending = None
