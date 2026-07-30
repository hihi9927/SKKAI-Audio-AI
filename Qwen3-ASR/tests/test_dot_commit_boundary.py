#!/usr/bin/env python3
"""dot-commit 문장 경계 판정 로직 단위 테스트 (GPU/모델 불필요, 순수 함수).

<SEG>처럼 "디코딩된 순간 바로 커밋 가능한" 마침표 경계를 찾는 로직 검증.
기존 정규식(`\\.\\s+(?=\\S)`)은 뒤에 단어가 더 와야만 매치되어 발화의
마지막 문장은 절대 못 잡았다 — 이 테스트는 문자열 끝 마침표도 경계로
잡히는지, 그리고 소수점/약어는 여전히 제외되는지를 확인한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qwen_asr.inference.sentence_boundary import count_dot_commit_boundaries


class DotCommitBoundaryTests(unittest.TestCase):
    def test_matches_period_followed_by_next_word(self):
        self.assertEqual(count_dot_commit_boundaries("Hello. World"), 1)

    def test_matches_period_at_end_of_string(self):
        # 핵심 회귀 케이스: 발화의 마지막 문장 — 뒤에 더 이상 텍스트가 없음.
        self.assertEqual(count_dot_commit_boundaries("Hello world."), 1)

    def test_matches_period_at_end_with_trailing_space(self):
        self.assertEqual(count_dot_commit_boundaries("Hello world. "), 1)

    def test_does_not_match_decimal_point(self):
        self.assertEqual(count_dot_commit_boundaries("Pi is 3.14 roughly."), 1)

    def test_does_not_match_known_abbreviation(self):
        self.assertEqual(count_dot_commit_boundaries("Mr. Smith left."), 1)

    def test_does_not_match_bare_abbreviation_at_end(self):
        # "Mr."로 끝나면 아직 이름이 안 나온 것 — 경계 아님.
        self.assertEqual(count_dot_commit_boundaries("Ask Mr."), 0)

    def test_matches_seg_token(self):
        self.assertEqual(count_dot_commit_boundaries("Hello world.<SEG>"), 1)

    def test_matches_question_and_exclamation_at_end(self):
        self.assertEqual(count_dot_commit_boundaries("Are you sure?"), 1)
        self.assertEqual(count_dot_commit_boundaries("Watch out!"), 1)

    def test_matches_cjk_punctuation(self):
        self.assertEqual(count_dot_commit_boundaries("你好。"), 1)

    def test_counts_multiple_sentences(self):
        self.assertEqual(count_dot_commit_boundaries("One. Two. Three."), 3)

    def test_empty_string_has_no_boundary(self):
        self.assertEqual(count_dot_commit_boundaries(""), 0)

    def test_count_is_monotonic_as_text_grows(self):
        # on_dot 콜백이 "count > prev_count"로 델타 트리거하므로, 텍스트가 늘어나도
        # 이미 확정된 경계의 매치 수가 줄어들면 안 된다.
        partial = "There's iron, they say, in all our blood, and a grain or two perhaps is good, but his—he makes me harshly feel—has got a little too much of steel."
        full = partial + " Anon."
        self.assertGreaterEqual(count_dot_commit_boundaries(full), count_dot_commit_boundaries(partial))


if __name__ == '__main__':
    unittest.main(verbosity=2)
