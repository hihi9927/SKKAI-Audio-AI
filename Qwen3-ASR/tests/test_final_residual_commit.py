#!/usr/bin/env python3
"""스트림 종료(규칙 4) 잔여 커밋 단위 테스트 (GPU/모델 불필요).

두 가지를 검증한다.

1. **extract 단계 스킵의 커서 갭.** 커밋 가드가 문장을 스킵할 때 커서를 전진시키지
   않으면, Phase 1 의 tail 이 스킵된 문장에서 시작한 채 다음 문장의 startswith 검사를
   받아 실패 → break 하고, 뒤 문장 전부가 미커밋으로 남아 flush(finish)로 흘러간다.
   commit 단계 스킵에는 이 처리가 있었지만(771d7bb) extract 단계에는 없었다.
   실측: CoVoST2-spk dot 축 `en_de_2c41823446e3` — DOT-SLOT-SWITCH 후 재디코딩으로
   앞머리 'Fine.' 이 중복되자 그 1건이 뒤의 'I agree.' / 'This could be the case.' 를
   인질로 잡아 finish 3건이 되었다.

2. **규칙 4의 축 적용 범위.** finish_streaming 이 `enable_dot_commit and
   dot_commit_confirm` 으로 막혀 있어 seg/always 축은 잔여 커밋을 통째로 건너뛰고
   전부 flush(finish)로 나갔다. 실측: CoVoST2-spk seg 축 18,655 세그 중 5,793 개.

모델을 띄우지 않으므로 핸들러는 `object.__new__` 로 만들고 필요한 속성만 채운다.
"""
import asyncio
import importlib.util
import logging
import os
import sys
import unittest

_SERVER = os.path.join(os.path.dirname(__file__), "..", "examples",
                       "streaming_websocket_server.py")
_spec = importlib.util.spec_from_file_location("_sws_under_test", _SERVER)
_sws = importlib.util.module_from_spec(_spec)
sys.modules["_sws_under_test"] = _sws
_spec.loader.exec_module(_sws)

H = _sws.Qwen3ASRStreamingHandler


class _FakeState:
    """init_streaming_state() 가 돌려주는 것 중 이 경로가 읽는 필드만."""

    def __init__(self, text):
        self.text = text
        self.language = "en"
        self.audio_accum = None
        self.unfixed_token_num = 20
        self._raw_decoded = text


def _make_handler(*, enable_dot_commit, dot_commit_confirm, always_commit,
                  rep_dedup=True):
    h = object.__new__(H)
    h.log = logging.getLogger("test_final_residual_commit")
    h.asr_lock = asyncio.Lock()
    h.active_slot = "A"
    h.standby_slot = "B"
    h.current_time = 5.63
    h._in_generate_loop = False
    h._pending_gpt_tasks = []
    h.enable_dot_commit = enable_dot_commit
    h.dot_commit_confirm = dot_commit_confirm
    h.dot_commit_stall_chunks = 1
    h.always_commit = always_commit
    h.rep_dedup = rep_dedup
    h.emitted = []  # [(reason, original)]

    async def _correct_and_translate(text, lang, audio_end_sec=None):
        return text, "<de>" + text, "en", {}

    async def _emit_final_payload(**kw):
        h.emitted.append((kw["reason"], kw["original"]))

    h._correct_and_translate = _correct_and_translate
    h._emit_final_payload = _emit_final_payload
    return h


def _make_slot(h, *, full_text, committed_display="", committed_seg_count=0,
               committed_asr=(), last_text=None, dot_switch_prev=None):
    slot = {
        "state": _FakeState(full_text),
        "flush_lock": asyncio.Lock(),
        "last_text": full_text if last_text is None else last_text,
        "last_text_lang": "en",
        "committed_len": 0,
        "committed_prefix": "",
        "committed_display": committed_display,
        "committed_seg_count": committed_seg_count,
        "audio_anchor_sec": 0.0,
        "committed_asr_set": set(committed_asr),
        "committed_fuzzy_keys": [H._fuzzy_key(t) for t in committed_asr],
    }
    if dot_switch_prev:
        # DOT-SLOT-SWITCH 가 새 슬롯에 남기는 마커 — dot-suffix-dedup 의 발동 조건.
        slot["dot_switch_prev_committed"] = dot_switch_prev
    h.stream_slots = {"A": slot, "B": None}
    return slot


def _leftover(slot):
    """flush_uncommitted 가 집어갈 잔여 = finish 로 나갈 텍스트."""
    cur = H._strip_asr_text((slot["state"].text or "").strip())
    unc = H._uncommitted_from(cur, slot["committed_display"],
                              slot["committed_seg_count"])
    # flush 는 <SEG> 를 뗀 display 를 내보내므로 같은 형태로 비교한다.
    return unc.replace("<SEG>", "").strip()


def _texts(h):
    return [t for _, t in h.emitted]


def _reasons(h):
    return {r for r, _ in h.emitted}


class FinalResidualCommitTests(unittest.IsolatedAsyncioTestCase):

    # ── 1. extract 단계 스킵의 커서 갭 ──────────────────────────────────────
    async def test_extract_skip_does_not_hold_later_sentences_hostage(self):
        """실측 en_de_2c41823446e3: 중복 앞머리 1건이 뒤 2문장을 finish 로 끌고 갔다."""
        h = _make_handler(enable_dot_commit=True, dot_commit_confirm=True,
                          always_commit=False, rep_dedup=False)
        slot = _make_slot(h, full_text="Fine. I agree. This could be the case.",
                          dot_switch_prev="How about Egypt? Fine.")
        await h._process_slot_updates("A", chunk_end=True, final=True)
        self.assertEqual(_texts(h), ["I agree.", "This could be the case."])
        self.assertEqual(_leftover(slot), "")

    async def test_fully_duplicate_residual_is_consumed_not_reemitted(self):
        """잔여가 통째로 중복이면 배출 0건이되, flush 로 재방출되어서도 안 된다."""
        h = _make_handler(enable_dot_commit=True, dot_commit_confirm=True,
                          always_commit=False, rep_dedup=False)
        slot = _make_slot(h, full_text="Fine.",
                          dot_switch_prev="How about Egypt? Fine.")
        await h._process_slot_updates("A", chunk_end=True, final=True)
        self.assertEqual(h.emitted, [])
        self.assertEqual(_leftover(slot), "")

    async def test_extract_skip_on_normal_chunk_path(self):
        """final=False(평상시 청크)에서도 스킵이 뒤 문장을 막지 않아야 한다."""
        h = _make_handler(enable_dot_commit=True, dot_commit_confirm=False,
                          always_commit=False)
        slot = _make_slot(h, full_text="Fine. I agree.", last_text="",
                          dot_switch_prev="How about Egypt? Fine.")
        await h._process_slot_updates("A", chunk_end=True)
        self.assertEqual(_texts(h), ["I agree."])
        self.assertEqual(_leftover(slot), "")

    # ── 2. 규칙 4가 모든 축에 적용되는지 ────────────────────────────────────
    async def test_seg_axis_tail_commits_as_seg_not_finish(self):
        """seg 축: SEG 로 닫히지 않는 꼬리가 축의 사유로 커밋되어야 한다."""
        h = _make_handler(enable_dot_commit=False, dot_commit_confirm=False,
                          always_commit=False)
        slot = _make_slot(h,
                          full_text="She'll be all right.<SEG> I agree. This could be the case.",
                          committed_display="She'll be all right.",
                          committed_seg_count=1,
                          committed_asr=("She'll be all right.",))
        await h._process_slot_updates("A", chunk_end=True, final=True)
        self.assertEqual(_texts(h), ["I agree. This could be the case."])
        self.assertEqual(_reasons(h), {"seg"})  # dot 라벨이 섞이면 안 된다
        self.assertEqual(_leftover(slot), "")

    async def test_always_axis_tail_commits_as_always(self):
        """always 축: 마지막 미완성 청크 잔여도 축의 사유로 나가야 한다."""
        h = _make_handler(enable_dot_commit=False, dot_commit_confirm=False,
                          always_commit=True)
        slot = _make_slot(h, full_text="and then he went", last_text="and then he went")
        await h._process_slot_updates("A", chunk_end=True, final=True)
        self.assertEqual(_texts(h), ["and then he went"])
        self.assertEqual(_reasons(h), {"always"})
        self.assertEqual(_leftover(slot), "")

    async def test_final_runs_even_when_text_unchanged(self):
        """종료 직전 청크와 텍스트가 같아도 final=True 면 조기 리턴하면 안 된다.

        dot_commit_confirm 이 꺼진 축(seg/always)은 _recheck_pending 이 False 라
        current_text == last_text 에서 그대로 리턴했고, 잔여가 통째로 finish 가 됐다.
        """
        h = _make_handler(enable_dot_commit=False, dot_commit_confirm=False,
                          always_commit=False)
        txt = "Nothing was committed yet."
        slot = _make_slot(h, full_text=txt, last_text=txt)
        await h._process_slot_updates("A", chunk_end=True, final=True)
        self.assertEqual(_texts(h), [txt])
        self.assertEqual(_leftover(slot), "")

    # ── 3. 평상시 경로 회귀 ─────────────────────────────────────────────────
    async def test_normal_seg_chunk_still_commits_only_up_to_boundary(self):
        h = _make_handler(enable_dot_commit=False, dot_commit_confirm=False,
                          always_commit=False)
        slot = _make_slot(h, full_text="First one.<SEG> second one", last_text="")
        await h._process_slot_updates("A", chunk_end=True)
        self.assertEqual(_texts(h), ["First one."])
        self.assertEqual(_leftover(slot), "second one")
        self.assertEqual(slot["committed_seg_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
