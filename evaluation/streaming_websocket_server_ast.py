#!/usr/bin/env python3
"""공용 AST(음성번역) 평가 서버.

FSL 서버(`LibriSpeech/servers/streaming_websocket_server_fsl.py`)를 상속해
**LAAL 계산에 필요한 필드만** 얹는다. SEG 커밋의 `audioEndSec` 는 토큰비율 역추정
로직이 있어야 나오므로 그걸 다시 쓰지 않고 그대로 물려받는다.

    python evaluation/streaming_websocket_server_ast.py --no-idle-shutdown

추가되는 `final` 필드
--------------------
decisionAudioSec  커밋을 **결정한 순간**까지 읽은 소스 오디오 길이(초). LAAL 의 d_i.
                  기존 `audioEndSec` 와 다르다 — 그건 세그먼트 *내용*의 경계(SEG는
                  토큰비율 역추정)이지 결정 시점의 읽기 지점이 아니다.
                    SEG/dot : SEG 를 감지한 청크의 끝 오디오 위치
                    vad     : VAD 트리거 시점 (speech_end + VAD_MIN_SILENCE_MS)
                    finish  : 그 시점까지 수신한 전체 오디오
emitElapsedSec    스트림 시작부터 이 payload 를 send 하기까지의 서버 wall-clock(초).
audioReceivedSec  send 시점까지 수신한 오디오 길이(초). 디버깅·검산용.

LAAL 자체는 서버가 계산하지 않는다 — 참조 번역과 소스 길이를 아는 쪽은 클라이언트다.
(`evaluation/ast/metrics_ast.py`)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
QWEN3_ROOT = PROJECT_ROOT / "Qwen3-ASR"
FSL_SERVER_DIR = HERE / "LibriSpeech" / "servers"

sys.path.insert(0, str(QWEN3_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))
sys.path.insert(2, str(FSL_SERVER_DIR))

import streaming_websocket_server_fsl as fsl_server  # noqa: E402

sys.path.insert(3, str(HERE / "ast"))
import trans_guard  # noqa: E402

logger = logging.getLogger(__name__)


class _StartSniffingWS:
    """`start` 메시지의 uttId 만 훔쳐보고 나머지는 그대로 흘려보내는 래퍼.

    FSL 이 `handle()` 을 통째로 오버라이드하고 있어서 start 처리부에 훅을 걸 자리가 없다.
    140줄을 복사하는 대신 websocket 을 감싸 수신 프레임만 들여다본다.
    """

    def __init__(self, ws):
        self._ws = ws
        self.state = {"utt_id": None, "epoch": 0}

    def _sniff(self, msg):
        if not isinstance(msg, str):
            return
        try:
            data = json.loads(msg)
        except Exception:
            return
        if data.get("type") == "start":
            self.state["utt_id"] = data.get("uttId")
            self.state["epoch"] += 1

    def __getattr__(self, name):
        return getattr(self._ws, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self._ws.__aiter__().__anext__()
        self._sniff(msg)
        return msg

    async def recv(self, *a, **kw):
        msg = await self._ws.recv(*a, **kw)
        self._sniff(msg)
        return msg


class ASTStreamingHandler(fsl_server.FSLStreamingHandler):
    """FSL 핸들러 + LAAL 용 결정시점/전송시점 계측 + 발화 단위 귀속(uttId)."""

    def __init__(self, websocket, *args, **kwargs):
        sniffer = _StartSniffingWS(websocket)
        super().__init__(sniffer, *args, **kwargs)
        self._ast_sniffer = sniffer
        self._ast_flush_inflight = 0   # 실행 중인 _flush_pending_gpt_tasks 수
        self._ast_flush_overlaps = 0   # 핸들이 덮어써진 횟수

    # ── 밀림(late final) 추적 ────────────────────────────────────────────────
    # base 는 청크마다 `self._gpt_flush_task = asyncio.create_task(...)` 로 **대입**한다.
    # 이전 flush 가 아직 돌고 있으면 그 핸들은 그대로 사라지고, `_drain_pending_gpt` 는
    # 마지막 핸들만 await 하므로 앞의 flush 를 놓친다. 놓친 flush 가 종료 뒤에 끝나면
    # 그 final 이 다음 발화로 밀려 나간다.
    #
    # 핸들 대신 **실행 중인 개수**를 세면 덮어쓰기와 무관하게 전부 기다릴 수 있다.

    async def _flush_pending_gpt_tasks(self) -> None:
        if self._ast_flush_inflight > 0:
            self._ast_flush_overlaps += 1
            logger.warning("[AST-OVERLAP] flush 태스크 중첩 (실행 중 %d개) — 핸들 덮어쓰기 발생",
                           self._ast_flush_inflight)
        self._ast_flush_inflight += 1
        try:
            await super()._flush_pending_gpt_tasks()
        finally:
            self._ast_flush_inflight -= 1

    async def _drain_pending_gpt(self) -> None:
        await super()._drain_pending_gpt()
        # 핸들 덮어쓰기로 놓친 in-flight flush 를 마저 기다린다.
        waited = 0.0
        while self._ast_flush_inflight > 0 and waited < 10.0:
            await asyncio.sleep(0.01)
            waited += 0.01
        if waited > 0:
            logger.info("[AST-DRAIN] 덮어써진 flush 를 %.2f초 더 기다림", waited)
        if self._pending_gpt_tasks:
            await self._flush_pending_gpt_tasks()

    @property
    def _ast_utt_id(self):
        return self._ast_sniffer.state.get("utt_id")

    def _decision_audio_sec(self, slot_key: str, reason: str) -> float:
        """커밋 결정 시점까지 읽은 소스 오디오 길이(초).

        주의: 이 메서드는 `super()._emit_final_payload()` **이전**에 호출해야 한다.
        상위 구현이 `_pending_vad_trigger_sec` 와 `_slot_seg_detected[slot_key]` 를
        소비(pop)하기 때문이다.
        """
        if reason == "vad" and self._pending_vad_trigger_sec is not None:
            # VAD 는 speech_end 가 아니라 트리거(= speech_end + 침묵대기) 시점에 결정한다.
            # 침묵을 기다린 비용은 정책이 치른 값이므로 지연에 포함되는 게 맞다.
            return float(self._pending_vad_trigger_sec)

        seg_info = self._slot_seg_detected.get(slot_key)
        if seg_info is not None and seg_info.get("audio_sec") is not None:
            return float(seg_info["audio_sec"])

        # dot 커밋(SEG 미감지)과 finish: 현재까지 수신·투입된 오디오 전량
        return float(self.current_time)

    # ── 발화 귀속 ────────────────────────────────────────────────────────────
    # uttId 는 **커밋이 결정되는 순간** 찍어야 한다. payload 를 만드는 시점이나 전송
    # 시점에 찍으면 늦은 emit 이 다음 발화의 id 를 달고 나가서, 정확히 막으려던 오염을
    # 그대로 재현한다. `_translate` / `_correct_and_translate` 는 커밋 직후(번역 태스크
    # 진입 시점)에 실행되므로 여기서 읽는 값이 그 커밋이 속한 발화다.

    # ── 번역 호출 계측 ──────────────────────────────────────────────────────
    # call count 는 보고할 지표이고, **번역 실패는 사후에 복구할 수 없다.** 실패하면
    # 번역문이 빈 문자열로 돌아와 "정책이 아무 말도 못 만든 커밋"과 구분되지 않는다.
    # 그래서 커밋 시점에 세어서 payload 에 실어 보낸다(`trans_guard` 참고).
    #
    # `begin_local` 은 이미 상위가 집계 중이면 None 을 준다 — base 의 방향 자가교정이
    # `_translate` 를 한 번 더 부르므로, 가장 바깥(`_correct_and_translate`)만 센다.

    @staticmethod
    def _attach_trans_stats(extra: dict, local: Optional[dict]) -> None:
        if not local:
            return
        extra["transCalls"] = local.get("calls", 0)
        extra["transRetries"] = local.get("retries", 0)
        extra["transFailed"] = bool(local.get("failed", 0))

    async def _translate(self, text, target_lang, audio_end_sec=None):
        utt_id = self._ast_utt_id
        tok = trans_guard.begin_local()
        try:
            translation, lang, extra = await super()._translate(
                text, target_lang, audio_end_sec
            )
        finally:
            local = trans_guard.end_local(tok)
        extra["uttId"] = utt_id
        self._attach_trans_stats(extra, local)
        return translation, lang, extra

    async def _correct_and_translate(self, text, current_lang, audio_end_sec):
        utt_id = self._ast_utt_id
        tok = trans_guard.begin_local()
        try:
            corrected, translation, lang, extra = await super()._correct_and_translate(
                text, current_lang, audio_end_sec
            )
        finally:
            local = trans_guard.end_local(tok)
        extra["uttId"] = utt_id
        self._attach_trans_stats(extra, local)
        return corrected, translation, lang, extra

    async def _emit_final_payload(
        self,
        *,
        slot_key: str,
        original: str,
        translation: str,
        language: str,
        reason: str,
        audio_end_sec: float,
        extra: Optional[dict] = None,
    ) -> None:
        # extra 는 상위에서 timing 으로 쓰이고 payload 에 그대로 병합된다.
        merged = dict(extra or {})
        merged["decisionAudioSec"] = round(
            self._decision_audio_sec(slot_key, reason), 3
        )
        await super()._emit_final_payload(
            slot_key=slot_key,
            original=original,
            translation=translation,
            language=language,
            reason=reason,
            audio_end_sec=audio_end_sec,
            extra=merged,
        )

    async def finish_streaming(self):
        """스트림 종료 전에 진행 중인 번역 태스크를 결정적으로 완결시킨다.

        base 의 `finish_streaming` 은 `_drain_pending_gpt()` 를 호출하지 않는다 — 그 호출은
        VAD 커밋 경로에만 있다. 그래서 `--no-vad` 로 돌리면 이런 순서가 만들어진다:

            <SEG> 감지 → 슬롯 리셋 → 번역을 백그라운드 태스크로 발사 → (스트림 종료)
            → finish_streaming: 슬롯이 이미 비었으니 flush_uncommitted 는 낼 게 없음
            → 번역 태스크는 뒤늦게 끝나지만 그 시점엔 클라이언트가 다음 발화로 넘어간 뒤

        결과는 **발화 통째 유실**이다. 실측(CoVoST2 200발화, seg 축, 침묵 500ms): 26%가
        `final` 을 한 건도 받지 못했고, 서버 로그에는 `[SEG-IN-TEXT]` 와 `[SEG-SLOT-RESET]` 은
        찍혀 있는데 `[TRANS-SEG-ASYNC]`/`[FINAL]` 이 없다.

        같은 문제가 프로덕션 서버에도 있지만, 거기선 VAD 가 켜져 있어 `_on_vad_commit` 의
        drain 이 대부분 먼저 걷어낸다. 프로덕션 수정은 base `finish_streaming` 맨 앞에
        같은 한 줄을 넣는 것인데, 그건 모든 벤치마크에 영향을 주므로 여기서는 평가 서버만
        고친다.
        """
        # ① 앞 세그먼트의 진행 중인 번역을 먼저 배출한다(세그먼트 순서 보장).
        await self._drain_pending_gpt()

        # 스트림 종료 시 **남은 오디오를 마저 디코딩**한다.
        #
        # base 의 최종 디코딩(`_asr_finish_streaming`)은 VAD 커밋 경로에만 있다. 그래서
        # `--no-vad` 로 돌리면 마지막 미완성 청크에 남은 음성이 **전사조차 되지 않는다** —
        # 커밋이 늦는 게 아니라 텍스트가 존재하지 않는다. 그 상태로 `flush_uncommitted` 가
        # 돌아봐야 `state.text` 에는 마지막으로 완성된 청크까지의 내용만 있다.
        #
        # 규칙 4(스트림 종료 시 보류 경계 전부 확정)는 이미 디코딩된 텍스트의 경계를
        # 확정하는 것이라 미디코딩 구간은 못 살린다 — 최종 디코딩과 역할이 다르다.
        # 여기서 최종 디코딩을 돌려주면 뒤 침묵에 기대지 않고 꼬리를 살릴 수 있고,
        # 이어지는 규칙 4가 그 꼬리를 축의 커밋 경로로 문장 단위 배출한다.
        # A/B 측정용 스위치. AST_NO_FINISH_DECODE=1 로 띄우면 base 와 동일한(수정 전) 동작이
        # 되어 이 디코딩이 있고 없고의 차이를 같은 데이터로 잴 수 있다.
        if os.environ.get("AST_NO_FINISH_DECODE") != "1":
            try:
                await self._asr_finish_streaming(self.active_slot)
            except Exception as exc:
                logger.warning("finish 최종 디코딩 실패: %s", exc)

        await super().finish_streaming()

        # ② 최종 디코딩과 flush 가 **새로** 만든 커밋/번역을 반드시 여기서 배출한다.
        #
        # 이걸 빠뜨리면 최종 디코딩이 감지한 <SEG> 의 번역 태스크가 뜬 채로 finish_done 이
        # 나가고, 클라이언트는 다음 발화로 넘어간다. 번역이 끝난 뒤 emit 되는 final 은
        # **다음 발화의 스트림에 섞여** 나가고(실측 로그: `Received start` 뒤에 이전 발화의
        # [FINAL] 이 찍히며 segment_id/audioStartSec 도 다음 발화 커서를 소비했다),
        # 클라이언트는 그걸 버리므로 발화가 통째로 유실된다.
        #
        # 즉 드레인은 최종 디코딩 **앞뒤 모두** 필요하다. 앞은 순서 보장, 뒤는 유실 방지다.
        await self._drain_pending_gpt()
        await self._drain_deferred_commits()

    async def send_message(self, msg_type: str, **kwargs) -> None:
        # 16병렬 로그는 연결이 섞여 순서를 읽을 수 없다. final 과 finish_done 에만
        # 연결 id 를 붙여, "final 이 그 연결의 finish_done 뒤에 나갔는가"(= 클라이언트가
        # 이미 다음 발화로 넘어간 뒤라 유실)를 단정할 수 있게 한다.
        if msg_type in ("final", "finish_done"):
            conn = getattr(self, "_ast_conn_id", None)
            if conn is None:
                conn = f"c{id(self) % 10000:04d}"
                self._ast_conn_id = conn
            logger.info("[AST-SEND] conn=%s %s text=%r", conn, msg_type,
                        (kwargs.get("original") or "")[:40] if msg_type == "final" else "")

        if msg_type == "final":
            # 커밋 시점에 못 박힌 uttId 가 없으면(가드에 걸려 번역을 안 탄 경로 등)
            # 현재 발화로 채운다. sentAtUttId 는 전송 시점의 발화 — 둘이 다르면 그
            # payload 는 발화 경계를 넘어 늦게 나간 것이다.
            kwargs.setdefault("uttId", self._ast_utt_id)
            kwargs["sentAtUttId"] = self._ast_utt_id
            if kwargs.get("uttId") != kwargs["sentAtUttId"]:
                logger.warning("[AST-LATE] 발화 경계를 넘은 final: commit=%s send=%s text=%r",
                               kwargs.get("uttId"), kwargs["sentAtUttId"],
                               (kwargs.get("original") or "")[:40])

            # 실제 송신 직전 시각. SEG deferred flush 경로와 즉시 전송 경로가 서로 다른
            # 함수라 여기서 한 번에 잡는 게 유일하게 정확하다.
            kwargs["emitElapsedSec"] = round(self._stream_elapsed_sec(), 4)
            kwargs["audioReceivedSec"] = round(self.current_time, 3)

            # 불변식: 오디오 T 지점까지의 내용을 담은 세그먼트를 T 이전에 결정할 수는 없다.
            # SEG 커밋은 payload 를 만든 뒤 deferred 큐에 넣고, 나중에 토큰비율로
            # audioEndSec 를 보정해서 내보낸다. 그래서 생성 시점에 찍은 decisionAudioSec 가
            # 보정된 audioEndSec 보다 앞서는 경우가 생긴다(실측: dec=6.40 인데 audioEnd=12.00).
            # 그대로 두면 그 세그먼트의 지연이 실제보다 작게 잡혀 LAAL 이 낙관적으로 나온다.
            decision = kwargs.get("decisionAudioSec")
            audio_end = kwargs.get("audioEndSec")
            if decision is not None and audio_end is not None and audio_end > decision:
                kwargs["decisionAudioSecRaw"] = decision  # 감사를 위해 원값 보존
                kwargs["decisionAudioSec"] = round(float(audio_end), 3)
        elif msg_type == "hello":
            kwargs["message"] = "Qwen3-ASR Streaming Server (AST)"
            cfg = dict(kwargs.get("serverConfig") or {})
            cfg["server"] = "ast"
            kwargs["serverConfig"] = cfg
        await super().send_message(msg_type, **kwargs)


class ASTStreamingServer(fsl_server.FSLStreamingServer):
    """FSL 서버 + static/punct 축에서 `<SEG>` 를 **서버 눈에만** 안 보이게 한다."""

    def _ast_hide_seg_token(self) -> None:
        """seg 축이 아니면 `<SEG>` 를 파싱 단계에서 걷어낸다. **생성은 건드리지 않는다.**

        왜 필요한가
        -----------
        세 축을 같은 가중치(en-dailytalk-seg)로 돌린다. 그 모델은 축과 무관하게 항상
        `<SEG>` 를 뱉으므로, static/punct 축이 그걸 무시하지 못하면 축이 조용히 섞인다.
        실측(20발화, punct 축): 커밋 사유가 `{'dot': 21, 'seg': 12}` — 36% 가 seg 였다.

        단순 라벨 문제가 아니다. base 는 커밋 구간에 `<SEG>` 가 섞이면

            trigger = "seg" if "<SEG>" in matched_text else "dot"
            ...
            if trigger == "dot" and self.dot_commit_confirm:   # ← 확정 게이트

        라벨을 seg 로 바꾸면서 **확정 게이트를 건너뛴다.** 그 커밋만 확정 없이 즉시
        나가 punct 축의 LAAL 이 낙관적으로 잡힌다. FSL 도 `"<SEG>" in state.text` 만
        보고 `_slot_seg_detected` 를 채우므로 dot 커밋에 SEG 감지 시점이 새어 든다.

        왜 생성을 막으면 안 되나
        ------------------------
        처음엔 vLLM `bad_words=["<SEG>"]` 로 생성을 막았는데 **전사가 망가졌다.**
        모델이 `<SEG>` 에 둔 확률 질량이 차선 토큰으로 흘러 무의미한 단어가 끼어든다.
        실측(같은 20발화, en→de, 정답 대비 WER):

            seg   (차단 없음)  WER  9.26%   'icky' 0회
            punct (차단)       WER 11.73%   'icky' 12회 / 7발화
            static(차단)       WER 29.63%   'icky' 18회 / 10발화

        정책과 무관한 ASR 열화를 static/punct 에만 씌우는 셈이라, seg 가 실제보다
        좋아 보이게 만드는 반대 방향의 교란이 된다.

        어떻게 하나
        -----------
        `parse_asr_output` 을 감싼다. 디코딩된 원문(`state._raw_decoded`)은 그대로 두고,
        **서버가 받아보는 텍스트에서만** `<SEG>` 를 지운다. 이 한 지점이 전부를 덮는다:

            state.text          ← txt_p 가 여기 대입된다 (FSL 의 SEG 감지도 이걸 본다)
            on_seg 콜백          ← `txt_p.count("<SEG>")` 로 발동하므로 0 이 되어 안 뜬다
            커밋 trigger 판정     ← matched_text 에 `<SEG>` 가 없어 항상 "dot"

        결과적으로 세 축이 **완전히 동일한 ASR 출력**을 공유하고 정책만 갈린다.
        """
        if not (self.config.always_commit or self.config.enable_dot_commit):
            return                      # seg 축 — 그대로 둔다
        if getattr(self, "_ast_seg_hidden", False):
            return
        # `parse_asr_output` 은 qwen3_asr 모듈에 이름으로 바인딩돼 호출된다. 그 모듈의
        # 속성을 갈아끼워야 호출부가 우리 래퍼를 본다.
        mod = sys.modules.get(type(self.asr).__module__)
        orig = getattr(mod, "parse_asr_output", None) if mod else None
        if orig is None:
            logger.error("[AST-HIDE-SEG] parse_asr_output 을 찾지 못했습니다 — "
                         "static/punct 축이 <SEG> 에 오염된 채 돌 수 있습니다 (모듈=%s)",
                         getattr(mod, "__name__", None))
            return

        def _hide_seg(*args, **kwargs):
            lang, txt = orig(*args, **kwargs)
            if txt and "<SEG>" in txt:
                txt = re.sub(r"\s*<SEG>\s*", " ", txt).strip()
            return lang, txt

        mod.parse_asr_output = _hide_seg
        self._ast_seg_hidden = True
        logger.info("[AST-HIDE-SEG] static/punct 축 — 파싱 단계에서 `<SEG>` 제거 "
                    "(생성·디코딩은 그대로, 모듈=%s)", mod.__name__)

    async def handle_connection(self, websocket):
        self._ast_hide_seg_token()
        async with self.connection_lock:
            self.active_connections += 1
            if self.idle_task and not self.idle_task.done():
                self.idle_task.cancel()
            logger.info("Client connected (%s)", self.active_connections)

        try:
            handler = ASTStreamingHandler(
                websocket,
                self.asr,
                self.config,
                self.pairing_hub,
                http_session=self._http_session,
                vad_model_bytes=self.vad_model_bytes,
                corrector=self.corrector,
                gpt_translator=self.gpt_translator,
            )
            await handler.handle()
        finally:
            async with self.connection_lock:
                self.active_connections -= 1
                logger.info("Client disconnected (%s)", self.active_connections)
                if self.active_connections == 0:
                    self._restart_idle_timer()


def _install_trans_guard() -> None:
    """AST 전용 인자를 sys.argv 에서 걷어내고 번역 계측을 설치한다.

    FSL 의 `main()` 이 argparse 를 쥐고 있으므로 여기서 먼저 떼어내야 한다
    (FSL 자신도 `parse_known_args` 로 같은 일을 한다).
    """
    import argparse as _argparse

    pre = _argparse.ArgumentParser(add_help=False)
    pre.add_argument("--trans-backend", default="v2", choices=["v2", "gtx"],
                     help="v2=공식 Cloud Translation Basic(API 키 필요, 기본값), "
                          "gtx=무료 위젯 엔드포인트(대량 호출 시 IP 차단됨)")
    pre.add_argument("--trans-api-key", default=None,
                     help="미지정 시 GOOGLE_TRANSLATE_API_KEY 환경변수를 쓴다")
    pre.add_argument("--trans-retries", type=int, default=3,
                     help="번역 총 시도 횟수(첫 시도 포함). 1이면 재시도 없음")
    pre.add_argument("--trans-timeout", type=float, default=10.0)
    pre.add_argument("--trans-backoff", type=float, default=0.5,
                     help="재시도 지수 백오프 기준(초)")
    pre.add_argument("--trans-backoff-429", type=float, default=5.0,
                     help="429/403 일 때의 백오프 기준(초). rate-limit 은 더 물러선다")
    pre.add_argument("--trans-alert-rate", type=float, default=0.005,
                     help="번역 실패율이 이 값을 넘으면 CRITICAL 로 경보")
    pre.add_argument("--trans-alert-min-calls", type=int, default=200)
    pre.add_argument("--trans-dump-every", type=int, default=50,
                     help="이 호출 수마다 통계 파일을 갱신한다(중간에 죽어도 남게)")
    pre.add_argument("--trans-stats-out", default=None,
                     help="번역 통계 JSON 을 쓸 경로. 주기적으로 갱신된다")
    args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # `stop_server.sh` 는 SIGTERM 을 보낸다. 파이썬의 기본 SIGTERM 처리는 프로세스를
    # 즉시 끝내고 **finally 블록을 실행하지 않으므로**, 아래 main() 의 finally 에 있는
    # 통계 기록이 통째로 날아간다. 축마다 서버를 내리는 실험에서는 그게 곧 "이 런을
    # 믿어도 되는가"의 근거가 사라진다는 뜻이다.
    #
    # 핸들러에서 기록만 하고 기본 동작으로 되돌린 뒤 같은 시그널을 자신에게 다시
    # 보낸다 — vLLM 종료 경로를 바꾸지 않으면서 통계만 건진다.
    import signal as _signal

    def _on_term(signum, _frame):
        trans_guard.log_summary(f"[TRANS-STATS/sig{signum}]")
        trans_guard.dump()
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for _sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(_sig, _on_term)
        except (ValueError, OSError):
            pass  # 메인 스레드가 아닌 경우 등 — 기록을 못 할 뿐 런은 계속된다

    # 정상 종료 경로용 보조. 시그널 경로와 중복 기록돼도 같은 내용이라 무해하다.
    import atexit as _atexit
    _atexit.register(trans_guard.dump)

    api_key = args.trans_api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
    trans_guard.install(
        fsl_server.base_server,
        backend=args.trans_backend,
        api_key=api_key,
        retries=args.trans_retries,
        timeout=args.trans_timeout,
        backoff=args.trans_backoff,
        backoff_429=args.trans_backoff_429,
        alert_rate=args.trans_alert_rate,
        alert_min_calls=args.trans_alert_min_calls,
        stats_path=args.trans_stats_out,
        dump_every=args.trans_dump_every,
    )


def main():
    # FSL main() 의 인자 파싱 + StreamingConfig 구성을 그대로 쓴다. 서버 클래스만
    # 갈아끼우는 이유는 중복 때문이다 — config 필드를 복사해두면 base 에 플래그가
    # 하나 늘 때마다 이 파일이 조용히 낡는다.
    fsl_server.FSLStreamingServer = ASTStreamingServer
    _install_trans_guard()
    try:
        fsl_server.main()
    finally:
        # 서버가 어떻게 끝나든(정상 종료, Ctrl-C, 예외) 번역 통계는 남겨야 한다.
        # 이게 없으면 "이 런을 믿어도 되는가"를 나중에 판정할 근거가 사라진다.
        trans_guard.log_summary("[TRANS-STATS/final]")
        trans_guard.dump()


if __name__ == "__main__":
    main()
