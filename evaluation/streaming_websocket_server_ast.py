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

    async def _translate(self, text, target_lang, audio_end_sec=None):
        utt_id = self._ast_utt_id
        translation, lang, extra = await super()._translate(text, target_lang, audio_end_sec)
        extra["uttId"] = utt_id
        return translation, lang, extra

    async def _correct_and_translate(self, text, current_lang, audio_end_sec):
        utt_id = self._ast_utt_id
        corrected, translation, lang, extra = await super()._correct_and_translate(
            text, current_lang, audio_end_sec
        )
        extra["uttId"] = utt_id
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
        # dot 축은 이 문제를 규칙 4(스트림 종료 시 보류 경계 전부 확정)로 우회하지만,
        # 그건 이미 디코딩된 텍스트의 경계를 확정하는 것이라 미디코딩 구간은 못 살린다.
        # 여기서 최종 디코딩을 돌려주면 seg 축도 뒤 침묵에 기대지 않고 꼬리를 살릴 수 있고,
        # 이어지는 flush 가 문장 경계로 쪼개 커밋하므로 커밋 격리도 유지된다.
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
    async def handle_connection(self, websocket):
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


def main():
    # FSL main() 의 인자 파싱 + StreamingConfig 구성을 그대로 쓴다. 서버 클래스만
    # 갈아끼우는 이유는 중복 때문이다 — config 필드를 복사해두면 base 에 플래그가
    # 하나 늘 때마다 이 파일이 조용히 낡는다.
    fsl_server.FSLStreamingServer = ASTStreamingServer
    fsl_server.main()


if __name__ == "__main__":
    main()
