"""출력 헤더(``language X<asr_text>``)의 언어 이름을 허용 집합으로 제한한다.

**왜 토큰을 막는 방식으로는 안 되나.** 허용되지 않은 언어 이름의 *토큰* 에 -100 을
주는 게 자연스러워 보이지만, 모델은 같은 글자를 다른 토큰 조합으로 쓸 수 있다.
실측(Qwen3-ASR-1.7B, 한국어 오디오, 허용=['Spanish'])::

    제한 없음   : 'language'  ' Korean'(16134)          <asr_text> …
    토큰 -100   : 'language'  ' K'(730) 'orean'(45195)  <asr_text> …
    이 처리기   : 'language'  ' Spanish'(15154)         <asr_text> …

텍스트는 글자 하나 다르지 않은 ``language Korean`` 이라 파서는 그대로 Korean 으로
읽는다. 토큰을 더 막는 식으로는 못 이긴다 — 조합이 여럿이고, ``" K"`` 를 전역으로
막으면 영어 전사의 ``Korea``·``Kim`` 까지 망가진다.

**그래서 토큰이 아니라 글자로 막는다.** 지금까지 생성된 헤더 바이트를 보고, 후보
토큰을 붙였을 때 여전히 ``language <허용 이름>`` 의 접두사인지만 본다. 어떤 식으로
쪼개 넣든 결과 문자열이 허용 이름이 아니면 그 자리에서 걸린다.

제한은 **헤더까지만** 이다. 언어 이름이 완성되면 마스크를 풀어 전사 본문은 손대지
않는다. 조용한 구간에 나오는 ``language None<asr_text>`` 도 살려 둔다 — 막으면 무음에
억지로 언어를 붙이게 된다.

**태그를 고정할 뿐 전사 문자를 강제하지는 않는다.** 스페인어 발화를 한글로 받아쓰는
문제는 이걸로 안 없어진다.

쓰는 법 — 엔진에 등록하고, 요청마다 허용 언어를 실어 보낸다::

    AsyncEngineArgs(model=..., logits_processors=[
        "qwen_asr.inference.lang_header_lock:LanguageHeaderProcessor"])   # 점이 아니라 콜론
    SamplingParams(..., extra_args={"allowed_languages": ["Spanish"]})

``extra_args`` 가 없는 요청은 건드리지 않는다. 평가 서버처럼 제한이 필요 없는 경로는
그대로 지나간다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# 헤더 앞부분은 모델이 항상 이 형태로 낸다. 이름이 끝나면 <asr_text> 가 따라온다.
_HEADER = b"language "
# 무음일 때 나오는 이름. 허용 목록과 무관하게 열어 둔다.
_ALWAYS_OK = ("None",)


class LanguageHeaderProcessor(LogitsProcessor):
    """헤더의 언어 이름이 허용 집합에 들어가도록 글자 단위로 제한한다."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        self.device = device
        # 요청 인덱스 -> (허용 이름 튜플, 생성 토큰 리스트 참조)
        self.rows: dict[int, tuple[tuple[str, ...], list[int]]] = {}
        self._tok = None
        self._token_bytes: list[bytes] = []
        self._model = vllm_config.model_config.tokenizer or vllm_config.model_config.model
        # (접두사 바이트, 허용 이름 튜플) -> 통과 토큰 id 텐서
        self._cache: dict[tuple[bytes, tuple[str, ...]], torch.Tensor] = {}

    # ── 토크나이저는 처음 쓸 때 한 번만 올린다 ────────────────────────────────
    def _ensure_tokenizer(self) -> None:
        if self._tok is not None:
            return
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self._model)
        # 토큰 하나하나의 바이트열. 헤더 비교는 **바이트로** 해야 한다 — byte-level
        # BPE 라 한 토큰이 UTF-8 글자 중간에서 끊길 수 있고, 문자열로 비교하면
        # 그 자리에서 잘못 걸러진다.
        vocab_size = len(self._tok)
        self._token_bytes = [
            self._tok.decode([i], skip_special_tokens=False).encode("utf-8", "ignore")
            for i in range(vocab_size)
        ]

    def is_argmax_invariant(self) -> bool:
        """마스킹은 argmax 를 바꾼다 — 그게 목적이다."""
        return False

    # ── 배치 상태 추적 ────────────────────────────────────────────────────────
    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        for idx in batch_update.removed:
            self.rows.pop(idx, None)
        for idx, params, _prompt_ids, out_ids in batch_update.added:
            allowed = (getattr(params, "extra_args", None) or {}).get("allowed_languages")
            if allowed:
                self.rows[idx] = (tuple(allowed) + _ALWAYS_OK, out_ids)
            else:
                # 같은 인덱스를 재사용하는 경우가 있으므로 반드시 지운다.
                self.rows.pop(idx, None)
        for i1, i2, direction in batch_update.moved:
            a, b = self.rows.pop(i1, None), self.rows.pop(i2, None)
            # UNIDIRECTIONAL 은 i1 -> i2 이동, SWAP 은 맞바꿈이다.
            if b is not None and str(direction).endswith("SWAP"):
                self.rows[i1] = b
            if a is not None:
                self.rows[i2] = a

    # ── 통과 토큰 계산 ────────────────────────────────────────────────────────
    def _targets(self, allowed: tuple[str, ...]) -> list[bytes]:
        return [_HEADER + name.encode("utf-8") for name in allowed]

    def _allowed_ids(self, prefix: bytes, allowed: tuple[str, ...]) -> Optional[torch.Tensor]:
        """``prefix`` 다음에 와도 되는 토큰 id. None 이면 제한하지 않는다."""
        key = (prefix, allowed)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        targets = self._targets(allowed)
        live = [t for t in targets if t.startswith(prefix)]
        if not live or prefix in targets:
            # 이름이 완성됐거나 헤더 밖으로 나갔다 — 더 볼 것이 없다.
            self._cache[key] = None
            return None
        ids = [i for i, b in enumerate(self._token_bytes)
               if b and any(t.startswith(prefix + b) for t in live)]
        out = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._cache[key] = out
        return out

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.rows:
            return logits
        self._ensure_tokenizer()
        done: list[int] = []
        for row, (allowed, out_ids) in self.rows.items():
            if row >= logits.shape[0]:
                continue
            prefix = b"".join(self._token_bytes[t] for t in out_ids if t < len(self._token_bytes))
            ids = self._allowed_ids(prefix, allowed)
            if ids is None:
                # 헤더가 끝났다. 행을 빼야 다음 스텝부터 접두사를 다시 만들지 않는다
                # — 안 빼면 생성이 길어질수록 매 스텝 비용이 함께 는다.
                done.append(row)
                continue
            keep = logits[row, ids].clone()
            logits[row].fill_(float("-inf"))
            logits[row, ids] = keep
        for row in done:
            self.rows.pop(row, None)
        return logits
