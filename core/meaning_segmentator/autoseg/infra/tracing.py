"""LangSmith 연동 — **선택**. 키가 없으면 통째로 no-op 이 된다.

## 왜 필요했나

`Usage` 는 런 전체 합계만 들고 있어서 "어디에 썼는가"가 안 나온다. run05 에서 비용
병목을 찾을 때 호출 수 비율로 역산해야 했고(분절 호출이 78~90%), 그건 호출당 비용이
균일하다는 가정 위에서만 맞다 — 실제로는 분절(사고 3,700토큰)과 판정자·Critic 이
자릿수로 다르다. 그래서 **호출마다 용도(`purpose`) 라벨**을 붙인다.

라벨은 두 곳에 쌓인다.

1. `Usage.by_purpose` — 의존성 없이 항상 켜져 있다. 런 끝에 표로 찍힌다.
2. LangSmith — `LANGSMITH_API_KEY` 가 있을 때만. 호출 단위 추적·비용 집계·시계열.

## 설계 원칙 — 관측이 런을 죽이면 안 된다

추적 실패(네트워크·인증·스키마)는 **전부 삼킨다.** 6시간짜리 런이 관측 코드 때문에
죽는 것이 관측을 못 하는 것보다 나쁘다. 그래서 이 모듈의 공개 함수는 예외를 밖으로
내보내지 않으며, 실패 시 조용히 null 핸들로 물러난다(최초 1회만 경고).
"""

from __future__ import annotations

import os
import sys
import time

_WARNED = False


def _warn_once(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        print(f"[tracing] {msg}", file=sys.stderr)


def enabled() -> bool:
    """`LANGSMITH_API_KEY` 가 있고 `langsmith` 가 import 되면 켠다.

    `LANGSMITH_TRACING=0` 으로 명시적으로 끌 수 있다 — 키를 지우지 않고 비교 런을
    돌려야 할 때 쓴다.
    """
    if os.environ.get("LANGSMITH_TRACING", "1").lower() in ("0", "false", "no"):
        return False
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        return False
    try:
        import langsmith  # noqa: F401
    except ImportError:
        _warn_once("LANGSMITH_API_KEY 는 있는데 langsmith 가 설치되지 않았다 — 추적 끔")
        return False
    return True


def project_name() -> str:
    return (os.environ.get("LANGSMITH_PROJECT")
            or os.environ.get("LANGCHAIN_PROJECT") or "autoseg")


class _NullRun:
    """추적이 꺼졌을 때의 핸들. 호출부가 분기하지 않아도 되게 한다."""

    def finish(self, payload: dict | None = None, error: str | None = None) -> None:
        pass


class _LangSmithRun:
    def __init__(self, rt, model: str, started: float):
        self._rt = rt
        self._model = model
        self._started = started

    def finish(self, payload: dict | None = None, error: str | None = None) -> None:
        try:
            if error is not None:
                self._rt.end(error=error)
            else:
                payload = payload or {}
                u = payload.get("usage") or {}
                prompt = u.get("prompt_tokens", 0) or 0
                completion = u.get("completion_tokens", 0) or 0
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                reasoning = ((u.get("completion_tokens_details") or {})
                             .get("reasoning_tokens", 0) or 0)
                choice = (payload.get("choices") or [{}])[0]
                # LangSmith 는 `usage_metadata` 로 토큰을 읽고 모델명으로 단가를 붙인다.
                # 사고 토큰은 output_token_details.reasoning 자리에 넣어야 출력 토큰과
                # 따로 보인다 — 이 런의 비용은 사실상 전부 여기서 나온다.
                self._rt.end(outputs={
                    "content": (choice.get("message") or {}).get("content"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage_metadata": {
                        "input_tokens": prompt,
                        "output_tokens": completion,
                        "total_tokens": prompt + completion,
                        "input_token_details": {"cache_read": cached},
                        "output_token_details": {"reasoning": reasoning},
                    },
                })
            self._rt.patch()
        except Exception as e:                                   # noqa: BLE001
            _warn_once(f"LangSmith 기록 실패(무시하고 계속): {e}")


def start_llm_run(purpose: str, model: str, system: str, user: str,
                  metadata: dict | None = None):
    """LLM 호출 1건의 추적을 시작한다. 반환 핸들에 `.finish(payload=|error=)`.

    실패하면 `_NullRun` 을 준다 — 호출부는 성공 여부를 알 필요가 없다.
    """
    if not enabled():
        return _NullRun()
    try:
        from langsmith.run_trees import RunTree

        rt = RunTree(
            name=purpose,
            run_type="llm",
            project_name=project_name(),
            tags=[purpose, model],
            inputs={"messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}]},
            extra={"metadata": {"ls_provider": "openai", "ls_model_name": model,
                                "purpose": purpose, **(metadata or {})}},
        )
        rt.post()
        return _LangSmithRun(rt, model, time.time())
    except Exception as e:                                       # noqa: BLE001
        _warn_once(f"LangSmith 시작 실패(추적 없이 계속): {e}")
        return _NullRun()
