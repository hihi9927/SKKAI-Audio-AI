"""Letsur AI Gateway 클라이언트.

OpenAI 호환 엔드포인트(https://gw.letsur.ai/v1)를 httpx로 직접 호출한다.
SDK를 쓰지 않는 이유: 게이트웨이가 응답에 실어 보내는 estimated_cost 필드를
그대로 읽어 런 전체 비용을 집계하기 위함.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "https://gw.letsur.ai/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_api_key(env_path: Path | None = None) -> str:
    """.env 에서 CLAUDE_API_KEY 를 읽는다. 환경변수가 있으면 그쪽이 우선."""
    key = os.environ.get("LETSUR_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if key:
        return key
    env_path = env_path or (_REPO_ROOT / ".env")
    if not env_path.exists():
        raise RuntimeError(f"API 키를 찾을 수 없습니다. {env_path} 또는 CLAUDE_API_KEY 환경변수 필요.")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in ("CLAUDE_API_KEY", "LETSUR_API_KEY"):
            return v.strip().strip('"').strip("'")
    raise RuntimeError(f"{env_path} 에 CLAUDE_API_KEY 가 없습니다.")


@dataclass
class Usage:
    """런 전체 누적 사용량. 스레드 안전."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    truncated: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, payload: dict) -> None:
        u = payload.get("usage") or {}
        c = payload.get("estimated_cost") or {}
        with self._lock:
            self.calls += 1
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)
            self.cached_tokens += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            if (payload.get("choices") or [{}])[0].get("finish_reason") == "length":
                self.truncated += 1
            try:
                self.cost += float(c.get("amount", 0) or 0)
            except (TypeError, ValueError):
                pass

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens,
                "truncated": self.truncated,
                "cost": round(self.cost, 6),
            }


class BudgetExceeded(RuntimeError):
    pass


class Gateway:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-large",
        budget: float | None = None,
        timeout: float = 420.0,   # thinking 모델 + max_tokens 12000 인 PE 호출이 180s 를 넘겼다
        max_retries: int = 5,
    ):
        self.api_key = api_key or load_api_key()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.budget = budget
        self.max_retries = max_retries
        self.usage = Usage()
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        )

    def close(self) -> None:
        self._client.close()

    # ── 저수준 ───────────────────────────────────────────────────────────
    def _post(self, path: str, body: dict) -> dict:
        if self.budget is not None and self.usage.cost >= self.budget:
            raise BudgetExceeded(f"예산 초과: {self.usage.cost:.4f} >= {self.budget}")

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(f"{self.base_url}{path}", json=body)
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    continue
                r.raise_for_status()
                payload = r.json()
                self.usage.add(payload)
                return payload
            except httpx.HTTPError as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"게이트웨이 호출 실패({self.max_retries}회 재시도): {last_err}")

    # ── 채팅 ─────────────────────────────────────────────────────────────
    def chat(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        body = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        payload = self._post("/chat/completions", body)
        choice = payload["choices"][0]
        out = (choice["message"].get("content") or "").strip()
        # thinking 모델은 사고 토큰도 max_tokens 에 함께 잡힌다. 예산을 사고에 다 쓰면
        # finish_reason=length 인 채 content 가 비어서 돌아오는데, 그대로 "" 를 반환하면
        # 하위에서 "모델이 텍스트를 고쳐 씀"(text_modified) 으로 오진된다 — run04 iter0 에서
        # 긴 문장 6/60 이 이 경로로 빈 출력이 났고, 원인이 로그에 전혀 남지 않았다.
        if not out and choice.get("finish_reason") == "length":
            print(f"[gateway] 경고: max_tokens={max_tokens} 안에서 출력이 끊겼다 "
                  f"(사고 토큰이 전부 소진). 입력 {len(user)}자", file=sys.stderr)
        return out

    def chat_json(self, system: str, user: str, **kw) -> dict:
        """JSON 응답을 기대하는 호출.

        에이전트 출력에는 원문(따옴표·인용부호 포함)이 그대로 들어가므로 이스케이프가
        깨지는 경우가 실제로 발생한다. 파싱 실패 시 깨진 출력을 모델에 되돌려
        한 번 복구시킨다."""
        raw = self.chat(system, user, **kw)
        try:
            return parse_json_loose(raw)
        except (ValueError, json.JSONDecodeError) as e:
            repaired = self.chat(
                "You repair malformed JSON. Return ONLY the corrected JSON object. "
                "Preserve all content; fix only syntax (unescaped quotes, missing commas, "
                "trailing commas, unterminated strings).",
                f"Parse error: {e}\n\n=== MALFORMED JSON ===\n{raw}",
                **{**kw, "max_tokens": kw.get("max_tokens", 8000)},
            )
            return parse_json_loose(repaired)

    # ── 임베딩 ───────────────────────────────────────────────────────────
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            chunk = texts[i : i + 64]
            payload = self._post("/embeddings", {"model": self.embed_model, "input": chunk})
            out.extend(d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"]))
        return out


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json_loose(raw: str) -> dict:
    s = _FENCE.sub("", raw).strip()
    for candidate in (s, s[s.find("{") : s.rfind("}") + 1] if "{" in s and "}" in s else ""):
        if not candidate:
            continue
        for strict in (True, False):
            try:
                return json.loads(candidate, strict=strict)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"JSON 파싱 실패: {raw[:300]}")
