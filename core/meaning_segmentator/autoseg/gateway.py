"""LLM 게이트웨이 클라이언트 — Letsur AI Gateway / OpenAI 양쪽.

OpenAI 호환 엔드포인트를 httpx로 직접 호출한다. SDK를 쓰지 않는 이유: Letsur가
응답에 실어 보내는 estimated_cost 필드를 그대로 읽어 런 전체 비용을 집계하기 위함.

**OpenAI는 estimated_cost를 주지 않는다.** 그쪽으로 붙을 때는 _PRICES 표로 직접
계산한다 — 이게 없으면 `usage.cost`가 0에 고정되고 `--budget` 예산 가드가 영영
발동하지 않는다(설계상 유일한 비용 상한이다).

두 백엔드의 실측 차이 (2026-08, gpt-5.4-mini):

    max_tokens             → HTTP 400. gpt-5 계열은 max_completion_tokens 만 받는다
    temperature=0.0        → 정상
    cache_control 마커     → 무해하게 무시된다(200). OpenAI는 1024토큰 이상 자동 캐싱이라
                             마커 없이도 prompt_tokens_details.cached_tokens 가 잡힌다
    estimated_cost         → 없음 (위 참조)
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

from . import tracing

LETSUR_BASE_URL = "https://gw.letsur.ai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
LOCAL_BASE_URL = "http://localhost:11434/v1"   # ollama 기본 포트
_REPO_ROOT = Path(__file__).resolve().parents[3]

# `chat(reasoning_effort=...)` 의 기본값. `None` 은 "파라미터를 아예 빼라"(=모델 기본
# 사고량)라는 **명시적 지시**이므로, "지정 안 함"과 구분할 센티널이 따로 필요하다.
_INHERIT = object()

# ── 프로바이더 ────────────────────────────────────────────────────────────
# **어디로 붙을지는 `--provider` 하나가 정한다.** 종전에는 환경변수를 순서대로 뒤져
# (`LETSUR_API_KEY` → `OPENAI_API_KEY` → `CLAUDE_API_KEY`) 먼저 잡히는 키로 엔드포인트를
# 역추론하고, `AUTOSEG_BASE_URL` 이 있으면 그걸로 덮었다. 문제가 셋이었다:
#
#   - `.env` 에 키가 둘 이상이면 **명령줄만 봐서는 어디로 갔는지 알 수 없다.** 이 레포의
#     `.env` 가 실제로 그렇다 (LETSUR + OPENAI) — 늘 앞의 것이 이겼다.
#   - 키를 하나 **추가**한 것만으로 다음 런의 상대가 조용히 바뀐다.
#   - `config.json` 에 안 남아 사후 확인이 불가능했다.
#
# 이제 프로바이더가 키 환경변수·엔드포인트·API 방언을 함께 정하고, 그 값이 런 기록에
# 남는다. 폴백은 없다 — 지정한 프로바이더의 키가 없으면 그냥 죽는다.


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_env: str | None      # None = 키가 필요 없다 (로컬 서버)
    # OpenAI 방언인가. `max_completion_tokens` 사용 여부와 `_PRICES` 기반 비용 계산이
    # 여기 걸린다. Letsur 는 `estimated_cost` 를 응답에 실어 주므로 표가 필요 없고,
    # 로컬 서버는 `max_tokens` 를 받고 비용이 0 이다.
    openai_dialect: bool


PROVIDERS: dict[str, Provider] = {
    "letsur": Provider(LETSUR_BASE_URL, "LETSUR_API_KEY", False),
    "openai": Provider(OPENAI_BASE_URL, "OPENAI_API_KEY", True),
    # ollama 등 OpenAI 호환 로컬 서버. 포트가 다르면 `--base-url` 로 바꾼다.
    "local": Provider(LOCAL_BASE_URL, None, False),
}
DEFAULT_PROVIDER = "letsur"


def load_api_key(key_env: str, env_path: Path | None = None) -> str:
    """`key_env` 환경변수 > 레포 루트 `.env` 의 **같은 이름**. 다른 이름은 안 본다."""
    if os.environ.get(key_env):
        return os.environ[key_env]
    env_path = env_path or (_REPO_ROOT / ".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key_env:
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    raise RuntimeError(f"{key_env} 를 찾을 수 없습니다 — 환경변수 또는 {env_path} 에 넣으세요.")


def add_provider_args(p) -> None:
    """`--provider` / `--base-url`. Gateway 를 만드는 CLI 는 전부 이걸 쓴다 —
    한 군데서 붙여야 인자 이름과 기본값이 갈라지지 않는다."""
    p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PROVIDERS),
                   help="LLM 엔드포인트. 키 환경변수도 이게 정한다 "
                        "(letsur=LETSUR_API_KEY, openai=OPENAI_API_KEY, local=키 불필요). "
                        f"기본 {DEFAULT_PROVIDER}")
    p.add_argument("--base-url", default=None,
                   help="프로바이더 기본 엔드포인트 대신 쓸 URL (로컬 서버 포트 등)")


# OpenAI 단가 (USD / 1M 토큰, 2026-08 developers.openai.com/api/docs/pricing).
# (입력, 캐시된 입력, 출력). 접두사 일치라 날짜 붙은 변종도 잡힌다.
#
# **이 표가 예산 가드의 유일한 근거다.** Letsur 는 estimated_cost 를 응답에 실어 주지만
# OpenAI 는 안 준다 — 표에 없는 모델을 쓰면 cost 가 0 으로 고정되어 `--budget` 이
# 무력화되므로, 미등록 모델은 생성 시점에 경고한다.
_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "o4-mini": (1.10, 0.275, 4.40),
    "o3-mini": (1.10, 0.55, 4.40),
}


def _price_of(model: str) -> tuple[float, float, float] | None:
    best: tuple[str, tuple[float, float, float]] | None = None
    for name, p in _PRICES.items():
        if model.startswith(name) and (best is None or len(name) > len(best[0])):
            best = (name, p)
    return best[1] if best else None


# 프롬프트 캐싱 최소 길이. Anthropic 계열은 일정 토큰 미만이면 캐시 블록을 만들지 않는다
# (sonnet 기준 1024 토큰). 짧은 판정자·비평가 시스템 프롬프트에까지 마커를 붙여도
# 캐시가 안 잡히므로, 반복 호출로 이득이 나는 긴 프롬프트에만 붙인다.
# 한글은 토큰당 글자수가 적어 보수적으로 잡았다 — 분절 프롬프트는 8391자 = 3914토큰이었다.
_CACHE_MIN_CHARS = 3000


def _cacheable(system: str):
    """긴 시스템 프롬프트에 `cache_control` 을 붙여 반복 호출의 입력 과금을 없앤다.

    Letsur(Anthropic 계열)에서 캐싱은 **명시적 opt-in** 이다 — 마커가 없으면 아무리 같은
    프롬프트를 반복해도 `cached_tokens` 는 0 이다. 실측(분절 프롬프트 8391자, 같은 문장
    8건): 신규 입력 31,818 → 506 토큰. **다만 총비용은 10.9% 감소에 그쳤다** — 사고 토큰이
    문장당 ~2,900 이라 출력 쪽이 비용을 지배하기 때문이다. 캐싱은 부수적 절감이지
    이 루프의 비용 구조를 바꾸지는 않는다.

    **OpenAI 에서는 이 마커가 필요 없다** — 1024토큰 이상 프리픽스를 자동 캐싱한다.
    실측에서 마커를 붙이든(D) 안 붙이든(E) 동일하게 cached=1280 이 잡혔고, 붙여도
    거부되지 않는다. 그래서 분기 없이 그대로 둔다 — Letsur 로 되돌아갈 때 필요하다.
    """
    if len(system) < _CACHE_MIN_CHARS:
        return system
    return [{"type": "text", "text": system,
             "cache_control": {"type": "ephemeral"}}]


@dataclass
class Usage:
    """런 전체 누적 사용량. 스레드 안전."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    truncated: int = 0
    # OpenAI 처럼 estimated_cost 를 안 주는 백엔드에서 쓰는 단가 (입력, 캐시입력, 출력).
    price: tuple[float, float, float] | None = None
    # 용도별 집계 — "어디에 썼는가". 합계만으로는 병목이 안 보인다 (tracing.py 참조).
    by_purpose: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, payload: dict, purpose: str = "other") -> None:
        u = payload.get("usage") or {}
        c = payload.get("estimated_cost") or {}
        prompt = u.get("prompt_tokens", 0) or 0
        completion = u.get("completion_tokens", 0) or 0
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        reasoning = ((u.get("completion_tokens_details") or {})
                     .get("reasoning_tokens", 0) or 0)
        with self._lock:
            before_cost = self.cost
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cached_tokens += cached
            if (payload.get("choices") or [{}])[0].get("finish_reason") == "length":
                self.truncated += 1
            try:
                amount = float(c.get("amount", 0) or 0)
            except (TypeError, ValueError):
                amount = 0.0
            if amount:
                self.cost += amount
            elif self.price is not None and "choices" in payload:
                # 임베딩 응답은 단가가 다르므로 제외한다 ("choices" 가 없다).
                p_in, p_cached, p_out = self.price
                self.cost += ((prompt - cached) * p_in + cached * p_cached
                              + completion * p_out) / 1_000_000
            delta_cost = self.cost - before_cost
            b = self.by_purpose.setdefault(
                purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                          "reasoning_tokens": 0, "cached_tokens": 0, "cost": 0.0})
            b["calls"] += 1
            b["prompt_tokens"] += prompt
            b["completion_tokens"] += completion
            b["reasoning_tokens"] += reasoning
            b["cached_tokens"] += cached
            b["cost"] += delta_cost

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens,
                "truncated": self.truncated,
                "cost": round(self.cost, 6),
                "by_purpose": {k: {**v, "cost": round(v["cost"], 6)}
                               for k, v in sorted(self.by_purpose.items(),
                                                  key=lambda kv: -kv[1]["cost"])},
            }


class BudgetExceeded(RuntimeError):
    pass


class Gateway:
    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5-mini",
        embed_model: str = "text-embedding-3-large",
        budget: float | None = None,
        # 에이전트(Profiler/Judge/Critic/PE) 호출의 기본 사고량. gpt-5.4-mini 는
        # 명시하지 않으면 사고를 **아예 안 한다** — gpt-5-mini 의 기본(medium)에서
        # 모델만 갈아끼우면 에이전트 품질이 조용히 떨어진다. 분절 호출은 호출부에서
        # 따로 낮춰 잡는다(`--seg-reasoning-effort`).
        reasoning_effort: str | None = None,
        # thinking 모델 + max_tokens 12000 인 PE 호출이 180s 를 넘겼다. 분절 예산을
        # 32768 로 올린 뒤로는 사고가 길어진 호출이 420s 도 넘길 수 있어 함께 올렸다.
        timeout: float = 900.0,
        max_retries: int = 5,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"모르는 provider: {provider!r}. {sorted(PROVIDERS)} 중 하나여야 한다")
        spec = PROVIDERS[provider]
        self.provider = provider
        if api_key is None:
            # 키가 필요 없는 프로바이더(로컬)에도 Authorization 헤더는 실어 보낸다 —
            # ollama 는 무시하고, 인증을 켠 로컬 서버는 자기 키를 환경변수로 받는다.
            api_key = load_api_key(spec.key_env) if spec.key_env else "local"
        self.api_key = api_key
        self.base_url = (base_url or spec.base_url).rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.budget = budget
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        # **base_url 이 아니라 프로바이더가 정한다.** URL 접두사로 보면 `--base-url` 로
        # OpenAI 호환 프록시를 물렸을 때 방언 판정이 조용히 뒤집힌다.
        self.is_openai = spec.openai_dialect
        self._warned_temperature = False
        self._warned_reasoning = False
        # **이 플래그는 게이트웨이 전체에 걸린다.** 에이전트 호출 때문에 한 번 서면
        # 분절 호출도 temperature 를 잃어 **비결정론이 된다**. Claude 계열에서 사고를
        # 켜면 temperature 가 거부되므로, 결정론이 필요하면 `--seg-reasoning-effort`
        # 와 `--agent-reasoning-effort` 를 **둘 다** none 으로 둘 것.
        self.omit_temperature = False    # 이 모델이 temperature 를 거부한다고 확인되면 True
        self.omit_reasoning_effort = False   # 이 모델이 reasoning_effort 를 거부하면 True
        self.omit_json_mode = False          # 이 모델이 response_format 을 거부하면 True
        price = _price_of(model) if self.is_openai else None
        if self.is_openai and price is None:
            print(f"[gateway] 경고: '{model}' 이 _PRICES 에 없다. 비용이 0 으로 집계되어 "
                  f"--budget 예산 가드가 작동하지 않는다.", file=sys.stderr)
        self.usage = Usage(price=price)
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        )

    @classmethod
    def from_args(cls, args, **kw) -> "Gateway":
        """`add_provider_args` 로 받은 인자를 그대로 넘겨 만든다."""
        return cls(provider=args.provider, base_url=args.base_url, **kw)

    def close(self) -> None:
        self._client.close()

    # ── 저수준 ───────────────────────────────────────────────────────────
    def _post(self, path: str, body: dict, purpose: str = "other") -> dict:
        if self.budget is not None and self.usage.cost >= self.budget:
            raise BudgetExceeded(f"예산 초과: {self.usage.cost:.4f} >= {self.budget}")

        def _rejected(resp, name: str) -> bool:
            """400 응답이 `name` 파라미터를 문제 삼았는가.

            **에러 형식이 게이트웨이마다 다르다.** OpenAI 는 `error.param` 에 이름을
            넣지만 Letsur 는 `{"type":"service_error","detail":"- {\\"message\\": ...}"}`
            로 감싸 보내 `param` 이 없다. `param` 만 보던 종전 코드는 그 400 을 못 알아보고
            5회 재시도 뒤 런을 죽였다 — run12 가 시작 즉시 그렇게 죽었다
            (claude-haiku + temperature 0 + 사고: "`temperature` may only be set to 1
            when thinking is enabled"). 그래서 본문 문자열도 함께 본다.
            """
            try:
                j = resp.json()
            except Exception:                                    # noqa: BLE001
                return name in (resp.text or "")
            if (j.get("error") or {}).get("param") == name:
                return True
            blob = json.dumps(j, ensure_ascii=False)
            if name in blob:
                return True
            # ollama 는 파라미터 이름을 아예 안 쓰고 기능 이름으로 거부한다:
            # `{"error":{"message":"\"granite-16k\" does not support thinking"}}`.
            # 이름만 찾던 종전 코드는 이 400 을 못 알아보고 5회 재시도 뒤 런을 죽였다
            # (granite4 / qwen3-next 라벨링이 그렇게 죽었다).
            if name == "reasoning_effort" and "does not support thinking" in blob:
                return True
            return False

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(f"{self.base_url}{path}", json=body)
                # 추론 계열(o4-mini, gpt-5-mini …)은 temperature 를 기본값 1 로만 받는다.
                # **이 모델들은 분절기를 비결정론적으로 만든다** — 루프가 검출하려는
                # 프롬프트 차이가 0.003 규모라 표집 잡음이 신호를 덮을 수 있다
                # (설계 §8.6-5: 모델이 흔들리면 점수 변화의 귀속이 불가능).
                # 죽이지는 않되, 무엇이 바뀌었는지 반드시 로그에 남긴다.
                if r.status_code == 400 and "temperature" in body:
                    if _rejected(r, "temperature"):
                        body = {k: v for k, v in body.items() if k != "temperature"}
                        # **플래그로 기억한다.** 호출마다 다시 붙이면 매 호출이 400 을 한 번씩
                        # 먹고 재시도 예산도 하나씩 깎는다 (run05 초반 로그에서 실제로 그랬다).
                        self.omit_temperature = True
                        if not self._warned_temperature:
                            self._warned_temperature = True
                            print(f"[gateway] 경고: '{body.get('model')}' 가 temperature 를 "
                                  f"거부해 기본값으로 돈다 — **분절이 비결정론적이 된다**. "
                                  f"Claude 계열은 사고를 켜면 temperature 를 1 로만 받으므로, "
                                  f"결정론이 필요하면 `--seg-reasoning-effort none` 으로 "
                                  f"사고를 끌 것.", file=sys.stderr)
                        continue
                # `reasoning_effort` 미지원 모델도 같은 형태로 400 을 준다. 사고량 조절은
                # 비용 최적화지 정확성 요건이 아니므로, 거부하면 떼고 계속 간다.
                if r.status_code == 400 and "reasoning_effort" in body:
                    if _rejected(r, "reasoning_effort"):
                        body = {k: v for k, v in body.items() if k != "reasoning_effort"}
                        self.omit_reasoning_effort = True
                        if not self._warned_reasoning:
                            self._warned_reasoning = True
                            print(f"[gateway] 경고: '{body.get('model')}' 는 reasoning_effort "
                                  f"를 지원하지 않는다 — 모델 기본 사고량으로 돈다.",
                                  file=sys.stderr)
                        continue
                if r.status_code == 400 and "response_format" in body:
                    if _rejected(r, "response_format"):
                        body = {k: v for k, v in body.items() if k != "response_format"}
                        self.omit_json_mode = True
                        continue
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    continue
                r.raise_for_status()
                payload = r.json()
                self.usage.add(payload, purpose)
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
        reasoning_effort=_INHERIT,
        purpose: str = "other",
        json_mode: bool = False,
    ) -> str:
        """`reasoning_effort` 는 **비용의 유일한 실질 손잡이**다.

        run05 실측: 총비용의 98% 가 출력 토큰이고(입력은 캐시 적중 90% 라 무시 가능),
        분절 1콜의 출력 3,700 토큰 중 본문은 76 토큰뿐 — 나머지가 전부 사고 토큰이다.
        같은 4문장 기준 gpt-5-mini 는 medium(기본) 14,332 → low 5,240 토큰으로 2.7배
        싸지는데 태그 수와 원문 보존이 오히려 같거나 낫다. `minimal` 은 원문 보존이
        4/4 → 2/4 로 무너져 쓸 수 없다.

        gpt-5.4-mini 는 **기본이 사고 없음**이라 그대로 두면 태그를 필요량의 1/3 밖에
        안 찍어 `too_few_tags` 로 전량 재시도된다 — 반드시 명시해야 한다. 단 사고를
        켜면 `temperature=0` 이 거부되므로(사고 끈 상태에서만 허용) 결정론은 포기해야
        한다. 결정론이 요건이면 모델을 Letsur `claude-sonnet-5` 로 바꿀 것.
        """
        # gpt-5 계열은 max_tokens 를 400 으로 거부한다 (param='max_tokens',
        # "Use 'max_completion_tokens' instead"). Letsur 는 반대로 max_tokens 만 받는다.
        effort = (self.reasoning_effort if reasoning_effort is _INHERIT
                  else reasoning_effort)
        budget_key = "max_completion_tokens" if self.is_openai else "max_tokens"
        body = {
            "model": model or self.model,
            budget_key: max_tokens,
            **({} if self.omit_temperature else {"temperature": temperature}),
            # **`is_openai` 로 막으면 안 된다.** Letsur 게이트웨이도 이 파라미터를
            # 그대로 받는다 — 실측(gpt-5-mini, 분절 프롬프트 + 1문장):
            #     low 640 / medium 2,752 / high 6,464 사고 토큰
            # 종전에는 OpenAI 직결일 때만 실었기 때문에 **Letsur 런에서
            # `--seg-reasoning-effort` 가 아무 일도 안 했다** (run09 실측: low 로 돌렸는데
            # 사고가 15,330tok/콜 로 medium 런의 13,876 보다 오히려 컸다). 비용의 94% 가
            # 분절 사고인데 유일한 손잡이가 조용히 끊겨 있었다.
            #
            # 거부하는 모델은 아래 400 처리가 `omit_reasoning_effort` 를 세워 다음
            # 호출부터 뺀다 — 그게 이 플래그의 존재 이유다. 엔드포인트로 미리 막을 일이
            # 아니다.
            **({"reasoning_effort": effort}
               if (effort and not self.omit_reasoning_effort) else {}),
            # 구문상 유효한 JSON 을 서버가 보장한다. temperature 를 0 으로 못 박는
            # 모델에서는 이게 유일한 방어다 — en-de run01 에서 Profiler 가 깨진 JSON 을
            # 냈고 **복구 호출까지 같은 실패**를 반복해 런이 죽었다.
            # 같은 이유로 `is_openai` 를 뺀다 — Letsur 도 `response_format` 을 받는다
            # (실측: `{"type":"json_object"}` 로 정상 응답). 거부하면 400 처리가 끈다.
            **({"response_format": {"type": "json_object"}}
               if (json_mode and not self.omit_json_mode) else {}),
            "messages": [
                {"role": "system", "content": _cacheable(system)},
                {"role": "user", "content": user},
            ],
        }
        # 추적은 실패해도 런을 죽이지 않는다 (tracing.py 참조).
        run = tracing.start_llm_run(purpose, body["model"], system, user,
                                    metadata={"max_tokens": max_tokens,
                                              "reasoning_effort": effort})
        try:
            payload = self._post("/chat/completions", body, purpose)
        except Exception as e:
            run.finish(error=str(e))
            raise
        run.finish(payload)
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
        raw = self.chat(system, user, **{**kw, "json_mode": True})
        try:
            return parse_json_loose(raw)
        except (ValueError, json.JSONDecodeError) as e:
            # **잘린 출력은 복구 대상이 아니다.** 아래 복구 호출은 깨진 JSON 을 되돌려
            # "문법만 고쳐라"라고 시키는데, 예산을 다 써서 잘린 출력은 뒤 내용 자체가
            # 없으므로 고칠 수가 없다. 게다가 복구 예산(8000)이 원본(16000~32000)보다
            # 작아 재생산도 불가능하다 — 실측에서 원본과 복구가 **같은 실패**를 반복하고
            # 런이 죽었다 (로컬 Qwen, profiler 가 16,000 토큰을 다 쓰고 JSON 을 안 닫음).
            # 잘렸을 때는 같은 요청을 간결 제약과 함께 다시 던지는 것이 유일한 수다.
            if _looks_truncated(raw):
                retried = self.chat(
                    system + _BREVITY_SUFFIX, user,
                    **{**kw, "json_mode": True,
                       "purpose": f"{kw.get('purpose', 'other')}:brevity_retry"},
                )
                return parse_json_loose(retried)
            repaired = self.chat(
                "You repair malformed JSON. Return ONLY the corrected JSON object. "
                "Preserve all content; fix only syntax (unescaped quotes, missing commas, "
                "trailing commas, unterminated strings).",
                f"Parse error: {e}\n\n=== MALFORMED JSON ===\n{raw}",
                **{**kw, "max_tokens": kw.get("max_tokens", 8000), "json_mode": True,
                   "purpose": f"{kw.get('purpose', 'other')}:json_repair"},
            )
            return parse_json_loose(repaired)

    # ── 임베딩 ───────────────────────────────────────────────────────────
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            chunk = texts[i : i + 64]
            payload = self._post("/embeddings",
                                 {"model": self.embed_model, "input": chunk}, "embed")
            out.extend(d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"]))
        return out


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


# 출력이 예산을 다 써서 잘렸는지 본다. 완결된 JSON 객체는 `}` 로 끝나고 중괄호가
# 맞는다 — 문자열 안의 중괄호까지 세지는 않으므로 완벽하지는 않지만, 여기서 필요한
# 것은 "명백히 안 닫힌 출력"의 판별뿐이다.
def _looks_truncated(raw: str) -> bool:
    t = _FENCE.sub("", raw or "").strip()
    if not t:
        return True
    return not t.endswith(("}", "]")) or t.count("{") > t.count("}")


# 잘린 재시도에 붙이는 제약. 필드를 짧게 강제해야 예산 안에서 JSON 이 닫힌다.
_BREVITY_SUFFIX = (
    "\n\nHARD LIMIT: keep every field value under 25 words. Do not quote source "
    "sentences. Do not add fields beyond those requested. Close the JSON object."
)


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
