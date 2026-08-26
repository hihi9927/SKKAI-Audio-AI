"""A2 Segmenter / A3 Format Validator / A5 Translation Tools.

전부 결정론적 래퍼이거나 프롬프트 주입형 LLM 호출이다. 판단하는 에이전트는 없다.
번역기의 모델·프롬프트는 런 전체에서 고정한다 — 여기가 흔들리면 점수 변화가
분절 때문인지 번역 때문인지 구분할 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .gateway import Gateway

# 태그는 **순위를 달고 나온다** — `<SEG:1>` 이 가장 확실한 경계다.
# 순위가 있어야 사후 절단(Truncator)으로 지연 노브를 돌릴 수 있다: 경계를 빼기만
# 하므로 조각 수가 반드시 줄고, LLM 의 지시 준수에 의존하지 않는다.
#
# 순위 없는 `<SEG>` 도 파싱만은 받아준다. 사람이 쓴 비교군 프롬프트(human_prompts/)와
# 기계분절 앵커가 그 형태이기 때문이다. 다만 순위가 없으면 절단이 불가능하므로
# 그런 분절은 곡선 위의 **점 하나**로만 평가된다 (설계 v2 §11.1).
SEG = "<SEG>"
TAG_RE = re.compile(r"<SEG(?::(\d+))?>")
CONSECUTIVE = re.compile(r"<SEG(?::\d+)?>\s*<SEG(?::\d+)?>")


def tag(priority: int | None = None) -> str:
    return SEG if priority is None else f"<SEG:{priority}>"


def priorities(seg_text: str) -> list[int | None]:
    """등장 순서대로의 순위 목록. 순위 없는 태그는 None."""
    return [int(m.group(1)) if m.group(1) else None
            for m in TAG_RE.finditer(seg_text)]


def strip_tags(seg_text: str, spaced: bool = True) -> str:
    s = TAG_RE.sub(" ", seg_text)
    return re.sub(r"\s+", " ", s).strip() if spaced else re.sub(r"\s+", "", s)


def round_half_up(x: float) -> int:
    """0.5 는 **항상 위로**. 파이썬 `round()` 는 짝수 반올림이라 `round(2.5)==2` 다.

    조각 수와 T 격자가 같은 파라미터 사슬에 있는데 예전에는 반올림이 서로 달랐다
    (`chunk_budget` 은 `round()`, 격자 유도는 `int(x+0.5)`). 정확히 .5 가 되는 길이에서
    조각 수가 갈렸다 — min_gap=3 격자에서 (길이, T) 조합의 7.1%, 예를 들어 T=4·10어절이
    2조각과 3조각으로 나뉘었다. 격자 쪽도 `t_floor=7` 이면 `7×1.5=10.5` 가 10 과 11 로
    갈렸다. 규칙 하나로 통일한다.
    """
    return int(x + 0.5)


def unit_count(text: str, spaced: bool) -> int:
    """T(목표 조각 크기)의 단위. 띄어쓰기 언어는 어절, 아니면 문자."""
    return len(text.split()) if spaced else len(re.sub(r"\s+", "", text))

# 분절 호출의 출력 예산. **thinking 모델에서는 사고 토큰이 여기 같이 잡힌다.**
# 1024 로 두면 긴 문장에서 사고가 예산을 전부 먹고 content 가 빈 문자열로 돌아온다
# (finish_reason='length', completion_tokens=1024, content=''). 그러면 검증기가
# text_modified 로 잡고, 복구 재시도도 같은 한도라 똑같이 실패해 V 가 1.0 에
# 영원히 도달하지 못한다. 실측: 103자 문장이 사고에만 918~1464 토큰을 썼다.
# 프롬프트로는 절대 고칠 수 없는 문제이므로 예산을 넉넉히 준다.
#
# 8192 도 부족했다 — run04(kspon-train) iter0 에서 60행 중 6행이 같은 증상으로 빈
# 출력이 났고, gateway 경고로 finish_reason='length' 를 직접 확인했다 (입력 86·125·140자).
# 사고량은 글자수에 비례하지 않는다: 위 103자=~1.4k 를 외삽하면 193자라도 ~2.7k 여야
# 하는데 실제로는 8192 를 넘겼다. 경계 수가 많고 순위까지 매겨야 하는 문장에서
# 사고가 초선형으로 늘기 때문으로 보인다. kspon(eval_clean_1000) 은 p99 가 28어절이라
# 안 드러났고, 풀을 kspon-train(p99 54어절)으로 바꾸면서 드러났다.
#
# max_tokens 는 상한이지 과금 단위가 아니다 — 짧은 문장은 여기 닿지 않으므로 올려도
# 비용이 늘지 않는다. 반대로 부족하면 8192 토큰을 전부 쓰고 결과가 0 이다.
SEG_MAX_TOKENS = 32768

# 배치 호출에만 덧붙인다. 인덱스 접두사로 되돌려 붙일 수 있어야 하고, 순위는 문장마다
# 1 부터 다시 시작해야 절단 규약이 문장 단위로 유지된다.
BATCH_PROTOCOL = """

[Batch Protocol]
You will receive SEVERAL independent sentences, each on its own line prefixed with an index
like [1], [2], .... Treat every sentence completely independently — never let one sentence
influence the boundaries or ranking of another.
Output ONE line per input sentence, in the same order, with the SAME [n] prefix, followed by
a single space and then that sentence with <SEG:k> tags inserted.
Restart the confidence ranking at <SEG:1> inside EVERY sentence.
Output nothing else — no blank lines, no commentary.
"""

_BATCH_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")

# 어느 문자가 "앞 텍스트에 붙는" 구두점인지는 언어마다 다르다. 목록을 코드에
# 박으면 검증기가 언어 종속이 되므로, 정상 경로는 `data.measure_profile` 의 실측이다.
# 여기는 그 실측이 빈 목록을 낼 때(구두점이 거의 없는 코퍼스)만 도는 대비책이다.
#
# **`Pf`(닫는 곡선따옴표)를 뺀 것이 실측 규칙과의 정합이다.** `data.measure_profile`
# 이 따옴표(`Pi`/`Pf`)를 통째로 제외하는 이유가 여기도 그대로 적용된다 — 여닫이가
# 언어마다 뒤집혀서 되돌리면 인용문 한복판을 자른다 (그 docstring 의 zh 실측 참조).
# `¿¡` 는 표본 하나로는 붙음비율을 못 재므로 여기서만 목록으로 남는다.
_PO_SENTENCE_OPENERS = "¿¡"


def default_trailing_punct(sample: str) -> str:
    """프로파일에 trailing_punctuation 이 없을 때 쓰는 언어 무관 추정."""
    import unicodedata
    out = set()
    for ch in set(sample):
        if ch in _PO_SENTENCE_OPENERS:
            continue
        if unicodedata.category(ch) in ("Po", "Pe"):
            out.add(ch)
    return "".join(sorted(out))


# ── 캐시 ───────────────────────────────────────────────────────────────────

class JsonCache:
    """디스크 영속 캐시. 이터레이션 간 분절이 안 바뀐 문장의 재번역을 막는다."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, object] = {}
        self._lock = threading.Lock()
        self._dirty = 0
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]

    def get(self, k: str):
        with self._lock:
            return self._data.get(k)

    def put(self, k: str, v) -> None:
        with self._lock:
            self._data[k] = v
            self._dirty += 1
            if self._dirty >= 20:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        self._dirty = 0


# ── A2 Segmenter ─────────────────────────────────────────────────────────

def segment_batch(
    gw: Gateway,
    prompt: str,
    texts: list[str],
    cache: JsonCache | None = None,
    workers: int = 8,
    validate_fn=None,
    normalize_fn=None,
    reasoning_effort: str | None = None,
    batch_size: int = 1,
    first_pass_sink: list | None = None,
) -> tuple[list[str], list[bool]]:
    """프롬프트 주입형 분절. 동일 (프롬프트, 문장) 조합은 캐시 재사용.

    복구는 두 단계다.
      1. `normalize_fn(원문, 출력)` — 표기 오류를 결정론적으로 고친다 (번호·태그 위치·공백).
         **`validate_fn` 과 같은 (원문, 출력) 형태다** — 무엇을 고쳤는지 기록하려면
         어느 문장인지 알아야 하는데, 출력만 받으면 그 연결이 끊긴다.
         LLM 호출이 없고 경계 위치를 안 바꾸므로 무료이고 안전하다.
      2. LLM 복구 재시도 1회 — 정규화로 못 고치는 것(`text_modified`)만 남는다.

    복구가 프롬프트 품질을 가리면 안 되므로 **1차 통과 여부를 따로 반환**한다.
    다만 1차 판정은 정규화 **이후**에 한다 — 표기 흔들림은 프롬프트 품질이 아니다.

    반환: (분절 결과, 1차 시도에서 포맷을 지켰는지 여부)

    `first_pass_sink` 를 주면 **재시도 전** 위반을 그대로 담는다. `violations.json` 은
    복구 뒤 최종 상태라, 1차 통과율이 떨어져도 무엇이 깨졌는지 산출물에 남지 않는다 —
    run03 iter1 에서 1차 0.97 → 0.53 의 원인을 재현 실험으로 다시 부르고서야 알았다
    (`too_few_tags` 5 → 19, 개정본이 마킹을 줄임).
    """
    # **배치.** 한 콜에 여러 문장을 넣으면 문장당 사고 토큰이 크게 준다 (실측 24문장:
    # b=1 5,237 → b=3 2,994 → b=6 1,680). 비용의 90% 가 분절 사고이므로 가장 큰 레버다.
    # 재시도는 **단건으로 떨어뜨린다** — 배치 하나가 실패했다고 성공한 형제까지 다시
    # 부르면 절감이 사라지고, 복구 프롬프트도 단문 전제로 쓰여 있다.
    batch_prompt = prompt if batch_size <= 1 else prompt + BATCH_PROTOCOL
    prompt_hash = JsonCache.key(prompt)

    def cache_key(t: str) -> str:
        # "seg3" 는 정규화 도입 시점 표시다. 이전 캐시는 정규화 전 문자열이라 그대로 쓰면
        # 새 코드와 옛 결과가 섞인다.
        # 사고량이 바뀌면 분절도 바뀌므로 키에 넣는다. 안 넣으면 effort=low 런이 medium 으로
        # 만든 옛 결과를 그대로 돌려받아 비교가 조용히 깨진다.
        # **모델도 같은 이유로 키에 들어간다.** 없으면 모델을 바꿔 돌린 평가가 이전 모델의
        # 캐시를 그대로 맞아 호출 0 회로 "동일한 결과"를 내놓는다 — 두 모델을 비교하려던
        # 실험이 조용히 같은 분절을 두 번 채점하는 것으로 바뀐다.
        return JsonCache.key("seg3", prompt_hash, gw.model, reasoning_effort or "-",
                             str(batch_size), t)

    def cached(t: str) -> list | None:
        """쓸 수 있는 캐시 값. 없거나 빈 출력이면 None.

        빈 출력은 **실패한 호출**이지 결과가 아니다. 예전 캐시에 남아 있는 빈 값을 그대로
        돌려주면 호출 없이(calls=0) 포맷 통과율이 떨어져, 원인이 캐시라는 사실이 로그
        어디에도 안 남는다 — run04 에서 실제로 이것 때문에 score 가 0 이 되고 judge 가 빈
        리스트를 인덱싱해 루프 전체가 중단됐다. 미스로 처리해 재호출한다.

        **키 계산과 빈 값 판정은 여기 한 곳에만 둔다.** 예전에는 `one()` 과 배치 제외 목록이
        같은 6조각 키를 각자 만들어, 한쪽만 고치면 배치 경로가 캐시를 통째로 놓치면서도
        겉으로는 정상 동작으로 보였다.
        """
        if cache is None:
            return None
        hit = cache.get(cache_key(t))
        if hit is None or not (hit[0] or "").strip():
            return None
        return hit

    def one(t: str, pre: str | None = None) -> tuple[str, bool]:
        hit = cached(t)
        if hit is not None:
            return hit[0], hit[1]

        out = pre if pre is not None else gw.chat(
            system=prompt, user=t, max_tokens=SEG_MAX_TOKENS,
            reasoning_effort=reasoning_effort, purpose="segment")
        if normalize_fn is not None:
            out = normalize_fn(t, out)
        first_ok = True
        if validate_fn is not None:
            vs = validate_fn(t, out)
            if vs:
                first_ok = False
                if first_pass_sink is not None:
                    first_pass_sink.extend(
                        {"rule": v.rule, "detail": v.detail, "text": t, "seg_text": out}
                        for v in vs)
                detail = "; ".join(f"{v.rule}: {v.detail}" for v in vs)
                # **규칙을 여기 다시 쓰지 않는다.** `system=prompt` 로 프롬프트를 통째로
                # 다시 주므로 `[Output Rules]` 가 이미 들어가 있다. 예전에는 번호 규약
                # ("numbered 1..N by confidence")과 커버리지 지침("marking one is free")을
                # 이 문자열에 복사해 뒀는데, 그건 `agents.output_rules()` 가 만드는 문장과
                # 같은 말이다. A1 쪽에서 규칙을 고치면 여기만 옛말로 남아 **1차 호출과
                # 재시도가 서로 다른 규칙을 요구**하게 되고, 재시도가 고칠 수 없는 것을
                # 시키는 무한 실패가 된다. 이 콜이 더할 것은 위반 내역과 "원문 그대로
                # 다시 내라"뿐이다.
                out = gw.chat(
                    system=prompt,
                    user=(
                        f"{t}\n\n"
                        f"[Your previous answer violated the output rules: {detail}]\n"
                        f"[Previous answer: {out}]\n"
                        f"[Re-emit the ORIGINAL text above, character for character, with only "
                        f"<SEG:n> tags inserted. Fix every violation listed above by following "
                        f"the [Output Rules] section of your instructions — it already states "
                        f"how tags must be numbered and what to do when you cannot find enough "
                        f"safe positions. Do not shorten, rewrite, or add anything.]"
                    ),
                    max_tokens=SEG_MAX_TOKENS,
                    reasoning_effort=reasoning_effort,
                    purpose="segment_retry",
                )
                if normalize_fn is not None:
                    out = normalize_fn(t, out)
        # 실패(빈 출력)는 캐싱하지 않는다. 캐싱하면 다음 런이 재시도조차 못 하고
        # 같은 실패를 영구히 재생한다.
        if cache is not None and (out or "").strip():
            cache.put(cache_key(t), [out, first_ok])
        return out, first_ok

    def call_group(g: list[str]) -> dict[str, str]:
        user = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(g))
        raw = gw.chat(system=batch_prompt, user=user, max_tokens=SEG_MAX_TOKENS,
                      reasoning_effort=reasoning_effort, purpose="segment")
        got: dict[str, str] = {}
        for line in raw.splitlines():
            m = _BATCH_LINE.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            if 1 <= idx <= len(g):
                got[g[idx - 1]] = m.group(2).strip()
        return got

    # 배치는 **미리 채우기**일 뿐이다. `batch_size <= 1` 이면 pre_map 이 비고, 아래 단건
    # 경로가 전부 직접 부른다 — 그래서 두 경우가 같은 코드다.
    pre_map: dict[str, str] = {}
    if batch_size > 1:
        # 캐시에 있는 것은 배치에서 뺀다 — 캐시 적중분까지 다시 부르면 배치의 의미가 없다.
        todo = [t for t in texts if cached(t) is None]
        groups = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        if groups:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for d in ex.map(call_group, groups):
                    pre_map.update(d)
    # 배치가 안 돌았거나 파싱이 안 된 문장은 pre=None 이라 one() 이 단건으로 부른다
    # — 조용한 누락 방지.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda t: one(t, pre_map.get(t)), texts))
    if cache is not None:
        cache.flush()
    return [r[0] for r in results], [r[1] for r in results]


# ── A3 Format Validator ──────────────────────────────────────────────────

@dataclass
class Violation:
    id: str
    rule: str
    detail: str


def _note(sink: list[dict] | None, kind: str, detail: str) -> None:
    """결정론적 수정 하나를 기록. `sink` 가 없으면 아무 일도 안 한다."""
    if sink is not None:
        sink.append({"kind": kind, "detail": detail})


# 정규화가 지키기로 한 약속. 어기면 **프롬프트 문제가 아니라 이 함수의 버그**다.
#
# 여기 있는 항목은 전부 예전에 `validate` 의 위반 규칙이었다. 정규화가 생기면서
# 구조적으로 못 뜨게 됐고(마지막 실측이 ko-en/run01 — 정규화 도입 직전이다), 위반
# 목록에 남겨 두면 죽은 규칙이 살아 있는 것처럼 보인다. 그렇다고 그냥 지우면 정규화가
# 약속을 어겨도 아무도 모른다 — 실제로 순위 뒤집힘(연속 태그가 낮은 번호로 살아남음)과
# 구두점 조각(쉼표가 빈 조각을 되살림) 두 버그가 여기서 났고, 둘 다 위반 목록이 아니라
# 이 점검으로 잡혔어야 했다.
#
# 위반이 아니라 **버그 신고**이므로 채점에 섞지 않는다. `sink` 에 남기고 종류마다 한 번
# 경고만 찍는다 — 예외를 던지면 100문장 런이 통째로 죽는다.
_SELF_CHECK_SEEN: set[str] = set()


def _self_check(out: str, spaced: bool, punct: str,
                sink: list[dict] | None = None) -> list[str]:
    """정규화 결과가 약속을 지켰는지 본다. 반환은 위반한 약속 목록(정상이면 빈 리스트)."""
    bad: list[str] = []
    tags = list(TAG_RE.finditer(out))
    if tags and tags[0].start() == 0:
        bad.append("맨 앞 태그가 남음")
    if tags and tags[-1].end() == len(out):
        bad.append("맨 뒤 태그가 남음")
    if CONSECUTIVE.search(out):
        bad.append("연속 태그가 남음")
    if punct and re.search(r"<SEG(?::\d+)?>\s*([" + re.escape(punct) + r"])", out):
        bad.append("태그 직후 구두점이 남음")
    for m in tags:
        before, after = out[: m.start()], out[m.end():]
        if (before and not before.endswith(" ")) or (after and not after.startswith(" ")):
            bad.append("태그 좌우 공백이 하나가 아님")
            break
    prios = [int(m.group(1)) if m.group(1) else None for m in tags]
    numbered = [p for p in prios if p is not None]
    if numbered:
        # 하나라도 번호가 있으면 **전부** 번호가 있어야 하고 1..N 이어야 한다.
        if len(numbered) != len(prios):
            bad.append(f"번호 있는 태그와 없는 태그가 섞임: {prios}")
        elif sorted(numbered) != list(range(1, len(numbered) + 1)):
            bad.append(f"번호가 1..N 이 아님: {sorted(numbered)}")
    for b in bad:
        _note(sink, "self_check_failed", b)
        if b not in _SELF_CHECK_SEEN:
            _SELF_CHECK_SEEN.add(b)
            print(f"[normalize] 버그: {b} — {out!r}")
    return bad


def normalize_tags(seg_text: str, spaced: bool, trailing_punct: str | None = None,
                   sink: list[dict] | None = None) -> str:
    """표기 오류만 결정론적으로 고친다. **경계 위치는 절대 건드리지 않는다.**

    **`sink` 가 주어지면 무엇을 고쳤는지 기록한다.** 이 함수는 조용히 산출물을 바꾸는데,
    그게 어디에도 안 남으면 규칙이 틀렸을 때 영영 모른다 — 실제로 중국어 여는 따옴표
    `“` 가 `trailing_punct` 에 잘못 들어가 13건이 앞 조각 꼬리로 끌려갔는데
    (`涂鸦活动和“ <SEG:6> 合法”涂鸦墙` — 인용어 한복판이 잘린다), 검증기의
    `punct_after_tag` 는 **이 함수가 먼저 돌아 다 고쳐 놓기 때문에** v2 런 전체에서
    0건이었다. 위반이 아니라 기록으로 남겨야 하는 이유다.

    포맷 위반은 프롬프트의 성질이 아니라 분절 모델의 표본 사건이다. run01 에서 30문장 중
    1건의 위반(`priority_gap [2,3,4]`)이 프롬프트 전체를 `score −9.03` 으로 폐기시켰다 —
    실제 프롬프트 차이가 0.003 인데 잡음이 −9.8 을 만들어 신호가 완전히 묻혔다.

    잔여 위반 2건이 모두 여기서 처리되는 유형이었다. 남는 것은 `text_modified` 뿐이고,
    그건 모델이 원문을 고쳐 쓴 것이라 결정론적으로 복구할 수 없다 (LLM 재시도 몫).

    **`punct_after_tag` 가 v2 런에서 0건이었던 건 이 함수가 다 고쳐서가 아니었다.** 구두점을
    바로 왼쪽(=연속 태그 사이의 빈) 조각에 얹으면 이 함수가 스스로 `<SEG:n> ,` 를 만들어
    냈고, 그건 `text_modified` 쪽으로 먼저 잡혔다. 받는 조각을 "가장 가까운 내용 있는" 쪽으로
    바꾼 뒤에야 그 경로가 실제로 닫혔다 — 지금은 `_self_check` 가 계속 지켜본다.

    고치는 것: 번호 조밀화(**확신 순위 보존**), 맨 앞/뒤 태그 삭제, 연속 태그 축약,
    태그 직후 구두점 재배치, 태그 좌우 공백.

    **연속 태그는 합치되 번호는 가장 확신한 것을 쓴다.** 사이에 글자가 없으니 두 태그는
    같은 경계이고, 남는 정보는 위치가 아니라 번호다. 예전에는 뒤엣것을 남겨
    `<SEG:1> <SEG:4>` 가 4 위로 살아남았다 — 모델이 1등으로 꼽은 자리가 `truncate` 에서
    먼저 잘리는 경로였다.

    **번호는 등장 순서가 아니라 모델이 매긴 순위 관계로 다시 붙인다.** 예전에는
    좌->우로 1..N 을 붙였는데, 그러면 `<SEG:n>` 이 확신도가 아니라 위치를 뜻하게 되고
    `truncate()` 가 언제나 **가장 앞쪽** 경계만 남긴다. 설계 §6.1 의 순위 절단이
    통째로 무효가 되는 경로였다 (실측: gpt-5-mini 8문장 전부 모델은 위치와 다른 순열을
    냈는데 정규화 후 8/8 이 위치 순서가 됐고, T=6 절단 결과가 7/8 에서 달라졌다.
    한국어 자연발화는 문장 앞쪽에 군말이 몰려 있어 하필 최악의 경계가 선택된다 —
    프롬프트가 최하위로 매기라고 명시한 부류다).

    순위 없는 `<SEG>` 는 **번호를 붙이지 않고 그대로 둔다.** 비교군(사람 프롬프트,
    기계분절)이 그 형태이고, `truncate()` 는 순위가 없으면 절단하지 않기로 되어 있다
    (설계 §9.2). 예전처럼 번호를 붙이면 비교군이 절단 대상이 되어 규약이 깨진다.
    """
    punct = trailing_punct if trailing_punct is not None else default_trailing_punct(seg_text)
    parts = TAG_RE.split(seg_text.strip())
    pieces = [p.strip() for p in parts[::2]]
    raw_prios: list[int | None] = [int(p) if p else None for p in parts[1::2]]

    # 태그 직후 구두점 -> 앞 조각 끝으로 옮긴다 (원문 문자는 보존)
    #
    # **받는 쪽은 "가장 가까운 내용 있는" 왼쪽 조각이다.** 바로 왼쪽이 연속 태그 사이의
    # 빈 조각이면 구두점이 거기 얹혀 그 조각이 구두점 하나짜리로 되살아난다 —
    # `aa <SEG:2> <SEG:1> , bb` 가 `aa <SEG:2> , <SEG:1> bb` 가 되어 경계 하나를 쉼표에
    # 쓰고, 태그를 떼도 `aa , bb` 라 원문과 달라져 `text_modified` 로 잡혔다. 그건 유일한
    # 채점 차단 위반이라 그 행이 점수에서 빠지고 LLM 재시도까지 나간다.
    # 원문에서 구두점이 붙어 있던 자리가 곧 그 조각이므로, 이쪽이 원문 보존이기도 하다.
    for i in range(len(raw_prios)):
        nxt = pieces[i + 1]
        moved = ""
        while nxt and punct and nxt[0] in punct:
            moved += nxt[0]
            nxt = nxt[1:].lstrip()
        if not moved:
            continue
        j = i
        while j >= 0 and not pieces[j]:
            j -= 1
        if j < 0:
            continue        # 왼쪽에 내용이 아예 없다 = 맨 앞 구두점. 옮기면 원문이 바뀐다
        pieces[j] = pieces[j] + moved
        pieces[i + 1] = nxt
        _note(sink, "punct_moved", f"태그 뒤 {moved!r} 를 앞 조각 끝으로 옮김")

    # 1) 어느 태그가 살아남는지 먼저 정한다. 태그는 **양쪽에 실제 내용이 있을 때만** 산다.
    #    미리 keep 배열을 만들면 연속 태그 `A <SEG:1> <SEG:2> B` 에서 가운데 빈 조각을
    #    양쪽 태그가 각각 보고 둘 다 탈락시킨다. 순차 조립은 앞 태그만 버리고 뒤를 살린다.
    #    맨 앞/뒤 태그도 같은 규칙에서 자동으로 떨어진다.
    chunks: list[str] = [pieces[0].strip()]
    survivors: list[int] = []                 # 살아남은 태그의 원래 인덱스
    absorbed: dict[int, list[int | None]] = {}   # 살아남은 태그가 흡수한 이웃의 원래 번호
    pending: list[int | None] = []
    for i, nxt in enumerate(pieces[1:]):
        nxt = nxt.strip()
        if chunks[-1] and nxt:
            survivors.append(i)
            if pending:
                absorbed[i] = pending
                pending = []
            chunks.append(nxt)
        elif chunks[-1]:
            # 다음 조각이 비었다 = 바로 뒤 태그와 **같은 자리**다. 위치가 같으니 남는
            # 정보는 번호뿐이고, 더 확신한 번호를 살려야 한다. 예전에는 그냥 뒤엣것을
            # 남겨서 `<SEG:1> <SEG:4>` 가 4 로 살아남았고, 모델이 1등이라 한 경계가
            # 절단에서 먼저 잘렸다.
            pending.append(raw_prios[i])
            chunks[-1] = chunks[-1] + nxt
            # 뒤 조각이 비는 경우는 둘이다 — 연속 태그이거나, **마지막 태그**라 뒤에
            # 아무것도 없거나. 후자는 아래에서 `tag_dropped` 로 따로 기록되므로 여기선
            # 세지 않는다. 같은 사건이 두 종류로 잡히면 집계가 어긋난다.
            if i < len(pieces) - 2:
                _note(sink, "tags_merged",
                      f"연속 태그 — {tag(raw_prios[i])} 를 다음 태그와 같은 자리로 합침")
        else:
            # 맨 앞 태그 — **자리 자체가 무효**라 번호도 버린다. 뒤 태그는 실제 텍스트
            # 뒤에 있어 위치가 다르므로, 무효 자리의 확신도를 거기 이식하면 안 된다.
            if pending:
                _note(sink, "tag_dropped",
                      f"맨 앞 태그 {len(pending)}개 — 자리가 무효라 번호까지 버림")
            pending = []
            chunks[-1] = chunks[-1] + nxt
            if i == 0:
                _note(sink, "tag_dropped", f"맨 앞 태그 {tag(raw_prios[i])} 삭제")
    # 맨 뒤에 남은 pending 은 trailing tag 몫 — 물려줄 상대가 없으니 버린다.

    # 2) 살아남은 태그에만 번호를 다시 매긴다 — 원래 번호의 **순위 관계**를 보존한 채
    #    1..N 으로 조밀화한다. 중복·결번은 등장 순서로 깨고, 번호 없는 태그는 뒤로 민다.
    #    같은 자리로 합쳐진 태그들은 그중 **가장 확신한 번호**로 대표된다.
    def _best(i: int) -> int | None:
        nums = [x for x in (raw_prios[i], *absorbed.get(i, [])) if x is not None]
        return min(nums) if nums else None

    vals = [_best(i) for i in survivors]
    if all(v is None for v in vals):
        labels: list[int | None] = [None] * len(vals)          # 비교군 — 그대로 둔다
    else:
        order = sorted(range(len(vals)),
                       key=lambda j: (vals[j] is None, vals[j] if vals[j] is not None else 0, j))
        labels = [0] * len(vals)
        for rank, j in enumerate(order, 1):
            labels[j] = rank

    if pending:
        _note(sink, "tag_dropped", f"맨 뒤 태그 {len(pending)}개 삭제")

    # 번호가 실제로 바뀌었을 때만 기록한다 — 조밀화는 대부분의 문장에서 무동작이다.
    kept = [raw_prios[i] for i in survivors]
    if labels != kept and any(v is not None for v in vals):
        _note(sink, "renumbered", f"{kept} -> {labels} (순위 관계 보존)")

    out = chunks[0]
    for label, nxt in zip(labels, chunks[1:]):
        out = f"{out} {tag(label)} {nxt}"
    out = out.strip()
    _self_check(out, spaced, punct, sink)
    return out


# 위반 중 **채점을 무의미하게 만드는 것**만 행을 제외한다. 나머지는 `format_pass_rate`
# 로 보고되고 Critic 조향에 쓰이되 점수는 그대로 낸다.
#
# 특히 `too_few_tags` 를 제외 대상으로 두면 안 된다 — 마킹이 부족한 문장(=긴 문장)이
# 통째로 빠져 **짧은 문장만으로 채점되는 우회로**가 열린다. 덜 찍을수록 점수가 오르는
# 구조를 막으려고 만든 규칙이 정반대로 작동하게 된다.
SCORING_BLOCKERS = {"text_modified"}


def blocks_scoring(violations: list["Violation"]) -> bool:
    return any(v.rule in SCORING_BLOCKERS for v in violations)


def validate(sent_id: str, original: str, seg_text: str, spaced: bool,
             trailing_punct: str | None = None,
             require_priority: bool = True,
             min_tags: int | None = None,
             min_gap: int = 0) -> list[Violation]:
    """번역 호출 전에 도는 하드 게이트. LLM 없이 순수 문자열 검사.

    **표기 규칙은 여기 없다.** 맨앞/맨뒤 태그·연속 태그·태그 좌우 공백·번호 중복·번호
    결번은 전부 `normalize_tags` 가 먼저 고쳐 놓아 구조적으로 못 걸린다(마지막 실측이
    ko-en/run01, 정규화 도입 직전이다). 죽은 규칙을 위반 목록에 두면 산출물이 "검사하고
    있다"고 거짓말한다. 같은 항목은 `_self_check` 가 **버그 신고**로 본다 — 거기서 걸리면
    프롬프트가 아니라 정규화가 잘못한 것이라 대응이 완전히 다르다.

    여기 남은 것은 정규화가 **원리적으로 못 고치는** 것뿐이다: 원문 훼손, 태그 개수 부족,
    조각 간격 부족, 무번호 태그.

    trailing_punct 는 언어 프로파일에서 온다 — 검증 규칙 자체는 언어 무관이고,
    언어 지식은 데이터로 주입된다. 출처는 LLM 추정이 아니라 코퍼스 실측이다
    (`data.measure_profile`) — ja 런에서 LLM 이 이 필드를 통째로 빠뜨린 적이 있다.

    require_priority=False 는 순위 없는 `<SEG>` 를 허용한다. 비교군(사람 프롬프트,
    기계분절)을 같은 검증기로 통과시키기 위한 것이며 루프에서는 쓰지 않는다.
    """
    v: list[Violation] = []
    s = seg_text.strip()
    punct = trailing_punct if trailing_punct is not None else default_trailing_punct(original)
    tags = list(TAG_RE.finditer(s))

    if strip_tags(s, spaced) != strip_tags(original, spaced):
        v.append(Violation(sent_id, "text_modified",
                           "태그를 제거한 결과가 원문과 다름 (모델이 텍스트를 고쳐 씀)"))
    # ── 순위 규칙 ──────────────────────────────────────────────────────
    # 순위가 깨지면 Truncator 가 "상위 k−1개"를 정의할 수 없어 노브 자체가 죽는다.
    #
    # 남는 검사는 **무번호 태그** 하나뿐이다. 번호 중복·결번은 정규화가 1..N 으로 다시
    # 매기지만, 전부 무번호인 출력은 비교군(사람 프롬프트·기계분절)일 수 있어 그대로
    # 두기 때문이다. 루프에서는 순위가 없으면 절단이 불가능하므로 위반이다.
    #
    # **부분 순위(`priority_depth`)는 기각됐다** — 상위 N 개만 번호를 요구해 사고량을
    # 아끼려 했으나, 정렬 생략은 평가 생략이 아니라 사고가 오히려 +17% 늘었다
    # (docs/RANK_METRIC_DIAGNOSIS.md §8.3). 배선은 본 루프에서 한 번도 켜진 적이 없어
    # 지웠다. 되살리려면 그 절을 먼저 읽을 것.
    if require_priority and tags:
        if any(m.group(1) is None for m in tags):
            v.append(Violation(sent_id, "bad_priority_format",
                               "순위 없는 태그. 모든 태그는 <SEG:n> 형태여야 함"))

    # ── 커버리지 요건 (v2, C 방식) ──────────────────────────────────────
    # 노브는 경계를 **빼기만** 하므로 프롬프트가 안 찍은 경계는 만들어낼 수 없다.
    # 마킹이 부족하면 T 를 조여도 k 가 안 오르고, 노브가 지연을 통제한다는 설계 전제가
    # 깨진다. run02 실측: 20어절 이상 문장의 충족률 62%, T=2 에서 missing 3.18 —
    # T=2 와 T=3 이 같은 분절로 수렴해 곡선 왼쪽이 뭉갰다.
    #
    # 이걸 지표(목적함수와 싸우는 축)가 아니라 **입력 요건**으로 올린다. 포맷과 같은 층에
    # 두면 복구 재시도가 개수를 명시해 다시 시킬 수 있고, "덜 찍어서 점수 얻기"가
    # 애초에 성립하지 않는다.
    if min_tags and len(tags) < min_tags:
        v.append(Violation(sent_id, "too_few_tags",
                           f"경계 {len(tags)}개 — 최소 {min_tags}개 필요"))

    # ── 간격 요건 ────────────────────────────────────────────────────
    # `truncate` 의 min_gap 과 **같은 규칙을 마킹 시점에** 건다. 사후에만 걸면 절단기가
    # 간격 위반인 상위 순위를 건너뛰고 아래 순위를 집게 되어(실측 남긴 경계 평균 순위
    # 1.92 -> 2.98), 정보가 없다고 측정된 순위축(rank_contra_gap ≈ 0)에 선택이 통째로
    # 실린다. 마킹이 이미 간격을 지키면 절단기 필터는 항등이 되고, 순위는 "쓸 수 있는
    # 자리들 사이"에서만 작동한다.
    #
    # 문면만으로는 안 움직인다 — 개수 요건과 같은 이유로 검증기가 강제해야 한다
    # (docs/RANK_METRIC_DIAGNOSIS.md §8.1: 문면으로만 시킨 dense 는 밀도 불변).
    if min_gap > 0 and tags:
        edges, total = tag_positions(s, spaced)
        for a, b in zip([0] + edges, edges + [total]):
            if b - a < min_gap:
                v.append(Violation(sent_id, "gap_too_small",
                                   f"조각 {b - a}단위 — 최소 간격 {min_gap} 미만"))
                break
    return v


def tag_positions(seg_text: str, spaced: bool) -> tuple[list[int], int]:
    """태그마다 "그 앞까지의 단위 수", 그리고 문장 전체 단위 수.

    **마킹 시점(`validate` 의 간격 검사)과 절단 시점(`truncate`)이 같은 규칙을 써야**
    절단기 필터가 항등이 된다. 예전에는 같은 6줄이 두 군데 복붙돼 있어서 한쪽만 고치면
    조용히 어긋났다 — 설계의 요점이 바로 그 둘을 일치시키는 것이라 특히 나쁜 중복이었다.
    """
    pieces = [q.strip() for q in TAG_RE.split(seg_text.strip())[::2]]
    wc = [unit_count(q, spaced) for q in pieces]
    pos, acc = [], 0
    for w in wc[:-1]:
        acc += w
        pos.append(acc)
    return pos, acc + wc[-1]


def split_segments(seg_text: str) -> list[str]:
    return [p.strip() for p in TAG_RE.split(seg_text)[::2] if p and p.strip()]


# ── A4 Truncator — 지연 노브, 결정론적 ───────────────────────────────────────────

def chunk_budget(text: str, target_chunk_words: int, spaced: bool) -> int:
    """이 문장을 몇 조각으로 낼 것인가.

    노브는 조각 **수**가 아니라 조각 **크기** `T` 다. 수를 고정하면 6어절 문장은
    무의미한 1.5어절 조각이 되고 30어절 문장은 7.5어절 조각으로 여전히 느리다.
    또 짧은 문장이 큰 k 를 못 만들어 도달률이 무너진다 (실측: 최소 3어절 가정에서
    k=4 도달률 0.52). 크기를 고정하면 모든 문장이 자기 길이에 맞는 k 를 받는다.

    **하한은 1 이다 — 종전 2 에서 내렸다.** 하한이 2 면 어떤 T 를 줘도 경계를 최소
    1 개 요구하므로 **무분절을 노브로 표현할 수 없다.** 그 결과 곡선 오른쪽이 죽는다
    (저장된 test 런 재절단, 5 언어쌍):

        en-de  T=16,24,32  -> k 2.00 고정, laal 6.56/6.59/6.59
        de-en  T=16~48     -> k 2.00 고정, laal 6.59~6.61
        ja-en  T=42,64,84,128 -> k 2.00 고정, laal 9.23 **네 점이 동일**
        zh-en  T=32,48,64,96  -> k 2.00 고정, laal 11.00

    하한 1 이면 `round(길이/T)` 가 그대로 살아 T 를 키울수록 무분절 비율이 100% 로
    수렴한다 — **무분절이 곡선의 오른쪽 끝점**이 되고 별도 기준선으로 병기할 이유가
    사라진다. 같은 재절단에서 도달 laal 이 en-de 6.59->21.4, ja-en 9.23->45.4 로
    2~5 배 넓어졌다. 작은 T 에서는 하한이 안 걸려 결과가 **정확히 동일**하다.

    커버리지 요건(`coverage_need`)에는 영향이 거의 없다. 하한이 걸릴 만큼 짧은 문장은
    `min_gap` 용량 상한(`units // min_gap - 1`)에서 이미 0 으로 깎이기 때문이다.
    """
    return max(1, round_half_up(unit_count(text, spaced) / target_chunk_words))


def boundaries(text: str, target_chunk_words: int, spaced: bool) -> int:
    """T 로 나눌 때 **필요한 경계 수**. `chunk_budget >= 1` 이라 음수가 안 나온다."""
    return chunk_budget(text, target_chunk_words, spaced) - 1


def capacity(text: str, min_gap: int, spaced: bool) -> int:
    """`min_gap` 을 지키며 **넣을 수 있는 경계 수의 상한**.

    조각이 각각 min_gap 이상이어야 하므로 조각 수가 `길이 // min_gap` 을 못 넘는다.
    목표치(`boundaries`)는 반올림이지만 이쪽은 **버림**이다 — 한계라서 넘을 수 없다.
    """
    return max(0, unit_count(text, spaced) // min_gap - 1)


def coverage_need(text: str, min_t: int, spaced: bool, min_gap: int) -> int:
    """검증기가 요구할 최소 태그 수 = min(필요한 수, 넣을 수 있는 수).

    개수 요건과 간격 요건은 서로 모른 채 각각 하한/상한을 건다. 용량으로 깎지 않으면
    짧은 문장에서 **만족 불가능한 요건**이 되어 그 문장이 영원히 재시도를 돈다
    (kspon 150문장 중 min_gap=3 에서 1건, =4 에서 7건).

    깎여서 0 이 되면 그 문장은 태그 없이 통과하고 **무분절**로 나간다 — 절단기에서
    무분절이 나오는 경로와 같은 결론이고, 짧은 발화가 통째로 나가야 하는 경우다.
    """
    need = boundaries(text, min_t, spaced)
    if min_gap > 0:
        need = min(need, capacity(text, min_gap, spaced))
    return need


def truncate(seg_text: str, target_chunk_words: int,
             spaced: bool = True, min_gap: int = 0) -> tuple[str, int]:
    """순위 상위 `k−1` 개 경계만 남긴다. 반환 `(절단된 seg_text, missing_boundaries)`.

    `missing_boundaries` 는 예산이 요구한 경계 중 프롬프트가 못 준 개수다. 프롬프트가 충분히 공격적으로 자르지
    않았다는 신호이며, Critic 의 `focus = "coverage"` 판정에 쓰인다.

    경계를 **빼기만** 하므로 조각 수가 반드시 줄고 `laal_words` 가 반드시 오른다 —
    단조성이 구조적으로 보장된다. 순위가 없으면(비교군) 절단하지 않고 그대로 둔다.

    `min_gap` — 이미 고른 경계(및 문장 양끝)와 이 어절 수 미만이면 건너뛴다.
    **T 는 조각 크기의 평균이지 하한이 아니다.** T=6 인데 문장의 절반(47/100)이 2어절
    이하 조각을 하나 이상 갖고 있었고, 그런 문장은 effective 가 0.728 로 최단 조각이
    4어절 이상인 문장(0.784)보다 크게 낮았다. 순위만 보고 고르면 1·2위가 붙어 있을 때
    1어절 조각이 나온다 — 절단기가 **간격을 안 보기** 때문이다.

    en-de test 100문장 오프라인 시뮬 (min_gap 0/2/3/4):
        T=6   eff 0.7533 / 0.7543 / **0.7635** / 0.7511,  laal 4.50 -> 4.17
        T=8   eff 0.7743 / 0.7785 / **0.7814** / 0.7729,  laal 5.45 -> 5.22
    3 이 두 T 모두에서 최적이고 **지연도 함께 준다**. 4 는 과해서 좋은 자리를 너무
    배제한다. 이득은 contradiction 에서 나온다(0.0724 -> 0.0526) — 다만 조각이 길어지면
    NLI 잡음 바닥도 함께 내려가므로, 실제 조기방출 감소인지 바닥 효과인지는 미분리다.

    **제약을 못 채우면 덜 자른다 — 순위 순 보충을 하지 않는다.** 예전에는 보충했고
    "실측 미달 0건" 이라 무해해 보였는데, 그 실측이 en-de 였다. ko-en/run05 를 오프라인
    재절단하면 min_gap=3 에서 보충이 T=2 150/150, T=3 107/150, T=4 11/150, T=6 1/150
    으로 **거의 항상** 발동한다. 보충이 있으면 min_gap 이 선언만 하고 강제를 못 해
    1어절 조각이 51% -> 43% 로 밖에 안 준다. 빼면 하한이 진짜 하한이 된다 — 전 T 에서
    1어절 조각 0%, 최단 조각 = min_gap 정확히.

    **무분절이 나오는 길은 둘이다.**
      (1) **노브가 요청** — `chunk_budget` 하한이 1 이라 T 가 문장 길이에 가까워지면
          요청 자체가 1 조각이 된다. 정상 동작이고 곡선의 오른쪽 끝이다.
      (2) **문장이 짧아 자리가 없음** — `units < 2*min_gap` 이면 양끝에서 각각
          min_gap 이상 떨어진 자리가 하나도 없다. T 와 무관한 **문장 고유 성질**이다
          (run05 실측 min_gap=3 에서 1/150, =4 에서 7/150).
    둘 다 `want`(요청 경계 수)와 일치하므로 `missing_boundaries` 는 0 이다. "마킹은
    했는데 min_gap 때문에 하나도 못 놓았다"는 경우는 여기서 안 센다 — 그건 마킹 시점에
    `validate` 의 `gap_too_small` 이 이미 막으므로 절단기까지 오지 않는다.

    **T 는 min_gap 보다 커야 한다.** T 는 조각 크기의 평균이고 min_gap 은 최소라,
    둘이 같으면 모든 조각이 정확히 min_gap 이어야 해서 사실상 불가능하다. `T <= min_gap`
    은 요청이 용량(`길이 // min_gap`)을 넘어 **전부 같은 결과**가 된다 — min_gap=3 이면
    T=1·2·3 이 소수점까지 같다. 처음 달라지는 값이 `min_gap + 1` 이고, 저장된 런 5개
    재절단에서 5/5 가 그랬다. loop.py 가 격자에 그 아래 값이 있으면 경고한다.

    반환하는 `missing_boundaries` 는 **마킹한 태그 수**로만 계산한다 (`want - len(tags)`).
    min_gap 때문에 못 놓은 몫은 빼고 센다 — 그건 프롬프트가 태그를 더 찍어서 고칠 수 있는
    문제가 아니므로, Critic 의 `focus="coverage"` 조향에 섞으면 잘못된 압력이 된다.
    """
    tags = list(TAG_RE.finditer(seg_text))
    if not tags:
        return seg_text, boundaries(seg_text, target_chunk_words, spaced)

    prios = [int(m.group(1)) if m.group(1) else None for m in tags]
    body = strip_tags(seg_text, spaced)
    want = boundaries(body, target_chunk_words, spaced)
    if any(p is None for p in prios):
        return seg_text, max(0, want - len(tags))      # 순위 없음 — 절단 불가

    # `min_gap <= 0` 을 따로 분기하지 않는다 — 거리 조건이 절대 참이 안 되므로 아래
    # 루프가 그대로 `order[:want]` 가 된다 (랜덤 20,000건 검증, 불일치 0).
    order = sorted(range(len(prios)), key=lambda i: prios[i])
    pos, total = tag_positions(seg_text, spaced)
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= want:
            break
        edges = [0, total] + [pos[j] for j in chosen]
        if min(abs(pos[i] - e) for e in edges) < min_gap:
            continue
        chosen.append(i)
    # 못 채우면 덜 자른다 — 순위 순 보충을 하지 않는다
    return _rebuild(seg_text, set(chosen), spaced), max(0, want - len(tags))


def shuffle_priorities(seg_text: str, rng: random.Random) -> str:
    """후보 위치는 그대로 두고 **순위 번호만 무작위로 치환한다.**

    `rank_lift` 의 대조군을 만드는 용도다. 같은 후보 집합·같은 `want`·같은 `min_gap` 을
    통과하므로 `truncate` 가 고르는 **keep 집합만** 달라진다 — 순위가 하는 일(버릴 것과
    남길 것을 가르기)만 분리해서 잴 수 있다.

    순위가 없는 태그가 하나라도 있으면 그대로 돌려준다 (비교군 프롬프트는 절단 자체가
    없으므로 대조군도 성립하지 않는다).
    """
    prios = priorities(seg_text)
    if not prios or any(p is None for p in prios):
        return seg_text
    perm = list(range(1, len(prios) + 1))
    rng.shuffle(perm)
    it = iter(perm)
    return TAG_RE.sub(lambda _m: tag(next(it)), seg_text)


def _rebuild(seg_text: str, keep: set[int], spaced: bool) -> str:
    """버릴 태그를 지우고 공백을 원래 표기로 되돌린다.

    태그는 좌우 공백 하나씩을 달고 나온다 (`OUTPUT_RULES_*` 가 두 표기 체계 모두에서
    이를 요구한다). 그냥 지우면 공백이 둘 남으므로, 띄어쓰기 언어는 하나로 접고
    아닌 언어는 둘 다 없앤다 — 안 그러면 되돌린 문장이 원문과 달라져
    `text_modified` 로 잡힌다."""
    parts = TAG_RE.split(seg_text)
    pieces = [p.strip() for p in parts[::2]]
    prios = parts[1::2]
    joiner = " " if spaced else ""
    out = pieces[0]
    for i, p in enumerate(prios):
        nxt = pieces[i + 1]
        if i in keep:
            out = f"{out} {tag(int(p) if p else None)} {nxt}".strip()
        elif out and nxt:
            out = out + joiner + nxt
        else:
            out = out + nxt
    return out.strip()


# ── A5 Translation Tools ─────────────────────────────────────────────────

# **번역기는 Google 하나다.** LLM 번역기(`--translator llm`) 분기를 들어냈다.
#
# 근거: v2 런 26개의 `config.json` 이 **전부 `translator: google`** 이다 — LLM 경로는
# 한 번도 안 돌았다. 그런데 두 클래스가 `seg_batch` 를 글자까지 같은 코드로 들고 있어
# (15줄 중 12줄 동일), 한쪽만 고치면 안 쓰이는 쪽이 조용히 갈라진다. 안 도는 코드라
# 갈라져도 안 드러나는 것이 복제의 최악 조건이다.
#
# 함께 사라진 것: 번역 규범 3종(`_NORMS`/`FULL_SYSTEM`/`SEG_*_SYSTEM`),
# 에코 검출(`looks_untranslated`), `max_tokens=2048` 3벌, `--translator*` 플래그 2개.
# 되살릴 일이 생기면 이 커밋을 보면 된다.
#
# 판단 근거는 "안 쓰니까"만이 아니다 — 목적함수가 **운영에 옮겨 붙는지**를 재는 것이라
# 운영이 쓰는 번역기로 재야 한다. 운영 서버는 Google 이다
# (`Qwen3-ASR/examples/streaming_websocket_server.py`).


# ── Google Translate 번역기 ─────────────────────────────────────────────────

# 운영 서버(`streaming_websocket_server.py`)의 기본 번역 경로가 이것이다. LLM 번역기로
# 잰 점수가 운영에 옮겨 붙는지 확인하려면 같은 번역기로 재야 한다.
#
# LLM 번역기와 두 가지가 근본적으로 다르다.
#   1. 시스템 프롬프트가 없다. "조각을 완성하지 말 것", "앞 번역은 확정" 같은 지시를
#      줄 방법이 자체가 없으므로 _NORMS 를 적용할 수 없다. 실측상 gtx 는 애초에
#      조각을 완성하지 않으므로 그 지시가 필요 없기도 하다.
#   2. 결정론적이다. 같은 문장 3회 번역이 5/5 문장에서 완전히 일치했다. 따라서
#      상한 앵커(무분절 재번역)가 정확히 1.0 이 되고, Q 에서 번역기 잡음이 사라진다.

GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"
# 지수 백오프. 비공식 엔드포인트라 429 를 맞으면 IP 단위로 걸리므로 넉넉히 기다린다.
BACKOFF_BASE = 2
BACKOFF_MAX = 30
GOOGLE_HEADERS = {"User-Agent": "Mozilla/5.0"}

# tgt_name 은 사람이 읽는 언어명("English")이라 gtx 의 tl 코드로 바꿔야 한다.
LANG_CODES = {
    "english": "en", "korean": "ko", "japanese": "ja",
    "chinese": "zh-CN", "spanish": "es", "french": "fr", "german": "de",
}


def to_lang_code(name: str) -> str:
    key = name.strip().lower()
    if key in LANG_CODES:
        return LANG_CODES[key]
    if len(key) <= 5 and "-" in key or len(key) == 2:
        return name.strip()      # 이미 코드로 넘어온 경우
    raise ValueError(f"Google 번역 대상 언어 코드를 알 수 없습니다: {name!r}. "
                     f"--tgt-code 로 직접 지정하세요.")


@dataclass
class GoogleTranslator:
    """운영 서버의 Google Translate 경로를 그대로 재현한다.

    `use_context` 는 서버의 `--google-context` 플래그에 대응한다.
      False — 조각을 각각 독립 번역 (서버 기본값)
      True  — 앞 조각 원문들을 개행으로 붙여 통째 번역하고 **마지막 줄만** 취함
              (`google_translate_with_context_async` 와 동일)

    두 경우 모두 미래 문맥은 보지 않으므로 스트리밍 조건은 지켜진다.
    """

    tgt_code: str
    cache: JsonCache | None = None
    workers: int = 4          # 비공식 무료 엔드포인트다. 동시성을 낮게 잡는다
    use_context: bool = True
    timeout: float = 30.0
    max_retries: int = 5
    calls: int = 0
    # 컨텍스트 번역은 앞 조각들을 개행으로 붙여 보내고 마지막 줄만 취한다. gtx 가
    # 줄을 합치면 마지막 줄이 엉뚱한 것이 되어 조각 번역이 조용히 오염된다 —
    # 발생 건수를 세서 런 끝에 경고한다.
    context_line_mismatches: int = 0
    _client: httpx.Client = field(init=False, repr=False, default=None)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def __post_init__(self):
        self._client = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=self.workers,
                                max_keepalive_connections=self.workers),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    # ── 저수준 ───────────────────────────────────────────────────────────
    def _raw(self, text: str) -> str:
        if not text.strip():
            return ""
        params = {"client": "gtx", "sl": "auto", "tl": self.tgt_code, "dt": "t", "q": text}
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.get(GOOGLE_URL, params=params, headers=GOOGLE_HEADERS)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"HTTP {r.status_code}")
                else:
                    r.raise_for_status()
                    data = r.json()
                    with self._lock:
                        self.calls += 1
                    return "".join(item[0] for item in data[0] if item and item[0])
            except (httpx.HTTPError, ValueError, IndexError, TypeError) as e:
                last = e
            # 대기는 **한 곳에서만** 계산한다. 예전에는 HTTP 실패 경로와 예외 경로가
            # 같은 식을 각자 갖고 있어, 한쪽만 고치면 상황에 따라 대기가 달라졌다.
            time.sleep(min(BACKOFF_BASE ** attempt, BACKOFF_MAX))
        raise RuntimeError(f"Google 번역 실패({self.max_retries}회 재시도): {last}")

    def _call(self, text: str) -> str:
        """캐시를 거친 번역 1회.

        **빈 값은 결과가 아니라 실패다 — 읽지도 쓰지도 않는다.** `segment_batch` 에서
        이미 한 번 크게 터진 유형이다: 빈 출력이 캐시에 박히면 다음 런이 호출조차
        안 하고(calls=0) 그 문장을 영영 빈 번역으로 돌려주는데, 원인이 캐시라는 사실이
        로그 어디에도 안 남는다 (run04 에서 score 가 0 이 되고 판정자가 빈 리스트를
        인덱싱해 루프 전체가 중단됐다).

        입력 자체가 공백이면 빈 출력이 정상이므로 그때만 통과시킨다.
        """
        if not text.strip():
            return ""
        k = JsonCache.key("gtx", self.tgt_code, text)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None and hit.strip():
                return hit
        out = self._raw(text)
        if self.cache is not None and (out or "").strip():
            self.cache.put(k, out)
        return out

    # ── 공개 인터페이스 ──────────────────────────────────────────────────
    def full(self, texts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            out = list(ex.map(self._call, texts))
        if self.cache is not None:
            self.cache.flush()
        return out

    def full_uncached(self, texts: list[str]) -> list[str]:
        """상한 앵커용 독립 재번역.

        gtx 는 결정론적이라 이 값이 `full()` 과 일치하고 상한이 1.0 이 된다.
        그것 자체가 측정 결과다 — 특수 처리하지 않고 그대로 재서 리포트한다."""
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._raw, texts))

    def streaming_segments(self, segments: list[str]) -> list[str]:
        if not self.use_context:
            return [self._call(s) for s in segments]
        done: list[str] = []
        for i, seg in enumerate(segments):
            if i == 0:
                done.append(self._call(seg))
                continue
            # 서버와 동일: 앞 **원문**들을 개행으로 붙여 통째 번역 후 마지막 줄만
            combined = "\n".join(segments[: i + 1])
            whole = self._call(combined)
            lines = [l.strip() for l in whole.split("\n") if l.strip()]
            if len(lines) != i + 1:
                with self._lock:
                    self.context_line_mismatches += 1
            done.append(lines[-1] if lines else whole)
        return done

    def seg_batch(self, seg_texts: list[str], full_fallback: list[str]) -> tuple[list[str], list[list[str]]]:
        def one(pair):
            seg_text, fb = pair
            parts = split_segments(seg_text)
            if len(parts) <= 1:
                return fb, [fb]
            pieces = self.streaming_segments(parts)
            return " ".join(p for p in pieces if p), pieces

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(one, zip(seg_texts, full_fallback)))
        if self.cache is not None:
            self.cache.flush()
        return [r[0] for r in results], [r[1] for r in results]
