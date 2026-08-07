"""A2 Segmenter / A3 Format Validator / A4 Translation Tools.

전부 결정론적 래퍼이거나 프롬프트 주입형 LLM 호출이다. 판단하는 에이전트는 없다.
번역기의 모델·프롬프트는 런 전체에서 고정한다 — 여기가 흔들리면 점수 변화가
분절 때문인지 번역 때문인지 구분할 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .gateway import Gateway

SEG = "<SEG>"
TAG_RE = re.compile(re.escape(SEG))
CONSECUTIVE = re.compile(re.escape(SEG) + r"\s*" + re.escape(SEG))

# 분절 호출의 출력 예산. **thinking 모델에서는 사고 토큰이 여기 같이 잡힌다.**
# 1024 로 두면 긴 문장에서 사고가 예산을 전부 먹고 content 가 빈 문자열로 돌아온다
# (finish_reason='length', completion_tokens=1024, content=''). 그러면 검증기가
# text_modified 로 잡고, 복구 재시도도 같은 한도라 똑같이 실패해 V 가 1.0 에
# 영원히 도달하지 못한다. 실측: 103자 문장이 사고에만 918~1464 토큰을 썼다.
# 프롬프트로는 절대 고칠 수 없는 문제이므로 예산을 넉넉히 준다.
SEG_MAX_TOKENS = 8192

# 어느 문자가 "앞 텍스트에 붙는" 구두점인지는 언어마다 다르다. 목록을 코드에
# 박으면 검증기가 언어 종속이 되므로, 기본값은 언어 프로파일이 제공한다
# (Language Profiler 의 trailing_punctuation). 프로파일에 없을 때만 유니코드
# 범주로 추정한다 — Pe(닫는 괄호)/Pf(닫는 인용부호)는 항상 뒤따르는 문자이고,
# Po 중에서는 문장을 여는 용도로 쓰이는 것(스페인어 ¿¡ 등)만 제외한다.
_PO_SENTENCE_OPENERS = "¿¡"


def default_trailing_punct(sample: str) -> str:
    """프로파일에 trailing_punctuation 이 없을 때 쓰는 언어 무관 추정."""
    import unicodedata
    out = set()
    for ch in set(sample):
        if ch in _PO_SENTENCE_OPENERS:
            continue
        if unicodedata.category(ch) in ("Po", "Pe", "Pf"):
            out.add(ch)
    return "".join(sorted(out))


# ── 캐시 ─────────────────────────────────────────────────────────────────

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
) -> tuple[list[str], list[bool]]:
    """프롬프트 주입형 분절. 동일 (프롬프트, 문장) 조합은 캐시 재사용.

    포맷 위반 시 위반 내용을 되돌려 1회 복구를 시도한다. 이는 분절 판단이 아니라
    결정론적 툴 정책이므로 에이전트가 아니다. 다만 복구가 프롬프트 품질을 가리면
    안 되므로 **1차 통과 여부를 따로 반환**해 지표로 함께 보고한다.

    반환: (분절 결과, 1차 시도에서 포맷을 지켰는지 여부)
    """
    prompt_hash = JsonCache.key(prompt)

    def one(t: str) -> tuple[str, bool]:
        k = JsonCache.key("seg2", prompt_hash, t)
        if cache is not None:
            hit = cache.get(k)
            if hit is not None:
                return hit[0], hit[1]

        out = gw.chat(system=prompt, user=t, max_tokens=SEG_MAX_TOKENS)
        first_ok = True
        if validate_fn is not None:
            vs = validate_fn(t, out)
            if vs:
                first_ok = False
                detail = "; ".join(f"{v.rule}: {v.detail}" for v in vs)
                out = gw.chat(
                    system=prompt,
                    user=(
                        f"{t}\n\n"
                        f"[Your previous answer violated the output rules: {detail}]\n"
                        f"[Previous answer: {out}]\n"
                        f"[Re-emit the ORIGINAL text above, character for character, with only "
                        f"<SEG> tags inserted. Do not shorten, rewrite, or add anything.]"
                    ),
                    max_tokens=SEG_MAX_TOKENS,
                )
        if cache is not None:
            cache.put(k, [out, first_ok])
        return out, first_ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, texts))
    if cache is not None:
        cache.flush()
    return [r[0] for r in results], [r[1] for r in results]


# ── A3 Format Validator ──────────────────────────────────────────────────

@dataclass
class Violation:
    id: str
    rule: str
    detail: str


def _canon(s: str, spaced: bool) -> str:
    s = TAG_RE.sub(" ", s)
    if spaced:
        return re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+", "", s)


def validate(sent_id: str, original: str, seg_text: str, spaced: bool,
             trailing_punct: str | None = None) -> list[Violation]:
    """번역 호출 전에 도는 하드 게이트. LLM 없이 순수 문자열 검사.

    trailing_punct 는 언어 프로파일에서 온다 — 검증 규칙 자체는 언어 무관이고,
    언어 지식은 데이터로 주입된다."""
    v: list[Violation] = []
    s = seg_text.strip()
    punct = trailing_punct if trailing_punct is not None else default_trailing_punct(original)

    if _canon(s, spaced) != _canon(original, spaced):
        v.append(Violation(sent_id, "text_modified",
                           "태그를 제거한 결과가 원문과 다름 (모델이 텍스트를 고쳐 씀)"))
    if s.startswith(SEG):
        v.append(Violation(sent_id, "leading_tag", "맨 앞에 태그"))
    if s.endswith(SEG):
        v.append(Violation(sent_id, "trailing_tag", "맨 뒤에 태그"))
    if CONSECUTIVE.search(s):
        v.append(Violation(sent_id, "consecutive_tags", "연속 태그"))
    if punct:
        m = re.search(re.escape(SEG) + r"\s*([" + re.escape(punct) + r"])", s)
        if m:
            v.append(Violation(sent_id, "punct_after_tag", f"태그 직후 구두점: {m.group(1)!r}"))
    for m in re.finditer(re.escape(SEG), s):
        before, after = s[: m.start()], s[m.end():]
        if before and not before.endswith(" "):
            v.append(Violation(sent_id, "missing_space", "태그 앞 공백 없음"))
            break
        if after and not after.startswith(" "):
            v.append(Violation(sent_id, "missing_space", "태그 뒤 공백 없음"))
            break
    return v


def split_segments(seg_text: str) -> list[str]:
    return [p.strip() for p in seg_text.split(SEG) if p.strip()]


def looks_untranslated(source: str, output: str, n: int = 8) -> bool:
    """출력이 원문 조각을 그대로 담고 있으면 번역 실패로 본다.

    원문의 n-gram(공백 제거) 중 하나라도 출력에 그대로 나타나면 참.
    언어 무관 휴리스틱 — 스크립트를 가정하지 않는다."""
    src = re.sub(r"\s+", "", source)
    out = re.sub(r"\s+", "", output)
    if len(src) < n or not out:
        return False
    return any(src[i : i + n] in out for i in range(len(src) - n + 1))


# ── A4 Translation Tools ─────────────────────────────────────────────────

# 번역기 변동이 Q 잡음의 주원인이었다 (동일 문장 2회 번역의 Q가 0.9273, 최악 0.5866).
# 자유도가 큰 축 — 고유명사 표기, 인용부호, 절 순서 — 을 세 프롬프트에 동일하게 못 박아
# reference 쪽 변동을 줄인다. 규범 자체는 언어 무관이다.
_NORMS = (
    "- Proper nouns (people, places): always render them phonetically in {tgt_name}. "
    "Never leave them in the source script.\n"
    "- Keep the source's quotation and punctuation marks as they are; do not convert them "
    "to {tgt_name} conventions.\n"
    "- Keep the order of clauses as in the source. Do not reorder or merge them.\n"
    "- Translate faithfully — do not add, omit, or infer meaning beyond what is stated."
)

FULL_SYSTEM = (
    "You are a precise translator from {src_name} into {tgt_name}, specializing in "
    "spoken and conversational language.\n"
    "Rules:\n"
    "- Output ONLY the {tgt_name} translation. No explanations, notes, or alternatives.\n"
    "- Translate the entire passage as one coherent unit.\n"
    "- Preserve the register (casual/formal) exactly as in the source.\n"
    "- Fillers and disfluencies: render naturally or omit if they carry no meaning.\n"
    + _NORMS
)

SEG_FIRST_SYSTEM = (
    "You are a precise translator from {src_name} into {tgt_name}, specializing in "
    "spoken and conversational language.\n"
    "Rules:\n"
    "- Output ONLY the {tgt_name} translation. Nothing else.\n"
    "- The input may be a sentence fragment — translate exactly what is given. "
    "Do NOT complete or extend it.\n"
    "- Preserve the register exactly as in the source.\n"
    + _NORMS
)

SEG_CONTEXT_SYSTEM = (
    "You are a precise translator from {src_name} into {tgt_name}, specializing in "
    "spoken and conversational language.\n"
    "You will receive already-confirmed preceding segment translations as context, "
    "then a new segment to translate.\n"
    "Rules:\n"
    "- Output ONLY the {tgt_name} translation of the NEW segment. Nothing else.\n"
    "- The preceding translations are FINAL. Do NOT reproduce, paraphrase, or continue them.\n"
    "- The new segment may be a grammatical fragment — translate exactly what is given, "
    "do NOT complete it.\n"
    "- Match the register, terminology, and tone established in the preceding translations.\n"
    + _NORMS
)


@dataclass
class Translator:
    """번역 툴 2종. 스트리밍 번역기는 세그먼트 i를 1..i-1 컨텍스트만으로 번역해
    실시간 커밋 조건을 재현한다 — 미래 문맥을 절대 보지 않는다."""

    gw: Gateway
    src_name: str
    tgt_name: str
    model: str
    cache: JsonCache | None = None
    workers: int = 8
    _fmt: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._fmt = {"src_name": self.src_name, "tgt_name": self.tgt_name}

    def _call(self, system: str, user: str, source: str | None = None) -> str:
        k = JsonCache.key("tr", self.model, system, user)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None:
                return hit
        out = self.gw.chat(system=system, user=user, model=self.model, max_tokens=2048)
        # 번역기가 원문을 그대로 되돌리는 실패가 실제로 관측된다. 그대로 두면
        # 분절 탓이 아닌 손실이 Q에 섞여 루프가 엉뚱한 방향으로 간다.
        if source and looks_untranslated(source, out):
            out = self.gw.chat(
                system=system + "\n- CRITICAL: never echo the source text. Output must be "
                                f"written entirely in {self.tgt_name}.",
                user=user, model=self.model, max_tokens=2048,
            )
        if self.cache is not None:
            self.cache.put(k, out)
        return out

    def full(self, texts: list[str]) -> list[str]:
        sys_p = FULL_SYSTEM.format(**self._fmt)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            out = list(ex.map(lambda t: self._call(sys_p, t, source=t), texts))
        if self.cache is not None:
            self.cache.flush()
        return out

    def full_uncached(self, texts: list[str]) -> list[str]:
        """캐시를 우회한 독립 재번역.

        Q 의 상한 앵커(지표 잡음 바닥)를 재려면 같은 입력에 대한 **두 번째 독립
        표본**이 필요하다. 캐시를 타면 같은 문자열이 돌아와 잡음이 0으로 보인다."""
        sys_p = FULL_SYSTEM.format(**self._fmt)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(
                lambda t: self.gw.chat(system=sys_p, user=t, model=self.model,
                                       max_tokens=2048),
                texts))

    def streaming_segments(self, segments: list[str]) -> list[str]:
        """한 문장의 세그먼트를 순서대로 번역. 앞선 번역은 확정되어 수정하지 않는다."""
        first_p = SEG_FIRST_SYSTEM.format(**self._fmt)
        ctx_p = SEG_CONTEXT_SYSTEM.format(**self._fmt)
        done: list[str] = []
        for i, seg in enumerate(segments):
            if i == 0:
                done.append(self._call(first_p, seg, source=seg))
                continue
            ctx = "\n".join(f"[{j+1}] SRC: {segments[j]} -> TGT: {done[j]}" for j in range(i))
            user = (
                "=== Preceding segments (FINAL — do NOT reproduce or modify) ===\n"
                f"{ctx}\n\n"
                "=== Translate ONLY this new segment ===\n"
                f"{seg}"
            )
            done.append(self._call(ctx_p, user, source=seg))
        return done

    def seg_batch(self, seg_texts: list[str], full_fallback: list[str]) -> tuple[list[str], list[list[str]]]:
        """분절이 없는 문장은 full 번역을 그대로 쓴다 (재호출 낭비 방지)."""
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
        joined = [r[0] for r in results]
        pieces = [r[1] for r in results]
        return joined, pieces


# ── Google Translate 번역기 ──────────────────────────────────────────────

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
                    time.sleep(min(2 ** attempt, 30))
                    last = RuntimeError(f"HTTP {r.status_code}")
                    continue
                r.raise_for_status()
                data = r.json()
                with self._lock:
                    self.calls += 1
                return "".join(item[0] for item in data[0] if item and item[0])
            except (httpx.HTTPError, ValueError, IndexError, TypeError) as e:
                last = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Google 번역 실패({self.max_retries}회 재시도): {last}")

    def _call(self, text: str) -> str:
        k = JsonCache.key("gtx", self.tgt_code, text)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None:
                return hit
        out = self._raw(text)
        if self.cache is not None:
            self.cache.put(k, out)
        return out

    # ── Translator 와 동일한 인터페이스 ──────────────────────────────────
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
