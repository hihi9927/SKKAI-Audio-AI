"""LLM 에이전트 3종 — Language Profiler / Critic / Prompt Engineer.

루프에서 LLM 판단이 들어가는 곳은 여기뿐이다. 포맷 검증·점수 계산·채택 판정·
재시도는 전부 결정론적 코드로 처리한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .gateway import Gateway

# ── 출력 토큰 예산 ─────────────────────────────────────────────────────────────
# **추론 모델은 사고 토큰이 max_tokens 에 함께 잡힌다.** 예산이 모자라면 사고가 그걸
# 다 먹고 content 가 빈 문자열로 돌아오는데(finish_reason='length'), 상위에서는 그게
# "모델이 답을 못 냈다"가 아니라 "이상한 답을 냈다"로 보인다 — 판정자에서는
# verdict='error' 로, 분절기에서는 text_modified 로 오진된다.
#
# **max_tokens 는 상한이지 과금 단위가 아니다** — 짧은 응답은 여기 닿지 않으므로
# 넉넉히 줘도 비용이 늘지 않는다. 반대로 부족하면 예산을 전부 쓰고 결과가 0 이다.
# 그러므로 여유는 크게 잡는 것이 정답이다.
#
# 실측 근거 (gpt-5-mini, `premature_cases.json` 6케이스): 판정자 사고량은 989~1300
# 토큰이었다. 종전 예산 1500 은 여유가 15% 뿐이라, temperature 를 0 으로 고정할 수 없는
# 추론 모델에서 사고 길이가 흔들리면 그대로 절단됐다 — 관문에서 ko-en-p02 가 3회 모두
# verdict='error' 를 냈는데도 safe/not-safe 이진 판정에 error 가 안 잡혀 **통과로
# 찍혔다.** 모델을 바꿀 때 이 값들을 먼저 확인할 것.
JUDGE_MAX_TOKENS = 16000        # 산출물은 작은 JSON 하나. 사실상 전부 사고 몫
PROFILER_MAX_TOKENS = 16000     # 언어 프로파일 JSON
PROMPT_MAX_TOKENS = 32000       # prompt_v0 생성·Critic·PE·Compressor — 출력 자체가 길다

# ── 프롬프트 골격 ──────────────────────────────────────────────────────────────
# 골격은 고정한다. Prompt Engineer는 섹션 '내용'만 바꾼다 — 구조를 흔들면 루프가
# 발산하고, 점수 변화를 어떤 변경에 귀속시킬 수 없게 된다.

REQUIRED_SECTIONS = [
    "[Role]",
    "[Core Principles]",
    "[When to Segment]",
    "[Never Segment]",
    "[Priority Rules]",
    "[Decision Procedure]",
    "[Output Rules]",
    "[Examples — Segment]",
    "[Examples — Do NOT Segment]",
]

# 태그는 순위를 달고 나온다. 순위가 있어야 사후 절단으로 지연 노브를 돌릴 수 있고,
# 그러면 지연이 목적함수에서 빠져 단일축이 된다 (설계 v2 §6·§7).
_PRIORITY_RULE = (
    "- Number every tag: <SEG:1>, <SEG:2>, ... where 1 marks the boundary you are MOST\n"
    "  confident about. Use each number exactly once, starting at 1 with no gaps.\n"
    "  The numbers are confidence ranks, NOT positions — <SEG:1> may appear anywhere.\n"
)

OUTPUT_RULES_SPACED = """[Output Rules]
- Insert <SEG:n> tags only. Do NOT change, correct, or paraphrase the original text in any way.
""" + _PRIORITY_RULE + """- No tag at the very start or the very end of the text.
- Never place two tags consecutively.
- Punctuation must stay attached to the text before it — never immediately after a tag.
- Always place exactly one space on both sides of every tag.
- If no segmentation is needed, output the original text unchanged.
- Output the tagged text and nothing else. No explanation, label, or commentary."""

OUTPUT_RULES_UNSPACED = """[Output Rules]
- Insert <SEG:n> tags only. Do NOT change, correct, or paraphrase the original text in any way.
- The source script does not use spaces between words. Do NOT add, remove, or move any
  character of the original text — the only thing you add is the tag itself.
""" + _PRIORITY_RULE + """- No tag at the very start or the very end of the text.
- Never place two tags consecutively.
- Punctuation must stay attached to the text before it — never immediately after a tag.
- Always place exactly one space on both sides of every tag. These two spaces are the
  only whitespace you may introduce.
- If no segmentation is needed, output the original text unchanged.
- Output the tagged text and nothing else. No explanation, label, or commentary."""


_COVERAGE_RULE = (
    "- Mark AT LEAST one boundary per {min_t} {unit} of input (a {example_len}-{unit} sentence\n"
    "  needs at least {example_n}). A deterministic step later keeps only the top-ranked ones,\n"
    "  so a boundary you never marked can never be used.\n"
    "  If you cannot find enough safe positions, mark the least-risky remaining ones and rank\n"
    "  them last. Output with too few boundaries is rejected.\n"
)

# 간격 규칙. 예전 문면은 "extra boundaries cost nothing" 이라고 했지만 그건 절단기가
# 간격을 안 볼 때 얘기다. 붙어 있는 경계는 절단기가 어차피 버리므로 찍어봐야 순위만
# 흐리고, 버려진 자리가 상위 순위면 절단이 아래 순위를 집게 된다 (실측 남긴 경계 평균
# 순위 1.92 -> 2.98). 그래서 **간격을 마킹 시점에 지키게** 한다.
#
# 문면만으로는 안 움직인다 — 밀도를 문면으로 시킨 `dense` 변종이 0.354 로 사실상
# 불변이었다 (docs/RANK_METRIC_DIAGNOSIS.md §8.1). 그래서 강제는 문면이 아니라
# `normalize_tags(min_gap=)` 가 한다 — 너무 가까운 태그를 결정론으로 쳐낸다.
_GAP_RULE = (
    "- Leave AT LEAST {gap} {unit} between any two tags, and the same distance between a tag\n"
    "  and either end of the text. A piece shorter than that cannot be translated on its own,\n"
    "  so such a boundary is unusable no matter how confident you are about it. If two good\n"
    "  positions are closer than {gap} {unit}, mark only the better one. Output that places\n"
    "  tags closer than this is rejected.\n"
)


def output_rules(spaced: bool, min_t: int = 3, min_gap: int = 0) -> str:
    unit = "words" if spaced else "characters"
    base = OUTPUT_RULES_SPACED if spaced else OUTPUT_RULES_UNSPACED
    rules = _COVERAGE_RULE.format(min_t=min_t, unit=unit, example_len=min_t * 6,
                                  example_n=5)
    if min_gap > 0:
        rules += _GAP_RULE.format(gap=min_gap, unit=unit)
    return base.replace("- If no segmentation is needed,", rules + "- If no segmentation is needed,")


# 타깃 언어명·문법 근거 검출. 프롬프트는 소스에만 종속돼야 한다 — 지시만으로는 안 지켜졌다
# (run04 산출물에 독일어 언급 8곳, 순위 규칙 8~11 이 그 위에 세워졌다).
_TARGET_LANG_WORDS = (
    "german", "korean", "japanese", "chinese", "spanish", "french", "italian",
    "portuguese", "russian", "arabic", "hindi", "vietnamese", "thai", "dutch",
    "polish", "turkish", "deutsch",
)
_TARGET_GRAMMAR_WORDS = ("case/gender", "case assignment", "grammatical gender",
                         "case marking", "verb-final")


def check_target_agnostic(prompt: str, src_lang: str | None = None) -> list[str]:
    """프롬프트가 특정 타깃 언어에 기대고 있으면 사유를 돌려준다.

    소스 언어명은 허용한다 — 프롬프트는 소스에 종속돼야 하므로 "English source text" 는
    정상이다. 걸러야 하는 것은 **타깃** 언어명과 그 문법 근거다.
    """
    low = prompt.lower()
    src = (src_lang or "").strip().lower()
    hits = [w for w in _TARGET_LANG_WORDS if w != src and w in low]
    gram = [w for w in _TARGET_GRAMMAR_WORDS if w in low]
    out = []
    if hits:
        out.append(f"타깃 언어명 언급: {sorted(set(hits))}")
    if gram:
        out.append(f"타깃 문법 근거: {sorted(set(gram))}")
    return out


def split_sections(prompt: str) -> dict[str, str]:
    """`[Section]` 헤더로 프롬프트를 가른다. 헤더 자체는 값에 넣지 않는다."""
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in prompt.splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]"):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur = line.strip()
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


# focus 별로 손대도 되는 섹션. `priority` 만 엄격하다 — 지시문이 "[Priority Rules] ONLY"
# 라고 못박는데 run03 iter1 에서 focus=placement 인 채로 [Priority Rules] 를 고쳤다.
_FOCUS_SECTIONS = {
    "priority": {"[Priority Rules]"},
    "coverage": {"[When to Segment]", "[Never Segment]", "[Decision Procedure]",
                 "[Priority Rules]"},
    "placement": {"[When to Segment]", "[Never Segment]", "[Decision Procedure]",
                  "[Examples — Segment]", "[Examples — Do NOT Segment]"},
    "format": {"[Decision Procedure]", "[Core Principles]", "[Role]",
               "[Examples — Segment]", "[Examples — Do NOT Segment]"},
}

# **신뢰 영역의 하한**이다 (수렴 상태에서의 허용 폭). 상한은 `TRUST_*`.
MAX_SECTIONS_CHANGED = 2
MAX_SECTION_GROWTH = 1.25

# 아직 아무 개정도 채택되지 않은 상태의 허용 폭. 걸음이 통하면 넓히고 나쁘면 좁힌다
# (`loop.py` 의 trust region). 고정 임계값 하나로는 **부트스트랩**(v0 가 구조적으로 틀려
# 큰 수리가 필요)과 **수렴**(진동을 막아야 함)을 구별할 수 없다 — de/zh/ja 실측에서
# 세 언어 모두 개정이 거부됐고 둘은 1.25 를 0.01~0.06 초과한 것이었다.
TRUST_SECTIONS_MAX = 4
TRUST_GROWTH_MAX = 2.5


def check_revision(old: str, new: str, focus: str | None = None,
                   max_sections: int | None = None,
                   max_growth: float | None = None) -> list[str]:
    """개정본이 **국소적인지** 검사하는 하드 게이트. 위반 사유 목록을 돌려준다.

    종전에는 "AT MOST TWO sections", "focus 를 따르라" 가 전부 지시문상의 권고였고,
    실측에서 지켜지지 않았다 (en-de run01~03):
      - run03 iter1: `focus=placement` 인데 `[Priority Rules]` 를 고쳤다
      - run03 iter1: 한 섹션 개정에 프롬프트가 +29%(10,392 -> 13,401자) 부풀었다
      - dev 분절이 62~95% 바뀌어 쌍체 비교의 이점(안 바뀐 문장은 분산 기여 0)이 사라졌고
        se 가 0.007~0.009 로 커졌다
    그 결과 dev 까지 간 개정 3건이 **전부 음수**(t = −0.8 ~ −3.2)로 기각됐다. 채택 문턱이
    아니라 개정 품질이 원인이므로, 범위를 코드로 강제한다.

    **다만 고정 임계값은 두 상황을 구별하지 못한다.** 위 사례는 *이미 수렴한 상태에서의
    대규모 재작성*이고, 새 소스 언어의 v0 는 반대로 *구조적으로 틀린 출발점*이라 큰 수리가
    필요하다 (zh 실측: `[Never Segment]` 43줄 vs en 7줄). 실제로 de·zh·ja 세 언어에서
    개정이 전부 거부됐고 그중 둘은 1.25 를 **0.01~0.06** 초과한 것이었다 — 세 언어 모두
    채택이 v0 하나뿐이라 **루프의 최적화 단계가 한 번도 실행되지 않았다.**

    그래서 `max_sections`/`max_growth` 를 인자로 받는다. 호출자(`loop.py`)가 신뢰 영역으로
    조절한다 — 개정이 dev 에서 채택되면 넓히고 기각되면 좁힌다. 상한 2.5 에서 기각 1회면
    2.5/2 = 1.25 로 **곧장 하한**이라, 큰 걸음은 한 번 측정될 기회를 얻고 나쁘면 즉시
    종전 동작으로 돌아간다. 측정 한 번을 지불하고 배우는 구조이지, 시도를 영구히 막는
    구조가 아니다 (en run03 의 나쁜 개정도 이 경로로 한 번 재어진 뒤 반경이 닫힌다).
    """
    max_sections = MAX_SECTIONS_CHANGED if max_sections is None else max_sections
    max_growth = MAX_SECTION_GROWTH if max_growth is None else max_growth
    problems: list[str] = []
    a, b = split_sections(old), split_sections(new)
    changed = [k for k in b if k in a and a[k] != b[k]]
    added = [k for k in b if k not in a]
    removed = [k for k in a if k not in b]
    if added or removed:
        problems.append(f"섹션 추가/삭제: +{added} -{removed}")
    if len(changed) > max_sections:
        problems.append(f"{len(changed)}개 섹션 변경 — 최대 {max_sections}개 ({changed})")
    for k in changed:
        if len(a[k]) >= 200 and len(b[k]) > len(a[k]) * max_growth:
            problems.append(f"{k} 가 {len(a[k])} -> {len(b[k])}자 "
                            f"({len(b[k])/len(a[k]):.2f}배 > {max_growth:.2f})")
    allowed = _FOCUS_SECTIONS.get(focus or "")
    if allowed:
        stray = [k for k in changed if k not in allowed]
        if stray and focus == "priority":
            problems.append(f"focus=priority 인데 {stray} 를 고침 — [Priority Rules] 만 허용")
    return problems


def check_skeleton(prompt: str) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if s not in prompt]


# ── A1 Language Profiler ─────────────────────────────────────────────────

PROFILER_SYSTEM = """You are a computational linguist preparing a meaning-based segmentation
system for a real-time speech translation pipeline.

You will be shown raw sentences in an unknown source language. Your job is to characterise the
language empirically FROM THE SAMPLE — do not rely on textbook generalities — so that a
segmentation prompt can be written for it without any human intervention.

The downstream system inserts <SEG> tags to split each sentence into translation units. Each
unit is translated as soon as it is emitted, seeing only the units before it and never the
units after it. Splitting earlier lowers latency; splitting in the wrong place destroys the
translation.

Do NOT report anything that can be COUNTED from the sample — writing system, punctuation
inventory, or how often punctuation appears. Those are measured directly and given to the
prompt writer alongside your profile. Report only what counting cannot reach: word order,
register, and the concrete surface forms below.

Return ONLY a JSON object with exactly these keys:
{
  "source_language": "language name",
  "word_order": "SOV | SVO | VSO | other, with a one-line justification from the sample",
  "register": "what kind of text this is (spontaneous speech, read-aloud prose, meeting, ...)",
  "fillers_and_hesitations": ["actual tokens observed or expected in this language"],
  "discourse_markers": ["tokens that open or link clauses"],
  "clause_boundary_signals": ["concrete surface forms that mark a clause boundary in THIS language"],
  "non_boundary_traps": ["surface forms that LOOK like boundaries but are not"],
  "unstable_prefix_signals": ["surface forms in THIS language after which a prefix's reading is\n                              still likely to be overturned by what follows — stated WITHOUT naming\n                              or assuming any particular target language"],
  "notes_for_prompt_writer": "2-4 sentences of practical guidance"
}
No prose outside the JSON."""

def measured_facts(measured: dict | None) -> str:
    """실측 프로파일을 프롬프트 작성기·PE 가 읽을 문단으로. 없으면 빈 문자열.

    **셀 수 있는 것은 Profiler 가 아니라 여기서 온다.** 종전에는 같은 다섯 값을
    Profiler(LLM)가 추측해 `language_profile.json` 에 넣었는데, 코드는 실측을 쓰고
    프롬프트 작성기만 LLM 값을 봤다 — 26개 런 중 **25개에서 `trailing_punctuation`
    이 실측과 달랐다**. zh-en/run01 은 LLM 이 `] } "` 를 지어내고 실제로 있는
    `、 · - /` 를 빠뜨렸고, 그 목록이 그대로 프롬프트 규칙이 됐다.

    나머지 넷도 셀 수 있는 값이었다: `uses_spaces_between_words`(26/26 맞혔으나 코드는
    실측을 쓰므로 잉여), `punctuation_present`(자유서술. 3건은 "sometimes" 인데 실측
    1.000), `head_final`(39/39 가 `word_order` 로 결정 — SOV->True, SVO->False. 같은
    독일어 데이터에서 런마다 뒤집혔다), `source_code`(코드 참조 0건).

    **수치가 아니라 프롬프트가 쓸 수 있는 문장으로 준다** — 작성기는 이걸 근거로
    `[Never Segment]`·`[Priority Rules]` 를 쓰므로 "0.9947" 보다 "거의 모든 문장이
    구두점으로 끝난다"가 그대로 규칙이 된다.
    """
    if not measured:
        return ""
    # 옛 런의 `measured_profile.json` 에는 `unit` 이 없다(뒤에 추가된 필드). 이어 돌릴 때
    # 죽지 않도록 `uses_spaces_between_words` 에서 되살린다 — 같은 규칙으로 정해지는 값이다.
    is_spaced = bool(measured.get("uses_spaces_between_words"))
    unit = ("whitespace-separated words"
            if measured.get("unit", "word" if is_spaced else "char") == "word"
            else "characters")
    spaced = ("This language DOES separate words with spaces." if is_spaced else
              "This language does NOT put spaces between words.")
    trailing = measured.get("trailing_punctuation") or []
    rate = measured.get("punctuation_final_rate")
    if rate is None:
        punct_line = ""
    elif rate >= 0.9:
        punct_line = f"Almost every sentence ends in punctuation ({rate:.0%})."
    elif rate >= 0.1:
        punct_line = (f"Only {rate:.0%} of sentences end in punctuation — most utterances "
                      f"simply stop. Rules that depend on punctuation will not fire.")
    else:
        punct_line = (f"Sentences essentially never end in punctuation ({rate:.0%}). "
                      f"Do not write rules that rely on it.")
    lines = [
        "Measured facts about this corpus (counted directly from the text, not estimated — "
        "these override anything the profile implies):",
        f"- {spaced} Segment length is therefore counted in {unit}.",
    ]
    if punct_line:
        lines.append(f"- {punct_line}")
    if trailing:
        lines.append(
            "- These characters attach to the text BEFORE them and must never start a new "
            f"segment: {' '.join(trailing)}. This list is exhaustive for this corpus — do "
            "not add characters to it from general knowledge of the language.")
    else:
        lines.append("- No punctuation attaches to the preceding text in this corpus.")
    return "\n".join(lines)


PROMPT_WRITER_SYSTEM = """You write system prompts for a meaning-based segmentation model.

HARD CONSTRAINT — the prompt must be TARGET-LANGUAGE-AGNOSTIC.
The same prompt is reused for every target language, so it may not name a target language
and may not justify any rule with a target language's grammar (case, gender, articles,
verb-final order, agreement). Segmentation is decided on the SOURCE text alone.
Express risk the target-neutral way instead: "the following words can still overturn what
was already emitted". That statement is true for every target; "German case assignment"
is not.

Given a language profile, write the initial segmentation system prompt.

Downstream mechanism you must write for: the model marks EVERY defensible boundary and ranks
them by confidence. A deterministic step afterwards keeps only the top-ranked ones, as many as
the current latency budget allows. So the model is never choosing HOW MANY pieces to make —
that is decided later. It is choosing WHERE boundaries can go and WHICH ones are safest.

THE MOST IMPORTANT THING TO GET RIGHT — candidacy and ranking are separate decisions.

A position is a CANDIDATE if the text before it is a coherent stretch a translator could render
without inventing content that has not arrived. That is a low bar, and most positions between
phrases clear it. How likely the later text is to force a revision decides the RANK, not
whether the boundary is marked at all.

Marking a boundary is FREE. A risky one that ends up ranked last will simply never be kept
under a tight budget. Withholding it, by contrast, permanently removes an option and can only
make the system slower. So the failure you must avoid is under-marking, not over-marking.

Concretely, when you write the prompt:
- NEVER write a rule of the form "never place a boundary before/until X". If X is risky, say
  it ranks LOW in [Priority Rules]. [Never Segment] is only for positions that are wrong at
  ANY budget — inside a word, inside a self-repair fragment, or where the preceding text is
  not a renderable stretch at all.
- Do NOT require that the emitted translation survive unchanged once the rest arrives. That
  test is far too strict for a head-final source language and would leave almost every sentence
  unsegmented. It belongs in [Priority Rules] as a ranking signal, never as an admission test.
- A spontaneous-speech sentence of N words should typically receive on the order of N/{min_t}
  marked boundaries. If your rules would leave a 20-word utterance with one or two tags, they
  are too restrictive — loosen them.

Hard requirements:
- The prompt MUST contain these section headers verbatim, in this order:
  [Role], [Core Principles], [When to Segment], [Never Segment], [Priority Rules],
  [Decision Procedure], [Output Rules], [Examples — Segment], [Examples — Do NOT Segment]
- The [Output Rules] section MUST be copied verbatim from the block given to you. Do not
  reword it — a deterministic validator depends on it exactly as written.
- [When to Segment] and [Never Segment] must cite CONCRETE surface forms of the source
  language from the profile, not generic advice like "split at clause boundaries".
- [Priority Rules] must say what makes one boundary MORE confident than another in this
  language — which surface forms are safest to cut after, and which are riskier but still
  allowed. Rank by how little the following text can change the meaning already emitted.
  Every position you were tempted to forbid belongs here, at the bottom of the ranking.
- [Decision Procedure] must be two steps in this order: (1) mark every position where the
  preceding stretch is renderable on its own — be generous; (2) rank the marked positions by
  how much the remaining text could still overturn what was emitted. Step 1 must not consult
  step 2's risk judgement.
- Examples must be written in the source language with numbered tags inserted, in the form:
    Input: <sentence>
    Output: <sentence with numbered tags>
  Give 6-10 examples per example section. Invent realistic sentences in the source language.
- Mark every boundary that is defensible, then rank. Do NOT hold boundaries back out of
  caution — holding back cannot improve the score, it only removes options from the ranking.

Return ONLY the prompt text. No commentary, no code fences."""


@dataclass
class Profiler:
    gw: Gateway

    def profile(self, samples: list[str], target_language: str | None = None) -> dict:
        """소스 언어만 프로파일한다. `target_language` 는 호환용이며 **쓰지 않는다**.

        분절은 소스 쪽 문제라는 것이 설계 전제인데, 타깃 언어명을 넘기면 LLM 이 측정되지
        않은 타깃 문법 지식을 프롬프트에 써넣는다 — run04 산출물에 독일어 격·성 근거가
        8곳 들어갔고 순위 규칙 8~11(Medium/Lower 구간 전체)이 그 위에 세워졌다.
        `core/CLAUDE.md` 의 "언어 지식은 측정으로만" 원칙과 어긋난다.
        """
        user = (
            "The segmentation prompt you are profiling for must work for ANY target "
            "language. Describe only properties of the SOURCE text.\n\n"
            f"Source sentences ({len(samples)} samples):\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(samples))
        )
        return self.gw.chat_json(PROFILER_SYSTEM, user, max_tokens=PROFILER_MAX_TOKENS,
                                 purpose="profiler")

    def initial_prompt(self, profile: dict, target_language: str | None, spaced: bool,
                       min_t: int = 3, min_gap: int = 0,
                       measured: dict | None = None) -> str:
        """`target_language` 는 **의도적으로 쓰지 않는다** (Profiler.profile 참조).

        `measured` 는 실측 프로파일이다. 종전에는 작성기가 구두점 목록을 LLM 프로파일
        에서 받았는데 그게 검증기가 쓰는 실측과 25/26 런에서 달랐다 (`measured_facts`).
        """
        # 밀도 지침(N/{min_t})은 검증기의 커버리지 요건과 **같은 값**이어야 한다.
        # run03 에서 지침 N/3 vs 요건 N/2 불일치가 1차 통과율을 깎았다 (재시도로 복구되나
        # 프롬프트 품질 신호인 1차 통과율이 오염된다). 시스템 프롬프트에 JSON 중괄호가
        # 많아 .format 은 못 쓰고 표적 치환만 한다.
        sys_p = PROMPT_WRITER_SYSTEM.replace("N/{min_t}", f"N/{min_t}")
        facts = measured_facts(measured)
        user = (
            f"Language profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            + (facts + "\n\n" if facts else "")
            + "The prompt must be TARGET-LANGUAGE-AGNOSTIC. It will be reused for many "
            "different target languages without modification, so it may not name one, "
            "nor lean on one's grammar (no case, gender, article, or word-order arguments "
            "that belong to a specific target).\n\n"
            f"Copy this [Output Rules] section verbatim into the prompt:\n\n{output_rules(spaced, min_t, min_gap)}"
        )
        # thinking 모델은 사고 토큰이 max_tokens 에 같이 잡힌다 (SEG_MAX_TOKENS 와 같은
        # 문제). 6000 에서는 사고가 예산을 먹고 프롬프트 꼬리 섹션이 잘렸다 — run04 에서
        # 재시도까지 연속으로 [Examples — Do NOT Segment] 가 누락된 채 통과할 뻔했다.
        return self.gw.chat(sys_p, user, max_tokens=PROMPT_MAX_TOKENS,
                            purpose="prompt_v0")


# ── A7 Judge — 경계별 조기 방출 판정 ──────────────────────────────────────────────

# 왜 별도 에이전트인가: `adequacy` 는 `(조각 원문, 조각 번역)` 만 본다. "그건 문제가 →
# That's a problem" 은 그 조각의 번역으로 완벽하므로 만점이 나온다. 뒤에 "안 될 것
# 같은데" 가 오면 t1 에 사용자가 본 것은 정반대 의미인데, 무수정 제약상 되돌릴 수 없다.
# `consistency` 도 못 잡는다 — 최종 합본만 보기 때문이다. **방출 시점의 중간 상태를
# 보는 판정이 따로 필요하다.**
#
# 왜 LLM 인가: 접두사 검사(MU)는 offline 어순을 기준으로 삼아 "좋은 경계인데 어순만
# 다른 경우"를 오탐한다. NLI 는 이진이지만 이유와 이동 방향이 안 나온다. LLM 만이
# **"어순 차이는 무시하라"를 지시로 배제**할 수 있고, 진단을 곧 편집 지시(shift)로
# 바꿔 준다.
#
# 목적함수에는 들어가지 않는다. Critic 의 입력을 고르고 이유를 붙이는 역할만 한다.
JUDGE_SYSTEM = """You judge whether a streaming translation segment was emitted TOO EARLY.

Setup: a sentence is cut into pieces. Each piece is translated as soon as it arrives, seeing
only the pieces before it, and once emitted it can NEVER be revised. You are given an oracle:
the translation of the whole sentence, made with full knowledge of what comes after.

You judge ONE boundary at a time. The hypothesis is everything the user has seen up to that
moment — all emitted pieces concatenated, in order.

WHAT DOES NOT COUNT AS A PROBLEM. Ignore all of these:
- Different word order from the oracle. Streaming output is expected to be more monotonic than
  an offline translation. Reordering is not an error.
- Different wording, register, or phrasing that carries the same proposition.
- Incompleteness. A hypothesis that simply stops mid-thought ("the meeting materials for
  tomorrow") is fine — the rest is still coming. Missing information is NOT prematurity.

WHAT COUNTS. Exactly two things:
1. premature — the emitted text asserts a proposition that the rest of the sentence
   CONTRADICTS. Typical mechanisms: polarity settled before the negation arrived; the wrong
   participant assigned to the action; a modifier attached to the wrong scope; a head that had
   not arrived yet was guessed.
2. mistranslated — the emitted text is simply a wrong rendering of its own source piece,
   independent of anything that follows.

If neither applies, the verdict is "safe" — even if the hypothesis is short, awkward, or
ordered differently from the oracle.

Use "reference_suspect" when the oracle translation is itself wrong, so the loop does not chase
a bad reference.

For anything not "safe", say where the boundary should have gone instead. Express it as a shift
to the RIGHT (later) by a number of words, and name the source words it should now follow.

Return ONLY JSON:
{
  "verdict": "safe | premature | mistranslated | reference_suspect",
  "conflict": "the proposition that clashes with the oracle, in one short clause; null if safe",
  "cause": "polarity not yet settled | wrong participant | modifier scope | head not yet arrived | referent lost | other",
  "shift": {"units": 2, "to_after": "the source words the boundary should follow"},
  "generalized_rule": "one rule that would prevent this class of boundary, phrased for unseen sentences"
}

"cause", "shift" and "generalized_rule" must ALL be null when the verdict is "safe".
Never quote more than 40 characters of source text. Keep every field short."""


@dataclass
class Judge:
    gw: Gateway

    def judge(self, src_sentence: str, oracle: str, pieces_src: list[str],
              pieces_tgt: list[str], boundary: int) -> dict:
        """경계 `boundary` (0-based, 조각 boundary 와 boundary+1 사이) 를 판정한다.

        hypothesis 를 **누적**으로 넘기는 이유: 조각 하나만 보면 중립인데 앞 조각과
        합쳐져야 모순이 되는 형태가 있다. 사용자가 보는 것이 누적이므로 판정도
        누적이어야 한다. 귀속은 방금 추가된 조각을 따로 넘겨 지목하게 한다."""
        upto = boundary + 1
        user = (
            f"Full source sentence:\n{src_sentence}\n\n"
            f"Oracle (whole-sentence translation):\n{oracle}\n\n"
            f"Emitted so far (what the user has seen at this moment):\n"
            + "\n".join(f"[{i+1}] {s}  ->  {t}"
                        for i, (s, t) in enumerate(zip(pieces_src[:upto], pieces_tgt[:upto])))
            + f"\n\nHypothesis (concatenated): {' '.join(pieces_tgt[:upto]).strip()}\n"
            f"The piece just added: {pieces_src[boundary]}  ->  {pieces_tgt[boundary]}\n"
            f"Source still unseen: {' '.join(pieces_src[upto:])}"
        )
        return self.gw.chat_json(JUDGE_SYSTEM, user, max_tokens=JUDGE_MAX_TOKENS,
                                 purpose="judge")


def max_contra(row: dict, T: int) -> float | None:
    """문장 안 경계 중 가장 크게 반박당한 값. 경계별 점수가 없으면 `None`."""
    ps = (row.get("by_T", {}).get(str(T)) or {}).get("pieces_contra")
    return max(ps) if ps else None


def rank_by_failure(rows: list[dict], T: int, max_rows: int) -> list[dict]:
    """실패 유형 **둘 다** 대표되도록 예산을 반씩 나눈다.

    `adequacy` 최하위만 뽑으면 조기 방출(F2)이 통째로 빠진다 — 실측에서 두 순위는
    무상관이고 최상위에서는 오히려 반대였다 (모순 1위 문장이 adequacy 로는 중위권).
    거꾸로 모순 최상위만 뽑으면 조각 오역(F1)이 빠진다. 그래서 절반씩이다.

    경계별 점수가 없으면(`--no-contradiction`, 구버전 런) 예전대로 adequacy 만 쓴다."""
    if not rows or max_rows <= 0:
        return []
    by_adq = sorted(rows, key=lambda r: r["by_T"][str(T)].get("adequacy", 1.0))
    scored = [r for r in rows if max_contra(r, T) is not None]
    if not scored:
        return by_adq[:max_rows]
    by_con = sorted(scored, key=lambda r: -max_contra(r, T))

    picked, seen = [], set()
    for r in _interleave(by_con[: (max_rows + 1) // 2], by_adq):
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        picked.append(r)
        if len(picked) >= max_rows:
            break
    return picked


def _interleave(a: list[dict], b: list[dict]):
    """a 를 먼저 소진하지 않고 번갈아 낸다 — 어느 한 축이 예산을 독점하지 않게."""
    for i in range(max(len(a), len(b))):
        if i < len(a):
            yield a[i]
        if i < len(b):
            yield b[i]


def judge_rows(judge: Judge, rows: list[dict], T: int, max_rows: int = 8,
               sample: str = "failure", seed: int = 20260810) -> list[dict]:
    """N문장의 모든 경계를 판정한다. 마지막 경계는 대상이 아니다 — 뒤에
    미래가 없으므로 반박당할 수 없다.

    `sample="failure"` — 실패 의심 문장 조준 (`rank_by_failure`). 루프 중 Critic 입력용.
    `sample="random"`  — 시드 고정 무작위. **보고용 `premature_rate` 는 이쪽으로 재야
    한다** — 실패 조준 표본으로 재면 조건부 상향 추정치가 된다 (run03 test 0.2727 이
    그 값이다).
    """
    import random as _random
    out: list[dict] = []
    targets = [r for r in rows
               if r.get("by_T", {}).get(str(T), {}).get("pieces_tgt")
               and len(r["by_T"][str(T)]["pieces_tgt"]) > 1]
    if sample == "random":
        targets = _random.Random(seed).sample(targets, min(max_rows, len(targets)))
    else:
        targets = rank_by_failure(targets, T, max_rows)
    for r in targets:
        d = r["by_T"][str(T)]
        for b in range(len(d["pieces_tgt"]) - 1):
            try:
                v = judge.judge(r["text"], r.get("full_trans") or "",
                                d["pieces_src"], d["pieces_tgt"], b)
            except Exception as e:                      # 판정 실패로 루프를 죽이지 않는다
                v = {"verdict": "error", "conflict": str(e)[:200]}
            out.append({"id": r["id"], "boundary": b,
                        "seg_text": d["seg_text"], **v})
    return out


SCORED_VERDICTS = ("safe", "premature", "mistranslated")
UNSAFE_VERDICTS = ("premature", "mistranslated")


def premature_rate(judgements: list[dict]) -> float | None:
    """미래가 반박한 경계의 비율. `adequacy` 가 원리적으로 못 보는 축이다."""
    scored = [j for j in judgements if j.get("verdict") in SCORED_VERDICTS]
    if not scored:
        return None
    return sum(1 for j in scored if j["verdict"] == "premature") / len(scored)


def reference_suspect_rate(judgements: list[dict]) -> float | None:
    """판정자가 "오라클(full 번역) 자체가 틀렸다"고 본 경계의 비율.

    contradiction·consistency 는 full 번역을 정답지로 쓴다 — 정답지가 틀리면 옳은
    조각이 벌받고, 그 오염은 지표 숫자에서 안 보인다. 이 비율이 높으면 지표가 아니라
    번역기(오라클)를 의심해야 한다는 신호다."""
    judged = [j for j in judgements
              if j.get("verdict") in SCORED_VERDICTS + ("reference_suspect",)]
    if not judged:
        return None
    return sum(1 for j in judged if j["verdict"] == "reference_suspect") / len(judged)


def unsafe_rate(judgements: list[dict]) -> float | None:
    """`premature` + `mistranslated`. 둘 다 "이 경계를 고쳐야 한다"로 같은 행동을 부른다.

    라벨이 둘 사이에서 흔들려도 이 값은 안정적이다 — 관문 실측에서 같은 경계가
    `mistranslated` 1회 / `premature` 2회로 갈렸는데 `cause` 와 `conflict` 는 3회
    동일했다. 조향에 쓰이는 것은 세부 라벨이 아니라 "표시되었는가"다."""
    scored = [j for j in judgements if j.get("verdict") in SCORED_VERDICTS]
    if not scored:
        return None
    return sum(1 for j in scored if j["verdict"] in UNSAFE_VERDICTS) / len(scored)


# ── A8 Critic ────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You diagnose failures of a meaning-based segmentation prompt.

Setup: a model marks EVERY defensible boundary with a numbered tag <SEG:n>, ranked by
confidence (1 = most confident). A deterministic step then keeps only the top-ranked boundaries,
as many as the latency budget allows, and each resulting piece is translated in order — seeing
only the already-final translations before it, never what comes after, and never revisable.

Because the piece COUNT is decided by that later step and not by the prompt, "too many" and
"too few" tags are not failures you can diagnose. Only two things are:

- PLACEMENT — a boundary sits somewhere that damages the translation.
- PRIORITY — the boundaries are in defensible places, but ranked wrong, so the ones kept under
  a tight budget are the risky ones.

You do NOT assign scores — scores are computed separately. Your job is to explain WHY specific
cases failed and to propose GENERALISED rules that would prevent them.

MEASURED EVIDENCE — JUDGEMENTS. Cases may carry "judgements": a per-boundary verdict produced
by re-examining what the user had actually seen at that moment against an oracle translation of
the whole sentence.

  "premature"      the emitted text asserted something the rest of the sentence CONTRADICTS.
                   This is the boundary being in the wrong place. It is invisible to the
                   quality scores, because the later pieces repair the final concatenation.
  "mistranslated"  the piece was rendered wrong on its own terms.
  "safe"           fine. Different word order from the oracle is NOT a defect.

A "premature" verdict comes with "cause" and "shift" — where the boundary should have gone
instead. Turn that into a general condition, never into a rule about that one sentence.

MEASURED EVIDENCE — CONTRADICTION. Cases may carry "contradiction_after_each_piece": one number
per piece, the probability that the text visible AFTER that piece was emitted is contradicted by
the oracle translation of the whole sentence. It is a mechanical entailment check, not a
judgement, so it covers every boundary rather than a sampled few.

  near 0.0   the emitted text is incomplete but not wrong. Incompleteness is NOT a defect.
  near 1.0   the emitted text committed to something the rest of the sentence overturns —
             usually polarity, the main predicate, or a modal that had not arrived yet.

The LAST number of the list is always 0.0: nothing follows the final piece, so there is nothing
that could contradict it. Never read that as evidence the final boundary was good.

MEASURED EVIDENCE — PRIORITY. Cases may carry "adequacy_by_T", keyed by target piece size.
A LARGE key means few pieces, so only the TOP-RANKED boundaries survived. A SMALL key means
many pieces, so lower-ranked boundaries survived too.

  worse at a LARGE key than at a SMALL key  ->  PRIORITY problem. The positions are
                                                defensible; the ranking put a risky boundary
                                                at rank 1. Fix [Priority Rules].
  bad at every key                           ->  PLACEMENT problem. Fix [When to Segment] /
                                                [Never Segment] / [Decision Procedure].

Also flag reference_suspect when the oracle (whole-sentence) translation is itself wrong, so
the loop does not chase a bad reference.

Never propose banning a connective ending outright. The same ending is safe in most sentences;
a blanket ban removes boundaries from the ranking for a small quality gain. Your rule must state
the CONDITION that separates the safe uses from the harmful ones.

Never propose "mark fewer boundaries". Holding a boundary back cannot raise the score — it only
removes an option from the ranking. If a boundary is risky, it belongs LOW in the ranking, not
absent.

Return ONLY JSON:
{
  "cases": [
    {
      "id": "case id",
      "error_type": "placement | priority | format_violation | reference_suspect",
      "span": "the exact source substring where the problem is",
      "evidence": "what specifically went wrong at the moment of emission",
      "cause": "short mechanism, e.g. polarity not yet settled / modifier scope changed / head had not arrived",
      "proposed_rule": "one generalised rule, phrased so it applies to unseen sentences — never mention this specific sentence",
      "example_pair": {"input": "source sentence", "output": "source sentence with correct numbered tags"}
    }
  ],
  "summary": "2-3 sentences on what the current prompt is systematically getting wrong"
}

Keep every field short. Do not quote more than 40 characters of source text in any field."""


@dataclass
class Critic:
    gw: Gateway

    def review(self, cases: list[dict], metrics: dict, violations: list[dict],
               avoid: str | None = None,
               priority_audit: list[dict] | None = None) -> dict:
        user = (
            f"Current metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"(adequacy = quality of each piece against ITS OWN source, with no reference "
            f"translation — word order differences from an offline translation do not lower "
            f"it. consistency = similarity of the concatenated pieces to the whole-sentence "
            f"translation, reported only. laal_words = lag in source words, lower is faster; "
            f"it is set by the latency budget, NOT by the prompt. by_T is keyed by the target "
            f"piece size: a LARGE key means few pieces, so only the TOP-RANKED boundaries "
            f"survive; a SMALL key means many pieces, so lower-ranked boundaries survive too. "
            f"missing_boundaries is how many boundaries the prompt failed to provide when "
            f"the budget asked for more.)\n\n"
            f"Format violations ({len(violations)}):\n"
            f"{json.dumps(violations[:10], ensure_ascii=False, indent=2)}\n\n"
            f"Cases to diagnose:\n{json.dumps(cases, ensure_ascii=False, indent=2)}"
        )
        # 순위 감사 — 모델이 **어떤 종류의 위치를 과신하는지**. gap 이 음수라는 사실만으로는
        # [Priority Rules] 의 어느 줄이 틀렸는지 알 수 없다 (metrics.priority_audit).
        if priority_audit:
            user += (
                "\n\nRanking audit — for each surface feature: the AVERAGE CONFIDENCE RANK "
                "the prompt assigned (0 = ranked most confident, 1 = ranked least) versus the "
                "MEASURED contradiction at those boundaries. A feature with a LOW rank "
                "percentile but a HIGH contradiction is one the prompt over-trusts: it tells "
                "the model these positions are safe when the measurement says they are not. "
                "Cite the feature by name when you propose a priority rule.\n"
                + json.dumps(priority_audit, ensure_ascii=False, indent=2))
        out = self.gw.chat_json(CRITIC_SYSTEM, user, max_tokens=PROMPT_MAX_TOKENS,
                                purpose="critic")
        out["aggregate"] = summarize_critique(out.get("cases") or [], metrics,
                                              out.get("summary"), avoid, priority_audit)
        return out


# 임계값은 **잠정값**이다. 실측 전에 확정하면 v1 의 `q_weight`·`ratio` 처럼 근거 없는
# 상수가 하나 더 생긴다. 첫 런들의 분포를 보고 고정할 것.
MISSING_BOUNDARIES_LIMIT = 0.5       # 문장당 평균 부족 경계 수
PREMATURE_LIMIT = 0.15      # 판정 대상 경계 중 조기 방출 비율

# 순위축 조향 문턱. `rank_lift` 가 **오차 한 칸도 못 넘으면** 순위가 값을 못 하는 것으로
# 본다 (`lift < 1·se`, 즉 t < 1).
#
# **부등호 방향이 종전과 반대다.** 예전 게이트는 `gap + 1·se <= 0`, 즉 사실상 `t <= −1`
# 이라 "순위가 **해롭다**는 증거"를 요구했다. 그런데 잡고 싶은 상태는 "이득이 없다"이지
# "해롭다"가 아니다 — 그래서 순위가 정말 무가치할 때조차 다섯 번에 한 번밖에 안 울렸다.
#
# 문턱은 실측 발화율로 잡았다 (metric_probes/runs/rank_ablation/, 두 언어쌍 × 최대 T):
#   순위가 값을 하는 상태(real vs 셔플)에서 오작동  0/40
#   순위가 무가치한 상태(셔플 vs 셔플, 참값 0)에서 검출  156/190, 164/190 (82~86%)
# 상수 1.0 은 종전 `RANK_GAP_SE_MULT` 를 그대로 옮긴 것이라 새로 생긴 임의 상수가 아니다.
RANK_LIFT_T_MIN = 1.0


def summarize_critique(cases: list[dict], metrics: dict, summary: str | None,
                       avoid: str | None = None,
                       priority_audit: list[dict] | None = None) -> dict:
    """집계는 세는 일이지 판단이 아니다 — LLM 에 맡기면 누락되거나 틀린다.

    `focus` 는 **측정된 지표**에서 도출한다. 사례 카운트로 정하면 안 된다: Critic
    에게는 망가진 사례를 골라 보내므로 특정 유형이 항상 다수가 되고, 방향이 영구히
    거기 고정된다 (v1 실측: direction 5회 고착, 분절률 0.72 -> 0.38).

    v1 의 "더/덜 잘라라" 방향이 사라진 자리가 크다. 조각 수는 노브가 정하므로
    과소분절·과분절이라는 실패 자체가 없다. 남는 것은 위치와 순위뿐이다.

    **순위 축은 순위를 망가뜨려 잰다** (`rank_lift`). 종전에 쓰던 `rank_contra_gap` 은
    **절단 후 살아남은 경계들끼리의 순서**만 보는데, 순위가 실제로 하는 일은 keep-vs-discard
    다 — 폐기된 경계는 렌더링이 없어 contra 값 자체가 없으므로 원리적으로 안 보인다
    (en-de run04 T=6 실측: 후보 15.4개 중 생존 2.7개, **결정의 82% 가 지표 밖**).
    실측에서 두 값은 어긋났다: 순위를 섞으면 effective 가 0.024~0.061 떨어지는데
    (20/20 셔플 완승, 순열 p=0.048) `rank_contra_gap` 은 en-de 에서 오히려 무작위보다
    낮게 나왔다. 근거: `../metric_probes/runs/rank_ablation/`, `docs/RANK_METRIC_DIAGNOSIS.md`.
    """
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.get("error_type", "unknown")] = counts.get(c.get("error_type", "unknown"), 0) + 1

    by_T = metrics.get("by_T") or {}
    # by_T 는 **비어 있을 수 있다** — 포맷이 무너져 저비용 게이트가 번역을 건너뛰면
    # (`evaluate` 의 `skip_translation_below`) 지표가 하나도 안 만들어진다. 예전에 여기서
    # 빈 dict 를 인덱싱해 Critic 호출 전체가 실패하고 루프가 최종 평가로 튕겨나간 적이
    # 있다 — run04 iter0 (train fmt=0.93, by_T={}). 정작 그 상황의 판정은 첫 분기(format)
    # 라 by_T 가 필요 없으므로, 아래는 전부 빈 dict 에서도 도는 형태로만 쓴다.
    missing = max((v.get("missing_boundaries") or 0.0) for v in by_T.values()) if by_T else 0.0
    prem = max((v.get("premature_rate") or 0.0) for v in by_T.values()) if by_T else 0.0

    # 순위 진단은 **순위를 망가뜨려 본다** (`metrics.rank_lift`). 절단기가 순위를 쓰는
    # 곳은 keep-vs-discard 한 군데뿐이므로, 그 결정만 무작위로 바꿔 손실을 재는 것이
    # 순위축의 직접 측정이다. `loop.evaluate` 가 최대 T 에서 한 번 계산해 싣는다.
    lift = metrics.get("rank_lift")
    lift_t = metrics.get("rank_lift_t")

    focus_reason = ""
    if metrics.get("format_pass_rate", 1.0) < 1.0:
        focus = "format"
        focus_reason = f"format_pass_rate {metrics.get('format_pass_rate'):.4f} < 1.0"
    # 예산이 요구한 경계를 프롬프트가 못 내놓으면 순위도 위치도 논할 수 없다.
    elif missing > MISSING_BOUNDARIES_LIMIT:
        focus = "coverage"
        focus_reason = f"missing_boundaries {missing:.3f} > {MISSING_BOUNDARIES_LIMIT}"
    # 순위를 무작위로 섞어도 품질이 안 떨어지면 순위가 값을 못 하는 것 = 순위 문제.
    # **폴백은 두지 않는다.** 종전의 T 대비(`adequacy(작은 T) − adequacy(큰 T)`)는 중첩
    # 집합 비교인 데다 QE 길이 편향이 섞여 있어, 근거 없는 조향을 만드는 경로였다.
    # 값이 없으면(--no-contradiction, 순위 없는 프롬프트) 순위축은 판단하지 않는다.
    elif lift_t is not None and lift_t < RANK_LIFT_T_MIN:
        focus = "priority"
        focus_reason = (f"rank_lift {lift:+.4f} (t {lift_t:+.2f}) — 순위를 무작위로 "
                        + ("섞으면 품질이 오히려 오름" if (lift or 0.0) < 0 else
                           "섞어도 품질이 안 떨어짐"))
    # 종전에는 이 두 갈래가 **같은 값을 넣어** PREMATURE_LIMIT 이 죽은 코드였다.
    # 결론은 어차피 placement 지만 **근거가 다르다** — 측정된 조기방출이냐, 아무 지표도
    # 안 걸린 기본값이냐. PE 가 그 차이를 알아야 확신 없는 개정을 덜 한다.
    elif prem > PREMATURE_LIMIT:
        focus = "placement"
        focus_reason = f"premature_rate {prem:.4f} > {PREMATURE_LIMIT}"
    else:
        focus = "placement"
        focus_reason = ("측정 지표가 아무것도 안 걸림 — 기본값. "
                        "확실한 근거가 없으니 작은 수정만 할 것")

    # **고착 방지.** 같은 방향이 반복되는데 dev 가 나아지지 않으면 탐색이 죽는다.
    #
    # 단 **없는 근거를 만들어내지는 않는다.** 종전에는 placement 가 막히면 지표를 아예
    # 안 보고 priority 로 뒤집었는데, 순위가 이미 값을 하고 있는 상태에서 그쪽을 고치라고
    # 보내는 것은 잘 돌아가는 섹션을 건드리게 하는 것이다 — 실측상 순위는 effective
    # +0.024~0.061 을 벌고 있다 (metric_probes/runs/rank_ablation/).
    # priority 로 갈 근거가 있었으면 위 분기에서 이미 그렇게 됐으므로, 여기서 남은 경우는
    # "근거가 없다"뿐이다. 그때는 축을 바꾸는 대신 **직전 실패를 PE 에게 알린다.**
    if avoid and focus == avoid:
        if focus == "priority":
            focus = "placement"
            focus_reason = f"고착 방지 — 직전 {avoid} 가 채택 실패해 방향 전환"
        else:
            focus_reason += (" | 고착 — 직전 개정이 채택 실패했다. 순위축으로 옮길 근거는 "
                             "없으니 같은 축에서 **다른 각도**로 볼 것")

    return {
        "dominant_error": max(counts, key=counts.get) if counts else None,
        "error_counts": counts,
        "focus": focus,
        "focus_reason": focus_reason,
        "priority_audit": (priority_audit or [])[:6],
        "max_missing_boundaries": round(missing, 4),
        "max_premature_rate": round(prem, 4),
        "rank_lift": lift,
        "rank_lift_t": lift_t,
        "summary": summary,
    }


def select_cases(rows: list[dict], main_T: int, judgements: list[dict] | None = None,
                 n_worst: int = 5, n_invalid: int = 5, n_short: int = 3) -> list[dict]:
    """비평 대상 선정 — 전량이 아니라 정보량이 있는 것만.

    v1 은 "COMET 하위 N개 문장"이었다. v2 는 **조기 방출로 판정된 경계**를 우선한다 —
    점수가 아니라 위치와 이유가 붙어 있으므로 PE 가 바로 고칠 수 있다.
    과소분절 쿼터는 없앴다. 조각 수를 노브가 정하므로 그 실패 유형이 존재하지 않는다.

    `n_worst` 쿼터는 `rank_by_failure` 로 뽑는다. `adequacy` 최하위만 보면 조기 방출이
    빠지는데, `flagged` 가 그걸 메워주지 못한다 — `flagged` 는 판정자가 이미 본 문장에서만
    나오므로 판정자 조준이 놓친 문장은 여기서도 못 본다.
    """
    key = str(main_T)
    by_id = {r["id"]: r for r in rows}
    judged: dict[str, list[dict]] = {}
    for j in (judgements or []):
        judged.setdefault(j["id"], []).append(j)

    invalid = [r for r in rows if not r["valid"]][:n_invalid]
    flagged = [by_id[i] for i, js in judged.items()
               if i in by_id and any(j.get("verdict") in ("premature", "mistranslated")
                                     for j in js)]
    valid = [r for r in rows if r["valid"] and r.get("by_T", {}).get(key)]
    worst = rank_by_failure(valid, main_T, n_worst)
    short = sorted([r for r in valid if r["by_T"][key].get("missing_boundaries", 0) > 0],
                   key=lambda r: -r["by_T"][key]["missing_boundaries"])[:n_short]

    picked, seen = [], set()
    for r in invalid + flagged + worst + short:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        adequacy_by_T = {t: d.get("adequacy") for t, d in (r.get("by_T") or {}).items()}
        picked.append({
            "id": r["id"], "text": r["text"], "seg_text": r["seg_text"],
            "valid": r["valid"], "full_trans": r.get("full_trans"),
            "segmentation_at_main_budget": (r.get("by_T", {}).get(key) or {}).get("seg_text"),
            "pieces_tgt": (r.get("by_T", {}).get(key) or {}).get("pieces_tgt"),
            "adequacy_by_T": adequacy_by_T,
            "contradiction_after_each_piece":
                (r.get("by_T", {}).get(key) or {}).get("pieces_contra"),
            "judgements": [{k: v for k, v in j.items() if k != "seg_text"}
                           for j in judged.get(r["id"], [])],
            "selected_because": (
                "format violation" if not r["valid"]
                else "a boundary was judged premature or mistranslated"
                     if r["id"] in judged and any(
                         j.get("verdict") in ("premature", "mistranslated")
                         for j in judged[r["id"]])
                else "the prompt did not provide enough boundaries for the budget"
                     if (r.get("by_T", {}).get(key) or {}).get("missing_boundaries", 0) > 0
                else "a boundary was strongly contradicted by the rest of the sentence"
                     if (max_contra(r, main_T) or 0.0) >= 0.5
                else "lowest adequacy at the main budget"),
        })
    return picked


# ── A10 Compressor ───────────────────────────────────────────────────────

# 길이 예산을 넘긴 개정본을 줄인다. PE 에게 "짧게 다시 써라"를 맡기지 않는 이유는
# 개선과 축소를 한 호출에 섞으면 **방금 추가한 규칙을 스스로 지우기** 때문이다.
# PE 는 이미 그 이터레이션의 2섹션 예산을 썼다. 압축기는 이번에 바꾼 섹션을 보호하고
# 누적된 옛 규칙에서만 깎으므로, 과적합 기전(규칙 누적)을 정확히 겨냥한다.
COMPRESSOR_SYSTEM = """You shorten a segmentation system prompt so it fits a length budget.

You are NOT improving it and NOT changing what it asks the model to do. You are removing
accumulated redundancy so the prompt stays a set of rules instead of growing into a case list.

Hard constraints — violating any of these makes your output unusable:
1. Keep the section skeleton exactly: [Role], [Core Principles], [When to Segment],
   [Never Segment], [Priority Rules], [Decision Procedure], [Output Rules],
   [Examples — Segment], [Examples — Do NOT Segment]. Same headers, same order.
2. Copy [Output Rules] verbatim. A deterministic validator depends on it exactly as written.
3. Do NOT weaken or remove anything in these sections — they hold the change being measured
   this iteration: {protected}
4. Cut in this order:
   a. examples that duplicate a condition already stated in prose,
   b. the weakest / most specific examples (each example section may keep at most 8),
   c. conditions that restate another condition in different words — merge them into one,
   d. hedging and repetition in [Core Principles].
5. NEVER delete a condition whose content is not covered anywhere else. If you cannot reach
   the budget without doing that, get as close as you can and stop.
6. Keep every remaining rule in the source language forms it already uses. Do not translate,
   generalise, or reword conditions beyond merging duplicates.

Target: at most {budget} characters. Current: {current} characters.

Return ONLY the prompt text. No commentary, no code fences."""


@dataclass
class Compressor:
    gw: Gateway

    def compress(self, prompt: str, budget: int, protected: list[str]) -> str:
        sys_p = COMPRESSOR_SYSTEM.format(
            protected=", ".join(protected) if protected else "(none)",
            budget=budget, current=len(prompt))
        return self.gw.chat(sys_p, prompt, max_tokens=PROMPT_MAX_TOKENS,
                            purpose="compressor")


# ── A9 Prompt Engineer ───────────────────────────────────────────────────

ENGINEER_SYSTEM = """You revise a meaning-based segmentation system prompt, one iteration at a time.

Hard constraints — violating any of these makes your output unusable:
1. Keep the section skeleton exactly: [Role], [Core Principles], [When to Segment],
   [Never Segment], [Priority Rules], [Decision Procedure], [Output Rules],
   [Examples — Segment], [Examples — Do NOT Segment]. Same headers, same order.
2. Copy [Output Rules] verbatim from the current prompt. It is frozen — a deterministic
   validator depends on it.
3. Change AT MOST TWO sections this iteration. Leave the rest byte-identical. A full rewrite
   makes it impossible to attribute the score change to anything.
   **A changed section may not grow beyond 1.25x its current length.** These two limits are
   enforced by a deterministic gate after you answer — a revision that breaks either is
   DISCARDED and the iteration is wasted. Edit the specific lines that are wrong; do not
   rewrite a section wholesale to express one new idea.
   Aim for the smallest edit that could plausibly move the metric. Measured failure: revisions
   that rewrote a section wholesale changed 62-95% of all segmentations, which destroyed the
   paired comparison's power (it only has signal on sentences whose segmentation changed) and
   were rejected 3 times out of 3.
4. At most 12 examples per example section. If you add one, remove a weaker one. The prompt
   must stay a set of rules, not a memorised dataset.
5. Consult the attempt history. Every entry with "adopted": false is a revision that was
   MEASURED AND REJECTED — its scores are shown. Do not repeat it or any minor variant of it.
   If your last attempt was rejected, move in the OPPOSITE direction within the section the
   focus points at, or edit a different rule inside it. **Rule 7 (focus) outranks this one** —
   never leave the focus's section just because your last attempt there failed. When focus is
   "priority" a gate rejects any edit outside [Priority Rules].
6. Rules must generalise. Never write a rule that names a specific sentence from the data.
   **Never name a target language or justify a rule with its grammar** (case, gender,
   articles, verb-final order, agreement). The prompt is reused for every target language;
   a deterministic gate rejects revisions that mention one. Say "the following words can
   still overturn what was emitted" instead — that holds for every target.
7. Obey the critic's "focus" field. It is computed from measurements, not opinion:
   - "format"    → touch the sections governing adherence and the decision procedure,
                   including how tags are numbered. Do NOT add new segmentation restrictions.
   - "coverage"  → the prompt is not marking enough boundaries to fill the latency budget.
                   Remove or narrow over-broad prohibitions and add concrete permissive rules
                   so more defensible positions get marked. Do NOT add restrictions. Marking a
                   boundary is free — a risky one can simply be ranked last.
   - "placement" → boundaries sit where they damage the translation. Edit [When to Segment],
                   [Never Segment] and [Decision Procedure] to describe better positions.
                   Use the judgement "cause" and "shift" evidence to say where the boundary
                   should have gone instead.
                   **Do not fix placement by forbidding more positions.** [Output Rules]
                   demands a minimum number of tags per sentence and that minimum does not
                   move; every prohibition you add makes it harder to reach, and output below
                   it is REJECTED and re-generated. A risky-but-defensible position belongs at
                   the BOTTOM of [Priority Rules], never in [Never Segment] — the truncator
                   drops it for free, whereas withholding it costs a retry and removes an
                   option the ranking could have used. [Never Segment] is only for positions
                   that are wrong at ANY latency budget.
                   Prefer rewriting a prohibition as a ranking signal. If your edit would
                   plausibly reduce how many boundaries the model marks, it is the wrong edit.
                   A prohibition leaks in through THREE sections, not one — guard all three:
                     * [Never Segment] — the obvious one.
                     * [When to Segment] — do NOT add negative admission conditions here
                       ("do not mark if the left side ends in a determiner / an auxiliary /
                       an unheaded modifier ..."). Phrased as an admission test they remove the
                       candidate entirely; the same linguistic fact belongs in [Priority Rules]
                       as a demotion, where it costs nothing.
                     * [Decision Procedure] step 1 — this step marks GENEROUSLY and must not
                       consult risk at all. Risk is step 2's job. Never move a coherence or
                       safety check into step 1, and never make step 1 "enforce" a rule from
                       another section.
                   Measured failure this guard exists for: a placement revision rewrote
                   [When to Segment] with coherence pre-checks and wired them into step 1;
                   first-pass `too_few_tags` went 5/40 -> 19/40 and retries went 25 -> 44 calls
                   (en-de run03 iter1), and the revision was rejected anyway.
   - "priority"  → the positions are defensible but ranked wrong: a risky boundary was given
                   rank 1, so it survives even under the tightest budget. Edit [Priority Rules]
                   ONLY. Do not move or remove any boundary — reorder the confidence criteria.
                   Use "priority_audit": each row pairs a surface feature with the average
                   confidence rank the current prompt gives it (0 = most confident) and the
                   MEASURED contradiction there. Demote features whose rank percentile is low
                   while contradiction is high — those are the rules that are actively wrong.
                   Do NOT rewrite the whole section; move the specific offending criteria.
   Also read "focus_reason". If it says the focus is a DEFAULT with no metric triggered,
   make a small, low-risk edit — there is no measured defect to chase.

MEASURED EVIDENCE IN THE CRITIQUE.

"judgements" — a per-boundary verdict from re-examining what the user had actually seen at that
moment against an oracle translation of the whole sentence. A "premature" verdict means the
emitted text asserted something the rest of the sentence contradicts, and the user could never
see it corrected. This is invisible to the quality scores because later pieces repair the final
concatenation. Each carries "cause" and "shift" — where the boundary should have gone.

"contradiction_after_each_piece" — one number per piece: the probability that the text visible
after that piece was emitted is contradicted by the oracle translation of the whole sentence.
Near 0.0 means incomplete but not wrong, which is fine; near 1.0 means the emission committed to
a polarity, predicate or modal that the rest of the sentence overturns. The last number is always
0.0 because nothing follows the final piece — never read it as evidence.

"adequacy_by_T" — the same sentence scored under different piece sizes. A LARGE key means few
pieces, so only top-ranked boundaries survived; a SMALL key means many pieces. Worse at a large
key than a small one is a RANKING failure, not a position failure.

- Turn evidence into a general condition stated in surface forms of the source language.
  "A tag may follow a discourse filler when a full subject+verb clause follows it" is a rule.
  "Split sentence X after word 3" is not — never write that.
- NEVER respond to a bad case by telling the model to mark fewer boundaries. The piece count is
  set by a separate deterministic step, not by the prompt. Holding a boundary back cannot raise
  the score; it only removes an option from the ranking. If a boundary is risky, it belongs LOW
  in [Priority Rules], not in [Never Segment].
- [Never Segment] is only for positions that are wrong at ANY budget.

Remember the objective:

    score = average adequacy across latency budgets

where adequacy scores each piece against ITS OWN source text, with no reference translation.
The only hard condition is 100% valid output format.

Three consequences:
- Word order differing from an offline translation costs nothing. Do not add rules that try to
  preserve the offline word order.
- Latency is NOT in the score. You cannot gain by splitting more or lose by splitting less —
  the budget decides that. What you control is WHERE boundaries may go and WHICH are safest.
- A boundary that is safe only sometimes should still be marked, and ranked below the ones that
  are always safe.

Return ONLY JSON:
{
  "sections_changed": ["[When to Segment]", "..."],
  "changelog": ["one line per change, stating what and why"],
  "prompt": "the complete revised prompt text"
}"""


@dataclass
class PromptEngineer:
    gw: Gateway

    def revise(
        self,
        current_prompt: str,
        critique: dict,
        history: list[dict],
        profile: dict,
        t_grid: list[int],
        only_rule: str | None = None,
        max_sections: int | None = None,
        max_growth: float | None = None,
        measured: dict | None = None,
    ) -> dict:
        """`only_rule` 이 있으면 **그 규칙 하나만** 반영하게 한다.

        Critic 은 케이스마다 `proposed_rule` 을 낸다. 그걸 한 번에 다 넣으면 섹션이
        배로 부풀고(run03 iter1: +29%) 어느 규칙이 도움됐는지 영영 알 수 없다 —
        점수는 하나인데 규칙은 여럿이라 신용 배분이 안 된다. 규칙 하나짜리 개정을
        여러 개 만들어 **probe 로 고르면** 그 둘이 동시에 풀린다.
        """
        hist = json.dumps(history[-8:], ensure_ascii=False, indent=2)
        facts = measured_facts(measured)
        user = (
            f"Language profile (fixed):\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            + (facts + "\n\n" if facts else "")
            + f"Latency budgets in use (target piece size in source words): {t_grid}. "
            f"A budget of T keeps roughly (sentence length / T) pieces, so the LARGEST T "
            f"exercises only your highest-ranked boundaries.\n\n"
            f"Attempt history (prompt version -> scores, and whether it was adopted):\n{hist}\n\n"
            f"Critic feedback on the current prompt:\n{json.dumps(critique, ensure_ascii=False, indent=2)}\n\n"
            f"=== CURRENT PROMPT ===\n{current_prompt}"
        )
        if only_rule:
            user += (
                "\n\n=== THIS REVISION'S SINGLE TARGET ===\n"
                "Implement EXACTLY ONE change, expressing this idea and nothing else:\n"
                f"  {only_rule}\n"
                "Ignore every other proposal in the critique for now — they are being tried "
                "separately and the results are compared. Adding more than this one idea makes "
                "the comparison meaningless. Touch ONE section if you can."
            )
        # **지시문과 게이트가 같은 숫자를 말해야 한다.** 실측상 PE 는 이 값에 비례해
        # 반응하지 않지만(1.25/2.5/4.0 지시 → 1.39/1.36/1.79 산출), 게이트가 2.5 인데
        # 지시문이 1.25 라고 말하는 상태는 유지보수를 망가뜨린다.
        sys_p = ENGINEER_SYSTEM
        if max_sections is not None:
            sys_p = sys_p.replace("Change AT MOST TWO sections",
                                  f"Change AT MOST {max_sections} sections")
        if max_growth is not None:
            sys_p = sys_p.replace("may not grow beyond 1.25x its current length",
                                  f"may not grow beyond {max_growth:.2f}x its current length")
        return self.gw.chat_json(sys_p, user, max_tokens=PROMPT_MAX_TOKENS,
                                 purpose="prompt_engineer")
