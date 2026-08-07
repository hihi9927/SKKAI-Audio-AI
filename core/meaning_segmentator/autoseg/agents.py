"""LLM 에이전트 3종 — Language Profiler / Critic / Prompt Engineer.

루프에서 LLM 판단이 들어가는 곳은 여기뿐이다. 포맷 검증·점수 계산·채택 판정·
재시도는 전부 결정론적 코드로 처리한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .gateway import Gateway

# ── 프롬프트 골격 ────────────────────────────────────────────────────────
# 골격은 고정한다. Prompt Engineer는 섹션 '내용'만 바꾼다 — 구조를 흔들면 루프가
# 발산하고, 점수 변화를 어떤 변경에 귀속시킬 수 없게 된다.

REQUIRED_SECTIONS = [
    "[Role]",
    "[Core Principles]",
    "[When to Segment]",
    "[Never Segment]",
    "[Decision Procedure]",
    "[Output Rules]",
    "[Examples — Segment]",
    "[Examples — Do NOT Segment]",
]

OUTPUT_RULES_SPACED = """[Output Rules]
- Insert <SEG> tags only. Do NOT change, correct, or paraphrase the original text in any way.
- No tag at the very start or the very end of the text.
- Never place two tags consecutively.
- Punctuation must stay attached to the text before it — never immediately after a tag.
- Always place exactly one space on both sides of every <SEG> tag.
- If no segmentation is needed, output the original text unchanged.
- Output the tagged text and nothing else. No explanation, label, or commentary."""

OUTPUT_RULES_UNSPACED = """[Output Rules]
- Insert <SEG> tags only. Do NOT change, correct, or paraphrase the original text in any way.
- The source script does not use spaces between words. Do NOT add, remove, or move any
  character of the original text — the only thing you add is the tag itself.
- No tag at the very start or the very end of the text.
- Never place two tags consecutively.
- Punctuation must stay attached to the text before it — never immediately after a tag.
- Always place exactly one space on both sides of every <SEG> tag. These two spaces are the
  only whitespace you may introduce.
- If no segmentation is needed, output the original text unchanged.
- Output the tagged text and nothing else. No explanation, label, or commentary."""


def output_rules(spaced: bool) -> str:
    return OUTPUT_RULES_SPACED if spaced else OUTPUT_RULES_UNSPACED


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

Return ONLY a JSON object with exactly these keys:
{
  "source_language": "language name",
  "source_code": "ISO 639-1 code",
  "uses_spaces_between_words": true or false,
  "word_order": "SOV | SVO | VSO | other, with a one-line justification from the sample",
  "head_final": true or false,
  "punctuation_present": "always | sometimes | never — as observed in the sample",
  "trailing_punctuation": ["characters that attach to the END of the preceding text and must never start a new segment, e.g. sentence-final and clause-separating marks and closing quotes/brackets in this language. Give the characters themselves, no descriptions. Empty list if the language writes none."],
  "register": "what kind of text this is (spontaneous speech, read-aloud prose, meeting, ...)",
  "fillers_and_hesitations": ["actual tokens observed or expected in this language"],
  "discourse_markers": ["tokens that open or link clauses"],
  "clause_boundary_signals": ["concrete surface forms that mark a clause boundary in THIS language"],
  "non_boundary_traps": ["surface forms that LOOK like boundaries but are not"],
  "target_language_risks": ["what breaks when a fragment of this language is translated into the target without following context"],
  "notes_for_prompt_writer": "2-4 sentences of practical guidance"
}
No prose outside the JSON."""

PROMPT_WRITER_SYSTEM = """You write system prompts for a meaning-based segmentation model.

Given a language profile, write the initial segmentation system prompt.

Hard requirements:
- The prompt MUST contain these section headers verbatim, in this order:
  [Role], [Core Principles], [When to Segment], [Never Segment], [Decision Procedure],
  [Output Rules], [Examples — Segment], [Examples — Do NOT Segment]
- The [Output Rules] section MUST be copied verbatim from the block given to you. Do not
  reword it — a deterministic validator depends on it exactly as written.
- [When to Segment] and [Never Segment] must cite CONCRETE surface forms of the source
  language from the profile, not generic advice like "split at clause boundaries".
- [Decision Procedure] must make the model check, for each candidate split "A <SEG> B",
  whether translating A alone and then B alone (with A's translation already fixed and
  unmodifiable) preserves what a single translation of "A B" would convey.
- Examples must be written in the source language with the tag inserted, in the form:
    Input: <sentence>
    Output: <sentence with tags>
  Give 6-10 examples per example section. Invent realistic sentences in the source language.
- Segmentation exists to reduce latency. The prompt must bias toward splitting whenever a
  split is safe, and must not encourage the model to avoid splitting out of caution.

Return ONLY the prompt text. No commentary, no code fences."""


@dataclass
class Profiler:
    gw: Gateway

    def profile(self, samples: list[str], target_language: str) -> dict:
        user = (
            f"Target language for translation: {target_language}\n\n"
            f"Source sentences ({len(samples)} samples):\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(samples))
        )
        return self.gw.chat_json(PROFILER_SYSTEM, user, max_tokens=3000)

    def initial_prompt(self, profile: dict, target_language: str, spaced: bool) -> str:
        user = (
            f"Language profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Target language: {target_language}\n\n"
            f"Copy this [Output Rules] section verbatim into the prompt:\n\n{output_rules(spaced)}"
        )
        return self.gw.chat(PROMPT_WRITER_SYSTEM, user, max_tokens=6000)


# ── A6 Critic ────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You diagnose failures of a meaning-based segmentation prompt.

Setup: a model inserts <SEG> tags into a source sentence. Each segment is then translated in
order, each one seeing only the already-final translations of the segments before it and never
what comes after. Concatenating those segment translations is the HYPOTHESIS. Translating the
whole sentence at once is the REFERENCE. A low similarity between them means the segmentation
damaged the translation.

You do NOT assign scores — scores are computed separately. Your job is to explain WHY specific
cases failed and to propose GENERALISED rules that would prevent them.

Two failure directions matter equally:
- over_segmentation / wrong_boundary: a split broke the translation.
- under_segmentation: the sentence was left whole or barely split even though safe split points
  existed. This is a real failure — it costs latency, which is the entire point of the system.
  Cases with a perfect score but no tags are usually under-segmentation, not success.

Also flag reference_suspect when the reference (whole-sentence) translation is itself wrong, so
the loop does not chase a bad reference.

MEASURED PLACEMENT EVIDENCE. Some cases carry "boundary_diagnostics". This is a real
experiment, not inference: the sentence was re-split at many alternative positions, every
alternative was kept AT LEAST AS FAST as the current one (same piece count, and no worse on the
latency proxy), then re-translated and re-scored.

  free_quality_headroom = best_alt_quality - current_quality

Because every alternative is at least as fast, this headroom is quality you could have had for
FREE — with no latency cost at all.

- headroom LARGE (verdict "placement"): compare current_segmentation with
  best_alt_segmentation and state what the better placement did differently. Say explicitly
  that fixing it costs no latency.
- headroom SMALL (verdict "not placement"): no equally-fast split does better. The damage is
  inherent to splitting that sentence that many times. Only here may you suggest splitting such
  sentences less.

Never propose banning a connective ending outright. The same ending is safe in most sentences;
a blanket ban trades a large latency loss for a small quality gain. Your rule must state the
CONDITION that separates the safe uses from the harmful ones.

Return ONLY JSON:
{
  "cases": [
    {
      "id": "case id",
      "error_type": "over_segmentation | under_segmentation | wrong_boundary | format_violation | reference_suspect",
      "span": "the exact source substring where the problem is",
      "evidence": "what specifically differs between the segmented translation and the reference translation",
      "cause": "short mechanism, e.g. pronoun lost its referent / modifier scope changed / connective ending left the following clause unable to stand alone",
      "proposed_rule": "one generalised rule, phrased so it applies to unseen sentences — never mention this specific sentence",
      "example_pair": {"input": "source sentence", "output": "source sentence with correct <SEG> placement"}
    }
  ],
  "summary": "2-3 sentences on what the current prompt is systematically getting wrong"
}

Keep every field short. Do not quote more than 40 characters of source text in any field."""


@dataclass
class Critic:
    gw: Gateway

    def review(self, cases: list[dict], metrics: dict, violations: list[dict],
               q_floor: float = 0.0) -> dict:
        user = (
            f"Current metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"(quality = similarity of segmented translation to whole-sentence translation, "
            f"1.0 is identical; latency_gain higher is better; mean_segments is the average "
            f"number of pieces per sentence; segmented_rate is the fraction of sentences that "
            f"got at least one tag.)\n\n"
            f"Format violations ({len(violations)}):\n"
            f"{json.dumps(violations[:10], ensure_ascii=False, indent=2)}\n\n"
            f"Cases to diagnose:\n{json.dumps(cases, ensure_ascii=False, indent=2)}"
        )
        out = self.gw.chat_json(CRITIC_SYSTEM, user, max_tokens=12000)
        out["aggregate"] = summarize_critique(out.get("cases") or [], metrics,
                                              out.get("summary"), q_floor)
        return out


def summarize_critique(cases: list[dict], metrics: dict, summary: str | None,
                       q_floor: float = 0.0) -> dict:
    """집계는 세는 일이지 판단이 아니다 — LLM에 맡기면 누락되거나 틀린다.

    direction 은 **측정된 지표**에서 도출한다. 사례 카운트로 정하면 안 된다:
    Critic 에게는 망가진 사례를 골라 보내므로 wrong_boundary 가 항상 다수가 되고,
    그러면 direction 이 영구히 "경계 수정"에 고정돼 PE 가 단조 보수화한다
    (ja 실측: k 1.40→1.30, 분절률 0.35→0.25). 우선순위는 목적함수 제약 순서와
    동일하게 포맷 → 품질 → 지연.
    """
    counts: dict[str, int] = {}
    for c in cases:
        t = c.get("error_type", "unknown")
        counts[t] = counts.get(t, 0) + 1

    # **목적함수와 같은 기준을 써야 한다.** 목적함수는 LCB(평균의 하한)로 거부하는데
    # 여기서 평균만 보면 판정이 어긋난다 — 실측에서 Qs 0.8846 > floor 0.8821 이라
    # "품질 통과"로 보고 `segment more aggressively` 를 냈지만, 목적함수는 같은
    # 이터레이션을 LCB 0.8647 < floor 로 거부했다. 병목이 품질인데 PE 에게 더 자르라고
    # 지시하는 셈이라 이터레이션이 통째로 낭비된다.
    qs = metrics.get("quality_segmented", 1.0)
    n_seg = metrics.get("n_segmented", 0) or 0
    sd = metrics.get("quality_segmented_std", 0.0) or 0.0
    lcb = qs - sd / (n_seg ** 0.5) if n_seg > 1 else qs

    if metrics.get("valid_rate", 1.0) < 1.0:
        direction = "fix output format"
    elif lcb < q_floor:
        direction = "fix boundary placement"
    elif metrics.get("segmented_rate", 0.0) < 0.6 or metrics.get("mean_segments", 1.0) < 2.0:
        direction = "segment more aggressively"
    elif counts.get("over_segmentation", 0) + counts.get("wrong_boundary", 0) > \
            counts.get("under_segmentation", 0):
        direction = "fix boundary placement"
    else:
        direction = "segment more aggressively"

    return {
        "dominant_error": max(counts, key=counts.get) if counts else None,
        "error_counts": counts,
        "over_seg_count": counts.get("over_segmentation", 0) + counts.get("wrong_boundary", 0),
        "under_seg_count": counts.get("under_segmentation", 0),
        "direction": direction,
        "summary": summary,
    }


def select_cases(rows: list[dict], n_worst: int = 5, n_unsegmented: int = 5,
                 n_invalid: int = 5) -> list[dict]:
    """비평 대상 선정 — 전량이 아니라 정보량이 있는 것만.

    **과소분절 사례에 고정 쿼터를 준다.** 저품질 사례만 보내면 Critic 이 보는
    세계에는 망가진 분절밖에 없어서 진단이 항상 "너무 많이 잘랐다"로 쏠린다.
    무분절 문장 중 **가장 긴 것**을 뽑는다 — 길수록 자를 데가 있었을 개연성이
    높으므로 과소분절의 증거력이 크다.
    """
    invalid = [r for r in rows if not r["valid"]]
    valid = [r for r in rows if r["valid"]]
    segmented = [r for r in valid if r["n_segments"] > 1]
    unsegmented = [r for r in valid if r["n_segments"] <= 1]

    worst = sorted(segmented, key=lambda r: r["quality"])[:n_worst]
    longest_unseg = sorted(unsegmented, key=lambda r: -len(r["text"]))[:n_unsegmented]

    picked, seen = [], set()
    for r in invalid[:n_invalid] + worst + longest_unseg:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        # Critic 이 사례의 성격을 혼동하지 않도록 무엇 때문에 뽑혔는지 명시한다
        picked.append({**r, "selected_because": (
            "format violation" if not r["valid"]
            else "lowest quality among segmented sentences" if r["n_segments"] > 1
            else "long sentence that received NO segmentation — "
                 "judge whether safe split points were missed"
        )})
    return picked


# ── A7 Prompt Engineer ───────────────────────────────────────────────────

ENGINEER_SYSTEM = """You revise a meaning-based segmentation system prompt, one iteration at a time.

Hard constraints — violating any of these makes your output unusable:
1. Keep the section skeleton exactly: [Role], [Core Principles], [When to Segment],
   [Never Segment], [Decision Procedure], [Output Rules], [Examples — Segment],
   [Examples — Do NOT Segment]. Same headers, same order.
2. Copy [Output Rules] verbatim from the current prompt. It is frozen — a deterministic
   validator depends on it.
3. Change AT MOST TWO sections this iteration. Leave the rest byte-identical. A full rewrite
   makes it impossible to attribute the score change to anything.
4. At most 12 examples per example section. If you add one, remove a weaker one. The prompt
   must stay a set of rules, not a memorised dataset.
5. Consult the attempt history. Every entry with "adopted": false is a revision that was
   MEASURED AND REJECTED — its scores are shown. Do not repeat it or any minor variant of it.
   If your last attempt was rejected, change a DIFFERENT section this time, or move in the
   opposite direction. Repeating a reverted change makes the loop oscillate forever.
6. Rules must generalise. Never write a rule that names a specific sentence from the data.
7. Obey the critic's "direction" field. It is computed from measurements, not opinion:
   - "fix output format"        → touch the sections governing adherence and the decision
                                  procedure. Do NOT add new segmentation restrictions.
   - "fix boundary placement"   → the splits being made are damaging the translation. Make
                                  [Never Segment] more precise about WHERE, not more sweeping.
                                  The critic may carry measured boundary evidence: each blamed
                                  boundary was actually deleted and re-scored. Target ONLY the
                                  conditions it identifies. Do NOT ban a connective ending
                                  outright — the same ending is safe in most sentences, and a
                                  blanket ban trades a large latency loss for a small quality
                                  gain. State the condition, not the form.
   - "segment more aggressively" → too few splits. Remove or narrow over-broad prohibitions
                                  and add concrete permissive rules. Do NOT add restrictions.

Remember the objective: maximise latency gain (split as early and as often as possible)
SUBJECT TO the translation quality of the SPLIT sentences staying above the floor and the
output format being 100% valid. Quality is measured only on sentences you actually split, so
leaving a sentence whole never buys you quality — it only costs latency. A prompt that never
splits is worthless no matter how safe it looks.

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
        q_floor: float,
    ) -> dict:
        hist = json.dumps(history[-8:], ensure_ascii=False, indent=2)
        user = (
            f"Language profile (fixed):\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Quality floor that must be met: {q_floor:.4f}\n\n"
            f"Attempt history (prompt version -> scores, and whether it was adopted):\n{hist}\n\n"
            f"Critic feedback on the current prompt:\n{json.dumps(critique, ensure_ascii=False, indent=2)}\n\n"
            f"=== CURRENT PROMPT ===\n{current_prompt}"
        )
        return self.gw.chat_json(ENGINEER_SYSTEM, user, max_tokens=12000)
