"""
GPT API를 이용한 의미 분절 마킹 스크립트 (영어 버전)
eval 데이터의 각 텍스트에 <SEG> 태그로 분절 지점을 표시하고
"seg_text" 필드로 저장합니다.
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
import anthropic

SYSTEM_PROMPT = (
"""You are an expert in meaning-based segmentation of spoken English text from ASR (Automatic Speech Recognition) output.
Your task is to insert <SEG> tags to divide conversational English into translation units.
The goal is for each segment to be a semantically self-contained unit that a translator can handle without surrounding context.

[Spoken English Characteristics to Be Aware Of]
ASR output reflects natural speech, which means:
- Filler words and hesitation markers: uh, um, uh-huh, hmm
- Discourse markers that open or link clauses: well, like, you know, I mean, right, okay, so, but, and, actually, basically, honestly, literally
- Hedges attached to the following clause: kind of, sort of, I think, I guess, I suppose, I feel like
- False starts and self-corrections: "I was gonna — I mean I actually changed my mind"
- Reduced/contracted forms: gonna, wanna, gotta, kinda, sorta, 'cause, yeah, nah
- Tag questions trailing the main clause: "It was great, wasn't it?" / "That's weird, right?"
- Dislocated elements: left dislocation ("That movie, it was amazing") / right dislocation ("It was amazing, that movie")
These features are normal and must be handled carefully when deciding whether to segment.

[Core Principles]
- Minimize segmentation: when in doubt, do NOT segment
- A grammatically incomplete clause can still be a single coherent meaning unit — do not segment based on grammar alone
- Segmentation is based on semantic completeness and translation independence, NOT clause count or sentence length

[When to Segment]
Segment ONLY when BOTH conditions are met.
Even if both conditions are met, do NOT segment if it falls under [Never Segment] below — those rules always take priority.

1. There is a clear semantic boundary between two clauses:
   - Subject, tense, or topic changes meaningfully
   - OR the first clause describes a completed event/state and the second introduces a new, independent one
   - OR a contrastive or temporal shift (but, however, and then, after that) separates two independently meaningful clauses
   - OR a punctuation mark (. ? !) ends a grammatically complete sentence and a new sentence begins
   - OR a greeting (Hello, Hi, Hey) or response word (Yes, No, Sure, Okay, Alright) is followed by a period/comma and an independent clause
     e.g. "Hello. <SEG> How can I help you?" / "Yes. <SEG> I'd like to make a reservation."
   - OR the main clause is semantically complete and is followed by a purpose clause (so that, in order to) that can be independently translated
2. After segmenting, each piece can stand alone as a translation unit — a translator with no context could handle it correctly

[Note on "so" — three distinct roles]
- "so" as a causal/result conjunction linking two clauses mid-sentence → do NOT segment
  e.g. "I was tired so I left early" — tight causal link, keep together
- "So" as a discourse marker at the start of a clause after a sentence boundary → valid segment point before it
  e.g. "She quit her job. <SEG> So now she's looking for something new."
- "so that" as a purpose clause after a complete main clause → valid segment boundary
  e.g. "I saved money <SEG> so that I could travel."

[Never Segment — even if both conditions above are met]
- A filler or hesitation marker (uh, um, hmm, etc.) would become a standalone piece
  → Must be attached to the following content
- A discourse marker appears at the very start of the entire input with no preceding clause (well, like, you know, I mean, but, and, actually, basically, right, etc.)
  → Must be attached to the following content
  → NOTE: "but" or "and" between two complete clauses is a valid segment boundary — only "but"/"and" with nothing before it is forbidden as standalone
- A hedge expression (I think, I guess, kind of, sort of, I feel like, maybe, probably) introduces the following clause
  → Must stay with the clause it introduces
- A subordinating conjunction opens a clause that cannot stand alone: because, since, when, while, if, although, even though, unless, until, as long as
  → The subordinate clause and its main clause must stay together
  → "so that" and "in order to" are exceptions — see [Note on "so"] above
- A relative clause (who, which, that, where, when) modifies the preceding noun
  → Restrictive relative clauses (no comma) — always keep together
  → Non-restrictive relative clauses (comma + which/who) — generally keep together unless clearly separable
- A complement clause follows a reporting or cognitive verb: "I think that...", "she said that...", "I know...", "I feel like..."
  → Keep the verb and its complement together
- A participial phrase, infinitive phrase, or gerund phrase modifies the main clause
  → Keep together
- Inside a noun/phrase-level enumeration (items of the same type with no independent subject+predicate): A and B, A or B, both A and B, not only A but B, A as well as B
  → Keep the entire list as one unit
  → e.g. "I had to clean and do laundry and meal prep" — these share one subject, keep together
  → EXCEPTION: If each item in the list is a full clause with its own subject and predicate and can be independently translated, treat each clause boundary as a valid segment point
  → e.g. "Country code is eighty one, <SEG> area code is thirty eight, <SEG> and the number is eight four six eight nine seven two."
- A tag question trails the main clause: "right?", "you know?", "isn't it?", "didn't you?"
  → Must stay with its main clause
- A false start or self-correction is mid-utterance
  → No internal segmentation
- An utterance is cut off or incomplete
  → No internal segmentation

[Decision Procedure — apply internally before confirming any split]
Do NOT include this reasoning in your output.

1. Identify the segmentation candidate: "A <SEG> B"
2. Attempt to translate A alone, then B alone, into Korean without any surrounding context
3. Attempt to translate the full original (A + B) into Korean as one unit
4. Compare: does combining the step-2 results match step 3 in meaning and nuance?
   - Yes → confirm segmentation
   - No, or any of the following apply → cancel, output original as-is:
     · A modifier's scope changes (e.g. "really" now applies differently)
     · A context-dependent word (then, that, there, this way) becomes ambiguous
     · The logical or narrative connection between A and B is lost in translation
     · A pronoun in B (he, she, it, they, that, there, this) loses its referent — see also [Never Segment]

[Output Rules]
- Insert <SEG> tags only — do NOT change, correct, or paraphrase the original text in any way
- No tag at the very start or very end of the text
- Never place two tags consecutively (<SEG> <SEG>)
- Punctuation (. ? ! ,) must stay attached to the text before it — never immediately after a <SEG> tag
- Always place a space on both sides of every <SEG> tag
- If no segmentation is needed, output the original text unchanged
- Do NOT add any explanation, label, comment, or extra text

---

[Examples — Segment ✓]

# Greeting followed by independent clause
Input: Hi. Welcome to the store. Is there anything I can help you with?
Output: Hi. <SEG> Welcome to the store. <SEG> Is there anything I can help you with?

# Response word (Yes/No) followed by independent clause
Input: Yes. I'd like to place an order please.
Output: Yes. <SEG> I'd like to place an order please.

# Complete main clause + purpose clause (so that) → segment
Input: I'm making a shopping budget so that I don't spend too much money.
Output: I'm making a shopping budget <SEG> so that I don't spend too much money.

# Two complete independent clauses, clear topic shift
Input: I woke up really early this morning and then I went for a run
Output: I woke up really early this morning <SEG> and then I went for a run

# Completed event + new event, punctuation stays with preceding text
Input: It was honestly so good. I definitely want to go back.
Output: It was honestly so good. <SEG> I definitely want to go back.

# Discourse marker (so) opens a new clause — groups with following content
Input: She ended up leaving the company. So now she's looking for something new.
Output: She ended up leaving the company. <SEG> So now she's looking for something new.

# Multiple boundaries in a longer utterance
Input: I met up with my friend yesterday and we grabbed food and went to a café. But then we ran into an old classmate. It had been so long I was genuinely happy to see them.
Output: I met up with my friend yesterday and we grabbed food and went to a café. <SEG> But then we ran into an old classmate. <SEG> It had been so long I was genuinely happy to see them.

# Filler + complete clause, then contrastive clause with "but" — segment between the two clauses
Input: Uh I'm not really sure but I think I want to try something different
Output: Uh I'm not really sure <SEG> but I think I want to try something different

# Subject shift mid-utterance, no punctuation
Input: I got the pasta but she ordered the risotto
Output: I got the pasta <SEG> but she ordered the risotto

# Temporal/situational contrast without punctuation
Input: Yesterday was rough but today actually feels okay
Output: Yesterday was rough <SEG> but today actually feels okay

# Contrastive shift with discourse marker, no punctuation
Input: The vibe was really nice but the prices were kind of steep
Output: The vibe was really nice <SEG> but the prices were kind of steep

# Sequential events with "and then"
Input: I finished the report and then I just completely crashed
Output: I finished the report <SEG> and then I just completely crashed

---

[Examples — Do NOT Segment ✗]

# Filler at the start — must attach to following content
Input: uh it was actually super crowded in there
Output: uh it was actually super crowded in there

# Discourse marker alone at the start — must attach to following content
Input: but I thought that was kind of strange honestly
Output: but I thought that was kind of strange honestly

# Hedge + following clause — keep together
Input: I think he was trying to be nice but it came off kind of weird
Output: I think he was trying to be nice but it came off kind of weird

# Subordinate clause with because — cannot stand alone
Input: I was in such a good mood today because the weather was perfect
Output: I was in such a good mood today because the weather was perfect

# Cause-effect with so — continuous logical structure
Input: I stayed up way too late last night so I'm running on nothing today
Output: I stayed up way too late last night so I'm running on nothing today

# Enumeration — keep entire list together
Input: I had to clean and do laundry and meal prep over the weekend so it was a lot
Output: I had to clean and do laundry and meal prep over the weekend so it was a lot

# Complement clause after reporting verb — keep together
Input: She told me she wasn't sure if she was going to make it
Output: She told me she wasn't sure if she was going to make it

# Tag question — must stay with its main clause
Input: That's kind of a weird thing to say right
Output: That's kind of a weird thing to say right

# False start mid-utterance — no internal segmentation
Input: I was gonna bring it up but I kind of just — I don't know I just let it go
Output: I was gonna bring it up but I kind of just — I don't know I just let it go

# Incomplete utterance — no internal segmentation
Input: but at that point I was about to say something and
Output: but at that point I was about to say something and

# Pronoun would lose its referent after split — no segmentation
# ("she said something" alone doesn't tell you who "she" is)
Input: I got a call from my friend and she said something that kind of got to me
Output: I got a call from my friend and she said something that kind of got to me

# Restrictive relative clause — must stay with the noun it modifies
Input: the guy that I was talking to earlier actually knows you
Output: the guy that I was talking to earlier actually knows you

"""
)


def _gdt_translate(text: str, src: str = "en", tgt: str = "ko") -> str | None:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception as e:
        print(f"  [번역 실패] '{text}' → {e}")
        return None


def _gdt_translate_seg(seg_text: str | None, src: str = "en", tgt: str = "ko") -> str | None:
    if not seg_text:
        return None
    from deep_translator import GoogleTranslator
    segments = seg_text.split("<SEG>")
    translated = [_gdt_translate(s.strip(), src, tgt) for s in segments if s.strip()]
    return " ".join(t for t in translated if t is not None)


def run_gdt(data: list, output_path: Path, delay: float = 0.2, resume: bool = True,
            save_fn=None) -> None:
    """GDT(Google 번역)로 full/seg 번역을 실행."""

    for i, entry in enumerate(data):
        has_full = bool(entry.get("gdt_full_trans"))
        has_seg  = bool(entry.get("gdt_seg_trans"))

        if resume and has_full and has_seg:
            print(f"[GDT {i+1}/{len(data)}] 건너뜀: {entry['file']}")
            continue

        if "seg_text" not in entry:
            print(f"[GDT {i+1}/{len(data)}] seg_text 없음, 스킵: {entry['file']}")
            continue

        print(f"[GDT {i+1}/{len(data)}] {entry['file']}")

        if not (resume and has_full):
            entry["gdt_full_trans"] = _gdt_translate(entry["text"])
            print(f"  gdt_full  : {entry['gdt_full_trans']}")
            if delay > 0:
                time.sleep(delay)

        if not (resume and has_seg):
            if "<SEG>" not in entry.get("seg_text", ""):
                entry["gdt_seg_trans"] = entry.get("gdt_full_trans")
                print(f"  gdt_seg   : (분절 없음, gdt_full_trans 사용)")
            else:
                entry["gdt_seg_trans"] = _gdt_translate_seg(entry["seg_text"])
                print(f"  gdt_seg   : {entry['gdt_seg_trans']}")
                if delay > 0:
                    time.sleep(delay)

        if save_fn:
            save_fn()

    print(f"\nGDT 번역 완료. 저장: {output_path}")


def mark_segmentation(client, text: str, model: str, provider: str = "openai", max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            if provider == "claude":
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": text}],
                )
                return response.content[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": text},
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower() or "overloaded" in msg.lower():
                m = re.search(r"try again in (\d+(?:\.\d+)?)(m?s)", msg)
                if m:
                    wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                    wait = max(wait + 0.2, 1.0)
                else:
                    wait = 2 ** attempt
                print(f"  Rate limit — {wait:.1f}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"최대 재시도({max_retries}회) 초과: {text[:30]}")


def main():
    _base = Path(__file__).resolve().parent.parent.parent.parent / "evaluation" / "DailyTalk"
    _transcribe_dir = _base / "transcribe"
    _results_dir    = _base / "results"

    parser = argparse.ArgumentParser(description="GPT 기반 의미 분절 마킹 (영어 텍스트)")
    parser.add_argument("--input",   type=str, required=True,
                        help="입력 JSON 파일명 (evaluation/KsponSpeech/transcribe/ 기준, 예: eval_clean.json)")
    parser.add_argument("--output",  type=str, default=None,
                        help="출력 JSON 파일명 (기본: 입력 파일명에 _seg_eng 추가, evaluation/KsponSpeech/results/ 저장)")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"],
                        help="사용할 API 제공자 (openai 또는 claude)")
    parser.add_argument("--model",   type=str, default=None,
                        help="모델명 (openai 기본값: gpt-5.4-mini, claude 기본값: claude-haiku-4-5)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API 키 (미입력 시 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 환경변수 사용)")
    parser.add_argument("--delay",     type=float, default=0.5, help="요청 간 딜레이(초)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="이미 seg_text가 있는 항목도 재처리")
    parser.add_argument("--gdt",       action="store_true",
                        help="분절 완료 후 GDT(Google 번역) + COMET 평가 자동 실행")
    parser.add_argument("--gdt-delay", type=float, default=0.2,
                        help="GDT 번역 요청 간 딜레이(초, 기본: 0.2)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.provider == "claude":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API 키가 필요합니다. --api-key 또는 ANTHROPIC_API_KEY 환경변수를 설정하세요.")
        client = anthropic.Anthropic(api_key=api_key)
        model = args.model or "claude-haiku-4-5"
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
        client = OpenAI(api_key=api_key)
        model = args.model or "gpt-5.4-mini"

    input_path = _transcribe_dir / args.input

    _results_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        p = Path(args.output)
        output_path = p if p.is_absolute() or len(p.parts) > 1 else _results_dir / p
    else:
        output_path = _results_dir / (input_path.stem + "_seg_eng" + input_path.suffix)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    raw_in = json.loads(input_path.read_text(encoding="utf-8"))

    # DailyTalk 구조: {"0": {"data": [...]}, "1": {"data": [...]}, ...}
    # 그룹 키 순서와 각 entry의 그룹 소속을 기억해두고 flat list로 펼침
    group_keys = list(raw_in.keys())
    base_data = []
    entry_group = {}  # file → group_key
    for gk in group_keys:
        for e in raw_in[gk]["data"]:
            base_data.append(dict(e))
            entry_group[e["file"]] = gk

    if output_path.exists():
        raw_out = json.loads(output_path.read_text(encoding="utf-8"))
        # 출력 파일도 동일한 DailyTalk 구조이므로 flat하게 펼침
        existing = {}
        for gk in raw_out:
            for e in raw_out[gk]["data"]:
                existing[e["file"]] = e
        data = [existing.get(e["file"], e) for e in base_data]
    else:
        data = base_data
        print(f"출력 파일 생성: {output_path}")

    def save():
        """flat data → DailyTalk 그룹 구조로 복원 후 저장"""
        grouped = {gk: {"data": []} for gk in group_keys}
        for e in data:
            grouped[entry_group[e["file"]]]["data"].append(e)
        output_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")

    budget_exceeded = False
    for i, entry in enumerate(data):
        if args.resume and entry.get("seg_text"):
            print(f"[{i+1}/{len(data)}] 건너뜀 (이미 처리됨): {entry['file']}")
            continue

        text = entry["text"]
        print(f"[{i+1}/{len(data)}] {entry['file']}")
        print(f"  원문: {text}")

        try:
            seg_text = mark_segmentation(client, text, model, args.provider)
            entry["seg_text"] = seg_text
            print(f"  분절: {seg_text}")
        except Exception as e:
            msg = str(e)
            if any(k in msg.lower() for k in ("insufficient_quota", "billing", "budget", "exceeded", "402")):
                print(f"  예산 초과 오류 — 분절 중단, 번역으로 이동합니다.\n  ({msg})")
                budget_exceeded = True
                save()
                break
            print(f"  오류: {e}")
            entry["seg_text"] = None

        save()

        if args.delay > 0:
            time.sleep(args.delay)

    if budget_exceeded:
        print(f"\n분절 중단 (예산 초과). 저장: {output_path}")
    else:
        print(f"\n완료. 저장: {output_path}")

    if args.gdt:
        print("\nGDT 번역 시작...")
        run_gdt(data, output_path, delay=args.gdt_delay, resume=args.resume, save_fn=save)


if __name__ == "__main__":
    main()
