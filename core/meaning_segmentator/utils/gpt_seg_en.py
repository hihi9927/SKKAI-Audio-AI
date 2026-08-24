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
"""You are an expert in meaning-based segmentation of spoken English ASR output.
Insert <SEG> tags to divide text into translation units for Korean translation.
The goal is to minimize latency: segment as aggressively as possible while keeping each piece translatable.

[Fundamental Constraint]
Segmentation occurs ONLY at clause boundaries.
Never split inside a single clause — never separate:
- a verb from its object or complement
- a noun from its modifier
- a preposition from its object
- a subject from its predicate
A valid split point must have a complete clause on BOTH sides.

[Decision Rule]
For every candidate split point "A <SEG> B":
1. Can a Korean translator produce an acceptable translation of A alone?
2. Can a Korean translator produce an acceptable translation of B alone?
3. If yes to both → SEGMENT. Minor stylistic differences from a combined translation are acceptable.
   Only reject a split if translation genuinely breaks: a pronoun loses its referent, a word becomes untranslatable, or the meaning reverses.

Bias toward splitting. When in doubt, SEGMENT.

[Segmentation Signals]
- Sentence boundary punctuation (. ? !) followed by a new clause
- Two clauses joined by "but", "and", "and then", or a contrastive/temporal shift
- Conditional "if/when" clause + main clause → split between them
- ", then" or ", and then" between sequential actions → split before "then"/"and then"
- Greeting or response word (Hi / Yes / No / OK / Sure / Gotcha / Alright) followed by another clause
- Ellipsis (...) followed by a new utterance
- Complete main clause + purpose clause ("so that", "in order to")
- Adverbial clause (although, even though, while, once) + main clause → split between them
- "No matter..." / "Whatever..." / "Regardless..." + main clause → split between them

[Do NOT Segment]
- Inside a single clause (verb+object, subject+predicate, noun+modifier, preposition+object)
- A filler (uh, um, hmm) would become standalone
- A pronoun in B genuinely loses its referent (no way to infer who/what)
- Noun-level enumeration without independent predicate per item: "A, B, or C"
- Tag question trailing main clause: "right?", "isn't it?"
- "so" as causal conjunction mid-sentence: "I was tired so I left"
- False start, self-correction, or incomplete utterance
- Conjunction/discourse marker at the very start of the entire input with nothing before it

[Output Rules]
- Insert <SEG> tags only. Do NOT change the original text.
- No tag at the very start or very end.
- Punctuation stays attached to the text before it.
- Space on both sides of every <SEG> tag.
- If no segmentation is needed, output the original text unchanged.
- No explanation or extra text.

[Examples — Segment]
Input: Please have a seat and the doctor will be with you shortly.
Output: Please have a seat <SEG> and the doctor will be with you shortly.

Input: If you need anything else, just let me know.
Output: If you need anything else, <SEG> just let me know.

Input: Sure. I'll get that ready for you right away.
Output: Sure. <SEG> I'll get that ready for you right away.

Input: Even though it was raining, we decided to go ahead with the picnic.
Output: Even though it was raining, <SEG> we decided to go ahead with the picnic.

Input: No matter how busy you are, you should try to get some rest.
Output: No matter how busy you are, <SEG> you should try to get some rest.

Input: Chop the onions, then add them to the pan.
Output: Chop the onions, <SEG> then add them to the pan.

Input: Alright... So what do we do next?
Output: Alright... <SEG> So what do we do next?

Input: The room was way too cold but nobody said anything about it.
Output: The room was way too cold <SEG> but nobody said anything about it.

Input: I checked the schedule and the next train leaves at five.
Output: I checked the schedule <SEG> and the next train leaves at five.

Input: Sorry about that. Let me fix it for you. It should only take a minute.
Output: Sorry about that. <SEG> Let me fix it for you. <SEG> It should only take a minute.

Input: Once you're done with the form, bring it to the front desk.
Output: Once you're done with the form, <SEG> bring it to the front desk.

[Examples — Do NOT Segment]
Input: Tea, coffee, or juice?
Output: Tea, coffee, or juice?

Input: She was exhausted so she went straight to bed.
Output: She was exhausted so she went straight to bed.

Input: I talked to my neighbor and he told me something interesting.
Output: I talked to my neighbor and he told me something interesting.

Input: um I think we should probably leave soon.
Output: um I think we should probably leave soon.

Input: That was pretty impressive right?
Output: That was pretty impressive right?

Input: We should grab the blue one from the top shelf.
Output: We should grab the blue one from the top shelf.

Input: She wants to finish her homework before dinner.
Output: She wants to finish her homework before dinner.
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


SEG_TAG_RE = re.compile(r"\s*<SEG:(\d+)>\s*")

BATCH_SUFFIX = """

[Batch Mode]
You will receive several numbered input lines, one sentence per line, in the form "1. <sentence>".
Apply all rules above to EACH line independently — confidence ranks restart at 1 within each line.
Output exactly one line per input line, in the same order, prefixed with the same number and a period,
and nothing else. Do not merge, reorder, drop, or add lines. Do not add commentary.
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_seg(s: str) -> str:
    """SEG 태그 제거 후 공백 정규화."""
    return _norm(SEG_TAG_RE.sub(" ", s))


def _is_exact_seg_output(original: str, output: str) -> bool:
    return _strip_seg(output) == _norm(original)


def _split_seg(seg_text: str) -> tuple[list[str], list[int]]:
    """태그 기준 분리 → (조각 리스트, 등장 순 랭크 번호 리스트)."""
    parts = SEG_TAG_RE.split(seg_text)
    return [x.strip() for x in parts[0::2]], [int(n) for n in parts[1::2]]


def _renumber_ranks(seg_text: str) -> str:
    """랭크 번호를 상대 순서 유지한 채 1..k로 재부여 (번호 누락/중복 복구)."""
    _, nums = _split_seg(seg_text)
    if not nums or sorted(nums) == list(range(1, len(nums) + 1)):
        return seg_text
    order = sorted(range(len(nums)), key=lambda i: (nums[i], i))
    new = [0] * len(nums)
    for rank, i in enumerate(order, start=1):
        new[i] = rank
    it = iter(new)
    return SEG_TAG_RE.sub(lambda m: f" <SEG:{next(it)}> ", seg_text).strip()


def _recover_seg_positions(original: str, modified_with_seg: str) -> str | None:
    """
    모델이 원문을 수정했을 때 SEG 경계를 원문에 복원.
    각 경계의 좌측 마지막 단어(anchor)를 원문에서 찾아 그 뒤에 태그를 삽입한다.
    anchor를 못 찾으면 None.
    """
    segments, nums = _split_seg(modified_with_seg)
    if len(segments) <= 1:
        return None

    inserts: list[tuple[int, int]] = []   # (원문 위치, 랭크 번호)
    search_start = 0

    for seg, num in zip(segments[:-1], nums):
        words = seg.split()
        if not words:
            return None
        for n_anchor in (2, 1):
            if len(words) < n_anchor:
                continue
            anchors = [re.sub(r"[?!.,;:]+$", "", w) for w in words[-n_anchor:]]
            pattern = re.compile(r"\s+".join(re.escape(a) for a in anchors) + r"[?!.,;:]*")
            m = pattern.search(original, search_start)
            if m:
                break
        else:
            return None
        inserts.append((m.end(), num))
        search_start = m.end()

    result = original
    for pos, num in sorted(inserts, reverse=True):
        result = result[:pos] + f" <SEG:{num}>" + result[pos:]

    return _renumber_ranks(result) if _is_exact_seg_output(original, result) else None


def _postprocess(original: str, result: str) -> tuple[str, str]:
    """(정리된 seg_text, 상태) — 상태: ok / recovered / failed."""
    result = _renumber_ranks(result)
    if _is_exact_seg_output(original, result):
        return result, "ok"
    recovered = _recover_seg_positions(original, result)
    if recovered:
        return recovered, "recovered"
    return original, "failed"


def _reasoning_extra(effort: str | None) -> dict:
    """reasoning 제어 파라미터. gpt-oss는 low/medium/high만 인식(true/false 무시),
    nemotron류는 think 불리언만 인식 — 모델별로 다르므로 명시 지정 시 그대로 전달."""
    return {"reasoning_effort": effort} if effort else {"think": False}


def _clean_output(text: str, result: str) -> str:
    """로컬 모델이 붙이는 라벨/코드펜스/설명 줄을 제거하고 태그된 한 줄만 남긴다."""
    result = re.sub(r"^```[a-zA-Z]*\n|```$", "", result.strip()).strip()
    result = re.sub(r"^(Output|출력)\s*:\s*", "", result, flags=re.IGNORECASE).strip()
    lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
    if len(lines) > 1:
        tagged = [ln for ln in lines if "<SEG" in ln]
        if tagged:
            return tagged[0]
        return lines[0]
    return result if lines else text


def mark_segmentation(client, text: str, model: str, provider: str = "openai",
                      max_retries: int = 5, system_prompt: str | None = None,
                      reasoning_effort: str | None = None) -> str:
    system_prompt = system_prompt or SYSTEM_PROMPT
    for attempt in range(max_retries):
        try:
            if provider == "claude":
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": text}],
                )
                raw = response.content[0].text.strip()
            elif provider == "ollama":
                # assistant prefill로 출력만 생성하도록 유도, think=False로 reasoning 모드 차단
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": f"Input: {text}"},
                        {"role": "assistant", "content": "Output: "},
                    ],
                    temperature=0,
                    extra_body=_reasoning_extra(reasoning_effort),
                )
                raw = _clean_output(text, response.choices[0].message.content)
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": text},
                    ],
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()

            seg, status = _postprocess(text, raw)
            if status == "recovered":
                print(f"  [복원] 원문 수정 감지 → SEG 위치 복원 성공")
            elif status == "failed":
                print(f"  [경고] 원문 수정 감지, 복원 실패 ({attempt+1}/{max_retries}): {raw[:80]!r}")
                if attempt < max_retries - 1:
                    continue
                print("  [폴백] 원문 그대로 반환")
            return seg
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



def mark_segmentation_batch(client, texts: list[str], model: str, provider: str = "openai",
                            system_prompt: str | None = None, max_retries: int = 3,
                            reasoning_effort: str | None = None) -> list[str | None]:
    """여러 문장을 한 요청으로 분절. 줄 누락/복원 실패 항목은 None으로 반환(호출부가 단건 재시도)."""
    system_prompt = (system_prompt or SYSTEM_PROMPT) + BATCH_SUFFIX
    user = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    for attempt in range(max_retries):
        try:
            if provider == "claude":
                response = client.messages.create(
                    model=model, max_tokens=4096,
                    system=[{"type": "text", "text": system_prompt,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                )
                out = response.content[0].text.strip()
            else:
                kwargs = {"extra_body": _reasoning_extra(reasoning_effort)} if provider == "ollama" else {}
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user}],
                    temperature=0, **kwargs,
                )
                out = response.choices[0].message.content.strip()
            break
        except Exception as e:
            msg = str(e)
            if attempt < max_retries - 1 and ("429" in msg or "rate_limit" in msg.lower()
                                              or "overloaded" in msg.lower()):
                time.sleep(2 ** attempt)
                continue
            raise

    got: dict[int, str] = {}
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)\s*[.)]\s*(.+)$", line)
        if m:
            got[int(m.group(1))] = m.group(2).strip()

    results: list[str | None] = []
    for i, src in enumerate(texts):
        raw = got.get(i + 1)
        if raw is None:
            results.append(None)
            continue
        seg, status = _postprocess(src, raw)
        results.append(None if status == "failed" else seg)
    return results


def main():
    _base = Path(__file__).resolve().parent.parent.parent.parent / "evaluation" / "DailyTalk"
    _transcribe_dir = _base / "transcribe"
    _results_dir    = _base / "results"

    parser = argparse.ArgumentParser(description="GPT 기반 의미 분절 마킹 (영어 텍스트)")
    parser.add_argument("--input",   type=str, required=True,
                        help="입력 JSON 파일명 (evaluation/KsponSpeech/transcribe/ 기준, 예: eval_clean.json)")
    parser.add_argument("--output",  type=str, default=None,
                        help="출력 JSON 파일명 (기본: 입력 파일명에 _seg_eng 추가, evaluation/KsponSpeech/results/ 저장)")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "claude", "ollama"],
                        help="사용할 API 제공자 (openai, claude, ollama)")
    parser.add_argument("--model",   type=str, default=None,
                        help="모델명 (openai 기본값: gpt-5.4-mini, claude 기본값: claude-haiku-4-5, ollama 기본값: llama3.3:70b)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="시스템 프롬프트 txt 경로 (미지정 시 내장 SYSTEM_PROMPT 사용). 예: prompt_en_ranked.txt")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama 호스트 (기본: OLLAMA_HOST 환경변수 또는 localhost:11434)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API 키 (미입력 시 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 환경변수 사용)")
    parser.add_argument("--delay",     type=float, default=0.5, help="요청 간 딜레이(초, workers>1이면 무시)")
    parser.add_argument("--workers",   type=int, default=1,
                        help="동시 요청 수 (ollama 병렬 배치용, 기본 1=순차). ollama는 OLLAMA_NUM_PARALLEL도 함께 키울 것")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="reasoning 모델의 사고 길이 (gpt-oss 등). 미지정 시 think:false 전달")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="한 요청에 넣을 문장 수 (기본 1). >1이면 묶음 모드: 프롬프트 prefill 비용을 나눠 처리량↑")
    parser.add_argument("--save-every", type=int, default=20,
                        help="병렬 처리 시 중간 저장 주기(건수)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="이미 seg_text가 있는 항목도 재처리")
    parser.add_argument("--gdt",       action="store_true",
                        help="분절 완료 후 GDT(Google 번역) + COMET 평가 자동 실행")
    parser.add_argument("--gdt-delay", type=float, default=0.2,
                        help="GDT 번역 요청 간 딜레이(초, 기본: 0.2)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="처리할 최대 항목 수 (기본: 전체)")
    parser.add_argument("--overwrite", action="store_true",
                        help="출력 파일을 입력 파일에 덮어쓰기 (--output 무시)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.provider == "claude":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API 키가 필요합니다. --api-key 또는 ANTHROPIC_API_KEY 환경변수를 설정하세요.")
        client = anthropic.Anthropic(api_key=api_key)
        model = args.model or "claude-haiku-4-5"
    elif args.provider == "ollama":
        host = args.ollama_host or os.environ.get("OLLAMA_HOST", "localhost:11434")
        client = OpenAI(base_url=f"http://{host}/v1", api_key="ollama")
        model = args.model or "llama3.3:70b"
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
        client = OpenAI(api_key=api_key)
        model = args.model or "gpt-5.4-mini"

    if args.prompt_file:
        pf = Path(args.prompt_file)
        if not pf.is_absolute() and not pf.exists():
            pf = Path(__file__).resolve().parent / args.prompt_file
        if not pf.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {args.prompt_file}")
        system_prompt = pf.read_text(encoding="utf-8")
        print(f"프롬프트: {pf} ({len(system_prompt)}자)")
    else:
        system_prompt = SYSTEM_PROMPT
        print("프롬프트: 내장 SYSTEM_PROMPT")

    input_path = _transcribe_dir / args.input

    _results_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path = input_path
    elif args.output:
        p = Path(args.output)
        output_path = p if p.is_absolute() or len(p.parts) > 1 else _results_dir / p
    else:
        output_path = _results_dir / (input_path.stem + "_seg_en" + input_path.suffix)

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

    target = data[:args.limit] if args.limit else data

    def save():
        """flat target → DailyTalk 그룹 구조로 복원 후 저장 (비어있는 그룹 제외)"""
        grouped = {}
        for e in target:
            gk = entry_group[e["file"]]
            if gk not in grouped:
                grouped[gk] = {"data": []}
            grouped[gk]["data"].append(e)
        output_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 병렬/묶음 분절: 전량 선처리 후 아래 루프는 GDT만 담당 ──
    if args.workers > 1 or args.batch_size > 1:
        from concurrent.futures import ThreadPoolExecutor

        pending = [i for i, e in enumerate(target)
                   if not (args.resume and bool(e.get("seg_text")))]
        skipped = len(target) - len(pending)
        if skipped:
            print(f"건너뜀 (이미 처리됨): {skipped}개")

        if pending:
            bs = max(1, args.batch_size)
            chunks = [pending[i:i + bs] for i in range(0, len(pending), bs)]
            print(f"분절 시작: {len(pending)}개 / 요청 {len(chunks)}건 "
                  f"(batch={bs}, workers={args.workers}), model={model}")

            def work(idxs):
                texts = [target[i]["text"] for i in idxs]
                try:
                    if len(idxs) == 1:
                        return idxs, [mark_segmentation(client, texts[0], model, args.provider,
                                                        system_prompt=system_prompt,
                                                        reasoning_effort=args.reasoning_effort)], None
                    return idxs, mark_segmentation_batch(client, texts, model, args.provider,
                                                         system_prompt=system_prompt,
                                                         reasoning_effort=args.reasoning_effort), None
                except Exception as e:
                    return idxs, [None] * len(idxs), str(e)

            done = 0
            retry_single: list[int] = []
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                for idxs, outs, err in ex.map(work, chunks):
                    if err:
                        print(f"  [요청 오류] {err}")
                    for i, seg_text in zip(idxs, outs):
                        if seg_text is None:
                            retry_single.append(i)
                        else:
                            target[i]["seg_text"] = seg_text
                            print(f"[{i+1}/{len(target)}] {target[i]['file']}: {seg_text}")
                    done += len(idxs)
                    if done % args.save_every < len(idxs):
                        save()
                        print(f"  ── 중간 저장 ({done}/{len(pending)}) ──")

            # 묶음에서 누락/복원 실패한 항목만 단건 재시도
            if retry_single:
                print(f"단건 재시도: {len(retry_single)}개")
                def work_one(i):
                    try:
                        return i, mark_segmentation(client, target[i]["text"], model, args.provider,
                                                    system_prompt=system_prompt,
                                                    reasoning_effort=args.reasoning_effort), None
                    except Exception as e:
                        return i, None, str(e)
                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                    for i, seg_text, err in ex.map(work_one, retry_single):
                        if err:
                            print(f"  [{i+1}] 오류: {err}")
                        target[i]["seg_text"] = seg_text
                        print(f"[재시도 {i+1}] {seg_text}")

            save()
            filled = sum(1 for e in target if e.get("seg_text"))
            print(f"분절 완료: {filled}/{len(target)}")

        # 분절은 방금 끝났으므로 아래 루프에서 재요청하지 않도록 고정
        args.resume = True

    budget_exceeded = False
    total = len(target)
    for i, entry in enumerate(target):
        seg_done  = bool(entry.get("seg_text"))
        trans_done = bool(entry.get("gdt_full_trans")) and bool(entry.get("gdt_seg_trans"))

        if args.resume and seg_done and (not args.gdt or trans_done):
            print(f"[{i+1}/{total}] 건너뜀 (이미 처리됨): {entry['file']}")
            continue

        text = entry["text"]
        print(f"[{i+1}/{total}] {entry['file']}")
        print(f"  원문: {text}")

        # ── 1. 분절 ──────────────────────────────────────────────
        if not (args.resume and seg_done):
            try:
                seg_text = mark_segmentation(client, text, model, args.provider, system_prompt=system_prompt,
                                             reasoning_effort=args.reasoning_effort)
                entry["seg_text"] = seg_text
                print(f"  분절: {seg_text}")
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in ("insufficient_quota", "billing", "budget", "exceeded", "402")):
                    print(f"  예산 초과 오류 — 분절 중단.\n  ({msg})")
                    budget_exceeded = True
                    save()
                    break
                print(f"  오류: {e}")
                entry["seg_text"] = None

            save()

            if args.delay > 0:
                time.sleep(args.delay)

        # ── 2. GDT 번역 (--gdt 옵션일 때만) ──────────────────────
        if args.gdt and entry.get("seg_text") is not None:
            if not (args.resume and bool(entry.get("gdt_full_trans"))):
                entry["gdt_full_trans"] = _gdt_translate(entry["text"])
                print(f"  gdt_full  : {entry['gdt_full_trans']}")
                if args.gdt_delay > 0:
                    time.sleep(args.gdt_delay)

            if not (args.resume and bool(entry.get("gdt_seg_trans"))):
                if "<SEG>" not in entry.get("seg_text", ""):
                    entry["gdt_seg_trans"] = entry.get("gdt_full_trans")
                    print(f"  gdt_seg   : (분절 없음, gdt_full_trans 사용)")
                else:
                    entry["gdt_seg_trans"] = _gdt_translate_seg(entry["seg_text"])
                    print(f"  gdt_seg   : {entry['gdt_seg_trans']}")
                    if args.gdt_delay > 0:
                        time.sleep(args.gdt_delay)

            save()

    if budget_exceeded:
        print(f"\n분절 중단 (예산 초과). 저장: {output_path}")
    else:
        print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
