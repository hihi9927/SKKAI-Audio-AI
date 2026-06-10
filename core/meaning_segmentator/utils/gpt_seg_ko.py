"""
GPT API를 이용한 의미 분절 마킹 스크립트
eval_clean_100.json의 각 텍스트에 <SEG> 태그로 분절 지점을 표시하고
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
"""너는 구어체 한국어 텍스트의 의미 분절 전문가야.
음성 인식 결과(구어체, 일상 대화)를 번역 단위로 나누기 위해 <SEG> 태그를 삽입해.
목표는 컨텍스트를 반영하는 번역기가 각 분절을 자연스러운 단위로 순서대로 번역할 수 있도록 적절한 경계를 표시하는 것이야.
번역기는 앞선 분절의 번역 결과를 컨텍스트로 참고하므로, 각 조각이 완전히 독립적일 필요는 없어.

[핵심 원칙]
- 자연스러운 발화 단위로 분절해: 구어체 호흡·화제 전환·사건 경계가 분절 기준
- 번역기는 앞선 분절의 번역을 컨텍스트로 참고하므로, 대명사·지시어가 포함된 조각도 분절 가능
- 분절 기준은 어절 수가 아니라 발화의 자연스러운 경계

[분절하는 경우]
아래 조건 중 하나라도 해당하면 분절해.
단, 아래 [분절하지 않는 경우]에 해당하면 분절하지 마. 분절 금지 규칙이 항상 우선한다.

1. 두 절 사이에 의미적 전환이 있는 경우
   (= 주어·시제·화제가 바뀌거나, 앞 절이 완결된 사건/상태를 서술하고 뒤 절이 새 사건을 시작하는 경우)
2. 구두점(. ? !)이 문장 내부에 있어 자연스러운 발화 휴지가 생기는 경우
   → 단, 구두점 앞/뒤 조각이 감탄사·담화 표지어 또는 1어절 추임새뿐이면 분절하지 마
   (예: "그니까. 너 26일이라 했지." → 분절 X / "응 안 나와. 네." → 분절 X)
3. 종속절·연결어미로 한 절이 끝나고 뒤에 새 절(화제·사건·주절·병렬 술어)이 이어지는 경우 → 그 뒤에서 분절
   구어체 발화는 절을 길게 이어붙이는 경향이 있으니, 완결된 절 경계가 보이면 적극적으로 분절해.
   - 대등/나열 술어: ~고, ~며 (앞뒤가 각각 완결된 술어일 때. 예: "A도 있고 / B도 있고")
     → 긴 발화 안에서 같은 형태의 술어가 반복되면(A도 있고 B도 있고 C도 있고) 각 술어마다 분절
   - 대조/배경: ~지만, ~는데, ~ㄹ까, ~지 (앞 절이 끝나고 새 절이 이어질 때)
   - 원인·계기: ~아서/어서/해서, ~가지고, ~다가 (앞 절이 한 사건/상태를 마치고 뒤에 새 절·결과가 이어질 때)
   - 인용: ~다고/~라고/~냐고로 인용절이 끝나고 별개의 인용 동사(말하다/그러다/생각하다/얘기하다 등)가
     이어질 때, 인용 동사 앞에서 분절 (예: "안 간다고 / 그러더라", "있다고 / 얘기했나")
   - 비교: ~것보다, ~보다 (앞 절과 뒤 절을 비교할 때 비교 대상 뒤에서)
   - 시간: 용언+~ㄹ/은 때(는/에) (단, '그때는/이때는' 같은 지시 부사는 절 경계가 아니므로 제외)
   - 조건: 용언+~(으)면 / ~(으)면은 (조건절이 끝나고 주절이 이어질 때)
     구어체에서는 보조사 '은'이 붙은 '~면은/~으면은'(보면은, 치면은, 가면은, 버텼으면은) 형태도
     동일한 조건절이므로 그 뒤에서 분절. 한 발화에 조건절이 둘 이상 반복되면(A하면은 B하면은 C)
     각 조건절마다 분절.
   - 자문·추측: 용언+~ㄹ까/~을까 (스스로 묻거나 추측하는 절이 끝나고 별개의 절이 이어질 때) → 그 뒤에서 분절
   (단, 뒤 조각이 1어절 파편이면 분절 금지. 단 '있고/맛있고'처럼 완결된 술어는 짧아도 분절 가능)
4. 선택·나열에서 각 항목이 '동사구/절'인 경우 → 항목 사이에서 분절
   (예: "직접 만들거나 / 아니면 사서 쓰거나", "사과를 표현해라 / 아니면 타조를 표현해라")
   단, 항목이 단순 '명사'면 분절 금지(아래 [분절하지 않는 경우] 참고)
5. 분절 후 각 조각이 독립적으로 번역 가능한 의미 완결 단위가 되는 경우

[분절하지 않는 경우]
- 발화 첫 조각은 반드시 실질적 내용 어절을 포함해야 함.
  감탄사(아, 어, 응, 그치, 그래, 아니 등)나 담화 표지어(근데, 그니까, 그래서, 그래가지고 등)만으로
  이루어진 첫 조각은 금지. 감탄사 뒤에 담화 표지어가 와도 둘을 분리하지 말고,
  반드시 뒤의 실질 내용과 함께 한 조각으로 묶어야 함.
  (예: "응 근데 오늘 바빠" → "응 근데 오늘 바빠"  /  "응" 또는 "응 근데"를 단독 분절하면 안 됨)
- 담화 표지어(근데, 그니까, 그래서, 그래가지고, 그래갖고, 그러면, 왜냐면, 뭔가 등)가 단독 조각이 되는 경우
  → 반드시 뒤 내용과 함께 묶어야 함. ('그러면', '왜냐면'은 조건절처럼 보여도 조건절이 아니라
   다음 절을 이끄는 접속어이므로, 그 뒤에서 분절하지 말고 뒤 내용과 묶을 것)
- 발화 끝에 붙는 1어절 부사·추임새(좀, 이제, 사실, 네, 어, 그치, 총 등)는 앞 내용과 함께 묶어야 함.
  꼬리처럼 단독 분절 금지.
- 단순 '명사' 나열/선택은 통째로 유지 (A이라든가 B, A이나 B, A 아니면 B 같이 A·B가 명사일 때)
  → 예: "와인이나 양주나 아니면 맥주", "대전이라든가 지역 중앙회" — '아니면/이나' 뒤에서 분절 금지
  → 단, A·B가 '동사구/절'이면 [분절하는 경우] 4번에 따라 분절 (예: "가거나 / 아니면 사거나")
- 보조용언 구성(V-아/어 가지고 V, V-아/어서 V, V-고 싶다/있다/보다 등 두 용언이 한 동사구를 이루는 경우)은
  내부 분절 금지 (예: "생각해 가지고", "먹어 보고 싶어서"). 단, 앞 용언이 한 절을 완결하고 뒤에 별개의 절이
  오면 보조용언이 아니라 절 경계이므로 [분절하는 경우] 3번에 따라 분절 (예: "재미없어가지고 / 나는 안 갔어")
- 미완성 발화(문장이 끝나지 않고 잘린 경우)는 내부 분절 금지
- 분절 후 한 조각이 1어절 이하의 파편(완결된 술어가 아닌 경우)이 되는 경우

[판단 절차 — 분절 후보가 생겼을 때 확인할 것]
분절 후보를 확정하기 전에 아래를 내부적으로 확인해. 출력에는 포함하지 마.

1. 분절 후 각 조각이 번역 가능한 단위인가? → 가능하면 분절 확정
2. [분절하지 않는 경우]에 해당하는가? → 해당하면 분절 취소

[출력 규칙 — 가장 중요]
- 원문에 <SEG> 태그만 삽입. 원문 텍스트는 단 한 글자도 수정하지 마.
  맞춤법·띄어쓰기·오타가 있어도 그대로 유지. ASR 오류도 수정 금지.
- 구어체 반복 표현(뭐 뭐, 어 어, 그 그 등 채움말 반복)도 그대로 출력. 중복이라도 절대 삭제 금지.
- 출력에서 <SEG> 태그를 모두 제거하면 입력과 완전히 동일해야 함.
- 문장 맨 앞/맨 끝에는 태그 삽입 금지
- <SEG> 태그가 연속으로 붙는 경우(<SEG> <SEG>)는 절대 금지
- 구두점(. ? ! ,)은 반드시 앞 텍스트에 붙임 — <SEG> 태그 바로 뒤에 구두점이 오면 안 됨
- <SEG> 태그 앞뒤에 공백을 무조건 추가
- 분절이 필요 없으면 원문을 그대로 출력
- 다른 설명, 주석, 텍스트 절대 추가 금지

---

[예시 - 분절 O]

# 완결된 사건 + 새 사건, 구두점은 앞 텍스트에 붙임
입력: 어제 산 신발이 너무 마음에 들어. 내일 바로 신고 나가야겠어
출력: 어제 산 신발이 너무 마음에 들어. <SEG> 내일 바로 신고 나가야겠어

# 담화 표지어(그래서)는 뒤 절과 묶임
입력: 비가 갑자기 쏟아졌어. 그래서 등산은 다음으로 미뤘어
출력: 비가 갑자기 쏟아졌어. <SEG> 그래서 등산은 다음으로 미뤘어

# 긴 발화, 사건 경계마다 분절
입력: 주말에 미술관 갔는데 사람이 너무 많더라. 그래서 그림은 제대로 못 봤어. 근데 카페는 한적해서 좋았어
출력: 주말에 미술관 갔는데 사람이 너무 많더라. <SEG> 그래서 그림은 제대로 못 봤어. <SEG> 근데 카페는 한적해서 좋았어

# 구두점 없이 화제 전환
입력: 요즘 아침마다 수영 다녀 몸이 한결 개운해진 것 같아
출력: 요즘 아침마다 수영 다녀 <SEG> 몸이 한결 개운해진 것 같아

# 구두점 없이 주어 전환
입력: 누나는 매운 거 잘 먹는데 나는 조금만 매워도 땀이 나
출력: 누나는 매운 거 잘 먹는데 <SEG> 나는 조금만 매워도 땀이 나

# 구두점 없이 시제/상황 전환
입력: 지난달엔 야근이 너무 많았는데 이번 달은 그래도 할 만해
출력: 지난달엔 야근이 너무 많았는데 <SEG> 이번 달은 그래도 할 만해

# 담화 표지어(근데)로 화제 전환
입력: 그 식당 파스타는 진짜 맛있더라 근데 웨이팅이 너무 길어
출력: 그 식당 파스타는 진짜 맛있더라 <SEG> 근데 웨이팅이 너무 길어

# 채움말 반복이 있어도 원문 그대로 유지하면서 SEG 삽입
입력: 음 음 그건 이따 정하고 우선 자료부터 정리하자
출력: 음 음 그건 이따 정하고 <SEG> 우선 자료부터 정리하자

# 두 질문 사이 분절
입력: 그 드라마 마지막 회 봤어? 결말 어땠어?
출력: 그 드라마 마지막 회 봤어? <SEG> 결말 어땠어?

# 대명사가 포함되어도 컨텍스트로 처리하므로 분절 가능
입력: 후배한테 연락이 왔는데 걔가 갑자기 조언을 구하더라고
출력: 후배한테 연락이 왔는데 <SEG> 걔가 갑자기 조언을 구하더라고

# 연결어미(~지만)로 절이 끝나고 화제 전환
입력: 발표 준비는 열심히 했지만 막상 무대에선 좀 긴장됐어
출력: 발표 준비는 열심히 했지만 <SEG> 막상 무대에선 좀 긴장됐어

# 첫 조각의 감탄사+담화 표지어는 묶고, 뒤 사건 경계(~다가)에서 분절
입력: 응 근데 오늘은 집에서 좀 쉬다가 저녁때 잠깐 나가려고
출력: 응 근데 오늘은 집에서 좀 쉬다가 <SEG> 저녁때 잠깐 나가려고

# 비교(~것보다)로 앞뒤 절을 비교 → 비교 대상 뒤에서 분절
입력: 그냥 대충 빨리 끝내는 것보다 시간이 걸려도 제대로 하는 게 낫더라고
출력: 그냥 대충 빨리 끝내는 것보다 <SEG> 시간이 걸려도 제대로 하는 게 낫더라고

# 시간 종속절(용언+~ㄹ 때는)이 끝나고 주절로 전환
입력: 어제 처음 거기 갔을 때는 사람이 너무 많아서 정신이 하나도 없었어
출력: 어제 처음 거기 갔을 때는 <SEG> 사람이 너무 많아서 정신이 하나도 없었어

# 조건절(용언+~으면)이 끝나고 주절로 전환
입력: 이번 주말에 시간이 되면 같이 그 전시회 보러 가자
출력: 이번 주말에 시간이 되면 <SEG> 같이 그 전시회 보러 가자

# 구어체 조건절(보조사 '은'이 붙은 ~면은/~으면은)도 동일하게 그 뒤에서 분절
입력: 주말에 날씨가 풀리면은 오랜만에 자전거나 타러 나가려고
출력: 주말에 날씨가 풀리면은 <SEG> 오랜만에 자전거나 타러 나가려고

# 한 발화에 조건절이 둘 이상 반복되면 각 조건절마다 분절
입력: 이 앱은 사진을 올리면은 자동으로 보정도 해 주고 위치까지 찍히면은 지도에 표시까지 돼
출력: 이 앱은 사진을 올리면은 <SEG> 자동으로 보정도 해 주고 <SEG> 위치까지 찍히면은 <SEG> 지도에 표시까지 돼

# 자문·추측절(~ㄹ까/~을까)이 끝나고 별개의 절이 이어지면 그 뒤에서 분절
입력: 이걸 굳이 새로 사는 게 맞을까 그냥 있는 걸로 버텨도 될 것 같은데
출력: 이걸 굳이 새로 사는 게 맞을까 <SEG> 그냥 있는 걸로 버텨도 될 것 같은데

# 대등/나열 술어(~고)는 각 완결 술어마다 분절 (짧아도 완결되면 분절)
입력: 그 식당은 분위기도 좋고 음식도 맛있고 직원도 친절하더라
출력: 그 식당은 분위기도 좋고 <SEG> 음식도 맛있고 <SEG> 직원도 친절하더라

# 원인·계기(~해서/~가지고)로 한 절이 끝나고 결과 절이 이어짐
입력: 어제 늦게까지 일을 해가지고 오늘은 진짜 하루 종일 피곤하더라고
출력: 어제 늦게까지 일을 해가지고 <SEG> 오늘은 진짜 하루 종일 피곤하더라고

# 선택지가 동사구/절이면 항목 사이에서 분절
입력: 그냥 집에서 쉬거나 아니면 가까운 데 산책이라도 갈까 봐
출력: 그냥 집에서 쉬거나 <SEG> 아니면 가까운 데 산책이라도 갈까 봐

# 인용절(~다고/~라고)이 끝나고 별개의 인용 동사가 이어지면 인용 동사 앞에서 분절
입력: 걔가 다음 주에 이사 간다고 엄마한테 얘기했대
출력: 걔가 다음 주에 이사 간다고 <SEG> 엄마한테 얘기했대

---

[예시 - 분절 X]

# 감탄사+담화 표지어만으로 첫 조각이 되면 안 됨 → 뒤 실질 내용과 묶기
입력: 응 근데 그건 좀 아닌 것 같아
출력: 응 근데 그건 좀 아닌 것 같아

# 담화 표지어(근데)가 단독 조각이 되면 안 됨 → 뒤와 묶기
입력: 근데 나는 그 소식 아직 못 들었어
출력: 근데 나는 그 소식 아직 못 들었어

# 단순 명사 나열은 통째로 유지 (동사구 나열이 아님)
입력: 주말엔 마트랑 약국이랑 세탁소까지 들러야 해서 좀 바빠
출력: 주말엔 마트랑 약국이랑 세탁소까지 들러야 해서 좀 바빠

# 미완성 발화는 내부 분절 금지
입력: 그러니까 그때 내가 뭐라고 하려다가
출력: 그러니까 그때 내가 뭐라고 하려다가

# ~해서가 절을 마치지 않고 뒤 용언을 수식하는(방식) 짧은 한 호흡은 분절 금지
입력: 어제 너무 무리해서 운동했더니 다음 날 온몸이 다 쑤시더라
출력: 어제 너무 무리해서 운동했더니 다음 날 온몸이 다 쑤시더라

# 보조용언 구성(생각해 가지고)은 한 동사구 → 분절 금지
입력: 이게 제일 낫다고 생각해 가지고
출력: 이게 제일 낫다고 생각해 가지고

# 발화 끝 1어절 부사 꼬리(사실)는 앞과 묶기 → 단독 분절 금지
입력: 내 생각엔 그 방법이 제일 나은 것 같아 사실
출력: 내 생각엔 그 방법이 제일 나은 것 같아 사실

# 연결어미로 끝나도 뒤 조각이 1어절 파편이면 분절 금지
입력: 한참을 여기저기 찾아봤는데 결국
출력: 한참을 여기저기 찾아봤는데 결국

# '그러면'은 조건절이 아니라 다음 절을 이끄는 접속어 → 뒤에서 분절 금지
입력: 그러면 내가 내일 다시 한 번 확인해서 알려 줄게
출력: 그러면 내가 내일 다시 한 번 확인해서 알려 줄게

# 명사 선택('A 이나/아니면 B'에서 A·B가 명사)은 통째로 유지
입력: 마실 건 그냥 커피나 아니면 주스 정도면 충분할 것 같아
출력: 마실 건 그냥 커피나 아니면 주스 정도면 충분할 것 같아

# '~려고 하다/생각하다'는 한 동사구 → 분절 금지
입력: 내년에는 자격증을 하나 따려고 준비하고 있어
출력: 내년에는 자격증을 하나 따려고 준비하고 있어

"""
)


def _gdt_translate(text: str, src: str = "ko", tgt: str = "en") -> str | None:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception as e:
        print(f"  [번역 실패] '{text}' → {e}")
        return None


def _gdt_translate_seg(seg_text: str | None, src: str = "ko", tgt: str = "en") -> str | None:
    if not seg_text:
        return None
    from deep_translator import GoogleTranslator
    segments = seg_text.split("<SEG>")
    translated = [_gdt_translate(s.strip(), src, tgt) for s in segments if s.strip()]
    return " ".join(t for t in translated if t is not None)


def run_gdt(data: list, raw: dict, output_path: Path, delay: float = 0.2, resume: bool = True) -> None:
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
            entry["gdt_seg_trans"] = _gdt_translate_seg(entry["seg_text"])
            print(f"  gdt_seg   : {entry['gdt_seg_trans']}")
            if delay > 0:
                time.sleep(delay)

        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGDT 번역 완료. 저장: {output_path}")


def _extract_seg_line(original: str, output: str) -> str:
    """출력에서 원문 기반 SEG 태그 줄만 추출. 설명/주석 제거."""
    orig_tokens = set(re.sub(r'[^\w]', ' ', original).split())
    if not orig_tokens:
        return output.splitlines()[0].strip()

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # 괄호/주석/레이블 줄 스킵
        if line.startswith('(') or line.startswith('※') or line.startswith('*'):
            continue
        if re.match(r'^(입력|출력)\s*[:：]', line):
            continue
        cleaned = re.sub(r'\s*<SEG>\s*', ' ', line).strip()
        out_tokens = set(re.sub(r'[^\w]', ' ', cleaned).split())
        overlap = len(orig_tokens & out_tokens) / len(orig_tokens)
        if overlap >= 0.5:
            return line

    # 유효한 줄이 없으면 첫 번째 줄 반환
    return output.splitlines()[0].strip()


def _is_exact_seg_output(original: str, output: str) -> bool:
    """SEG 태그 제거 후 원문과 정확히 일치하는지 검증 (공백 정규화 후 비교)."""
    cleaned  = re.sub(r'\s*<SEG>\s*', ' ', output)
    cleaned  = re.sub(r'\s+', ' ', cleaned).strip()
    norm_orig = re.sub(r'\s+', ' ', original).strip()
    return cleaned == norm_orig


def _recover_seg_positions(original: str, modified_with_seg: str) -> str | None:
    """
    모델이 원문을 수정했을 때 SEG 경계를 원문에 복원.

    각 SEG 경계의 좌측 마지막 단어(anchor)를 원문에서 찾아 그 뒤에 SEG를 삽입.
    anchor를 원문에서 찾지 못하면 None 반환.
    """
    segments = [s.strip() for s in modified_with_seg.split('<SEG>')]
    if len(segments) <= 1:
        return None

    insert_positions: list[int] = []
    search_start = 0

    for seg in segments[:-1]:
        words = seg.split()
        if not words:
            return None

        # 마지막 단어(구두점 제거)를 anchor로 사용. 2개 단어로 시도 후 1개로 fallback.
        for n_anchor in (2, 1):
            anchors = [re.sub(r'[?!.,]+$', '', w) for w in words[-n_anchor:]]
            pattern = re.compile(
                r'\s+'.join(re.escape(a) for a in anchors) + r'[?!.,]*'
            )
            m = pattern.search(original, search_start)
            if m:
                break
        else:
            return None

        insert_positions.append(m.end())
        search_start = m.end()

    # 뒤에서부터 삽입해야 앞 위치가 밀리지 않음
    result = original
    for pos in reversed(sorted(insert_positions)):
        result = result[:pos] + ' <SEG>' + result[pos:]

    return result if _is_exact_seg_output(original, result) else None


# 단독으로 분절될 수 없는 비자립 토큰: 담화표지·감탄사·접속부사·후치 부사
# (모델이 프롬프트 금지에도 불구하고 단독 분절하는 경향이 강해 결정론적으로 병합)
_FILLER_TOKENS = {
    '어', '음', '으음', '아', '응', '그', '네', '예', '자', '막', '뭐', '뭔가',
    '근데', '그러니까', '그니까', '그래', '그치', '그러고', '맞아', '맞어',
    '완전', '아니', '그럼', '그러면', '그러면은', '어쨌든', '하여튼', '그리고', '그래서',
    '그래가지고', '그래갖고', '가지고', '또', '또한', '왜냐면', '왜',
    '좀', '이제', '사실', '그냥',
}


def _strip_token_punct(w: str) -> str:
    return re.sub(r'[?!.,…]+$', '', re.sub(r'^[?!.,…]+', '', w))


def _seg_tokens(seg: str) -> list[str]:
    return [t for t in (_strip_token_punct(w) for w in seg.split()) if t]


def _is_filler_only(seg: str) -> bool:
    """조각이 비자립 토큰(담화표지·감탄사·접속부사)만으로 이루어졌는지."""
    toks = _seg_tokens(seg)
    return bool(toks) and all(t in _FILLER_TOKENS for t in toks)


def _merge_nonstanding_segments(seg_text: str) -> str:
    """
    프롬프트가 금지하지만 모델이 자주 만드는 두 패턴을 결정론적으로 병합.
    - A: 담화표지·감탄사만으로 된 비-마지막 조각 → 다음 조각에 병합 (예: '근데 <SEG> 그때...')
    - B: 1어절 꼬리 파편 또는 비자립 토큰뿐인 마지막 조각 → 앞 조각에 병합 (예: '...해서 <SEG> 좀')
    SEG를 제거하면 원문과 동일해야 하는 불변식은 유지된다(토큰 내용 변경 없음).
    """
    segs = [s.strip() for s in seg_text.split('<SEG>') if s.strip()]
    if len(segs) <= 1:
        return seg_text

    # A: 비자립 첫/중간 조각을 다음 조각 앞에 병합
    merged: list[str] = []
    i = 0
    while i < len(segs):
        cur = segs[i]
        if i < len(segs) - 1 and _is_filler_only(cur):
            segs[i + 1] = cur + ' ' + segs[i + 1]
        else:
            merged.append(cur)
        i += 1
    segs = merged

    # B: 1어절 꼬리 파편 / 비자립 마지막 조각을 앞 조각에 병합 (연쇄 적용)
    while len(segs) >= 2:
        last = segs[-1]
        if len(_seg_tokens(last)) <= 1 or _is_filler_only(last):
            segs[-2] = segs[-2] + ' ' + last
            segs.pop()
        else:
            break

    return ' <SEG> '.join(segs)


def mark_segmentation(client, text: str, model: str, provider: str = "openai", max_retries: int = 5) -> str:
    user_msg = text

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
                    messages=[{"role": "user", "content": user_msg}],
                )
                result = response.content[0].text.strip()
            elif provider == "ollama":
                # "입력 ... 출력:" 형식으로 모델이 출력만 생성하도록 유도
                # extra_body: think=False → Qwen3 thinking 모드 비활성화
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": f"입력:\n{text}"},
                        {"role": "assistant", "content": "출력:\n"},
                    ],
                    temperature=0,
                    extra_body={"think": False},
                )
                result = response.choices[0].message.content.strip()
                result = _extract_seg_line(text, result)
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0,
                )
                result = response.choices[0].message.content.strip()

            if not _is_exact_seg_output(text, result):
                recovered = _recover_seg_positions(text, result)
                if recovered:
                    print(f"  [복원] 원문 수정 감지 → SEG 위치 복원 성공: {recovered[:80]!r}")
                    result = recovered
                else:
                    print(f"  [경고] 원문 수정 감지, 복원 실패 ({attempt+1}/{max_retries}): {result[:80]!r}")
                    if attempt < max_retries - 1:
                        continue
                    print(f"  [폴백] 원문 그대로 반환")
                    return text

            # 문장 맨 앞/끝 SEG 제거, 연속 SEG 제거
            result = re.sub(r'^(\s*<SEG>\s*)+', '', result)
            result = re.sub(r'(\s*<SEG>\s*)+$', '', result)
            result = re.sub(r'(\s*<SEG>\s*){2,}', ' <SEG> ', result)
            result = result.strip()
            # 비자립 조각(담화표지 단독 / 1어절 꼬리 파편) 결정론적 병합
            result = _merge_nonstanding_segments(result)
            return result

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
    _base = Path(__file__).resolve().parent.parent.parent.parent / "evaluation" / "KsponSpeech"
    _transcribe_dir = _base / "transcribe"
    _results_dir    = _base / "results"

    parser = argparse.ArgumentParser(description="GPT 기반 의미 분절 마킹")
    parser.add_argument("--input",   type=str, required=True,
                        help="입력 JSON 파일명 (evaluation/KsponSpeech/transcribe/ 기준, 예: eval_clean.json)")
    parser.add_argument("--output",  type=str, default=None,
                        help="출력 JSON 파일명 (기본: 입력 파일명에 _seg 추가, evaluation/KsponSpeech/results/ 저장)")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "claude", "ollama"],
                        help="사용할 API 제공자 (openai, claude, ollama)")
    parser.add_argument("--model",   type=str, default=None,
                        help="모델명 (openai 기본값: gpt-4o-mini, claude 기본값: claude-haiku-4-5)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API 키 (미입력 시 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 환경변수 사용)")
    parser.add_argument("--delay",     type=float, default=0.5, help="요청 간 딜레이(초, workers>1이면 무시)")
    parser.add_argument("--workers",   type=int, default=1,
                        help="동시 요청 수 (ollama 병렬 배치용, 기본 1=순차). ollama는 OLLAMA_NUM_PARALLEL을 함께 키울 것")
    parser.add_argument("--save-every", type=int, default=50,
                        help="병렬 모드에서 N개 완료마다 중간 저장 (기본 50)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="원문 수정 감지 시 재시도 횟수 (기본 5, 낮추면 폴백 빨라짐)")
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
    elif args.provider == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
        client = OpenAI(base_url=f"http://{ollama_host}/v1", api_key="ollama")
        model = args.model or "exaone3.5:32b"
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
        client = OpenAI(api_key=api_key)
        model = args.model or "gpt-4o-mini"
    input_path = _transcribe_dir / args.input

    _results_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        p = Path(args.output)
        output_path = p if p.is_absolute() or len(p.parts) > 1 else _results_dir / p
    else:
        output_path = _results_dir / (input_path.stem + "_seg" + input_path.suffix)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    raw_in = json.loads(input_path.read_text(encoding="utf-8"))
    base_data = raw_in["data"] if isinstance(raw_in, dict) else raw_in

    if output_path.exists():
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        data = raw["data"] if isinstance(raw, dict) else raw
        # 출력 파일의 데이터를 file 기준으로 병합
        existing = {e["file"]: e for e in data}
        data = [existing.get(e["file"], e) for e in base_data]
        raw = {"data": data}
    else:
        raw = {"data": [dict(e) for e in base_data]}
        print(f"출력 파일 생성: {output_path}")

    data = raw["data"]

    pending = [i for i, e in enumerate(data) if not (args.resume and e.get("seg_text"))]
    skipped = len(data) - len(pending)
    if skipped:
        print(f"건너뜀 (이미 처리됨): {skipped}개")

    budget_exceeded = False

    if args.workers <= 1:
        # ── 순차 처리 ────────────────────────────────────────────────
        for i in pending:
            entry = data[i]
            text = entry["text"]
            print(f"[{i+1}/{len(data)}] {entry['file']}")
            print(f"  원문: {text}")
            try:
                seg_text = mark_segmentation(client, text, model, args.provider, args.max_retries)
                entry["seg_text"] = seg_text
                print(f"  분절: {seg_text}")
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in ("insufficient_quota", "billing", "budget", "exceeded", "402")):
                    print(f"  예산 초과 오류 — 분절 중단, 번역으로 이동합니다.\n  ({msg})")
                    budget_exceeded = True
                    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                    break
                print(f"  오류: {e}")
                entry["seg_text"] = None
            output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.delay > 0:
                time.sleep(args.delay)
    else:
        # ── 병렬 처리 (ollama 동시 배치) ──────────────────────────────
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        save_lock = threading.Lock()

        def work(i):
            entry = data[i]
            try:
                return i, mark_segmentation(client, entry["text"], model, args.provider, args.max_retries), None
            except Exception as e:
                return i, None, str(e)

        print(f"병렬 처리 시작: {len(pending)}개, workers={args.workers}")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(work, i): i for i in pending}
            for fut in as_completed(futures):
                i, seg_text, err = fut.result()
                entry = data[i]
                if err is not None:
                    print(f"  [{i+1}] 오류: {err}")
                    entry["seg_text"] = None
                else:
                    entry["seg_text"] = seg_text
                    print(f"[{i+1}/{len(data)}] {entry['file']}: {seg_text}")
                done += 1
                if done % args.save_every == 0:
                    with save_lock:
                        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"  ── 중간 저장 ({done}/{len(pending)}) ──")
        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    if budget_exceeded:
        print(f"\n분절 중단 (예산 초과). 저장: {output_path}")
    else:
        print(f"\n완료. 저장: {output_path}")

    if args.gdt:
        print("\nGDT 번역 시작...")
        run_gdt(data, raw, output_path, delay=args.gdt_delay, resume=args.resume)


if __name__ == "__main__":
    main()
