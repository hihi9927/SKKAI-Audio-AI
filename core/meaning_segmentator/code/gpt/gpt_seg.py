"""
GPT API를 이용한 의미 분절 마킹 스크립트
eval_clean_100.json의 각 텍스트에 <seg> 태그로 분절 지점을 표시하고
"seg_text" 필드로 저장합니다.
"""

import json
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI

SYSTEM_PROMPT = (
"""너는 구어체 한국어 텍스트의 의미 분절 전문가야.
음성 인식 결과(구어체, 일상 대화)를 번역 단위로 나누기 위해 <seg> 태그를 삽입해.
목표는 각 분절이 번역기가 독립적으로 처리할 수 있는 의미 완결 단위가 되는 것이야.

[핵심 원칙]
- 분절은 최소화가 기본: 확실하지 않으면 분절하지 마
- 구어체 특성상 문법이 불완전해도 의미상 한 덩어리면 분절하지 마
- 분절 기준은 어절 수가 아니라 의미 완결성과 번역 독립성

[분절하는 경우 - 모두 충족해야 함]
1. 두 절 사이에 의미적 전환이 명확하고
2. 분절 후 각 조각이 번역기가 독립적으로 처리할 수 있는 의미 완결 단위인 경우

[절대 분절하지 않는 경우]
- 감탄사/추임새(아, 어, 응, 그치, 그래, 아니 등)가 단독으로 한 조각이 되는 경우
  → 반드시 뒤 내용과 함께 묶어야 함
- 담화 표지어(근데, 그니까, 그래서, 그래가지고, 그래갖고, 뭔가 등)가 단독 조각이 되는 경우
  → 반드시 뒤 내용과 함께 묶어야 함
- 나열/열거 구조 내부 (A이라든가, A든지 B든지, A이나 B 형태의 나열은 통째로 유지)
- 분절 후 어느 한 조각이 번역 문맥 없이는 의미를 알 수 없는 파편인 경우
- 미완성 발화(문장이 끝나지 않고 잘린 경우)는 내부 분절 금지
- 연결어미(~고, ~서, ~면, ~는데 등) 뒤라도 뒤 절이 단독으로 의미를 전달하지 못하는 경우

[출력 규칙]
- 원문에 <seg> 태그만 삽입, 원문 텍스트는 절대 수정하지 마
- 문장 맨 앞/맨 끝에는 태그 삽입 금지
- 구두점(. ? ! ,)은 반드시 앞 텍스트에 붙임 — <seg> 태그 바로 뒤에 구두점이 오면 안 됨
- <seg> 태그 앞뒤에 공백을 넣지 마
- 다른 설명, 주석, 텍스트 절대 추가 금지

[예시 - 분절 O: 독립적인 두 절, 구두점은 앞 텍스트에 붙임]
입력: 저는 매일 아침 일찍 일어나서 조깅을 한 시간씩 합니다
출력: 저는 매일 아침 일찍 일어나서<seg>조깅을 한 시간씩 합니다

입력: 진짜 맛있더라. 나 다음에 또 가고 싶어
출력: 진짜 맛있더라.<seg>나 다음에 또 가고 싶어

입력: 걔 결국 회사 그만뒀어. 그래서 지금 다른 데 알아보고 있대
출력: 걔 결국 회사 그만뒀어.<seg>그래서 지금 다른 데 알아보고 있대

[예시 - 분절 X: 감탄사/추임새는 뒤 내용과 묶기]
입력: 아 진짜 거기 사람이 엄청 많았어
출력: 아 진짜 거기 사람이 엄청 많았어

입력: 어 맞아 그거 나도 들었는데 되게 신기하더라
출력: 어 맞아 그거 나도 들었는데<seg>되게 신기하더라

[예시 - 분절 X: 나열 구조는 통째로 유지]
입력: 주말에 청소도 하고 빨래도 하고 장도 봐야 해서 바빴어
출력: 주말에 청소도 하고 빨래도 하고 장도 봐야 해서 바빴어

[예시 - 분절 X: 미완성 발화는 내부 분절 금지]
입력: 근데 그때 내가 뭔가 말하려고 했는데
출력: 근데 그때 내가 뭔가 말하려고 했는데

입력: 거기서 좀 일하고 왔어
출력: 거기서 좀 일하고 왔어
"""
)


def mark_segmentation(client: OpenAI, text: str, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="GPT 기반 의미 분절 마킹")
    parser.add_argument("--input",   type=str, default=r"C:\Users\jduh1\Desktop\STiTy\core\meaning_segmentator\data\transcribe\eval_clean_3_change_prompt.json")
    parser.add_argument("--output",  type=str, default=None, help="출력 JSON 경로 (기본: 입력 파일명에 _seg 추가)")
    parser.add_argument("--model",   type=str, default="gpt-4o-mini", help="OpenAI 모델명")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API 키 (미입력 시 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--delay",   type=float, default=0.5, help="요청 간 딜레이(초)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="이미 seg_text가 있는 항목도 재처리")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")

    client = OpenAI(api_key=api_key)
    input_path = Path(args.input)

    if args.output:
        p = Path(args.output)
        output_path = p if p.is_absolute() or len(p.parts) > 1 else input_path.parent / p
    else:
        output_path = input_path.parent / (input_path.stem + "_seg" + input_path.suffix)

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

    for i, entry in enumerate(data):
        if args.resume and "seg_text" in entry:
            print(f"[{i+1}/{len(data)}] 건너뜀 (이미 처리됨): {entry['file']}")
            continue

        text = entry["text"]
        print(f"[{i+1}/{len(data)}] {entry['file']}")
        print(f"  원문: {text}")

        try:
            seg_text = mark_segmentation(client, text, args.model)
            entry["seg_text"] = seg_text
            print(f"  분절: {seg_text}")
        except Exception as e:
            print(f"  오류: {e}")
            entry["seg_text"] = None

        # 진행 중 중간 저장
        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
