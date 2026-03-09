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
- 담화 표지어가 단독으로 앞 조각이 되는 경우
  (근데, 그니까, 그래서, 아니, 그치, 그래 가지고, 그래가지고, 어, 응, 뭔가 등)
- 분절 후 어느 한 조각이 번역 문맥 없이는 의미를 알 수 없는 파편인 경우
- 의문문 내부 (하나의 질문은 통째로 유지)
- 연결어미(~고, ~서, ~면, ~는데 등) 뒤라도 뒤 절이 단독으로 의미를 전달하지 못하는 경우
- 문장 중간에 등장하는 감탄사/추임새(아, 어, 응, 그치 등)

[출력 규칙]
- 원문에 <seg> 태그만 삽입, 원문 텍스트는 절대 수정하지 마
- 문장 맨 앞/맨 끝에는 태그 삽입 금지
- 다른 설명, 주석, 텍스트 절대 추가 금지

[예시 - 분절 O: 두 절 모두 독립적이고 5어절 이상]
입력: 저는 매일 아침 일찍 일어나서 조깅을 한 시간씩 합니다
출력: 저는 매일 아침 일찍 일어나서<seg>조깅을 한 시간씩 합니다

입력: 형이 대학원을 졸업하고 나서 바로 취업을 하게 됐는데 연봉도 꽤 높다고 들었어
출력: 형이 대학원을 졸업하고 나서<seg> 바로 취업을 하게 됐는데<seg>연봉도 꽤 높다고 들었어

입력: 여기 사람이 워낙 많으니까 예약을 미리 해두는 게 좋을 것 같아서 어제 저녁에 전화를 해뒀어
출력: 여기 사람이 워낙 많으니까<seg> 예약을 미리 해두는 게 좋을 것 같아서<seg>어제 저녁에 전화를 해뒀어

[예시 - 분절 X: 짧은 문장 또는 단일 의미 단위]
입력: 오늘 날씨가 너무 좋아서 기분이 좋다
출력: 오늘 날씨가 너무 좋아서 기분이 좋다

입력: 그냥 좀 쉬고 싶어서 집에 있었어
출력: 그냥 좀 쉬고 싶어서 집에 있었어

[예시 - 분절 X: 담화 표지어가 단독 조각이 되는 경우]
입력: 그니까 걔가 원래 그런 스타일이잖아
출력: 그니까 걔가 원래 그런 스타일이잖아

입력: 근데 사실 나도 잘 모르겠어
출력: 근데 사실 나도 잘 모르겠어

[예시 - 분절 X: 의문문은 통째로 유지]
입력: 너 오늘 저녁에 시간 있으면 같이 밥 먹을 수 있어
출력: 너 오늘 저녁에 시간 있으면 같이 밥 먹을 수 있어

[예시 - 분절 X: 연결어미 뒤 절이 짧거나 불완전]
입력: 거기서 좀 일하고 왔어
출력: 거기서 좀 일하고 왔어

입력: 어제 친구 만나고 밥도 먹었어
출력: 어제 친구 만나고 밥도 먹었어
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
    parser.add_argument("--input",   type=str, default=r"C:\Users\jduh1\Desktop\STiTy\core\meaning_segmentator\data\transcribe\eval_clean_100.json")
    parser.add_argument("--output",  type=str, default=None, help="출력 JSON 경로 (기본: 입력 파일 덮어쓰기)")
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
    output_path = Path(args.output or args.input)

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
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
