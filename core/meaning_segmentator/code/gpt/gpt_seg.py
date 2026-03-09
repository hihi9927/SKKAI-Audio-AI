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
"""너는 국어 텍스트 분석 전문가야.
주어진 문장에서 의미 분절 지점에 <seg> 태그를 삽입해야 해.

[분절 기준 - 우선순위 순]
1. 절/구 경계: 주어+서술어 구조가 완결되는 지점
2. 연결어미 뒤: ~고, ~며, ~서, ~면, ~지만, ~는데 등 이후
3. 부사절/관형절이 끝나는 지점
4. 쉼표(,) 또는 세미콜론(;) 위치

[분절하지 않는 경우]
- 조사, 보조용언, 의존명사 앞뒤
- 짧은 명사구 내부
- 전체 문장이 단일 의미 단위인 경우

[출력 규칙]
- 원문에 <seg> 태그만 삽입
- 문장 맨 앞/맨 끝에는 태그 삽입 금지
- 다른 설명, 주석, 텍스트 절대 추가 금지

[예시]
입력: 비가 오고 바람이 불었다
출력: 비가 오고<seg>바람이 불었다

입력: 나는 어제 학교에 갔다
출력: 나는 어제 학교에 갔다

입력: 그가 웃으면서 말했는데 아무도 듣지 않았다
출력: 그가 웃으면서 말했는데<seg>아무도 듣지 않았다
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

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

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
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
