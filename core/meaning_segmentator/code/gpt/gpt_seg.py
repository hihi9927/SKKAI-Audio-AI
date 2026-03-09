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
    "너는 지금 국어교사야. "
    "주어진 문장에서 의미가 분절되는 지점을 <seg>로 표시해야 해. "
    "의미가 분절되는 지점은 해당 지점을 끊어도 앞뒤 맥락이 자연스러운 곳이야"
    "어미 같은 품사를 고려해도 되고 구두점을 고려해도 돼. "
    "어떤 문장에는 분절 지점이 없을 수도 있어. "
    "원문 텍스트에 <seg> 태그만 삽입해서 반환해. 다른 설명이나 텍스트는 절대 추가하지 마."
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
    parser.add_argument("--input",   type=str, default="C:\Users\jduh1\Desktop\STiTy\core\meaning_segmentator\data\transcribe\eval_clean_100.json")
    parser.add_argument("--output",  type=str, default=None, help="출력 JSON 경로 (기본: 입력 파일 덮어쓰기)")
    parser.add_argument("--model",   type=str, default="gpt-4o-mini", help="OpenAI 모델명")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API 키 (미입력 시 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--delay",   type=float, default=0.5, help="요청 간 딜레이(초)")
    parser.add_argument("--resume",  action="store_true", help="이미 seg_text가 있는 항목은 건너뜀")
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
