"""
GPT API를 이용한 의미 분절 마킹 스크립트
eval_clean_100.json의 각 텍스트에 <SEG> 태그로 분절 지점을 표시하고
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
음성 인식 결과(구어체, 일상 대화)를 번역 단위로 나누기 위해 <SEG> 태그를 삽입해.
목표는 각 분절이 번역기가 독립적으로 처리할 수 있는 의미 완결 단위가 되는 것이야.

[핵심 원칙]
- 분절은 최소화가 기본: 확실하지 않으면 분절하지 마
- 구어체 특성상 문법이 불완전해도 의미상 한 덩어리면 분절하지 마
- 분절 기준은 어절 수가 아니라 의미 완결성과 번역 독립성

[분절하는 경우]
아래 두 조건을 모두 충족할 때만 분절해.
단, 조건을 충족하더라도 아래 [절대 분절하지 않는 경우]에 해당하면 분절하지 마. 절대 분절 금지 규칙이 항상 우선한다.

1. 두 절 사이에 의미적 전환이 명확한 경우
   (= 주어·시제·화제가 바뀌거나, 앞 절이 완결된 사건/상태를 서술하고 뒤 절이 독립된 새 사건을 시작하는 경우)
2. 분절 후 각 조각이 번역기가 앞뒤 문맥 없이도 독립적으로 번역할 수 있는 의미 완결 단위인 경우

[절대 분절하지 않는 경우 — 위 조건을 충족해도 아래에 해당하면 분절 금지]
- 감탄사/추임새(아, 어, 응, 그치, 그래, 아니 등)가 단독으로 한 조각이 되는 경우
  → 반드시 뒤 내용과 함께 묶어야 함
- 담화 표지어(근데, 그니까, 그래서, 그래가지고, 그래갖고, 뭔가 등)가 단독 조각이 되는 경우
  → 반드시 뒤 내용과 함께 묶어야 함
- 나열/열거 구조 내부 (A이라든가, A든지 B든지, A도 B도, A이나 B 형태는 통째로 유지)
- 연결어미(~고, ~서, ~면, ~는데 등) 뒤 절이 단독으로 의미를 전달하지 못하는 경우
- 분절 후 어느 한 조각이 번역 문맥 없이는 의미를 알 수 없는 파편이 되는 경우
- 미완성 발화(문장이 끝나지 않고 잘린 경우)는 내부 분절 금지

[출력 규칙]
- 원문에 <SEG> 태그만 삽입, 원문 텍스트는 절대 수정하지 마
- 문장 맨 앞/맨 끝에는 태그 삽입 금지
- <SEG> 태그가 연속으로 붙는 경우(<SEG> <SEG>)는 절대 금지
- 구두점(. ? ! ,)은 반드시 앞 텍스트에 붙임 — <SEG> 태그 바로 뒤에 구두점이 오면 안 됨
- <SEG> 태그 앞뒤에 공백을 무조건 추가
- 분절이 필요 없으면 원문을 그대로 출력
- 다른 설명, 주석, 텍스트 절대 추가 금지

---

[예시 - 분절 O]

# 독립된 두 절, 의미 전환 명확
입력: 저는 매일 아침 일찍 일어나서 조깅을 한 시간씩 합니다
출력: 저는 매일 아침 일찍 일어나서 <SEG> 조깅을 한 시간씩 합니다

# 완결된 사건 + 새 사건, 구두점은 앞 텍스트에 붙임
입력: 진짜 맛있더라. 나 다음에 또 가고 싶어
출력: 진짜 맛있더라. <SEG> 나 다음에 또 가고 싶어

# 담화 표지어(그래서)가 앞 절이 아닌 뒤 내용과 묶임
입력: 걔 결국 회사 그만뒀어. 그래서 지금 다른 데 알아보고 있대
출력: 걔 결국 회사 그만뒀어. <SEG> 그래서 지금 다른 데 알아보고 있대

# 긴 발화에서 분절 여러 개
입력: 어제 친구 만났는데 밥도 먹고 카페도 갔어. 근데 거기서 옛날 동창을 딱 마주쳤어. 진짜 오랜만이어서 되게 반가웠어
출력: 어제 친구 만났는데 밥도 먹고 카페도 갔어. <SEG> 근데 거기서 옛날 동창을 딱 마주쳤어. <SEG> 진짜 오랜만이어서 되게 반가웠어

# 구두점 없이 화제 전환
입력: 나 오늘 회사에서 발표했어 생각보다 잘 됐어
출력: 나 오늘 회사에서 발표했어 <SEG> 생각보다 잘 됐어

# 구두점 없이 주어 전환
입력: 나는 짜장면 시켰는데 걔는 짬뽕 먹더라
출력: 나는 짜장면 시켰는데 <SEG> 걔는 짬뽕 먹더라

# 구두점 없이 시제/상황 전환
입력: 어제는 진짜 힘들었는데 오늘은 좀 괜찮아
출력: 어제는 진짜 힘들었는데 <SEG> 오늘은 좀 괜찮아

# 담화 표지어(근데)로 화제 전환, 구두점 없음
입력: 거기 분위기 되게 좋더라 근데 가격이 좀 세더라고
출력: 거기 분위기 되게 좋더라 <SEG> 근데 가격이 좀 세더라고

---

[예시 - 분절 X]

# 감탄사/추임새는 뒤 내용과 묶기
입력: 아 진짜 거기 사람이 엄청 많았어
출력: 아 진짜 거기 사람이 엄청 많았어

# 담화 표지어(근데)가 단독 조각이 되면 안 됨 → 뒤와 묶기
입력: 근데 나는 그 얘기 듣고 좀 이상하다고 생각했어
출력: 근데 나는 그 얘기 듣고 좀 이상하다고 생각했어

# 연결어미 뒤 절이 단독으로 의미 전달 불가
입력: 오늘 날씨가 좋아서 기분이 좋았어
출력: 오늘 날씨가 좋아서 기분이 좋았어

# 나열 구조는 통째로 유지
입력: 주말에 청소도 하고 빨래도 하고 장도 봐야 해서 바빴어
출력: 주말에 청소도 하고 빨래도 하고 장도 봐야 해서 바빴어

# 미완성 발화는 내부 분절 금지
입력: 근데 그때 내가 뭔가 말하려고 했는데
출력: 근데 그때 내가 뭔가 말하려고 했는데

# 분절 불필요 시 원문 그대로 출력
입력: 나 어제 진짜 피곤했어
출력: 나 어제 진짜 피곤했어

# 구두점 없지만 ~서로 연결된 이유-결과 구조 → 분절 X
입력: 어제 늦게 자서 오늘 진짜 피곤해
출력: 어제 늦게 자서 오늘 진짜 피곤해

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
    _base = Path(__file__).resolve().parent.parent.parent.parent / "evaluation" / "KsponSpeech"
    _transcribe_dir = _base / "transcribe"
    _results_dir    = _base / "results"

    parser = argparse.ArgumentParser(description="GPT 기반 의미 분절 마킹")
    parser.add_argument("--input",   type=str, required=True,
                        help="입력 JSON 파일명 (evaluation/KsponSpeech/transcribe/ 기준, 예: eval_clean.json)")
    parser.add_argument("--output",  type=str, default=None,
                        help="출력 JSON 파일명 (기본: 입력 파일명에 _seg 추가, evaluation/KsponSpeech/results/ 저장)")
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
