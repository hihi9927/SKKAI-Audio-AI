"""
SAT (Segment Any Text) 기반 한국어 문장 분절 스크립트
모델: segment-any-text/sat-12l-sm
설치: pip install wtpsplit
"""

import argparse
import json
from pathlib import Path

from wtpsplit import SaT


def segment_text(model, text: str, language: str = "ko") -> list[str]:
    segments = model.split(text, lang_code=language)
    return [s.strip() for s in segments if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="SAT 기반 한국어 문장 분절")
    parser.add_argument("--text",     type=str, default=None,          help="분절할 텍스트 (직접 입력)")
    parser.add_argument("--file",     type=str, default=None,          help="입력 텍스트 파일 경로")
    parser.add_argument("--lang",     type=str, default="ko",          help="언어 코드 (기본: ko)")
    parser.add_argument("--model",    type=str, default="sat-12l-sm", help="SAT 모델 이름 (sat-3l-sm / sat-6l-sm / sat-12l-sm / sat-12l)")
    parser.add_argument("--output",   type=str, default=None,          help="결과 저장 JSON 경로 (선택)")
    parser.add_argument("--device",   type=str, default="cuda",        help="디바이스 (cuda/cpu)")
    args = parser.parse_args()

    if args.text is None and args.file is None:
        # 기본 예시 텍스트
        args.text = (
            "안녕하세요 오늘 날씨가 정말 좋네요 "
            "점심은 뭐 드셨나요 저는 오늘 비빔밥을 먹었어요 "
            "다음 주에 회의가 있는데 준비를 잘 해야 할 것 같아요"
            "이번 주말에는 친구들과 영화를 보러 갈 예정입니다"
            "새로운 프로젝트를 시작했는데 기대가 됩니다"
            "한 문장이 너무 길어서 분절이 필요한 경우도 있죠"
        )

    print(f"모델 로딩 중: {args.model}")
    model = SaT(args.model)
    model.half().to(args.device)
    print("모델 로드 완료\n")

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text

    print(f"입력 텍스트:\n{text}\n")
    print("-" * 50)

    segments = segment_text(model, text, language=args.lang)

    print(f"분절 결과 ({len(segments)}개 문장):")
    for i, seg in enumerate(segments, 1):
        print(f"  [{i:2d}] {seg}")

    if args.output:
        result = {"input": text, "language": args.lang, "segments": segments}
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
