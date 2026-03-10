"""
KsponSpeech .trn 파일을 eval_clean.json 형식으로 변환하는 스크립트

정규화 처리:
  - 노이즈 태그 (b/, o/, l/, n/, u/) 제거
  - 이중 표기 (숫자표기)/(한글표기) → 한글 표기 채택
  - 한국어 필러 어절 (어/, 뭐/ 등) 제거
  - 여분 공백 정리

출력 형식: {"stats": {...}, "data": [{"file": "...", "text": "..."}, ...]}
"""

import re
import json
import argparse
from pathlib import Path


# 단순 노이즈 태그: b/ o/ l/ n/ u/ (단독 토큰)
_NOISE_TAG = re.compile(r'\b[bolnu]/\s*')

# 이중 표기: (표기1)/(표기2)
#   - 표기2가 *로 끝나면 (잘린 형태) → 표기1 채택
#   - 그 외 → 표기2 채택 (한글 정규 표기)
_DUAL_NOTATION = re.compile(r'\(([^)]+)\)/\(([^)]*)\)')

# 한국어 어절 뒤 /: 슬래시만 제거, 어절 자체는 유지  예) 그/ → 그, 어/ → 어
_KOREAN_SLASH = re.compile(r'([가-힣]+)/')


def _replace_dual(m: re.Match) -> str:
    first, second = m.group(1), m.group(2)
    # second가 *로 끝나면 잘린 형태 → first(완전한 말) 선택
    return first if second.endswith('*') else second


def normalize(text: str) -> str:
    text = _DUAL_NOTATION.sub(_replace_dual, text)  # (A)/(B) → 규칙에 따라 선택
    text = _NOISE_TAG.sub('', text)                 # b/ o/ l/ n/ u/ 제거
    text = _KOREAN_SLASH.sub(r'\1', text)           # 한글어절/ → 슬래시만 제거
    text = re.sub(r'\+', '', text)                  # + (음 연장 기호) 제거
    text = re.sub(r'\s+', ' ', text).strip()        # 공백 정리
    return text


def word_count(text: str) -> int:
    """어절 수 (공백 기준 분리)"""
    return len(text.split())


def parse_trn(trn_path: Path, min_words: int, count: int) -> list[dict]:
    entries = []
    with trn_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ' :: ' not in line:
                continue

            path_part, raw_text = line.split(' :: ', 1)
            file_name = Path(path_part).stem  # 확장자 없는 파일명

            text = normalize(raw_text)

            if not text:
                continue
            if word_count(text) < min_words:
                continue

            entries.append({"file": file_name, "text": text})

            if count > 0 and len(entries) >= count:
                break

    return entries


def main():
    parser = argparse.ArgumentParser(description=".trn → eval_clean.json 변환")
    parser.add_argument("--input",     type=str,   default=str(Path(__file__).resolve().parent.parent / "data" / "transcribe" / "eval_clean.trn"),
                        help=".trn 입력 파일 경로")
    parser.add_argument("--output",    type=str,   default=None,
                        help="출력 JSON 경로 (기본: 입력 파일명.json)")
    parser.add_argument("--min-words", type=int,   default=5,
                        help="포함할 최소 어절 수 (기본: 5)")
    parser.add_argument("--count",     type=int,   default=0,
                        help="수집할 최대 문장 수 (기본: 0 = 전체)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    print(f"파싱 중: {input_path}")
    print(f"  최소 어절: {args.min_words}  |  최대 수집: {args.count if args.count > 0 else '전체'}")

    entries = parse_trn(input_path, min_words=args.min_words, count=args.count)

    raw = {
        "stats": {
            "total_files": len(entries),
            "min_words_filter": args.min_words,
            "source": input_path.name,
        },
        "data": entries,
    }

    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"완료: {len(entries)}개 문장 저장 → {output_path}")


if __name__ == "__main__":
    main()
