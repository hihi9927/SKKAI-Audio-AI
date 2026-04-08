"""
KsponSpeech train.trn → JSON 변환 스크립트.

train.trn 포맷:
    KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm :: 아/ 몬 소리야, 그건 또. b/

텍스트 정규화 규칙:
    - (구어체)/(문어체) → 문어체(오른쪽) 선택
    - b/, n/, o/ 등 단일 영문자 노이즈 마커 제거
    - +, * (반복/불명확 마커) 제거
    - 나머지 / 제거
    - 공백 정리

Usage:
    # 기본 (KsponSpeech_0001~0010, 10000개 → train.json)
    python extract_trn_to_json.py

    # 폴더 범위 지정
    python extract_trn_to_json.py --folders 0011 0020 --output ../transcribe/train_0011_0020.json

    # trn 파일 직접 지정
    python extract_trn_to_json.py --trn ../transcribe/train.trn --folders 0001 0010 --output ../transcribe/train.json
"""
import argparse
import json
import re
import sys
from pathlib import Path


_FOLDER_PATTERN_CACHE: dict[tuple[int, int], re.Pattern] = {}


def build_folder_pattern(start: int, end: int) -> re.Pattern:
    key = (start, end)
    if key not in _FOLDER_PATTERN_CACHE:
        parts = [f"KsponSpeech_{n:04d}/" for n in range(start, end + 1)]
        _FOLDER_PATTERN_CACHE[key] = re.compile("|".join(re.escape(p) for p in parts))
    return _FOLDER_PATTERN_CACHE[key]


def clean_text(raw: str) -> str:
    t = raw
    # (구어체)/(문어체) → 문어체 선택
    t = re.sub(r"\([^)]*\)/\(([^)]*)\)", r"\1", t)
    # 단일 영문자 노이즈 마커 제거: b/, n/, o/, l/, u/ 등
    t = re.sub(r"\b[a-zA-Z]\/", "", t)
    # 반복(+), 불명확(*) 마커 제거
    t = re.sub(r"[+*]", "", t)
    # 나머지 / 제거
    t = re.sub(r"/", "", t)
    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract(trn_path: str, folder_start: int, folder_end: int) -> list[dict]:
    pattern = build_folder_pattern(folder_start, folder_end)
    records = []

    with open(trn_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not pattern.search(line):
                continue
            parts = line.split(" :: ", 1)
            if len(parts) != 2:
                continue
            path, raw_text = parts
            fname = Path(path).stem  # 확장자 제거
            text = clean_text(raw_text)
            if text:
                records.append({"file": fname, "text": text})

    return records


def main():
    default_trn = Path(__file__).parent.parent / "transcribe" / "train.trn"

    p = argparse.ArgumentParser(description="KsponSpeech train.trn → JSON")
    p.add_argument(
        "--trn",
        default=str(default_trn),
        help="train.trn 경로 (기본값: ../transcribe/train.trn)",
    )
    p.add_argument(
        "--folders",
        nargs=2,
        metavar=("START", "END"),
        default=["0001", "0010"],
        help="추출할 폴더 범위 (기본값: 0001 0010)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="출력 JSON 경로 (기본값: ../transcribe/train_XXXX_YYYY.json)",
    )
    args = p.parse_args()

    folder_start = int(args.folders[0])
    folder_end = int(args.folders[1])

    if args.output is None:
        out_name = f"train_{folder_start:04d}_{folder_end:04d}.json"
        output_path = Path(__file__).parent.parent / "transcribe" / out_name
    else:
        output_path = Path(args.output)

    print(f"TRN: {args.trn}")
    print(f"범위: KsponSpeech_{folder_start:04d} ~ KsponSpeech_{folder_end:04d}")

    records = extract(args.trn, folder_start, folder_end)

    result = {
        "stats": {
            "total_files": len(records),
            "min_words_filter": 0,
            "source": Path(args.trn).name,
            "folders": f"KsponSpeech_{folder_start:04d} ~ KsponSpeech_{folder_end:04d}",
        },
        "data": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path} ({len(records)}개)")


if __name__ == "__main__":
    main()
