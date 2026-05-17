"""
split_train2.json → 파인튜닝용 JSONL 변환 스크립트.

Usage:
    # KSponSpeech
    python convert_split_to_jsonl.py \
        --split_json  ../../evaluation/KsponSpeech/transcribe/split_train2.json \
        --audio_dir   ../data/KSponSpeech/split_audio \
        --language    Korean \
        --output      ../data/KSponSpeech/split_train.jsonl

    # DailyTalk
    python convert_split_to_jsonl.py \
        --split_json  ../../evaluation/DailyTalk/transcribe/split_train2.json \
        --audio_dir   ../data/DailyTalk/split_audio \
        --language    English \
        --output      ../data/DailyTalk/split_train.jsonl
"""
import argparse
import json
import os


def convert(split_json: str, audio_dir: str, language: str, output: str) -> None:
    audio_dir = os.path.abspath(audio_dir)
    output    = os.path.abspath(output)

    with open(split_json, encoding="utf-8") as f:
        raw = json.load(f)

    # KSponSpeech: {"data": [...]}  /  DailyTalk: {"conv_id": {"data": [...]}, ...}
    if "data" in raw and isinstance(raw["data"], list):
        items = raw["data"]
    else:
        items = [item for conv in raw.values() for item in conv["data"]]

    prefix = f"language {language}<asr_text>"
    written, skipped = 0, 0
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "w", encoding="utf-8") as out:
        for item in items:
            wav = os.path.join(audio_dir, item["file"] + ".wav")
            if not os.path.exists(wav):
                skipped += 1
                continue

            seg_text = item["seg_text"].strip()
            # 빈 텍스트(무음 구간이 잘린 경우)는 건너뜀
            if not seg_text:
                skipped += 1
                continue

            record = {
                "audio": wav,
                "text": prefix + seg_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"완료: {written}개 저장 → {output}  (건너뜀: {skipped}개)")


def main():
    p = argparse.ArgumentParser(description="split_train2.json → 파인튜닝 JSONL 변환")
    p.add_argument("--split_json", required=True, help="split_train2.json 경로")
    p.add_argument("--audio_dir",  required=True, help="split WAV 파일이 있는 디렉토리")
    p.add_argument("--language",   required=True, help="언어 레이블 (Korean / English 등)")
    p.add_argument("--output",     required=True, help="출력 JSONL 경로")
    args = p.parse_args()
    convert(args.split_json, args.audio_dir, args.language, args.output)


if __name__ == "__main__":
    main()
