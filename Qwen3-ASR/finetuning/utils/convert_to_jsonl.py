import argparse
import json
import os

# 이 파일: Qwen3-ASR/finetuning/utils/ → 저장소 루트는 세 단계 위.
# 종전에는 개발자 한 명의 홈 디렉터리 절대경로가 박혀 있어 다른 머신에서는 인자를
# 전부 주지 않으면 무조건 실패했다.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

PRESETS = {
    "Korean": {
        "input":     "evaluation/KsponSpeech/results/eval_clean_1000_seg.json",
        "audio_dir": "Qwen3-ASR/finetuning/data/KSponSpeech/audio",
        "output":    "Qwen3-ASR/finetuning/data/KSponSpeech/test.jsonl",
    },
    "English": {
        "input":     "evaluation/DailyTalk/results/eval_dailytalk_1008_seg_en.json",
        "audio_dir": "Qwen3-ASR/finetuning/data/DailyTalk/audio",
        "output":    "Qwen3-ASR/finetuning/data/DailyTalk/test.jsonl",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", "-l", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--input",     default=None)
    p.add_argument("--audio_dir", default=None)
    p.add_argument("--output",    default=None)
    args = p.parse_args()

    preset = PRESETS[args.language]
    train_json   = args.input     or os.path.join(REPO_ROOT, preset["input"])
    audio_dir    = args.audio_dir or os.path.join(REPO_ROOT, preset["audio_dir"])
    output_jsonl = args.output    or os.path.join(REPO_ROOT, preset["output"])

    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 포맷 감지: 최상위에 "data" 키가 있으면 KsponSpeech, 없으면 DailyTalk
    if "data" in data:
        entries = data["data"]
    else:
        entries = [entry for conv in data.values() for entry in conv["data"]]

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for entry in entries:
            audio_path = os.path.join(audio_dir, entry["file"] + ".wav")
            line = {
                "audio": audio_path,
                "text": f"language {args.language}<asr_text>{entry['seg_text']}"
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"완료: {len(entries)}개 항목 변환")
    print(f"출력: {output_jsonl}")


if __name__ == "__main__":
    main()
