import argparse
import json
import os

PRESETS = {
    "Korean": {
        "input":     "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/evaluation/KsponSpeech/transcribe/eval_clean_1000.json",
        "audio_dir": "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/Qwen3-ASR/finetuning/data/KSponSpeech/audio",
        "output":    "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/Qwen3-ASR/finetuning/data/KSponSpeech/test.jsonl",
    },
    "English": {
        "input":     "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/evaluation/DailyTalk/transcribe/eval_dailytalk_1008.json",
        "audio_dir": "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/Qwen3-ASR/finetuning/data/DailyTalk/audio",
        "output":    "/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/Qwen3-ASR/finetuning/data/DailyTalk/test.jsonl",
    },
}

p = argparse.ArgumentParser()
p.add_argument("--language", "-l", required=True, choices=list(PRESETS.keys()))
p.add_argument("--input",     default=None)
p.add_argument("--audio_dir", default=None)
p.add_argument("--output",    default=None)
args = p.parse_args()

preset = PRESETS[args.language]
train_json  = args.input     or preset["input"]
audio_dir   = args.audio_dir or preset["audio_dir"]
output_jsonl = args.output   or preset["output"]

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
            "text": f"language {args.language}<asr_text>{entry['text']}"
        }
        out.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"완료: {len(entries)}개 항목 변환")
print(f"출력: {output_jsonl}")
