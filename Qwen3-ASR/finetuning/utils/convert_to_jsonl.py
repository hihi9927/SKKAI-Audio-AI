import json
import os



TRAIN_JSON = "/home/skkai/Desktop/STiTy/evaluation/KsponSpeech/results/train_gpt.json"
AUDIO_DIR = "/home/skkai/Desktop/STiTy/evaluation/KsponSpeech/wav_data"
OUTPUT_JSONL = "/home/skkai/Desktop/STiTy/Qwen3-ASR/finetuning/train.jsonl"
LANGUAGE = "Korean"

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

entries = data["data"]

with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
    for entry in entries:
        audio_path = os.path.join(AUDIO_DIR, entry["file"] + ".wav")
        line = {
            "audio": audio_path,
            "text": f"language {LANGUAGE}<asr_text>{entry['seg_text']}"
        }
        out.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"완료: {len(entries)}개 항목 변환")
print(f"출력: {OUTPUT_JSONL}")
