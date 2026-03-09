"""
seg 단위 번역 vs 전체 번역 COMET 비교 스크립트

- reference : text 전체를 Google Translate로 번역
- hypothesis: seg_text를 <seg>로 분할 → 각각 번역 → 공백 join
- COMET     : hypothesis 품질을 reference 기준으로 산출
"""

import json
import time
import argparse
from pathlib import Path
from deep_translator import GoogleTranslator
from comet import download_model, load_from_checkpoint


def translate(text: str, src: str = "ko", tgt: str = "en") -> str:
    return GoogleTranslator(source=src, target=tgt).translate(text)


def translate_seg(seg_text: str, src: str = "ko", tgt: str = "en") -> str:
    segments = seg_text.split("<seg>")
    translated = [translate(s.strip(), src, tgt) for s in segments if s.strip()]
    return " ".join(translated)


def main():
    parser = argparse.ArgumentParser(description="seg 번역 vs 전체 번역 COMET 평가")
    parser.add_argument("--input",  type=str, default=r"C:\Users\jduh1\Desktop\STiTy\core\meaning_segmentator\data\transcribe\eval_clean_100.json")
    parser.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로 (기본: 입력 파일 덮어쓰기)")
    parser.add_argument("--model",  type=str, default="Unbabel/wmt22-comet-da", help="COMET 모델명")
    parser.add_argument("--delay",  type=float, default=0.2, help="번역 요청 간 딜레이(초)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="이미 번역된 항목도 재처리")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output or args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # ── 1. 번역 ──────────────────────────────────────────────────────────────
    for i, entry in enumerate(data):
        has_full = "full_trans" in entry
        has_seg  = "seg_trans"  in entry

        if args.resume and has_full and has_seg:
            print(f"[{i+1}/{len(data)}] 건너뜀: {entry['file']}")
            continue

        if "seg_text" not in entry:
            print(f"[{i+1}/{len(data)}] seg_text 없음, 스킵: {entry['file']}")
            continue

        print(f"[{i+1}/{len(data)}] {entry['file']}")

        if not (args.resume and has_full):
            entry["full_trans"] = translate(entry["text"])
            print(f"  full : {entry['full_trans']}")
            if args.delay > 0:
                time.sleep(args.delay)

        if not (args.resume and has_seg):
            entry["seg_trans"] = translate_seg(entry["seg_text"])
            print(f"  seg  : {entry['seg_trans']}")
            if args.delay > 0:
                time.sleep(args.delay)

        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 2. COMET 점수 산출 ───────────────────────────────────────────────────
    print("\nCOMET 모델 로드 중...")
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    comet_data = []
    for entry in data:
        if "full_trans" not in entry or "seg_trans" not in entry:
            continue
        comet_data.append({
            "src": entry["text"],
            "mt":  entry["seg_trans"],
            "ref": entry["full_trans"],
        })

    if not comet_data:
        print("COMET 계산할 데이터가 없습니다.")
        return

    output = model.predict(comet_data, batch_size=8, gpus=0)
    scores = output.scores

    # 결과를 JSON에 저장
    idx = 0
    for entry in data:
        if "full_trans" not in entry or "seg_trans" not in entry:
            continue
        entry["comet_score"] = round(scores[idx], 4)
        idx += 1

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    avg = sum(scores) / len(scores)
    print(f"\n── COMET 결과 ──────────────────────────")
    print(f"  샘플 수  : {len(scores)}")
    print(f"  평균 점수: {avg:.4f}")
    print(f"  최소     : {min(scores):.4f}")
    print(f"  최대     : {max(scores):.4f}")
    print(f"  저장     : {output_path}")


if __name__ == "__main__":
    main()