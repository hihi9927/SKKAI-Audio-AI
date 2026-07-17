#!/usr/bin/env python3
"""각 jsonl 줄의 text(문장) 끝에 <SEG>를 추가한다 (끝-SEG 누락 보정).

규칙:
- text를 rstrip 후 이미 '<SEG>'로 끝나면 그대로 둔다(멱등).
- 아니면 ' <SEG>'(앞에 공백 1개)를 붙인다.
  → 구두점이 있으면 '구두점 <SEG>'로 항상 공백이 들어가고, 없으면 '단어 <SEG>'.
- text 외 필드(audio 등)는 보존. 한글 유지를 위해 ensure_ascii=False.

사용:
  python utils/add_end_seg.py IN.jsonl OUT.jsonl
  python utils/add_end_seg.py --inplace FILE.jsonl   # 원본 .bak 백업 후 덮어쓰기
"""
import argparse
import json
import os
import shutil
import sys

SEG = "<SEG>"


def fix_text(text: str) -> str:
    t = text.rstrip()
    if t.endswith(SEG):
        return t
    return t + " " + SEG


def process(in_path: str, out_path: str, skip_audio_substr: str = "") -> tuple[int, int, int]:
    n = changed = skipped = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            obj = json.loads(line)
            # split_audio 등 지정 경로의 줄은 끝-SEG 추가 제외(원본 그대로 둠)
            if skip_audio_substr and skip_audio_substr in obj.get("audio", ""):
                skipped += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
                continue
            orig = obj.get("text", "")
            new = fix_text(orig)
            if new != orig:
                changed += 1
            obj["text"] = new
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n, changed, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path", nargs="?", default=None)
    ap.add_argument("--inplace", action="store_true", help="원본 .bak 백업 후 덮어쓰기")
    ap.add_argument("--skip-if-audio-contains", default="",
                    help="audio 경로에 이 문자열이 있으면 끝-SEG 제외(예: split_audio)")
    args = ap.parse_args()

    sub = args.skip_if_audio_contains

    if args.inplace:
        bak = args.in_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(args.in_path, bak)
        tmp = args.in_path + ".tmp"
        n, changed, skipped = process(args.in_path, tmp, sub)
        os.replace(tmp, args.in_path)
        print(f"{args.in_path}: {n}줄, {changed}줄 끝-SEG 추가, {skipped}줄 제외 (백업: {bak})")
    else:
        out = args.out_path or (os.path.splitext(args.in_path)[0] + "_segend.jsonl")
        n, changed, skipped = process(args.in_path, out, sub)
        print(f"{args.in_path} -> {out}: {n}줄, {changed}줄 끝-SEG 추가, {skipped}줄 제외")


if __name__ == "__main__":
    main()
