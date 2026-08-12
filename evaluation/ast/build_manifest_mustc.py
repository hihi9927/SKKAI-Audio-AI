#!/usr/bin/env python3
"""MuST-C → AST 평가용 manifest(JSONL) 생성기.

MuST-C 레이아웃:

    {root}/data/{split}/wav/ted_1096.wav          # talk 통짜 wav, 16kHz mono
    {root}/data/{split}/txt/{split}.yaml          # 세그먼트 인덱스
    {root}/data/{split}/txt/{split}.en            # 소스 전사 (라인 정렬)
    {root}/data/{split}/txt/{split}.de            # 참조 번역 (라인 정렬)

yaml 의 n번째 항목이 텍스트 파일의 n번째 줄과 1:1 대응한다.

사용:

    python evaluation/ast/build_manifest_mustc.py \
        --mustc-root ~/datasets/mustc/en-de --split tst-COMMON \
        --out evaluation/ast/manifests/mustc_en-de_tst-COMMON.jsonl

오디오는 자르지 않는다 — manifest 에 (wav, offset, duration) 만 담고 슬라이싱은
클라이언트가 로드할 때 한다. 4시간짜리 오디오를 한 벌 더 만들 이유가 없고, wav 가
아직 없어도 manifest 를 먼저 만들어 볼 수 있다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_yaml_index(yaml_path: Path) -> list[dict]:
    """세그먼트 인덱스를 읽는다.

    train 스플릿은 항목이 20만 개가 넘어 순수 파이썬 로더로는 느리다. CSafeLoader 를
    우선 쓰고, 그것도 없으면 줄 단위 정규식으로 떨어진다 (MuST-C 의 yaml 은 한 줄에
    flow mapping 하나인 고정 포맷이라 안전하다).
    """
    try:
        import yaml

        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=loader)
        if isinstance(data, list):
            return data
    except ImportError:
        pass

    entries: list[dict] = []
    kv_re = re.compile(r"(\w+):\s*([^,}]+)")
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("-"):
                continue
            item: dict = {}
            for key, raw in kv_re.findall(line):
                val = raw.strip().strip("'\"")
                if key in ("duration", "offset"):
                    try:
                        val = float(val)
                    except ValueError:
                        continue
                item[key] = val
            if item:
                entries.append(item)
    return entries


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build(args) -> int:
    root = Path(args.mustc_root).expanduser().resolve()
    split_dir = root / "data" / args.split
    txt_dir = split_dir / "txt"
    wav_dir = split_dir / "wav"

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang or root.name.split("-")[-1]

    yaml_path = txt_dir / f"{args.split}.yaml"
    src_path = txt_dir / f"{args.split}.{src_lang}"
    tgt_path = txt_dir / f"{args.split}.{tgt_lang}"

    missing = [p for p in (yaml_path, src_path, tgt_path) if not p.exists()]
    if missing:
        print("[ERROR] 다음 파일이 없습니다:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            f"\n--mustc-root 는 언어쌍 디렉토리({src_lang}-{tgt_lang})를 가리켜야 합니다.\n"
            f"기대 구조: {root}/data/{args.split}/txt/{args.split}.yaml",
            file=sys.stderr,
        )
        return 2

    index = load_yaml_index(yaml_path)
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)

    if not (len(index) == len(src_lines) == len(tgt_lines)):
        print(
            f"[ERROR] 라인 수 불일치 — yaml={len(index)} "
            f"{src_lang}={len(src_lines)} {tgt_lang}={len(tgt_lines)}. "
            "정렬이 깨진 배포본이므로 그대로 쓰면 참조가 어긋납니다.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = {"empty_text": 0, "duration": 0, "missing_wav": 0}
    per_talk_counter: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as out:
        for entry, src_text, tgt_text in zip(index, src_lines, tgt_lines):
            wav_name = str(entry.get("wav", "")).strip()
            duration = float(entry.get("duration", 0.0) or 0.0)
            offset = float(entry.get("offset", 0.0) or 0.0)

            talk_id = Path(wav_name).stem
            seg_idx = per_talk_counter.get(talk_id, 0)
            per_talk_counter[talk_id] = seg_idx + 1

            if not src_text.strip() or not tgt_text.strip():
                skipped["empty_text"] += 1
                continue
            if duration < args.min_duration or duration > args.max_duration:
                skipped["duration"] += 1
                continue

            wav_path = wav_dir / wav_name
            if args.verify_audio and not wav_path.exists():
                skipped["missing_wav"] += 1
                continue

            out.write(
                json.dumps(
                    {
                        "utt_id": f"{talk_id}_{seg_idx:04d}",
                        "wav": str(wav_path),
                        "offset": round(offset, 3),
                        "duration": round(duration, 3),
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "src_text": src_text.strip(),
                        "tgt_text": tgt_text.strip(),
                        "speaker_id": entry.get("speaker_id", ""),
                        "talk_id": talk_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            if args.limit and kept >= args.limit:
                break

    total_sec = 0.0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            total_sec += json.loads(line)["duration"]

    print(f"manifest: {out_path}")
    print(f"  세그먼트 {kept}개 / 오디오 {total_sec / 3600:.2f}시간")
    print(f"  제외 — 빈 텍스트 {skipped['empty_text']}, 길이 {skipped['duration']}, "
          f"wav 없음 {skipped['missing_wav']}")
    if kept:
        print(f"  평균 길이 {total_sec / kept:.2f}초")
        # 실시간 페이싱 기준 예상 소요 (세그먼트당 VAD 대기 ~1초 가정)
        est_min = (total_sec + kept * 1.0) / 60.0
        print(f"  실시간 페이싱 예상 소요: 약 {est_min:.0f}분")
    return 0


def main():
    parser = argparse.ArgumentParser(description="MuST-C → AST manifest(JSONL)")
    parser.add_argument("--mustc-root", required=True,
                        help="언어쌍 디렉토리 (예: ~/datasets/mustc/en-de)")
    parser.add_argument("--split", default="tst-COMMON",
                        help="tst-COMMON(표준 테스트셋) / tst-HE / dev / train")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default=None,
                        help="미지정 시 --mustc-root 이름에서 추론 (en-de → de)")
    parser.add_argument("--out", required=True, help="출력 manifest 경로(.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서 N개만. 반복 개발용 서브셋 생성")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="이보다 짧은 세그먼트 제외(초)")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="이보다 긴 세그먼트 제외(초)")
    parser.add_argument("--verify-audio", action="store_true",
                        help="wav 파일 존재를 확인하고 없으면 제외")
    sys.exit(build(parser.parse_args()))


if __name__ == "__main__":
    main()
