#!/usr/bin/env python3
"""FLEURS → AST 평가용 manifest(JSONL) 생성기.

FLEURS 는 FLoRes-101 문장을 103개 언어로 낭독한 **n-way 병렬** 코퍼스다. 문장 id 가
언어 간에 공유되므로 `소스 언어의 오디오` + `타깃 언어의 전사` 를 붙이면 그대로
음성번역 평가쌍이 된다. MuST-C 와 달리 소스가 영어로 고정되지 않아 ko→en 처럼
제품이 실제로 서비스하는 방향도 평가할 수 있다.

레이아웃 (hf download 후):

    {root}/data/en_us/test.tsv           # 소스: id, filename, raw_transcription, ...
    {root}/data/en_us/audio/test.tar.gz  # 오디오 (풀면 test/*.wav)
    {root}/data/de_de/test.tsv           # 타깃: 참조 번역으로 쓴다

TSV 는 헤더가 없고 7열이다:
    0 id  1 filename  2 raw_transcription  3 transcription  4 phonemes
    5 num_samples  6 gender

참조로는 `raw_transcription`(2열)을 쓴다 — 구두점과 대소문자가 살아 있는 원문이라
BLEU 대상으로 적절하다. `transcription`(3열)은 소문자·구두점 제거본이다.

사용:

    python evaluation/ast/build_manifest_fleurs.py \
        --fleurs-root ~/datasets/fleurs --src en_us --tgt de_de \
        --out evaluation/ast/manifests/fleurs_en-de_test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from pathlib import Path

SAMPLING_RATE = 16000

# FLEURS 언어 코드 → 서버/번역기가 쓰는 2글자 코드
LANG_CODE = {
    "en_us": "en", "de_de": "de", "ko_kr": "ko", "ja_jp": "ja",
    "cmn_hans_cn": "zh", "yue_hant_hk": "zh", "es_419": "es",
    "fr_fr": "fr", "it_it": "it", "nl_nl": "nl", "pt_br": "pt",
    "ru_ru": "ru", "pl_pl": "pl", "tr_tr": "tr", "ar_eg": "ar",
    "hi_in": "hi", "vi_vn": "vi", "th_th": "th", "id_id": "id",
}


def read_tsv(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [r for r in csv.reader(f, delimiter="\t") if len(r) >= 6]


def ensure_audio_dir(root: Path, lang: str, split: str) -> Path:
    """오디오 디렉토리를 돌려준다. tar.gz 만 있으면 한 번 풀어둔다."""
    audio_dir = root / "data" / lang / "audio" / split
    if audio_dir.is_dir() and any(audio_dir.iterdir()):
        return audio_dir

    tar_path = root / "data" / lang / "audio" / f"{split}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(
            f"오디오가 없습니다: {audio_dir} 도 {tar_path} 도 없음.\n"
            f"  hf download google/fleurs --repo-type dataset "
            f'--include "data/{lang}/audio/{split}.tar.gz" --local-dir {root}'
        )
    print(f"오디오 압축 해제: {tar_path} → {tar_path.parent}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(tar_path.parent)
    if not audio_dir.is_dir():
        # 배포본에 따라 최상위 디렉토리 이름이 다를 수 있다
        candidates = [p for p in tar_path.parent.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            audio_dir = candidates[0]
        else:
            raise FileNotFoundError(f"압축 해제 후 오디오 디렉토리를 찾지 못했습니다: {candidates}")
    return audio_dir


def build(args) -> int:
    root = Path(args.fleurs_root).expanduser().resolve()
    src_tsv = root / "data" / args.src / f"{args.split}.tsv"
    tgt_tsv = root / "data" / args.tgt / f"{args.split}.tsv"

    missing = [p for p in (src_tsv, tgt_tsv) if not p.exists()]
    if missing:
        print("[ERROR] 다음 파일이 없습니다:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            f'\n  hf download google/fleurs --repo-type dataset \\\n'
            f'    --include "data/{args.src}/*" "data/{args.tgt}/test.tsv" \\\n'
            f"    --local-dir {root}",
            file=sys.stderr,
        )
        return 2

    audio_dir = ensure_audio_dir(root, args.src, args.split)

    # 타깃: 문장 id → 참조 문장. 같은 id 에 여러 낭독이 있어도 문장은 같으므로 첫 줄만.
    refs: dict[str, str] = {}
    for row in read_tsv(tgt_tsv):
        refs.setdefault(row[0], row[2].strip())

    src_lang = LANG_CODE.get(args.src, args.src.split("_")[0])
    tgt_lang = LANG_CODE.get(args.tgt, args.tgt.split("_")[0])

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, total_sec = 0, 0.0
    seen_ids: set[str] = set()
    skipped = {"no_ref": 0, "no_audio": 0, "duration": 0, "dup_sentence": 0}

    with open(out_path, "w", encoding="utf-8") as out:
        for row in read_tsv(src_tsv):
            sent_id, filename, raw_src = row[0], row[1], row[2].strip()
            try:
                duration = int(row[5]) / SAMPLING_RATE
            except (ValueError, IndexError):
                skipped["duration"] += 1
                continue

            ref = refs.get(sent_id)
            if not ref or not raw_src:
                skipped["no_ref"] += 1
                continue
            # 한 문장에 낭독이 여럿 있다. 기본은 문장당 하나만 써서 BLEU 가 특정 문장에
            # 중복 가중되지 않게 한다.
            if not args.all_recordings:
                if sent_id in seen_ids:
                    skipped["dup_sentence"] += 1
                    continue
                seen_ids.add(sent_id)
            if duration < args.min_duration or duration > args.max_duration:
                skipped["duration"] += 1
                continue

            wav_path = audio_dir / filename
            if args.verify_audio and not wav_path.exists():
                skipped["no_audio"] += 1
                continue

            out.write(json.dumps({
                "utt_id": f"{args.src}_{sent_id}_{Path(filename).stem[:8]}",
                "wav": str(wav_path),
                "offset": 0.0,
                "duration": round(duration, 3),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_text": raw_src,
                "tgt_text": ref,
                "speaker_id": row[6] if len(row) > 6 else "",
                "talk_id": sent_id,
            }, ensure_ascii=False) + "\n")
            kept += 1
            total_sec += duration
            if args.limit and kept >= args.limit:
                break

    print(f"manifest: {out_path}")
    print(f"  {args.src} 음성 → {args.tgt} 참조 / 세그먼트 {kept}개 / "
          f"오디오 {total_sec / 3600:.2f}시간")
    print(f"  제외 — 참조 없음 {skipped['no_ref']}, 중복 문장 {skipped['dup_sentence']}, "
          f"길이 {skipped['duration']}, 오디오 없음 {skipped['no_audio']}")
    if kept:
        print(f"  평균 길이 {total_sec / kept:.2f}초")
        print(f"  실시간 페이싱 예상 소요: 약 {(total_sec + kept * 1.0) / 60:.0f}분")
        print(f"\n  실행: python evaluation/ast/test_ast.py --manifest {out_path} \\\n"
              f"          --dataset FLEURS --src-lang {src_lang} --target-lang {tgt_lang} \\\n"
              f"          --model <모델별칭> --scope sample --tag run_01")
    return 0


def main():
    p = argparse.ArgumentParser(description="FLEURS → AST manifest(JSONL)")
    p.add_argument("--fleurs-root", default="~/datasets/fleurs",
                   help="hf download 로 받은 디렉토리")
    p.add_argument("--src", default="en_us", help="소스 언어 (오디오를 쓸 쪽)")
    p.add_argument("--tgt", default="de_de", help="타깃 언어 (참조 번역으로 쓸 쪽)")
    p.add_argument("--split", default="test", choices=["test", "dev", "train"])
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-duration", type=float, default=0.5)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--all-recordings", action="store_true",
                   help="한 문장의 낭독을 전부 사용 (기본은 문장당 1개)")
    p.add_argument("--verify-audio", action="store_true")
    sys.exit(build(p.parse_args()))


if __name__ == "__main__":
    main()
