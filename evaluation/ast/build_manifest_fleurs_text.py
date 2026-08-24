#!/usr/bin/env python3
"""FLEURS n-way **텍스트** manifest 생성기 — 오디오 없이 소스/참조만 뽑는다.

`build_manifest_fleurs.py` 는 오디오 디렉토리를 요구하므로 텍스트만 필요한 실험
(분절 프롬프트 평가 등)에는 쓸 수 없다. 이 스크립트는 FLEURS 가 n-way 병렬이라는
성질만 이용해 **소스 문장 id 교집합**을 잡고 타깃별 참조를 붙인다.

기본 동작은 `core/meaning_segmentator/BLEU_COMPARISON_PLAN.md` 의 설계를 그대로 따른다.

  - 소스 en_us, 타깃 de/ja/zh 의 **4언어 교집합**만 남긴다 (타깃별 n 이 같아야
    언어 간 비교가 문장 차이와 섞이지 않는다).
  - `--exclude-run` 이 가리키는 autoseg 런이 이미 쓴 문장을 뺀다 (프롬프트 최적화 데이터).
  - `--splits train dev` 로 FLEURS **test 스플릿을 제외**한다 — 오디오를 가진 유일한
    분할이라 파인튜닝 모델 end-to-end 평가용으로 봉인한다.
  - `data.split_data` 와 같은 층화 규칙으로 전체를 정렬한 뒤 **앞 n 개**를 취한다.
    prefix 로 뽑아야 나중에 표본을 늘릴 때 앞 n 개가 그대로 유지돼 캐시가 살아난다.
    정렬 전체는 `order.json` 으로 남긴다.

사용:

    python evaluation/ast/build_manifest_fleurs_text.py \
        --fleurs-root ~/datasets/fleurs --src en_us --tgts de_de ja_jp cmn_hans_cn \
        --splits train dev --exclude-run core/meaning_segmentator/runs/en-multi/run06 \
        --n 500 --out-dir evaluation/ast/manifests --tag clean500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from core.meaning_segmentator.autoseg import data as autoseg_data  # noqa: E402

LANG_CODE = {
    "en_us": "en", "de_de": "de", "ko_kr": "ko", "ja_jp": "ja",
    "cmn_hans_cn": "zh", "es_419": "es", "fr_fr": "fr", "ru_ru": "ru",
}


def read_split(root: Path, lang: str, split: str) -> dict[str, str]:
    """{문장 id: raw_transcription}. 같은 문장의 여러 화자 행은 첫 행만 취한다."""
    path = root / "data" / lang / f"{split}.tsv"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    # **QUOTE_NONE 이 필수다.** FLEURS 전사에는 따옴표가 그대로 들어 있어 기본 파싱은
    # 그것을 인용 시작으로 읽고 뒤따르는 행들을 한 행으로 합쳐 버린다 (en_us 실측:
    # 3,643행이 3,559행으로 줄고, 한 "문장"이 2,916어절짜리 덩어리가 된다).
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 6:
                text = row[2].strip()
                if text and "\n" not in text:
                    out.setdefault(row[0], text)
    return out


def load_lang(root: Path, lang: str, splits: list[str]) -> dict[str, tuple[str, str]]:
    """{문장 id: (텍스트, 원 스플릿)}. 앞 스플릿이 이긴다."""
    out: dict[str, tuple[str, str]] = {}
    for split in splits:
        for sid, text in read_split(root, lang, split).items():
            if text:
                out.setdefault(sid, (text, split))
    return out


def used_sentence_ids(run_dir: Path) -> set[str]:
    """autoseg 런이 소비한 FLEURS 문장 id. `utt_id` 가운데 필드가 문장 id 다."""
    ids: set[str] = set()
    for name in ("train", "dev", "test"):
        p = run_dir / "data" / f"{name}.json"
        if p.exists():
            for row in json.loads(p.read_text(encoding="utf-8")):
                parts = str(row["id"]).split("_")
                if len(parts) >= 3:
                    ids.add(parts[2])
    return ids


def stratified_order(items: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    """층화 라운드로빈 정렬. `autoseg.data` 의 것을 그대로 쓴다 — 예전에는 같은 규칙이
    양쪽에 복사돼 있어 한쪽만 고치면 조용히 갈라졌다."""
    return autoseg_data.stratified_order(items, seed, text_of=lambda x: x[1])

def main() -> int:
    p = argparse.ArgumentParser(description="FLEURS n-way 텍스트 manifest (오디오 불필요)")
    p.add_argument("--fleurs-root", default="~/datasets/fleurs")
    p.add_argument("--src", default="en_us")
    p.add_argument("--tgts", nargs="+", default=["de_de", "ja_jp", "cmn_hans_cn"])
    p.add_argument("--splits", nargs="+", default=["train", "dev"],
                   help="쓸 FLEURS 스플릿. 기본은 test 제외 (모델 평가용 봉인)")
    p.add_argument("--exclude-run", default=None,
                   help="이 autoseg 런이 쓴 문장을 제외 (프롬프트 최적화 데이터)")
    p.add_argument("--min-chars", type=int, default=25)
    p.add_argument("--max-words", type=int, default=60,
                   help="이보다 긴 행은 문장이 아니라 파싱 잔해일 가능성이 높다")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--order-from", default=None,
                   help="기존 *_order.json 의 정렬을 그대로 재사용한다. 층화를 다시 계산하지 "
                        "않으므로 소스 언어를 바꿔도 **같은 문장 집합**이 나온다 — 방향이 "
                        "다른 트랙끼리 내용을 맞추려면 이것을 쓸 것")
    p.add_argument("--skip", type=int, default=0,
                   help="정렬에서 앞 N 개를 건너뛰고 그 다음 --n 개를 취한다. 이미 소진한 "
                        "구간과 겹치지 않는 새 셋을 떼는 용도")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--out-dir", default="evaluation/ast/manifests")
    p.add_argument("--tag", default="clean500")
    args = p.parse_args()

    root = Path(args.fleurs_root).expanduser().resolve()
    langs = {l: load_lang(root, l, args.splits) for l in [args.src, *args.tgts]}
    for l, d in langs.items():
        if not d:
            print(f"TSV 없음/빈 값: {l} ({args.splits})", file=sys.stderr)
            return 1
        print(f"  {l:14s} 고유 문장 {len(d):5d}")

    inter = set.intersection(*[set(d) for d in langs.values()])
    print(f"{len(langs)}언어 교집합: {len(inter)}")

    excluded = used_sentence_ids(Path(args.exclude_run)) if args.exclude_run else set()
    if excluded:
        print(f"제외(기존 런 사용분): {len(excluded)} → 교집합에서 {len(inter & excluded)}건 빠짐")

    src = langs[args.src]
    if args.order_from:
        # 정렬은 **다른 언어 기준으로 이미 정해져 있다.** 여기서 다시 층화하면 소스 언어의
        # 길이 분포를 타서 트랙마다 다른 문장이 뽑히고, 언어 간 비교가 문장 차이와 섞인다.
        saved = json.loads(Path(args.order_from).read_text(encoding="utf-8"))
        ref_ids = [str(o["id"]).split("_")[-1] for o in saved["order"]]
        missing = [i for i in ref_ids if i not in src or i not in inter]
        if missing:
            print(f"정렬에 있으나 {args.src}/교집합에 없는 문장 {len(missing)}건 — 건너뜀")
        order = [(sid, src[sid][0]) for sid in ref_ids
                 if sid in src and sid in inter and sid not in excluded]
        print(f"정렬 재사용: {args.order_from} ({len(saved['order'])}) → 사용 가능 {len(order)}")
    else:
        pool = [(sid, src[sid][0]) for sid in inter
                if sid not in excluded and len(src[sid][0]) >= args.min_chars
                and len(src[sid][0].split()) <= args.max_words]
        print(f"모집단: {len(pool)} (min_chars={args.min_chars})")
        order = stratified_order(pool, args.seed)

    chosen = order[args.skip: args.skip + args.n]
    if len(chosen) < args.n:
        print(f"문장 부족: skip={args.skip} 이후 {len(chosen)} < {args.n}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_code = LANG_CODE[args.src]
    for tgt in args.tgts:
        tcode = LANG_CODE[tgt]
        path = out_dir / f"fleurs_nway_{src_code}-{tcode}_{args.tag}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for sid, text in chosen:
                f.write(json.dumps({
                    "utt_id": f"{args.src}_{sid}",
                    "src_lang": src_code, "tgt_lang": tcode,
                    "src_text": text, "tgt_text": langs[tgt][sid][0],
                    "talk_id": sid, "fleurs_split": src[sid][1],
                }, ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(chosen)}행)")

    (out_dir / f"fleurs_nway_{src_code}_{args.tag}_order.json").write_text(json.dumps({
        "seed": args.seed, "splits": args.splits, "min_chars": args.min_chars,
        "n_taken": args.n, "n_pool": len(order), "skip": args.skip,
        "order_from": args.order_from, "exclude_run": args.exclude_run,
        "order": [{"id": f"{args.src}_{sid}", "text": t} for sid, t in order],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote order.json (전체 {len(order)} — 확장 시 앞 {args.n} 은 그대로)")

    # autoseg 런 디렉토리에 그대로 깔 수 있는 형식도 같이 낸다.
    (out_dir / f"fleurs_nway_{src_code}_{args.tag}_split.json").write_text(json.dumps(
        [{"id": f"{args.src}_{sid}", "text": t} for sid, t in chosen],
        ensure_ascii=False, indent=2), encoding="utf-8")

    measured = autoseg_data.measure_profile([t for _, t in chosen])
    spaced = measured["uses_spaces_between_words"]
    # **`T` 와 `min_gap` 의 단위가 여기서 갈린다** (`pipeline.unit_count`): 띄어쓰기가 있으면
    # 어절, 없으면 문자다. 같은 T 를 다른 언어에 그대로 쓰면 조각 길이가 전혀 달라진다.
    units = sorted((len(t.split()) if spaced else len(t.replace(" ", "")))
                   for _, t in chosen)
    mid = units[len(units) // 2]
    print(f"  측정 프로파일: 공백비율 {measured['space_ratio']} "
          f"(spaced={spaced}), 문말 부호 {measured['trailing_punctuation']}")
    print(f"  소스 단위({'어절' if spaced else '문자'}): 중앙 {mid}, "
          f"평균 {sum(units) / len(units):.1f}, p10 {units[len(units) // 10]}, "
          f"p90 {units[len(units) * 9 // 10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
