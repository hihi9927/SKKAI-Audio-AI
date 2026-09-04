#!/usr/bin/env python3
"""CoVoST2 → AST 평가용 manifest(JSONL) 생성기. **같은 화자의 클립을 이어붙인다.**

CoVoST2 는 Common Voice 기반이라 클립이 단문 한 문장(en_de test 평균 5.72초)이다.
그 길이로는 커밋 정책이 발화당 1회로 수렴해 세 축(always/dot/seg)이 구분되지 않는다.
그래서 `client_id` 가 같은 클립을 이어붙여 하나의 발화로 만든다.

    en_de test 실측: 15,531 클립 / 화자 9,472명 / 화자당 클립 1~3개
                     화자별 총길이 중앙값 7.7초 (95% 20.6초, 최대 142.5초)

**알고 써야 하는 편향** — 이어붙인 문장들은 서로 무관하다(Common Voice 문장은 임의 추출).
  - 무관한 문장 사이의 경계는 더 뚜렷하다 → SEG 커밋에 **유리하게** 편향된다.
  - 문맥을 쓰는 모드(`--gpt-translation`, `--google-context`)에는 엉뚱한 앞문장이
    주입되어 **불리하게** 편향된다. 문맥 모드 비교에는 이 manifest 를 쓰면 안 된다.
화자는 동일하므로 음향 조건은 일관된다(이어붙이기 인공물 중 화자 변화는 없다).

데이터:

    hf download fixie-ai/covost2 --repo-type dataset \
        --include "en_de/test-*.parquet" --local-dir ~/datasets/covost2

사용:

    # 화자별 이어붙이기 (기존 동작)
    python evaluation/ast/build_manifest_covost2.py \
        --covost-root ~/datasets/covost2 --config en_de --split test \
        --audio-cache ~/datasets/covost2_concat \
        --out evaluation/ast/manifests/covost2_en-de_spk.jsonl

    # 단클립 — 이어붙이지 않고 원본 클립 하나가 발화 하나
    python evaluation/ast/build_manifest_covost2.py \
        --config en_de --mode single \
        --id-list evaluation/ast/subsets/covost2_en_test_n3000.json \
        --out evaluation/ast/manifests/covost2_en-de_n3000.jsonl

`--mode single` 은 이어붙이기의 편향(무관한 문장 사이의 뚜렷한 경계)이 없는 대신
발화가 짧아(평균 5.7초) 커밋이 발화당 1~2회로 수렴한다. **단문 성능**을 재는 조건이다.
`--id-list` 는 `select_covost2_subset.py` 가 만든 발화 id 목록으로, 세 언어가 같은
파일을 써야 언어 간 비교가 매칭된다.

단클립 모드의 wav 캐시는 언어와 무관하다(영어 오디오는 세 config 이 동일). 그래서
캐시 경로에 config 을 넣지 않고 `{split}` 만 쓴다 — 세 언어 manifest 가 **같은 wav
파일을 가리키므로** 디스크도 3분의 1이고 오디오가 정말 동일하다는 것도 보장된다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLING_RATE = 16000


def csv_unescape(t: str) -> str:
    """`sentence` 필드의 **CSV 이스케이프를 되돌린다**. 정규화가 아니라 복원이다.

    업스트림 parquet(`fixie-ai/covost2`)의 `sentence` 는 CSV 필드를 파싱하지 않은 채로
    들어 있다 — 필드 전체가 따옴표로 감싸이고 내부 따옴표는 두 번씩 적힌 상태다:

        parquet: '"A ""chanson"", by contrast, is a folk or popular song."'
        복원후 : 'A "chanson", by contrast, is a folk or popular song.'

    en_de test 15,531건 실측: 감싸진 것 19.1%, 겹따옴표 포함 5.1%. 이걸 그대로 쓰면
    분절기가 따옴표를 지우거나 옮겨 `text_modified` 로 떨어진다 (n3000 라벨링 실측:
    위반 28건 중 26건이 이 원인).

    **따옴표를 없애는 것이 아니다.** 복원 후 큰따옴표를 가진 문장은 5.8% 남는데, 그건
    진짜 인용이다 — Qwen3-ASR 출력도 3.9%, FLEURS 도 7.9% 가 큰따옴표를 갖는다.
    지우면 오히려 도메인이 어긋난다.

    **`translation` 에는 쓰지 않는다.** 세 config 모두 겹따옴표가 0.00% 라 CSV
    이스케이프된 적이 없다 (감싸진 것도 en_zh-CN 의 1건뿐이고, 겹따옴표가 없으므로
    그건 진짜 인용이다). 거기 적용하면 멀쩡한 인용부호를 벗긴다.

    감싸기만 있고 내부 겹따옴표가 없는 경우는 원리상 "필드 전체가 CSV 로 감싸인 것"과
    "문장 전체가 인용문인 것"을 구분할 수 없다. 19.1% 라는 비율이 후자로는 설명되지
    않으므로(Common Voice 문장은 단문이다) CSV 로 본다.
    """
    t = (t or "").strip()
    if len(t) > 1 and t[0] == '"' and t[-1] == '"':
        return t[1:-1].replace('""', '"').strip()
    return t


def load_rows(root: Path, config: str, split: str) -> list[dict]:
    """parquet 샤드를 읽어 행 목록으로. 오디오 바이트는 그대로 들고 온다."""
    import pyarrow.parquet as pq

    files = sorted((root / config).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"parquet 이 없습니다: {root / config}/{split}-*.parquet\n"
            f'  hf download fixie-ai/covost2 --repo-type dataset '
            f'--include "{config}/{split}-*.parquet" --local-dir {root}'
        )
    rows: list[dict] = []
    for f in files:
        t = pq.read_table(f, columns=["client_id", "audio", "sentence", "translation", "id"])
        chunk = t.to_pylist()
        for r in chunk:
            r["sentence"] = csv_unescape(r["sentence"])
        rows.extend(chunk)
        del t
    return rows


def decode_16k(audio_bytes: bytes) -> np.ndarray:
    """mp3 바이트 → 16kHz mono float32."""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLING_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLING_RATE,
                                res_type="soxr_hq")
    return data


def trim_silence(audio: np.ndarray, top_db: float) -> np.ndarray:
    """앞뒤 묵음 제거. 이어붙일 때 이음매 묵음이 통제되지 않으면 VAD 가 그 지점에서
    자명하게 발동해 분절이 공짜가 된다."""
    import librosa

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size > 0 else audio


def load_id_list(path: Path) -> list[str]:
    """select_covost2_subset.py 산출물에서 발화 id 목록을 읽는다."""
    d = json.loads(path.read_text(encoding="utf-8"))
    ids = d["utt_ids"] if isinstance(d, dict) else d
    if not ids:
        raise ValueError(f"id 목록이 비었습니다: {path}")
    return list(ids)


def build_single(args) -> int:
    """이어붙이지 않고 클립 하나 = 발화 하나. `--id-list` 로 부분집합을 고른다."""
    root = Path(args.covost_root).expanduser().resolve()
    src_lang, tgt_lang = args.config.split("_", 1)
    tgt_lang = tgt_lang.split("-")[0]
    # 영어 오디오는 세 config 이 동일하므로 캐시 경로에 config 을 넣지 않는다.
    cache = (Path(args.audio_cache).expanduser().resolve()
             / f"{src_lang}_{args.split}")
    cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids: list[str] | None = None
    if args.id_list:
        ids = load_id_list(Path(args.id_list).expanduser().resolve())
        print(f"id 목록: {args.id_list} ({len(ids):,}개)")

    print(f"parquet 로드: {root / args.config}")
    rows = load_rows(root, args.config, args.split)
    print(f"  {len(rows)}행")

    by_id = {r["id"]: r for r in rows}
    if ids is not None:
        missing = set(ids) - set(by_id)
        if missing:
            print(f"  [경고] id 목록 중 {len(missing)}개가 parquet 에 없습니다 "
                  f"(예: {sorted(missing)[:3]})")
        order = [i for i in ids if i in by_id]
    else:
        order = [r["id"] for r in rows]

    kept, total_sec, skipped = 0, 0.0, 0
    raw_sec, trimmed_sec = 0.0, 0.0
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for uid in order:
            if args.limit and kept >= args.limit:
                break
            it = by_id[uid]
            if not (it["sentence"] or "").strip() or not (it["translation"] or "").strip():
                skipped += 1
                continue
            try:
                a = decode_16k(it["audio"]["bytes"])
            except Exception:
                skipped += 1
                continue
            raw_sec += len(a) / SAMPLING_RATE
            if args.trim_db > 0:
                a = trim_silence(a, args.trim_db)
            trimmed_sec += len(a) / SAMPLING_RATE
            duration = len(a) / SAMPLING_RATE
            if a.size == 0 or duration < args.min_duration or duration > args.max_duration:
                skipped += 1
                continue

            wav_path = cache / f"{uid}.wav"
            # 세 언어 manifest 가 같은 wav 를 가리킨다 — 이미 있으면 다시 쓰지 않는다.
            if not wav_path.exists():
                sf.write(wav_path, a, SAMPLING_RATE, subtype="PCM_16")

            out.write(json.dumps({
                "utt_id": uid,
                "wav": str(wav_path),
                "offset": 0.0,
                "duration": round(duration, 3),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_text": it["sentence"].strip(),
                "tgt_text": it["translation"].strip(),
                "speaker_id": it["client_id"][:16],
                "talk_id": it["client_id"][:12],
                "n_clips": 1,
            }, ensure_ascii=False) + "\n")
            kept += 1
            total_sec += duration
            if kept % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  ... {kept}개 / {total_sec / 3600:.2f}시간 ({el:.0f}초 경과)")

    print(f"\nmanifest: {out_path}")
    print(f"  발화 {kept}개 (클립 1개 = 발화 1개) / 오디오 {total_sec / 3600:.2f}시간 / 제외 {skipped}")
    if kept:
        print(f"  평균 길이 {total_sec / kept:.2f}초")
        if args.trim_db > 0 and raw_sec > 0:
            print(f"  트림(top_db={args.trim_db}) 전 {raw_sec/3600:.2f}시간 "
                  f"→ 후 {trimmed_sec/3600:.2f}시간 "
                  f"({(trimmed_sec-raw_sec)/raw_sec*100:+.1f}%)")
        print(f"  오디오 캐시: {cache}")
        for n_cli in (1, 16):
            print(f"  실시간 페이싱 예상 소요({n_cli}병렬): "
                  f"{(total_sec + kept * 1.0) / 60 / n_cli:.0f}분")
    return 0


def build(args) -> int:
    if args.mode == "single":
        return build_single(args)
    root = Path(args.covost_root).expanduser().resolve()
    cache = Path(args.audio_cache).expanduser().resolve() / f"{args.config}_{args.split}"
    cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_lang, tgt_lang = args.config.split("_", 1)
    tgt_lang = tgt_lang.split("-")[0]  # zh-CN → zh

    print(f"parquet 로드: {root / args.config}")
    rows = load_rows(root, args.config, args.split)
    print(f"  {len(rows)}행")

    # 화자별 그룹 (등장 순서 유지 — 같은 화자의 클립은 대체로 같은 세션이다)
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        groups.setdefault(r["client_id"], []).append(r)
    print(f"  화자 {len(groups)}명 / 화자당 클립 "
          f"평균 {len(rows) / len(groups):.2f}개")

    gap = np.zeros(int(SAMPLING_RATE * args.concat_gap_ms / 1000), dtype=np.float32)

    kept, total_sec, skipped = 0, 0.0, 0
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, (cid, items) in enumerate(groups.items()):
            if args.limit and kept >= args.limit:
                break
            if len(items) < args.min_clips:
                skipped += 1
                continue

            parts, sents, trans = [], [], []
            for it in items:
                if not (it["sentence"] or "").strip() or not (it["translation"] or "").strip():
                    continue
                try:
                    a = decode_16k(it["audio"]["bytes"])
                except Exception:
                    continue
                if args.trim_db > 0:
                    a = trim_silence(a, args.trim_db)
                if a.size == 0:
                    continue
                if parts:
                    parts.append(gap)
                parts.append(a)
                sents.append(it["sentence"].strip())
                trans.append(it["translation"].strip())

            if not parts:
                skipped += 1
                continue

            audio = np.concatenate(parts)
            duration = len(audio) / SAMPLING_RATE
            if duration < args.min_duration or duration > args.max_duration:
                skipped += 1
                continue

            utt_id = f"{args.config}_{cid[:12]}"
            wav_path = cache / f"{utt_id}.wav"
            sf.write(wav_path, audio, SAMPLING_RATE, subtype="PCM_16")

            out.write(json.dumps({
                "utt_id": utt_id,
                "wav": str(wav_path),
                "offset": 0.0,
                "duration": round(duration, 3),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_text": " ".join(sents),
                "tgt_text": " ".join(trans),
                "speaker_id": cid[:16],
                "talk_id": cid[:12],
                "n_clips": len(sents),
            }, ensure_ascii=False) + "\n")
            kept += 1
            total_sec += duration

            if kept % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  ... {kept}개 / {total_sec / 3600:.2f}시간 "
                      f"({el:.0f}초 경과, {kept / el:.1f}발화/초)")

    print(f"\nmanifest: {out_path}")
    print(f"  발화 {kept}개 (화자 1명 = 발화 1개) / 오디오 {total_sec / 3600:.2f}시간 / 제외 {skipped}")
    if kept:
        print(f"  평균 길이 {total_sec / kept:.2f}초")
        print(f"  오디오 캐시: {cache}")
        for n_cli in (1, 16):
            print(f"  실시간 페이싱 예상 소요({n_cli}병렬): "
                  f"{(total_sec + kept * 1.0) / 60 / n_cli:.0f}분")
    return 0


def main():
    p = argparse.ArgumentParser(description="CoVoST2(화자별 이어붙이기) → AST manifest")
    p.add_argument("--covost-root", default="~/datasets/covost2")
    p.add_argument("--config", default="en_de", help="언어쌍 (en_de, en_ja, en_zh-CN, de_en ...)")
    p.add_argument("--split", default="test", choices=["test", "validation", "train"])
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="concat", choices=["concat", "single"],
                   help="concat=화자별 이어붙이기(기본), single=원본 클립 하나가 발화 하나")
    p.add_argument("--id-list", default=None,
                   help="single 모드에서 쓸 발화 id 목록 JSON "
                        "(select_covost2_subset.py 산출물). 미지정 시 전체")
    p.add_argument("--audio-cache", default="~/datasets/covost2_concat",
                   help="wav 를 저장할 곳 (리포 밖). single 모드는 미지정 시 "
                        "~/datasets/covost2_single 로 자동 대체된다")
    p.add_argument("--limit", type=int, default=None, help="앞에서 N명(=N발화)만")
    p.add_argument("--min-clips", type=int, default=1,
                   help="이 개수 미만의 클립만 가진 화자는 제외 (1이면 전부 사용)")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=600.0)
    p.add_argument("--concat-gap-ms", type=int, default=300,
                   help="클립 사이에 넣을 무음 길이")
    p.add_argument("--trim-db", type=float, default=30.0,
                   help="이어붙이기 전 앞뒤 묵음 트림 기준(top_db). 0이면 트림 안 함")
    args = p.parse_args()
    # concat 전용 기본 경로를 single 모드가 물려받으면 두 종류 wav 가 한 폴더에 섞인다.
    if args.mode == "single" and args.audio_cache == "~/datasets/covost2_concat":
        args.audio_cache = "~/datasets/covost2_single"
    if args.mode == "single" and args.min_clips != 1:
        p.error("--min-clips 는 concat 모드 전용입니다")
    sys.exit(build(args))


if __name__ == "__main__":
    main()
