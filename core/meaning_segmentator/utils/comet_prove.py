"""분절이 번역을 흔들었는지 본다 — **참조 없는 자기일관성 점검**.

`autoseg/scoring/comet_eval.py` 와 헷갈리지 말 것. 이름이 같아 혼동돼서 갈랐다.

|  | 이 스크립트 | `autoseg/scoring/comet_eval.py` |
|---|---|---|
| reference | **같은 번역기의 무분절 번역** (`*_full_trans`) | 골드 참조 (FLEURS/CoVoST2 매니페스트) |
| 재는 것 | 잘라 번역한 게 통째로 번역한 것과 같은가 | 정답 대비 번역 품질 |
| 번역 | 필드가 없으면 직접 번역한다 (deep_translator) | 안 한다 (`bleu_eval` 산출 재사용) |

참조가 기계 번역이므로 점수가 높아도 "번역이 정확하다"가 아니라 "분절이 결과를 안
흔들었다"는 뜻이다. 무분절 번역이 틀렸으면 그 틀림까지 정답으로 친다 — 논문 수치로
쓰지 말고, 라벨링 데이터 품질 점검에만 쓸 것.

- reference : --field 접두어에 맞는 full 번역 필드 (gdt_full_trans / gpt_full_trans)
- hypothesis: --field 로 지정한 번역 필드 (기본: gdt_seg_trans)
- COMET     : hypothesis 품질을 reference 기준으로 산출

점수 필드 및 통계 키는 --field 값에서 자동 결정:
  gdt_seg_trans       → ref: gdt_full_trans,      score: gdt_comet_score,       stats: gdt_comet_*
  gpt_seg_trans       → ref: gpt_full_trans,      score: gpt_comet_score,       stats: gpt_comet_*
  gpt_seg_nc_trans    → ref: gpt_full_trans,      score: gpt_nc_comet_score,    stats: gpt_nc_comet_*
  finetuned_seg_trans → ref: finetuned_full_trans, score: finetuned_comet_score, stats: finetuned_comet_*
                    (finetuned_text를 번역 → finetuned_seg_trans, text를 번역 → finetuned_full_trans)
"""

import json
import time
import argparse
from pathlib import Path
from deep_translator import GoogleTranslator
from comet import download_model, load_from_checkpoint


def translate(text: str, src: str = "ko", tgt: str = "en") -> str | None:
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception as e:
        print(f"  [번역 실패] '{text}' → {e}")
        return None


def translate_seg(seg_text: str, src: str = "ko", tgt: str = "en") -> str:
    segments = seg_text.split("<SEG>")
    translated = [translate(s.strip(), src, tgt) for s in segments if s.strip()]
    translated = [t for t in translated if t is not None]
    return " ".join(translated)


# 기본 split("_")[0] 로 처리할 수 없는 필드의 prefix / ref 매핑
_FIELD_PREFIX_MAP: dict[str, str] = {
    "gpt_seg_nc_trans": "gpt_nc",
}
_PREFIX_REF_MAP: dict[str, str] = {
    "gpt_nc": "gpt_full_trans",
}


def field_prefix(field: str) -> str:
    """gdt_seg_trans → gdt,  gpt_seg_nc_trans → gpt_nc"""
    return _FIELD_PREFIX_MAP.get(field, field.split("_")[0])


def score_field_name(field: str) -> str:
    """gdt_seg_trans → gdt_comet_score,  gpt_seg_nc_trans → gpt_nc_comet_score"""
    return f"{field_prefix(field)}_comet_score"


def ref_field_name(field: str) -> str:
    """gdt_seg_trans → gdt_full_trans,  gpt_seg_nc_trans → gpt_full_trans"""
    pfx = field_prefix(field)
    return _PREFIX_REF_MAP.get(pfx, f"{pfx}_full_trans")


def main():
    parser = argparse.ArgumentParser(description="seg 번역 vs 전체 번역 COMET 평가")
    _kspon_dir = Path(__file__).parents[3] / "evaluation" / "KsponSpeech"
    results_dir = _kspon_dir / "results"
    finetuned_results_dir = _kspon_dir / "finetuned_results"
    parser.add_argument("--input",     type=str, required=True,
                        help="결과 디렉토리 내 파일명 (예: eval_clean_seg.json)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="입력 파일 디렉토리")
    parser.add_argument("--field",     type=str, default="gdt_seg_trans",
                        help="COMET hypothesis로 사용할 번역 필드 (기본: gdt_seg_trans)")
    parser.add_argument("--model",     type=str, default="Unbabel/wmt22-comet-da",
                        help="COMET 모델명")
    parser.add_argument("--delay",     type=float, default=0.2,
                        help="번역 요청 간 딜레이(초)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="이미 번역된 항목도 재처리")
    parser.add_argument("--gpus",     type=int, default=0,
                        help="COMET 추론에 사용할 GPU 수 (기본: 0, CPU 사용)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = str(finetuned_results_dir if args.field == "finetuned_seg_trans" else results_dir)

    hyp_field   = args.field                  # e.g. gdt_seg_trans
    ref_field   = ref_field_name(hyp_field)   # e.g. gdt_full_trans
    score_field = score_field_name(hyp_field) # e.g. gdt_comet_score
    stat_pfx    = field_prefix(hyp_field)     # e.g. gdt

    input_path  = Path(args.input_dir) / args.input
    output_path = input_path
    raw  = json.loads(input_path.read_text(encoding="utf-8"))

    # 구조 감지: KsponSpeech {"data": [...]} vs DailyTalk {"0": {"data": [...]}, ...}
    is_dailytalk = "data" not in raw
    if is_dailytalk:
        group_keys = list(raw.keys())
        data = []
        entry_group = {}
        for gk in group_keys:
            if isinstance(raw[gk], dict) and "data" in raw[gk]:
                for e in raw[gk]["data"]:
                    data.append(e)
                    entry_group[e["file"]] = gk
    else:
        data = raw["data"]

    def save_raw(stats=None):
        if is_dailytalk:
            grouped = {gk: {"data": []} for gk in group_keys}
            for e in data:
                grouped[entry_group[e["file"]]]["data"].append(e)
            if stats:
                grouped = {"stats": stats} | grouped
            output_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            raw_out = {"stats": raw.get("stats", {})} | {k: v for k, v in raw.items() if k != "stats"}
            output_path.write_text(json.dumps(raw_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 1. Google Translate 번역 ─────────────────────────────────────────────
    if hyp_field == "gdt_seg_trans":
        for i, entry in enumerate(data):
            has_full = bool(entry.get("gdt_full_trans"))
            has_seg  = bool(entry.get("gdt_seg_trans"))

            if args.resume and has_full and has_seg:
                print(f"[{i+1}/{len(data)}] 건너뜀: {entry['file']}")
                continue

            if not entry.get("seg_text"):
                print(f"[{i+1}/{len(data)}] seg_text 없음, 스킵: {entry['file']}")
                continue

            print(f"[{i+1}/{len(data)}] {entry['file']}")

            if not (args.resume and has_full):
                entry["gdt_full_trans"] = translate(entry["text"])
                print(f"  gdt_full  : {entry['gdt_full_trans']}")
                if args.delay > 0:
                    time.sleep(args.delay)

            if not (args.resume and has_seg):
                entry["gdt_seg_trans"] = translate_seg(entry["seg_text"])
                print(f"  gdt_seg   : {entry['gdt_seg_trans']}")
                if args.delay > 0:
                    time.sleep(args.delay)

            save_raw()

    elif hyp_field == "finetuned_seg_trans":
        for i, entry in enumerate(data):
            has_full      = bool(entry.get("finetuned_full_trans"))
            has_finetuned = bool(entry.get("finetuned_seg_trans"))

            if args.resume and has_full and has_finetuned:
                print(f"[{i+1}/{len(data)}] 건너뜀: {entry['file']}")
                continue

            if not entry.get("finetuned_text"):
                print(f"[{i+1}/{len(data)}] finetuned_text 없음, 스킵: {entry['file']}")
                continue

            print(f"[{i+1}/{len(data)}] {entry['file']}")

            if not (args.resume and has_full):
                # gdt_full_trans 재사용, 없으면 새로 번역
                entry["finetuned_full_trans"] = (
                    entry.get("gdt_full_trans") or translate(entry["text"])
                )
                print(f"  full_trans: {entry['finetuned_full_trans']}")
                if args.delay > 0 and not entry.get("gdt_full_trans"):
                    time.sleep(args.delay)

            if not (args.resume and has_finetuned):
                entry["finetuned_seg_trans"] = translate_seg(entry["finetuned_text"])
                print(f"  finetuned : {entry['finetuned_seg_trans']}")
                if args.delay > 0:
                    time.sleep(args.delay)

            save_raw()

    # ── 2. COMET 점수 산출 ───────────────────────────────────────────────────
    print(f"\nCOMET 모델 로드 중... (hyp={hyp_field}, ref={ref_field}, score={score_field})")
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    require_seg = hyp_field not in ("finetuned_seg_trans",)
    comet_data    = []
    comet_entries = []
    skipped = 0
    for entry in data:
        if not entry.get(ref_field) or not entry.get(hyp_field):
            continue
        if require_seg and "<SEG>" not in entry.get("seg_text", ""):
            skipped += 1
            continue
        comet_data.append({
            "src": entry["text"],
            "mt":  entry[hyp_field],
            "ref": entry[ref_field],
        })
        comet_entries.append(entry)
    if require_seg:
        print(f"  분절 없는 항목 제외: {skipped}개")

    if not comet_data:
        print("COMET 계산할 데이터가 없습니다.")
        return

    output = model.predict(comet_data, batch_size=8, gpus=args.gpus)
    scores = output.scores

    for entry, score in zip(comet_entries, scores):
        entry[score_field] = round(score, 4)

    avg = sum(scores) / len(scores)

    # stats: 기존 stats 유지하면서 해당 field 통계만 덮어씀
    all_scores = [e[score_field] for e in data if score_field in e]
    stats = raw.get("stats", {"total_files": len(data)}) if not is_dailytalk else {"total_files": len(data)}
    stats["total_files"] = len(data)
    stats[f"{stat_pfx}_comet_evaluated"] = len(all_scores)
    stats[f"{stat_pfx}_comet_max"]       = round(max(all_scores), 4)
    stats[f"{stat_pfx}_comet_min"]       = round(min(all_scores), 4)
    stats[f"{stat_pfx}_comet_avg"]       = round(sum(all_scores) / len(all_scores), 4)
    if not is_dailytalk:
        raw["stats"] = stats
    save_raw(stats=stats)

    print(f"\n── COMET 결과 ({hyp_field}) ──────────────────────────")
    print(f"  샘플 수  : {len(scores)}")
    print(f"  평균 점수: {avg:.4f}")
    print(f"  최소     : {min(scores):.4f}")
    print(f"  최대     : {max(scores):.4f}")
    print(f"  저장     : {output_path}")


if __name__ == "__main__":
    main()
