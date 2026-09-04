#!/usr/bin/env python3
"""ACL 60/60 → 장문 AST manifest. **발표 하나가 발화 하나다.**

CoVoST2 단클립과 근본적으로 다르다. 10분짜리 발표를 끊지 않고 통째로 흘려보내므로,
시스템이 내는 조각은 참조 문장과 경계가 전혀 맞지 않는다. 그래서 채점은 이 manifest 만으로
안 되고, 사후에 **mwerSegmenter 재분절**을 거쳐야 한다(`score_acl6060.py`).

그래서 각 항목에 참조를 **두 벌** 담는다:

    src_text / tgt_text   문장을 전부 이어붙인 것 — 클라이언트의 기본 채점이 쓴다.
                          문서 단위 BLEU 라 주지표가 아니다(재분절 전이므로).
    sentences[]           문장별 {seg_id, offset, duration, src, tgt}.
                          재분절 채점기가 이걸 보고 참조 경계와 시각을 잡는다.

`offset`/`duration` 은 통짜 발표 전체다. gold 문장이 발표의 99% 이상을 덮으므로
(앞 여백 0.5~2.4초, 뒤 0.3~5.9초) 잘라내지 않는다 — 원본 그대로가 가장 방어하기 쉽다.

    python evaluation/ast/recover_acl6060_timings.py --split dev   # 선행 (타임스탬프)
    python evaluation/ast/build_manifest_acl6060.py --split dev --tgt de

산출물: `manifests/acl6060_{split}_en-{tgt}.jsonl` (발표 5개 = 5줄)
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# XML 파일명의 언어 코드. 우리가 쓰는 짧은 코드 → 배포판 코드.
XML_LANG = {"de": "de", "ja": "ja", "zh": "zh", "en": "en",
            "ar": "ar", "fa": "fa", "fr": "fr", "nl": "nl",
            "pt": "pt", "ru": "ru", "tr": "tr"}


def parse_segs(xml_path: Path) -> dict[int, str]:
    """seg id → 텍스트. XML 엔티티는 풀고 공백은 정규화한다."""
    x = xml_path.read_text(encoding="utf-8")
    out = {}
    for did, body in re.findall(r'<doc docid="([^"]+)"[^>]*>(.*?)</doc>', x, re.S):
        for sid, txt in re.findall(r'<seg id="(\d+)">(.*?)</seg>', body, re.S):
            out[int(sid)] = re.sub(r"\s+", " ", html.unescape(txt)).strip()
    return out


def build(a) -> int:
    root = Path(a.acl_root).expanduser().resolve()
    split_dir = root / "acl_6060" / a.split
    timings_path = root / f"timings_{a.split}.json"
    if not timings_path.exists():
        print(f"타임스탬프가 없습니다: {timings_path}\n"
              f"  먼저: python evaluation/ast/recover_acl6060_timings.py --split {a.split}",
              file=sys.stderr)
        return 2

    tim = json.loads(timings_path.read_text(encoding="utf-8"))
    by_seg = {s["seg_id"]: s for s in tim["segments"]}

    xdir = split_dir / "text" / "xml"
    src = parse_segs(xdir / f"ACL.6060.{a.split}.en-xx.en.xml")
    tgt = parse_segs(xdir / f"ACL.6060.{a.split}.en-xx.{XML_LANG[a.tgt]}.xml")

    # 전제 검증 — 언어 간 seg id 집합이 같아야 문장 단위 대응이 성립한다.
    if set(src) != set(tgt):
        only_s, only_t = sorted(set(src) - set(tgt)), sorted(set(tgt) - set(src))
        print(f"!! seg id 불일치: en 에만 {len(only_s)}개, {a.tgt} 에만 {len(only_t)}개",
              file=sys.stderr)
        return 2
    missing = sorted(set(src) - set(by_seg))
    if missing:
        print(f"!! 타임스탬프 없는 seg {len(missing)}개: {missing[:5]}", file=sys.stderr)
        return 2

    # 발표별로 묶는다. 문장 순서는 **seg id** 가 아니라 **시각** 기준이다
    # (gold 경계가 4건 겹쳐 있어 id 순서와 미세하게 어긋날 수 있다).
    talks: dict[str, list[int]] = {}
    for sid in sorted(src):
        talks.setdefault(by_seg[sid]["talk_id"], []).append(sid)

    out_path = Path(a.out) if a.out else (
        HERE / "manifests" / f"acl6060_{a.split}_en-{a.tgt}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    total_sec = 0.0
    with out_path.open("w", encoding="utf-8") as f:
        for talk_id, sids in talks.items():
            sids = sorted(sids, key=lambda i: (by_seg[i]["offset"], i))
            info = tim["talks"][talk_id]
            sentences = []
            for sid in sids:
                s_txt, t_txt = src[sid], tgt[sid]
                if not s_txt or not t_txt:
                    n_empty += 1
                sentences.append({
                    "seg_id": sid,
                    "offset": by_seg[sid]["offset"],
                    "duration": by_seg[sid]["duration"],
                    "src": s_txt,
                    "tgt": t_txt,
                })
            f.write(json.dumps({
                "utt_id": f"acl6060_{a.split}_{talk_id}",
                "wav": info["wav"],
                "offset": 0.0,
                "duration": round(info["duration"], 3),
                "src_lang": "en",
                "tgt_lang": a.tgt,
                # 클라이언트 기본 채점용(문서 단위). 주지표는 재분절 후에 낸다.
                "src_text": " ".join(s["src"] for s in sentences),
                "tgt_text": " ".join(s["tgt"] for s in sentences),
                "speaker_id": talk_id,
                "talk_id": talk_id,
                "n_sentences": len(sentences),
                "sentences": sentences,
            }, ensure_ascii=False) + "\n")
            total_sec += info["duration"]

    print(f"manifest: {out_path}")
    print(f"  발표 {len(talks)}개 / 문장 {sum(len(v) for v in talks.values())}개 "
          f"/ 오디오 {total_sec/60:.1f}분")
    if n_empty:
        print(f"  [주의] 빈 텍스트 문장 {n_empty}개")
    per = [len(v) for v in talks.values()]
    print(f"  발표당 문장 {min(per)}~{max(per)}개 (평균 {sum(per)/len(per):.0f})")
    print(f"  발표당 길이 {total_sec/len(talks)/60:.1f}분")
    for n_cli in (5, 10):
        print(f"  실시간 페이싱 예상({n_cli}병렬): "
              f"{total_sec/60/min(n_cli, len(talks)):.0f}분/런")
    return 0


def main():
    p = argparse.ArgumentParser(description="ACL 60/60 → 장문 AST manifest")
    p.add_argument("--acl-root", default="~/datasets/acl6060")
    p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--tgt", default="de", choices=sorted(XML_LANG))
    p.add_argument("--out", default=None)
    sys.exit(build(p.parse_args()))


if __name__ == "__main__":
    main()
