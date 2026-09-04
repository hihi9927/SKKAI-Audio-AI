#!/usr/bin/env python3
"""ACL 60/60 gold 문장의 **통짜 발표 내 타임스탬프**를 복원한다.

배포판에는 문장별 오디오(`segmented_wavs/gold/sent_N.wav`)와 통짜 발표
(`full_wavs/<talk>.wav`)가 둘 다 있는데 **시각 정보가 없다.** XML 에도 없고,
offset/duration 이 담긴 yaml 은 SHAS 자동 분절용뿐이라 gold 에는 쓸 수 없다.

다행히 gold 문장 wav 는 통짜에서 **그대로 잘라낸 것**이라 샘플이 바이트 단위로
일치한다. 그래서 통짜 안에서 문장 바이트열을 찾으면 시각이 정확히 복원된다
(실측: sent_1 이 2022.acl-long.268 의 2.44초 지점, 바이트 완전 일치).

이 시각이 있어야 **통짜를 끊지 않고 흘려보내면서 문장 단위로 채점**할 수 있다.

    python evaluation/ast/recover_acl6060_timings.py --split dev
    python evaluation/ast/recover_acl6060_timings.py --split eval

산출물: `<acl-root>/timings_<split>.json`
    {"talks": {...}, "segments": [{"seg_id", "talk_id", "offset", "duration"}, ...]}
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import wave
from pathlib import Path

SR = 16000
SAMPLE_WIDTH = 2          # 16-bit PCM


def read_pcm(path: Path) -> bytes:
    """wav 의 원시 PCM 바이트. 포맷이 다르면 즉시 실패시킨다 — 조용히 리샘플하면
    바이트 일치 탐색이 통째로 무의미해진다."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name}: 예상과 다른 포맷 "
                f"{w.getframerate()}Hz/{w.getnchannels()}ch/{w.getsampwidth()*8}bit")
        return w.readframes(w.getnframes())


def parse_docs(xml_path: Path) -> list[tuple[str, list[int]]]:
    """[(talk_id, [seg_id...])] — XML 등장 순서 그대로."""
    x = xml_path.read_text(encoding="utf-8")
    out = []
    for did, body in re.findall(r'<doc docid="([^"]+)"[^>]*>(.*?)</doc>', x, re.S):
        ids = [int(i) for i in re.findall(r'<seg id="(\d+)">', body)]
        out.append((did, ids))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="ACL 60/60 gold 문장 타임스탬프 복원")
    p.add_argument("--acl-root", default="~/datasets/acl6060")
    p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--probe-sec", type=float, default=0.5,
                   help="전체 일치가 실패했을 때 시도할 앞부분 길이(초)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.acl_root).expanduser().resolve() / "acl_6060" / a.split
    xml = root / "text" / "xml" / f"ACL.6060.{a.split}.en-xx.en.xml"
    gold = root / "segmented_wavs" / "gold"

    docs = parse_docs(xml)
    print(f"[{a.split}] talk {len(docs)}개 / 문장 {sum(len(s) for _, s in docs)}개")

    segments, talks = [], {}
    n_exact = n_probe = n_fail = 0

    for talk_id, seg_ids in docs:
        full_path = root / "full_wavs" / f"{talk_id}.wav"
        full = read_pcm(full_path)
        talks[talk_id] = {
            "wav": str(full_path),
            "duration": round(len(full) / SAMPLE_WIDTH / SR, 4),
            "n_segments": len(seg_ids),
        }
        # 문장은 시간 순서이므로 앞에서부터 훑되, 커서는 **직전 문장의 시작**에 둔다.
        # 끝에 두면 안 된다 — gold 경계는 인접 문장끼리 조금씩 겹친다(실측 4건:
        # sent_77 이 492.55s 에서 시작하는데 sent_76 은 493.11s 에 끝난다, 겹침 0.56s).
        # 끝을 커서로 삼으면 그런 문장을 영영 못 찾는다.
        cursor = 0
        prev_start = 0.0
        n_overlap = 0
        for sid in seg_ids:
            seg_path = gold / f"sent_{sid}.wav"
            seg = read_pcm(seg_path)
            idx = full.find(seg, cursor)
            how = "exact"
            if idx < 0:
                # 전체가 안 맞으면 앞부분만으로 재시도(끝단이 다듬어진 경우 대비)
                probe = seg[: int(a.probe_sec * SR) * SAMPLE_WIDTH]
                idx = full.find(probe, cursor) if probe else -1
                how = "probe"
            if idx < 0 or idx % SAMPLE_WIDTH != 0:
                print(f"  !! sent_{sid} ({talk_id}) 탐색 실패")
                n_fail += 1
                continue
            off = idx / SAMPLE_WIDTH / SR
            dur = len(seg) / SAMPLE_WIDTH / SR
            if off < prev_start - 1e-6:
                print(f"  !! sent_{sid} 시각 역전 (offset {off:.2f} < 이전 시작 {prev_start:.2f})")
                n_fail += 1
                continue
            if segments and segments[-1]["talk_id"] == talk_id:
                gap = off - (segments[-1]["offset"] + segments[-1]["duration"])
                if gap < -1e-6:
                    n_overlap += 1
            segments.append({
                "seg_id": sid, "talk_id": talk_id,
                "offset": round(off, 4), "duration": round(dur, 4),
            })
            cursor = idx          # 다음 문장은 이 문장 **시작** 이후에서 찾는다
            prev_start = off
            n_exact += how == "exact"
            n_probe += how == "probe"
        if n_overlap:
            talks[talk_id]["n_overlapping"] = n_overlap

    total = sum(len(s) for _, s in docs)
    print(f"\n복원: 완전일치 {n_exact} / 앞부분일치 {n_probe} / 실패 {n_fail}  (전체 {total})")
    if segments:
        speech = sum(s["duration"] for s in segments)
        audio = sum(t["duration"] for t in talks.values())
        print(f"발표 오디오 {audio/60:.1f}분 / 문장 합계 {speech/60:.1f}분 "
              f"(문장 사이 정적 {(audio-speech)/60:.1f}분, {(1-speech/audio)*100:.1f}%)")
        durs = sorted(s["duration"] for s in segments)
        print(f"문장 길이: 중앙 {durs[len(durs)//2]:.2f}s "
              f"최소 {durs[0]:.2f}s 최대 {durs[-1]:.2f}s")

    out = Path(a.out) if a.out else (Path(a.acl_root).expanduser().resolve()
                                     / f"timings_{a.split}.json")
    out.write_text(json.dumps({"split": a.split, "talks": talks,
                               "segments": segments}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"저장: {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
