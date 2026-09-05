"""시연용 테스트 음성을 언어별로 만든다 (en / ja / ko).

    python tools/partial_demo/make_demo_wavs.py            # 세 언어 모두
    python tools/partial_demo/make_demo_wavs.py --lang ja --n 4

발화 사이에 침묵을 넣는다. VAD 커밋과 슬롯 리셋을 실제로 태워, 문장이 하나씩
확정되며 화면에 쌓이는 모습을 보기 위해서다. 침묵이 없으면 한 덩어리로 붙어
버려 시연에서 문장 경계가 안 보인다.

출력은 web/partial_test_<lang>.wav (16kHz mono s16le) 와, 무엇이 나와야 하는지
대조할 수 있게 web/partial_test_<lang>.txt 다. 둘 다 gitignore 대상이라
필요할 때 이 스크립트로 다시 만든다.

데이터 출처:
  en, ja  FLEURS dev — 낭독체라 문장이 또렷하고 구두점이 있는 원문 전사가 붙는다
  ko      KsponSpeech sample_data/eval_clean — FLEURS ko_kr 에는 오디오가 없다.
          대화체라 낭독체인 나머지 둘과 결은 다르다
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
WEB = HERE / "web"
REPO = HERE.parent.parent

FLEURS = pathlib.Path("/home/mobility/datasets/fleurs/data")
KSPON = REPO / "evaluation/KsponSpeech"

SR = 16000
FLEURS_LOCALE = {"en": "en_us", "ja": "ja_jp"}


def load_fleurs(lang: str, n: int, max_sec: float) -> list[tuple[np.ndarray, str]]:
    """FLEURS dev 에서 짧은 발화부터 n 개."""
    locale = FLEURS_LOCALE[lang]
    root = FLEURS / locale
    tsv, audio_dir = root / "dev.tsv", root / "audio/dev"
    if not tsv.is_file() or not audio_dir.is_dir():
        sys.exit(f"FLEURS {locale} 없음: {root}")

    rows = []
    # FLEURS TSV 는 본문에 따옴표가 그대로 들어 있다. QUOTE_NONE 이 아니면
    # 여러 행이 한 필드로 붙어 수천 어절짜리 잔해가 섞인다.
    with tsv.open(encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 6:
                continue
            wav = audio_dir / r[1]
            if not wav.is_file():
                continue
            try:
                dur = int(r[5]) / SR
            except ValueError:
                continue
            rows.append((dur, wav, r[2].strip()))

    rows.sort(key=lambda x: x[0])
    picked = [r for r in rows if 3.0 <= r[0] <= max_sec][:n]
    if len(picked) < n:
        sys.exit(f"{lang}: {max_sec}초 이하 발화가 {len(picked)}개뿐이다 — --max-sec 를 올려라")

    out = []
    for _, wav, text in picked:
        audio, sr = sf.read(wav, dtype="float32")
        if sr != SR:
            sys.exit(f"{wav} 의 샘플레이트가 {sr} 다 — {SR} 를 기대했다")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out.append((audio, text))
    return out


def load_kspon(n: int, max_sec: float) -> list[tuple[np.ndarray, str]]:
    """KsponSpeech 샘플. pcm 은 헤더 없는 16kHz s16le 라 그대로 읽는다."""
    meta = KSPON / "transcribe/eval_clean.json"
    pcm_dir = KSPON / "sample_data/eval_clean"
    if not meta.is_file() or not pcm_dir.is_dir():
        sys.exit(f"KsponSpeech 샘플 없음: {pcm_dir}")

    text_of = {d["file"]: d["text"] for d in json.loads(meta.read_text(encoding="utf-8"))["data"]}
    out = []
    for pcm in sorted(pcm_dir.glob("*.pcm")):
        text = text_of.get(pcm.stem)
        if not text:
            continue
        raw = pcm.read_bytes()
        if len(raw) % 2:            # 홀수 바이트로 끝나는 파일이 섞여 있다
            raw = raw[:-1]
        audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if not (3.0 <= len(audio) / SR <= max_sec):
            continue
        out.append((audio, text))
        if len(out) >= n:
            break
    if len(out) < n:
        sys.exit(f"ko: 조건에 맞는 발화가 {len(out)}개뿐이다 — --max-sec 를 올려라")
    return out


def build(lang: str, n: int, gap: float, lead: float, max_sec: float) -> None:
    items = load_kspon(n, max_sec) if lang == "ko" else load_fleurs(lang, n, max_sec)

    silence = np.zeros(int(SR * gap), dtype="float32")
    chunks = [np.zeros(int(SR * lead), dtype="float32")]
    lines, at = [], lead
    for audio, text in items:
        lines.append(f"[{at:6.2f}s] {text}")
        at += len(audio) / SR + gap
        chunks.append(audio)
        chunks.append(silence)
    merged = np.concatenate(chunks)

    # 크게 튀는 발화가 섞여도 클리핑되지 않게 최댓값만 맞춘다.
    peak = float(np.abs(merged).max())
    if peak > 0:
        merged = merged / peak * 0.9

    WEB.mkdir(parents=True, exist_ok=True)
    wav_out = WEB / f"partial_test_{lang}.wav"
    txt_out = WEB / f"partial_test_{lang}.txt"
    sf.write(wav_out, merged, SR, subtype="PCM_16")
    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{lang}: {wav_out.name}  {len(merged)/SR:.1f}s  발화 {len(items)}개  침묵 {gap}s")
    for l in lines:
        print("   ", l)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "ja", "ko", "all"], default="all")
    ap.add_argument("--n", type=int, default=3, help="이어 붙일 발화 수")
    ap.add_argument("--gap", type=float, default=1.2, help="발화 사이 침묵(초)")
    ap.add_argument("--lead", type=float, default=0.3, help="맨 앞 침묵(초)")
    ap.add_argument("--max-sec", type=float, default=12.0, help="발화 하나의 최대 길이(초)")
    a = ap.parse_args()

    for lang in (["en", "ja", "ko"] if a.lang == "all" else [a.lang]):
        build(lang, a.n, a.gap, a.lead, a.max_sec)


if __name__ == "__main__":
    main()
