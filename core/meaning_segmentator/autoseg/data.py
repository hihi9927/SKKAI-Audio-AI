"""A0 Data Preparer — 결정론적.

임의의 문장 소스를 {"id", "text"} 리스트로 정규화하고 train/dev/test 로 층화 분할한다.
루프는 train/dev만 본다. test는 최종 1회 리포트에서만 사용.
"""

from __future__ import annotations

import csv
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Sentence:
    id: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text}


# ── 로더 ─────────────────────────────────────────────────────────────────

_JA_TERMINATORS = "。！？"


def load_kokoro(path: Path | None = None) -> list[Sentence]:
    """KokoroSpeech metadata.csv (ja).

    형식: id|분かち書き 텍스트|음소열

    두 가지 정규화가 필요하다.

    1. 2번째 필드는 형태소 단위로 공백 분리되어 있으나 일본어 원문에는 공백이 없다.
       공백을 제거해 자연스러운 표기로 되돌린다.
    2. **각 행은 문장이 아니라 낭독 호흡 단위다.** 행이 어중에서 끊겨
       "た。" 로 시작하거나 "ごんは、「ふふん、" 로 끝난다. 그대로 쓰면 분절 대상이
       문장이 아니게 되어 실험 자체가 무의미해진다. 인용부호 깊이를 추적하면서
       문말 부호가 나올 때까지 이어 붙여 문장을 복원한다.
    """
    path = path or (_REPO_ROOT / "evaluation" / "KokoroSpeech" / "metadata.csv")
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 2:
                continue
            text = row[1].replace(" ", "").replace("　", "").strip()
            if text:
                rows.append((row[0], text))

    stories: dict[str, str] = {}
    for rid, text in rows:
        stories.setdefault(rid.rsplit("-", 1)[0], "")
        stories[rid.rsplit("-", 1)[0]] += text

    out: list[Sentence] = []
    for story, blob in stories.items():
        for i, sent in enumerate(_split_ja_sentences(blob)):
            out.append(Sentence(id=f"{story}-s{i:04d}", text=sent))
    return out


def _split_ja_sentences(blob: str, max_chars: int = 120) -> list[str]:
    """문말 부호에서 자르되 인용부호 안은 자르지 않는다.

    원문에 인용부호 짝이 맞지 않는 구간이 있어 그대로 두면 버퍼가 무한 누적된다.
    max_chars 를 넘으면 강제로 끊되, **그렇게 끊긴 조각은 버린다** — 인용부호가
    열린 채 끝나거나 문말 부호 없이 잘린 텍스트는 문장이 아니고, 분절 모델이
    빈 출력을 내놓는 원인이 된다 (실측 확인). 깨진 입력으로 프롬프트를 평가하면
    포맷 유효율이 영원히 1.0에 도달하지 못한다.
    """
    out, buf, depth = [], "", 0
    for ch in blob:
        buf += ch
        if ch == "「":
            depth += 1
        elif ch == "」":
            depth = max(0, depth - 1)
        elif ch in _JA_TERMINATORS and depth == 0:
            out.append(buf)
            buf = ""
        if len(buf) > max_chars:      # 강제 절단분은 채택하지 않는다
            buf, depth = "", 0
    if buf.strip():
        out.append(buf)
    return [s for s in (x.strip() for x in out) if _is_wellformed(s)]


def _is_wellformed(s: str) -> bool:
    """인용부호 짝이 맞고 문말 부호로 끝나는 것만 문장으로 인정."""
    if not s or s.count("「") != s.count("」"):
        return False
    return s[-1] in _JA_TERMINATORS or s[-1] == "」"


def load_json_entries(path: Path, text_field: str = "text", id_field: str = "file") -> list[Sentence]:
    """평가 데이터셋 JSON 두 구조를 모두 흡수.

    KsponSpeech:  {"data": [...]}
    DailyTalk  :  {"0": {"data": [...]}, "1": {...}}
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict] = []
    if isinstance(raw, list):
        entries = raw
    elif "data" in raw:
        entries = raw["data"]
    else:
        for v in raw.values():
            if isinstance(v, dict) and "data" in v:
                entries.extend(v["data"])
    out = []
    for i, e in enumerate(entries):
        t = (e.get(text_field) or "").strip()
        if t:
            out.append(Sentence(id=str(e.get(id_field, i)), text=t))
    return out


def load_kspon(path: Path | None = None) -> list[Sentence]:
    """KsponSpeech eval_clean_1000.json (ko).

    실제 자발 발화 ASR 전사. KokoroSpeech 와 달리 문장 재구성이 필요 없다 —
    각 항목이 이미 하나의 발화 단위다. 구두점이 없는 항목이 많은 것이 정상이며,
    그것이 바로 실시간 분절이 풀어야 하는 조건이다."""
    path = path or (_REPO_ROOT / "evaluation" / "KsponSpeech" / "transcribe" / "eval_clean_1000.json")
    return load_json_entries(path, text_field="text", id_field="file")


# 로더는 데이터셋별로 다를 수밖에 없다 (파일 포맷·전처리가 데이터셋 고유이므로).
# 언어 무관이어야 하는 것은 에이전트와 지표이지 로더가 아니다.
LOADERS = {
    "kokoro": load_kokoro,
    "kspon": load_kspon,
}


# ── 분할 ─────────────────────────────────────────────────────────────────

def _stratum(s: Sentence) -> tuple[int, bool]:
    """길이 3분위 × 구두점 유무로 층을 만든다.

    구두점 판정은 유니코드 범주로 한다 — 문자 목록을 박으면 층화가 언어 종속이 된다."""
    n = len(s.text)
    bucket = 0 if n < 25 else (1 if n < 40 else 2)
    has_punct = any(unicodedata.category(ch).startswith("P") for ch in s.text)
    return bucket, has_punct


def split_data(
    sentences: list[Sentence],
    n_train: int,
    n_dev: int,
    n_test: int,
    min_chars: int = 20,
    seed: int = 20260806,
) -> dict[str, list[Sentence]]:
    """층화 후 라운드로빈으로 train/dev/test 를 겹치지 않게 뽑는다."""
    pool = [s for s in sentences if len(s.text) >= min_chars]
    if len(pool) < n_train + n_dev + n_test:
        raise ValueError(
            f"문장 부족: 사용 가능 {len(pool)}개 < 요청 {n_train + n_dev + n_test}개 "
            f"(min_chars={min_chars})"
        )

    rng = random.Random(seed)
    strata: dict[tuple, list[Sentence]] = {}
    for s in pool:
        strata.setdefault(_stratum(s), []).append(s)
    for v in strata.values():
        rng.shuffle(v)

    keys = sorted(strata, key=lambda k: (-len(strata[k]), str(k)))
    order: list[Sentence] = []
    idx = {k: 0 for k in keys}
    while len(order) < len(pool):
        progressed = False
        for k in keys:
            i = idx[k]
            if i < len(strata[k]):
                order.append(strata[k][i])
                idx[k] = i + 1
                progressed = True
        if not progressed:
            break

    return {
        "train": order[:n_train],
        "dev": order[n_train : n_train + n_dev],
        "test": order[n_train + n_dev : n_train + n_dev + n_test],
    }


def write_splits(splits: dict[str, list[Sentence]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def read_split(path: Path) -> list[Sentence]:
    return [Sentence(**r) for r in json.loads(path.read_text(encoding="utf-8"))]
