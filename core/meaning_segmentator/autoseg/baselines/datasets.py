"""평가 데이터셋 추상화 — FLEURS 전용 하드코딩을 걷어낸다.

`bleu_eval` 은 원래 FLEURS 매니페스트 경로와 FLEURS TSV 를 직접 읽었다. CoVoST2 처럼
길이·오디오 경로를 매니페스트가 직접 들고 있는 데이터셋을 붙이려면 그 부분이 갈려야 한다.

데이터셋마다 다른 것은 넷뿐이다:
    1. 매니페스트 파일명 규칙
    2. 발화 길이를 어디서 읽나 (FLEURS 는 별도 TSV, CoVoST2 는 매니페스트 안)
    3. 오디오 파일을 어떻게 찾나
    4. 길이·타임스탬프의 **조회 키** (FLEURS 는 talk_id — 한 문장에 화자별 녹음이 여럿,
       CoVoST2 는 utt_id — 발화가 곧 파일)
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_DIR = _REPO_ROOT / "evaluation" / "ast" / "manifests"


@dataclass
class Entry:
    utt_id: str
    src: str
    ref: str
    key: str                 # 길이·타임스탬프 조회 키
    dur_ms: float | None
    wav: Path | None
    split: str | None = None


class DatasetSpec:
    name = "?"

    def manifest(self, tag: str, tgt: str) -> Path:
        raise NotImplementedError

    def entries(self, tag: str, tgt: str) -> dict[str, Entry]:
        raise NotImplementedError

    def wordtimes_path(self, tag: str, source: str) -> Path:
        suffix = "" if source == "ctc" else f"_{source}"
        return MANIFEST_DIR / f"{self.name}_en_{tag}_wordtimes{suffix}.json"


class Fleurs(DatasetSpec):
    """길이는 TSV 6번 열(샘플 수), 오디오는 `audio/<split>/<wav>`.

    한 문장 id 에 화자별 녹음이 여럿이라 **길이 중앙값 녹음 하나**를 대표로 쓴다.
    """

    name = "fleurs_nway"

    def __init__(self, lang_dir: str = "en_us"):
        self.base = Path.home() / "datasets" / "fleurs" / "data" / lang_dir
        self._tsv: dict[str, list[tuple[str, int, str]]] | None = None

    def _load_tsv(self) -> dict[str, list[tuple[str, int, str]]]:
        if self._tsv is not None:
            return self._tsv
        out: dict[str, list[tuple[str, int, str]]] = {}
        for split in ("train", "dev", "test"):
            f = self.base / f"{split}.tsv"
            if not f.exists():
                continue
            with f.open(encoding="utf-8") as fh:
                # TSV 는 따옴표 이스케이프가 없다 — QUOTE_NONE 필수.
                for c in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if len(c) >= 6 and c[5].isdigit():
                        out.setdefault(c[0], []).append((c[1], int(c[5]), split))
        self._tsv = out
        return out

    def manifest(self, tag: str, tgt: str) -> Path:
        return MANIFEST_DIR / f"fleurs_nway_en-{tgt}_{tag}.jsonl"

    def entries(self, tag: str, tgt: str) -> dict[str, Entry]:
        tsv = self._load_tsv()
        out: dict[str, Entry] = {}
        with self.manifest(tag, tgt).open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                tid = str(e.get("talk_id", ""))
                recs = sorted(tsv.get(tid, []), key=lambda r: r[1])
                dur = wav = None
                if recs:
                    name, n, split = recs[len(recs) // 2]
                    dur = statistics.median(r[1] for r in recs) / 16000 * 1000
                    wav = self.base / "audio" / split / name
                out[e["utt_id"]] = Entry(e["utt_id"], e["src_text"], e["tgt_text"],
                                         tid, dur, wav, e.get("fleurs_split"))
        return out


class CoVoST2(DatasetSpec):
    """길이·오디오 경로를 매니페스트가 직접 들고 있다. 발화 하나 = 파일 하나."""

    name = "covost2"

    def manifest(self, tag: str, tgt: str) -> Path:
        return MANIFEST_DIR / f"covost2_en-{tgt}_{tag}.jsonl"

    def entries(self, tag: str, tgt: str) -> dict[str, Entry]:
        out: dict[str, Entry] = {}
        with self.manifest(tag, tgt).open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                out[e["utt_id"]] = Entry(
                    e["utt_id"], e["src_text"], e["tgt_text"], e["utt_id"],
                    float(e["duration"]) * 1000, Path(e["wav"]), None)
        return out


_REGISTRY = {"fleurs": Fleurs, "covost2": CoVoST2}


def get(name: str, **kw) -> DatasetSpec:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown dataset: {name} (사용 가능: {list(_REGISTRY)})")
    return _REGISTRY[name](**kw)


def alignment_jobs(dataset: str, tag: str, tgt: str,
                   limit: int = 0) -> list[tuple[str, str, Path, float]]:
    """강제정렬에 넣을 (조회키, 소스문장, wav 경로, 길이ms) 목록.

    두 정렬기 스크립트가 공유한다 — 같은 녹음·같은 텍스트를 써야 차이가 정렬기에서만 난다.
    오디오나 길이가 없는 항목은 조용히 빠진다 (호출부에서 개수로 확인할 것).
    """
    spec = get(dataset)
    out = []
    for e in spec.entries(tag, tgt).values():
        if e.wav is None or not e.wav.exists() or not e.dur_ms:
            continue
        out.append((e.key, e.src, e.wav, e.dur_ms))
        if limit and len(out) >= limit:
            break
    return out
