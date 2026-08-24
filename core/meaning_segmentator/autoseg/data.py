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


def load_kspon_train(path: Path | None = None) -> list[Sentence]:
    """KsponSpeech train.json — 대용량 풀 (10000발화, 25자 이상 4884).

    `kspon`(eval_clean_1000)은 25자 필터 후 337문장뿐이라 train-pool·dev·test 를
    키우면 바닥난다 (run04 에서 420 요청 > 337 로 실패). 같은 코퍼스의 학습 분할
    전사이고 id·텍스트 모두 유니크 실측 확인. autoseg 는 ASR 을 평가하지 않으므로
    학습 분할 사용이 오염을 만들지 않는다."""
    path = path or (_REPO_ROOT / "evaluation" / "KsponSpeech" / "transcribe" / "train.json")
    return load_json_entries(path, text_field="text", id_field="file")


def load_ast_manifest(path: Path, text_field: str = "src_text",
                      id_field: str = "utt_id") -> list[Sentence]:
    """`evaluation/ast/manifests/*.jsonl` — AST 평가 트랙의 공용 매니페스트.

    한 줄이 한 발화이고 오디오 경로·정답 번역까지 들고 있으나, autoseg 는 **소스
    텍스트만** 쓴다. 정답 번역(`tgt_text`)은 일부러 안 읽는다 — 목적함수가 참조 없는
    QE(`adequacy`)와 실제 번역기 출력 기반 NLI(`contradiction`)이므로, 참조를 끌어오면
    루프가 재는 것과 다른 자가 섞인다.
    """
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            t = (e.get(text_field) or "").strip()
            if t:
                out.append(Sentence(id=str(e.get(id_field, i)), text=t))
    return out


def load_fleurs_en_de(path: Path | None = None) -> list[Sentence]:
    """FLEURS en-de test (346발화). 소스는 영어 낭독체.

    KsponSpeech(자발 발화, 구두점 없음)와 성격이 정반대다 — 문어체에 구두점이 완비돼
    있어 `measured_profile` 의 `trailing_punctuation` 이 실제로 잡히고, 문장이 길다
    (어절 중앙값 21). **총 346개뿐이라** `train_pool + dev + test <= 346` 을 지켜야
    `split_data` 가 죽지 않는다.
    """
    path = path or (_REPO_ROOT / "evaluation" / "ast" / "manifests"
                    / "fleurs_en-de_test.jsonl")
    return load_ast_manifest(path)


def load_fleurs_en_ko(path: Path | None = None) -> list[Sentence]:
    """FLEURS en-ko test (270발화). 소스 영어는 `fleurs-en-de` 와 같은 낭독체다.

    FLEURS 는 n-way 병렬이라 소스 문장은 공유되지만, ko_kr 전사가 있는 문장이 더
    적어 en-de(346)의 부분집합에 가깝다 (교집합 268, ko 전용 2). 즉 **en-de 와
    en-ko 는 사실상 같은 영어 문장을 타깃만 바꿔 돌리는 대조군**이다 — autoseg 는
    `tgt_text` 를 읽지 않으므로 두 런의 차이는 번역기 타깃 언어와 지표뿐이다.

    총 270개뿐이라 `train_pool + dev + test <= 270` 을 지켜야 `split_data` 가
    죽지 않는다.

    기본 `--contradiction-backend xlmr-anli` 가 다국어라 타깃별 지정이 필요 없다.
    """
    path = path or (_REPO_ROOT / "evaluation" / "ast" / "manifests"
                    / "fleurs_en-ko_test.jsonl")
    return load_ast_manifest(path)


# 로더는 데이터셋별로 다를 수밖에 없다 (파일 포맷·전처리가 데이터셋 고유이므로).
# 언어 무관이어야 하는 것은 에이전트와 지표이지 로더가 아니다.
def _load_multi2en(src: str, path: Path | None = None) -> list[Sentence]:
    """FLEURS {de,ja,zh}→en 루프용 240문장. 소스 언어만 다르고 **문장은 셋이 동일**하다.

    en 기준 층화 정렬(`fleurs_nway_en_clean500_order.json`)의 500~739 구간이라
    `en-multi/run06`(프롬프트 생성)·`clean500`(en→X 평가) 어느 쪽과도 겹치지 않는다.
    같은 정렬의 740~1239 구간이 최종 평가용 `multi2en_eval500` 이다.

    **길이 하한 필터는 없앴다** (`split_data` 참조). 예전에는 `--min-chars 0` 을 손으로
    줘야 했다 — 기본 25자로 두면 밀도가 높은 zh 판만 240 중 34개가 빠져 세 트랙이 같은
    문장을 쓴다는 전제가 깨졌기 때문이다. 이제 그 우회가 필요 없다.

    소스 단위 중앙값이 언어마다 다르므로 (`pipeline.unit_count`: de 어절 20,
    ja 문자 52, zh 문자 38) **`--t-grid`·`--min-gap` 을 그대로 쓰면 안 된다** —
    환산은 `../MULTI2EN_DATASET.md` 참조.
    """
    path = path or (_REPO_ROOT / "evaluation" / "ast" / "manifests"
                    / f"fleurs_nway_{src}-en_multi2en_loop240.jsonl")
    return load_ast_manifest(path)


LOADERS = {
    "kokoro": load_kokoro,
    "kspon": load_kspon,
    "kspon-train": load_kspon_train,
    "fleurs-en-de": load_fleurs_en_de,
    "fleurs-en-ko": load_fleurs_en_ko,
    "fleurs-de-en": lambda: _load_multi2en("de"),
    "fleurs-ja-en": lambda: _load_multi2en("ja"),
    "fleurs-zh-en": lambda: _load_multi2en("zh"),
}


# ── 측정 프로파일 ────────────────────────────────────────────────────────

# **결정론적 코드를 움직이는 필드는 측정으로 채운다.** v1 은 이 값들을 Language
# Profiler(LLM)가 추측하게 뒀는데, 실제로 틀렸다:
#   - runs/ja-ko/ja-ko-test04 는 `trailing_punctuation` 이 통째로 null 이었다
#     (같은 데이터의 test06 은 6종을 냈다). 유니코드 폴백이 받아줘서 우연히 무사했다.
#   - ko 런들은 `['.','?']`(실측 일치), `['.','?','!','…']`(잉여),
#     `['.','?',',',…]`(코퍼스에 실제로 있는 `,` 를 포함 — 동작이 갈림)로 제각각이었다.
# 지표는 틀리면 숫자로 드러나지만 검증기 규칙은 조용히 틀린다.
#
# **language_profile.json 은 고치지 않는다.** JSON 을 바꾸면 그게 prompt_v0 writer 와
# Prompt Engineer 컨텍스트로 들어가 프롬프트가 달라지고 기존 런과 비교가 깨진다.
# 덮어쓰기는 소비 지점(loop.py)에서만 한다.

_PO_SENTENCE_OPENERS = "¿¡"


def measure_profile(texts: list[str], min_count: int = 2, attach_ratio: float = 0.9) -> dict:
    """코퍼스에서 직접 재는 언어 표기 특성. LLM 을 쓰지 않는다.

    `trailing_punctuation` 의 정의는 **"앞 텍스트에 붙는 구두점"** 이다. 그래서
    문말 등장만 세면 안 된다 — 일본어 `、` 는 절 구분자라 문말에 안 나오지만 태그
    직후에 오면 안 되는 문자다. 대신 **거의 항상 비공백 문자 뒤에 붙어 나오는가**로
    판정한다. 이 규칙은 언어 무관이고, 여는 부호(`「`, 스페인어 `¿¡`)는 자동으로
    빠진다 — 그것들은 공백이나 문장 시작 뒤에 오기 때문이다.
    """
    total_chars = sum(len(t) for t in texts) or 1
    space_ratio = sum(t.count(" ") for t in texts) / total_chars
    # **판정은 코퍼스 집계가 아니라 문장별 중앙값으로 한다.** 중국어 FLEURS 는 삽입된 라틴
    # 고유명사·숫자(`CafeNet El Sol`, `30 美元`) 때문에 소수 문장이 공백을 갖는데, 그것만으로
    # 집계 비율이 0.0213 이 되어 임계값 0.02 를 아슬하게 넘는다 (240문장 중 공백 보유는 62개,
    # 중앙값은 0). 그대로 두면 zh 가 spaced 로 잡혀 `unit_count` 가 문자 대신 어절을 세고
    # (중앙 1), 조각 예산과 `min_gap` 이 통째로 무의미해진다. 중앙값이면 de 0.135 / en 0.159
    # vs ja 0.000 / zh 0.000 으로 양쪽 여유가 6배 이상이다. 규칙은 여전히 언어 무관이다.
    per_sentence = sorted(t.count(" ") / len(t) for t in texts if t)
    space_ratio_median = (per_sentence[len(per_sentence) // 2]
                          if per_sentence else 0.0)
    space_sentence_ratio = (sum(1 for t in texts if " " in t) / len(texts)
                            if texts else 0.0)

    attached: dict[str, int] = {}
    seen: dict[str, int] = {}
    finals: dict[str, int] = {}
    n_punct_final = 0
    for t in texts:
        s = t.strip()
        if not s:
            continue
        for i, ch in enumerate(s):
            if not unicodedata.category(ch).startswith("P"):
                continue
            seen[ch] = seen.get(ch, 0) + 1
            if i > 0 and not s[i - 1].isspace():
                attached[ch] = attached.get(ch, 0) + 1
        if unicodedata.category(s[-1]).startswith("P"):
            finals[s[-1]] = finals.get(s[-1], 0) + 1
            n_punct_final += 1

    trailing = sorted(
        c for c, n in seen.items()
        if n >= min_count and c not in _PO_SENTENCE_OPENERS
        and unicodedata.category(c) != "Ps"
        and attached.get(c, 0) / n >= attach_ratio
    )

    # **`min_gap`·`T` 유도가 이 값을 쓴다** (`loop.derive_min_gap`). 단위는 spaced 여부로
    # 갈린다(어절 vs 문자) — 그래서 여기서 함께 재야 판정과 단위가 어긋나지 않는다.
    spaced_ = space_ratio_median > 0.02
    units = sorted((len(t.split()) if spaced_ else len(t.replace(" ", "")))
                   for t in texts if t.strip())
    return {
        "n": len(texts),
        "median_units": units[len(units) // 2] if units else 0,
        "unit": "word" if spaced_ else "char",
        "space_ratio": round(space_ratio, 4),
        "space_ratio_median": round(space_ratio_median, 4),
        "space_sentence_ratio": round(space_sentence_ratio, 4),
        "uses_spaces_between_words": space_ratio_median > 0.02,
        "trailing_punctuation": trailing,
        "punctuation_counts": dict(sorted(seen.items(), key=lambda x: -x[1])),
        "final_punctuation_counts": dict(sorted(finals.items(), key=lambda x: -x[1])),
        "punctuation_final_rate": round(n_punct_final / len(texts), 4) if texts else 0.0,
    }


def reconcile_profile(measured: dict, profile: dict) -> tuple[bool, str | None, list[str]]:
    """측정값과 LLM 프로파일을 대조한다. 반환 `(spaced, trailing_punct, 경고들)`.

    **측정값이 이긴다.** 불일치는 경고로 남겨 프로파일러 품질 진단에 쓴다."""
    warn: list[str] = []
    spaced = bool(measured["uses_spaces_between_words"])
    if profile.get("uses_spaces_between_words") not in (None, spaced):
        warn.append(f"uses_spaces_between_words: LLM={profile.get('uses_spaces_between_words')} "
                    f"측정={spaced} (공백비율 {measured['space_ratio']}) — 측정값 채택")

    llm_punct = set(profile.get("trailing_punctuation") or [])
    measured_punct = set(measured["trailing_punctuation"])
    if not llm_punct:
        warn.append(f"trailing_punctuation: LLM 이 비었음 — 측정값 {sorted(measured_punct)} 채택")
    elif llm_punct != measured_punct:
        warn.append(f"trailing_punctuation: LLM={sorted(llm_punct)} "
                    f"측정={sorted(measured_punct)} — 측정값 채택")
    return spaced, ("".join(sorted(measured_punct)) or None), warn


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
    seed: int = 20260806,
) -> dict[str, list[Sentence]]:
    """층화 후 라운드로빈으로 train/dev/test 를 겹치지 않게 뽑는다.

    **길이 하한 필터(`min_chars`)는 없앴다.** 짧은 문장을 거르는 선은 이미
    `min_gap` 이 갖고 있다 — `truncate` 는 양끝에서 각각 `min_gap` 이상 떨어진 자리에만
    경계를 놓으므로 `unit_count < 2*min_gap` 인 문장은 **구조적으로 무분절**이다.
    `min_chars` 는 그 선을 문자 수로 근사한 두 번째 절단선이었고, 단위가 달라
    언어마다 어긋났다 (en/ko 는 25자 ≈ 2*min_gap 으로 우연히 일치, zh 는 무분절 경계가
    12자인데 25자에서 잘려 240 중 34 문장이 과다 탈락 — 그래서 multi2en 트랙이
    `--min-chars 0` 을 손으로 꺼야 했다).

    거를 이유도 사라졌다. 구조적 무분절 문장은 `coverage_need == 0` 이라 분절 호출을
    건너뛸 수 있고(결과가 호출 전에 확정), `effective` 가 프롬프트 불변 상수라 채택
    판정의 쌍체 차이에 **정확히 0** 을 기여한다.
    """
    pool = list(sentences)
    if len(pool) < n_train + n_dev + n_test:
        raise ValueError(
            f"문장 부족: 사용 가능 {len(pool)}개 < 요청 {n_train + n_dev + n_test}개"
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

    # **test -> dev -> train 순으로 배분한다.** train 을 앞에서 떼면 train 크기를 바꿀
    # 때마다 test/dev 가 통째로 밀려 런 간 비교가 깨진다 (실측: train 30 -> 60 에서
    # test 겹침 70/100, pool 120 에서 10/100). 평가 분할을 고정해야 train 크기를
    # 실험 변수로 쓸 수 있다.
    return {
        "test": order[:n_test],
        "dev": order[n_test : n_test + n_dev],
        "train": order[n_test + n_dev : n_test + n_dev + n_train],
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
