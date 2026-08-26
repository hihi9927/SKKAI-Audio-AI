"""A0 Data Preparer — 결정론적.

임의의 문장 소스를 {"id", "text"} 리스트로 정규화하고 train/dev/test 로 층화 분할한다.
루프는 train/dev만 본다. test는 최종 1회 리포트에서만 사용.
"""

from __future__ import annotations

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

def _load_json_entries(path: Path, text_field: str = "text", id_field: str = "file") -> list[Sentence]:
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

    실제 자발 발화 ASR 전사. 각 항목이 이미 하나의 발화 단위다. 구두점이 없는 항목이 많은 것이 정상이며,
    그것이 바로 실시간 분절이 풀어야 하는 조건이다."""
    path = path or (_REPO_ROOT / "evaluation" / "KsponSpeech" / "transcribe" / "eval_clean_1000.json")
    return _load_json_entries(path, text_field="text", id_field="file")


def load_kspon_train(path: Path | None = None) -> list[Sentence]:
    """KsponSpeech train.json — 10000발화. `kspon` 과 **분포가 다르다**.

    도입 당시 사유는 길이 하한 필터 때문에 `kspon` 이 337문장으로 말라붙는 것이었는데,
    그 필터는 없어졌다 (`split_data` 참조). 그럼에도 남기는 이유는 **꼬리 길이**다 — 어절 수 p99 가 `kspon` 26 vs 여기 47, 최대 36 vs 72.
    긴 문장에서만 나타나는 실패(`pipeline.SEG_MAX_TOKENS` 주석의 빈 출력)는
    `kspon` 으로는 재현되지 않는다. ko-en run04·run05 가 이 풀로 돌았다.

    같은 코퍼스의 학습 분할 전사이고 id·텍스트 모두 유니크 실측 확인. autoseg 는
    ASR 을 평가하지 않으므로 학습 분할 사용이 오염을 만들지 않는다."""
    path = path or (_REPO_ROOT / "evaluation" / "KsponSpeech" / "transcribe" / "train.json")
    return _load_json_entries(path, text_field="text", id_field="file")


def _load_ast_manifest(path: Path, text_field: str = "src_text",
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


# **FLEURS 트랙은 로더 함수가 아니라 매니페스트 경로로 등록한다.** 예전에는 트랙마다
# 함수가 하나씩 있었는데(en-de, en-ko, de/ja/zh-en) 전부 `_load_ast_manifest(경로)` 에
# 경로만 다른 래퍼였다. 경로로 두면 함수 5개가 사라지고 **새 언어를 추가할 때 코드를
# 고칠 필요가 없다** — `--dataset <경로.jsonl>` 로 바로 돌린다.
#
# 발화 속도를 재려면 발화 길이가 필요한데, 그건 매니페스트 옆의 강제정렬 산출물
# (`*_unittimes.json`)에 들어 있다. 없으면 `--units-per-sec` 나 `--min-gap` 을 직접 준다.
_AST = _REPO_ROOT / "evaluation" / "ast" / "manifests"
MANIFESTS = {
    "fleurs-en-de": _AST / "fleurs_en-de_test.jsonl",
    "fleurs-en-ko": _AST / "fleurs_en-ko_test.jsonl",
    "fleurs-de-en": _AST / "fleurs_nway_de-en_multi2en_loop240.jsonl",
    "fleurs-ja-en": _AST / "fleurs_nway_ja-en_multi2en_loop240.jsonl",
    "fleurs-zh-en": _AST / "fleurs_nway_zh-en_multi2en_loop240.jsonl",
}

# 파일 포맷이 고유한 것만 로더 함수로 남는다. 나머지는 위 MANIFESTS.
LOADERS = {
    "kspon": load_kspon,               # ko. eval_clean_1000
    "kspon-train": load_kspon_train,   # ko. 같은 코퍼스, 꼬리가 2배 길다
}
DATASETS = sorted(LOADERS) + sorted(MANIFESTS)


def _manifest_path(dataset: str) -> Path | None:
    """등록된 이름이거나 `.jsonl` 경로면 매니페스트 경로를 준다. 아니면 None."""
    if dataset in MANIFESTS:
        return MANIFESTS[dataset]
    q = Path(dataset)
    return q if q.suffix == ".jsonl" and q.exists() else None


def load(dataset: str) -> list[Sentence]:
    """`--dataset` 하나로 받는다 — 등록된 이름이거나 매니페스트 경로(`.jsonl`)."""
    if dataset in LOADERS:
        return LOADERS[dataset]()
    q = _manifest_path(dataset)
    if q is None:
        raise ValueError(f"모르는 데이터셋: {dataset!r}. "
                         f"등록된 이름 {DATASETS} 이거나 .jsonl 경로여야 한다")
    return _load_ast_manifest(q)


def units_per_sec(dataset: str, sentences: list[Sentence],
                  spaced: bool) -> tuple[float | None, str]:
    """코퍼스 발화 속도 (단위/초). 반환 `(값, 출처)`.

    **발화 구간으로 잰다** — `speech_ms` 는 첫 span 시작부터 마지막 span 끝까지다.
    녹음 전체 길이(`dur_ms`)를 쓰면 앞뒤 무음이 섞여 속도가 과소평가된다:
    FLEURS 실측에서 무음이 de/ja/zh 모두 24~25% 로, 총 길이 기준과 발화 구간 기준이
    de 1.84 vs 2.43, ja 4.36 vs 5.77, zh 3.55 vs 4.74 (단위/초) 로 갈렸다.
    녹음 관행이지 언어 특성이 아니므로 빼는 것이 맞다.

    **두 기준을 섞으면 안 된다** — 25% 차이가 그대로 `min_gap` 으로 들어가 런 간
    비교가 깨진다. 그래서 출처 문자열을 함께 돌려주고 호출자가 config 에 남긴다.
    """
    man = _manifest_path(dataset)
    if man is None:
        return None, "none"
    ut = man.with_name(man.stem + "_unittimes.json")
    if not ut.exists():
        return None, "none"
    times = json.loads(ut.read_text(encoding="utf-8"))
    unit = (lambda t: len(t.split())) if spaced else (lambda t: len(t.replace(" ", "")))
    tot_u = tot_ms = 0.0
    for s in sentences:
        e = times.get(s.id)
        if not e or not e.get("speech_ms"):
            continue
        tot_u += unit(s.text)
        tot_ms += e["speech_ms"]
    if tot_ms <= 0:
        return None, "none"
    return tot_u / (tot_ms / 1000.0), f"alignment:{ut.name}"


# ── 측정 프로파일 ────────────────────────────────────────────────────────

# **결정론적 코드를 움직이는 필드는 측정으로 채운다.** v1 은 이 값들을 Language
# Profiler(LLM)가 추측하게 뒀는데, 실제로 틀렸다:
#   - runs/ja-ko/ja-ko-test04 는 `trailing_punctuation` 이 통째로 null 이었다
#     (같은 데이터의 test06 은 6종을 냈다). 유니코드 폴백이 받아줘서 우연히 무사했다.
#   - ko 런들은 `['.','?']`(실측 일치), `['.','?','!','…']`(잉여),
#     `['.','?',',',…]`(코퍼스에 실제로 있는 `,` 를 포함 — 동작이 갈림)로 제각각이었다.
# 지표는 틀리면 숫자로 드러나지만 검증기 규칙은 조용히 틀린다.
#
# **Profiler 는 이제 이 다섯을 내지 않는다** (`agents.PROFILER_SYSTEM`). 종전에는
# 코드가 실측을 쓰면서 `language_profile.json` 은 그대로 뒀는데, 그 JSON 이 프롬프트
# 작성기·PE 컨텍스트로 들어가 **모델이 읽는 규칙만 틀린 목록을 근거로 쓰였다**
# (26개 런 중 25개에서 `trailing_punctuation` 이 실측과 불일치). 지금은 실측이
# `agents.measured_facts()` 로 양쪽에 같이 들어간다.

# **모양이 여닫이를 확정하는 글자는 분류로 뺀다.** 괄호(`Ps`/`Pe`)와 곡선 따옴표
# (`Pi`/`Pf`)가 그것이다. 어느 쪽이 여는 것인지는 언어마다 뒤집히지만 — 독일어
# `„…“` 는 `“` 로 닫고 중국어 `“…”` 는 `“` 로 연다 — **"따옴표인가"는 안 뒤집힌다.**
# 그래서 여닫이를 가르려 하지 않고 따옴표를 통째로 뺀다. `Pe`(닫는 괄호)만 남겨
# 실측을 태우는데, 닫는 괄호는 어느 언어에서도 앞말에 붙는다.
_SHAPE_OPENING = {"Ps", "Pi", "Pf"}


def measure_profile(texts: list[str], min_count: int = 2, attach_ratio: float = 0.9) -> dict:
    """코퍼스에서 직접 재는 언어 표기 특성. LLM 을 쓰지 않는다.

    `trailing_punctuation` 의 정의는 **"앞 텍스트에 붙는 구두점"** 이다. 그래서
    문말 등장만 세면 안 된다 — 일본어 `、` 는 절 구분자라 문말에 안 나오지만 태그
    직후에 오면 안 되는 문자다. 대신 **거의 항상 비공백 문자 뒤에 붙어 나오는가**로
    판정한다.

    **판정은 두 갈래다. 모양이 확정하는 것은 분류로, 모호한 것만 실측으로.**

    실측 규칙("앞이 공백이 아닌가")은 **공백을 쓰는 언어에서만 신호가 있다.** 중국어
    실측: `“`(여는) 9회·`”`(닫는) 9회가 **둘 다 붙음 1.00** 이다 — 공백이 없으니
    여닫이가 구별되지 않는다. 그래서 zh 런 5개 전부 `“` 가 목록에 들어갔고,
    `normalize_tags` 가 여는 따옴표를 앞 조각 꼬리로 끌어당겼다 (실측 13건.
    `涂鸦活动和“ <SEG:6> 合法”涂鸦墙` — 인용어 한복판이 잘린다). `normalize_tags` 가
    조용히 고치는 경로라 검증기의 태그-뒤-구두점 검사는 v2 런 전체에서 **0건**이었다
    (그래서 그 검사는 지웠다 — `validate` 참조).

    반대로 실측이 꼭 필요한 글자도 있다 — 유니코드가 `Po` 로 뭉뚱그려 모양으로는
    못 가르는 것들이다:

        `"`  영어 17회 = 여는 8(앞이 공백) + 닫는 9(앞이 글자) -> 0.53. **일관되지
             않으므로** 제외한다. 강제로 옮기면 절반이 틀린다.
        `¿¡` 스페인어. 항상 문장 맨앞이거나 공백 뒤 -> 0.00 으로 제외된다.
             종전의 하드코딩 목록 `_PO_SENTENCE_OPENERS` 가 하던 일이고, 실측이
             이미 하고 있어 지웠다.

    **둘은 안 겹친다** — 실측이 눈머는 조건(공백 없음)과 실측이 꼭 필요한 글자
    (`"` `¿¡`)는 같은 언어에 안 나타난다. 그래서 나눠 맡기면 빈틈이 없다.

    닫는 따옴표를 더는 되돌리지 않는 것이 이 변경의 대가다. 모델이 스스로 제자리에
    놓는다는 근거는 있다 — 어느 목록에도 없어 한 번도 정규화되지 않은 `"` 가 새 조각을
    시작한 62건이 **62건 모두 여는 따옴표**였고, 독일어 `„` 19/19·영어 `“` 12/12 도
    같다. 다만 닫는 쪽은 늘 옮겨져 와서 자연 배치를 관측한 적이 없다. 바꾼 뒤
    `<SEG> ”` 패턴이 나오는지 확인할 것.
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
        if n >= min_count
        and unicodedata.category(c) not in _SHAPE_OPENING
        and attached.get(c, 0) / n >= attach_ratio
    )

    # 단위는 spaced 여부로 갈린다(어절 vs 문자). `pipeline.unit_count` 와 같은 규칙이라
    # 여기서 함께 정해야 판정과 단위가 어긋나지 않는다.
    spaced_ = space_ratio_median > 0.02
    return {
        "n": len(texts),
        "unit": "word" if spaced_ else "char",
        "space_ratio": round(space_ratio, 4),
        "space_ratio_median": round(space_ratio_median, 4),
        "space_sentence_ratio": round(space_sentence_ratio, 4),
        "uses_spaces_between_words": spaced_,
        "trailing_punctuation": trailing,
        "punctuation_counts": dict(sorted(seen.items(), key=lambda x: -x[1])),
        "final_punctuation_counts": dict(sorted(finals.items(), key=lambda x: -x[1])),
        "punctuation_final_rate": round(n_punct_final / len(texts), 4) if texts else 0.0,
    }


def profile_settings(measured: dict) -> tuple[bool, str | None]:
    """검증기·절단기가 쓸 `(spaced, trailing_punct)`. **측정값만 본다.**

    종전에는 LLM 프로파일과 대조해 불일치를 경고했는데, 측정값이 무조건 이기는 구조라
    경고는 로그 한 줄로 끝나고 아무 동작도 바꾸지 않았다.
    LLM 이 이 두 필드를 내놓을 이유도 없다 — 코퍼스에서 직접 세는 값이다.
    (Profiler 프롬프트에서 필드를 빼는 것은 prompt_v0 를 바꾸므로 별도 판단.)
    """
    return (bool(measured["uses_spaces_between_words"]),
            "".join(sorted(measured["trailing_punctuation"])) or None)


# ── 분할 ─────────────────────────────────────────────────────────────────

# **분할 시드는 config 에 기록된다** (`loop.py --seed`). 값 자체에 의미는 없지만,
# 바꾸면 train/dev/test 가 통째로 달라져 런 간 비교가 깨진다. 어떤 시드로 나눈
# 분할인지가 산출물에 안 남으면 재현이 불가능하다.
DEFAULT_SEED = 20260806


def stratified_order(items: list, seed: int, text_of=lambda x: x.text,
                     limit: int | None = None) -> list:
    """길이 3분위 × 구두점 유무로 층을 만들고 라운드로빈으로 섞는다.

    분할이 성격에 치우치지 않게 하는 장치다. 그냥 앞에서부터 자르면 test 에 짧은
    문장만, dev 에 긴 문장만 몰릴 수 있고, 그러면 점수가 프롬프트가 아니라 분할
    난이도를 재게 된다.

    **경계는 이 코퍼스에서 계산한다.** 종전에는 25/40자 고정이라 문자 밀도가 높은
    언어에서 층이 붕괴했다 — zh 는 중앙 38자라 첫 층(<25)이 거의 비고, ja 는 중앙
    52자라 거의 전부가 마지막 층(>=40)에 들어간다. 주석은 "3분위"라고 적혀 있었지만
    실제로는 고정값이었다. 실제 분위수를 쓰면 세 층이 항상 비슷한 크기가 되고
    상수 두 개가 사라진다.

    구두점 판정은 유니코드 범주로 한다 — 문자 목록을 박으면 층화가 언어 종속이 된다.

    `evaluation/ast/build_manifest_fleurs_text.py` 가 이 함수를 그대로 쓴다. 예전에는
    같은 규칙이 양쪽에 복사돼 있었다. **2026-08-24 이전에 만든 매니페스트는 고정
    경계(25/40)로 정렬된 것이라 재생성하면 순서가 달라진다.**
    """
    if not items:
        return []
    lens = sorted(len(text_of(x)) for x in items)
    lo, hi = lens[len(lens) // 3], lens[2 * len(lens) // 3]
    rng = random.Random(seed)
    strata: dict[tuple, list] = {}
    for x in items:
        t = text_of(x)
        bucket = 0 if len(t) < lo else (1 if len(t) < hi else 2)
        has_punct = any(unicodedata.category(ch).startswith("P") for ch in t)
        strata.setdefault((bucket, has_punct), []).append(x)
    for v in strata.values():
        rng.shuffle(v)

    # `limit` 만큼 뽑으면 멈춘다 — 분할이 앞에서부터 떼가므로 뒤쪽은 안 쓴다.
    # kspon-train(10000문장)에서 400개만 쓰는데 전량을 순서화하고 있었다.
    need = len(items) if limit is None else min(limit, len(items))
    keys = sorted(strata, key=lambda k: (-len(strata[k]), str(k)))
    order, idx = [], {k: 0 for k in keys}
    while len(order) < need:
        progressed = False
        for k in keys:
            i = idx[k]
            if i < len(strata[k]):
                order.append(strata[k][i]); idx[k] = i + 1; progressed = True
        if not progressed:
            break
    return order


def split_data(
    sentences: list[Sentence],
    n_train: int,
    n_dev: int,
    n_test: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[Sentence]]:
    """층화 후 라운드로빈으로 train/dev/test 를 겹치지 않게 뽑는다.

    **길이 하한 필터를 두지 않는다.** 짧은 문장을 거르는 선은 이미 `min_gap` 이 갖고
    있다 — `truncate` 는 양끝에서 각각 `min_gap` 이상 떨어진 자리에만 경계를 놓으므로
    `unit_count < 2*min_gap` 인 문장은 **구조적으로 무분절**이다. 예전에 있던 문자 수
    기준 하한은 그 선을 근사한 두 번째 절단선이었고, 단위가 달라 언어마다 어긋났다
    (en/ko 는 25자 ≈ 2*min_gap 으로 우연히 일치, zh 는 무분절 경계가 12자인데 25자에서
    잘려 240 중 34 문장이 과다 탈락했다).

    거를 이유도 사라졌다. 구조적 무분절 문장은 `coverage_need == 0` 이라 분절 호출을
    건너뛸 수 있고(결과가 호출 전에 확정), `effective` 가 프롬프트 불변 상수라 채택
    판정의 쌍체 차이에 **정확히 0** 을 기여한다.
    """
    pool = list(sentences)
    if len(pool) < n_train + n_dev + n_test:
        raise ValueError(
            f"문장 부족: 사용 가능 {len(pool)}개 < 요청 {n_train + n_dev + n_test}개"
        )

    order = stratified_order(pool, seed, limit=n_train + n_dev + n_test)

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
