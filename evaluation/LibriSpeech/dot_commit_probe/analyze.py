"""probe JSONL 리플레이 — naive(현재 브랜치) vs gate(안 A) 비교."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import DotCommitGate, DOT_COMMIT_BOUNDARY_RE

PUNCT = re.compile(r"[^\w\s']")


def norm(s):
    return PUNCT.sub(" ", s.upper()).split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sentence_latencies(rec, commits, gap=1.0):
    """발화(utterance)별 전달 지연 = 그 발화 마지막 단어가 커밋된 시각 - 발화 끝난 시각.

    concat 스트림에서만 의미 있음 (marks가 2개 이상). gap은 발화 사이 무음 길이.
    """
    marks = rec.get("utterance_marks") or []
    if len(marks) < 2:
        return []
    ends = []
    for i, (start, _) in enumerate(marks):
        if i + 1 < len(marks):
            ends.append(marks[i + 1][0] - gap)
        else:
            ends.append(rec["duration_sec"] - gap)

    # 참조 단어 -> 그 단어가 실린 커밋 시각. 편집거리 정렬로 매핑(삭제/삽입 내성).
    ref_words, ref_owner = [], []
    for i, (_, ref_text) in enumerate(marks):
        w = norm(ref_text)
        ref_words += w
        ref_owner += [i] * len(w)
    hyp_words, hyp_time = [], []
    for sec, text, _ in commits:
        w = norm(text)
        hyp_words += w
        hyp_time += [sec] * len(w)
    if not hyp_words:
        return []

    times = align_ref_to_hyp_times(ref_words, hyp_words, hyp_time)
    lats = []
    for i in range(len(marks)):
        # 발화 i의 마지막으로 매칭된 참조 단어가 전달된 시각
        ts = [times[j] for j in range(len(ref_words)) if ref_owner[j] == i and times[j] is not None]
        if ts:
            lats.append(max(ts) - ends[i])
    return lats


def align_ref_to_hyp_times(ref, hyp, hyp_time):
    """Levenshtein 정렬 backtrace로 ref 단어별 전달 시각을 구한다."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    times = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            times[i - 1] = hyp_time[j - 1]  # match 또는 substitution
            i -= 1
            j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1  # deletion: 전달 안 됨
        else:
            j -= 1  # insertion
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--show", type=int, default=0, help="상세 출력할 파일 수")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    count_tokens = lambda s: len(tok.encode(s)) if s else 0

    recs = [json.loads(l) for l in open(args.jsonl)]

    surv_total = surv_kept = 0
    agg = {}
    for mode in ("naive", "gate"):
        agg[mode] = dict(commits=0, err=0, words=0, ftl=[], tail=[], conflicts=0, empty=0,
                         reasons={}, sent_lat=[])

    for ri, r in enumerate(recs):
        chunks = r["chunks"]
        # --- 청크 끝 온점 생존율 ---
        for i in range(len(chunks) - 1):
            h, nxt = chunks[i]["text"], chunks[i + 1]["text"]
            if not h or not re.search(r"[.?!。？！]$", h.strip()):
                continue
            surv_total += 1
            if nxt.startswith(h.strip()):
                surv_kept += 1

        ref = norm(r["reference"])
        for mode in ("naive", "gate"):
            g = DotCommitGate(count_tokens, args.unfixed_token_num, naive=(mode == "naive"))
            for c in chunks:
                if c["text"]:
                    g.feed(c["text"], c["audio_sec"])
            g.finish(r["final_text"], chunks[-1]["audio_sec"] if chunks else 0.0)

            hyp = norm(" ".join(t for _, t, _ in g.commits))
            a = agg[mode]
            a["commits"] += len(g.commits)
            a["err"] += edit_distance(ref, hyp)
            a["words"] += len(ref)
            a["conflicts"] += g.conflicts
            for _, _, why in g.commits:
                a["reasons"][why] = a["reasons"].get(why, 0) + 1
            if not g.commits:
                a["empty"] += 1
            else:
                a["ftl"].append(g.commits[0][0])
                a["tail"].append(g.commits[-1][0] - r["duration_sec"])
            a["sent_lat"] += sentence_latencies(r, g.commits)

            if ri < args.show:
                if mode == "naive":
                    print("=" * 78)
                    print(f'{r["file"]} dur={r["duration_sec"]}s')
                    print("  REF :", r["reference"])
                print(f"  {mode:5} ({len(g.commits)} commits, conflicts={g.conflicts}):")
                for sec, t, why in g.commits:
                    print(f"    @{sec:>6.2f}s [{why:12}] {t!r}")

    n = len(recs)
    print("\n" + "=" * 78)
    print(f"files={n}  chunk-terminal dots={surv_total}  survived next chunk={surv_kept} "
          f"({100*surv_kept/max(1,surv_total):.1f}%)")
    print(f'{"mode":6} {"commits":>8} {"cmt/strm":>9} {"WER%":>7} {"FTL(s)":>8} {"tail(s)":>8} '
          f'{"sentLat":>8} {"conflict":>9} {"empty":>6}')
    for mode in ("naive", "gate"):
        a = agg[mode]
        wer = 100 * a["err"] / max(1, a["words"])
        ftl = sum(a["ftl"]) / max(1, len(a["ftl"]))
        tail = sum(a["tail"]) / max(1, len(a["tail"]))
        slat = sum(a["sent_lat"]) / len(a["sent_lat"]) if a["sent_lat"] else float("nan")
        print(f'{mode:6} {a["commits"]:>8} {a["commits"]/n:>9.2f} {wer:>7.2f} {ftl:>8.2f} {tail:>8.2f} '
              f'{slat:>8.2f} {a["conflicts"]:>9} {a["empty"]:>6}')
    for mode in ("naive", "gate"):
        print(f'  {mode:5} reasons: {agg[mode]["reasons"]}')


if __name__ == "__main__":
    main()
