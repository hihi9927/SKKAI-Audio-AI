"""라벨링 중간 결과를 본다. 캐시 키가 해시라 문장에서 역산해 되짚는다.

`segment_batch` 의 cache_key = key("seg4", prompt_hash, model, effort, batch_size, text).
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.meaning_segmentator.autoseg.pipeline import JsonCache, coverage_need

SEG = re.compile(r"<SEG(?::\d+)?>")

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--manifest", default="evaluation/ast/manifests/covost2_en-de_sample300.jsonl")
ap.add_argument("--effort", default="-")
ap.add_argument("--batch-size", type=int, default=6)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--show", type=int, default=8)
a = ap.parse_args()

def unwrap(t):
    t = t.strip()
    return t[1:-1].strip() if len(t) > 1 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2 else t

rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
if a.limit: rows = rows[: a.limit]
ph = JsonCache.key(Path(a.prompt).read_text(encoding="utf-8"))
data = json.loads(Path(a.cache).read_text(encoding="utf-8")) if Path(a.cache).exists() else {}

done, shown = 0, 0
for r in rows:
    t = unwrap(r["src_text"])
    k = JsonCache.key("seg4", ph, a.model, a.effort, str(a.batch_size), t)
    v = data.get(k)
    if not v: continue
    done += 1
    if shown < a.show:
        out = v[0] if isinstance(v, list) else v
        need = coverage_need(t, 4, True, 3)
        print(f"[{len(SEG.findall(out))}/{need}] {out[:150]}")
        shown += 1
print(f"\n완료 {done}/{len(rows)}  (캐시 항목 {len(data)})")
