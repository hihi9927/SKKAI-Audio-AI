import json
import asyncio
import aiohttp
import warnings
warnings.filterwarnings("ignore")

from comet import download_model, load_from_checkpoint

RESULTS = {
    "1.0.0_r5": "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/fcl_1.0.0_results_5.json",
    "1.0.0_r6": "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/fcl_1.0.0_results_6/qwen3_test_other_fcl.json",
    "1.0.1_r7": "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/fcl_1.0.1_results_7.json",
    "1.0.1_r8": "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/fcl_1.0.1_results_8.json",
    "1.0.1_r9": "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/fcl_1.0.1_results_9.json",
}

async def google_translate_async(session, text, target_lang="ko"):
    if not text.strip():
        return ""
    try:
        params = {"client": "gtx", "sl": "auto", "tl": target_lang,
                  "dt": "t", "q": text}
        async with session.get(
            "https://translate.googleapis.com/translate_a/single",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as resp:
            data = await resp.json(content_type=None)
            return "".join(item[0] for item in data[0] if item and item[0])
    except Exception as e:
        print(f"  [translate error] {e}")
        return ""

async def translate_all_references(raw_results):
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for r in raw_results:
            ref_en = r.get("reference", "").strip().lower().capitalize()
            tasks.append(google_translate_async(session, ref_en))
        return await asyncio.gather(*tasks)

print("Loading COMET model (wmt20-comet-qe-da with reference)...")
model_path = download_model("Unbabel/wmt20-comet-da")
model = load_from_checkpoint(model_path)
print("Model loaded.\n")

all_results = {}

for name, path in RESULTS.items():
    print(f"=== {name} ===")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    raw = d["policy_3"]["raw_results"]

    print(f"  Translating {len(raw)} full-sentence references via Google Translate...")
    ref_translations = asyncio.run(translate_all_references(raw))

    data = []
    for r, ref_ko in zip(raw, ref_translations):
        src_en = r.get("reference", "").strip().lower().capitalize()
        hyp_ko = " ".join(
            s.get("translation", "").strip()
            for s in r.get("segment_metrics", [])
            if s.get("translation", "").strip()
        )
        if src_en and hyp_ko and ref_ko:
            data.append({"src": src_en, "mt": hyp_ko, "ref": ref_ko})

    print(f"  files with valid data: {len(data)}")
    output = model.predict(data, batch_size=64, gpus=1, progress_bar=True)
    scores = output.scores
    avg = sum(scores) / len(scores)
    sys_score = output.system_score

    all_results[name] = {
        "n_files": len(data),
        "avg_comet_da": round(avg, 4),
        "system_score": round(sys_score, 4),
    }
    print(f"  avg_comet_da: {avg:.4f}")
    print(f"  system_score: {sys_score:.4f}\n")

print("=== SUMMARY ===")
for name, r in all_results.items():
    print(f"{name}: n={r['n_files']}  avg_comet_da={r['avg_comet_da']}  sys={r['system_score']}")

out_path = "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/comet_seg_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
