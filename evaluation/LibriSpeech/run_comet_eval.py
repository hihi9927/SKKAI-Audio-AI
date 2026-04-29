import json
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

print("Loading COMET-QE model (wmt20-comet-qe-da)...")
model_path = download_model("Unbabel/wmt20-comet-qe-da")
model = load_from_checkpoint(model_path)
print("Model loaded.\n")

all_results = {}

for name, path in RESULTS.items():
    print(f"=== {name} ===")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    raw = d["policy_3"]["raw_results"]

    data = []
    for r in raw:
        for s in r.get("segment_metrics", []):
            src = s.get("text", "").strip()
            mt = s.get("translation", "").strip()
            if src and mt:
                data.append({"src": src, "mt": mt})

    print(f"  segments: {len(data)}")
    output = model.predict(data, batch_size=128, gpus=1, progress_bar=True)
    scores = output.scores
    seg_score = sum(scores) / len(scores)
    sys_score = output.system_score

    all_results[name] = {
        "n_segments": len(data),
        "avg_segment_score": round(seg_score, 4),
        "system_score": round(sys_score, 4),
    }
    print(f"  avg_segment_score: {seg_score:.4f}")
    print(f"  system_score:      {sys_score:.4f}\n")

print("=== SUMMARY ===")
for name, r in all_results.items():
    print(f"{name}: n={r['n_segments']}  avg_seg={r['avg_segment_score']}  sys={r['system_score']}")

out_path = "/home/ubuntu/STiTy/evaluation/LibriSpeech/results/fcl/comet_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
