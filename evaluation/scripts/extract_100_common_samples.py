#!/usr/bin/env python3
"""
Extract 100 random samples from common 1619 files and recalculate metrics
"""
import json
import random
from collections import defaultdict

# Configuration
RANDOM_SEED = 42
SAMPLE_SIZE = 100

# Input files
input_files = {
    'chunked': 'common/chunked-common-files.json',
    'original': 'common/original-common-files.json',
    'simul_meaning': 'common/simul_meaning_policy1-common-files.json',
    'simul_no_meaning': 'common/simul_no_meaning_policy1-common-files.json'
}

print(f"Extracting {SAMPLE_SIZE} random samples from common files (seed={RANDOM_SEED})")
print("="*70)

# Load all data
data = {}
for name, filepath in input_files.items():
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data[name] = json.load(f)
        print(f"Loaded {name}: {len(data[name]['raw_results'])} files")
    except Exception as e:
        print(f"Error loading {name}: {e}")
        exit(1)

# Get common file IDs from first dataset
all_file_ids = {r['file_id'] for r in data['chunked']['raw_results']}
print(f"\nTotal common files: {len(all_file_ids)}")

# Random sampling with fixed seed
random.seed(RANDOM_SEED)
file_ids_sorted = sorted(list(all_file_ids))
sampled_file_ids = set(random.sample(file_ids_sorted, SAMPLE_SIZE))

print(f"Sampled {len(sampled_file_ids)} files")
print(f"First sampled file: {min(sampled_file_ids)}")
print(f"Last sampled file: {max(sampled_file_ids)}")

# Filter and recalculate for each model
def filter_and_recalculate(result_data, sampled_ids):
    """Filter to sampled files and recalculate all metrics"""
    import jiwer

    if 'raw_results' not in result_data:
        return None

    # Filter to sampled files only
    filtered_results = [
        r for r in result_data['raw_results']
        if r.get('file_id') in sampled_ids
    ]

    print(f"    Filtered: {len(result_data['raw_results'])} -> {len(filtered_results)} files")

    if len(filtered_results) == 0:
        return None

    # Define text normalization
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip()
    ])

    # Filter out empty results
    valid_results = [
        r for r in filtered_results
        if r.get('reference', '').strip() and r.get('hypothesis', '').strip()
    ]

    if len(valid_results) == 0:
        return None

    # Calculate overall WER - apply transformation first
    references = [r['reference'] for r in valid_results]
    hypotheses = [r['hypothesis'] for r in valid_results]

    transformed_refs = [transformation(r) for r in references]
    transformed_hyps = [transformation(h) for h in hypotheses]

    # Filter out empty after transformation
    final_pairs = [(r, h, orig_r)
                   for r, h, orig_r in zip(transformed_refs, transformed_hyps, valid_results)
                   if r.strip() and h.strip()]

    if not final_pairs:
        return None

    final_refs = [p[0] for p in final_pairs]
    final_hyps = [p[1] for p in final_pairs]
    valid_results = [p[2] for p in final_pairs]

    overall_wer = jiwer.wer(final_refs, final_hyps)

    # Calculate overall averages
    first_token_latencies = [r['first_token_latency'] for r in valid_results
                            if r.get('first_token_latency') is not None]
    overall_first_token_latency = (sum(first_token_latencies) / len(first_token_latencies)
                                  if first_token_latencies else None)

    processing_times = [r.get('avg_processing_time') or r.get('total_time')
                       for r in valid_results
                       if r.get('avg_processing_time') or r.get('total_time')]
    overall_avg_processing_time = (sum(processing_times) / len(processing_times)
                                  if processing_times else None)

    # Calculate per-folder metrics
    folders = defaultdict(list)
    for r in valid_results:
        speaker_id = r.get('speaker_id')
        if speaker_id:
            folders[speaker_id].append(r)

    folder_stats = {}
    for speaker_id, folder_results in sorted(folders.items()):
        folder_refs = [r['reference'] for r in folder_results]
        folder_hyps = [r['hypothesis'] for r in folder_results]

        # Apply transformation and filter
        trans_refs = [transformation(r) for r in folder_refs]
        trans_hyps = [transformation(h) for h in folder_hyps]

        valid_pairs = [(r, h) for r, h in zip(trans_refs, trans_hyps) if r.strip() and h.strip()]
        if not valid_pairs:
            continue
        folder_refs, folder_hyps = zip(*valid_pairs)

        folder_wer = jiwer.wer(list(folder_refs), list(folder_hyps))

        folder_first_token = [r['first_token_latency'] for r in folder_results
                             if r.get('first_token_latency') is not None]
        avg_first_token = (sum(folder_first_token) / len(folder_first_token)
                          if folder_first_token else None)

        folder_proc_times = [r.get('avg_processing_time') or r.get('total_time')
                            for r in folder_results
                            if r.get('avg_processing_time') or r.get('total_time')]
        avg_proc_time = (sum(folder_proc_times) / len(folder_proc_times)
                        if folder_proc_times else None)

        folder_stats[speaker_id] = {
            'num_files': len(folder_results),
            'wer': folder_wer,
            'first_token_latency': avg_first_token,
            'avg_processing_time': avg_proc_time
        }

    return {
        'timestamp': result_data.get('timestamp', ''),
        'overall': {
            'num_files': len(valid_results),
            'wer': overall_wer,
            'first_token_latency': overall_first_token_latency,
            'avg_processing_time': overall_avg_processing_time
        },
        'folders': folder_stats,
        'raw_results': filtered_results,
        'sample_info': {
            'sample_size': SAMPLE_SIZE,
            'random_seed': RANDOM_SEED,
            'total_common_files': len(all_file_ids)
        }
    }

# Process each model
print("\n" + "="*70)
print("Filtering and recalculating metrics for 100 samples...")
print("="*70)

filtered_data = {}
for name, result_data in data.items():
    print(f"\nProcessing {name}...")
    filtered = filter_and_recalculate(result_data, sampled_file_ids)
    if filtered:
        filtered_data[name] = filtered
        print(f"  WER: {filtered['overall']['wer']:.4f}")
        print(f"  FTL: {filtered['overall']['first_token_latency']:.2f}s")
        print(f"  Files: {filtered['overall']['num_files']}")

# Save filtered results
print("\n" + "="*70)
print("Saving 100-sample results...")
print("="*70)

for name, filtered in filtered_data.items():
    output_file = f"100-common/{name}-100-samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_file}")

# Create comparison summary
summary = {
    'sample_info': {
        'sample_size': SAMPLE_SIZE,
        'random_seed': RANDOM_SEED,
        'total_common_files': len(all_file_ids),
        'sampled_files': sorted(list(sampled_file_ids))
    },
    'models': {}
}

for name, filtered in filtered_data.items():
    summary['models'][name] = {
        'overall': filtered['overall'],
        'num_speakers': len(filtered['folders'])
    }

with open('100-common/100-samples-summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("  Saved: 100-common/100-samples-summary.json")

# Save list of sampled file IDs
with open('100-common/sampled-file-ids.txt', 'w', encoding='utf-8') as f:
    for file_id in sorted(sampled_file_ids):
        f.write(f"{file_id}\n")
print("  Saved: 100-common/sampled-file-ids.txt")

print("\n" + "="*70)
print("SUMMARY - 100 Random Samples Comparison")
print("="*70)
print(f"Sample size: {SAMPLE_SIZE}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Total common files: {len(all_file_ids)}")
print(f"\nModel Performance (WER):")
for name in sorted(filtered_data.keys()):
    wer = filtered_data[name]['overall']['wer']
    ftl = filtered_data[name]['overall']['first_token_latency']
    proc = filtered_data[name]['overall']['avg_processing_time']
    print(f"  {name:20s}: WER={wer:.4f}, FTL={ftl:.2f}s, ProcTime={proc:.2f}s")

print("\nDone!")
