#!/usr/bin/env python3
"""
Recalculate WER from whisperlivekit_results-old.json with proper text normalization
"""
import json
import jiwer
from collections import defaultdict

def recalculate_wer(input_file, output_file):
    """Recalculate WER with text normalization"""

    # Load the old results
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Text normalization function
    def normalize_text(text):
        """Normalize text: lowercase, remove punctuation, strip"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        return text.strip()

    # Process each policy
    for policy_key, policy_data in data.items():
        if 'raw_results' not in policy_data:
            print(f"No raw_results found for {policy_key}")
            continue

        results = policy_data['raw_results']

        # Filter valid results
        valid_results = [r for r in results if r.get('reference', '').strip() and r.get('hypothesis', '').strip()]

        if len(valid_results) == 0:
            print(f"No valid results for {policy_key}")
            continue

        # Calculate overall WER
        all_references = [r['reference'] for r in valid_results]
        all_hypotheses = [r['hypothesis'] for r in valid_results]

        # Normalize texts
        final_refs = []
        final_hyps = []
        for idx, (ref, hyp) in enumerate(zip(all_references, all_hypotheses)):
            ref_norm = normalize_text(ref)
            hyp_norm = normalize_text(hyp)
            if ref_norm and hyp_norm:
                final_refs.append(ref_norm)
                final_hyps.append(hyp_norm)
            else:
                print(f"  Skipping empty result {idx}: ref='{ref}' -> '{ref_norm}', hyp='{hyp}' -> '{hyp_norm}'")

        if len(final_refs) == 0:
            print(f"All results became empty after normalization for {policy_key}")
            continue

        print(f"Processing {len(final_refs)} valid results for {policy_key}")

        if len(final_refs) == 0:
            print(f"No valid results remaining for {policy_key}")
            continue

        # Calculate WER with normalized texts (no additional transformation needed)
        overall_wer = jiwer.wer(final_refs, final_hyps)

        # Calculate per-folder WER
        folders = defaultdict(list)
        for r in valid_results:
            speaker_id = r['speaker_id']
            folders[speaker_id].append(r)

        folder_stats = {}
        for speaker_id, folder_results in sorted(folders.items()):
            references = [normalize_text(r['reference']) for r in folder_results]
            hypotheses = [normalize_text(r['hypothesis']) for r in folder_results]

            # Filter out empty normalized results
            valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r and h]
            if not valid_pairs:
                continue
            references, hypotheses = zip(*valid_pairs)

            folder_wer = jiwer.wer(list(references), list(hypotheses))

            # Calculate average metrics
            first_token_latencies = [r['first_token_latency'] for r in folder_results if r.get('first_token_latency') is not None]
            avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else None

            processing_times = [r['avg_processing_time'] for r in folder_results if r.get('avg_processing_time') is not None]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else None

            folder_stats[speaker_id] = {
                'num_files': len(folder_results),
                'wer': folder_wer,
                'first_token_latency': avg_first_token_latency,
                'avg_processing_time': avg_processing_time
            }

        # Calculate overall averages
        all_first_token_latencies = [r['first_token_latency'] for r in valid_results if r.get('first_token_latency') is not None]
        overall_first_token_latency = sum(all_first_token_latencies) / len(all_first_token_latencies) if all_first_token_latencies else None

        all_processing_times = [r['avg_processing_time'] for r in valid_results if r.get('avg_processing_time') is not None]
        overall_avg_processing_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else None

        # Update the policy data
        policy_data['overall']['wer'] = overall_wer
        policy_data['overall']['first_token_latency'] = overall_first_token_latency
        policy_data['overall']['avg_processing_time'] = overall_avg_processing_time
        policy_data['folders'] = folder_stats

        print(f"\n{policy_key}:")
        print(f"  Overall WER: {overall_wer:.4f} (old: {policy_data['overall'].get('wer', 'N/A')})")
        print(f"  Files: {len(valid_results)}")
        print(f"  Avg first token latency: {overall_first_token_latency:.3f}s")
        print(f"  Avg processing time: {overall_avg_processing_time:.3f}s")

    # Save updated results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nRecalculated results saved to {output_file}")

if __name__ == "__main__":
    recalculate_wer(
        'whisperlivekit_results2.json',
        'simul_results_meaning.json'
    )
