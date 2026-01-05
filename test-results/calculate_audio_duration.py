#!/usr/bin/env python3
"""
Calculate total audio duration for 100 sampled files
"""
import json

# Load one of the 100-sample files (they all have the same files)
with open('100-common/chunked-100-samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

raw_results = data['raw_results']

# Calculate total duration
total_duration_seconds = 0
durations = []

for result in raw_results:
    duration = result.get('duration')
    if duration is not None:
        total_duration_seconds += duration
        durations.append(duration)

# Statistics
num_files = len(durations)
avg_duration = total_duration_seconds / num_files if num_files > 0 else 0
min_duration = min(durations) if durations else 0
max_duration = max(durations) if durations else 0

# Convert to minutes and hours
total_minutes = total_duration_seconds / 60
total_hours = total_duration_seconds / 3600

print("="*70)
print("Audio Duration Analysis for 100 Sampled Files")
print("="*70)
print(f"\nNumber of files: {num_files}")
print(f"\nTotal duration:")
print(f"  {total_duration_seconds:.2f} seconds")
print(f"  {total_minutes:.2f} minutes")
print(f"  {total_hours:.2f} hours")
print(f"\nAverage duration per file:")
print(f"  {avg_duration:.2f} seconds")
print(f"\nDuration range:")
print(f"  Min: {min_duration:.2f} seconds")
print(f"  Max: {max_duration:.2f} seconds")

# Duration distribution
short_files = len([d for d in durations if d < 5])
medium_files = len([d for d in durations if 5 <= d < 10])
long_files = len([d for d in durations if d >= 10])

print(f"\nDuration distribution:")
print(f"  < 5 seconds:  {short_files} files ({short_files/num_files*100:.1f}%)")
print(f"  5-10 seconds: {medium_files} files ({medium_files/num_files*100:.1f}%)")
print(f"  >= 10 seconds: {long_files} files ({long_files/num_files*100:.1f}%)")

# Save to summary
duration_summary = {
    'num_files': num_files,
    'total_duration_seconds': total_duration_seconds,
    'total_duration_minutes': total_minutes,
    'total_duration_hours': total_hours,
    'average_duration_seconds': avg_duration,
    'min_duration_seconds': min_duration,
    'max_duration_seconds': max_duration,
    'distribution': {
        'short_files_under_5s': short_files,
        'medium_files_5_to_10s': medium_files,
        'long_files_over_10s': long_files
    }
}

with open('100-common/audio-duration-summary.json', 'w', encoding='utf-8') as f:
    json.dump(duration_summary, f, indent=2, ensure_ascii=False)

print(f"\nSaved: 100-common/audio-duration-summary.json")
print("="*70)
