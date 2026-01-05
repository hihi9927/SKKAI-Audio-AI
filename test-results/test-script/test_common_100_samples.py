#!/usr/bin/env python3
"""
Test WhisperLiveKit with random samples from common files

This script demonstrates how to use the new --common-files and --random-sample options
to test on a subset of common files that are present across all models.

Usage:
    python test_common_100_samples.py --policy 1 --sample-size 10
    python test_common_100_samples.py --policy 2 --sample-size 100 --output localagreement_results.json
"""
import subprocess
import sys
import os
import argparse

# Default configuration
DEFAULT_COMMON_FILES_JSON = r"c:\Users\VRSTUDIO3\Desktop\STiTy-main1\test-results\common\chunked-common-files.json"
DEFAULT_TEST_DIR = r"c:\Users\VRSTUDIO3\Desktop\STiTy-main1\LibriSpeech\test-other"
DEFAULT_RANDOM_SEED = 42
DEFAULT_WS_HOST = "localhost"
DEFAULT_WS_PORT = 8001

def main():
    parser = argparse.ArgumentParser(
        description='Test WhisperLiveKit with random samples from common files'
    )

    parser.add_argument('--policy', type=str, required=True,
                        choices=['1', '2', 'simulstreaming', 'localagreement'],
                        help='Policy to test: 1/simulstreaming or 2/localagreement')
    parser.add_argument('--sample-size', type=int, default=10,
                        help='Number of files to test (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: auto-generated based on policy)')
    parser.add_argument('--common-files', type=str, default=DEFAULT_COMMON_FILES_JSON,
                        help='Path to common files JSON')
    parser.add_argument('--test-dir', type=str, default=DEFAULT_TEST_DIR,
                        help='Path to test directory')
    parser.add_argument('--host', type=str, default=DEFAULT_WS_HOST,
                        help='WebSocket server host')
    parser.add_argument('--port', type=int, default=DEFAULT_WS_PORT,
                        help='WebSocket server port')
    parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level (default: INFO)')

    args = parser.parse_args()

    # Normalize policy name
    policy = args.policy
    if policy == 'simulstreaming':
        policy = '1'
    elif policy == 'localagreement':
        policy = '2'

    # Auto-generate output filename if not specified
    if args.output is None:
        policy_name = 'simulstreaming' if policy == '1' else 'localagreement'
        args.output = f"{policy_name}_{args.sample_size}_results.json"

    # Check if common files JSON exists
    if not os.path.exists(args.common_files):
        print(f"ERROR: Common files JSON not found: {args.common_files}")
        print("Please run extract_common_files.py first to generate common files list")
        sys.exit(1)

    # Get policy name for display
    policy_display = "SimulStreaming" if policy == '1' else "LocalAgreement"

    # Test selected policy
    print("\n" + "="*70)
    print(f"Testing Policy {policy}: {policy_display}")
    print("="*70)

    cmd = [
        sys.executable,
        "test_whisperlivekit.py",
        "--test-dir", args.test_dir,
        "--policy", policy,
        "--host", args.host,
        "--port", str(args.port),
        "--output", args.output,
        "--common-files", args.common_files,
        "--random-sample", str(args.sample_size),
        "--random-seed", str(args.random_seed),
        "--calculate-wer",
        "--log-level", args.log_level
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Policy {policy} testing failed with return code {result.returncode}")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"Policy {policy} testing completed successfully!")
    print("="*70)

    print(f"\n✓ Testing completed!")
    print(f"Results saved to: {args.output}")
    print(f"\nTested {args.sample_size} randomly sampled files (seed={args.random_seed})")
    print(f"Source common files: {args.common_files}")

if __name__ == "__main__":
    main()
