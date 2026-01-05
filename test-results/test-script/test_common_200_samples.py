#!/usr/bin/env python3
"""
Test WhisperLiveKit with 200 random samples from common files

This script demonstrates how to use the new --common-files and --random-sample options
to test on a subset of common files that are present across all models.

Usage:
    python test_common_200_samples.py
"""
import subprocess
import sys
import os

# Configuration
COMMON_FILES_JSON = r"c:\Users\VRSTUDIO3\Desktop\STiTy-main1\test-results\chunked-common-files.json"
TEST_DIR = r"c:\Users\VRSTUDIO3\Desktop\STiTy-main1\LibriSpeech\test-other"
OUTPUT_FILE = "whisperlivekit_common_200_results.json"
RANDOM_SEED = 42  # Fixed seed for reproducibility
SAMPLE_SIZE = 200  # Number of files to test
WS_HOST = "localhost"
WS_PORT = 8001

def main():
    # Check if common files JSON exists
    if not os.path.exists(COMMON_FILES_JSON):
        print(f"ERROR: Common files JSON not found: {COMMON_FILES_JSON}")
        print("Please run extract_common_files.py first to generate common files list")
        sys.exit(1)

    # Test policy 1 (SimulStreaming with meaning segmentation)
    print("\n" + "="*70)
    print("Testing Policy 1: SimulStreaming with meaning segmentation")
    print("="*70)

    cmd_policy1 = [
        sys.executable,
        "test_whisperlivekit.py",
        "--test-dir", TEST_DIR,
        "--policy", "1",
        "--host", WS_HOST,
        "--port", str(WS_PORT),
        "--output", OUTPUT_FILE,
        "--common-files", COMMON_FILES_JSON,
        "--random-sample", str(SAMPLE_SIZE),
        "--random-seed", str(RANDOM_SEED),
        "--calculate-wer",
        "--log-level", "INFO"
    ]

    print(f"Command: {' '.join(cmd_policy1)}\n")

    result = subprocess.run(cmd_policy1)

    if result.returncode != 0:
        print(f"\nERROR: Policy 1 testing failed with return code {result.returncode}")
        sys.exit(1)

    print("\n" + "="*70)
    print("Policy 1 testing completed successfully!")
    print("="*70)

    # Optionally test policy 2 (uncomment to enable)
    # print("\n" + "="*70)
    # print("Testing Policy 2: LocalAgreement")
    # print("="*70)
    #
    # cmd_policy2 = cmd_policy1.copy()
    # cmd_policy2[cmd_policy2.index("1")] = "2"  # Change policy to 2
    #
    # result = subprocess.run(cmd_policy2)
    #
    # if result.returncode != 0:
    #     print(f"\nERROR: Policy 2 testing failed with return code {result.returncode}")
    #     sys.exit(1)
    #
    # print("\n" + "="*70)
    # print("Policy 2 testing completed successfully!")
    # print("="*70)

    print(f"\n✓ All testing completed!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"\nTested {SAMPLE_SIZE} randomly sampled files (seed={RANDOM_SEED})")
    print(f"Source common files: {COMMON_FILES_JSON}")

if __name__ == "__main__":
    main()
