#!/usr/bin/env python3
"""
Command-line utility to validate cache integrity.

This script validates that all cached attack benchmark files only differ
from their original counterparts in the attack_emails attribute.
"""

import sys
from pathlib import Path
import os

# Ensure src/ is on sys.path for package imports (repo root = parent of useful_temp)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.attack_utils import validate_cache_integrity, print_cache_validation_report


def main():
    """Main entry point for cache validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate attack benchmark cache integrity")
    parser.add_argument("--cache-dir", default="data/benchmark/attack_bench/train_cache",
                       help="Directory containing cached files (default: data/benchmark/attack_bench/train_cache)")
    parser.add_argument("--original-dir", default="data/benchmark/attack_bench/train",
                       help="Directory containing original train files (default: data/benchmark/attack_bench/train)")
    parser.add_argument("--quiet", action="store_true", 
                       help="Only show summary, no detailed report")
    
    args = parser.parse_args()
    
    # Validate cache integrity
    validation_result = validate_cache_integrity(args.cache_dir, args.original_dir)
    
    if args.quiet:
        print(validation_result['summary'])
        return 0 if validation_result['invalid_files'] == 0 else 1
    else:
        print_cache_validation_report(validation_result)
        return 0 if validation_result['invalid_files'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
