#!/usr/bin/env python3
"""
Cleanup Old Test Environments

Removes old test environment directories that are no longer needed.
Test environments are created for each test run and should be cleaned up
immediately after the test completes. This script removes any that were
left behind (e.g., from crashed processes or interrupted runs).

Usage:
    # Clean up environments older than 24 hours (default)
    python scripts/cleanup_test_envs.py
    
    # Clean up environments older than 1 hour
    python scripts/cleanup_test_envs.py --max-age-hours 1
    
    # Dry run (show what would be deleted without actually deleting)
    python scripts/cleanup_test_envs.py --dry-run
    
    # Clean up all test environments (use with caution!)
    python scripts/cleanup_test_envs.py --max-age-hours 0
"""

import argparse
import sys
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.benchmark_utils import cleanup_old_test_environments


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up old test environment directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum age in hours before cleanup (default: 24). Use 0 to clean up all."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "--test-envs-dir",
        type=str,
        default="data/benchmark/test_envs",
        help="Base directory containing test environments (default: data/benchmark/test_envs)"
    )
    
    args = parser.parse_args()
    
    # Validate max_age_hours
    if args.max_age_hours < 0:
        parser.error("--max-age-hours must be >= 0")
    
    if args.max_age_hours == 0:
        print("⚠️  WARNING: --max-age-hours 0 will delete ALL test environments!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
    
    test_envs_dir = Path(args.test_envs_dir)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print()
    
    # Clean up old test environments
    deleted_count, failed_count = cleanup_old_test_environments(
        base_dir=test_envs_dir,
        max_age_hours=args.max_age_hours,
        dry_run=args.dry_run
    )
    
    # Exit with appropriate code
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

