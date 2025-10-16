#!/usr/bin/env python3
"""
Simple script to run the test bench with proper setup.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_bench import TestBench


def main():
    """Run the test bench."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Email Agent Test Bench Runner")
    parser.add_argument("--attack", action="store_true", help="Run attack benchmark instead of test benchmark")
    
    args = parser.parse_args()
    
    # Determine which benchmark to run
    if args.attack:
        print("Email Agent Attack Benchmark - Pure Python Mode")
        test_dir = "attack_bench"
    else:
        print("Email Agent Test Bench - Pure Python Mode")
        test_dir = "test_bench"
    
    print("=" * 50)
    
    # Initialize test bench
    bench = TestBench()
    
    # Run all tests
    results = bench.run_all_tests(test_dir)
    
    if not results:
        print("No tests found or all tests failed to run.")
        return 1
    
    # Check if any tests failed
    failed_tests = [r for r in results if not r["overall_success"]]
    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) failed. Check the results directory for details.")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
