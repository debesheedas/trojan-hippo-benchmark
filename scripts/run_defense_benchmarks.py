#!/usr/bin/env python3
"""
Run test bench for all defense configurations.

This script runs all test cases in the benign folder for each defense type:
1. disable_memory (baseline)
2. none (regular mem0, no defense)
3. user_only
4. no_untrusted_tools
"""

import sys
import subprocess
from pathlib import Path
import yaml

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

def update_config_defense(defense_type: str, config_path: str = "config.yaml"):
    """Update config.yaml with the specified defense type."""
    config_file = BASE_DIR / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update defense type
    if "memory" not in config:
        config["memory"] = {}
    if "mem0_memory" not in config["memory"]:
        config["memory"]["mem0_memory"] = {}
    config["memory"]["mem0_memory"]["defense_type"] = defense_type
    
    # Write back
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"✅ Updated config.yaml: defense_type = {defense_type}")

def cleanup_old_results(defense_type: str):
    """Clean up old result files for a specific defense type."""
    base_dir = BASE_DIR / "data" / "benchmark" / "test_bench_results_mem0"
    
    # Map "none" to "no_defense" for consistency
    if defense_type == "none":
        defense_folder = "no_defense"
    else:
        defense_folder = defense_type
    
    results_dir = base_dir / f"defense_{defense_folder}"
    if results_dir.exists():
        import shutil
        # Remove the entire defense directory to ensure clean slate
        print(f"🧹 Cleaning old results: {results_dir}")
        shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

def run_tests(defense_type: str, test_path: str):
    """Run test bench for a specific defense type."""
    print(f"\n{'='*80}")
    print(f"Running tests with defense: {defense_type}")
    print(f"{'='*80}\n")
    
    # Clean up old results first (but only for the specific test file)
    cleanup_old_results_for_test(defense_type, test_path)
    
    # Update config
    update_config_defense(defense_type)
    
    # Run test bench
    test_file = BASE_DIR / test_path
    cmd = [
        sys.executable,
        "-m", "src.benchmark.test_bench",
        "--test", str(test_file)
    ]
    
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✅ Completed tests for defense: {defense_type}")
    else:
        print(f"\n❌ Tests failed for defense: {defense_type}")
    
    return result.returncode == 0

def cleanup_old_results_for_test(defense_type: str, test_path: str):
    """Clean up old result files for a specific defense type and test file."""
    base_dir = BASE_DIR / "data" / "benchmark" / "test_bench_results_mem0"
    
    # Map "none" to "no_defense" for consistency
    if defense_type == "none":
        defense_folder = "no_defense"
    else:
        defense_folder = defense_type
    
    results_dir = base_dir / f"defense_{defense_folder}"
    
    # Get the test file name
    test_file = Path(test_path)
    test_file_name = test_file.stem + ".json"  # e.g., "02_tool_memory.json"
    
    # Find and remove only the specific test result file
    if results_dir.exists():
        result_files = list(results_dir.rglob(test_file_name))
        for result_file in result_files:
            print(f"🧹 Removing old result: {result_file}")
            result_file.unlink()

def main():
    """Run all defense benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run defense benchmarks")
    parser.add_argument("--test", type=str, help="Specific test file to run (e.g., data/benchmark/attack_bench_mem0/benign/02_tool_memory.json)")
    parser.add_argument("--folder", type=str, help="Test folder to run all tests from")
    
    args = parser.parse_args()
    
    if args.test:
        test_path = args.test
    elif args.folder:
        test_path = args.folder
    else:
        # Default: run all tests in benign folder
        test_path = "data/benchmark/attack_bench_mem0/benign"
    
    defense_types = [
        "disable_memory",
        "none",
        "user_only",
        "no_untrusted_tools"
    ]
    
    results = {}
    for defense_type in defense_types:
        success = run_tests(defense_type, test_path)
        results[defense_type] = success
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for defense_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{defense_type:20s}: {status}")
    
    print(f"\n✅ All defense benchmarks completed!")
    print(f"Results saved in: data/benchmark/test_bench_results_mem0/")

if __name__ == "__main__":
    main()

