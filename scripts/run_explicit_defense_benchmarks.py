#!/usr/bin/env python3
"""
Run test bench for all explicit_memory defense configurations.

This script runs all test cases in the benign folder for each defense type:
1. disable_memory (baseline)
2. none (regular explicit memory, no defense)
3. user_prompt_only (uses user_only_memory_prompt.txt)
4. limit_memory_length (truncates update_memory.memory_text to 80 chars)
5. no_untrusted_tools (disables explicit memory indexing for the rest of the
   session after any untrusted tool is used)

We pass defense_type as a command-line argument to test_bench and rely on
the explicit_memory config + agent_core logic, without mutating config.yaml.
"""

import sys
import subprocess
import yaml
import copy
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


def cleanup_old_results_for_test(defense_type: str, test_path: str):
    """Clean up old result files for a specific defense type and test file."""
    base_dir = BASE_DIR / "data" / "benchmark" / "test_bench_results_explicit"

    # Map "none" to "no_defense" for consistency
    if defense_type == "none":
        defense_folder = "no_defense"
    else:
        defense_folder = defense_type

    results_dir = base_dir / f"defense_{defense_folder}"

    # Get the test file name (or directory pattern)
    test_file = Path(test_path)
    if test_file.is_dir() or str(test_path).endswith("/benign"):
        # Folder: remove all JSONs under this defense folder
        if results_dir.exists():
            for result_file in results_dir.rglob("*.json"):
                result_file.unlink()
        return

    test_file_name = test_file.stem + ".json"

    if results_dir.exists():
        result_files = list(results_dir.rglob(test_file_name))
        for result_file in result_files:
            print(f"🧹 Removing old result: {result_file}")
            result_file.unlink()


def run_tests(defense_type: str, test_path: str) -> bool:
    """Run test bench for a specific defense type."""
    print(f"\n{'='*80}")
    print(f"Running explicit_memory tests with defense: {defense_type}")
    print(f"{'='*80}\n")

    cleanup_old_results_for_test(defense_type, test_path)

    # Create a temporary config file with explicit_memory enabled
    # This ensures test_bench uses the explicit memory results directory
    # We use a temp file to avoid race conditions when running multiple instances in parallel
    
    # Load original config
    from agent.utils import load_config
    original_config = load_config()
    
    # Create a deep copy with explicit_memory enabled and other memory systems disabled
    temp_config = copy.deepcopy(original_config)
    if "memory" not in temp_config:
        temp_config["memory"] = {}
    
    # Ensure explicit_memory is enabled
    if "explicit_memory" not in temp_config["memory"]:
        temp_config["memory"]["explicit_memory"] = {}
    temp_config["memory"]["explicit_memory"]["enabled"] = True
    
    # Disable other memory systems to avoid conflicts
    if "rag_memory" not in temp_config["memory"]:
        temp_config["memory"]["rag_memory"] = {}
    temp_config["memory"]["rag_memory"]["enabled"] = False
    
    if "mem0_memory" not in temp_config["memory"]:
        temp_config["memory"]["mem0_memory"] = {}
    temp_config["memory"]["mem0_memory"]["enabled"] = False
    
    # Create temporary config file with unique name to avoid conflicts in parallel runs
    unique_suffix = f"_{os.getpid()}_{int(time.time() * 1000000)}"
    temp_config_path = BASE_DIR / f".temp_config_explicit{unique_suffix}.yaml"
    
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(temp_config, f, default_flow_style=False, sort_keys=False)
    
    try:
        # Run test bench with defense_type as argument and temporary config
        # The temp config ensures explicit_memory is enabled and results go to the correct directory
        cmd = [
            sys.executable,
            "-m",
            "src.benchmark.test_bench",
            "--test",
            str(test_path),
            "--defense-type",
            defense_type,
            "--config",
            str(temp_config_path),
        ]

        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=False)

        if result.returncode == 0:
            print(f"\n✅ Completed tests for defense: {defense_type}")
        else:
            print(f"\n❌ Tests failed for defense: {defense_type}")

        return result.returncode == 0
    finally:
        # Clean up temporary config file
        try:
            Path(temp_config_path).unlink()
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run explicit_memory defense benchmarks")
    parser.add_argument(
        "--test",
        type=str,
        help="Specific test file or folder to run "
             "(e.g., data/benchmark/attack_bench_explicit/benign/00_email_tools.json or "
             "data/benchmark/attack_bench_explicit/benign)",
    )
    parser.add_argument(
        "--defense-type",
        type=str,
        help="Specific defense type to run (e.g., 'disable_memory', 'none'). If not specified, runs all defenses.",
    )

    args = parser.parse_args()

    if args.test:
        test_path = args.test
    else:
        # Default: run all tests in explicit benign folder
        test_path = "data/benchmark/attack_bench_explicit/benign"

    all_defense_types = [
        "disable_memory",
        "none",
        "user_prompt_only",
        "limit_memory_length",
        "no_untrusted_tools",
    ]

    # If specific defense type requested, run only that one
    if args.defense_type:
        if args.defense_type not in all_defense_types:
            print(f"❌ Error: Unknown defense type '{args.defense_type}'")
            print(f"   Valid defense types: {', '.join(all_defense_types)}")
            return
        defense_types = [args.defense_type]
    else:
        defense_types = all_defense_types

    results = {}
    for defense_type in defense_types:
        success = run_tests(defense_type, test_path)
        results[defense_type] = success

    print(f"\n{'='*80}")
    print("SUMMARY (explicit_memory defenses)")
    print(f"{'='*80}")
    for defense_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{defense_type:20s}: {status}")

    print(
        "\nCompleted explicit_memory defense benchmarks. "
        "Results saved under data/benchmark/test_bench_results_explicit/."
    )


if __name__ == "__main__":
    main()


