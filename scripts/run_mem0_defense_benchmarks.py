#!/usr/bin/env python3
"""
Run test bench for all mem0 defense configurations.

This script runs all test cases in the benign folder for each mem0 defense type:
1. disable_memory (baseline - mem0 disabled)
2. none (regular mem0, no defense)
3. user_only (only index user messages)
4. no_untrusted_tools (disable mem0 indexing for the rest of the session once
   an untrusted tool has been used)
5. limit_memory_length (truncate extracted mem0 memories to 80 characters)

The script passes defense_type as a command-line argument to test_bench,
avoiding modification of the global config.yaml file. This allows safe
parallel execution of multiple instances without race conditions.
"""

import sys
import subprocess
import json
import yaml
import copy
import time
import os
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


def load_config(config_path: str = "config.yaml"):
    """Load config file to get model name."""
    try:
        from agent.utils import load_config as load_config_util
        return load_config_util(config_path)
    except Exception:
        # Fallback: return default model name
        return {"agent": {"target_model_name": "gpt-5-mini"}}


def check_result_exists(defense_type: str, test_file_path: Path, model_name: str) -> bool:
    """
    Check if a result file already exists for a given test file and defense type.
    
    Args:
        defense_type: Defense type (e.g., "none", "user_only")
        test_file_path: Path to the test file
        model_name: Model name from config
        
    Returns:
        True if result exists, False otherwise
    """
    base_dir = BASE_DIR / "data" / "benchmark" / "test_bench_results_mem0"
    
    # Map "none" to "no_defense" for consistency
    if defense_type == "none":
        defense_folder = "no_defense"
    else:
        defense_folder = defense_type
    
    # Get attack_type from test file
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        attack_type = test_data.get("attack_type", "benign")
    except Exception:
        # Fallback: try to infer from path
        if "benign" in str(test_file_path):
            attack_type = "benign"
        elif "direct" in str(test_file_path):
            attack_type = "direct"
        elif "indirect" in str(test_file_path):
            attack_type = "indirect"
        else:
            attack_type = "benign"
    
    # Construct expected result file path
    results_dir = base_dir / f"defense_{defense_folder}" / model_name / attack_type
    result_file = results_dir / f"{test_file_path.stem}.json"
    
    return result_file.exists()


def should_skip_defense(defense_type: str, test_path: str, model_name: str, force: bool) -> tuple[bool, list[str]]:
    """
    Check if we should skip running tests for a defense type.
    
    Args:
        defense_type: Defense type to check
        test_path: Path to test file or directory
        model_name: Model name from config
        force: If True, don't skip even if results exist
        
    Returns:
        (should_skip, list_of_skipped_tests) tuple
    """
    if force:
        return False, []
    
    test_path_obj = BASE_DIR / test_path
    
    skipped_tests = []
    
    if test_path_obj.is_file():
        # Single test file
        if check_result_exists(defense_type, test_path_obj, model_name):
            return True, [str(test_path_obj)]
    elif test_path_obj.is_dir():
        # Directory: check all JSON test files
        test_files = list(test_path_obj.glob("*.json"))
        all_exist = True
        for test_file in test_files:
            if check_result_exists(defense_type, test_file, model_name):
                skipped_tests.append(str(test_file))
            else:
                all_exist = False
        
        if all_exist and test_files:
            return True, skipped_tests
    
    return False, []


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

def run_tests(defense_type: str, test_path: str, model_name: str, force: bool = False):
    """Run test bench for a specific defense type."""
    print(f"\n{'='*80}")
    print(f"Running mem0 tests with defense: {defense_type}")
    print(f"{'='*80}\n")
    
    # Check if we should skip this defense
    should_skip, skipped_tests = should_skip_defense(defense_type, test_path, model_name, force)
    
    if should_skip:
        print(f"⏭️  Skipping defense '{defense_type}' - results already exist")
        if skipped_tests:
            print(f"   Found existing results for {len(skipped_tests)} test(s)")
            if len(skipped_tests) <= 5:
                for test in skipped_tests:
                    print(f"     - {Path(test).name}")
            else:
                for test in skipped_tests[:3]:
                    print(f"     - {Path(test).name}")
                print(f"     ... and {len(skipped_tests) - 3} more")
        print(f"   Use --force to overwrite existing results")
        return True  # Return True (success) since we're skipping intentionally
    
    # Clean up old results first (only if force is True)
    if force:
        cleanup_old_results_for_test(defense_type, test_path)
    
    # Create a temporary config file with mem0_memory enabled
    # This ensures test_bench uses the mem0 results directory
    # We use a temp file to avoid race conditions when running multiple instances in parallel
    
    # Load original config
    original_config = load_config()
    
    # Create a deep copy with mem0 enabled and other memory systems disabled
    temp_config = copy.deepcopy(original_config)
    if "memory" not in temp_config:
        temp_config["memory"] = {}
    
    # Ensure mem0_memory is enabled
    if "mem0_memory" not in temp_config["memory"]:
        temp_config["memory"]["mem0_memory"] = {}
    temp_config["memory"]["mem0_memory"]["enabled"] = True
    
    # Disable other memory systems to avoid conflicts
    if "explicit_memory" not in temp_config["memory"]:
        temp_config["memory"]["explicit_memory"] = {}
    temp_config["memory"]["explicit_memory"]["enabled"] = False
    
    if "rag_memory" not in temp_config["memory"]:
        temp_config["memory"]["rag_memory"] = {}
    temp_config["memory"]["rag_memory"]["enabled"] = False
    
    # Create temporary config file with unique name to avoid conflicts in parallel runs
    # Use process ID and timestamp to ensure uniqueness
    unique_suffix = f"_{os.getpid()}_{int(time.time() * 1000000)}"
    temp_config_path = BASE_DIR / f".temp_config_mem0{unique_suffix}.yaml"
    
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(temp_config, f, default_flow_style=False, sort_keys=False)
    
    try:
        # Run test bench with defense_type as argument and temporary config
        # The temp config ensures mem0 is enabled and results go to the correct directory
        test_file = BASE_DIR / test_path
        cmd = [
            sys.executable,
            "-m", "src.benchmark.test_bench",
            "--test", str(test_file),
            "--defense-type", defense_type,
            "--config", str(temp_config_path)
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
    if test_file.is_file():
        test_file_name = test_file.stem + ".json"  # e.g., "02_tool_memory.json"
        
        # Find and remove only the specific test result file
        if results_dir.exists():
            result_files = list(results_dir.rglob(test_file_name))
            for result_file in result_files:
                print(f"🧹 Removing old result: {result_file}")
                result_file.unlink()
    elif test_file.is_dir():
        # If it's a directory, remove all results in that directory
        if results_dir.exists():
            import shutil
            print(f"🧹 Cleaning old results: {results_dir}")
            shutil.rmtree(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

def main():
    """Run all mem0 defense benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mem0 defense benchmarks")
    parser.add_argument("--test", type=str, help="Specific test file to run (e.g., data/benchmark/attack_bench_mem0/benign/02_tool_memory.json)")
    parser.add_argument("--folder", type=str, help="Test folder to run all tests from")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing results (default: skip if results exist)")
    parser.add_argument(
        "--defense-type",
        type=str,
        help="Specific defense type to run (e.g., 'disable_memory', 'none'). If not specified, runs all defenses.",
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_path = args.test
    elif args.folder:
        test_path = args.folder
    else:
        # Default: run all tests in benign folder
        test_path = "data/benchmark/attack_bench_mem0/benign"
    
    # Load config to get model name
    config = load_config()
    model_name = config.get("agent", {}).get("target_model_name", "gpt-5-mini")
    
    all_defense_types = [
        "disable_memory",
        "none",
        "user_only",
        "no_untrusted_tools",
        "limit_memory_length",
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
    skipped = {}
    for defense_type in defense_types:
        success = run_tests(defense_type, test_path, model_name, force=args.force)
        results[defense_type] = success
        # Track if it was skipped (success=True but we didn't actually run)
        should_skip, _ = should_skip_defense(defense_type, test_path, model_name, args.force)
        if should_skip and not args.force:
            skipped[defense_type] = True
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for defense_type, success in results.items():
        if defense_type in skipped:
            status = "⏭️  SKIPPED"
        elif success:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{defense_type:20s}: {status}")
    
    if skipped and not args.force:
        print(f"\n💡 Tip: Use --force to overwrite existing results")
    
    print(f"\n✅ All mem0 defense benchmarks completed!")
    print(f"Results saved in: data/benchmark/test_bench_results_mem0/")

if __name__ == "__main__":
    main()

