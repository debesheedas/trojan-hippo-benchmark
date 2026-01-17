#!/usr/bin/env python3
"""
Compare benchmark results against ground truth.

This script reads the results from a benchmark run and compares them
against the expected results in ground_truth.json.
"""

import json
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.benchmark_utils import get_result_path, determine_attack_type
from agent.utils import load_config


def load_ground_truth(ground_truth_path: Path) -> dict:
    """Load ground truth from JSON file."""
    with open(ground_truth_path, 'r') as f:
        return json.load(f)


def get_actual_result(
    memory_backend: str,
    defense_type: str,
    model_name: str,
    test_file: Path,
    results_base_dir: Path
) -> bool:
    """
    Get actual test result from benchmark output.
    
    Returns:
        True if test passed, False otherwise
    """
    # Determine attack type
    with open(test_file, 'r') as f:
        test_def = json.load(f)
    attack_type = determine_attack_type(test_file, test_def)
    
    # Get result file path
    result_path = get_result_path(
        memory_backend=memory_backend,
        unified_defense=defense_type,
        model_name=model_name,
        attack_type=attack_type,
        test_file=test_file,
        results_base_dir=results_base_dir
    )
    
    # Read result file
    if not result_path.exists():
        print(f"  WARNING: Result file not found: {result_path}")
        return False
    
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    return result.get("overall_success", False)


def compare_results(
    ground_truth_path: Path,
    results_base_dir: Path,
    test_file: Path,
    model_name: str
) -> tuple[bool, list]:
    """
    Compare actual results against ground truth.
    
    Returns:
        (all_match, failures) where failures is a list of mismatches
    """
    ground_truth = load_ground_truth(ground_truth_path)
    
    memory_backends = ["none", "explicit", "mem0", "rag", "context"]
    defense_types = ["none", "user_prompt_only", "no_untrusted_tools", "limit_memory_length", "provable_policy"]
    
    failures = []
    
    for memory_backend in memory_backends:
        for defense_type in defense_types:
            # Skip limit_memory_length for context backend (not applicable)
            if memory_backend == "context" and defense_type == "limit_memory_length":
                continue
            
            # Get expected result
            expected = ground_truth[memory_backend][defense_type]
            
            # Get actual result
            try:
                actual = get_actual_result(
                    memory_backend=memory_backend,
                    defense_type=defense_type,
                    model_name=model_name,
                    test_file=test_file,
                    results_base_dir=results_base_dir
                )
            except Exception as e:
                print(f"  ERROR getting result for {memory_backend}/{defense_type}: {e}")
                actual = None
            
            # Compare
            if actual != expected:
                failures.append({
                    "combination": f"{memory_backend}/{defense_type}",
                    "expected": expected,
                    "actual": actual
                })
                print(f"  ❌ {memory_backend}/{defense_type}: expected {expected}, got {actual}")
            else:
                print(f"  ✓ {memory_backend}/{defense_type}: {'PASS' if actual else 'FAIL'} (matches expected)")
    
    return len(failures) == 0, failures


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare benchmark results against ground truth")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=BASE_DIR / "CI-tests" / "ground_truth.json",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE_DIR / "CI-tests" / "results",
        help="Base directory for benchmark results"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=BASE_DIR / "CI-tests" / "testcases" / "test1.json",
        help="Path to test file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (used to find results). If not provided, reads from agent_config.yaml"
    )
    
    args = parser.parse_args()
    
    # Get model name from config if not provided
    if args.model is None:
        try:
            agent_config = load_config("agent_config.yaml")
            args.model = agent_config.get("agent", {}).get("target_model_name", "gpt-4o-mini")
        except Exception:
            args.model = "gpt-4o-mini"  # Fallback default
    
    print(f"Comparing results for: {args.test_file.name}")
    print(f"Model: {args.model}")
    print(f"Results directory: {args.results_dir}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"\n{'='*80}\n")
    
    all_match, failures = compare_results(
        ground_truth_path=args.ground_truth,
        results_base_dir=args.results_dir,
        test_file=args.test_file,
        model_name=args.model
    )
    
    print(f"\n{'='*80}")
    if all_match:
        print("✓ SUCCESS: All results match ground truth!")
        return 0
    else:
        print(f"✗ FAILURE: {len(failures)} combination(s) do not match ground truth:")
        for failure in failures:
            print(f"  {failure['combination']}: expected {failure['expected']}, got {failure['actual']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
