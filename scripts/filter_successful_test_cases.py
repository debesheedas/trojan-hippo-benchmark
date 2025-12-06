#!/usr/bin/env python3
"""
Filter test case to keep only successful steps based on results file.

Usage:
    python scripts/filter_successful_test_cases.py \
        --test-case data/benchmark/attack_bench_mem0/benign/01_persona_memory.json \
        --results data/benchmark/test_bench_results_mem0/defense_no_defense/gpt-5-mini/benign/01_persona_memory.json \
        --output data/benchmark/attack_bench_mem0/benign/01_persona_memory_filtered.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_successful_step_numbers(results: Dict[str, Any]) -> List[int]:
    """
    Extract step numbers that passed from results file.
    
    Returns list of step numbers (1-indexed) that passed.
    """
    steps = results.get('steps', [])
    successful_steps = []
    
    for step_result in steps:
        step_number = step_result.get('step')
        passed = step_result.get('passed', False)
        
        if passed:
            successful_steps.append(step_number)
            print(f"  ✓ Step {step_number}: PASSED")
        else:
            print(f"  ✗ Step {step_number}: FAILED")
    
    return sorted(successful_steps)


def filter_test_case(test_case: Dict[str, Any], successful_step_numbers: List[int]) -> Dict[str, Any]:
    """
    Filter test case to keep only successful steps.
    
    Args:
        test_case: Original test case dictionary
        successful_step_numbers: List of step numbers (1-indexed) that passed
    
    Returns:
        Filtered test case dictionary
    """
    filtered_case = test_case.copy()
    original_steps = test_case.get('steps', [])
    
    # Create mapping of step_number to step index
    # Steps in test case may or may not have explicit step_number field
    filtered_steps = []
    
    for idx, step in enumerate(original_steps, start=1):
        # Check if step has explicit step_number field
        if 'step_number' in step:
            step_num = step['step_number']
        else:
            # Use index (1-indexed) as step number
            step_num = idx
        
        if step_num in successful_step_numbers:
            filtered_steps.append(step)
    
    filtered_case['steps'] = filtered_steps
    
    return filtered_case


def main():
    parser = argparse.ArgumentParser(description='Filter test case to keep only successful steps')
    parser.add_argument('--test-case', type=str, required=True,
                       help='Path to original test case JSON file')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output filtered test case JSON file')
    
    args = parser.parse_args()
    
    test_case_path = Path(args.test_case)
    results_path = Path(args.results)
    output_path = Path(args.output)
    
    # Validate files exist
    if not test_case_path.exists():
        print(f"❌ Error: Test case file not found: {test_case_path}")
        return 1
    
    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        print(f"   Please run the test first to generate results.")
        return 1
    
    print(f"📖 Loading test case: {test_case_path}")
    test_case = load_json_file(test_case_path)
    original_step_count = len(test_case.get('steps', []))
    print(f"   Original test case has {original_step_count} steps")
    
    print(f"\n📊 Loading results: {results_path}")
    results = load_json_file(results_path)
    
    print(f"\n🔍 Analyzing results...")
    successful_step_numbers = get_successful_step_numbers(results)
    
    print(f"\n📈 Summary:")
    print(f"   Total steps in test case: {original_step_count}")
    print(f"   Total steps in results: {len(results.get('steps', []))}")
    print(f"   Successful steps: {len(successful_step_numbers)}")
    print(f"   Failed steps: {original_step_count - len(successful_step_numbers)}")
    print(f"   Success rate: {len(successful_step_numbers)/original_step_count*100:.1f}%")
    
    if not successful_step_numbers:
        print(f"\n⚠️  Warning: No successful steps found! Not creating filtered test case.")
        return 1
    
    print(f"\n✂️  Filtering test case...")
    filtered_case = filter_test_case(test_case, successful_step_numbers)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write filtered test case
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_case, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created filtered test case: {output_path}")
    print(f"   Kept {len(filtered_case.get('steps', []))} successful steps")
    print(f"   Removed {original_step_count - len(filtered_case.get('steps', []))} failed steps")
    
    return 0


if __name__ == '__main__':
    exit(main())

