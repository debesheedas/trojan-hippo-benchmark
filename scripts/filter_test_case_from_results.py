#!/usr/bin/env python3
"""
Automatically filter test case based on results file.
Finds the results file automatically based on test case path and config.

Usage:
    python scripts/filter_test_case_from_results.py \
        --test-case data/benchmark/attack_bench_mem0/benign/01_persona_memory.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: try to parse as JSON-like structure
        print("⚠️  yaml module not available, using default paths")
        return {}


def find_results_file(test_case_path: Path, config: Dict[str, Any]) -> Optional[Path]:
    """
    Find the corresponding results file for a test case.
    
    Results are stored in: data/benchmark/test_bench_results_mem0/defense_{defense_type}/{model_name}/{attack_type}/{test_file_name}.json
    """
    # Get model name and defense type from config
    model_name = config.get("agent", {}).get("target_model_name", "gpt-5-mini")
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    defense_type = mem0_config.get("defense_type", "none")
    
    # Map "none" to "no_defense"
    if defense_type == "none":
        defense_folder = "no_defense"
    else:
        defense_folder = defense_type
    
    # Get attack type from test case
    with open(test_case_path, 'r') as f:
        test_case = json.load(f)
    attack_type = test_case.get("attack_type", "benign")
    
    # Construct results path
    test_file_name = test_case_path.name
    results_path = Path(f"data/benchmark/test_bench_results_mem0/defense_{defense_folder}/{model_name}/{attack_type}/{test_file_name}")
    
    return results_path


def get_successful_step_numbers(results: Dict[str, Any]) -> List[int]:
    """Extract step numbers that passed from results file."""
    steps = results.get('steps', [])
    successful_steps = []
    
    print(f"\n📋 Step Results:")
    for step_result in steps:
        step_number = step_result.get('step')
        passed = step_result.get('passed', False)
        user_msg = step_result.get('user_message', '')[:60] + '...' if len(step_result.get('user_message', '')) > 60 else step_result.get('user_message', '')
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Step {step_number:2d}: {status} - {user_msg}")
        
        if passed:
            successful_steps.append(step_number)
    
    return sorted(successful_steps)


def filter_test_case(test_case: Dict[str, Any], successful_step_numbers: List[int]) -> Dict[str, Any]:
    """Filter test case to keep only successful steps."""
    filtered_case = test_case.copy()
    original_steps = test_case.get('steps', [])
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
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results JSON file (auto-detected if not provided)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output filtered test case (default: adds _filtered suffix)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    test_case_path = Path(args.test_case)
    
    if not test_case_path.exists():
        print(f"❌ Error: Test case file not found: {test_case_path}")
        return 1
    
    # Load config
    config = load_config(args.config)
    
    # Find results file
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = find_results_file(test_case_path, config)
        if results_path is None:
            print(f"❌ Error: Could not determine results file path")
            return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Add _filtered suffix before .json
        stem = test_case_path.stem
        output_path = test_case_path.parent / f"{stem}_filtered.json"
    
    print(f"📖 Test Case: {test_case_path}")
    print(f"📊 Results File: {results_path}")
    print(f"💾 Output File: {output_path}")
    
    if not results_path.exists():
        print(f"\n❌ Error: Results file not found: {results_path}")
        print(f"   Please run the test first:")
        print(f"   python src/benchmark/test_bench.py --test {test_case_path}")
        return 1
    
    # Load files
    print(f"\n📖 Loading test case...")
    with open(test_case_path, 'r', encoding='utf-8') as f:
        test_case = json.load(f)
    original_step_count = len(test_case.get('steps', []))
    
    print(f"📊 Loading results...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Get successful steps
    successful_step_numbers = get_successful_step_numbers(results)
    
    print(f"\n📈 Summary:")
    print(f"   Total steps in test case: {original_step_count}")
    print(f"   Total steps in results: {len(results.get('steps', []))}")
    print(f"   Successful steps: {len(successful_step_numbers)}")
    print(f"   Failed steps: {original_step_count - len(successful_step_numbers)}")
    if original_step_count > 0:
        print(f"   Success rate: {len(successful_step_numbers)/original_step_count*100:.1f}%")
    
    if not successful_step_numbers:
        print(f"\n⚠️  Warning: No successful steps found! Not creating filtered test case.")
        return 1
    
    # Filter test case
    print(f"\n✂️  Filtering test case...")
    filtered_case = filter_test_case(test_case, successful_step_numbers)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_case, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created filtered test case: {output_path}")
    print(f"   Kept {len(filtered_case.get('steps', []))} successful steps")
    print(f"   Removed {original_step_count - len(filtered_case.get('steps', []))} failed steps")
    print(f"\n   Successful step numbers: {successful_step_numbers}")
    
    return 0


if __name__ == '__main__':
    exit(main())

