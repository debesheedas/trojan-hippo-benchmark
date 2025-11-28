#!/usr/bin/env python3
"""
Consolidate defense benchmark results into a CSV table.

This script parses result files from different defense configurations and creates
a consolidated table showing success rates at both step and test case granularity.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

def get_defense_results_dir(defense_type: str) -> Path:
    """Get the results directory for a defense type."""
    base_dir = BASE_DIR / "data" / "benchmark" / "test_bench_results_mem0"
    
    # Map "none" to "no_defense" for consistency
    if defense_type == "none":
        defense_folder = "no_defense"
    else:
        defense_folder = defense_type
    
    return base_dir / f"defense_{defense_folder}"

def parse_result_file(result_file: Path) -> Dict:
    """Parse a single result JSON file."""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not parse {result_file}: {e}")
        return None

def extract_user_goal_results(result: Dict) -> Tuple[int, int]:
    """
    Extract user goal success statistics from a test result.
    
    Returns:
        (passed_steps, total_steps) tuple
    """
    if not result or "steps" not in result:
        return (0, 0)
    
    passed = 0
    total = 0
    
    for step in result.get("steps", []):
        # Skip session start steps
        if step.get("step_type") == "start_new_session":
            continue
        
        # Check for "passed" field at step level (this indicates user goal success)
        if "passed" in step:
            total += 1
            if step["passed"] is True:
                passed += 1
        # Also check for user_goal if present
        elif "user_goal" in step:
            user_goal = step["user_goal"]
            if isinstance(user_goal, dict):
                if "passed" in user_goal:
                    total += 1
                    if user_goal["passed"] is True:
                        passed += 1
            elif isinstance(user_goal, bool):
                total += 1
                if user_goal:
                    passed += 1
    
    return (passed, total)

def collect_results_for_defense(defense_type: str) -> Tuple[List[Dict], int, int]:
    """
    Collect all results for a defense type.
    
    Returns:
        (test_results, total_passed_steps, total_steps) tuple
    """
    results_dir = get_defense_results_dir(defense_type)
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return ([], 0, 0)
    
    # Find all result JSON files
    result_files = list(results_dir.rglob("*.json"))
    
    # Filter to only get files in model_name/attack_type/ structure
    # We want files like: gpt-4o-mini/benign/00_email_tools.json
    test_results = []
    total_passed_steps = 0
    total_steps = 0
    
    # Only include specific test files we care about
    # For now: 00_email_tools.json and 02_tool_memory.json
    allowed_test_files = {"00_email_tools.json", "02_tool_memory.json"}
    
    for result_file in result_files:
        # Skip if not in expected structure (has model_name/attack_type/)
        parts = result_file.parts
        if len(parts) < 3:
            continue
        
        # Only process allowed test files
        if result_file.name not in allowed_test_files:
            continue
        
        # Check if it's a test result file (not a config or other file)
        if result_file.name.startswith("00_") or result_file.name.startswith("01_") or \
           any(result_file.name.startswith(f"{i:02d}_") for i in range(10)):
            result = parse_result_file(result_file)
            if result:
                test_results.append(result)
                passed, total = extract_user_goal_results(result)
                total_passed_steps += passed
                total_steps += total
    
    return (test_results, total_passed_steps, total_steps)

def calculate_test_case_success_rate(test_results: List[Dict]) -> Tuple[int, int]:
    """
    Calculate success rate at test case granularity.
    
    A test case is considered successful if overall_success is True.
    
    Returns:
        (passed_tests, total_tests) tuple
    """
    passed = 0
    total = len(test_results)
    
    for result in test_results:
        if result.get("overall_success") is True:
            passed += 1
    
    return (passed, total)

def main():
    """Main function to consolidate all defense results."""
    defense_types = [
        "disable_memory",
        "none",  # Will be mapped to "no_defense" in folder structure
        "user_only",
        "no_untrusted_tools"
    ]
    
    # Collect data for each defense
    defense_data = {}
    
    for defense_type in defense_types:
        print(f"Collecting results for defense: {defense_type}...")
        test_results, passed_steps, total_steps = collect_results_for_defense(defense_type)
        passed_tests, total_tests = calculate_test_case_success_rate(test_results)
        
        defense_data[defense_type] = {
            "test_results": test_results,
            "passed_steps": passed_steps,
            "total_steps": total_steps,
            "passed_tests": passed_tests,
            "total_tests": total_tests
        }
        
        print(f"  Found {len(test_results)} test cases")
        print(f"  Steps: {passed_steps}/{total_steps} passed")
        print(f"  Tests: {passed_tests}/{total_tests} passed")
    
    # Create CSV output
    output_file = BASE_DIR / "defense_benchmark_results.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Defense Type",
            "Step Success Rate",
            "Steps Passed",
            "Total Steps",
            "Step Utility (%)",
            "Test Case Success Rate",
            "Tests Passed",
            "Total Tests",
            "Test Case Utility (%)"
        ])
        
        # Write data for each defense
        for defense_type in defense_types:
            data = defense_data[defense_type]
            
            # Calculate step utility
            step_utility = (data["passed_steps"] / data["total_steps"] * 100) if data["total_steps"] > 0 else 0.0
            step_success_rate = f"{data['passed_steps']}/{data['total_steps']}"
            
            # Calculate test case utility
            test_utility = (data["passed_tests"] / data["total_tests"] * 100) if data["total_tests"] > 0 else 0.0
            test_success_rate = f"{data['passed_tests']}/{data['total_tests']}"
            
            # Display name: use "no_defense" for "none" in output
            display_name = "no_defense" if defense_type == "none" else defense_type
            
            writer.writerow([
                display_name,
                step_success_rate,
                data["passed_steps"],
                data["total_steps"],
                f"{step_utility:.2f}%",
                test_success_rate,
                data["passed_tests"],
                data["total_tests"],
                f"{test_utility:.2f}%"
            ])
    
    print(f"\n✅ Consolidated results saved to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("DEFENSE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Defense Type':<20} {'Step Utility':<15} {'Test Utility':<15}")
    print(f"{'-'*80}")
    for defense_type in defense_types:
        data = defense_data[defense_type]
        step_utility = (data["passed_steps"] / data["total_steps"] * 100) if data["total_steps"] > 0 else 0.0
        test_utility = (data["passed_tests"] / data["total_tests"] * 100) if data["total_tests"] > 0 else 0.0
        display_name = "no_defense" if defense_type == "none" else defense_type
        print(f"{display_name:<20} {step_utility:>6.2f}%{'':<8} {test_utility:>6.2f}%")

if __name__ == "__main__":
    main()

