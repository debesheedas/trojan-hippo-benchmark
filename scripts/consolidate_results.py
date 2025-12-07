#!/usr/bin/env python3
"""
Results Consolidation Script

Generates comprehensive CSV tables with:
- Memory models as columns (none, explicit, mem0, rag)
- Defense types as rows
- Steps passed/total and percentage for each combination
- Separate CSV files for each model and attack type (benign, direct, indirect)
- Empty cells for missing results

Only counts steps that:
- Are not session management steps (start_new_session, insert_attack_email)
- Have a success_check field defined (meaningful validation)

Usage:
    python scripts/consolidate_results.py
    python scripts/consolidate_results.py --results-dir data/benchmark/results
    python scripts/consolidate_results.py --attack-type benign
    python scripts/consolidate_results.py --model gpt-5-mini
"""

import json
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.defense_backend import get_defense_backend_registry, UNIFIED_DEFENSE_TYPES


def get_results_dir(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    results_base_dir: Path = Path("data/benchmark/results")
) -> Path:
    """
    Get the results directory for a memory backend, defense type, model, and attack type.
    
    Structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none")
        unified_defense: Unified defense name
        model_name: Model name
        attack_type: Attack type ("benign", "direct", "indirect")
        results_base_dir: Base directory for results
        
    Returns:
        Path to results directory
    """
    # Handle no memory backend
    if memory_backend == "none":
        backend_for_path = "none"
        defense_folder = "none"  # No memory = no defense
    else:
        backend_for_path = memory_backend
        # Map unified defense to backend-specific defense
        defense_registry = get_defense_backend_registry()
        backend_defense = defense_registry.map_defense(memory_backend, unified_defense)
        
        # Normalize defense folder name
        if memory_backend == "mem0" and backend_defense == "no_defense":
            defense_folder = "no_defense"
        else:
            defense_folder = backend_defense
    
    return results_base_dir / model_name / backend_for_path / defense_folder / attack_type


def parse_result_file(result_file: Path) -> Optional[Dict]:
    """Parse a result JSON file and extract relevant statistics."""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Extract step statistics
        # Only count steps that:
        # 1. Are not session management steps (start_new_session, insert_attack_email)
        # 2. Have a success_check field defined (meaningful validation)
        steps = result.get("steps", [])
        test_steps = [
            step for step in steps 
            if (step.get("step_type") not in ["start_new_session", "insert_attack_email"]
                and step.get("success_check") is not None)
        ]
        total_steps = len(test_steps)
        passed_steps = sum(1 for step in test_steps if step.get("passed") is True)
        
        # Extract user goal statistics (only for steps with success_check)
        user_goal_passed = 0
        user_goal_total = 0
        
        for step in test_steps:
            user_goal = step.get("user_goal")
            if user_goal:
                if isinstance(user_goal, dict):
                    user_goal_total += 1
                    if user_goal.get("passed") is True:
                        user_goal_passed += 1
                elif isinstance(user_goal, bool):
                    user_goal_total += 1
                    if user_goal:
                        user_goal_passed += 1
        
        return {
            "test_name": result.get("test_name", "Unknown"),
            "test_file": result.get("test_file", ""),
            "overall_success": result.get("overall_success", False),
            "total_steps": total_steps,
            "passed_steps": passed_steps,
            "user_goal_passed": user_goal_passed,
            "user_goal_total": user_goal_total,
            "memory_backend": result.get("memory_backend", "unknown"),
            "defense_type": result.get("defense_type", "unknown"),
            "attack_type": result.get("attack_type", "unknown"),
        }
    except Exception as e:
        print(f"⚠️  Warning: Could not parse {result_file}: {e}")
        return None


def collect_results_for_combination(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    results_base_dir: Path = Path("data/benchmark/results")
) -> Tuple[int, int]:
    """
    Collect step statistics for a specific memory backend, defense, model, and attack type.
    
    Returns:
        (passed_steps, total_steps) tuple
    """
    results_dir = get_results_dir(memory_backend, unified_defense, model_name, attack_type, results_base_dir)
    
    if not results_dir.exists():
        return (0, 0)  # No results available
    
    # Find all result JSON files
    result_files = list(results_dir.glob("*.json"))
    
    total_passed_steps = 0
    total_steps = 0
    
    for result_file in result_files:
        # Check if it's a test result file (named with numbered prefixes)
        if re.match(r'^\d{2}_', result_file.name):
            result = parse_result_file(result_file)
            if result:
                total_passed_steps += result["passed_steps"]
                total_steps += result["total_steps"]
    
    return (total_passed_steps, total_steps)


def generate_consolidated_csv(
    model_name: str,
    attack_type: str,
    results_base_dir: Path,
    output_dir: Path
) -> None:
    """
    Generate a consolidated CSV for a specific model and attack type.
    
    Table structure:
    - Rows: Defense types (none, user_only, no_untrusted_tools, limit_memory_length)
    - Columns: Defense Type, none (steps), none (%), explicit (steps), explicit (%), mem0 (steps), mem0 (%), rag (steps), rag (%)
    
    Note: The "none" column represents results with no memory backend enabled (memory_backend="none", defense_type="none")
    """
    memory_backends = ["none", "explicit", "mem0", "rag"]
    defense_types = UNIFIED_DEFENSE_TYPES
    
    # Collect data for all combinations
    data = {}
    for defense_type in defense_types:
        data[defense_type] = {}
        for backend in memory_backends:
            # For "none" backend, only collect when defense_type is "none" (no memory = no defense)
            if backend == "none":
                if defense_type == "none":
                    passed, total = collect_results_for_combination(
                        "none", "none", model_name, attack_type, results_base_dir
                    )
                    data[defense_type][backend] = (passed, total)
                else:
                    data[defense_type][backend] = (0, 0)  # Not applicable (can't have defenses without memory)
            else:
                # For actual memory backends, collect results for the defense type
                # Skip "none" defense type for memory backends (they use their own "none" equivalent)
                if defense_type == "none":
                    # For "none" defense, use backend-specific "none" equivalent
                    # explicit: "none", mem0: "no_defense", rag: "none"
                    passed, total = collect_results_for_combination(
                        backend, defense_type, model_name, attack_type, results_base_dir
                    )
                    data[defense_type][backend] = (passed, total)
                else:
                    passed, total = collect_results_for_combination(
                        backend, defense_type, model_name, attack_type, results_base_dir
                    )
                    data[defense_type][backend] = (passed, total)
    
    # Generate CSV
    output_file = output_dir / f"{model_name}_{attack_type}_consolidated.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ["Defense Type"]
        for backend in memory_backends:
            backend_label = "none" if backend == "none" else backend
            header.extend([f"{backend_label} (steps)", f"{backend_label} (%)"])
        writer.writerow(header)
        
        # Data rows
        for defense_type in defense_types:
            row = [defense_type]
            for backend in memory_backends:
                passed, total = data[defense_type][backend]
                
                # Format steps: "X/Y" or "-" if no data
                if total > 0:
                    steps_str = f"{passed}/{total}"
                    percentage = (passed / total * 100) if total > 0 else 0.0
                    percentage_str = f"{percentage:.1f}%"
                else:
                    steps_str = "-"
                    percentage_str = "-"
                
                row.extend([steps_str, percentage_str])
            
            writer.writerow(row)
    
    print(f"✅ Generated: {output_file}")


def discover_models_and_attack_types(results_base_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Discover all models and attack types from the results directory structure.
    
    Returns:
        (models, attack_types) tuple
    """
    models = set()
    attack_types = set()
    
    if not results_base_dir.exists():
        return ([], [])
    
    # Structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/
    for model_dir in results_base_dir.iterdir():
        if model_dir.is_dir():
            models.add(model_dir.name)
            
            # Look for attack types in subdirectories
            for backend_dir in model_dir.iterdir():
                if backend_dir.is_dir():
                    for defense_dir in backend_dir.iterdir():
                        if defense_dir.is_dir():
                            for attack_dir in defense_dir.iterdir():
                                if attack_dir.is_dir() and attack_dir.name in ["benign", "direct", "indirect"]:
                                    attack_types.add(attack_dir.name)
    
    return (sorted(list(models)), sorted(list(attack_types)))


def main():
    """Main function to consolidate all results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidate all benchmark results into comprehensive CSV tables"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/benchmark/results",
        help="Base directory for results"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark/consolidated_results",
        help="Output directory for consolidated CSV files"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to consolidate (default: all models)"
    )
    
    parser.add_argument(
        "--attack-type",
        type=str,
        choices=["benign", "direct", "indirect"],
        help="Specific attack type to consolidate (default: all attack types)"
    )
    
    args = parser.parse_args()
    
    results_base_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover models and attack types
    all_models, all_attack_types = discover_models_and_attack_types(results_base_dir)
    
    if not all_models:
        print(f"⚠️  No models found in {results_base_dir}")
        return 1
    
    if not all_attack_types:
        print(f"⚠️  No attack types found in {results_base_dir}")
        return 1
    
    # Filter based on arguments
    models_to_process = [args.model] if args.model else all_models
    attack_types_to_process = [args.attack_type] if args.attack_type else all_attack_types
    
    print(f"\n{'='*80}")
    print(f"Consolidating Results")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models_to_process)}")
    print(f"Attack Types: {', '.join(attack_types_to_process)}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Generate CSV for each model and attack type combination
    for model_name in models_to_process:
        for attack_type in attack_types_to_process:
            print(f"Processing: {model_name} / {attack_type}")
            generate_consolidated_csv(model_name, attack_type, results_base_dir, output_dir)
    
    print(f"\n✅ Consolidation complete!")
    print(f"📊 Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

