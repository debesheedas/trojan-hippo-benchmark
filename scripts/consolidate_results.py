#!/usr/bin/env python3
"""
Results Consolidation Script with CSV and Plots

Generates comprehensive CSV tables and visualizations for each test suite:
- CSV files with success percentages for each memory backend and defense type
- Bar charts and heatmaps showing performance across configurations
- Separate files for each suite: benign, direct, indirect, memory_only

Usage:
    python scripts/consolidate_results.py
    python scripts/consolidate_results.py --results-dir data/benchmark/results
    python scripts/consolidate_results.py --suite benign
    python scripts/consolidate_results.py --model gpt-5-mini
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Try to import plotting libraries (optional)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
except ImportError:
    PLOTTING_AVAILABLE = False

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.defense_backend import UNIFIED_DEFENSE_TYPES
from benchmark.benchmark_utils import get_results_dir


def parse_result_file(result_file: Path) -> Optional[Dict]:
    """Parse a result JSON file and extract step statistics."""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Use total_user_steps and total_successful_user_steps if available
        # Otherwise, calculate from steps
        if "total_user_steps" in result and "total_successful_user_steps" in result:
            total_steps = result["total_user_steps"]
            passed_steps = result["total_successful_user_steps"]
        else:
            # Fallback: count steps with success_check
            steps = result.get("steps", [])
            test_steps = [
                step for step in steps 
                if (step.get("step_type") not in ["start_new_session", "insert_attack_email"]
                    and (step.get("success_check") is not None 
                         or step.get("user_goal") is not None
                         or step.get("attack_goal") is not None))
            ]
            total_steps = len(test_steps)
            passed_steps = sum(1 for step in test_steps if step.get("passed") is True)
        
        return {
            "test_name": result.get("test_name", "Unknown"),
            "total_steps": total_steps,
            "passed_steps": passed_steps,
            "success_rate": (passed_steps / total_steps * 100) if total_steps > 0 else 0.0,
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
) -> Tuple[int, int, float]:
    """
    Collect step statistics for a specific memory backend, defense, model, and attack type.
    
    Returns:
        (passed_steps, total_steps, success_rate) tuple
    """
    results_dir = get_results_dir(memory_backend, unified_defense, model_name, attack_type, results_base_dir)
    
    if not results_dir.exists():
        return (0, 0, 0.0)  # No results available
    
    # Find all result JSON files
    result_files = list(results_dir.glob("*.json"))
    
    total_passed_steps = 0
    total_steps = 0
    
    for result_file in result_files:
        result = parse_result_file(result_file)
        if result:
            total_passed_steps += result["passed_steps"]
            total_steps += result["total_steps"]
    
    success_rate = (total_passed_steps / total_steps * 100) if total_steps > 0 else 0.0
    
    return (total_passed_steps, total_steps, success_rate)


def discover_models_and_attack_types(results_base_dir: Path) -> Tuple[List[str], List[str]]:
    """Discover all models and attack types from the results directory structure."""
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
                                if attack_dir.is_dir() and attack_dir.name in ["benign", "direct", "indirect", "memory_only"]:
                                    attack_types.add(attack_dir.name)
    
    return (sorted(list(models)), sorted(list(attack_types)))


def collect_all_data(
    model_name: str,
    attack_type: str,
    results_base_dir: Path
) -> Dict[str, Dict[str, Tuple[int, int, float]]]:
    """
    Collect all data for a model and attack type.
    
    Returns:
        Dictionary: {defense_type: {memory_backend: (passed, total, success_rate)}}
    """
    memory_backends = ["none", "explicit", "mem0", "rag", "context"]
    defense_types = UNIFIED_DEFENSE_TYPES
    
    data = {}
    for defense_type in defense_types:
        data[defense_type] = {}
        for backend in memory_backends:
            # For "none" backend, results are the same for all defense types
            # (since memory is disabled, defenses don't matter)
            # Always use results from "none/none/" folder for all defense types
            if backend == "none":
                passed, total, rate = collect_results_for_combination(
                    "none", "none", model_name, attack_type, results_base_dir
                )
                data[defense_type][backend] = (passed, total, rate)
            else:
                # For actual memory backends, collect results for the defense type
                passed, total, rate = collect_results_for_combination(
                    backend, defense_type, model_name, attack_type, results_base_dir
                )
                data[defense_type][backend] = (passed, total, rate)
    
    return data


def generate_csv(
    model_name: str,
    attack_type: str,
    data: Dict[str, Dict[str, Tuple[int, int, float]]],
    output_dir: Path
) -> Path:
    """
    Generate CSV file for a model and attack type.
    
    Structure:
    - Rows: Defense types
    - Columns: Memory backends with success percentage
    """
    output_file = output_dir / f"{model_name}_{attack_type}_consolidated.csv"
    
    memory_backends = ["none", "explicit", "mem0", "rag", "context"]
    defense_types = UNIFIED_DEFENSE_TYPES
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ["Defense Type"]
        for backend in memory_backends:
            backend_label = "No Memory" if backend == "none" else backend.capitalize()
            header.append(f"{backend_label} (%)")
        writer.writerow(header)
        
        # Data rows
        for defense_type in defense_types:
            row = [defense_type.replace("_", " ").title()]
            for backend in memory_backends:
                _, _, rate = data[defense_type][backend]
                if rate > 0:
                    row.append(f"{rate:.1f}%")
                else:
                    row.append("-")
            writer.writerow(row)
    
    return output_file


def generate_bar_chart(
    model_name: str,
    attack_type: str,
    data: Dict[str, Dict[str, Tuple[int, int, float]]],
    output_dir: Path
) -> Path:
    """
    Generate a bar chart showing success rates by defense type and memory backend.
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
    
    memory_backends = ["none", "explicit", "mem0", "rag", "context"]
    backend_labels = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
    defense_types = UNIFIED_DEFENSE_TYPES
    defense_labels = [dt.replace("_", " ").title() for dt in defense_types]
    
    # Prepare data for plotting
    rates = []
    for defense_type in defense_types:
        row = []
        for backend in memory_backends:
            _, _, rate = data[defense_type][backend]
            row.append(rate)
        rates.append(row)
    
    rates = np.array(rates)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(defense_types))
    width = 0.15  # Width of bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot bars for each memory backend
    for i, (backend, label, color) in enumerate(zip(memory_backends, backend_labels, colors)):
        values = rates[:, i]
        # Only plot if there's data (rate > 0)
        mask = values > 0
        if np.any(mask):
            ax.bar(x[mask] + i * width, values[mask], width, label=label, color=color, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Defense Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - {attack_type.replace("_", " ").title()} Suite\nSuccess Rate by Defense Type and Memory Backend', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 2)  # Center ticks
    ax.set_xticklabels(defense_labels, rotation=45, ha='right')
    ax.set_ylim((0, 105))
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, defense_type in enumerate(defense_types):
        for j, backend in enumerate(memory_backends):
            _, _, rate = data[defense_type][backend]
            if rate > 0:
                ax.text(i + j * width, rate + 1, f'{rate:.1f}%', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{model_name}_{attack_type}_bar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_heatmap(
    model_name: str,
    attack_type: str,
    data: Dict[str, Dict[str, Tuple[int, int, float]]],
    output_dir: Path
) -> Path:
    """
    Generate a heatmap showing success rates.
    Rows: Defense types, Columns: Memory backends
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
    
    memory_backends = ["none", "explicit", "mem0", "rag", "context"]
    backend_labels = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
    defense_types = UNIFIED_DEFENSE_TYPES
    defense_labels = [dt.replace("_", " ").title() for dt in defense_types]
    
    # Prepare data matrix
    rates_matrix = []
    for defense_type in defense_types:
        row = []
        for backend in memory_backends:
            _, _, rate = data[defense_type][backend]
            row.append(rate)
        rates_matrix.append(row)
    
    rates_matrix = np.array(rates_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    sns.heatmap(rates_matrix, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                vmin=0, 
                vmax=100,
                cbar_kws={'label': 'Success Rate (%)'},
                xticklabels=backend_labels,
                yticklabels=defense_labels,
                linewidths=1,
                linecolor='gray',
                ax=ax)
    
    ax.set_xlabel('Memory Backend', fontsize=12, fontweight='bold')
    ax.set_ylabel('Defense Type', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - {attack_type.replace("_", " ").title()} Suite\nSuccess Rate Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{model_name}_{attack_type}_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    """Main function to consolidate all results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidate benchmark results into CSV files and visualizations"
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
        help="Output directory for consolidated files"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to consolidate (default: all models)"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        choices=["benign", "direct", "indirect", "memory_only"],
        help="Specific test suite to consolidate (default: all suites)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only generate CSV files)"
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
    attack_types_to_process = [args.suite] if args.suite else all_attack_types
    
    print(f"\n{'='*80}")
    print(f"Consolidating Results")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models_to_process)}")
    print(f"Test Suites: {', '.join(attack_types_to_process)}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Generate files for each model and attack type combination
    csv_files = []
    plot_files = []
    
    for model_name in models_to_process:
        for attack_type in attack_types_to_process:
            print(f"Processing: {model_name} / {attack_type}")
            
            # Collect data
            data = collect_all_data(model_name, attack_type, results_base_dir)
            
            # Generate CSV
            csv_file = generate_csv(model_name, attack_type, data, output_dir)
            csv_files.append(csv_file)
            print(f"  ✅ CSV: {csv_file.name}")
            
            # Generate plots
            if not args.no_plots:
                if not PLOTTING_AVAILABLE:
                    print(f"  ⚠️  Skipping plots: matplotlib/seaborn not installed")
                    print(f"     Install with: pip install matplotlib seaborn")
                else:
                    try:
                        bar_file = generate_bar_chart(model_name, attack_type, data, output_dir)
                        heatmap_file = generate_heatmap(model_name, attack_type, data, output_dir)
                        plot_files.extend([bar_file, heatmap_file])
                        print(f"  ✅ Bar Chart: {bar_file.name}")
                        print(f"  ✅ Heatmap: {heatmap_file.name}")
                    except Exception as e:
                        print(f"  ⚠️  Error generating plots: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ Consolidation complete!")
    print(f"📊 Generated {len(csv_files)} CSV file(s)")
    if not args.no_plots:
        print(f"📈 Generated {len(plot_files)} plot file(s)")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
