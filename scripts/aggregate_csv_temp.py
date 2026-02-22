#!/usr/bin/env python3
"""
Aggregate CSV Script - Temporary

Reads existing consolidated CSV files and generates combined visualizations and CSVs.
This script parses CSV files instead of result JSON files.

Usage:
    python scripts/aggregate_csv.py
    python scripts/aggregate_csv.py --input-dir data/benchmark/consolidated_results
    python scripts/aggregate_csv.py --model gpt-5-mini
"""

import csv
import sys
import re
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

from benchmark.benchmark_utils import UNIFIED_DEFENSE_TYPES, MEMORY_BACKENDS


def parse_csv_file(csv_file: Path) -> Optional[Dict]:
    """
    Parse a consolidated CSV file and extract data.
    
    Returns:
        Dictionary: {defense_type: {memory_backend: (passed, total, success_rate, has_execution_errors)}}
        Note: passed and total are dummy values (0, 0) since we only have rates from CSV
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            data = {}
            backend_mapping = {
                "No Memory (%)": "none",
                "Explicit (%)": "explicit",
                "Mem0 (%)": "mem0",
                "Rag (%)": "rag",
                "Context (%)": "context"
            }
            
            for row in reader:
                defense_type_raw = row.get("Defense Type", "").strip()
                if not defense_type_raw:
                    continue
                
                # Convert defense type back to internal format
                # CSV format: "None", "User Prompt Only", "No Untrusted Tools", etc.
                # Internal format: "none", "user_prompt_only", "no_untrusted_tools", etc.
                defense_type_lower = defense_type_raw.lower()
                
                # Map CSV defense names to internal format
                defense_mapping = {
                    "none": "none",
                    "user prompt only": "user_prompt_only",
                    "no untrusted tools": "no_untrusted_tools",
                    "limit memory length": "limit_memory_length",
                    "provable policy": "provable_policy"
                }
                
                defense_type = defense_mapping.get(defense_type_lower, defense_type_lower.replace(" ", "_"))
                
                data[defense_type] = {}
                
                # Parse each memory backend column
                for col_name, backend in backend_mapping.items():
                    value = row.get(col_name, "").strip()
                    
                    if value == "ERR":
                        # Execution error
                        data[defense_type][backend] = (0, 0, 0.0, True)
                    elif value == "-":
                        # No data
                        data[defense_type][backend] = (0, 0, 0.0, False)
                    elif value.endswith("%"):
                        # Extract percentage
                        try:
                            rate = float(value.rstrip("%"))
                            data[defense_type][backend] = (0, 0, rate, False)
                        except ValueError:
                            data[defense_type][backend] = (0, 0, 0.0, False)
                    else:
                        data[defense_type][backend] = (0, 0, 0.0, False)
            
            return data
    except Exception as e:
        print(f"WARNING: Could not parse {csv_file}: {e}")
        return None


def discover_models_and_suites(input_dir: Path) -> Tuple[List[str], List[str]]:
    """Discover all models and test suites from CSV files in the input directory."""
    models = set()
    suites = set()
    
    if not input_dir.exists():
        return ([], [])
    
    # Pattern: {model}_{suite}_consolidated.csv
    pattern = re.compile(r'^(.+?)_(.+?)_consolidated\.csv$')
    
    for csv_file in input_dir.glob("*_consolidated.csv"):
        match = pattern.match(csv_file.name)
        if match:
            model_name = match.group(1)
            suite_name = match.group(2)
            models.add(model_name)
            suites.add(suite_name)
    
    return (sorted(list(models)), sorted(list(suites)))


def collect_all_data_from_csvs(
    model_name: str,
    suite_name: str,
    input_dir: Path
) -> Dict[str, Dict[str, Tuple[int, int, float, bool]]]:
    """
    Collect data for a model and suite from CSV file.
    
    Returns:
        Dictionary: {defense_type: {memory_backend: (passed, total, success_rate, has_execution_errors)}}
    """
    csv_file = input_dir / f"{model_name}_{suite_name}_consolidated.csv"
    
    if not csv_file.exists():
        # Return empty data structure
        memory_backends = MEMORY_BACKENDS
        defense_types = UNIFIED_DEFENSE_TYPES
        data = {}
        for defense_type in defense_types:
            data[defense_type] = {}
            for backend in memory_backends:
                data[defense_type][backend] = (0, 0, 0.0, False)
        return data
    
    parsed_data = parse_csv_file(csv_file)
    if parsed_data is None:
        # Return empty data structure
        memory_backends = MEMORY_BACKENDS
        defense_types = UNIFIED_DEFENSE_TYPES
        data = {}
        for defense_type in defense_types:
            data[defense_type] = {}
            for backend in memory_backends:
                data[defense_type][backend] = (0, 0, 0.0, False)
        return data
    
    # Ensure all defense types and backends are present
    memory_backends = MEMORY_BACKENDS
    defense_types = UNIFIED_DEFENSE_TYPES
    
    # Fill in missing entries
    for defense_type in defense_types:
        if defense_type not in parsed_data:
            parsed_data[defense_type] = {}
        for backend in memory_backends:
            if backend not in parsed_data[defense_type]:
                parsed_data[defense_type][backend] = (0, 0, 0.0, False)
    
    return parsed_data


def generate_combined_heatmaps_subplot(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, Tuple[int, int, float, bool]]]],
    output_dir: Path
) -> Path:
    """
    Generate a combined plot with all test suites as subplots.
    Each subplot is a heatmap for one test suite.
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
    
    memory_backends = MEMORY_BACKENDS
    backend_labels = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
    defense_types = UNIFIED_DEFENSE_TYPES
    defense_labels = [dt.replace("_", " ").title() for dt in defense_types]
    
    # Get all attack types sorted
    attack_types = sorted(all_data.keys())
    n_suites = len(attack_types)
    
    # Calculate grid dimensions
    n_cols = min(3, n_suites)  # 3 columns max
    n_rows = (n_suites + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_suites == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Generate heatmap for each suite
    for idx, attack_type in enumerate(attack_types):
        ax = axes[idx]
        data = all_data[attack_type]
        
        # Prepare data matrix
        rates_matrix = []
        has_errors_matrix = []
        for defense_type in defense_types:
            rate_row = []
            error_row = []
            for backend in memory_backends:
                _, _, rate, has_errors = data[defense_type][backend]
                rate_row.append(rate)
                error_row.append(has_errors)
            rates_matrix.append(rate_row)
            has_errors_matrix.append(error_row)
        
        rates_matrix = np.array(rates_matrix)
        has_errors_matrix = np.array(has_errors_matrix)
        
        # For error cases, set to NaN
        display_matrix = rates_matrix.copy().astype(float)
        display_matrix[has_errors_matrix] = np.nan
        
        # Create heatmap
        sns.heatmap(display_matrix, 
                    annot=True, 
                    fmt='.1f',
                    cmap='RdYlGn',
                    vmin=0, 
                    vmax=100,
                    cbar_kws={'label': 'Success Rate (%)'},
                    xticklabels=backend_labels,
                    yticklabels=defense_labels,
                    linewidths=0.5,
                    linecolor='gray',
                    ax=ax,
                    mask=has_errors_matrix)
        
        # Add error annotations
        for i in range(len(defense_types)):
            for j in range(len(memory_backends)):
                if has_errors_matrix[i, j]:
                    ax.text(j + 0.5, i + 0.5, 'ERR', 
                           ha='center', va='center', fontsize=8, fontweight='bold', 
                           color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set title for subplot
        suite_label = attack_type.replace("_", " ").title()
        ax.set_title(f'{suite_label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Memory Backend', fontsize=9)
        ax.set_ylabel('Defense Type', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_suites, len(axes)):
        axes[idx].axis('off')
    
    # Main title
    fig.suptitle(f'{model_name.upper()} - All Test Suites\nSuccess Rate Heatmaps', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = output_dir / f"{model_name}_all_suites_combined_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_average_heatmap(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, Tuple[int, int, float, bool]]]],
    output_dir: Path
) -> Path:
    """
    Generate a heatmap showing average success rates across all test suites.
    Rows: Defense types, Columns: Memory backends
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
    
    memory_backends = MEMORY_BACKENDS
    backend_labels = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
    defense_types = UNIFIED_DEFENSE_TYPES
    defense_labels = [dt.replace("_", " ").title() for dt in defense_types]
    
    # Calculate averages across all suites
    rates_matrix = []
    has_errors_matrix = []
    
    for defense_type in defense_types:
        rate_row = []
        error_row = []
        for backend in memory_backends:
            rates = []
            has_any_errors = False
            
            # Collect rates from all suites for this combination
            for attack_type in all_data.keys():
                _, _, rate, has_errors = all_data[attack_type][defense_type][backend]
                if has_errors:
                    has_any_errors = True
                else:
                    # Only include non-error rates in average
                    rates.append(rate)
            
            if has_any_errors:
                # If any suite has errors, mark as error
                avg_rate = 0.0
                error_row.append(True)
            elif len(rates) > 0:
                avg_rate = np.mean(rates)
                error_row.append(False)
            else:
                avg_rate = 0.0
                error_row.append(False)
            
            rate_row.append(avg_rate)
        
        rates_matrix.append(rate_row)
        has_errors_matrix.append(error_row)
    
    rates_matrix = np.array(rates_matrix)
    has_errors_matrix = np.array(has_errors_matrix)
    
    # Create heatmap
    _, ax = plt.subplots(figsize=(10, 8))
    
    # For error cases, set to NaN
    display_matrix = rates_matrix.copy().astype(float)
    display_matrix[has_errors_matrix] = np.nan
    
    # Create heatmap
    sns.heatmap(display_matrix, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                vmin=0, 
                vmax=100,
                cbar_kws={'label': 'Average Success Rate (%)'},
                xticklabels=backend_labels,
                yticklabels=defense_labels,
                linewidths=1,
                linecolor='gray',
                ax=ax,
                mask=has_errors_matrix)
    
    # Add error annotations
    for i in range(len(defense_types)):
        for j in range(len(memory_backends)):
            if has_errors_matrix[i, j]:
                ax.text(j + 0.5, i + 0.5, 'ERR', 
                       ha='center', va='center', fontsize=10, fontweight='bold', 
                       color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Memory Backend', fontsize=12, fontweight='bold')
    ax.set_ylabel('Defense Type', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Average Across All Test Suites\nSuccess Rate Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{model_name}_average_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_combined_csv(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, Tuple[int, int, float, bool]]]],
    output_dir: Path
) -> Path:
    """
    Generate a combined CSV file listing all test suites one below the other, clearly labeled.
    """
    output_file = output_dir / f"{model_name}_all_suites_combined.csv"
    
    memory_backends = MEMORY_BACKENDS
    defense_types = UNIFIED_DEFENSE_TYPES
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ["Test Suite", "Defense Type"]
        for backend in memory_backends:
            backend_label = "No Memory" if backend == "none" else backend.capitalize()
            header.append(f"{backend_label} (%)")
        writer.writerow(header)
        
        # Write data for each suite
        for idx, attack_type in enumerate(sorted(all_data.keys())):
            suite_label = attack_type.replace("_", " ").title()
            data = all_data[attack_type]
            
            # Write a separator row (empty row before each suite, except the first)
            if idx > 0:
                writer.writerow([])
            
            # Write data rows for this suite
            for defense_type in defense_types:
                row = [suite_label, defense_type.replace("_", " ").title()]
                for backend in memory_backends:
                    _, total_steps, rate, has_errors = data[defense_type][backend]
                    if has_errors:
                        row.append("ERR")
                    elif total_steps > 0 or rate > 0:
                        # For CSV-based data, we might have rate > 0 even if total_steps is 0
                        row.append(f"{rate:.1f}%")
                    else:
                        row.append("-")
                writer.writerow(row)
    
    return output_file


def generate_average_summary_csv(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, Tuple[int, int, float, bool]]]],
    output_dir: Path
) -> Path:
    """
    Generate a CSV file with average scores for all 25 combinations (5 memory backends × 5 defense types).
    Same numbers as the averaged heatmap.
    """
    output_file = output_dir / f"{model_name}_average_summary.csv"
    
    memory_backends = MEMORY_BACKENDS
    defense_types = UNIFIED_DEFENSE_TYPES
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ["Defense Type"]
        for backend in memory_backends:
            backend_label = "No Memory" if backend == "none" else backend.capitalize()
            header.append(f"{backend_label} (%)")
        writer.writerow(header)
        
        # Calculate averages and write rows
        for defense_type in defense_types:
            row = [defense_type.replace("_", " ").title()]
            for backend in memory_backends:
                rates = []
                has_any_errors = False
                
                # Collect rates from all suites for this combination
                for attack_type in all_data.keys():
                    _, _, rate, has_errors = all_data[attack_type][defense_type][backend]
                    if has_errors:
                        has_any_errors = True
                    else:
                        # Only include non-error rates in average
                        rates.append(rate)
                
                if has_any_errors:
                    row.append("ERR")
                elif len(rates) > 0:
                    avg_rate = sum(rates) / len(rates)
                    row.append(f"{avg_rate:.1f}%")
                else:
                    row.append("-")
            
            writer.writerow(row)
    
    return output_file


def main():
    """Main function to aggregate CSV files and generate combined outputs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aggregate consolidated CSV files into combined visualizations and CSVs"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/benchmark/consolidated_results",
        help="Input directory containing consolidated CSV files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark/consolidated_results",
        help="Output directory for aggregated files"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to aggregate (default: all models)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only generate CSV files)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover models and suites
    all_models, all_suites = discover_models_and_suites(input_dir)
    
    if not all_models:
        print(f"WARNING: No models found in {input_dir}")
        return 1
    
    if not all_suites:
        print(f"WARNING: No test suites found in {input_dir}")
        return 1
    
    # Filter based on arguments
    models_to_process = [args.model] if args.model else all_models
    
    print(f"\n{'='*80}")
    print(f"Aggregating CSV Files")
    print(f"{'='*80}")
    print(f"Input Directory: {input_dir}")
    print(f"Models: {', '.join(models_to_process)}")
    print(f"Test Suites: {', '.join(all_suites)}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Generate files for each model
    csv_files = []
    plot_files = []
    
    for model_name in models_to_process:
        print(f"Processing: {model_name}")
        
        # Collect data from all suites
        all_suites_data = {}
        for suite_name in all_suites:
            data = collect_all_data_from_csvs(model_name, suite_name, input_dir)
            all_suites_data[suite_name] = data
        
        # Generate combined CSV files
        print(f"  Generating combined CSV files...")
        try:
            # Combined CSV with all suites
            combined_csv_file = generate_combined_csv(model_name, all_suites_data, output_dir)
            csv_files.append(combined_csv_file)
            print(f"  OK: Combined CSV: {combined_csv_file.name}")
            
            # Average summary CSV
            avg_csv_file = generate_average_summary_csv(model_name, all_suites_data, output_dir)
            csv_files.append(avg_csv_file)
            print(f"  OK: Average Summary CSV: {avg_csv_file.name}")
        except Exception as e:
            print(f"  WARNING: Error generating combined CSV files: {e}")
        
        # Generate combined visualizations
        if not args.no_plots:
            if not PLOTTING_AVAILABLE:
                print(f"  WARNING: Skipping plots: matplotlib/seaborn not installed")
                print(f"     Install with: pip install matplotlib seaborn")
            else:
                print(f"  Generating combined visualizations...")
                try:
                    # Combined heatmaps subplot
                    combined_heatmap_file = generate_combined_heatmaps_subplot(
                        model_name, all_suites_data, output_dir
                    )
                    plot_files.append(combined_heatmap_file)
                    print(f"  OK: Combined Heatmaps: {combined_heatmap_file.name}")
                    
                    # Average heatmap
                    avg_heatmap_file = generate_average_heatmap(
                        model_name, all_suites_data, output_dir
                    )
                    plot_files.append(avg_heatmap_file)
                    print(f"  OK: Average Heatmap: {avg_heatmap_file.name}")
                except Exception as e:
                    print(f"  WARNING: Error generating combined visualizations: {e}")
    
    print(f"\n{'='*80}")
    print(f"Aggregation complete!")
    print(f"Generated {len(csv_files)} CSV file(s)")
    if not args.no_plots:
        print(f"Generated {len(plot_files)} plot file(s)")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

