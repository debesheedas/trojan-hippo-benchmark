#!/usr/bin/env python3
"""
Attack Results Consolidation Script with CSV and Plots

Consolidates results from the attack benchmark (data/benchmark/attack_results)
into CSV tables and visualizations. Logs are read from data/benchmark/attack_logs
when available.

Discovers attack suites from the results directory (model/backend/defense/suite/)
and only generates CSVs and plots for suites that have result files present.

Usage:
    python scripts/consolidate_attack_results.py
    python scripts/consolidate_attack_results.py --results-dir data/benchmark/attack_results
    python scripts/consolidate_attack_results.py --model gpt-4o-mini
    python scripts/consolidate_attack_results.py --no-plots
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import plotting libraries (optional)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
except ImportError:
    PLOTTING_AVAILABLE = False

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.benchmark_utils import (
    UNIFIED_DEFENSE_TYPES,
    MEMORY_BACKENDS,
    get_combination_log_path,
    is_valid_combination,
)

# Constants
BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]

# Default paths for attack benchmark
DEFAULT_RESULTS_DIR = Path("data/benchmark/attack_results")
DEFAULT_LOGS_DIR = Path("data/benchmark/attack_logs")
DEFAULT_OUTPUT_DIR = Path("data/benchmark/consolidated_attack_results")

# Data tuple: (user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate, has_execution_errors)
MetricTuple = Tuple[int, int, float, int, int, float, bool]


def get_attack_results_dir(
    model_name: str,
    memory_backend: str,
    unified_defense: str,
    results_base_dir: Path,
    suite_name: Optional[str] = None,
) -> Path:
    """
    Get the results directory for attack benchmark.
    Path structure: {results_base_dir}/{model_name}/{memory_backend}/{defense_type}/
    With optional suite: {results_base_dir}/{model_name}/{memory_backend}/{defense_type}/{suite_name}/
    """
    p = results_base_dir / model_name / memory_backend / unified_defense
    if suite_name:
        p = p / suite_name
    return p


# Filename stem marker for train test cases (excluded from consolidation; only test case results count)
TRAIN_STEM_MARKER = "train"


def _iter_result_files_for_suite(results_dir: Path, suite_name: str):
    """
    Iterate over result JSON files for a suite, supporting both:
    - Legacy: result files directly in results_dir (model/backend/defense/*.json)
    - New: result files in results_dir/suite_name/*.json
    Skips any file whose stem contains TRAIN_STEM_MARKER (train test cases).
    """
    # New layout: defense/suite_name/*.json
    suite_dir = results_dir / suite_name
    if suite_dir.is_dir():
        for p in suite_dir.glob("*.json"):
            if suite_name in p.stem and TRAIN_STEM_MARKER not in p.stem:
                yield p
    # Legacy: defense/*.json (filter by suite in stem)
    for p in results_dir.glob("*.json"):
        if suite_name in p.stem and TRAIN_STEM_MARKER not in p.stem:
            yield p


def parse_result_file(result_file: Path) -> Optional[Dict]:
    """
    Parse a result JSON file and extract user (utility) and attack goal statistics.
    Returns user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate, execution_success.
    """
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)

        steps = result.get("steps", [])
        excluded_types = {"start_new_session", "insert_attack_email"}

        # User (utility) goal: use top-level if present, else compute from steps
        if "total_user_steps" in result and "total_successful_user_steps" in result:
            user_total = result["total_user_steps"]
            user_passed = result["total_successful_user_steps"]
        else:
            user_steps = [
                s for s in steps
                if s.get("step_type") not in excluded_types
                and (s.get("user_goal") is not None or s.get("success_check") is not None)
            ]
            user_total = len(user_steps)
            user_passed = sum(
                1 for s in user_steps
                if s.get("user_goal", {}).get("passed") is True or s.get("passed") is True
            )

        # Attack goal: steps that have attack_goal with passed not None
        attack_steps = [
            s for s in steps
            if s.get("step_type") not in excluded_types
            and s.get("attack_goal") is not None
            and isinstance(s.get("attack_goal"), dict)
            and s["attack_goal"].get("passed") is not None
        ]
        attack_total = len(attack_steps)
        attack_passed = sum(
            1 for s in attack_steps
            if s["attack_goal"].get("passed") is True
        )

        user_rate = (user_passed / user_total * 100) if user_total > 0 else 0.0
        attack_rate = (attack_passed / attack_total * 100) if attack_total > 0 else 0.0
        execution_success = result.get("execution_success", True)
        execution_errors = result.get("execution_errors", [])

        return {
            "test_name": result.get("test_name", "Unknown"),
            "user_passed": user_passed,
            "user_total": user_total,
            "user_rate": user_rate,
            "attack_passed": attack_passed,
            "attack_total": attack_total,
            "attack_rate": attack_rate,
            "execution_success": execution_success,
            "execution_errors": execution_errors if execution_errors else []
        }
    except Exception as e:
        print(f"WARNING: Could not parse {result_file}: {e}")
        return None


def parse_log_file(log_file: Path) -> Optional[Dict]:
    """Parse a log file and extract error information."""
    if not log_file.exists():
        return None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        errors = []
        for i, line in enumerate(lines, start=1):
            line_lower = line.lower()
            line_stripped = line.strip()
            if any(phrase in line_lower for phrase in [
                "rate limit error (final)",
                "connection error (final",
                "api error"
            ]) and "final" in line_lower:
                errors.append(f"Line {i}: {line_stripped}")
            elif ("error:" in line_lower or "exception:" in line_lower):
                if "traceback" in line_lower or "failed" not in line_lower[:50]:
                    errors.append(f"Line {i}: {line_stripped}")

        return {
            "has_errors": len(errors) > 0,
            "errors": errors,
            "warnings": []
        }
    except Exception as e:
        return {"has_errors": False, "errors": [f"Could not parse log: {e}"], "warnings": []}


def collect_results_for_combination(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    suite_name: str,
    results_base_dir: Path,
) -> MetricTuple:
    """
    Collect user (utility) and attack goal statistics for a specific combination.
    Only JSON files whose name contains the suite name are included.

    Returns:
        (user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate, has_execution_errors)
    """
    results_dir = get_attack_results_dir(
        model_name, memory_backend, unified_defense, results_base_dir
    )

    if not results_dir.exists():
        return (0, 0, 0.0, 0, 0, 0.0, False)

    result_files = list(_iter_result_files_for_suite(results_dir, suite_name))

    user_passed = user_total = attack_passed = attack_total = 0
    has_execution_errors = False

    for result_file in result_files:
        result = parse_result_file(result_file)
        if result:
            user_passed += result["user_passed"]
            user_total += result["user_total"]
            attack_passed += result["attack_passed"]
            attack_total += result["attack_total"]
            if not result.get("execution_success", True):
                has_execution_errors = True

    user_rate = (user_passed / user_total * 100) if user_total > 0 else 0.0
    attack_rate = (attack_passed / attack_total * 100) if attack_total > 0 else 0.0
    return (user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate, has_execution_errors)


def discover_models(results_base_dir: Path) -> List[str]:
    """Discover all models from the attack results directory structure."""
    models = []
    if not results_base_dir.exists():
        return models
    for model_dir in results_base_dir.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)
    return sorted(models)


def discover_suites(results_base_dir: Path, model_name: str) -> List[str]:
    """
    Discover attack suite names that actually have result files under this model.
    Scans the results directory structure and returns only suites that have at least
    one non-train result JSON file. This ensures we only generate CSVs/plots for
    suites that exist in the results folder.
    """
    suites = set()
    for defense_type in UNIFIED_DEFENSE_TYPES:
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            results_dir = results_base_dir / model_name / backend / defense_type
            if not results_dir.exists():
                continue
            # New layout: result files in results_dir/suite_name/*.json
            for subdir in results_dir.iterdir():
                if subdir.is_dir():
                    suite_name = subdir.name
                    for p in subdir.glob("*.json"):
                        if suite_name in p.stem and TRAIN_STEM_MARKER not in p.stem:
                            suites.add(suite_name)
                            break
            # Legacy layout: result files directly in results_dir/*.json
            for p in results_dir.glob("*.json"):
                if TRAIN_STEM_MARKER in p.stem:
                    continue
                # Stem e.g. "01_persistent_exfiltrate_tax" -> suite "persistent_exfiltrate_tax"
                parts = p.stem.split("_", 1)
                if len(parts) >= 2:
                    suites.add(parts[1])
    return sorted(suites)


def collect_all_data(
    model_name: str,
    suite_name: str,
    results_base_dir: Path,
) -> Dict[str, Dict[str, MetricTuple]]:
    """
    Collect all data for a model and suite.
    Returns: {defense_type: {memory_backend: MetricTuple}}
    """
    data = {}
    for defense_type in UNIFIED_DEFENSE_TYPES:
        data[defense_type] = {}
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            data[defense_type][backend] = collect_results_for_combination(
                backend, defense_type, model_name, suite_name, results_base_dir
            )
    return data


def _default_metric_tuple() -> MetricTuple:
    return (0, 0, 0.0, 0, 0, 0.0, False)


def generate_csv(
    model_name: str,
    suite_name: str,
    data: Dict[str, Dict[str, MetricTuple]],
    output_dir: Path,
) -> Path:
    """Generate CSV file for a model and suite with Utility and Attack success rates."""
    output_file = output_dir / f"{model_name}_{suite_name}_consolidated.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Defense Type", "Metric"]
        for backend in MEMORY_BACKENDS:
            backend_label = BACKEND_LABELS[MEMORY_BACKENDS.index(backend)]
            header.append(f"{backend_label} (%)")
        writer.writerow(header)

        for defense_type in UNIFIED_DEFENSE_TYPES:
            dt_label = defense_type.replace("_", " ").title()
            for metric_name, rate_idx, total_idx in [
                ("Utility", 2, 1),   # user_rate, user_total
                ("Attack", 5, 4),    # attack_rate, attack_total
            ]:
                row = [dt_label, metric_name]
                for backend in MEMORY_BACKENDS:
                    if not is_valid_combination(backend, defense_type):
                        row.append("-")
                        continue
                    t = data[defense_type].get(backend, _default_metric_tuple())
                    has_errors = t[6]
                    total = t[total_idx]
                    rate = t[rate_idx]
                    if has_errors:
                        row.append("ERR")
                    elif total > 0:
                        row.append(f"{rate:.1f}%")
                    else:
                        row.append("-")
                writer.writerow(row)

    return output_file


def _prepare_heatmap_data(
    data: Dict[str, Dict[str, MetricTuple]],
    metric: str = "utility",
) -> Tuple[Any, Any, Any]:
    """
    Prepare data matrices for heatmap. metric is 'utility' or 'attack'.
    Returns (rates_matrix, has_errors_matrix, no_data_matrix).
    Cells with no data (total==0) are set to NaN and flagged in no_data_matrix so they appear white.
    """
    rate_idx = 2 if metric == "utility" else 5
    total_idx = 1 if metric == "utility" else 4
    rates_matrix = []
    has_errors_matrix = []
    no_data_matrix = []
    for defense_type in UNIFIED_DEFENSE_TYPES:
        rate_row = []
        error_row = []
        no_data_row = []
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                rate_row.append(np.nan)
                error_row.append(False)
                no_data_row.append(True)
            else:
                t = data[defense_type].get(backend, _default_metric_tuple())
                total = t[total_idx]
                if total == 0:
                    rate_row.append(np.nan)
                    no_data_row.append(True)
                else:
                    rate_row.append(t[rate_idx])
                    no_data_row.append(False)
                error_row.append(t[6])
        rates_matrix.append(rate_row)
        has_errors_matrix.append(error_row)
        no_data_matrix.append(no_data_row)
    return (
        np.array(rates_matrix),
        np.array(has_errors_matrix),
        np.array(no_data_matrix),
    )


def _draw_one_heatmap(
    ax: Any,
    data: Dict[str, Dict[str, MetricTuple]],
    metric: str,
    title: str,
) -> None:
    """Draw a single heatmap (Utility or Attack). No-data cells (total==0) are white; ERR cells show 'ERR'."""
    rates_matrix, has_errors_matrix, no_data_matrix = _prepare_heatmap_data(data, metric=metric)
    defense_labels = [dt.replace("_", " ").title() for dt in UNIFIED_DEFENSE_TYPES]
    display_matrix = rates_matrix.copy().astype(float)
    # Mask both errors and no-data so they appear white; we'll draw 'ERR' only on error cells
    mask = has_errors_matrix | no_data_matrix
    display_matrix[mask] = np.nan

    sns.heatmap(
        display_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Success Rate (%)'},
        xticklabels=BACKEND_LABELS,
        yticklabels=defense_labels,
        linewidths=1,
        linecolor='gray',
        ax=ax,
        mask=mask,
    )
    for i in range(len(UNIFIED_DEFENSE_TYPES)):
        for j in range(len(MEMORY_BACKENDS)):
            if has_errors_matrix[i, j]:
                ax.text(
                    j + 0.5, i + 0.5, 'ERR',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            # no_data cells stay white (masked), no label
    ax.set_xlabel('Memory Backend', fontsize=12, fontweight='bold')
    ax.set_ylabel('Defense Type', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')


def generate_heatmap(
    model_name: str,
    suite_name: str,
    data: Dict[str, Dict[str, MetricTuple]],
    output_dir: Path,
) -> Path:
    """Generate a figure with two heatmaps: Utility (user goal) and Attack success rates."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    suite_label = suite_name.replace("_", " ").title()
    fig, (ax_util, ax_attack) = plt.subplots(1, 2, figsize=(18, 8))

    _draw_one_heatmap(
        ax_util, data, "utility",
        f"Utility Success Rate\n(user goal)"
    )
    _draw_one_heatmap(
        ax_attack, data, "attack",
        f"Attack Success Rate\n(attack goal)"
    )

    fig.suptitle(
        f'{model_name.upper()} - {suite_label} (Attack Suite)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    output_file = output_dir / f"{model_name}_{suite_name}_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_combined_heatmaps_subplot(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate a combined plot: for each suite, two subplots (Utility and Attack)."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    suite_names = sorted(all_data.keys())
    n_suites = len(suite_names)
    fig, axes = plt.subplots(n_suites, 2, figsize=(16, 5 * n_suites))
    if n_suites == 1:
        axes = axes.reshape(1, -1)

    for idx, suite_name in enumerate(suite_names):
        suite_label = suite_name.replace("_", " ").title()
        data = all_data[suite_name]
        _draw_one_heatmap(
            axes[idx, 0], data, "utility",
            f'{suite_label} - Utility'
        )
        _draw_one_heatmap(
            axes[idx, 1], data, "attack",
            f'{suite_label} - Attack'
        )

    fig.suptitle(
        f'{model_name.upper()} - Attack Suites (Utility & Attack Success Rates)',
        fontsize=16, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = output_dir / f"{model_name}_all_suites_combined_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_average_heatmap(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate two heatmaps: average Utility and average Attack success rates across suites."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    defense_labels = [dt.replace("_", " ").title() for dt in UNIFIED_DEFENSE_TYPES]

    def _avg_matrix(rate_idx: int, total_idx: int) -> Tuple[Any, Any, Any]:
        """Return (rates_matrix, has_errors_matrix, no_data_matrix). No-data cells are NaN and masked."""
        rates_matrix = []
        has_errors_matrix = []
        no_data_matrix = []
        for defense_type in UNIFIED_DEFENSE_TYPES:
            rate_row = []
            error_row = []
            no_data_row = []
            for backend in MEMORY_BACKENDS:
                if not is_valid_combination(backend, defense_type):
                    rate_row.append(np.nan)
                    error_row.append(False)
                    no_data_row.append(True)
                    continue
                rates = []
                has_any_errors = False
                for suite_name in all_data.keys():
                    if (defense_type in all_data[suite_name]
                            and backend in all_data[suite_name][defense_type]):
                        t = all_data[suite_name][defense_type][backend]
                        if t[6]:
                            has_any_errors = True
                        elif t[total_idx] > 0:
                            rates.append(t[rate_idx])
                if has_any_errors:
                    rate_row.append(np.nan)
                    error_row.append(True)
                    no_data_row.append(False)
                elif rates:
                    rate_row.append(float(np.mean(rates)))
                    error_row.append(False)
                    no_data_row.append(False)
                else:
                    rate_row.append(np.nan)
                    error_row.append(False)
                    no_data_row.append(True)
            rates_matrix.append(rate_row)
            has_errors_matrix.append(error_row)
            no_data_matrix.append(no_data_row)
        return (
            np.array(rates_matrix),
            np.array(has_errors_matrix),
            np.array(no_data_matrix),
        )

    fig, (ax_util, ax_attack) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, rate_idx, total_idx, label in [
        (ax_util, 2, 1, "Utility (user goal)"),
        (ax_attack, 5, 4, "Attack (attack goal)"),
    ]:
        rates_matrix, has_errors_matrix, no_data_matrix = _avg_matrix(rate_idx, total_idx)
        display_matrix = rates_matrix.copy().astype(float)
        mask = has_errors_matrix | no_data_matrix
        display_matrix[mask] = np.nan
        sns.heatmap(
            display_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Avg Success Rate (%)'},
            xticklabels=BACKEND_LABELS,
            yticklabels=defense_labels,
            linewidths=1,
            linecolor='gray',
            ax=ax,
            mask=mask,
        )
        for i in range(len(UNIFIED_DEFENSE_TYPES)):
            for j in range(len(MEMORY_BACKENDS)):
                if has_errors_matrix[i, j]:
                    ax.text(
                        j + 0.5, i + 0.5, 'ERR',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
        ax.set_xlabel('Memory Backend', fontsize=12, fontweight='bold')
        ax.set_ylabel('Defense Type', fontsize=12, fontweight='bold')
        ax.set_title(f'Average {label}', fontsize=12, fontweight='bold')

    fig.suptitle(
        f'{model_name.upper()} - Average Across Attack Suites',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    output_file = output_dir / f"{model_name}_average_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_combined_csv(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate a combined CSV listing all attack suites with Utility and Attack columns."""
    output_file = output_dir / f"{model_name}_all_suites_combined.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Test Suite", "Defense Type", "Metric"]
        for backend in MEMORY_BACKENDS:
            backend_label = BACKEND_LABELS[MEMORY_BACKENDS.index(backend)]
            header.append(f"{backend_label} (%)")
        writer.writerow(header)

        for idx, suite_name in enumerate(sorted(all_data.keys())):
            suite_label = suite_name.replace("_", " ").title()
            data = all_data[suite_name]
            if idx > 0:
                writer.writerow([])
            for defense_type in UNIFIED_DEFENSE_TYPES:
                dt_label = defense_type.replace("_", " ").title()
                for metric_name, rate_idx, total_idx in [("Utility", 2, 1), ("Attack", 5, 4)]:
                    row = [suite_label, dt_label, metric_name]
                    for backend in MEMORY_BACKENDS:
                        if not is_valid_combination(backend, defense_type):
                            row.append("-")
                            continue
                        t = data[defense_type].get(backend, _default_metric_tuple())
                        if t[6]:
                            row.append("ERR")
                        elif t[total_idx] > 0:
                            row.append(f"{t[rate_idx]:.1f}%")
                        else:
                            row.append("-")
                    writer.writerow(row)

    return output_file


def generate_average_summary_csv(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate a CSV with average Utility and Attack scores for all valid combinations."""
    output_file = output_dir / f"{model_name}_average_summary.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Defense Type", "Metric"]
        for backend in MEMORY_BACKENDS:
            backend_label = BACKEND_LABELS[MEMORY_BACKENDS.index(backend)]
            header.append(f"{backend_label} (%)")
        writer.writerow(header)

        for defense_type in UNIFIED_DEFENSE_TYPES:
            dt_label = defense_type.replace("_", " ").title()
            for metric_name, rate_idx, total_idx in [("Utility", 2, 1), ("Attack", 5, 4)]:
                row = [dt_label, metric_name]
                for backend in MEMORY_BACKENDS:
                    if not is_valid_combination(backend, defense_type):
                        row.append("-")
                        continue
                    rates = []
                    has_any_errors = False
                    for suite_name in all_data.keys():
                        if (defense_type in all_data[suite_name]
                                and backend in all_data[suite_name][defense_type]):
                            t = all_data[suite_name][defense_type][backend]
                            if t[6]:
                                has_any_errors = True
                            elif t[total_idx] > 0:
                                # Only average over suites that have data for this cell
                                rates.append(t[rate_idx])
                    if has_any_errors:
                        row.append("ERR")
                    elif rates:
                        row.append(f"{sum(rates) / len(rates):.1f}%")
                    else:
                        row.append("-")
                writer.writerow(row)

    return output_file


def generate_error_summary(
    model_name: str,
    suite_name: str,
    results_base_dir: Path,
    logs_base_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate an error summary file for execution errors.
    Returns path to summary file, or None if no errors found.
    """
    errors_found = []

    for memory_backend in MEMORY_BACKENDS:
        for unified_defense in UNIFIED_DEFENSE_TYPES:
            if not is_valid_combination(memory_backend, unified_defense):
                continue
            results_dir = get_attack_results_dir(
                model_name, memory_backend, unified_defense, results_base_dir
            )
            for result_file in _iter_result_files_for_suite(results_dir, suite_name):
                result_data = parse_result_file(result_file)
                if result_data and not result_data.get("execution_success", True):
                    errors_found.append({
                        "type": "result_file",
                        "backend": memory_backend,
                        "defense": unified_defense,
                        "test_file": result_file.name,
                        "errors": result_data.get("execution_errors", []),
                    })

            log_path = get_combination_log_path(
                memory_backend=memory_backend,
                unified_defense=unified_defense,
                model_name=model_name,
                attack_type=suite_name,
                logs_base_dir=logs_base_dir,
            )
            log_data = parse_log_file(log_path)
            if log_data and log_data.get("has_errors"):
                errors_found.append({
                    "type": "log_file",
                    "backend": memory_backend,
                    "defense": unified_defense,
                    "log_file": str(log_path),
                    "errors": log_data.get("errors", []),
                })

    if not errors_found:
        return None

    summary_file = output_dir / f"{model_name}_{suite_name}_execution_errors.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{'='*80}\n")
        f.write("EXECUTION ERROR SUMMARY (Attack Benchmark)\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Suite: {suite_name}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"WARNING: {len(errors_found)} combination(s) had execution errors!\n")
        f.write("   These results may be unreliable. Rerun these combinations.\n\n")
        for error in errors_found:
            f.write(f"{'-'*80}\n")
            f.write(f"Backend: {error['backend'].upper()}, Defense: {error['defense']}\n")
            f.write(f"Source: {error['type']}\n")
            if 'test_file' in error:
                f.write(f"Test File: {error['test_file']}\n")
            if 'log_file' in error:
                f.write(f"Log File: {error['log_file']}\n")
            f.write("\nErrors:\n")
            for err in error.get('errors', [])[:10]:
                f.write(f"  - {err}\n")
            if len(error.get('errors', [])) > 10:
                f.write(f"  ... and {len(error['errors']) - 10} more errors\n")
            f.write("\n")
        f.write(f"{'='*80}\n")
    return summary_file


def main() -> int:
    """Main function to consolidate attack results."""
    parser = argparse.ArgumentParser(
        description="Consolidate attack benchmark results into CSV files and visualizations"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Base directory for attack results",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=str(DEFAULT_LOGS_DIR),
        help="Base directory for attack logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for consolidated files",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to consolidate (default: all models)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only generate CSV files)",
    )
    args = parser.parse_args()

    results_base_dir = Path(args.results_dir)
    logs_base_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = discover_models(results_base_dir)
    if not all_models:
        print(f"WARNING: No models found in {results_base_dir}")
        return 1

    models_to_process = [args.model] if args.model else all_models

    print(f"\n{'='*80}")
    print("Consolidating Attack Results")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models_to_process)}")
    print(f"Results: {results_base_dir}")
    print(f"Logs: {logs_base_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    csv_files = []
    plot_files = []
    error_summaries = []

    for model_name in models_to_process:
        suites_to_process = discover_suites(results_base_dir, model_name)
        if not suites_to_process:
            print(f"Skipping {model_name}: no attack suites found in results")
            continue
        print(f"Suites for {model_name}: {', '.join(suites_to_process)}")
        all_suites_data = {}

        for suite_name in suites_to_process:
            print(f"Processing: {model_name} / {suite_name}")

            data = collect_all_data(model_name, suite_name, results_base_dir)
            all_suites_data[suite_name] = data

            csv_file = generate_csv(model_name, suite_name, data, output_dir)
            csv_files.append(csv_file)
            print(f"  OK: CSV: {csv_file.name}")

            error_summary = generate_error_summary(
                model_name, suite_name, results_base_dir, logs_base_dir, output_dir
            )
            if error_summary:
                error_summaries.append(error_summary)
                print(f"  WARNING: Error Summary: {error_summary.name}")
            else:
                print(f"  OK: No execution errors found")

            if not args.no_plots and PLOTTING_AVAILABLE:
                try:
                    heatmap_file = generate_heatmap(
                        model_name, suite_name, data, output_dir
                    )
                    plot_files.append(heatmap_file)
                    print(f"  OK: Heatmap: {heatmap_file.name}")
                except Exception as e:
                    print(f"  WARNING: Error generating heatmap: {e}")
            elif not args.no_plots:
                print("  WARNING: Skipping plots (matplotlib/seaborn not installed)")

        # Combined visualizations (one suite: still generate for consistency)
        if not args.no_plots and PLOTTING_AVAILABLE:
            print(f"\nGenerating combined visualizations for {model_name}...")
            try:
                combined_heatmap_file = generate_combined_heatmaps_subplot(
                    model_name, all_suites_data, output_dir
                )
                plot_files.append(combined_heatmap_file)
                print(f"  OK: Combined Heatmaps: {combined_heatmap_file.name}")
            except Exception as e:
                print(f"  WARNING: Error generating combined visualizations: {e}")

        # Combined and average CSV/heatmap
        print(f"\nGenerating combined CSV files for {model_name}...")
        try:
            combined_csv_file = generate_combined_csv(
                model_name, all_suites_data, output_dir
            )
            csv_files.append(combined_csv_file)
            print(f"  OK: Combined CSV: {combined_csv_file.name}")

            avg_csv_file = generate_average_summary_csv(
                model_name, all_suites_data, output_dir
            )
            csv_files.append(avg_csv_file)
            print(f"  OK: Average Summary CSV: {avg_csv_file.name}")

            if not args.no_plots and PLOTTING_AVAILABLE:
                try:
                    avg_heatmap_file = generate_average_heatmap(
                        model_name, all_suites_data, output_dir
                    )
                    plot_files.append(avg_heatmap_file)
                    print(f"  OK: Average Heatmap: {avg_heatmap_file.name}")
                except Exception as e:
                    print(f"  WARNING: Error generating average heatmap: {e}")
        except Exception as e:
            print(f"  WARNING: Error generating combined CSV files: {e}")

    print(f"\n{'='*80}")
    print("Consolidation complete!")
    print(f"Generated {len(csv_files)} CSV file(s)")
    if error_summaries:
        print(f"\nWARNING: Generated {len(error_summaries)} error summary file(s)")
        for ef in error_summaries:
            print(f"   - {ef}")
    else:
        print("OK: No execution errors found")
    if not args.no_plots:
        print(f"Generated {len(plot_files)} plot file(s)")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
