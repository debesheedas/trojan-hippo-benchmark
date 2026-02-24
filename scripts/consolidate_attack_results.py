#!/usr/bin/env python3
"""
Attack Results Consolidation Script with CSV and Plots

Consolidates TEST results from the attack benchmark. Reads from
data/benchmark/attack_results/test/ and data/benchmark/attack_logs/test/ by default
(train/test layout matches attack_bench). Generates CSV tables and visualizations.

Discovers suites from the results directory (model/backend/defense/suite/) and only
generates CSVs and plots for suites that have result files present.

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

# Default paths for attack benchmark (test results live under attack_results/test/)
DEFAULT_RESULTS_DIR = Path("data/benchmark/attack_results/test")
DEFAULT_TRAIN_RESULTS_DIR = Path("data/benchmark/attack_results/train")
DEFAULT_LOGS_DIR = Path("data/benchmark/attack_logs/test")
DEFAULT_OUTPUT_DIR = Path("data/benchmark/consolidated_attack_results")

# Data tuple: (user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate,
#              max_attack_passed, max_attack_total, has_execution_errors)
# max_attack_*: count of test cases with >=1 attack success / count of test cases with attack steps
MetricTuple = Tuple[int, int, float, int, int, float, int, int, bool]

# Per-session (trigger 1..5): list of (passed_count, total_count) for session 1..5
SessionDataTuple = List[Tuple[int, int]]


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


def _iter_result_files_for_suite(
    results_dir: Path,
    suite_name: str,
    train_only: bool = False,
):
    """
    Iterate over result JSON files for a suite, supporting both:
    - Legacy: result files directly in results_dir (model/backend/defense/*.json)
    - New: result files in results_dir/suite_name/*.json
    When train_only=False: skips files whose stem contains TRAIN_STEM_MARKER (test only).
    When train_only=True: only yields files whose stem contains TRAIN_STEM_MARKER (train only).
    """
    want_train = train_only
    # New layout: defense/suite_name/*.json
    suite_dir = results_dir / suite_name
    if suite_dir.is_dir():
        for p in suite_dir.glob("*.json"):
            if suite_name not in p.stem:
                continue
            if (TRAIN_STEM_MARKER in p.stem) == want_train:
                yield p
    # Legacy: defense/*.json (filter by suite in stem)
    for p in results_dir.glob("*.json"):
        if suite_name not in p.stem:
            continue
        if (TRAIN_STEM_MARKER in p.stem) == want_train:
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

        # Attack goal: steps that have attack_goal with passed not None (order preserved for session index)
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
        # Per-session (trigger index): list of bools, one per attack step in order
        attack_step_passed = [
            s["attack_goal"].get("passed") is True
            for s in attack_steps
        ]

        # Stealth goal: steps that have stealth_goal with passed not None (same steps as attack_goal in persistent_exfiltrate)
        stealth_steps = [
            s for s in steps
            if s.get("step_type") not in excluded_types
            and s.get("stealth_goal") is not None
            and isinstance(s.get("stealth_goal"), dict)
            and s["stealth_goal"].get("passed") is not None
        ]
        stealth_total = len(stealth_steps)
        stealth_passed = sum(
            1 for s in stealth_steps
            if s["stealth_goal"].get("passed") is True
        )
        stealth_step_passed = [
            s["stealth_goal"].get("passed") is True
            for s in stealth_steps
        ]

        user_rate = (user_passed / user_total * 100) if user_total > 0 else 0.0
        attack_rate = (attack_passed / attack_total * 100) if attack_total > 0 else 0.0
        stealth_rate = (stealth_passed / stealth_total * 100) if stealth_total > 0 else 0.0
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
            "attack_step_passed": attack_step_passed,
            "stealth_passed": stealth_passed,
            "stealth_total": stealth_total,
            "stealth_rate": stealth_rate,
            "stealth_step_passed": stealth_step_passed,
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
    train_only: bool = False,
) -> Tuple[MetricTuple, SessionDataTuple]:
    """
    Collect user (utility) and attack goal statistics for a specific combination.
    Only JSON files whose name contains the suite name are included.

    Returns:
        (MetricTuple, SessionDataTuple)
        MetricTuple: (user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate,
                     max_attack_passed, max_attack_total, has_execution_errors)
        SessionDataTuple: [(passed_1, total_1), ..., (passed_5, total_5)] for trigger sessions 1..5
    """
    results_dir = get_attack_results_dir(
        model_name, memory_backend, unified_defense, results_base_dir
    )

    default_session = [(0, 0)] * 5
    if not results_dir.exists():
        return ((0, 0, 0.0, 0, 0, 0.0, 0, 0, False), default_session, (0, 0, 0.0))

    result_files = list(_iter_result_files_for_suite(results_dir, suite_name, train_only=train_only))

    user_passed = user_total = attack_passed = attack_total = 0
    max_attack_passed = max_attack_total = 0
    stealth_passed = stealth_total = 0
    has_execution_errors = False
    session_passed = [0] * 5
    session_total = [0] * 5

    for result_file in result_files:
        result = parse_result_file(result_file)
        if result:
            user_passed += result["user_passed"]
            user_total += result["user_total"]
            attack_passed += result["attack_passed"]
            attack_total += result["attack_total"]
            stealth_passed += result.get("stealth_passed", 0)
            stealth_total += result.get("stealth_total", 0)
            if not result.get("execution_success", True):
                has_execution_errors = True
            # Max attack: one test case counts as 100% if any of its attack steps passed
            if result["attack_total"] > 0:
                max_attack_total += 1
                if result["attack_passed"] >= 1:
                    max_attack_passed += 1
            # Per-session (trigger 1..5)
            step_passed = result.get("attack_step_passed", [])
            for s in range(min(5, len(step_passed))):
                session_total[s] += 1
                if step_passed[s]:
                    session_passed[s] += 1

    user_rate = (user_passed / user_total * 100) if user_total > 0 else 0.0
    attack_rate = (attack_passed / attack_total * 100) if attack_total > 0 else 0.0
    stealth_rate = (stealth_passed / stealth_total * 100) if stealth_total > 0 else 0.0
    metric_tuple = (
        user_passed, user_total, user_rate,
        attack_passed, attack_total, attack_rate,
        max_attack_passed, max_attack_total, has_execution_errors,
    )
    session_data: SessionDataTuple = [(session_passed[s], session_total[s]) for s in range(5)]
    stealth_tuple = (stealth_passed, stealth_total, stealth_rate)
    return (metric_tuple, session_data, stealth_tuple)


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


def discover_suites_for_train(results_base_dir: Path, model_name: str) -> List[str]:
    """
    Discover suite names that have train result files under this model.
    Same layout as discover_suites but only includes files whose stem contains TRAIN_STEM_MARKER.
    """
    suites = set()
    for defense_type in UNIFIED_DEFENSE_TYPES:
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            results_dir = results_base_dir / model_name / backend / defense_type
            if not results_dir.exists():
                continue
            for subdir in results_dir.iterdir():
                if subdir.is_dir():
                    suite_name = subdir.name
                    for p in subdir.glob("*.json"):
                        if suite_name in p.stem and TRAIN_STEM_MARKER in p.stem:
                            suites.add(suite_name)
                            break
            for p in results_dir.glob("*.json"):
                if TRAIN_STEM_MARKER not in p.stem:
                    continue
                parts = p.stem.split("_", 1)
                if len(parts) >= 2:
                    suites.add(parts[1])
    return sorted(suites)


# Stealth tuple: (stealth_passed, stealth_total, stealth_rate)
StealthTuple = Tuple[int, int, float]


def collect_all_data(
    model_name: str,
    suite_name: str,
    results_base_dir: Path,
    train_only: bool = False,
) -> Tuple[Dict[str, Dict[str, MetricTuple]], Dict[str, Dict[str, SessionDataTuple]], Dict[str, Dict[str, StealthTuple]]]:
    """
    Collect all data for a model and suite.
    Returns: (data, session_data, data_stealth)
        data: {defense_type: {memory_backend: MetricTuple}}
        session_data: {defense_type: {memory_backend: SessionDataTuple}}
        data_stealth: {defense_type: {memory_backend: (stealth_passed, stealth_total, stealth_rate)}}
    """
    data: Dict[str, Dict[str, MetricTuple]] = {}
    session_data: Dict[str, Dict[str, SessionDataTuple]] = {}
    data_stealth: Dict[str, Dict[str, StealthTuple]] = {}
    for defense_type in UNIFIED_DEFENSE_TYPES:
        data[defense_type] = {}
        session_data[defense_type] = {}
        data_stealth[defense_type] = {}
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            metric_tuple, sess, stealth_tuple = collect_results_for_combination(
                backend, defense_type, model_name, suite_name, results_base_dir,
                train_only=train_only,
            )
            data[defense_type][backend] = metric_tuple
            session_data[defense_type][backend] = sess
            data_stealth[defense_type][backend] = stealth_tuple
    return (data, session_data, data_stealth)


def _default_metric_tuple() -> MetricTuple:
    return (0, 0, 0.0, 0, 0, 0.0, 0, 0, False)


def generate_csv(
    model_name: str,
    suite_name: str,
    data: Dict[str, Dict[str, MetricTuple]],
    output_dir: Path,
    data_stealth: Optional[Dict[str, Dict[str, StealthTuple]]] = None,
) -> Path:
    """Generate CSV file for a model and suite with Utility, Attack, and Stealth success rates. output_dir is the model subfolder (output_base/model_name)."""
    output_file = output_dir / f"{suite_name}_consolidated.csv"

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
                ("Utility", 2, 1),           # user_rate, user_total
                ("Attack (Total)", 5, 4),    # attack_rate, attack_total
                ("Max Attack", 6, 7),        # max: rate = passed/total*100 (indices 6,7; rate computed)
            ]:
                row = [dt_label, metric_name]
                for backend in MEMORY_BACKENDS:
                    if not is_valid_combination(backend, defense_type):
                        row.append("-")
                        continue
                    t = data[defense_type].get(backend, _default_metric_tuple())
                    has_errors = t[8]
                    total = t[total_idx]
                    if metric_name == "Max Attack":
                        rate = (t[6] / t[7] * 100) if t[7] > 0 else 0.0
                    else:
                        rate = t[rate_idx]
                    if has_errors:
                        row.append("ERR")
                    elif total > 0:
                        row.append(f"{rate:.1f}%")
                    else:
                        row.append("-")
                writer.writerow(row)
            # Stealth row (when data_stealth provided)
            if data_stealth:
                row = [dt_label, "Stealth"]
                for backend in MEMORY_BACKENDS:
                    if not is_valid_combination(backend, defense_type):
                        row.append("-")
                        continue
                    st = data_stealth[defense_type].get(backend, (0, 0, 0.0))
                    sp, stot, _ = st
                    if stot > 0:
                        row.append(f"{(sp / stot * 100):.1f}%")
                    else:
                        row.append("-")
                writer.writerow(row)

    return output_file


def _prepare_heatmap_data(
    data: Dict[str, Dict[str, MetricTuple]],
    metric: str = "utility",
) -> Tuple[Any, Any, Any]:
    """
    Prepare data matrices for heatmap. metric is 'utility', 'attack', or 'max_attack'.
    Returns (rates_matrix, has_errors_matrix, no_data_matrix).
    Cells with no data (total==0) are set to NaN and flagged in no_data_matrix so they appear white.
    """
    if metric == "utility":
        rate_idx, total_idx = 2, 1
    elif metric == "attack":
        rate_idx, total_idx = 5, 4
    else:  # max_attack
        rate_idx, total_idx = 6, 7  # passed/total; rate computed below
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
                    if metric == "max_attack":
                        rate_row.append((t[6] / t[7] * 100) if t[7] > 0 else np.nan)
                    else:
                        rate_row.append(t[rate_idx])
                    no_data_row.append(False)
                error_row.append(t[8])
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
    """Generate a figure with three heatmaps: Utility, Attack (total), and Max Attack success rates."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    suite_label = suite_name.replace("_", " ").title()
    fig, (ax_util, ax_attack, ax_max) = plt.subplots(1, 3, figsize=(22, 8))

    _draw_one_heatmap(
        ax_util, data, "utility",
        f"Utility Success Rate\n(user goal)"
    )
    _draw_one_heatmap(
        ax_attack, data, "attack",
        f"Attack Success Rate (Total)\n(attack goal)"
    )
    _draw_one_heatmap(
        ax_max, data, "max_attack",
        f"Max Attack Success Rate\n(any trigger succeeded)"
    )

    fig.suptitle(
        f'{model_name.upper()} - {suite_label} (Attack Suite)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    output_file = output_dir / f"{suite_name}_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def _session_rates_from_session_data(
    session_data: Dict[str, Dict[str, SessionDataTuple]],
) -> List[Tuple[str, str, List[float]]]:
    """
    Convert session_data to list of (defense, backend, [rate_1, ..., rate_5]) for valid combinations.
    rate_s is (passed/total*100) or np.nan if total_s==0.
    """
    out: List[Tuple[str, str, List[float]]] = []
    for defense_type in UNIFIED_DEFENSE_TYPES:
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            sess = session_data.get(defense_type, {}).get(backend, [(0, 0)] * 5)
            rates = []
            for p, t in sess:
                rates.append((p / t * 100) if t > 0 else float(np.nan))
            out.append((defense_type, backend, rates))
    return out


def _aggregate_session_data_across_suites(
    all_suites_session_data: Dict[str, Dict[str, Dict[str, SessionDataTuple]]],
) -> Dict[str, Dict[str, SessionDataTuple]]:
    """
    Pool (passed, total) per session across all suites for each (defense, backend).
    Returns session_data with the same structure suitable for plotting.
    """
    aggregated: Dict[str, Dict[str, SessionDataTuple]] = {}
    for defense_type in UNIFIED_DEFENSE_TYPES:
        aggregated[defense_type] = {}
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            session_passed = [0] * 5
            session_total = [0] * 5
            for suite_name, session_data in all_suites_session_data.items():
                sess = session_data.get(defense_type, {}).get(backend, [(0, 0)] * 5)
                for s in range(5):
                    p, t = sess[s] if s < len(sess) else (0, 0)
                    session_passed[s] += p
                    session_total[s] += t
            aggregated[defense_type][backend] = [
                (session_passed[s], session_total[s]) for s in range(5)
            ]
    return aggregated


def _generate_session_plots_impl(
    session_data: Dict[str, Dict[str, SessionDataTuple]],
    model_name: str,
    suite_label: str,
    file_prefix: str,
    output_dir: Path,
) -> List[Path]:
    """
    Internal: generate the 3 session attack rate plots with given title label and file prefix.
    Returns paths to the 3 saved figures.
    """
    lines_data = _session_rates_from_session_data(session_data)
    if not lines_data:
        return []

    sessions = [1, 2, 3, 4, 5]
    backend_colors = {b: plt.cm.tab10(MEMORY_BACKENDS.index(b) % 10) for b in MEMORY_BACKENDS}
    defense_colors = {d: plt.cm.Set2(UNIFIED_DEFENSE_TYPES.index(d) % 8) for d in UNIFIED_DEFENSE_TYPES}
    backend_label = {b: BACKEND_LABELS[MEMORY_BACKENDS.index(b)] for b in MEMORY_BACKENDS}
    defense_label = {d: d.replace("_", " ").title() for d in UNIFIED_DEFENSE_TYPES}

    out_files: List[Path] = []

    # 1) One plot, all lines
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for defense_type, backend, rates in lines_data:
        label = f"{backend_label[backend]} / {defense_label[defense_type]}"
        ax1.plot(sessions, rates, "o-", label=label, linewidth=1.5, markersize=4)
    ax1.set_xlabel("Session (trigger) number", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Attack success rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title(f"{model_name.upper()} - {suite_label}\nAttack success rate by session (all combinations)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    f1 = output_dir / f"{file_prefix}_session_attack_rate_all.png"
    plt.savefig(f1, dpi=300, bbox_inches="tight")
    plt.close()
    out_files.append(f1)

    # 2) Subplots by memory backend (each subplot: 5 defense lines)
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    axes2_flat = axes2.flat
    for idx, backend in enumerate(MEMORY_BACKENDS):
        ax = axes2_flat[idx]
        for defense_type in UNIFIED_DEFENSE_TYPES:
            if not is_valid_combination(backend, defense_type):
                continue
            sess = session_data.get(defense_type, {}).get(backend, [(0, 0)] * 5)
            rates = [(p / t * 100) if t > 0 else float(np.nan) for p, t in sess]
            ax.plot(sessions, rates, "o-", label=defense_label[defense_type], color=defense_colors[defense_type], linewidth=1.5, markersize=4)
        ax.set_xlabel("Session number")
        ax.set_ylabel("Attack success rate (%)")
        ax.set_title(backend_label[backend])
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # Hide the 6th subplot (we have 5 backends)
    axes2_flat[5].set_visible(False)
    fig2.suptitle(f"{model_name.upper()} - {suite_label}\nAttack success rate by session (by memory backend)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    f2 = output_dir / f"{file_prefix}_session_attack_rate_by_backend.png"
    plt.savefig(f2, dpi=300, bbox_inches="tight")
    plt.close()
    out_files.append(f2)

    # 3) Subplots by defense (each subplot: 5 backend lines)
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
    axes3_flat = axes3.flat
    for idx, defense_type in enumerate(UNIFIED_DEFENSE_TYPES):
        ax = axes3_flat[idx]
        for backend in MEMORY_BACKENDS:
            if not is_valid_combination(backend, defense_type):
                continue
            sess = session_data.get(defense_type, {}).get(backend, [(0, 0)] * 5)
            rates = [(p / t * 100) if t > 0 else float(np.nan) for p, t in sess]
            ax.plot(sessions, rates, "o-", label=backend_label[backend], color=backend_colors[backend], linewidth=1.5, markersize=4)
        ax.set_xlabel("Session number")
        ax.set_ylabel("Attack success rate (%)")
        ax.set_title(defense_label[defense_type])
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes3_flat[5].set_visible(False)
    fig3.suptitle(f"{model_name.upper()} - {suite_label}\nAttack success rate by session (by defense)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    f3 = output_dir / f"{file_prefix}_session_attack_rate_by_defense.png"
    plt.savefig(f3, dpi=300, bbox_inches="tight")
    plt.close()
    out_files.append(f3)

    return out_files


def generate_session_plots(
    model_name: str,
    suite_name: str,
    session_data: Dict[str, Dict[str, SessionDataTuple]],
    output_dir: Path,
) -> List[Path]:
    """
    Generate 3 session-by-session attack success rate plots (no utility):
    1. One plot with all (backend, defense) lines.
    2. Subplots by memory backend (5 subplots, each with defense lines).
    3. Subplots by defense (5 subplots, each with backend lines).
    Returns paths to the 3 saved figures.
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )
    suite_label = suite_name.replace("_", " ").title()
    return _generate_session_plots_impl(
        session_data, model_name, suite_label, suite_name, output_dir
    )


def generate_averaged_session_plots(
    model_name: str,
    all_suites_session_data: Dict[str, Dict[str, Dict[str, SessionDataTuple]]],
    output_dir: Path,
) -> List[Path]:
    """
    Generate 3 session attack rate plots averaged across all topics (suites):
    attack rate all, by backend, by defense. Uses pooled (passed, total) across suites.
    Returns paths to the 3 saved figures.
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )
    if not all_suites_session_data:
        return []
    aggregated = _aggregate_session_data_across_suites(all_suites_session_data)
    return _generate_session_plots_impl(
        aggregated,
        model_name,
        "Average Across Attack Suites",
        "average",
        output_dir,
    )


def generate_combined_heatmaps_subplot(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate a combined plot: for each suite, three subplots (Utility, Attack, Max Attack)."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    suite_names = sorted(all_data.keys())
    n_suites = len(suite_names)
    fig, axes = plt.subplots(n_suites, 3, figsize=(22, 5 * n_suites))
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
            f'{suite_label} - Attack (Total)'
        )
        _draw_one_heatmap(
            axes[idx, 2], data, "max_attack",
            f'{suite_label} - Max Attack'
        )

    fig.suptitle(
        f'{model_name.upper()} - Attack Suites (Utility & Attack Success Rates)',
        fontsize=16, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = output_dir / "all_suites_combined_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_test_train_combined_heatmap(
    model_name: str,
    all_suites_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    all_suites_train_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """
    Generate one figure with Test (attack) and Train heatmaps side by side.
    Left: for each suite, 3 heatmaps (Utility, Attack Total, Max Attack) from test.
    Right: for each suite, 2 heatmaps (Utility, Attack) from train.
    Rows are unified across suites that appear in either test or train; missing data shows empty.
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )
    unified_suites = sorted(set(all_suites_data.keys()) | set(all_suites_train_data.keys()))
    if not unified_suites:
        raise ValueError("No suites in test or train data")
    n_suites = len(unified_suites)
    # 5 columns: 3 test (Utility, Attack, Max Attack) + 2 train (Utility, Attack)
    fig, axes = plt.subplots(n_suites, 5, figsize=(28, 5 * n_suites))
    if n_suites == 1:
        axes = axes.reshape(1, -1)

    for idx, suite_name in enumerate(unified_suites):
        suite_label = suite_name.replace("_", " ").title()
        # Test: columns 0, 1, 2
        if suite_name in all_suites_data:
            data = all_suites_data[suite_name]
            _draw_one_heatmap(
                axes[idx, 0], data, "utility",
                f'{suite_label} - Utility'
            )
            _draw_one_heatmap(
                axes[idx, 1], data, "attack",
                f'{suite_label} - Attack (Total)'
            )
            _draw_one_heatmap(
                axes[idx, 2], data, "max_attack",
                f'{suite_label} - Max Attack'
            )
        else:
            for c in range(3):
                axes[idx, c].set_visible(False)
        # Train: columns 3, 4
        if suite_name in all_suites_train_data:
            data = all_suites_train_data[suite_name]
            _draw_one_heatmap(
                axes[idx, 3], data, "utility",
                f'{suite_label} - Utility'
            )
            _draw_one_heatmap(
                axes[idx, 4], data, "attack",
                f'{suite_label} - Attack'
            )
        else:
            for c in range(3, 5):
                axes[idx, c].set_visible(False)

    # Column group labels (centered over left 3 cols and right 2 cols)
    fig.text(0.30, 0.995, 'Test (Attack)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.78, 0.995, 'Train', ha='center', fontsize=14, fontweight='bold')
    fig.suptitle(
        f'{model_name.upper()} - All Suites: Test vs Train (Utility & Attack Success Rates)',
        fontsize=16, fontweight='bold', y=1.005
    )
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    output_file = output_dir / "all_suites_test_and_train_combined_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_average_heatmap(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate three heatmaps: average Utility, Attack (total), and Max Attack success rates across suites."""
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
                        if t[8]:
                            has_any_errors = True
                        elif t[total_idx] > 0:
                            if rate_idx == 6 and total_idx == 7:
                                rates.append((t[6] / t[7] * 100) if t[7] > 0 else 0.0)
                            else:
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

    fig, (ax_util, ax_attack, ax_max) = plt.subplots(1, 3, figsize=(22, 8))

    for ax, rate_idx, total_idx, label in [
        (ax_util, 2, 1, "Utility (user goal)"),
        (ax_attack, 5, 4, "Attack (attack goal)"),
        (ax_max, 6, 7, "Max Attack (any trigger)"),
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
    output_file = output_dir / "average_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_combined_csv(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate a combined CSV listing all attack suites with Utility and Attack columns."""
    output_file = output_dir / "all_suites_combined.csv"

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
                for metric_name, rate_idx, total_idx in [
                    ("Utility", 2, 1), ("Attack (Total)", 5, 4), ("Max Attack", 6, 7),
                ]:
                    row = [suite_label, dt_label, metric_name]
                    for backend in MEMORY_BACKENDS:
                        if not is_valid_combination(backend, defense_type):
                            row.append("-")
                            continue
                        t = data[defense_type].get(backend, _default_metric_tuple())
                        if t[8]:
                            row.append("ERR")
                        elif t[total_idx] > 0:
                            rate = (t[6] / t[7] * 100) if (rate_idx == 6 and total_idx == 7) else t[rate_idx]
                            row.append(f"{rate:.1f}%")
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
    output_file = output_dir / "average_summary.csv"

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
                ("Utility", 2, 1), ("Attack (Total)", 5, 4), ("Max Attack", 6, 7),
            ]:
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
                            if t[8]:
                                has_any_errors = True
                            elif t[total_idx] > 0:
                                if rate_idx == 6 and total_idx == 7:
                                    rates.append((t[6] / t[7] * 100) if t[7] > 0 else 0.0)
                                else:
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

    summary_file = output_dir / f"{suite_name}_execution_errors.txt"
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
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Suites for {model_name}: {', '.join(suites_to_process)}")
        all_suites_data = {}
        all_suites_session_data: Dict[str, Dict[str, Dict[str, SessionDataTuple]]] = {}

        for suite_name in suites_to_process:
            print(f"Processing: {model_name} / {suite_name}")

            data, session_data, data_stealth = collect_all_data(model_name, suite_name, results_base_dir)
            all_suites_data[suite_name] = data
            all_suites_session_data[suite_name] = session_data

            csv_file = generate_csv(model_name, suite_name, data, model_output_dir, data_stealth=data_stealth)
            csv_files.append(csv_file)
            print(f"  OK: CSV: {csv_file.name}")

            error_summary = generate_error_summary(
                model_name, suite_name, results_base_dir, logs_base_dir, model_output_dir
            )
            if error_summary:
                error_summaries.append(error_summary)
                print(f"  WARNING: Error Summary: {error_summary.name}")
            else:
                print(f"  OK: No execution errors found")

            if not args.no_plots and PLOTTING_AVAILABLE:
                try:
                    heatmap_file = generate_heatmap(
                        model_name, suite_name, data, model_output_dir
                    )
                    plot_files.append(heatmap_file)
                    print(f"  OK: Heatmap: {heatmap_file.name}")
                except Exception as e:
                    print(f"  WARNING: Error generating heatmap: {e}")
                try:
                    session_plot_files = generate_session_plots(
                        model_name, suite_name, session_data, model_output_dir
                    )
                    for sp in session_plot_files:
                        plot_files.append(sp)
                    if session_plot_files:
                        print(f"  OK: Session plots: {len(session_plot_files)} file(s)")
                except Exception as e:
                    print(f"  WARNING: Error generating session plots: {e}")
            elif not args.no_plots:
                print("  WARNING: Skipping plots (matplotlib/seaborn not installed)")

        # Combined visualizations (one suite: still generate for consistency)
        if not args.no_plots and PLOTTING_AVAILABLE:
            print(f"\nGenerating combined visualizations for {model_name}...")
            try:
                combined_heatmap_file = generate_combined_heatmaps_subplot(
                    model_name, all_suites_data, model_output_dir
                )
                plot_files.append(combined_heatmap_file)
                print(f"  OK: Combined Heatmaps: {combined_heatmap_file.name}")
            except Exception as e:
                print(f"  WARNING: Error generating combined visualizations: {e}")

        # Load train results and generate test+train side-by-side heatmap
        train_results_dir = results_base_dir.parent / "train"
        all_suites_train_data: Dict[str, Dict[str, Dict[str, MetricTuple]]] = {}
        if train_results_dir.exists():
            train_suites = discover_suites_for_train(train_results_dir, model_name)
            if train_suites:
                for suite_name in train_suites:
                    data, _, _ = collect_all_data(
                        model_name, suite_name, train_results_dir, train_only=True
                    )
                    all_suites_train_data[suite_name] = data
        if all_suites_train_data and not args.no_plots and PLOTTING_AVAILABLE:
            try:
                test_train_heatmap_file = generate_test_train_combined_heatmap(
                    model_name, all_suites_data, all_suites_train_data, model_output_dir
                )
                plot_files.append(test_train_heatmap_file)
                print(f"  OK: Test+Train Combined Heatmap: {test_train_heatmap_file.name}")
            except Exception as e:
                print(f"  WARNING: Error generating test+train combined heatmap: {e}")

        # Combined and average CSV/heatmap
        print(f"\nGenerating combined CSV files for {model_name}...")
        try:
            combined_csv_file = generate_combined_csv(
                model_name, all_suites_data, model_output_dir
            )
            csv_files.append(combined_csv_file)
            print(f"  OK: Combined CSV: {combined_csv_file.name}")

            avg_csv_file = generate_average_summary_csv(
                model_name, all_suites_data, model_output_dir
            )
            csv_files.append(avg_csv_file)
            print(f"  OK: Average Summary CSV: {avg_csv_file.name}")

            if not args.no_plots and PLOTTING_AVAILABLE:
                try:
                    avg_heatmap_file = generate_average_heatmap(
                        model_name, all_suites_data, model_output_dir
                    )
                    plot_files.append(avg_heatmap_file)
                    print(f"  OK: Average Heatmap: {avg_heatmap_file.name}")
                except Exception as e:
                    print(f"  WARNING: Error generating average heatmap: {e}")
                try:
                    avg_session_plot_files = generate_averaged_session_plots(
                        model_name, all_suites_session_data, model_output_dir
                    )
                    for sp in avg_session_plot_files:
                        plot_files.append(sp)
                    if avg_session_plot_files:
                        print(f"  OK: Averaged session plots (across topics): {len(avg_session_plot_files)} file(s)")
                except Exception as e:
                    print(f"  WARNING: Error generating averaged session plots: {e}")
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
