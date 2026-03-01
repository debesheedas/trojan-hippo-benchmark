#!/usr/bin/env python3
"""
Train Results Consolidation Script with CSV and Plots

Consolidates TRAIN results only from the attack benchmark. Reads from
data/benchmark/attack_results/train/ and data/benchmark/attack_logs/train/ by default
(train/test layout matches attack_bench). Includes only result files whose filename
stem contains "train". By default writes to two output folders: consolidated_train_results
(from attack_results/attack_logs) and consolidated_train_results_stealth (from
attack_results_stealth/attack_logs_stealth). Use --results-dir/--logs-dir/--output-dir
for a single custom pass.

Use --train-folder train_10 (or train_20, train_0, etc.) for train_N layout (model/topic/backend/defense);
outputs go to data/benchmark/consolidated_<train-folder>_results and heatmaps show per-topic
success rates.

Discovers suites that have at least one train result file and generates CSVs/plots
for those suites.

Usage:
    python scripts/consolidate_train.py
    python scripts/consolidate_train.py --train-folder train_10
    python scripts/consolidate_train.py --train-folder train_20
    python scripts/consolidate_train.py --results-dir data/benchmark/attack_results/train_20
    python scripts/consolidate_train.py --model gpt-4o-mini
    python scripts/consolidate_train.py --no-plots
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

# Default paths for train (train results live under attack_results/train/)
DEFAULT_RESULTS_DIR = Path("data/benchmark/attack_results/train")
DEFAULT_LOGS_DIR = Path("data/benchmark/attack_logs/train")
DEFAULT_OUTPUT_DIR = Path("data/benchmark/consolidated_train_results")

# train_N paths (topic/suite is second level: model/topic/backend/defense/)
# Used for train_10, train_20, train_0, etc. Base dirs are derived from --train-folder.
ATTACK_RESULTS_BASE = Path("data/benchmark/attack_results")
ATTACK_LOGS_BASE = Path("data/benchmark/attack_logs")
ATTACK_RESULTS_STEALTH_BASE = Path("data/benchmark/attack_results_stealth")
ATTACK_LOGS_STEALTH_BASE = Path("data/benchmark/attack_logs_stealth")
CONSOLIDATED_BASE = Path("data/benchmark")

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
    use_train_10_layout: bool = False,
) -> Path:
    """
    Get the results directory for attack benchmark.
    Default path: {results_base_dir}/{model_name}/{memory_backend}/{defense_type}/ or +/{suite_name}/
    train_10 layout: {results_base_dir}/{model_name}/{suite_name}/{memory_backend}/{defense_type}/
    """
    if use_train_10_layout and suite_name:
        return results_base_dir / model_name / suite_name / memory_backend / unified_defense
    p = results_base_dir / model_name / memory_backend / unified_defense
    if suite_name:
        p = p / suite_name
    return p


# Filename stem marker for train test cases (this script includes ONLY files with this marker)
TRAIN_STEM_MARKER = "train"


def _iter_result_files_for_suite(
    results_dir: Path, suite_name: str, use_train_10_layout: bool = False
):
    """
    Iterate over TRAIN result JSON files for a suite (stem must contain TRAIN_STEM_MARKER).
    Supports both:
    - New layout: result files in results_dir/suite_name/*.json
    - Legacy: result files directly in results_dir/*.json
    - train_10 layout: results_dir is already model/suite/backend/defense, all *.json count
    """
    if use_train_10_layout:
        for p in results_dir.glob("*.json"):
            yield p
        return
    # New layout: defense/suite_name/*.json
    suite_dir = results_dir / suite_name
    if suite_dir.is_dir():
        for p in suite_dir.glob("*.json"):
            if suite_name in p.stem and TRAIN_STEM_MARKER in p.stem:
                yield p
    # Legacy: defense/*.json (filter by suite + train in stem)
    for p in results_dir.glob("*.json"):
        if suite_name in p.stem and TRAIN_STEM_MARKER in p.stem:
            yield p


def parse_result_file(result_file: Path, stealth_run: bool = False) -> Optional[Dict]:
    """
    Parse a result JSON file and extract user (utility) and attack goal statistics.
    When stealth_run is True, attack success = attack_goal passed AND stealth_goal passed (for steps that have both).
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
        # In stealth mode: count as passed only when BOTH attack_goal and stealth_goal pass
        if stealth_run:
            attack_passed = sum(
                1 for s in attack_steps
                if s["attack_goal"].get("passed") is True
                and s.get("stealth_goal") is not None
                and isinstance(s.get("stealth_goal"), dict)
                and s["stealth_goal"].get("passed") is True
            )
            attack_step_passed = [
                s["attack_goal"].get("passed") is True
                and s.get("stealth_goal") is not None
                and isinstance(s.get("stealth_goal"), dict)
                and s["stealth_goal"].get("passed") is True
                for s in attack_steps
            ]
        else:
            attack_passed = sum(
                1 for s in attack_steps
                if s["attack_goal"].get("passed") is True
            )
            attack_step_passed = [
                s["attack_goal"].get("passed") is True
                for s in attack_steps
            ]

        # Stealth goal: steps that have stealth_goal with passed not None
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
    use_train_10_layout: bool = False,
    stealth_run: bool = False,
) -> Tuple[MetricTuple, SessionDataTuple, Tuple[int, int, float]]:
    """
    Collect user (utility), attack goal, and stealth goal statistics for a specific combination.
    Only JSON files whose name contains the suite name are included (or all *.json for train_10).

    Returns:
        (MetricTuple, SessionDataTuple, stealth_tuple)
        MetricTuple: (user_passed, user_total, user_rate, attack_passed, attack_total, attack_rate,
                     max_attack_passed, max_attack_total, has_execution_errors)
        SessionDataTuple: [(passed_1, total_1), ..., (passed_5, total_5)] for trigger sessions 1..5
        stealth_tuple: (stealth_passed, stealth_total, stealth_rate)
    """
    results_dir = get_attack_results_dir(
        model_name,
        memory_backend,
        unified_defense,
        results_base_dir,
        suite_name=suite_name if use_train_10_layout else None,
        use_train_10_layout=use_train_10_layout,
    )

    default_session = [(0, 0)] * 5
    if not results_dir.exists():
        return ((0, 0, 0.0, 0, 0, 0.0, 0, 0, False), default_session, (0, 0, 0.0))

    result_files = list(
        _iter_result_files_for_suite(
            results_dir, suite_name, use_train_10_layout=use_train_10_layout
        )
    )

    user_passed = user_total = attack_passed = attack_total = 0
    max_attack_passed = max_attack_total = 0
    stealth_passed = stealth_total = 0
    has_execution_errors = False
    session_passed = [0] * 5
    session_total = [0] * 5

    for result_file in result_files:
        result = parse_result_file(result_file, stealth_run=stealth_run)
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


def discover_train_folders(attack_results_base: Path) -> List[str]:
    """Discover train folder names under attack_results (e.g. train, train_0, train_10, train_20)."""
    if not attack_results_base.exists():
        return []
    folders = sorted(
        d.name for d in attack_results_base.iterdir()
        if d.is_dir() and (d.name == "train" or d.name.startswith("train_"))
    )
    return folders


def _train_split_index(folder_name: str) -> int:
    """Return numeric index for train folder for ordering: train_0 -> 0, train_10 -> 10, train_100 -> 100."""
    if folder_name == "train":
        return 0
    if folder_name.startswith("train_"):
        try:
            return int(folder_name.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def discover_models(results_base_dir: Path) -> List[str]:
    """Discover all models from the attack results directory structure."""
    models = []
    if not results_base_dir.exists():
        return models
    for model_dir in results_base_dir.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)
    return sorted(models)


def discover_suites(
    results_base_dir: Path, model_name: str, use_train_10_layout: bool = False
) -> List[str]:
    """
    Discover suite names that have at least one TRAIN result file under this model.
    Default: returns suites with result JSON files with TRAIN_STEM_MARKER in the stem.
    train_10 layout: suites are direct subdirs of results_base_dir/model_name (topics).
    """
    if use_train_10_layout:
        model_dir = results_base_dir / model_name
        if not model_dir.exists():
            return []
        return sorted(
            d.name for d in model_dir.iterdir() if d.is_dir()
        )
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
                        if suite_name in p.stem and TRAIN_STEM_MARKER in p.stem:
                            suites.add(suite_name)
                            break
            # Legacy layout: result files directly in results_dir/*.json
            for p in results_dir.glob("*.json"):
                if TRAIN_STEM_MARKER not in p.stem:
                    continue
                # Stem e.g. "00_persistent_exfiltrate_health_train" -> suite "persistent_exfiltrate_health"
                parts = p.stem.split("_", 1)
                if len(parts) >= 2:
                    suite_name = parts[1][:-6] if parts[1].endswith("_train") else parts[1]
                    suites.add(suite_name)
    return sorted(suites)


StealthTuple = Tuple[int, int, float]


def collect_all_data(
    model_name: str,
    suite_name: str,
    results_base_dir: Path,
    use_train_10_layout: bool = False,
    stealth_run: bool = False,
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
                backend,
                defense_type,
                model_name,
                suite_name,
                results_base_dir,
                use_train_10_layout=use_train_10_layout,
                stealth_run=stealth_run,
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
    """Generate CSV file for a model and suite with Utility and Attack success rates. output_dir is the model subfolder (output_base/model_name)."""
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
                ("Utility", 2, 1),
                ("Attack", 5, 4),
            ]:
                row = [dt_label, metric_name]
                for backend in MEMORY_BACKENDS:
                    if not is_valid_combination(backend, defense_type):
                        row.append("-")
                        continue
                    t = data[defense_type].get(backend, _default_metric_tuple())
                    has_errors = t[8]
                    total = t[total_idx]
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
    """Prepare data matrices for heatmap. metric is 'utility' or 'attack'."""
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
    """Generate a figure with two heatmaps: Utility and Attack success rates."""
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
        f'{model_name.upper()} - {suite_label} (Train)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    output_file = output_dir / f"{suite_name}_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_combined_heatmaps_subplot(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate a combined plot: for each suite, two subplots (Utility, Attack)."""
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
        f'{model_name.upper()} - Train Suites (Utility & Attack Success Rates)',
        fontsize=16, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = output_dir / "all_suites_combined_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def generate_average_heatmap(
    model_name: str,
    all_data: Dict[str, Dict[str, Dict[str, MetricTuple]]],
    output_dir: Path,
) -> Path:
    """Generate two heatmaps: average Utility and Attack success rates across suites."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    defense_labels = [dt.replace("_", " ").title() for dt in UNIFIED_DEFENSE_TYPES]

    def _avg_matrix(rate_idx: int, total_idx: int) -> Tuple[Any, Any, Any]:
        """Return (rates_matrix, has_errors_matrix, no_data_matrix)."""
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
        f'{model_name.upper()} - Average Across Train Suites',
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
                    ("Utility", 2, 1), ("Attack", 5, 4),
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
                ("Utility", 2, 1), ("Attack", 5, 4),
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
                                rates.append(t[rate_idx])
                    if has_any_errors:
                        row.append("ERR")
                    elif rates:
                        row.append(f"{sum(rates) / len(rates):.1f}%")
                    else:
                        row.append("-")
                writer.writerow(row)

    return output_file


def generate_train_splits_asr_plot(
    model_name: str,
    model_backend_splits: Dict[str, Dict[int, Tuple[int, int]]],
    train_folders_sorted: List[str],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate one plot: ASR vs train split (train_0, train_10, ..., train_100) for defense=none,
    one line per memory backend. Same style as test session_asr (all_topics_sessions_asr_defense_none).
    """
    if not PLOTTING_AVAILABLE:
        return None
    if not model_backend_splits or not train_folders_sorted:
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        split_indices = [_train_split_index(f) for f in train_folders_sorted]
        for backend in MEMORY_BACKENDS:
            if backend not in model_backend_splits:
                continue
            sess_map = model_backend_splits[backend]
            xs, ys = [], []
            for split_idx in split_indices:
                if split_idx not in sess_map:
                    continue
                p, t = sess_map[split_idx]
                if t > 0:
                    xs.append(split_idx)
                    ys.append(p / t * 100.0)
            if xs:
                label = BACKEND_LABELS[MEMORY_BACKENDS.index(backend)]
                ax.plot(xs, ys, marker="o", label=label)
        if ax.has_data():
            ax.set_xlabel("Train split (session %)")
            ax.set_ylabel("Attack success rate (%)")
            ax.set_title(f"{model_name.upper()} - All topics: ASR vs. train split (defense=None)")
            ax.set_ylim(-5, 105)
            ax.set_xticks(split_indices)
            ax.set_xticklabels([f"train_{i}" for i in split_indices])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "train_splits_asr_defense_none.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            return out_path
        plt.close(fig)
    except Exception:
        pass
    return None


def generate_error_summary(
    model_name: str,
    suite_name: str,
    results_base_dir: Path,
    logs_base_dir: Path,
    output_dir: Path,
    use_train_10_layout: bool = False,
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
                model_name,
                memory_backend,
                unified_defense,
                results_base_dir,
                suite_name=suite_name if use_train_10_layout else None,
                use_train_10_layout=use_train_10_layout,
            )
            for result_file in _iter_result_files_for_suite(
                results_dir, suite_name, use_train_10_layout=use_train_10_layout
            ):
                result_data = parse_result_file(result_file)
                if result_data and not result_data.get("execution_success", True):
                    errors_found.append({
                        "type": "result_file",
                        "backend": memory_backend,
                        "defense": unified_defense,
                        "test_file": result_file.name,
                        "errors": result_data.get("execution_errors", []),
                    })

            if use_train_10_layout:
                log_dir = logs_base_dir / model_name / suite_name / memory_backend / unified_defense
                if log_dir.exists():
                    log_path = next(
                        (f for f in sorted(log_dir.iterdir()) if f.suffix == ".log"),
                        None,
                    )
                else:
                    log_path = None
                if log_path is None:
                    log_path = Path("/nonexistent")
            else:
                log_path = get_combination_log_path(
                    memory_backend=memory_backend,
                    unified_defense=unified_defense,
                    model_name=model_name,
                    attack_type=suite_name,
                    logs_base_dir=logs_base_dir,
                )
            log_data = parse_log_file(log_path) if log_path else None
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

    summary_file = output_dir / f"{suite_name}_train_execution_errors.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{'='*80}\n")
        f.write("EXECUTION ERROR SUMMARY (Attack Benchmark - Train)\n")
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
        description="Consolidate attack benchmark TRAIN results into CSV files and visualizations"
    )
    parser.add_argument(
        "--train-folder",
        type=str,
        default=None,
        help="Train folder to consolidate (e.g. train, train_0, train_10, train_20). If not set, discover and process all train* folders under attack_results.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Base directory for attack results (default: attack_results/<train-folder>)",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Base directory for attack logs (default: attack_logs/<train-folder>)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for consolidated files (default: consolidated_<train-folder>_results)",
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

    # When no custom dirs are given, run both normal and stealth consolidation (same logic, two output folders).
    # When custom dirs are given, run a single pass with those dirs.
    if args.results_dir is not None or args.logs_dir is not None or args.output_dir is not None:
        runs = [
            (
                "custom",
                Path(args.results_dir) if args.results_dir else ATTACK_RESULTS_BASE,
                Path(args.logs_dir) if args.logs_dir else ATTACK_LOGS_BASE,
                None,  # output_dir from args
                "",    # run_suffix for single-folder output name
            ),
        ]
    else:
        runs = [
            ("normal", ATTACK_RESULTS_BASE, ATTACK_LOGS_BASE, "consolidated_train_results", ""),
            ("stealth", ATTACK_RESULTS_STEALTH_BASE, ATTACK_LOGS_STEALTH_BASE, "consolidated_train_results_stealth", "_stealth"),
        ]

    all_csv_files = []
    all_plot_files = []
    all_error_summaries = []

    for run_name, results_base_run, logs_base_run, output_subdir, run_suffix in runs:
        # Resolve output_dir for custom run
        if output_subdir is None:
            output_dir_for_run = Path(args.output_dir) if args.output_dir else CONSOLIDATED_BASE / "consolidated_train_results"
        else:
            output_dir_for_run = None  # computed per train_folder below

        # Determine which train folder(s) to process for this run
        if args.train_folder is not None:
            train_folders = [args.train_folder]
        else:
            train_folders = discover_train_folders(results_base_run)
            if not train_folders:
                print(f"WARNING: No train folders found under {results_base_run}, skipping [{run_name}] run.\n")
                continue
            print(f"[{run_name}] Discovered train folders: {', '.join(train_folders)}\n")

        # Cross-train-split ASR (defense=none): model -> backend -> split_idx -> (passed, total)
        # Only populated when processing multiple train folders (train_0, train_10, ..., train_100)
        model_backend_train_splits: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {}
        train_folders_sorted = sorted(train_folders, key=_train_split_index)

        for train_folder in train_folders:
            # Any train_N (train_10, train_20, train_0, ...) uses topic-under-model layout; plain "train" uses default layout.
            use_train_10_layout = train_folder != "train"
            if output_subdir is not None:
                results_base_dir = results_base_run / train_folder if use_train_10_layout else DEFAULT_RESULTS_DIR
                logs_base_dir = logs_base_run / train_folder if use_train_10_layout else DEFAULT_LOGS_DIR
                if len(train_folders) > 1:
                    output_dir = CONSOLIDATED_BASE / output_subdir / train_folder
                elif use_train_10_layout:
                    output_dir = CONSOLIDATED_BASE / f"consolidated_{train_folder}_results{run_suffix}"
                else:
                    output_dir = CONSOLIDATED_BASE / output_subdir
            else:
                results_base_dir = Path(args.results_dir) if args.results_dir else (results_base_run / train_folder if use_train_10_layout else DEFAULT_RESULTS_DIR)
                logs_base_dir = Path(args.logs_dir) if args.logs_dir else (logs_base_run / train_folder if use_train_10_layout else DEFAULT_LOGS_DIR)
                output_dir = output_dir_for_run

            if not results_base_dir.exists():
                print(f"Skipping {train_folder}: results dir does not exist: {results_base_dir}")
                continue

            all_models = discover_models(results_base_dir)
            if not all_models:
                print(f"Skipping {train_folder}: no models found in {results_base_dir}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            models_to_process = [args.model] if args.model else all_models

            print(f"\n{'='*80}")
            print(f"Consolidating Train Results [{train_folder}]")
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
                suites_to_process = discover_suites(
                    results_base_dir, model_name, use_train_10_layout=use_train_10_layout
                )
                if not suites_to_process:
                    print(f"Skipping {model_name}: no train suites found in results")
                    continue
                model_output_dir = output_dir / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Suites for {model_name}: {', '.join(suites_to_process)}")
                all_suites_data = {}

                for suite_name in suites_to_process:
                    print(f"Processing: {model_name} / {suite_name}")

                    data, session_data, data_stealth = collect_all_data(
                        model_name,
                        suite_name,
                        results_base_dir,
                        use_train_10_layout=use_train_10_layout,
                        stealth_run=(run_name == "stealth"),
                    )
                    all_suites_data[suite_name] = data

                    csv_file = generate_csv(model_name, suite_name, data, model_output_dir, data_stealth=data_stealth)
                    csv_files.append(csv_file)
                    print(f"  OK: CSV: {csv_file.name}")

                    error_summary = generate_error_summary(
                        model_name,
                        suite_name,
                        results_base_dir,
                        logs_base_dir,
                        model_output_dir,
                        use_train_10_layout=use_train_10_layout,
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
                except Exception as e:
                    print(f"  WARNING: Error generating combined CSV files: {e}")

                # Accumulate ASR for defense=none across suites for cross-train-split plot (when processing all train folders)
                if len(train_folders) > 1 and "none" in UNIFIED_DEFENSE_TYPES:
                    split_idx = _train_split_index(train_folder)
                    mb_splits = model_backend_train_splits.setdefault(model_name, {})
                    for backend in MEMORY_BACKENDS:
                        if not is_valid_combination(backend, "none"):
                            continue
                        total_passed = total_total = 0
                        for suite_name, data in all_suites_data.items():
                            t = data.get("none", {}).get(backend, _default_metric_tuple())
                            total_passed += t[3]  # attack_passed
                            total_total += t[4]   # attack_total
                        if total_total > 0:
                            bmap = mb_splits.setdefault(backend, {})
                            bmap[split_idx] = (total_passed, total_total)

            all_csv_files.extend(csv_files)
            all_plot_files.extend(plot_files)
            all_error_summaries.extend(error_summaries)
            print(f"  [{train_folder}] Done: {len(csv_files)} CSV(s), {len(plot_files)} plot(s) -> {output_dir}")

        # Generate cross-train-split ASR plot (defense=none, one line per backend) when we processed multiple train folders for this run
        if (
            len(train_folders) > 1
            and model_backend_train_splits
            and not args.no_plots
            and PLOTTING_AVAILABLE
        ):
            print(f"\n{'='*80}")
            print(f"Generating train splits ASR plot (defense=None) [{run_name}]")
            print(f"{'='*80}")
            cross_output_base = CONSOLIDATED_BASE / output_subdir if output_subdir else output_dir_for_run
            for model_name, backend_splits in model_backend_train_splits.items():
                try:
                    model_cross_dir = cross_output_base / model_name
                    out_path = generate_train_splits_asr_plot(
                        model_name,
                        backend_splits,
                        train_folders_sorted,
                        model_cross_dir,
                    )
                    if out_path:
                        all_plot_files.append(out_path)
                        print(f"  OK: Train splits ASR (defense=None): {out_path.relative_to(CONSOLIDATED_BASE)}")
                except Exception as e:
                    print(f"  WARNING: Error generating train splits ASR plot for {model_name}: {e}")

    print(f"\n{'='*80}")
    print("Consolidation complete!")
    print(f"Generated {len(all_csv_files)} CSV file(s)")
    if all_error_summaries:
        print(f"\nWARNING: Generated {len(all_error_summaries)} error summary file(s)")
        for ef in all_error_summaries:
            print(f"   - {ef}")
    else:
        print("OK: No execution errors found")
    if not args.no_plots:
        print(f"Generated {len(all_plot_files)} plot file(s)")
    if args.results_dir or args.logs_dir or args.output_dir:
        print(f"Results saved to custom output (see above).")
    else:
        print(f"Results saved under: {CONSOLIDATED_BASE / 'consolidated_train_results'} and {CONSOLIDATED_BASE / 'consolidated_train_results_stealth'}")
    print(f"{'='*80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
