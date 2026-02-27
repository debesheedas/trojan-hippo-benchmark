#!/usr/bin/env python3
"""
Aggregate multiple seed CSVs (results_seed*.csv) into mean ± std and regenerate plots.

Reads all results_seed{N}.csv from data/benchmark/aggregated_test_results/,
groups by (split, model, topic, memory_backend, defense), computes mean and std
of user_rate and attack_rate (ASR) across seeds, writes mean_std_summary.csv and
generates ASR-vs-session plots with error bars (defense=none, all topics).

Usage:
    python scripts/aggregate_seed_runs.py
    python scripts/aggregate_seed_runs.py --input-dir data/benchmark/aggregated_test_results
    python scripts/aggregate_seed_runs.py --no-plots
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AGGREGATED_DIR = BASE_DIR / "data" / "benchmark" / "aggregated_test_results"
MEMORY_BACKENDS = ["none", "explicit", "mem0", "rag", "context"]
BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]

try:
    import numpy as np
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    np = None
    PLOTTING_AVAILABLE = False


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(vals: List[float]) -> float:
    if not vals or len(vals) < 2:
        return 0.0
    m = _mean(vals)
    variance = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return variance ** 0.5


def session_index_from_split(split: str) -> int:
    """test -> 0, test_10 -> 10, test_100 -> 100."""
    if split == "test":
        return 0
    if split.startswith("test_"):
        try:
            return int(split.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def load_seed_csvs(input_dir: Path) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Load all results_seed*.csv; return [(seed, rows), ...]."""
    out = []
    for p in sorted(input_dir.glob("results_seed*.csv")):
        stem = p.stem  # results_seed42
        if not stem.startswith("results_seed"):
            continue
        try:
            seed = int(stem.replace("results_seed", ""))
        except ValueError:
            continue
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        out.append((seed, rows))
    return out


def numeric(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    if v == "" or v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def aggregate_per_seed_per_combo(
    seed_rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str, str, str], Dict[str, float]]:
    """
    For one seed's rows, aggregate by (split, model, topic, memory_backend, defense).
    Returns dict: key -> {user_rate, attack_rate, user_passed, user_total, attack_passed, attack_total}.
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {"user_passed": 0, "user_total": 0, "attack_passed": 0, "attack_total": 0})
    for row in seed_rows:
        key = (
            row.get("split", ""),
            row.get("model", ""),
            row.get("topic", ""),
            row.get("memory_backend", ""),
            row.get("defense", ""),
        )
        g = groups[key]
        g["user_passed"] += int(numeric(row, "user_passed", 0))
        g["user_total"] += int(numeric(row, "user_total", 0))
        g["attack_passed"] += int(numeric(row, "attack_passed", 0))
        g["attack_total"] += int(numeric(row, "attack_total", 0))
    result = {}
    for key, g in groups.items():
        u_total = g["user_total"] or 1
        a_total = g["attack_total"] or 1
        result[key] = {
            "user_rate": (g["user_passed"] / u_total) * 100,
            "attack_rate": (g["attack_passed"] / a_total) * 100,
            "user_passed": g["user_passed"],
            "user_total": g["user_total"],
            "attack_passed": g["attack_passed"],
            "attack_total": g["attack_total"],
        }
    return result


def aggregate_per_seed_all_topics(
    seed_rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    """
    For one seed, aggregate over topics: (split, model, memory_backend, defense) -> rates.
    Used for all-topics ASR vs session plot.
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {"attack_passed": 0, "attack_total": 0})
    for row in seed_rows:
        key = (
            row.get("split", ""),
            row.get("model", ""),
            row.get("memory_backend", ""),
            row.get("defense", ""),
        )
        g = groups[key]
        g["attack_passed"] += int(numeric(row, "attack_passed", 0))
        g["attack_total"] += int(numeric(row, "attack_total", 0))
    result = {}
    for key, g in groups.items():
        a_total = g["attack_total"] or 1
        result[key] = {
            "attack_rate": (g["attack_passed"] / a_total) * 100,
            "attack_passed": g["attack_passed"],
            "attack_total": g["attack_total"],
        }
    return result


def run(
    input_dir: Path,
    no_plots: bool = False,
    model_filter: str | None = None,
) -> int:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    seed_data = load_seed_csvs(input_dir)
    if not seed_data:
        print(f"No results_seed*.csv found in {input_dir}", file=sys.stderr)
        return 1

    seeds = [s for s, _ in seed_data]
    print(f"Found {len(seed_data)} seed CSV(s): seeds {seeds}")

    # Per (split, model, topic, backend, defense): collect rate per seed, then mean & std
    # key (split, model, topic, memory_backend, defense) -> list of {user_rate, attack_rate} per seed
    combo_rates: Dict[Tuple[str, str, str, str, str], Dict[str, List[float]]] = {}
    for seed, rows in seed_data:
        per_combo = aggregate_per_seed_per_combo(rows)
        for key, vals in per_combo.items():
            if key not in combo_rates:
                combo_rates[key] = {"user_rate": [], "attack_rate": []}
            combo_rates[key]["user_rate"].append(vals["user_rate"])
            combo_rates[key]["attack_rate"].append(vals["attack_rate"])

    # Mean/std summary rows
    summary_rows = []
    for key in sorted(combo_rates.keys()):
        split, model, topic, memory_backend, defense = key
        if model_filter and model != model_filter:
            continue
        ur = combo_rates[key]["user_rate"]
        ar = combo_rates[key]["attack_rate"]
        n = len(ur)
        user_mean = _mean(ur)
        user_std = _std(ur)
        attack_mean = _mean(ar)
        attack_std = _std(ar)
        summary_rows.append({
            "split": split,
            "session_index": session_index_from_split(split),
            "model": model,
            "topic": topic,
            "memory_backend": memory_backend,
            "defense": defense,
            "n_seeds": n,
            "user_rate_mean": round(user_mean, 2),
            "user_rate_std": round(user_std, 2),
            "attack_rate_mean": round(attack_mean, 2),
            "attack_rate_std": round(attack_std, 2),
        })

    out_csv = input_dir / "mean_std_summary.csv"
    summary_columns = [
        "split", "session_index", "model", "topic", "memory_backend", "defense",
        "n_seeds", "user_rate_mean", "user_rate_std", "attack_rate_mean", "attack_rate_std",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_csv} ({len(summary_rows)} rows)")

    # Plots: ASR vs session (defense=none, all topics), one line per backend, mean ± std
    if not no_plots and PLOTTING_AVAILABLE and summary_rows:
        models = sorted(set(r["model"] for r in summary_rows))
        if model_filter:
            models = [m for m in models if m == model_filter]
        for model_name in models:
            # Build (session_index, backend) -> (mean, std) for defense=none, aggregated over topics
            # For each seed we have (split, model, backend, defense) -> attack_rate
            backend_sessions: Dict[str, Dict[int, List[float]]] = {b: {} for b in MEMORY_BACKENDS}
            for seed, rows in seed_data:
                per_combo = aggregate_per_seed_all_topics(rows)
                for (split, model, backend, defense), vals in per_combo.items():
                    if model != model_name or defense != "none":
                        continue
                    sidx = session_index_from_split(split)
                    if sidx not in backend_sessions.get(backend, {}):
                        backend_sessions.setdefault(backend, {})[sidx] = []
                    backend_sessions[backend][sidx].append(vals["attack_rate"])

            # Convert to mean and std per (backend, session)
            sessions_sorted = sorted(set(
                sidx for b in backend_sessions for sidx in backend_sessions[b]
            ))
            fig, ax = plt.subplots(figsize=(8, 5))
            for backend in MEMORY_BACKENDS:
                if backend not in backend_sessions:
                    continue
                xs, ys, yerrs = [], [], []
                for s in sessions_sorted:
                    rates = backend_sessions[backend].get(s, [])
                    if not rates:
                        continue
                    xs.append(s)
                    mean_val = _mean(rates)
                    std_val = _std(rates)
                    ys.append(float(mean_val))
                    yerrs.append(float(std_val))
                if not xs:
                    continue
                label = BACKEND_LABELS[MEMORY_BACKENDS.index(backend)]
                # Plot mean line with shaded ±1 std region
                ax.plot(xs, ys, marker="o", label=label)
                lower = [m - e for m, e in zip(ys, yerrs)]
                upper = [m + e for m, e in zip(ys, yerrs)]
                ax.fill_between(xs, lower, upper, alpha=0.15)
            if ax.has_data():
                ax.set_xlabel("Session index")
                ax.set_ylabel("Attack success rate (%)")
                ax.set_title(f"{model_name} - All topics: ASR vs. session (defense=None)\nmean ± std over {len(seeds)} seed(s)")
                ax.set_ylim(-5, 105)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                fig.tight_layout()
                plot_path = input_dir / f"asr_vs_session_defense_none_{model_name.replace('.', '_')}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  Plot: {plot_path}")
    elif no_plots:
        print("Plots skipped (--no-plots)")
    elif not PLOTTING_AVAILABLE:
        print("Plots skipped (matplotlib/numpy not available)")

    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate results_seed*.csv into mean ± std and generate plots."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_AGGREGATED_DIR,
        help="Directory containing results_seed*.csv (default: data/benchmark/aggregated_test_results)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only include this model in summary and plots",
    )
    args = parser.parse_args()
    return run(
        input_dir=args.input_dir.resolve(),
        no_plots=args.no_plots,
        model_filter=args.model,
    )


if __name__ == "__main__":
    sys.exit(main())
