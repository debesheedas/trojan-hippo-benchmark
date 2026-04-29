#!/usr/bin/env python3
"""
Weighted Utility Results: use-case-weighted average across capability classes.

Reads consolidated_results per model, applies a use-case weight to each capability
class (reflecting how often a typical email-agent user would need that capability),
and outputs:
- weighted_average_summary.csv
- weighted_utility_heatmap.png
- weighted_utility_bars.png (bar chart: defense vs weighted utility, one series per backend)
- use_case_weights.csv (for the paper)

Usage:
  python scripts/weighted_utility_results.py
  python scripts/weighted_utility_results.py --output-figures thesis_manuscript/figures
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Optional plotting (use non-interactive backend for scripts)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

BASE_DIR = Path(__file__).resolve().parent.parent
CONSOLIDATED_DIR = BASE_DIR / "data" / "benchmark" / "consolidated_results"
# Canonical order: must match consolidate_results.py (UNIFIED_DEFENSE_TYPES, MEMORY_BACKENDS)
BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
DEFENSE_DISPLAY_ORDER = [
    "None", "User Prompt Only", "No Untrusted Tools",
    "Limit Memory Length", "Provable Policy",
]
SUITE_TO_KEY = {
    "Assistant Responses": "assistant_responses",
    "Disable Send": "disable_send",
    "Long Memory": "long_memory",
    "Memory Only": "memory_only",
    "Memory Tools": "memory_tools",
    "Untrusted Probe": "untrusted_probe",
    "Untrusted Send": "untrusted_send",
}

# Use-case weights: Balanced profile (typical personal assistant with email).
# Sum = 1.0. Must match deployment_profile_weights.tex (Balanced column).
USE_CASE_WEIGHTS = {
    "memory_only": 0.12,
    "assistant_responses": 0.08,
    "untrusted_probe": 0.10,
    "untrusted_send": 0.08,
    "disable_send": 0.08,
    "memory_tools": 0.14,
    "long_memory": 0.40,
}


def parse_pct(s: str):
    """Parse '85.0%' or 'ERR' or '-' to float or None."""
    s = (s or "").strip()
    if s in ("", "-", "ERR"):
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def load_all_suites_combined(model_dir: Path):
    """
    Load all_suites_combined.csv for a model.
    Returns: dict suite_key -> dict defense -> dict backend_label -> rate (float or None)
    """
    path = model_dir / "all_suites_combined.csv"
    if not path.exists():
        return {}
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # header: Test Suite, Defense Type, No Memory (%), Explicit (%), ...
        if len(header) < 3:
            return {}
        backend_cols = header[2:]
        # Normalize backend labels: "No Memory (%)" -> "No Memory"
        def norm(s):
            return s.replace(" (%)", "").strip() if s else s
        for row in reader:
            if len(row) < 2:
                continue
            suite_label, defense = row[0].strip(), row[1].strip()
            key = SUITE_TO_KEY.get(suite_label)
            if key is None:
                continue
            if key not in data:
                data[key] = {}
            data[key][defense] = {}
            for i, backend_label in enumerate(backend_cols):
                if i + 2 < len(row):
                    val = parse_pct(row[i + 2])
                    data[key][defense][norm(backend_label)] = val
    return data


def compute_weighted_average(suite_data: dict, weights: dict):
    """
    suite_data: from load_all_suites_combined (suite -> defense -> backend -> rate)
    weights: suite_key -> weight (float), sum 1.0
    Returns: defense -> backend -> weighted_rate (float), or None if no data
    """
    defenses = set()
    backends = set()
    for defense_dict in suite_data.values():
        defenses |= set(defense_dict.keys())
        for backend_dict in defense_dict.values():
            backends |= set(backend_dict.keys())
    # Keep canonical order for defenses; backends use BACKEND_LABELS order
    defenses = [d for d in DEFENSE_DISPLAY_ORDER if d in defenses] or sorted(defenses)
    backends = [b for b in BACKEND_LABELS if b in backends] or sorted(backends)
    out = {}
    for defense in defenses:
        out[defense] = {}
        for backend in backends:
            total_weight = 0.0
            weighted_sum = 0.0
            for suite_key, w in weights.items():
                if w <= 0 or suite_key not in suite_data:
                    continue
                rate = suite_data[suite_key].get(defense, {}).get(backend)
                if rate is not None:
                    weighted_sum += w * rate
                    total_weight += w
            if total_weight > 0:
                out[defense][backend] = weighted_sum / total_weight
            else:
                out[defense][backend] = None
    return out


def write_weighted_csv(weighted_rates: dict, out_path: Path, backends: list = None):
    """Write weighted_average_summary.csv. Uses canonical defense order."""
    if backends is None:
        backends = BACKEND_LABELS
    defenses = [d for d in DEFENSE_DISPLAY_ORDER if d in weighted_rates] or sorted(weighted_rates.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Defense Type"] + [f"{b} (%)" for b in backends])
        for defense in defenses:
            row = [defense.replace("_", " ").title()]
            for backend in backends:
                v = weighted_rates[defense].get(backend)
                if v is None:
                    row.append("-")
                else:
                    row.append(f"{v:.1f}%")
            w.writerow(row)


def write_use_case_weights_csv(out_path: Path):
    """Write use_case_weights.csv for the paper."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Capability class", "Weight", "Description (short)"])
        desc = {
            "memory_only": "Recall user-stated facts (no tools)",
            "assistant_responses": "Recall from assistant's prior answers",
            "untrusted_probe": "Recall after reading inbox",
            "untrusted_send": "Send email with recalled info after inbox read",
            "disable_send": "Send email with content given in query",
            "memory_tools": "Combine memory + inbox in one email",
            "long_memory": "Recall despite large intervening context",
        }
        for key, weight in USE_CASE_WEIGHTS.items():
            w.writerow([key, f"{weight:.2f}", desc.get(key, "")])


def plot_weighted_heatmap(weighted_rates: dict, model_name: str, out_path: Path):
    """Produce heatmap of weighted utility (defense x backend). Uses canonical defense/backend order."""
    if not PLOTTING_AVAILABLE:
        return
    defenses = [d for d in DEFENSE_DISPLAY_ORDER if d in weighted_rates]
    if not defenses:
        defenses = sorted(weighted_rates.keys())
    defense_labels = defenses
    backends = BACKEND_LABELS
    matrix = []
    for defense in defenses:
        row = []
        for backend in backends:
            v = weighted_rates[defense].get(backend)
            row.append(float(v) if v is not None else np.nan)
        matrix.append(row)
    matrix = np.array(matrix)
    mask = np.isnan(matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Weighted utility (%)"},
        xticklabels=backends,
        yticklabels=defense_labels,
        linewidths=1,
        linecolor="gray",
        ax=ax,
        mask=mask,
    )
    ax.set_xlabel("Memory backend", fontsize=12, fontweight="bold")
    ax.set_ylabel("Defense", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{model_name} — Weighted utility (use-case weighted average)\n"
        "Higher = better benign utility for a typical email-agent workload.",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_weighted_bars(weighted_rates: dict, model_name: str, out_path: Path):
    """Bar chart: x = defense, y = weighted utility (%), one bar per backend (grouped). Uses canonical order."""
    if not PLOTTING_AVAILABLE:
        return
    defenses = [d for d in DEFENSE_DISPLAY_ORDER if d in weighted_rates]
    if not defenses:
        defenses = sorted(weighted_rates.keys())
    defense_labels = defenses
    backends = BACKEND_LABELS
    x = np.arange(len(defense_labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, backend in enumerate(backends):
        vals = []
        for defense in defenses:
            v = weighted_rates[defense].get(backend)
            vals.append(v if v is not None else 0)
        offset = (i - len(backends) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=backend)
    ax.set_xticks(x)
    ax.set_xticklabels(defense_labels, rotation=45, ha="right")
    ax.set_ylabel("Weighted utility (%)", fontsize=11)
    ax.set_xlabel("Defense", fontsize=11)
    ax.set_title(
        f"{model_name} — Weighted utility by defense and backend\n"
        "Use-case weighted average across capability classes.",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute weighted utility and write figures.")
    parser.add_argument(
        "--consolidated-dir",
        type=Path,
        default=CONSOLIDATED_DIR,
        help="Consolidated results directory (per-model subdirs)",
    )
    parser.add_argument(
        "--output-figures",
        type=Path,
        default=None,
        help="Copy figures here (e.g. thesis_manuscript/figures). If not set, write next to consolidated dir.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Override weights: key=val key=val ... (e.g. memory_only=0.3 disable_send=0.2). Must sum to 1.",
    )
    args = parser.parse_args()
    consolidated_dir = args.consolidated_dir
    if not consolidated_dir.exists():
        print(f"Consolidated dir not found: {consolidated_dir}")
        return 1
    weights = dict(USE_CASE_WEIGHTS)
    if args.weights:
        for pair in args.weights:
            if "=" in pair:
                k, v = pair.split("=", 1)
                weights[k.strip()] = float(v.strip())
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            print(f"WARNING: Weights sum to {total}, normalizing to 1.0")
            for k in weights:
                weights[k] /= total

    models = [d.name for d in consolidated_dir.iterdir() if d.is_dir()]
    if not models:
        print("No model subdirs found in", consolidated_dir)
        return 1

    figures_out = args.output_figures
    if figures_out:
        figures_out = Path(figures_out)
        figures_out.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_dir = consolidated_dir / model_name
        suite_data = load_all_suites_combined(model_dir)
        if not suite_data:
            print(f"Skip {model_name}: no all_suites_combined.csv")
            continue
        weighted_rates = compute_weighted_average(suite_data, weights)
        # Write CSV into model dir
        write_weighted_csv(weighted_rates, model_dir / "weighted_average_summary.csv")
        print(f"Wrote {model_name}/weighted_average_summary.csv")
        if PLOTTING_AVAILABLE:
            plot_weighted_heatmap(
                weighted_rates,
                model_name,
                model_dir / "weighted_utility_heatmap.png",
            )
            plot_weighted_bars(
                weighted_rates,
                model_name,
                model_dir / "weighted_utility_bars.png",
            )
            print(f"  weighted_utility_heatmap.png, weighted_utility_bars.png")
            if figures_out:
                for name in ("weighted_utility_heatmap.png", "weighted_utility_bars.png"):
                    src = model_dir / name
                    if src.exists():
                        dest = figures_out / f"{model_name}_{name}"
                        import shutil
                        shutil.copy2(src, dest)
                        print(f"  Copied to {dest}")

    # Use-case weights CSV (single file, in first model dir or figures)
    use_case_path = consolidated_dir / models[0] / "use_case_weights.csv"
    write_use_case_weights_csv(use_case_path)
    print(f"Wrote use_case_weights.csv to {use_case_path}")
    if figures_out:
        write_use_case_weights_csv(figures_out / "use_case_weights.csv")
        # Copy average_heatmap and all_suites_combined heatmap for appendix
        for model_name in models:
            model_dir = consolidated_dir / model_name
            for name in ("average_heatmap.png", "all_suites_combined_heatmap.png"):
                src = model_dir / name
                if src.exists():
                    import shutil
                    dest = figures_out / f"{model_name}_{name}"
                    shutil.copy2(src, dest)
                    print(f"Copied {name} -> {figures_out.name}/{dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
