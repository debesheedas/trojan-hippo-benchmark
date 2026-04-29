#!/usr/bin/env python3
"""
Build space-efficient thesis figures: combined average + weighted heatmaps and weights table.

For each model, produces one figure with:
  - Left: unweighted average utility heatmap (same order as reference)
  - Right: weighted average utility heatmap
  - Weights table (use-case weights per capability class)

Uses the same canonical ordering as consolidate_results.py and weighted_utility_results.py.
Does not modify files in consolidated_results/; writes only to thesis_manuscript/figures/.

Usage:
  python scripts/build_thesis_utility_figures.py
  python scripts/build_thesis_utility_figures.py --figures-dir thesis_manuscript/figures
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
CONSOLIDATED_DIR = BASE_DIR / "data" / "benchmark" / "consolidated_results"
FIGURES_DIR = BASE_DIR / "thesis_manuscript" / "figures"

# Must match consolidate_results.py and weighted_utility_results.py
BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
DEFENSE_DISPLAY_ORDER = [
    "None", "User Prompt Only", "No Untrusted Tools",
    "Limit Memory Length", "Provable Policy",
]

# Use-case weights: Balanced profile (must match deployment_profile_weights.tex and weighted_utility_results.USE_CASE_WEIGHTS)
USE_CASE_WEIGHTS = {
    "memory_only": 0.12,
    "assistant_responses": 0.08,
    "untrusted_probe": 0.10,
    "untrusted_send": 0.08,
    "disable_send": 0.08,
    "memory_tools": 0.14,
    "long_memory": 0.40,
}
SUITE_DISPLAY_ORDER = [
    "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send",
    "disable_send", "memory_tools", "long_memory",
]


def parse_pct(s: str):
    if s is None or (isinstance(s, str) and s.strip() in ("", "-", "ERR")):
        return np.nan
    s = str(s).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_summary_csv(path: Path):
    """Load average or weighted summary CSV; return matrix (defenses × backends) in canonical order."""
    if not path.exists():
        return None, None
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        backend_cols = [h.replace(" (%)", "").strip() for h in header[1:]]
        for row in reader:
            if len(row) < 2:
                continue
            defense = row[0].strip()
            data[defense] = {}
            for i, col in enumerate(backend_cols):
                if i + 1 < len(row):
                    data[defense][col] = parse_pct(row[i + 1])
    defenses = [d for d in DEFENSE_DISPLAY_ORDER if d in data]
    if not defenses:
        defenses = sorted(data.keys())
    backends = [b for b in BACKEND_LABELS if b in (data[defenses[0]] if defenses else {})]
    if not backends:
        backends = BACKEND_LABELS
    matrix = np.array([
        [data[d].get(b, np.nan) for b in backends]
        for d in defenses
    ])
    return matrix, (defenses, backends)


def build_combined_figure(model_name: str, average_matrix, weighted_matrix, labels, out_path: Path):
    """Draw two heatmaps side by side and a weights table."""
    if not PLOTTING_AVAILABLE:
        return
    defenses, backends = labels
    mask_avg = np.isnan(average_matrix)
    mask_wgt = np.isnan(weighted_matrix)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(
        np.where(mask_avg, 0, average_matrix),
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Success (%)"},
        xticklabels=backends,
        yticklabels=defenses,
        linewidths=0.5,
        linecolor="gray",
        ax=ax0,
        mask=mask_avg,
    )
    ax0.set_title("Unweighted average", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Memory backend", fontsize=9)
    ax0.set_ylabel("Defense", fontsize=9)

    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(
        np.where(mask_wgt, 0, weighted_matrix),
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Weighted (%)"},
        xticklabels=backends,
        yticklabels=defenses,
        linewidths=0.5,
        linecolor="gray",
        ax=ax1,
        mask=mask_wgt,
    )
    ax1.set_title("Weighted average", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Memory backend", fontsize=9)
    ax1.set_ylabel("")

    # Weights table
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    suite_labels = [s.replace("_", " ").title() for s in SUITE_DISPLAY_ORDER]
    rows = [[s, f"{USE_CASE_WEIGHTS[s]:.2f}"] for s in SUITE_DISPLAY_ORDER]
    table = ax2.table(
        cellText=rows,
        colLabels=["Capability class", "Weight"],
        loc="center",
        cellLoc="left",
        colColours=["#e0e0e0"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax2.set_title("Use-case weights", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        f"{model_name} — Utility (average vs weighted)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.4)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build thesis utility combined figures.")
    parser.add_argument("--consolidated-dir", type=Path, default=CONSOLIDATED_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    consolidated_dir = args.consolidated_dir
    figures_dir = args.figures_dir
    if not consolidated_dir.exists():
        print(f"Consolidated dir not found: {consolidated_dir}")
        return 1
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not PLOTTING_AVAILABLE:
        print("matplotlib/seaborn required")
        return 1

    for model_dir in sorted(consolidated_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        avg_path = model_dir / "average_summary.csv"
        wgt_path = model_dir / "weighted_average_summary.csv"
        if not avg_path.exists() or not wgt_path.exists():
            print(f"Skip {model_name}: missing average or weighted summary CSV")
            continue
        average_matrix, labels = load_summary_csv(avg_path)
        weighted_matrix, _ = load_summary_csv(wgt_path)
        if average_matrix is None or weighted_matrix is None:
            continue
        build_combined_figure(
            model_name,
            average_matrix,
            weighted_matrix,
            labels,
            figures_dir / f"{model_name}_utility_combined.png",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
