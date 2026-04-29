#!/usr/bin/env python3
"""
Grouped bar charts of ASR (%) by memory backend and defense at test_100.

Reads the same consolidated CSVs as generate_defense_effectiveness_table.py.
Output: CCS_thesis_manuscript/figures/defense_effectiveness_asr_test_100.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_defense_effectiveness_table import (  # noqa: E402
    ATTACK,
    BACKENDS,
    DEFENSES_CSV,
    MODELS,
    load_metric_rows,
)

OUTPUT_DIR = REPO / "CCS_thesis_manuscript" / "figures"

# Subplot titles (figure-only; CSV paths still use MODELS directory names)
MODEL_TITLE = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5-mini": "GPT-5-mini",
}

# Short x-axis labels (aligned with table)
BACKEND_XLABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]

# Legend labels (figure display; consolidated CSVs still use DEFENSES_CSV keys).
DEFENSE_DISPLAY = {
    "None": "None",
    "User Prompt Only": "User prompt only",
    "No Untrusted Tools": "No untrusted write",
    "Limit Memory Length": "Limit memory length",
    "Provable Policy": "Provable policy",
}

# Colorblind-friendly distinct colors (one per defense)
DEFENSE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
]

# Y-axis extends below 0 so the "0" tick sits clearly above the bottom band. All bars share
# the same baseline y = -ZERO_STUB_DEPTH; true ASR=0% is a stub to y=0 (missing stays absent).
Y_MARGIN_BELOW_ZERO = 6.5
ZERO_STUB_DEPTH = 4.0
BAR_BASELINE = -ZERO_STUB_DEPTH

# Horizontal width: higher = wider bars / wider defense cluster (capped so adjacent backends do not overlap).
BAR_GROUP_SPAN = 0.96
X_AXIS_PAD = 0.06

# Typography controls
AXIS_LABEL_FONTSIZE = 21
TICK_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def plot_model_ax(ax, data: dict, title: str) -> None:
    """Grouped bars: x = memory backends, hue = defense; ASR only."""
    n_b = len(BACKENDS)
    n_d = len(DEFENSES_CSV)
    x = np.arange(n_b, dtype=float)
    width = min(0.24, BAR_GROUP_SPAN / (n_d + 1))
    offsets = (np.arange(n_d) - (n_d - 1) / 2.0) * width

    for i, d in enumerate(DEFENSES_CSV):
        row = data.get(d, {}).get("Attack (Total)", [None] * n_b)
        h = np.array(
            [row[j] if j < len(row) and row[j] is not None else np.nan for j in range(n_b)],
            dtype=float,
        )
        pos = x + offsets[i]
        mask = ~np.isnan(h)
        if not np.any(mask):
            continue
        height_plot = np.where(
            np.isnan(h),
            np.nan,
            np.where(h == 0, ZERO_STUB_DEPTH, h + ZERO_STUB_DEPTH),
        )
        bottom_plot = np.where(np.isnan(h), np.nan, BAR_BASELINE)
        ax.bar(
            pos[mask],
            height_plot[mask],
            width,
            bottom=bottom_plot[mask],
            align="center",
            label=DEFENSE_DISPLAY[d],
            color=DEFENSE_COLORS[i],
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

    ax.set_ylabel("ASR (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontweight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(BACKEND_XLABELS, rotation=0, fontsize=TICK_LABEL_FONTSIZE)
    ax.axhline(0.0, color="0.55", linewidth=1.05, zorder=1)
    ax.set_ylim(-Y_MARGIN_BELOW_ZERO, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.yaxis.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    half = width / 2.0
    x_left = 0.0 + offsets[0] - half
    x_right = float(n_b - 1) + offsets[n_d - 1] + half
    ax.set_xlim(x_left - X_AXIS_PAD, x_right + X_AXIS_PAD)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PDF/PNG output",
    )
    parser.add_argument(
        "--basename",
        default="defense_effectiveness_asr_test_100",
        help="Output filename without extension",
    )
    args = parser.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.1, 7.0),
        sharex=True,
        constrained_layout=True,
    )

    for ax, (model_dir, display) in zip(axes, MODELS):
        csv_path = ATTACK / model_dir / "test_100" / "average_summary.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path}", file=sys.stderr)
            return 1
        data = load_metric_rows(csv_path)
        plot_model_ax(ax, data, MODEL_TITLE.get(model_dir, display))

    # sharex=True hides x tick labels on upper axes by default; show backends on both panels.
    for ax in axes:
        ax.tick_params(labelbottom=True)

    handles, labels = axes[0].get_legend_handles_labels()
    top_legend = fig.legend(
        handles[:3],
        labels[:3],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=LEGEND_FONTSIZE,
    )
    fig.legend(
        handles[3:],
        labels[3:],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.085),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=LEGEND_FONTSIZE,
    )
    fig.add_artist(top_legend)

    out_pdf = out_dir / f"{args.basename}.pdf"
    out_png = out_dir / f"{args.basename}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
