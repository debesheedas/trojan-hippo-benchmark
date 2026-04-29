#!/usr/bin/env python3
"""
Plot undefended ASR (%) by memory backend for two models.

Reads consolidated attack summaries and uses:
  Defense Type = None, Metric = Attack (Total)

Output:
  CCS_thesis_manuscript/figures/undefended_asr_by_backend.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ATTACK_ROOT = REPO / "data" / "benchmark" / "consolidated_attack_results"
OUTPUT_DIR = REPO / "CCS_thesis_manuscript" / "figures"
TEST_SUITE = "test_100"

BACKENDS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
BACKEND_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
ZERO_MARKER_LINEWIDTH = 3.6
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 18

MODELS = [
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
    ("gpt-5-mini", "GPT-5-mini"),
]


def parse_pct(value: str) -> float | None:
    s = (value or "").strip().rstrip("%").strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_undefended_asr(csv_path: Path) -> list[float]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Defense Type", "").strip() == "None" and row.get("Metric", "").strip() == "Attack (Total)":
                values: list[float] = []
                for backend in BACKENDS:
                    raw = row.get(f"{backend} (%)", "")
                    v = parse_pct(raw)
                    values.append(float("nan") if v is None else v)
                return values
    raise RuntimeError(f"Could not find undefended ASR row in {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for output figure files")
    parser.add_argument("--basename", default="undefended_asr_by_backend", help="Output filename without extension")
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_values: list[list[float]] = []
    for model_dir, _ in MODELS:
        csv_path = ATTACK_ROOT / model_dir / TEST_SUITE / "average_summary.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path}", file=sys.stderr)
            return 1
        model_values.append(load_undefended_asr(csv_path))

    x = np.arange(len(MODELS), dtype=float)
    n_backends = len(BACKENDS)
    width = 0.15
    offsets = (np.arange(n_backends) - (n_backends - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(11.0, 7.2), constrained_layout=True)

    for i, backend in enumerate(BACKENDS):
        heights = np.array([model_values[m][i] for m in range(len(MODELS))], dtype=float)
        mask = ~np.isnan(heights)
        if not np.any(mask):
            continue
        x_pos = (x + offsets[i])[mask]
        y_vals = heights[mask]
        ax.bar(
            x_pos,
            y_vals,
            width=width,
            label=backend,
            color=BACKEND_COLORS[i],
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

        # Show true 0% values as a thin colored marker at y=0 (distinct from missing values).
        zero_mask = np.isclose(y_vals, 0.0)
        if np.any(zero_mask):
            for xp in x_pos[zero_mask]:
                ax.hlines(
                    y=0.0,
                    xmin=xp - width * 0.42,
                    xmax=xp + width * 0.42,
                    colors=BACKEND_COLORS[i],
                    linewidth=ZERO_MARKER_LINEWIDTH,
                    zorder=4,
                )

    ax.set_ylabel("ASR (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in MODELS], fontweight="bold", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.yaxis.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        title="Memory Backend",
        ncol=len(BACKENDS),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        edgecolor="0.8",
        fancybox=False,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )

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
