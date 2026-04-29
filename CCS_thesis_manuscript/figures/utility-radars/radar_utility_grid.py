"""Radar grid: one panel per (model, memory-backend, defense), sized for paper inclusion.

A single figure stacks both models vertically. Per model:
  rows    = 4 memory-backends (Explicit, Mem0, RAG, Context)
  columns = 4 defenses        (None baseline omitted; tautological)

Each panel overlays three series so each cell is self-contained:
  1. Cell:     this (memory-backend, defense) — primary, solid color
  2. Baseline: (memory-backend, None defense) — gray fill + dashed outline
  3. Floor:    (No Mem., None defense)  — dotted outline only

Sized for full text-width inclusion in the paper (a fig* figure
roughly equivalent in footprint to Table 3).

Self-contained bundle layout:
  ./pyproject.toml                  uv project + pinned deps
  ./uv.lock                         locked versions
  ./data/utility_by_capability.csv  input data (extracted from Table 3 of the paper)
  ./radar_utility_grid.py           this script
  ./radar_grid_utility.{pdf,png}    rendered output (also produced by this script)

Run:
  uv sync
  uv run radar_utility_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "utility_by_capability.csv"
OUT_DIR = ROOT

CAPABILITIES = [
    "Asst. Resp.",
    "Long Mem.",
    "Mem. Only",
    "Mem. Tools",
    "Untr. Probe",
    "Untr. Send",
    "Dis. Send",
]
# Short axis labels — repeated 20× on the grid, so abbreviating saves a lot of room.
CAP_SHORT = {
    "Asst. Resp.": "AR",
    "Long Mem.":   "LM",
    "Mem. Only":   "MO",
    "Mem. Tools":  "MT",
    "Untr. Probe": "UP",
    "Untr. Send":  "US",
    "Dis. Send":   "DS",
}
# Long names for the legend key — taken from the paper's capability-class
# definitions (App. A: assistant_responses, long_memory, memory_only,
# memory_tools, untrusted_probe, untrusted_send, disable_send).
CAP_LONG = {
    "Asst. Resp.": "Assistant Responses",
    "Long Mem.":   "Long Memory",
    "Mem. Only":   "Memory Only",
    "Mem. Tools":  "Memory Tools",
    "Untr. Probe": "Untrusted Probe",
    "Untr. Send":  "Untrusted Send",
    "Dis. Send":   "Disable Send",
}
BACKENDS = ["Explicit", "Mem0", "RAG", "Context"]  # No Mem. dropped — it's the floor reference, already overlaid in every cell
DEFENSES = [
    # "None (Baseline)" omitted — its cell == baseline shadow, no information.
    "User Prompt Only",
    "No Untrusted Write",
    "Limit Memory Length",
    "Provable Policy",
]
# Two-line column titles so we can keep cells tight without label collisions.
DEFENSE_TITLE = {
    "None (Baseline)":     "None\n(Baseline)",
    "User Prompt Only":    "User Prompt\nOnly",
    "No Untrusted Write":  "No Untrusted\nWrite",
    "Limit Memory Length": "Limit Memory\nLength",
    "Provable Policy":     "Provable\nPolicy",
}
# Muted, paper-friendly palette (Tableau-derived, desaturated). Avoids the
# default matplotlib primary-color look while staying colorblind-distinguishable.
DEFENSE_COLORS = {
    "None (Baseline)":     "#3D3D3D",
    "User Prompt Only":    "#3D6E9C",  # steel blue
    "No Untrusted Write":  "#C77B33",  # warm ochre
    "Limit Memory Length": "#4F8A6F",  # sage green
    "Provable Policy":     "#9B3A3A",  # muted brick red
}
BASELINE_FILL = "#9C9C9C"
BASELINE_EDGE = "#6E6E6E"
FLOOR_EDGE    = "#B8B8B8"
GRID_COLOR    = "#D6D6D6"
SPINE_COLOR   = "#B0B0B0"

R_MIN = -10
R_MAX = 100


def radar_axes(ax, n: int):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([CAP_SHORT[c] for c in CAPABILITIES], fontsize=6.5)
    ax.tick_params(axis="x", pad=-2)
    ax.set_ylim(R_MIN, R_MAX)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["", "", ""])  # radial ticks visible but unlabeled in cells
    ax.set_rlabel_position(180 / n)
    ax.grid(color=GRID_COLOR, linewidth=0.35)
    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.spines["polar"].set_linewidth(0.4)
    return angles


def values_for(df_model: pd.DataFrame, backend: str, defense: str) -> list[float] | None:
    sub = df_model[(df_model["backend"] == backend) & (df_model["defense"] == defense)]
    if sub.empty or sub["value"].isna().all():
        return None
    vals: list[float] = []
    for cap in CAPABILITIES:
        row = sub[sub["capability"] == cap]
        if row.empty or pd.isna(row["value"].iloc[0]):
            return None
        vals.append(float(row["value"].iloc[0]))
    return vals


def plot_cell(ax, df_model: pd.DataFrame, backend: str, defense: str):
    angles = radar_axes(ax, len(CAPABILITIES))
    closed = np.concatenate([angles, angles[:1]])

    # Layer 1: memory-backend baseline (None defense, same memory-backend) — faint gray fill
    # so it reads as a "shadow" the colored cell sits on top of.
    baseline = values_for(df_model, backend, "None (Baseline)")
    if baseline is not None:
        v = np.array(baseline + baseline[:1], dtype=float)
        ax.fill(closed, v, color=BASELINE_FILL, alpha=0.22)
        ax.plot(closed, v, color=BASELINE_EDGE, linestyle="--", linewidth=0.7)

    # Layer 2: the cell itself
    cell = values_for(df_model, backend, defense)
    if cell is not None:
        v = np.array(cell + cell[:1], dtype=float)
        color = DEFENSE_COLORS[defense]
        ax.plot(closed, v, color=color, linestyle="-", linewidth=1.2)
        ax.fill(closed, v, color=color, alpha=0.22)
    else:
        ax.text(0, 0, "n/a", ha="center", va="center", fontsize=7, color="#999",
                style="italic")


MODELS = [
    ("Gemini 3.1 Pro Preview", "Gemini 3.1 Pro Preview"),
    ("GPT-5-mini",             "GPT-5-mini"),
]


def make_combined_figure(df: pd.DataFrame, out_stem: Path):
    """Single figure: two model blocks side-by-side.

    Layout: 4 rows (memory-backends) × 8 cols (per-model 4 defenses × 2 models),
    with a small horizontal gap between model blocks and per-model band headers.
    Sized for a wide \\textwidth figure on a paper page (≈ Table 3 footprint).
    """
    nrows = len(BACKENDS)                          # 4
    cols_per_model = len(DEFENSES)                 # 4

    # Wide-figure target: ~ paper text width or slightly wider (figure*)
    cell = 1.30
    fig_w = (cell * cols_per_model * len(MODELS) + 0.95) * 1.02   # row labels + inter-block gap
    fig_h = (cell * nrows + 2.00) * 1.02                          # headers + legend + row padding

    fig = plt.figure(figsize=(fig_w, fig_h))

    left, right = 0.06, 0.99
    top, bottom = 0.93, 0.10
    header_h = 0.06                                # model band height (in fig coords)
    coltitle_h = 0.07                              # space reserved for column titles
    inter_block_w = 0.025                          # horizontal gap between models
    avail_w = right - left - inter_block_w
    block_w = avail_w / 2
    cells_top = top - header_h - coltitle_h
    row_gap = 0.020                                # vertical padding between rows
    col_gap = 0.018                                # horizontal padding between cells within a block
    cell_h = (cells_top - bottom - row_gap * (nrows - 1)) / nrows
    cell_w = (block_w - col_gap * (cols_per_model - 1)) / cols_per_model
    # Shrink the polar circle within its cell by reserving inner padding;
    # implemented by inset positioning of each polar Axes.
    inner_pad_x = 0.08                             # fraction of cell width
    inner_pad_y = 0.08                             # fraction of cell height

    for m_idx, (model, _) in enumerate(MODELS):
        sub = df[df["model"] == model]
        block_left = left + m_idx * (block_w + inter_block_w)
        for i, backend in enumerate(BACKENDS):
            for j, defense in enumerate(DEFENSES):
                x0 = block_left + j * (cell_w + col_gap)
                y0 = cells_top - (i + 1) * cell_h - i * row_gap
                # Shrink polar circle inside the cell rect via inset margins.
                ax_w = cell_w * (1 - 2 * inner_pad_x)
                ax_h = cell_h * (1 - 2 * inner_pad_y)
                ax_x = x0 + cell_w * inner_pad_x
                ax_y = y0 + cell_h * inner_pad_y
                ax = fig.add_axes((ax_x, ax_y, ax_w, ax_h), projection="polar")
                plot_cell(ax, sub, backend, defense)
                if i == 0:
                    ax.set_title(DEFENSE_TITLE[defense], fontsize=8, pad=3)
                # Row labels: only on the leftmost cell of the leftmost block
                if j == 0 and m_idx == 0:
                    ax.text(
                        -0.28, 0.5, backend,
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=9, fontweight="bold",
                        rotation=90,
                    )

        # Model band header centered above this block
        band_bot = top - header_h
        fig.text(
            block_left + block_w / 2, band_bot + header_h * 0.35,
            model,
            ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color="#222",
        )
        # Underline spanning just this block
        fig.add_artist(plt.Line2D(
            [block_left, block_left + block_w], [band_bot, band_bot],
            transform=fig.transFigure,
            color="#888", linewidth=0.5,
        ))

    # Legend + axis-abbreviation key under the figure
    legend_handles = [
        Line2D([0], [0], color="#3D3D3D", lw=1.3,
               label="this memory-backend × defense"),
        Patch(facecolor=BASELINE_FILL, alpha=0.35, edgecolor=BASELINE_EDGE,
              linestyle="--", label="baseline (same memory-backend, no defense)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2, frameon=False, fontsize=11,
        bbox_to_anchor=(0.5, 0.035),
    )
    abbrev_text = "   ".join(f"{CAP_SHORT[c]} = {CAP_LONG[c]}" for c in CAPABILITIES)
    fig.text(0.5, 0.021, "Capability Classes",
             ha="center", va="bottom",
             fontsize=10, color="#555", style="italic")
    fig.text(0.5, 0.001, abbrev_text,
             ha="center", va="bottom",
             fontsize=10, color="#555")

    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    make_combined_figure(df, OUT_DIR / "radar_grid_utility")


if __name__ == "__main__":
    main()
