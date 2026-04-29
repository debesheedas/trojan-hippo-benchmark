#!/usr/bin/env python3
"""
Produce two security–utility tradeoff plots per model (test_100 ASR):

1. Pareto frontier: Security (1 - ASR) vs Weighted Utility; mark frontier and dominated.
2. Weight sensitivity: same as (1) for 2–3 deployment weight profiles.

Uses: consolidated_results/{model}/all_suites_combined.csv (utility by capability class),
      consolidated_attack_results/{model}/test_100/average_summary.csv (ASR).
Output: figures in CCS_thesis_manuscript/figures/ (or --output-dir).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CONSOLIDATED_RESULTS = BASE_DIR / "data" / "benchmark" / "consolidated_results"
CONSOLIDATED_ATTACK = BASE_DIR / "data" / "benchmark" / "consolidated_attack_results"
TEST_100 = "test_100"
OUTPUT_DIR = BASE_DIR / "CCS_thesis_manuscript" / "figures"

BACKENDS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
DEFENSES = [
    "None",
    "User Prompt Only",
    "No Untrusted Tools",
    "Limit Memory Length",
    "Provable Policy",
]

# Suite name in CSV -> key for weights
SUITE_TO_KEY = {
    "Assistant Responses": "assistant_responses",
    "Disable Send": "disable_send",
    "Long Memory": "long_memory",
    "Memory Only": "memory_only",
    "Memory Tools": "memory_tools",
    "Untrusted Probe": "untrusted_probe",
    "Untrusted Send": "untrusted_send",
}

# Profile 1: Recall-focused — emphasis on memory recall, assistant follow-up, long context; less on send/probe.
RECALL_FOCUSED_WEIGHTS = {
    "memory_only": 0.18,
    "assistant_responses": 0.18,
    "untrusted_probe": 0.04,
    "untrusted_send": 0.04,
    "disable_send": 0.05,
    "memory_tools": 0.06,
    "long_memory": 0.45,
}

# Profile 2: Balanced — typical personal assistant with email; moderate across flows.
BALANCED_WEIGHTS = {
    "memory_only": 0.12,
    "assistant_responses": 0.08,
    "untrusted_probe": 0.10,
    "untrusted_send": 0.08,
    "disable_send": 0.08,
    "memory_tools": 0.14,
    "long_memory": 0.40,
}

# Profile 3: Email-heavy — client uses all email features; long context and memory matter; memory_only negligible.
EMAIL_HEAVY_WEIGHTS = {
    "memory_only": 0.01,
    "assistant_responses": 0.14,
    "untrusted_probe": 0.15,
    "untrusted_send": 0.12,
    "disable_send": 0.15,
    "memory_tools": 0.13,
    "long_memory": 0.30,
}

# Deployment profiles: display name -> weight dict (spectrum: Recall-focused -> Balanced -> Email-heavy).
# Order determines row order in security-utility scatter plots.
PROFILES = {
    "Recall-focused": RECALL_FOCUSED_WEIGHTS,
    "Balanced Personal Assistant with Email": BALANCED_WEIGHTS,
    "Email-heavy": EMAIL_HEAVY_WEIGHTS,
}


def parse_pct(s: str) -> float | None:
    if s is None or not str(s).strip() or str(s).strip() in ("-", "ERR", "---"):
        return None
    m = re.match(r"^([\d.]+)\s*%?\s*$", str(s).strip())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def load_utility_by_suite(csv_path: Path) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Load all_suites_combined.csv. Returns (defense, backend_index) -> [(suite_key, rate), ...]."""
    # suite -> defense -> [rate for 5 backends]
    by_suite: dict[str, dict[str, list[float | None]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suite = (row.get("Test Suite") or "").strip()
            defense = (row.get("Defense Type") or "").strip()
            if not suite or not defense:
                continue
            if suite not in by_suite:
                by_suite[suite] = {}
            rates = []
            for b in BACKENDS:
                col = b + " (%)"
                val = parse_pct(row.get(col, ""))
                rates.append(val)
            by_suite[suite][defense] = rates

    # Build (defense, backend_ix) -> [(suite_key, rate), ...]
    result: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for suite, def_map in by_suite.items():
        key = SUITE_TO_KEY.get(suite)
        if not key:
            continue
        for defense, rates in def_map.items():
            for backend_ix, r in enumerate(rates):
                if r is None:
                    continue
                k = (defense, backend_ix)
                if k not in result:
                    result[k] = []
                result[k].append((key, r))
    return result  # type: ignore


def load_asr_test100(csv_path: Path) -> dict[tuple[str, int], float]:
    """Load average_summary.csv (Attack Total row). Returns (defense, backend_ix) -> ASR."""
    out: dict[tuple[str, int], float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            defense = (row.get("Defense Type") or "").strip()
            metric = (row.get("Metric") or "").strip()
            if metric != "Attack (Total)":
                continue
            for backend_ix, b in enumerate(BACKENDS):
                col = b + " (%)"
                val = parse_pct(row.get(col, ""))
                if val is not None:
                    out[(defense, backend_ix)] = val
    return out


def weighted_utility(
    suite_rates: list[tuple[str, float]],
    weights: dict[str, float],
) -> float:
    total = 0.0
    w_sum = 0.0
    for key, rate in suite_rates:
        w = weights.get(key, 0.0)
        total += w * rate
        w_sum += w
    if w_sum <= 0:
        return 0.0
    return total / w_sum


def get_configs(utility_data: dict, asr_data: dict) -> list[tuple[str, int]]:
    """List (defense, backend_ix) that have both utility and ASR."""
    u_keys = set(utility_data.keys())
    a_keys = set(asr_data.keys())
    common = u_keys & a_keys
    return sorted(common, key=lambda x: (DEFENSES.index(x[0]) if x[0] in DEFENSES else 99, x[1]))


def compute_pareto_frontier(
    points: list[tuple[float, float]],
) -> list[bool]:
    """For each point, True if on Pareto frontier (maximize both coords)."""
    n = len(points)
    frontier = [True] * n
    for i in range(n):
        u_i, s_i = points[i]
        for j in range(n):
            if i == j:
                continue
            u_j, s_j = points[j]
            if u_j >= u_i and s_j >= s_i and (u_j > u_i or s_j > s_i):
                frontier[i] = False
                break
    return frontier


# Defense colors (distinct, colorblind-friendly)
# Bright, high-contrast colors for defenses (blue, red, green, violet, orange)
DEFENSE_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f97316"]
# Backend markers: circle, square, triangle up, diamond, triangle down
BACKEND_MARKERS = ["o", "s", "^", "D", "v"]


DEFENSE_DISPLAY = {
    "None": "No Defense",
    "User Prompt Only": "User Prompt Only",
    "No Untrusted Tools": "No untrusted write",
    "Limit Memory Length": "Limit Memory Length",
    "Provable Policy": "Provable Policy",
}

def backend_label(backend_ix: int) -> str:
    """Label for plot: memory backend name only."""
    return BACKENDS[backend_ix]


def add_gradient_line(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color_start: str,
    color_end: str,
    n_steps: int = 32,
) -> None:
    """Draw a line segment with a smooth color gradient from color_start to color_end."""
    from matplotlib.collections import LineCollection
    from matplotlib import colors as mcolors

    if n_steps < 2:
        n_steps = 2
    xs = np.linspace(x0, x1, n_steps)
    ys = np.linspace(y0, y1, n_steps)
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    c0 = np.array(mcolors.to_rgba(color_start))
    c1 = np.array(mcolors.to_rgba(color_end))
    t = np.linspace(0.0, 1.0, n_steps - 1)[:, None]
    colors = c0 + (c1 - c0) * t

    lc = LineCollection(segments, colors=colors, linewidths=1.0, alpha=0.9, zorder=2)
    ax.add_collection(lc)


def add_backend_legend_next_to_defense(ax, defense_handles: list[Any], fontsize: int = 5) -> None:
    """Add backend-shape legend next to the existing defense-color legend."""
    from matplotlib.lines import Line2D

    # First legend (defenses) is expected to be created by the caller? We create both here
    # so we can control placement to avoid overlap.
    leg_def = ax.legend(
        handles=defense_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=fontsize,
        ncol=1,
        frameon=True,
        borderaxespad=0.0,
    )
    ax.add_artist(leg_def)

    backend_handles: list[Any] = []
    for backend_ix, backend in enumerate(BACKENDS):
        backend_handles.append(
            Line2D(
                [0],
                [0],
                marker=BACKEND_MARKERS[backend_ix],
                color="none",
                markerfacecolor="#111827",
                markeredgecolor="none",
                markersize=5,
                linestyle="None",
                label=backend,
            )
        )

    # Second legend (backends) positioned right next to defense legend.
    ax.legend(
        handles=backend_handles,
        loc="lower left",
        bbox_to_anchor=(0.25, 0.02),
        fontsize=fontsize,
        ncol=1,
        frameon=True,
        borderaxespad=0.0,
    )


def slugify(name: str) -> str:
    """Simple slug for filenames: lowercase, alnum + dashes."""
    import re as _re

    s = name.strip().lower()
    s = _re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def run(
    models: list[str],
    consolidated_results: Path,
    consolidated_attack: Path,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    for model in models:
        util_path = consolidated_results / model / "all_suites_combined.csv"
        asr_path = consolidated_attack / model / TEST_100 / "average_summary.csv"
        if not util_path.exists() or not asr_path.exists():
            print(f"Skip {model}: missing data")
            continue

        utility_data = load_utility_by_suite(util_path)
        asr_data = load_asr_test100(asr_path)
        configs = get_configs(utility_data, asr_data)
        if not configs:
            print(f"Skip {model}: no common configs")
            continue

        # Normalize profile weights to sum 1
        weights_by_profile = {}
        for name, wdict in PROFILES.items():
            s = sum(wdict.values())
            weights_by_profile[name] = {k: v / s for k, v in wdict.items()} if s else wdict

        # Weighted utility per config for each profile
        util_by_profile = {}
        for name, weights in weights_by_profile.items():
            util_by_profile[name] = {}
            for c in configs:
                rates = utility_data.get(c, [])
                util_by_profile[name][c] = weighted_utility(rates, weights)

        # Security = 100 - ASR (so 100 = best)
        security = {c: 100.0 - asr_data[c] for c in configs}

        model_safe = model.replace(".", "-").replace(" ", "-")
        font_size = 10
        title_font = 11

        # ----- Plot 1: Pareto frontier (Use-case Fig. 2/3 weights) -----
        first_profile_name = next(iter(weights_by_profile))
        weights_default = weights_by_profile[first_profile_name]
        util_default = {c: weighted_utility(utility_data.get(c, []), weights_default) for c in configs}
        points = [(util_default[c], security[c]) for c in configs]
        frontier = compute_pareto_frontier(points)
        frontier_pts = [(points[i], c) for i, c in enumerate(configs) if frontier[i]]
        frontier_pts.sort(key=lambda x: x[0][0])

        fig1, ax1 = plt.subplots(figsize=(7, 5.5))

        # Four-quadrant background: vertical at 80% utility, horizontal at 90% security
        # Greenish top, reddish bottom; top-left yellowish, bottom-right orangish
        ax1.fill_betweenx([90, 105], 80, 105, alpha=0.18, color="green", zorder=0)   # top-right
        ax1.fill_betweenx([90, 105], -5, 80, alpha=0.12, color="#fff9c4", zorder=0)  # top-left (light yellow)
        ax1.fill_betweenx([-5, 90], 80, 105, alpha=0.12, color="#ffab40", zorder=0)  # bottom-right (more orangish)
        ax1.fill_betweenx([-5, 90], -5, 80, alpha=0.18, color="red", zorder=0)      # bottom-left
        ax1.axvline(80, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        ax1.axhline(90, color="gray", linewidth=0.8, linestyle="--", zorder=1)

        # Quadrant labels: light grey, just outside plot (data coords)
        label_color = "#999999"
        label_fs = 7
        ax1.text(40, 107, "Low Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
        ax1.text(90, 107, "High Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
        ax1.text(107, 95, "High Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)
        ax1.text(107, 40, "Low Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)

        # Plot points: color = defense, shape = backend (small markers, no border)
        for i, c in enumerate(configs):
            u, s = points[i]
            backend_ix = c[1]
            defense_ix = DEFENSES.index(c[0]) if c[0] in DEFENSES else 0
            ax1.scatter(u, s, s=22, zorder=3, color=DEFENSE_COLORS[defense_ix], marker=BACKEND_MARKERS[backend_ix], edgecolors="none")

        # Labels: vertically below or above marker to avoid covering it; alternate when below would overlap
        # Sort by y descending (top first); assign below/above to minimize overlap
        idx_by_y = sorted(range(len(configs)), key=lambda i: -points[i][1])
        placement = {}  # idx -> "below" or "above"
        for idx in idx_by_y:
            u, s = points[idx]
            # Default: below for upper half (room below), above for lower half (room above)
            if s >= 55:
                preferred = "below"
            elif s <= 45:
                preferred = "above"
            else:
                preferred = "below"
            # Check if preferred would overlap with already-placed labels in same vertical band
            overlap_below = False
            for j, pl in placement.items():
                uj, sj = points[j]
                if abs(s - sj) < 18 and abs(u - uj) < 25:  # nearby in y and x
                    if pl == preferred:
                        overlap_below = True
                        break
            placement[idx] = "above" if (preferred == "below" and overlap_below) else preferred
        # Draw labels (backend name only)
        for i, c in enumerate(configs):
            u, s = points[i]
            lbl = backend_label(c[1])
            pl = placement.get(i, "below")
            dy = -3 if pl == "below" else 3  # below: -3pt, above: +3pt
            va = "top" if pl == "below" else "bottom"  # anchor label edge to point
            ax1.annotate(
                lbl,
                (u, s),
                fontsize=2.5,
                ha="center",
                va=va,
                xytext=(0, dy),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9),
            )

        # Legend: defenses only (colors); inside plot, bottom left
        from matplotlib.lines import Line2D
        defense_legend = [DEFENSE_DISPLAY.get(d, d) for d in DEFENSES]
        defense_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=DEFENSE_COLORS[i], markeredgecolor="none", markersize=5, label=defense_legend[i]) for i in range(5)]
        add_backend_legend_next_to_defense(ax1, defense_handles, fontsize=5)

        ax1.set_xlabel("Weighted Utility (%)", fontsize=7)
        ax1.set_ylabel("Security = 1 − ASR (%)", fontsize=7)
        ax1.tick_params(axis="both", labelsize=5)
        ax1.set_xticks(range(0, 101, 10))
        ax1.set_yticks(range(0, 101, 10))
        ax1.set_xlim(-5, 105)
        ax1.set_ylim(-5, 105)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", adjustable="box")
        fig1.tight_layout()
        out1 = output_dir / f"security_utility_pareto_{model_safe}.pdf"
        fig1.savefig(out1, bbox_inches="tight")
        out1_png = output_dir / f"security_utility_pareto_{model_safe}.png"
        fig1.savefig(out1_png, bbox_inches="tight", dpi=300)
        plt.close(fig1)
        print(f"  {out1}")
        print(f"  {out1_png}")

        # ----- Plot 2: 3 rows × 1 col — scatter plots only (no heatmaps) -----
        from matplotlib.lines import Line2D
        from matplotlib.gridspec import GridSpec
        defense_legend = [DEFENSE_DISPLAY.get(d, d) for d in DEFENSES]
        defense_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=DEFENSE_COLORS[i], markeredgecolor="none", markersize=5, label=defense_legend[i]) for i in range(5)]
        label_color = "#999999"
        label_fs = 7

        # 3 rows of scatter plots; square-ish panels
        fig_width = 6
        row_height_in = 5
        fig_height = 3 * row_height_in / 0.92 + 0.5
        fig3 = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(3, 1, figure=fig3, hspace=0.28)
        for idx, (profile_name, w) in enumerate(weights_by_profile.items()):
            util_p = util_by_profile[profile_name]
            points_p = [(util_p[c], security[c]) for c in configs]
            frontier_p = compute_pareto_frontier(points_p)

            # ----- Security–utility tradeoff scatter plot -----
            ax = fig3.add_subplot(gs[idx, 0])
            ax.fill_betweenx([90, 105], 80, 105, alpha=0.18, color="green", zorder=0)
            ax.fill_betweenx([90, 105], -5, 80, alpha=0.12, color="#fff9c4", zorder=0)
            ax.fill_betweenx([-5, 90], 80, 105, alpha=0.12, color="#ffab40", zorder=0)
            ax.fill_betweenx([-5, 90], -5, 80, alpha=0.18, color="red", zorder=0)
            ax.axvline(80, color="gray", linewidth=0.8, linestyle="--", zorder=1)
            ax.axhline(90, color="gray", linewidth=0.8, linestyle="--", zorder=1)
            ax.text(40, 107, "Low Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
            ax.text(90, 107, "High Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
            ax.text(107, 95, "High Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)
            ax.text(107, 40, "Low Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)
            for i, c in enumerate(configs):
                u, s = points_p[i]
                backend_ix = c[1]
                defense_ix = DEFENSES.index(c[0]) if c[0] in DEFENSES else 0
                ax.scatter(u, s, s=22, zorder=3, color=DEFENSE_COLORS[defense_ix], marker=BACKEND_MARKERS[backend_ix], edgecolors="none")
            idx_by_y = sorted(range(len(configs)), key=lambda i: -points_p[i][1])
            placement = {}
            for iidx in idx_by_y:
                u, s = points_p[iidx]
                preferred = "below" if s >= 55 else ("above" if s <= 45 else "below")
                overlap_below = False
                for j, pl in placement.items():
                    uj, sj = points_p[j]
                    if abs(s - sj) < 18 and abs(u - uj) < 25 and pl == preferred:
                        overlap_below = True
                        break
                placement[iidx] = "above" if (preferred == "below" and overlap_below) else preferred
            for i, c in enumerate(configs):
                u, s = points_p[i]
                pl = placement.get(i, "below")
                dy = -3 if pl == "below" else 3
                va = "top" if pl == "below" else "bottom"
                ax.annotate(backend_label(c[1]), (u, s), fontsize=2.5, ha="center", va=va, xytext=(0, dy), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9))
            frontier_pts_p = [(points_p[i], c) for i, c in enumerate(configs) if frontier_p[i]]
            frontier_pts_p.sort(key=lambda x: x[0][0])
            if len(frontier_pts_p) >= 2:
                Uf = [p[0][0] for p in frontier_pts_p]
                Sf = [p[0][1] for p in frontier_pts_p]
                ax.step(Uf, Sf, where="post", color="darkblue", linewidth=1.2, alpha=0.8, zorder=2)
            add_backend_legend_next_to_defense(ax, defense_handles, fontsize=5)
            ax.set_xlabel("Weighted Utility (%)", fontsize=7)
            ax.set_ylabel("Security = 1 − ASR (%)", fontsize=7)
            ax.tick_params(axis="both", labelsize=5)
            ax.set_xticks(range(0, 101, 10))
            ax.set_yticks(range(0, 101, 10))
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
        fig3.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03, hspace=0.22, wspace=0.16)
        out3 = output_dir / f"security_utility_profiles_{model_safe}.pdf"
        fig3.savefig(out3, bbox_inches="tight")
        out3_png = output_dir / f"security_utility_profiles_{model_safe}.png"
        fig3.savefig(out3_png, bbox_inches="tight", dpi=300)
        plt.close(fig3)
        print(f"  {out3}")
        print(f"  {out3_png}")

        # ----- Per-profile standalone security–utility plots (PNG + PDF) -----
        # These correspond to the three rows above, but are saved as separate files
        # for easier inclusion as individual figures in the thesis.
        for profile_name, w in weights_by_profile.items():
            util_p = util_by_profile[profile_name]
            points_p = [(util_p[c], security[c]) for c in configs]
            frontier_p = compute_pareto_frontier(points_p)

            fig_single, ax = plt.subplots(figsize=(6, 5.5))
            ax.fill_betweenx([90, 105], 80, 105, alpha=0.18, color="green", zorder=0)
            ax.fill_betweenx([90, 105], -5, 80, alpha=0.12, color="#fff9c4", zorder=0)
            ax.fill_betweenx([-5, 90], 80, 105, alpha=0.12, color="#ffab40", zorder=0)
            ax.fill_betweenx([-5, 90], -5, 80, alpha=0.18, color="red", zorder=0)
            ax.axvline(80, color="gray", linewidth=0.8, linestyle="--", zorder=1)
            ax.axhline(90, color="gray", linewidth=0.8, linestyle="--", zorder=1)
            ax.text(40, 107, "Low Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
            ax.text(90, 107, "High Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
            ax.text(107, 95, "High Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)
            ax.text(107, 40, "Low Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)

            for i, c in enumerate(configs):
                u, s = points_p[i]
                backend_ix = c[1]
                defense_ix = DEFENSES.index(c[0]) if c[0] in DEFENSES else 0
                ax.scatter(u, s, s=22, zorder=3, color=DEFENSE_COLORS[defense_ix], marker=BACKEND_MARKERS[backend_ix], edgecolors="none")

            idx_by_y = sorted(range(len(configs)), key=lambda i: -points_p[i][1])
            placement = {}
            for iidx in idx_by_y:
                u, s = points_p[iidx]
                preferred = "below" if s >= 55 else ("above" if s <= 45 else "below")
                overlap_below = False
                for j, pl in placement.items():
                    uj, sj = points_p[j]
                    if abs(s - sj) < 18 and abs(u - uj) < 25 and pl == preferred:
                        overlap_below = True
                        break
                placement[iidx] = "above" if (preferred == "below" and overlap_below) else preferred

            for i, c in enumerate(configs):
                u, s = points_p[i]
                pl = placement.get(i, "below")
                dy = -3 if pl == "below" else 3
                va = "top" if pl == "below" else "bottom"
                ax.annotate(
                    backend_label(c[1]),
                    (u, s),
                    fontsize=2.5,
                    ha="center",
                    va=va,
                    xytext=(0, dy),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9),
                )

            frontier_pts_p = [(points_p[i], c) for i, c in enumerate(configs) if frontier_p[i]]
            frontier_pts_p.sort(key=lambda x: x[0][0])
            if len(frontier_pts_p) >= 2:
                Uf = [p[0][0] for p in frontier_pts_p]
                Sf = [p[0][1] for p in frontier_pts_p]
                ax.step(Uf, Sf, where="post", color="darkblue", linewidth=1.2, alpha=0.8, zorder=2)

            add_backend_legend_next_to_defense(ax, defense_handles, fontsize=5)
            ax.set_xlabel("Weighted Utility (%)", fontsize=7)
            ax.set_ylabel("Security = 1 − ASR (%)", fontsize=7)
            ax.tick_params(axis="both", labelsize=5)
            ax.set_xticks(range(0, 101, 10))
            ax.set_yticks(range(0, 101, 10))
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
            fig_single.tight_layout()

            prof_slug = slugify(profile_name)
            base = f"security_utility_profile_{prof_slug}_{model_safe}"
            out_pdf = output_dir / f"{base}.pdf"
            out_png = output_dir / f"{base}.png"
            fig_single.savefig(out_pdf, bbox_inches="tight")
            fig_single.savefig(out_png, bbox_inches="tight", dpi=300)
            plt.close(fig_single)
            print(f"  {out_pdf}")
            print(f"  {out_png}")

        # ----- RAG-only plot with 3 regimes × 5 defenses -----
        # One figure per model, showing only the RAG backend, with 3 points per defense
        # (recall-focused, balanced, email-heavy) in different colors, and lines joining
        # the 3 regime points belonging to the same defense.
        try:
            rag_ix = BACKENDS.index("RAG")
        except ValueError:
            rag_ix = None
        if rag_ix is not None:
            from matplotlib.lines import Line2D

            # Colors by regime (profile)
            regime_colors = {
                "Recall-focused": "#dc2626",  # red
                "Balanced Personal Assistant with Email": "#f97316",  # orange
                "Email-heavy": "#facc15",  # yellow
            }
            # Marker shapes by defense
            defense_markers = {
                "None": "o",
                "User Prompt Only": "s",
                "No Untrusted Tools": "^",
                "Limit Memory Length": "D",
                "Provable Policy": "v",
            }

            fig_rag, ax_rag = plt.subplots(figsize=(7, 5.5))

            # Quadrant background and guide lines (same as other plots)
            ax_rag.fill_betweenx([90, 105], 80, 105, alpha=0.18, color="green", zorder=0)
            ax_rag.fill_betweenx([90, 105], -5, 80, alpha=0.12, color="#fff9c4", zorder=0)
            ax_rag.fill_betweenx([-5, 90], 80, 105, alpha=0.12, color="#ffab40", zorder=0)
            ax_rag.fill_betweenx([-5, 90], -5, 80, alpha=0.18, color="red", zorder=0)
            ax_rag.axvline(80, color="gray", linewidth=0.8, linestyle="--", zorder=1)
            ax_rag.axhline(90, color="gray", linewidth=0.8, linestyle="--", zorder=1)

            # Quadrant labels
            label_color = "#999999"
            label_fs = 7
            ax_rag.text(40, 107, "Low Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
            ax_rag.text(90, 107, "High Utility", ha="center", va="bottom", fontsize=label_fs, color=label_color, zorder=2, clip_on=False)
            ax_rag.text(107, 95, "High Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)
            ax_rag.text(107, 40, "Low Security", ha="left", va="center", fontsize=label_fs, color=label_color, rotation=-90, zorder=2, clip_on=False)

            # Plot points and regime-connecting lines for each defense that has RAG data
            for defense in DEFENSES:
                cfg = (defense, rag_ix)
                if cfg not in configs:
                    continue
                # Compute and plot the three regime points for this defense.
                points_by_profile: dict[str, tuple[float, float]] = {}
                for profile_name in PROFILES.keys():
                    u = util_by_profile[profile_name][cfg]
                    s = security[cfg]
                    points_by_profile[profile_name] = (u, s)
                    color = regime_colors.get(profile_name, "#000000")
                    marker = defense_markers.get(defense, "o")
                    ax_rag.scatter(
                        u,
                        s,
                        s=22,
                        zorder=3,
                        color=color,
                        marker=marker,
                        edgecolors="none",
                    )

                # Draw gradient lines between Email-heavy -> Balanced -> Recall-focused for this defense,
                # with a yellow→orange gradient then orange→red.
                try:
                    (xe, ye) = points_by_profile["Email-heavy"]
                    (xb, yb) = points_by_profile["Balanced Personal Assistant with Email"]
                    (xr, yr) = points_by_profile["Recall-focused"]

                    add_gradient_line(ax_rag, xe, ye, xb, yb, regime_colors["Email-heavy"], regime_colors["Balanced Personal Assistant with Email"])
                    add_gradient_line(ax_rag, xb, yb, xr, yr, regime_colors["Balanced Personal Assistant with Email"], regime_colors["Recall-focused"])
                except KeyError:
                    # If any profile is missing for this defense, skip gradient lines.
                    pass

                # Add a single text label per defense near the cluster of its three points.
                if points_by_profile:
                    xs_def = [p[0] for p in points_by_profile.values()]
                    ys_def = [p[1] for p in points_by_profile.values()]
                    cx = float(np.mean(xs_def))
                    cy = float(np.mean(ys_def))
                    # Offset label slightly so it does not cover markers.
                    if defense == "User Prompt Only":
                        # Place label slightly down and further to the right of the cluster
                        xytext = (8, -4)
                        va = "top"
                    else:
                        # Place label slightly up and to the left of the cluster
                        xytext = (-4, 4)
                        va = "bottom"
                    ax_rag.annotate(
                        DEFENSE_DISPLAY.get(defense, defense),
                        (cx, cy),
                        xytext=xytext,
                        textcoords="offset points",
                        fontsize=5,
                        ha="right",
                        va=va,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8),
                        zorder=4,
                    )

            # Legends: shapes for defenses, colors for regimes
            defense_handles = []
            for d in DEFENSES:
                m = defense_markers.get(d, "o")
                display_name = DEFENSE_DISPLAY.get(d, d)
                defense_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=m,
                        color="none",
                        markerfacecolor="#6b7280",
                        markeredgecolor="none",
                        markersize=5,
                        linestyle="None",
                        label=display_name,
                    )
                )
            regime_handles = []
            for pname, color in regime_colors.items():
                regime_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=color,
                        markeredgecolor="none",
                        markersize=5,
                        linestyle="None",
                        label=pname,
                    )
                )

            leg1 = ax_rag.legend(handles=defense_handles, title="Defenses (marker shape)", loc="lower left", fontsize=5, title_fontsize=6, frameon=True)
            ax_rag.add_artist(leg1)
            ax_rag.legend(handles=regime_handles, title="Profiles (color)", loc="lower right", fontsize=5, title_fontsize=6, frameon=True)

            ax_rag.set_xlabel("Weighted Utility (%)", fontsize=7)
            ax_rag.set_ylabel("Security = 1 − ASR (%)", fontsize=7)
            ax_rag.tick_params(axis="both", labelsize=5)
            ax_rag.set_xticks(range(0, 101, 10))
            ax_rag.set_yticks(range(0, 101, 10))
            ax_rag.set_xlim(-5, 105)
            ax_rag.set_ylim(-5, 105)
            ax_rag.grid(True, alpha=0.3)
            ax_rag.set_aspect("equal", adjustable="box")
            fig_rag.tight_layout()

            rag_base = f"security_utility_rag_profiles_{model_safe}"
            out_rag_pdf = output_dir / f"{rag_base}.pdf"
            out_rag = output_dir / f"{rag_base}.png"
            fig_rag.savefig(out_rag_pdf, bbox_inches="tight")
            fig_rag.savefig(out_rag, bbox_inches="tight", dpi=300)
            plt.close(fig_rag)
            print(f"  {out_rag_pdf}")
            print(f"  {out_rag}")

    print("Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot security–utility tradeoff (Pareto, constrained, profiles).")
    parser.add_argument("--consolidated-results", type=Path, default=CONSOLIDATED_RESULTS)
    parser.add_argument("--consolidated-attack", type=Path, default=CONSOLIDATED_ATTACK)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["gemini-3.1-pro-preview", "gpt-5-mini"])
    args = parser.parse_args()
    run(
        models=args.models,
        consolidated_results=args.consolidated_results,
        consolidated_attack=args.consolidated_attack,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
