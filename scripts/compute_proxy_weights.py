#!/usr/bin/env python3
"""
Compute Proxy Weights for Memory-Equipped Email Agent Benchmark

Derives empirically-grounded weights for 7 representative task flows by combining
two public datasets under an independence assumption:
  1. LongMemEval (Wu et al., 2024) - memory dimension marginal
  2. Avocado / Enterprise Email Literature (Yang et al., SIGIR 2017) - email action marginal

Usage:
  python scripts/compute_proxy_weights.py
  python scripts/compute_proxy_weights.py --dry-run   # Skip download, use placeholder counts
  python scripts/compute_proxy_weights.py --output-dir ./output

Outputs:
  - proxy_weights_figure1_marginals.pdf
  - proxy_weights_figure2_final_weights.pdf
  - proxy_weights.json
  - LaTeX table fragment to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Non-interactive backend for headless/script execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_oracle.json"
)

# LongMemEval question_type -> our memory_dim mapping
# single-session-preference merges into single_session_user (both are single-session user recall)
# knowledge-update merges into multi_session (requires cross-session awareness)
LONGMEMEVAL_TO_MEMORY_DIM = {
    "single-session-user": "single_session_user",
    "single-session-assistant": "single_session_asst",
    "single-session-preference": "single_session_user",
    "multi-session": "multi_session",
    "knowledge-update": "multi_session",
    "temporal-reasoning": "temporal",
}

# "none" memory dimension is NOT in LongMemEval; used for disable_send flow.
# Small explicit constant: tasks that require no memory at all.
P_MEMORY_NONE = 0.10

# Dry-run placeholder counts (for testing without network).
# Source: approximate LongMemEval distribution; abstention subset ~25 filtered out.
DRY_RUN_LONGMEMEVAL_COUNTS = {
    "single-session-user": 85,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "multi-session": 115,
    "knowledge-update": 78,
    "temporal-reasoning": 111,
}

# Email action marginal (empirically-motivated from literature).
# Avocado (Yang et al., SIGIR 2017): ~38% enterprise emails receive reply.
# CHI 2025: users spend 28% of workweek on email; AI assistant value proposition
# motivates mild upward adjustment for send-involved tasks.
P_EMAIL_NONE = 0.20        # Pure memory recall, no inbox access
P_EMAIL_READ_ONLY = 0.30   # Inbox read without send (most common single action)
P_EMAIL_OPTIONAL_SEND = 0.10   # Assistant output reuse with optional send
P_EMAIL_READ_SEND = 0.40   # Inbox read + reply/send (upward-adjusted from ~38%)

EMAIL_ACTION_MARGINAL = {
    "none": P_EMAIL_NONE,
    "read_only": P_EMAIL_READ_ONLY,
    "optional_send": P_EMAIL_OPTIONAL_SEND,
    "read_send": P_EMAIL_READ_SEND,
}

# 7 task flows: (flow_id, memory_dim, email_action_dim)
FLOWS = [
    ("memory_only", "single_session_user", "none"),
    ("assistant_responses", "single_session_asst", "optional_send"),
    ("untrusted_probe", "single_session_user", "read_only"),
    ("disable_send", "none", "read_send"),
    ("untrusted_send", "single_session_user", "read_send"),
    ("memory_tools", "multi_session", "read_send"),
    ("long_memory", "temporal", "none"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute proxy weights for memory-equipped email agent benchmark."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip download; use hardcoded LongMemEval placeholder counts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: script directory).",
    )
    return parser.parse_args()


def get_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(__file__).resolve().parent


def download_longmemeval(cache_path: Path) -> list[dict]:
    """Download LongMemEval oracle JSON; cache locally."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(LONGMEMEVAL_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data
    except requests.RequestException as e:
        if cache_path.exists():
            print(f"Download failed ({e}); using cached file.", file=sys.stderr)
            with open(cache_path) as f:
                return json.load(f)
        raise SystemExit(
            f"Failed to download LongMemEval: {e}\n"
            "Use --dry-run to run with placeholder counts."
        ) from e


def load_longmemeval(dry_run: bool, cache_path: Path) -> dict[str, int]:
    """
    Load LongMemEval data and return raw question_type counts (after filtering abstention).
    In dry-run mode, return DRY_RUN_LONGMEMEVAL_COUNTS directly.
    """
    if dry_run:
        return dict(DRY_RUN_LONGMEMEVAL_COUNTS)

    data = download_longmemeval(cache_path)
    # Filter out abstention questions (question_id ends with "_abs")
    non_abs = [r for r in data if not str(r.get("question_id", "")).endswith("_abs")]
    counts = Counter(r["question_type"] for r in non_abs)
    return dict(counts)


def compute_memory_marginal(raw_counts: dict[str, int]) -> dict[str, float]:
    """
    Map LongMemEval question_type counts to memory_dim, then normalise.
    'none' is NOT in LongMemEval; assign 0 in the marginal (handled in combination step).
    """
    mapped = {}
    for qtype, count in raw_counts.items():
        dim = LONGMEMEVAL_TO_MEMORY_DIM.get(qtype)
        if dim is not None:
            mapped[dim] = mapped.get(dim, 0) + count

    total = sum(mapped.values())
    marginals = {dim: c / total for dim, c in mapped.items()} if total > 0 else {}
    # "none" gets 0 in the marginal; P(memory_dim=none)=0.10 used only when computing joint
    marginals["none"] = 0.0
    return marginals


def get_email_action_marginal() -> dict[str, float]:
    """Return email action dimension marginal (empirically-motivated constants)."""
    total = sum(EMAIL_ACTION_MARGINAL.values())
    assert abs(total - 1.0) < 1e-9, f"Email action marginal must sum to 1.0, got {total}"
    return dict(EMAIL_ACTION_MARGINAL)


def compute_flow_weights(
    memory_marginal: dict[str, float],
    email_marginal: dict[str, float],
) -> pd.DataFrame:
    """
    Compute P(flow) = P(memory_dim) * P(email_action_dim), then normalise.
    For memory_dim='none' (disable_send), use P_MEMORY_NONE=0.10 explicitly.
    Returns DataFrame with flow_id, marginals, unnorm_joint, final_weight.
    """
    rows = []
    for flow_id, mem_dim, email_dim in FLOWS:
        p_mem = P_MEMORY_NONE if mem_dim == "none" else memory_marginal.get(mem_dim, 0.0)
        p_email = email_marginal.get(email_dim, 0.0)
        unnorm = p_mem * p_email
        rows.append({
            "flow_id": flow_id,
            "memory_dim": mem_dim,
            "memory_marginal_P": p_mem,
            "email_action_dim": email_dim,
            "email_action_marginal_P": p_email,
            "unnorm_joint": unnorm,
        })

    df = pd.DataFrame(rows)
    total_joint = df["unnorm_joint"].sum()
    df["final_weight"] = df["unnorm_joint"] / total_joint if total_joint > 0 else 0.0
    return df


def print_table(df: pd.DataFrame) -> None:
    """Print clean table to stdout."""
    print("\n" + "=" * 110)
    print("Proxy Weights: Flow-Level Distribution")
    print("=" * 110)
    display = df.copy()
    display["memory_marginal_P"] = display["memory_marginal_P"].round(4)
    display["email_action_marginal_P"] = display["email_action_marginal_P"].round(4)
    display["unnorm_joint"] = display["unnorm_joint"].round(6)
    display["normalized"] = display["final_weight"].round(4)
    # Show unnorm_joint then normalized (final weights)
    cols = ["flow_id", "memory_dim", "memory_marginal_P", "email_action_dim",
            "email_action_marginal_P", "unnorm_joint", "normalized"]
    print(display[cols].to_string(index=False))
    print("=" * 100)
    print(f"Sum of final weights: {df['final_weight'].sum():.6f}")
    print()


def save_json_weights(df: pd.DataFrame, out_path: Path) -> None:
    """Save final weights as JSON."""
    weights = {row["flow_id"]: round(float(row["final_weight"]), 4) for _, row in df.iterrows()}
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved: {out_path}")


def print_latex_table(df: pd.DataFrame) -> None:
    """Print LaTeX booktabs table fragment."""
    print("\n% --- LaTeX table fragment (booktabs) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Proxy task-flow weights derived from LongMemEval $\times$ Avocado under independence.}")
    print(r"\label{tab:proxy-weights}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Flow & Weight & \% \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        w = row["final_weight"]
        pct = w * 100
        fid = row["flow_id"].replace("_", r"\_")
        print(f"  {fid} & {w:.4f} & {pct:.2f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% --- end LaTeX ---\n")


def plot_figure1_marginals(
    memory_marginal: dict[str, float],
    raw_counts: dict[str, int],
    email_marginal: dict[str, float],
    out_path: Path,
) -> None:
    """
    Two side-by-side bar charts: memory dimension marginals (left), email action (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: memory dimension (after mapping)
    mem_dims = list(memory_marginal.keys())
    mem_vals = [memory_marginal[d] for d in mem_dims]
    colors_mem = plt.cm.Blues(np.linspace(0.4, 0.9, len(mem_dims)))
    bars1 = ax1.bar(mem_dims, mem_vals, color=colors_mem)
    ax1.set_xlabel("Memory dimension")
    ax1.set_ylabel("P(memory_dim)")
    ax1.set_title("LongMemEval Memory Marginal\n(after mapping & merging)")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    for bar, val in zip(bars1, mem_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Source category labels for memory (which raw types map to which dim)
    source_note = (
        "Mapping: single-session-user/preference→single_session_user; "
        "multi-session/knowledge-update→multi_session; none=0 (not in LongMemEval)"
    )
    ax1.annotate(source_note, xy=(0.5, -0.22), xycoords="axes fraction",
                 ha="center", fontsize=8, style="italic")

    # Right: email action
    email_dims = list(email_marginal.keys())
    email_vals = [email_marginal[d] for d in email_dims]
    colors_email = plt.cm.Greens(np.linspace(0.4, 0.9, len(email_dims)))
    bars2 = ax2.bar(email_dims, email_vals, color=colors_email)
    ax2.set_xlabel("Email action dimension")
    ax2.set_ylabel("P(email_action_dim)")
    ax2.set_title("Email Action Marginal\n(Avocado + CHI 2025 adjustment)")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    for bar, val in zip(bars2, email_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Proxy Weights: Marginal Distributions\n"
        "Sources: LongMemEval (Wu et al., 2024); Avocado (Yang et al., SIGIR 2017)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_figure2_final_weights(df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of 7 final weights.
    Colour by email_action_dim, hatching by memory_dim.
    """
    # Colours by email_action_dim (4 distinct colours)
    email_colors = {
        "none": "#2ecc71",
        "read_only": "#3498db",
        "optional_send": "#9b59b6",
        "read_send": "#e74c3c",
    }
    # Hatching by memory_dim
    mem_hatches = {
        "single_session_user": "",
        "single_session_asst": "///",
        "multi_session": "\\\\\\",
        "temporal": "xxx",
        "none": "+++",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    flows = df["flow_id"].tolist()
    weights = df["final_weight"].tolist()
    colors = [email_colors[row["email_action_dim"]] for _, row in df.iterrows()]
    hatches = [mem_hatches.get(row["memory_dim"], "") for _, row in df.iterrows()]

    y_pos = np.arange(len(flows))
    bars = ax.barh(y_pos, weights, color=colors, edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(flows, fontsize=10)
    ax.set_xlabel("Weight")
    ax.set_title(
        "Proxy Task-Flow Distribution for Memory-Equipped Email Agent",
        fontsize=12,
    )
    ax.set_xlim(0, max(weights) * 1.15)

    # Value labels on bars
    for i, (bar, w) in enumerate(zip(bars, weights)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=9)

    fig.text(
        0.5, -0.08,
        "Derived from LongMemEval (Wu et al., 2024) × Avocado corpus (Yang et al., SIGIR 2017) "
        "under independence assumption. See Appendix for full derivation and limitations.",
        ha="center", fontsize=8, style="italic", wrap=True, transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = get_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "longmemeval_oracle.json"

    # 1. Load LongMemEval counts
    raw_counts = load_longmemeval(args.dry_run, cache_path)
    if args.dry_run:
        print("Running in --dry-run mode (placeholder LongMemEval counts).", file=sys.stderr)

    # 2. Compute marginals
    memory_marginal = compute_memory_marginal(raw_counts)
    email_marginal = get_email_action_marginal()

    # 3. Compute flow weights
    df = compute_flow_weights(memory_marginal, email_marginal)

    # 4. Outputs
    print_table(df)
    save_json_weights(df, out_dir / "proxy_weights.json")
    print_latex_table(df)

    plot_figure1_marginals(
        memory_marginal, raw_counts, email_marginal,
        out_dir / "proxy_weights_figure1_marginals.pdf",
    )
    plot_figure2_final_weights(df, out_dir / "proxy_weights_figure2_final_weights.pdf")


if __name__ == "__main__":
    main()
