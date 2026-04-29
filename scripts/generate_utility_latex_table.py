#!/usr/bin/env python3
"""
Generate LaTeX table of utility by capability class (transposed for readability).

Table layout (transposed):
  - Rows: capability classes (7) — Assistant Responses, Disable Send, Long Memory, Memory Only, Memory Tools, Untrusted Probe, Untrusted Send
  - Columns: 5 main column groups (defenses), each with 5 sub-columns (No Memory, Explicit, Mem0, RAG, Context)
  - Cell values: integer success rate (no decimal, no %); caption states all numbers are in %
  - "---" for invalid/missing

Output: one .tex file per model in CCS_thesis_manuscript/tables/, to be \\input in appendix.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONSOLIDATED_DIR = BASE_DIR / "data" / "benchmark" / "consolidated_results"
# Where the thesis actually keeps the tables (CCS_thesis_manuscript/).
OUTPUT_DIR = BASE_DIR / "CCS_thesis_manuscript"

BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
DEFENSE_DISPLAY_ORDER = [
    "None", "User Prompt Only", "No Untrusted Tools",
    "Limit Memory Length", "Provable Policy",
]
# Suite order in all_suites_combined.csv (sorted alphabetically by key)
SUITE_LABELS_ORDER = [
    "Assistant Responses", "Disable Send", "Long Memory", "Memory Only",
    "Memory Tools", "Untrusted Probe", "Untrusted Send",
]


def parse_pct(s: str):
    """Return numeric value or None for missing/invalid."""
    if s is None or not str(s).strip() or str(s).strip() in ("-", "ERR"):
        return None
    m = re.match(r"^([\d.]+)\s*%?\s*$", str(s).strip())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def load_all_suites_combined(path: Path):
    """
    Load all_suites_combined.csv.
    Returns: list of (suite_label, defense, list of 5 rates in backend order)
    """
    if not path.exists():
        return {}
    # suite -> defense -> [rate for No Memory, Explicit, Mem0, RAG, Context]
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        backend_cols = [h.replace(" (%)", "").strip() for h in header[2:]]
        current_suite = None
        for row in reader:
            if len(row) < 2:
                continue
            suite_label = row[0].strip()
            defense = row[1].strip()
            if suite_label:
                current_suite = suite_label
            if not current_suite:
                continue
            if current_suite not in data:
                data[current_suite] = {}
            rates = []
            for i in range(min(5, len(row) - 2)):
                val = parse_pct(row[i + 2])
                rates.append(val)
            while len(rates) < 5:
                rates.append(None)
            data[current_suite][defense] = rates
    return data


def format_cell(val):
    """
    Integer only, no % (stated in caption).

    Uses the same heatmap color macro (`\\heatcell{}`) as the other utility tables,
    so this big "Table 3" matches tables 6/7 in color scheme.
    """
    if val is None:
        return r"\cellcolor{gray!15}---"
    ival = int(round(val))
    return rf"\heatcell{{{ival}}}"


def generate_latex_table(model_name: str, data: dict, out_path: Path):
    """Write one LaTeX table: rows = capability classes, columns = defenses × backends."""
    lines = []
    lines.append("% Utility by capability class (rows) and defense × backend (columns)")
    lines.append(f"% Model: {model_name}")
    # Must span both columns in the two-column appendix.
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{" + model_name + ": utility by capability class and (defense, memory backend). "
                 "Rows: capability class; column groups: defense; sub-columns: No Mem., Explicit, Mem0, RAG, Context. "
                 "All values in \\%. ``---'' denotes that the defense is not defined for that memory backend, or data is missing.}")
    safe = model_name.replace(".", "-").replace(" ", "-")
    lines.append("\\label{tab:utility-by-suite-" + safe + "}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    # Table spec: l (capability class) then 5 defense groups × 5 backends
    col_spec = "l|" + "|".join(["ccccc"] * 5)
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\hline")
    # Header row 1: 5 multicolumns (defenses)
    header1 = "\\textbf{Capability class} & "
    header1 += " & ".join(
        "\\multicolumn{5}{c|}{\\textbf{" + d.replace(" ", "~") + "}}"
        for d in DEFENSE_DISPLAY_ORDER
    )
    header1 += " \\\\"
    lines.append(header1)
    lines.append("\\hline")
    # Header row 2: backend names repeated 5 times (short labels to save space)
    short_backends = ["NoM", "Expl", "Mem0", "RAG", "Ctx"]
    header2 = " & " + " & ".join(
        " & ".join(short_backends) for _ in DEFENSE_DISPLAY_ORDER
    )
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\hline")

    for suite in SUITE_LABELS_ORDER:
        row_cells = ["\\textbf{" + suite.replace(" ", "~") + "}"]
        for defense in DEFENSE_DISPLAY_ORDER:
            if suite not in data or defense not in data[suite]:
                row_cells.extend([r"\cellcolor{gray!15}---"] * 5)
            else:
                rates = data[suite][defense]
                for i in range(5):
                    row_cells.append(format_cell(rates[i]) if i < len(rates) else r"\cellcolor{gray!15}---")
        lines.append(" & ".join(row_cells) + " \\\\[0.4ex]")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}% end resizebox")
    lines.append("\\end{table*}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX utility tables per model.")
    parser.add_argument("--consolidated-dir", type=Path, default=CONSOLIDATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Only generate these models (e.g., --models gemini-3.1-pro-preview). Defaults to all models.",
    )
    args = parser.parse_args()
    consolidated_dir = args.consolidated_dir
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    if not consolidated_dir.exists():
        print(f"Consolidated dir not found: {consolidated_dir}")
        return 1

    model_filter = set(args.models) if args.models else None

    for model_dir in sorted(consolidated_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if model_filter is not None and model_name not in model_filter:
            continue
        csv_path = model_dir / "all_suites_combined.csv"
        if not csv_path.exists():
            print(f"Skip {model_name}: no all_suites_combined.csv")
            continue
        data = load_all_suites_combined(csv_path)
        if not data:
            continue
        label_safe = model_name.replace(" ", "-").replace(".", "-")
        generate_latex_table(model_name, data, tables_dir / f"utility_by_suite_{label_safe}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
