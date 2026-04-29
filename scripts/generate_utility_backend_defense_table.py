#!/usr/bin/env python3
"""
Generate utility_by_backend_defense.tex: rows = (backend, defense), columns = 7 capability classes (Val, Δ), AM, HM.
Produces two stacked tables: Gemini 3.1 Pro Preview and GPT-5-mini.
"""

import csv
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONSOLIDATED_DIR = BASE_DIR / "data" / "benchmark" / "consolidated_results"
OUTPUT_PATH = BASE_DIR / "CCS_thesis_manuscript" / "tables" / "utility_by_backend_defense.tex"

BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
DEFENSE_ORDER = ["None", "User Prompt Only", "No Untrusted Tools", "Limit Memory Length", "Provable Policy"]
SUITE_ORDER = [
    "Assistant Responses", "Disable Send", "Long Memory", "Memory Only",
    "Memory Tools", "Untrusted Probe", "Untrusted Send",
]
SUITE_TO_SHORT = {
    "Assistant Responses": "Asst. Resp.",
    "Disable Send": "Dis. Send",
    "Long Memory": "Long Mem.",
    "Memory Only": "Mem. Only",
    "Memory Tools": "Mem. Tools",
    "Untrusted Probe": "Untr. Probe",
    "Untrusted Send": "Untr. Send",
}


def parse_pct(s: str):
    if s is None or not str(s).strip() or str(s).strip() in ("-", "ERR"):
        return None
    m = re.match(r"^([\d.]+)\s*%?\s*$", str(s).strip())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def load_csv(path: Path):
    """Returns: suite -> defense -> [rate for No Memory, Explicit, Mem0, RAG, Context]"""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        current_suite = None
        for row in reader:
            if len(row) < 2:
                continue
            suite = row[0].strip()
            defense = row[1].strip()
            if suite:
                current_suite = suite
            if not current_suite:
                continue
            if current_suite not in data:
                data[current_suite] = {}
            rates = [parse_pct(row[i + 2]) if i + 2 < len(row) else None for i in range(5)]
            data[current_suite][defense] = rates
    return data


def value_to_color(val):
    """Return LaTeX cellcolor for value (0-100)."""
    if val is None:
        return "gray!15"
    v = int(round(val))
    if v == 0:
        return "hmred"
    if v <= 5:
        return "hmred!80!hmorange"
    if v <= 10:
        return "hmred!60!hmorange"
    if v <= 15:
        return "hmred!40!hmorange"
    if v <= 25:
        return "hmorange" if v >= 20 else "hmred!20!hmorange"
    if v <= 35:
        return "hmorange!80!hmyellow"
    if v <= 50:
        return "hmyellow" if v >= 45 else "hmorange!40!hmyellow"
    if v <= 65:
        return "hmyellow!40!hmlightgreen" if v >= 60 else "hmyellow!80!hmlightgreen"
    if v <= 80:
        return "hmlightgreen!80!hmgreen"
    if v <= 90:
        return "hmlightgreen!60!hmgreen" if v >= 85 else "hmlightgreen!40!hmgreen"
    return "hmgreen"


def delta_to_color(delta):
    if delta is None or delta == 0:
        return "deltaneutral"
    return "hmgreen!30!white" if delta > 0 else "hmred!25!white"


def format_delta(val, baseline):
    if val is None or baseline is None:
        return ("", "deltaneutral")
    delta = val - baseline
    if delta == 0:
        return ("", "deltaneutral")
    if delta > 0:
        return (f"$\\uparrow${int(delta)}", delta_to_color(delta))
    return (f"$\\downarrow${int(-delta)}", delta_to_color(delta))


def compute_am(values):
    """Arithmetic mean of non-None values. Returns (val, color) or None for all-missing."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    am = sum(nums) / len(nums)
    return am


def compute_hm(values):
    """Harmonic mean. None if any value is 0 or all missing."""
    nums = [v for v in values if v is not None and v > 0]
    if not nums or len(nums) < 7:
        # If we have any 0, HM = 0. If all missing, HM = None
        if any(v is not None and v == 0 for v in values):
            return 0.0
        return None
    return 7 / sum(1 / v for v in nums)


def build_table_data(data):
    """Build list of rows: (backend, defense, [(val, delta_str, delta_color), ...], am, hm)."""
    rows = []
    for bi, backend in enumerate(BACKEND_LABELS):
        baseline = {}  # suite -> value for None
        for suite in SUITE_ORDER:
            if suite in data and "None" in data[suite]:
                baseline[suite] = data[suite]["None"][bi]
            else:
                baseline[suite] = None

        for defense in DEFENSE_ORDER:
            cells = []
            values = []
            for suite in SUITE_ORDER:
                if suite not in data or defense not in data[suite]:
                    val = None
                else:
                    val = data[suite][defense][bi]
                values.append(val)
                if val is None:
                    cells.append(("---", "gray!15", "", "deltaneutral"))
                else:
                    ival = int(round(val))
                    delta_str, delta_col = format_delta(val, baseline.get(suite))
                    cells.append((str(ival), value_to_color(val), delta_str, delta_col))

            am = compute_am(values)
            hm = compute_hm(values)

            am_str = "---" if am is None else str(int(round(am)))
            am_col = "gray!15" if am is None else value_to_color(am)

            hm_str = "---" if hm is None else str(int(round(hm)))
            hm_col = "gray!15" if hm is None else value_to_color(hm)

            defense_label = "None (Baseline)" if defense == "None" else defense
            rows.append((backend, defense_label, cells, am_str, am_col, hm_str, hm_col))
    return rows


def emit_table(rows, model_name):
    lines = []
    lines.append("\\noindent\\textbf{" + model_name + "}")
    lines.append("")
    lines.append("\\begin{tabular}{ll|cc|cc|cc|cc|cc|cc|cc|cc|c}")
    lines.append("\\hline")
    cap_headers = " & ".join(
        f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{SUITE_TO_SHORT[s]} (\\%)}}}}"
        for s in SUITE_ORDER
    )
    lines.append(f"\\textbf{{Backend}} & \\textbf{{Defense}} & {cap_headers} & \\textbf{{AM (\\%)}} & \\textbf{{HM (\\%)}} \\\\")
    lines.append("\\cline{3-18}")
    lines.append("& & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & & \\\\")
    lines.append("\\hline")

    for i, (backend, defense, cells, am_str, am_col, hm_str, hm_col) in enumerate(rows):
        prefix = "\\multirow{5}{*}{" + backend + "}" if i % 5 == 0 else ""
        cell_parts = []
        for val, vcol, dstr, dcol in cells:
            # Match the shared utility heatmap scheme used by AM/HM tables (6/7).
            # Missing values use the gray placeholder used elsewhere.
            if val == "---":
                cell_parts.append(r"\cellcolor{gray!15}---")
            else:
                cell_parts.append(f"\\heatcell{{{val}}}")
            cell_parts.append(f"\\cellcolor{{{dcol}}}{dstr}" if dstr else f"\\cellcolor{{{dcol}}}")
        if am_str == "---":
            cell_parts.append(r"\cellcolor{gray!15}---")
        else:
            cell_parts.append(f"\\heatcell{{{am_str}}}")
        if hm_str == "---":
            cell_parts.append(r"\cellcolor{gray!15}---")
        else:
            cell_parts.append(f"\\heatcell{{{hm_str}}}")
        line = f"{prefix} & {defense} & " + " & ".join(cell_parts) + " \\\\"
        lines.append(line)
        if (i + 1) % 5 == 0 and i + 1 < len(rows):
            lines.append("\\hline")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main():
    models = [
        ("gemini-3.1-pro-preview", "Gemini~3.1~Pro~Preview"),
        ("gpt-5-mini", "GPT-5-mini"),
    ]
    tables = []
    for model_dir, display_name in models:
        csv_path = CONSOLIDATED_DIR / model_dir / "all_suites_combined.csv"
        if not csv_path.exists():
            print(f"Skip {model_dir}: no CSV")
            continue
        data = load_csv(csv_path)
        rows = build_table_data(data)
        tables.append((display_name, rows))

    if not tables:
        print("No data found")
        return 1

    out_lines = []
    out_lines.append("% Utility by (memory backend, defense) with 7 capability classes.")
    out_lines.append("% Value cells: RdYlGn spectrum. Delta cells: red (decrease), pale yellow (no change), green (increase).")
    out_lines.append("% AM = arithmetic mean, HM = harmonic mean of the seven capability values.")
    out_lines.append("")
    # Must span both columns in the two-column thesis.
    out_lines.append("\\begin{table*}[t]")
    out_lines.append("\\centering")
    out_lines.append("\\scriptsize")
    out_lines.append("\\caption{Utility by capability class for each (memory backend, defense). \\textbf{Value cells (Val):} success rate (0--100\\%); color scale red $\\to$ orange $\\to$ yellow $\\to$ green (low to high). \\textbf{Delta ($\\Delta$) cells:} percentage-point change vs.\\ the None (baseline) defense for that backend; $\\uparrow N$ = increase, $\\downarrow N$ = decrease; delta colors: green = improvement, pale yellow = unchanged, red = degradation. \\textbf{AM/HM:} arithmetic and harmonic mean of the seven capability values. ``---'' = inapplicable.}")
    out_lines.append("\\label{tab:utility-by-backend-defense}")
    out_lines.append("\\resizebox{\\textwidth}{!}{%")
    out_lines.append("\\begin{tabular}{ll|cc|cc|cc|cc|cc|cc|cc|cc|c}")
    out_lines.append("\\hline")
    # Single header for both tables
    cap_headers = " & ".join(
        f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{SUITE_TO_SHORT[s]} (\\%)}}}}"
        for s in SUITE_ORDER
    )
    out_lines.append(f"\\textbf{{Backend}} & \\textbf{{Defense}} & {cap_headers} & \\textbf{{AM (\\%)}} & \\textbf{{HM (\\%)}} \\\\")
    out_lines.append("\\cline{3-18}")
    out_lines.append("& & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & Val & $\\Delta$ & & \\\\")
    out_lines.append("\\hline")

    for idx, (display_name, rows) in enumerate(tables):
        if idx > 0:
            out_lines.append("")
        out_lines.append("\\multicolumn{18}{l}{\\textbf{" + display_name + "}} \\\\")
        out_lines.append("\\hline")
        for i, (backend, defense, cells, am_str, am_col, hm_str, hm_col) in enumerate(rows):
            prefix = "\\multirow{5}{*}{" + backend + "}" if i % 5 == 0 else ""
            cell_parts = []
            for val, vcol, dstr, dcol in cells:
                if val == "---":
                    cell_parts.append(r"\cellcolor{gray!15}---")
                else:
                    cell_parts.append(f"\\heatcell{{{val}}}")
                cell_parts.append(f"\\cellcolor{{{dcol}}}{dstr}" if dstr else f"\\cellcolor{{{dcol}}}")
            if am_str == "---":
                cell_parts.append(r"\cellcolor{gray!15}---")
            else:
                cell_parts.append(f"\\heatcell{{{am_str}}}")
            if hm_str == "---":
                cell_parts.append(r"\cellcolor{gray!15}---")
            else:
                cell_parts.append(f"\\heatcell{{{hm_str}}}")
            line = f"{prefix} & {defense} & " + " & ".join(cell_parts) + " \\\\"
            out_lines.append(line)
            if (i + 1) % 5 == 0 and i + 1 < len(rows):
                out_lines.append("\\hline")
        out_lines.append("\\hline")

    out_lines.append("\\end{tabular}")
    out_lines.append("}% end resizebox")
    out_lines.append("\\end{table*}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
