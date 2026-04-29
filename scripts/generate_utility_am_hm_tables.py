#!/usr/bin/env python3
"""
Regenerate utility AM/HM summary tables in CCS_thesis_manuscript/tables/.

Derived from:
  - data/benchmark/consolidated_results/{model}/all_suites_combined.csv
    (capability success rates by capability class)
  - data/benchmark/consolidated_attack_results/{model}/test_100/average_summary.csv
    (ASR at trigger session N=100)

Outputs:
  - CCS_thesis_manuscript/tables/utility_am_hm_by_defense_backend.tex
  - CCS_thesis_manuscript/tables/utility_am_hm_asr_by_defense_backend.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


BACKEND_LABELS = ["No Memory", "Explicit", "Mem0", "RAG", "Context"]
DEFENSE_ORDER = ["None", "User Prompt Only", "No Untrusted Tools", "Limit Memory Length", "Provable Policy"]
SUITE_ORDER = [
    "Assistant Responses",
    "Disable Send",
    "Long Memory",
    "Memory Only",
    "Memory Tools",
    "Untrusted Probe",
    "Untrusted Send",
]

DEFENSE_TEX_LABEL = {
    "None": "None (Baseline)",
    "User Prompt Only": "User-prompt-only",
    "No Untrusted Tools": "No-untrusted-tools",
    "Limit Memory Length": "Limit-memory-length",
    "Provable Policy": "Provable policy",
}

MODEL_DISPLAY = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gpt-5-mini": "GPT-5-mini",
}


def parse_pct_cell(val: str) -> float | None:
    """Parse '90.0%' or '-' into float/None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("-", "ERR", "NA"):
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None


def compute_am(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_hm(values: list[float | None]) -> float | None:
    # Match generate_utility_backend_defense_table.py:
    # - HM = 0 if any value is explicitly 0
    # - HM = None if fewer than 7 positive, non-missing values (and no explicit 0)
    nums = [v for v in values if v is not None and v > 0]
    if not nums or len(nums) < 7:
        if any(v is not None and v == 0 for v in values):
            return 0.0
        return None
    return 7 / sum(1 / v for v in nums)


def to_int_or_none(x: float | None) -> int | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return int(round(x))


def tex_cell_heat(val: int) -> str:
    return f"\\heatcell{{{val}}}"


def tex_cell_missing() -> str:
    return r"\cellcolor{gray!15}---"


def tex_cell_asr(val: int) -> str:
    return f"\\asrcellinv{{{val}}}"


def load_utility_all_suites_combined(model_dir: Path) -> dict[str, dict[str, list[float | None]]]:
    """
    Returns: suite -> defense -> [rate for No Memory, Explicit, Mem0, RAG, Context]
    """
    csv_path = model_dir / "all_suites_combined.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    out: dict[str, dict[str, list[float | None]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        backend_cols = header[2:]  # e.g., "No Memory (%)"

        # Map indices into our canonical BACKEND_LABELS ordering.
        backend_map: list[int] = []
        for b in BACKEND_LABELS:
            wanted = f"{b} (%)"
            if wanted in backend_cols:
                backend_map.append(backend_cols.index(wanted))
            else:
                idx = None
                for i, col in enumerate(backend_cols):
                    if col.replace(" (%)", "").strip() == b:
                        idx = i
                        break
                if idx is None:
                    raise KeyError(f"Could not map backend column for {b} in {csv_path.name}")
                backend_map.append(idx)

        for row in reader:
            if len(row) < 2:
                continue
            suite = row[0].strip()
            defense = row[1].strip()
            if suite not in SUITE_ORDER or defense not in DEFENSE_ORDER:
                continue

            rates: list[float | None] = []
            for backend_col_idx in backend_map:
                cell = row[2 + backend_col_idx] if (2 + backend_col_idx) < len(row) else "-"
                rates.append(parse_pct_cell(cell))

            out.setdefault(suite, {})[defense] = rates

    return out


def load_asr_test100(attack_model_dir: Path) -> dict[str, list[float | None]]:
    """
    Returns: defense -> [ASR for No Memory, Explicit, Mem0, RAG, Context] (float % or None).
    """
    csv_path = attack_model_dir / "test_100" / "average_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    out: dict[str, list[float | None]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        backend_cols = header[2:]

        backend_map: list[int] = []
        for b in BACKEND_LABELS:
            wanted = f"{b} (%)"
            if wanted in backend_cols:
                backend_map.append(backend_cols.index(wanted))
            else:
                idx = None
                for i, col in enumerate(backend_cols):
                    if col.replace(" (%)", "").strip() == b:
                        idx = i
                        break
                if idx is None:
                    raise KeyError(f"Could not map backend column for {b} in {csv_path.name}")
                backend_map.append(idx)

        for row in reader:
            if len(row) < 2:
                continue
            defense = row[0].strip()
            metric = row[1].strip()
            if defense not in DEFENSE_ORDER:
                continue
            # Metric string differs slightly across exports:
            # - "Attack (Total)" or "Attack ( Total)"
            # So we normalize whitespace before comparing.
            metric_norm = re.sub(r"\s+", "", metric)
            if metric_norm != "Attack(Total)" and metric_norm != "Attack(Total)":
                continue

            rates: list[float | None] = []
            for backend_col_idx in backend_map:
                cell = row[2 + backend_col_idx] if (2 + backend_col_idx) < len(row) else "-"
                rates.append(parse_pct_cell(cell))
            out[defense] = rates

    return out


def generate_utility_am_hm_table(models: list[str], consolidated_dir: Path, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("% Utility (benign): arithmetic mean and harmonic mean by defense and memory backend.")
    lines.append("% In two-column mode use table* (full text width). Standalone: plain block.")
    lines.append(r"\providecommand{\ifstandalone}{\iffalse}")
    lines.append(r"\ifstandalone")
    lines.append(
        r"  \par\centering\textbf{Utility (benign): arithmetic mean (AM) and harmonic mean (HM) of capability success rates by defense and memory backend, for Gemini~3.1~Pro~Preview and GPT-5-mini. All values in \%. ``---'' denotes not applicable or missing.}\par\smallskip"
    )
    lines.append(r"  \label{tab:utility-am-hm-by-defense-backend}")
    lines.append(r"  \par\smallskip")
    lines.append(r"\else")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Utility (benign): arithmetic mean (AM) and harmonic mean (HM) of capability success rates by defense and memory backend, for Gemini~3.1~Pro~Preview and GPT-5-mini. All values in \%. ``---'' denotes not applicable or missing.}"
    )
    lines.append(r"\label{tab:utility-am-hm-by-defense-backend}")
    lines.append(r"\fi")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|cc|cc|cc|cc|cc}")
    lines.append(r"\hline")

    for midx, model in enumerate(models):
        suite_by_def = load_utility_all_suites_combined(consolidated_dir / model)

        lines.append(rf"\multicolumn{{11}}{{l}}{{\textbf{{{MODEL_DISPLAY[model]}}}}} \\")
        lines.append(r"\hline")
        lines.append(
            r"& \multicolumn{2}{c|}{\textbf{No Mem.}} & \multicolumn{2}{c|}{\textbf{Explicit}} & \multicolumn{2}{c|}{\textbf{Mem0}} & \multicolumn{2}{c|}{\textbf{RAG}} & \multicolumn{2}{c}{\textbf{Context}} \\"
        )
        lines.append(r"\textbf{Defense} & AM & HM & AM & HM & AM & HM & AM & HM & AM & HM \\")
        lines.append(r"\hline")

        for defense in DEFENSE_ORDER:
            row_cells: list[str] = [DEFENSE_TEX_LABEL[defense]]
            for backend_ix, _backend in enumerate(BACKEND_LABELS):
                suite_vals: list[float | None] = []
                for suite in SUITE_ORDER:
                    suite_vals.append(suite_by_def.get(suite, {}).get(defense, [None] * 5)[backend_ix])
                am = compute_am(suite_vals)
                hm = compute_hm(suite_vals)
                am_i = to_int_or_none(am)
                hm_i = to_int_or_none(hm)

                row_cells.append(tex_cell_heat(am_i) if am_i is not None else tex_cell_missing())
                row_cells.append(tex_cell_heat(hm_i) if hm_i is not None else tex_cell_missing())

            lines.append(" & ".join(row_cells) + r" \\")

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\ifstandalone")
    lines.append(r"\par\medskip")
    lines.append(r"\else")
    lines.append(r"\end{table*}")
    lines.append(r"\fi")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_utility_am_hm_asr_table(
    models: list[str],
    consolidated_dir: Path,
    attack_consolidated_dir: Path,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("% Utility (benign) AM, HM and ASR at test_100 by defense and memory backend.")
    lines.append("% Same orientation as utility_am_hm_by_defense_backend; third subcolumn = ASR.")
    lines.append(r"% ASR from tab:defense-effectiveness-test100. ASR cells lightly shaded (green / yellow / red).")
    lines.append(r"% AM/HM use \heatcell (higher=greener); ASR uses \asrcellinv (lower=greener).")
    lines.append(r"% Two-column body: table* so \resizebox{\textwidth}{!}{...} matches full page width.")
    lines.append(r"\providecommand{\ifstandalone}{\iffalse}")
    lines.append(r"\ifstandalone")
    lines.append(
        r"  \par\centering\textbf{Utility (benign): AM and HM of capability success rates (\%); ASR at trigger session $N=100$ (\%). By defense and memory backend for Gemini~3.1~Pro~Preview and GPT-5-mini. ``---'' denotes not applicable or missing. ASR cells lightly shaded (low to high).}\par\smallskip"
    )
    lines.append(r"  \label{tab:utility-am-hm-asr-by-defense-backend}")
    lines.append(r"  \par\smallskip")
    lines.append(r"\else")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Utility (benign): AM and HM of capability success rates (\%); ASR at trigger session $N=100$ (\%). By defense and memory backend for Gemini~3.1~Pro~Preview and GPT-5-mini. ``---'' denotes not applicable or missing. ASR cells lightly shaded (low to high).}"
    )
    lines.append(r"\label{tab:utility-am-hm-asr-by-defense-backend}")
    lines.append(r"\fi")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}")
    lines.append(r"\toprule")

    for midx, model in enumerate(models):
        suite_by_def = load_utility_all_suites_combined(consolidated_dir / model)
        asr_by_def = load_asr_test100(attack_consolidated_dir / model)

        if midx == 0:
            lines.append(rf"\multicolumn{{16}}{{l}}{{{MODEL_DISPLAY[model]}}} \\")
            lines.append(r"\midrule")
        else:
            lines.append(r"\midrule")
            lines.append(r"\addlinespace[0.6ex]")
            lines.append(rf"\multicolumn{{16}}{{l}}{{{MODEL_DISPLAY[model]}}} \\")
            lines.append(r"\midrule")

        lines.append(
            r"& \multicolumn{3}{c|}{\textbf{No Mem.}} & \multicolumn{3}{c|}{\textbf{Explicit}} & \multicolumn{3}{c|}{\textbf{Mem0}} & \multicolumn{3}{c|}{\textbf{RAG}} & \multicolumn{3}{c}{\textbf{Context}} \\"
        )
        lines.append(
            r"& \multicolumn{2}{c}{Utility} & ASR & \multicolumn{2}{c}{Utility} & ASR & \multicolumn{2}{c}{Utility} & ASR & \multicolumn{2}{c}{Utility} & ASR & \multicolumn{2}{c}{Utility} & ASR \\"
        )
        lines.append(r"\textbf{Defense} & AM & HM & & AM & HM & & AM & HM & & AM & HM & & AM & HM & \\")
        lines.append(r"\midrule")

        for defense in DEFENSE_ORDER:
            row_cells: list[str] = [DEFENSE_TEX_LABEL[defense]]
            for backend_ix, _backend in enumerate(BACKEND_LABELS):
                suite_vals: list[float | None] = []
                for suite in SUITE_ORDER:
                    suite_vals.append(suite_by_def.get(suite, {}).get(defense, [None] * 5)[backend_ix])
                am = compute_am(suite_vals)
                hm = compute_hm(suite_vals)
                am_i = to_int_or_none(am)
                hm_i = to_int_or_none(hm)

                asr_val = asr_by_def.get(defense, [None] * 5)[backend_ix]
                asr_i = to_int_or_none(asr_val)

                row_cells.append(tex_cell_heat(am_i) if am_i is not None else tex_cell_missing())
                row_cells.append(tex_cell_heat(hm_i) if hm_i is not None else tex_cell_missing())
                row_cells.append(tex_cell_asr(asr_i) if asr_i is not None else tex_cell_missing())

            lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}%")
    lines.append(r"\ifstandalone")
    lines.append(r"\par\medskip")
    lines.append(r"\else")
    lines.append(r"\end{table*}")
    lines.append(r"\fi")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate utility AM/HM summary tables.")
    parser.add_argument(
        "--consolidated-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "benchmark" / "consolidated_results",
    )
    parser.add_argument(
        "--attack-consolidated-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "benchmark" / "consolidated_attack_results",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "CCS_thesis_manuscript" / "tables",
    )
    args = parser.parse_args()

    models_present = sorted([p.name for p in args.consolidated_dir.iterdir() if p.is_dir()])
    preferred = ["gemini-3.1-pro-preview", "gpt-5-mini"]
    models = [m for m in preferred if m in models_present] + [m for m in models_present if m not in preferred]
    if not models:
        print("No models found in consolidated_results", file=sys.stderr)
        return 1

    out_am_hm = args.tables_dir / "utility_am_hm_by_defense_backend.tex"
    generate_utility_am_hm_table(models, args.consolidated_dir, out_am_hm)

    out_asr = args.tables_dir / "utility_am_hm_asr_by_defense_backend.tex"
    generate_utility_am_hm_asr_table(models, args.consolidated_dir, args.attack_consolidated_dir, out_asr)

    print(f"Wrote: {out_am_hm}")
    print(f"Wrote: {out_asr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

