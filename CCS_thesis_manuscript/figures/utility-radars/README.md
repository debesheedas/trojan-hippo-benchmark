# Utility Radars

Radar-grid figure visualising per-capability utility for each
(memory-backend, defense) cell, for both models studied in the paper.
This figure is the radar-chart counterpart of Table 3 in the paper.

## Contents

| File | Purpose |
|---|---|
| `radar_utility_grid.py` | Renders the figure from the CSV. |
| `data/utility_by_capability.csv` | Input data, extracted from Table 3 (350 rows: model × memory-backend × defense × capability). |
| `pyproject.toml` / `uv.lock` | Pinned Python deps (matplotlib, pandas, numpy). |
| `radar_grid_utility.pdf` / `.png` | Rendered output. |

## Run

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13+).

```sh
uv sync                       # one-time: install deps from uv.lock
uv run radar_utility_grid.py  # writes radar_grid_utility.{pdf,png}
```

## What the figure shows

For each model (Gemini 3.1 Pro Preview, GPT-5-mini), a 4×4 grid of small
radars: rows are the four persistent memory-backends (Explicit, Mem0,
RAG, Context); columns are the four defenses
(`User Prompt Only`, `No Untrusted Write`, `Limit Memory Length`,
`Provable Policy`). The "None (Baseline)" defense is omitted because in
that column the cell would be identical to the baseline shadow.

Each cell overlays two shapes:

- **Solid colored polygon** — utility for this (memory-backend × defense) across
  the 7 capability classes.
- **Gray dashed shadow** — same memory-backend with no defense (the baseline).
  Visible gray sticking out from under the colored fill = utility lost to
  the defense.
Axes (clockwise from top): `AR` Assistant Responses, `LM` Long Memory,
`MO` Memory Only, `MT` Memory Tools, `UP` Untrusted Probe,
`US` Untrusted Send, `DS` Disable Send. Radial scale is utility
(0–100%); the inner ring is 0 (the center is at −10 so 0-valued
capabilities are visible as a small ring rather than collapsing to a
point).

`n/a` cells (e.g. User-Prompt-Only × Explicit) mark inapplicable
defense/memory-backend combinations.

## Tweaking

Layout knobs near the top of `radar_utility_grid.py`:

- `cell` — base size of each radar cell (inches).
- `row_gap`, `col_gap` — padding between cells (figure-fraction units).
- `inner_pad_x`, `inner_pad_y` — shrink the polar circle within its cell
  (helps if axis labels collide with neighbours).
- `DEFENSE_COLORS` — palette per defense.

To regenerate the input CSV from the paper's LaTeX tables, see the
`scripts/extract_data.py` script in the parent paper repository.
