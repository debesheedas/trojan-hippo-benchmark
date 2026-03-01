#!/usr/bin/env python3
"""
Process train adaptive logs from attack_logs and attack_logs_stealth to build
frequency tables of OpenEvolve iterations required to find a successful attack.

Only counts runs where a successful attack was achieved in the end.
Outputs two CSVs: one for attack_logs, one for attack_logs_stealth.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

# Paths relative to repo root (script is in scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "benchmark"
ATTACK_LOGS = DATA / "attack_logs"
ATTACK_LOGS_STEALTH = DATA / "attack_logs_stealth"

# Regex for iteration lines: [openevolve] ITERATION N/None or ITERATION N/50
ITERATION_RE = re.compile(r"\[openevolve\]\s+ITERATION\s+(\d+)/", re.IGNORECASE)


def parse_adaptive_log(log_path: Path, is_stealth: bool) -> tuple[bool, int | None]:
    """
    Parse a single *_adaptive.log file.
    Returns (success: bool, iterations: int | None).
    If success is False, iterations is None (caller should not count).
    If success is True, iterations is the number of OpenEvolve iterations (0 = first-try success).
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Warning: could not read {log_path}: {e}")
        return False, None

    # Success: attack was achieved in the end.
    # Non-stealth: last "Adaptive Test Result" in the log is the final outcome.
    # Stealth: "adaptive_success=True" means they cached a successful attack (attack+stealth passed).
    if is_stealth:
        success = "adaptive_success=True" in text and "Caching optimized attack" in text
    else:
        # Use last occurrence so verification result overrides any earlier step result
        last_passed = [m.end() for m in re.finditer(r"Adaptive Test Result:\s*✓\s*PASSED", text)]
        last_failed = [m.end() for m in re.finditer(r"Adaptive Test Result:\s*✗\s*FAILED", text)]
        if not last_passed and not last_failed:
            success = False
        elif not last_failed:
            success = True
        elif not last_passed:
            success = False
        else:
            success = max(last_passed) > max(last_failed)

    if not success:
        return False, None

    # First-try success: no optimization was run.
    if "attempting optimization" not in text and "Trying optimization strategy: openevolve" not in text:
        return True, 0

    # OpenEvolve was run: count iterations (max ITERATION N seen).
    matches = ITERATION_RE.findall(text)
    if not matches:
        # Success but no ITERATION line (e.g. success in initial batch before loop) -> 0
        return True, 0
    max_iter = max(int(m) for m in matches)
    return True, max_iter


def collect_train_logs(logs_dir: Path) -> list[Path]:
    """Collect all *_adaptive.log files under train_* folders only."""
    if not logs_dir.exists():
        return []
    paths = []
    for train_dir in sorted(logs_dir.iterdir()):
        if not train_dir.is_dir() or not train_dir.name.startswith("train_"):
            continue
        for log_path in train_dir.rglob("*_adaptive.log"):
            paths.append(log_path)
    return paths


def parse_path(log_path: Path, logs_root: Path) -> tuple[str, str] | None:
    """
    Extract (memory_backend, defense) from path.
    Expected: .../train_N/model/topic/memory_backend/defense/XX_adaptive.log
    """
    try:
        rel = log_path.relative_to(logs_root)
        parts = rel.parts
        # train_N, model, topic, memory_backend, defense, file
        if len(parts) >= 6:
            return parts[3], parts[4]
        return None
    except ValueError:
        return None


def build_frequency_table(logs_dir: Path, is_stealth: bool) -> dict[tuple[str, str], dict[int, int]]:
    """
    Returns dict mapping (memory_backend, defense) -> { iteration: count }.
    Only includes successful runs.
    """
    logs = collect_train_logs(logs_dir)
    # (backend, defense) -> list of iteration counts (only successful)
    combo_to_iterations: dict[tuple[str, str], list[int]] = defaultdict(list)

    for log_path in logs:
        key = parse_path(log_path, logs_dir)
        if key is None:
            continue
        success, iterations = parse_adaptive_log(log_path, is_stealth)
        if not success or iterations is None:
            continue
        combo_to_iterations[key].append(iterations)

    # Convert to frequency dicts
    result: dict[tuple[str, str], dict[int, int]] = {}
    for (backend, defense), iters_list in combo_to_iterations.items():
        freq: dict[int, int] = defaultdict(int)
        for i in iters_list:
            freq[i] += 1
        result[(backend, defense)] = dict(freq)
    return result


def write_csv(out_path: Path, freq_table: dict[tuple[str, str], dict[int, int]], source_name: str) -> None:
    """Write frequency table to CSV. Columns: memory_backend, defense, n_success, iter_0, iter_1, ..."""
    if not freq_table:
        out_path.write_text("memory_backend,defense,n_success\n", encoding="utf-8")
        return

    all_iterations: set[int] = set()
    for f in freq_table.values():
        all_iterations.update(f.keys())
    iter_cols = sorted(all_iterations)

    rows = []
    for (backend, defense), freq in sorted(freq_table.items()):
        n_success = sum(freq.values())
        row = [backend, defense, str(n_success)]
        for i in iter_cols:
            row.append(str(freq.get(i, 0)))
        rows.append(row)

    header = ["memory_backend", "defense", "n_success"] + [f"iter_{i}" for i in iter_cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"Wrote {source_name} -> {out_path} ({len(rows)} backend+defense combinations)")


def main() -> None:
    # attack_logs (non-stealth)
    table_standard = build_frequency_table(ATTACK_LOGS, is_stealth=False)
    out_standard = DATA / "openevolve_iteration_frequency_attack_logs.csv"
    write_csv(out_standard, table_standard, "attack_logs")

    # attack_logs_stealth
    table_stealth = build_frequency_table(ATTACK_LOGS_STEALTH, is_stealth=True)
    out_stealth = DATA / "openevolve_iteration_frequency_attack_logs_stealth.csv"
    write_csv(out_stealth, table_stealth, "attack_logs_stealth")


if __name__ == "__main__":
    main()
