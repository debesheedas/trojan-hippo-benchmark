#!/usr/bin/env python3
"""
Compare attack_bench results before vs after changing the semantic judge model.
Usage:
  PYTHONPATH=src python scripts/compare_semantic_judge_results.py

Expects:
  - Backup (before): data/benchmark/attack_results_backup_before_semantic_judge_change/finance/
  - Current (after): data/benchmark/attack_results/test_0/gemini-3.1-pro-preview/finance/
"""

import json
from pathlib import Path

BACKUP_BASE = Path("data/benchmark/attack_results_backup_before_semantic_judge_change/finance")
CURRENT_BASE = Path("data/benchmark/attack_results/test_0/gemini-3.1-pro-preview/finance")


def load_result(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error reading {p}: {e}")
        return None


def main():
    print("Comparing BEFORE (semantic judge: Pro) vs AFTER (semantic judge: Flash)")
    print("Backup:", BACKUP_BASE.resolve())
    print("Current:", CURRENT_BASE.resolve())
    print()

    if not BACKUP_BASE.exists():
        print("Backup directory not found. Run backup first.")
        return 1
    if not CURRENT_BASE.exists():
        print("Current results not found. Run the benchmark first:")
        print("  PYTHONPATH=src python scripts/run_benchmark.py --model gemini-3.1-pro-preview \\")
        print("    --test data/benchmark/attack_bench/test_0/finance/context/none \\")
        print("    --memory-backend context --defense-type none --force")
        return 1

    same = 0
    diff = 0
    only_backup = 0
    only_current = 0
    details = []

    for backend in ["context", "explicit", "mem0", "rag"]:
        backup_dir = BACKUP_BASE / backend / "none"
        current_dir = CURRENT_BASE / backend / "none"
        for name in ["01.json", "02.json", "03.json", "04.json"]:
            b = load_result(backup_dir / name)
            c = load_result(current_dir / name)
            if b is None and c is None:
                continue
            if b is None:
                only_current += 1
                details.append((f"{backend}/none/{name}", None, c))
                continue
            if c is None:
                only_backup += 1
                details.append((f"{backend}/none/{name}", b, None))
                continue
            succ_b = b.get("overall_success", None)
            succ_c = c.get("overall_success", None)
            if succ_b == succ_c:
                same += 1
            else:
                diff += 1
                details.append((f"{backend}/none/{name}", b, c))

    print("Result summary (overall_success):")
    print(f"  Same (before == after): {same}")
    print(f"  Different:               {diff}")
    print(f"  Only in backup:          {only_backup}")
    print(f"  Only in current:         {only_current}")
    print()

    if details:
        print("Differences (file -> before (Pro) -> after (Flash)):")
        for path, before, after in details:
            if before is None:
                print(f"  {path}: (no backup) -> overall_success={after.get('overall_success')}")
            elif after is None:
                print(f"  {path}: overall_success={before.get('overall_success')} -> (no current)")
            else:
                print(f"  {path}: {before.get('overall_success')} -> {after.get('overall_success')}")
    elif diff == 0:
        print("All compared results have the same overall_success.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
