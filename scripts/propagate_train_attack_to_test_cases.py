#!/usr/bin/env python3
"""
Propagate attack email into test cases: use cached train for (backend, defense) if successful,
else cached train for (backend, none), else original train file. Writes chosen attack into
attack_bench/test/{backend}/{suite}/*.json.

Cache layout: train_cache/{backend}/{defense}/{suite}/00_*_train.json (one per defense).

Workflow:
1. Generate cases; run adaptive on train for each (backend, defense). Cache is per defense.
2. Run this script with --defense D (or default: none). For each (backend, suite), attack =
   cache(backend, D) else cache(backend, none) else original train file. Copies into test cases.
3. Run run_benchmark.py on attack_bench/test/... for the desired defenses.

Usage:
    python scripts/propagate_train_attack_to_test_cases.py --defense none
    python scripts/propagate_train_attack_to_test_cases.py --defense user_prompt_only --dry-run
    python scripts/propagate_train_attack_to_test_cases.py --defense none --stealth   # prefer _stealth cache if present
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Default paths (attack_bench contains train/, test/, train_cache/)
DEFAULT_ATTACK_BENCH = BASE_DIR / "data" / "benchmark" / "attack_bench"
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "benchmark" / "attack_bench" / "train_cache"

# Train files are identified by having "train" in the filename stem (e.g. 00_*_train.json)
TRAIN_STEM_MARKER = "train"


def get_attack_email_from_cached(cached_path: Path) -> Optional[dict]:
    """Extract the attack email from the first insert_attack_email step in a cached test JSON."""
    if not cached_path.exists():
        return None
    with open(cached_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email" and "attack_email" in step:
            return step["attack_email"]
    return None


def propagate_attack_to_file(test_case_path: Path, attack_email: dict) -> bool:
    """Set step 1 (insert_attack_email) attack_email in test_case_path. Returns True if updated."""
    with open(test_case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    updated = False
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email":
            step["attack_email"] = attack_email
            updated = True
            break
    if updated:
        with open(test_case_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return updated


def discover_suites(attack_bench_base: Path) -> List[Tuple[str, Path]]:
    """Discover (backend, test_suite_dir) for each suite under attack_bench/test/<backend>/<suite>/."""
    suites = []
    test_base = attack_bench_base / "test"
    if not test_base.exists():
        return suites
    for backend_dir in test_base.iterdir():
        if not backend_dir.is_dir():
            continue
        for suite_dir in backend_dir.iterdir():
            if suite_dir.is_dir() and any(suite_dir.glob("*.json")):
                suites.append((backend_dir.name, suite_dir))
    return suites


def find_cached_train_file(cache_base: Path, backend: str, defense: str, suite_name: str) -> Optional[Path]:
    """Return path to the cached train file for this backend/defense/suite, or None. Cache layout: train_cache/<backend>/<defense>/<suite>/."""
    cache_suite_dir = cache_base / backend / defense / suite_name
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if TRAIN_STEM_MARKER in p.stem:
            return p
    return None


def find_cached_train_file_stealth(cache_base: Path, backend: str, defense: str, suite_name: str) -> Optional[Path]:
    """Return path to the cached train file with _stealth (saved when adaptive --stealth achieved both attack and stealth). Same layout as find_cached_train_file."""
    cache_suite_dir = cache_base / backend / defense / suite_name
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if TRAIN_STEM_MARKER in p.stem and "_stealth" in p.stem:
            return p
    return None


def find_train_file_in_bench(attack_bench_base: Path, backend: str, suite_name: str) -> Optional[Path]:
    """Return path to the train file in attack_bench/train/<backend>/<suite>/, or None."""
    train_suite_dir = attack_bench_base / "train" / backend / suite_name
    if not train_suite_dir.exists():
        return None
    for p in train_suite_dir.glob("*.json"):
        if TRAIN_STEM_MARKER in p.stem:
            return p
    return None


def get_attack_email_from_file(json_path: Path) -> Optional[dict]:
    """Extract the attack email from the first insert_attack_email step in a test JSON file."""
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email" and "attack_email" in step:
            return step["attack_email"]
    return None


def run(
    attack_bench_base: Path,
    cache_base: Path,
    defense: str,
    dry_run: bool = False,
    stealth: bool = False,
) -> int:
    """For each (backend, suite): use attack from cache(backend, defense) else cache(backend, none) else original train file; write into test cases. If stealth=True, prefer _stealth cache first."""
    updated_count = 0
    skipped_no_attack = []
    skipped_no_test_files = []

    for backend, suite_dir in discover_suites(attack_bench_base):
        suite_name = suite_dir.name
        attack_email = None
        source_label = None

        if stealth:
            # 1a) Prefer stealth cache for this defense
            cached_stealth = find_cached_train_file_stealth(cache_base, backend, defense, suite_name)
            if cached_stealth:
                attack_email = get_attack_email_from_cached(cached_stealth)
                if attack_email:
                    source_label = f"cached (stealth) {backend}/{defense}/{suite_name}"
            # 1b) Fallback: stealth cache for defense=none
            if not attack_email and defense != "none":
                cached_stealth_none = find_cached_train_file_stealth(cache_base, backend, "none", suite_name)
                if cached_stealth_none:
                    attack_email = get_attack_email_from_cached(cached_stealth_none)
                    if attack_email:
                        source_label = f"cached (stealth) {backend}/none/{suite_name} (fallback)"

        # 2) Normal cache for this defense (or first step when not stealth)
        if not attack_email:
            cached_train = find_cached_train_file(cache_base, backend, defense, suite_name)
            if cached_train:
                attack_email = get_attack_email_from_cached(cached_train)
                if attack_email:
                    source_label = source_label or f"cached {backend}/{defense}/{suite_name}"
        # 3) Fallback: cache for defense=none (unless we already used it)
        if not attack_email and defense != "none":
            cached_none = find_cached_train_file(cache_base, backend, "none", suite_name)
            if cached_none:
                attack_email = get_attack_email_from_cached(cached_none)
                if attack_email:
                    source_label = source_label or f"cached {backend}/none/{suite_name} (fallback)"
        # 4) Fallback: original train file
        if not attack_email:
            bench_train = find_train_file_in_bench(attack_bench_base, backend, suite_name)
            if bench_train:
                attack_email = get_attack_email_from_file(bench_train)
                if attack_email:
                    source_label = f"original train {backend}/{suite_name}"
                else:
                    skipped_no_attack.append(f"{backend}/{suite_name}")
                    continue
            else:
                skipped_no_attack.append(f"{backend}/{suite_name}")
                continue

        if not attack_email:
            skipped_no_attack.append(f"{backend}/{suite_name}")
            continue

        test_files = [
            p for p in suite_dir.glob("*.json")
            if TRAIN_STEM_MARKER not in p.stem
        ]
        if not test_files:
            skipped_no_test_files.append(f"{backend}/{suite_name}")
            continue

        print(f"  {backend}/{suite_name}: {source_label} → {len(test_files)} test file(s)")
        for test_path in sorted(test_files):
            if dry_run:
                print(f"    [dry-run] would update {test_path.name}")
                updated_count += 1
            else:
                if propagate_attack_to_file(test_path, attack_email):
                    print(f"    updated {test_path.name}")
                    updated_count += 1

    if skipped_no_attack:
        print(f"\nSkipped (no attack source): {', '.join(skipped_no_attack)}")
    if skipped_no_test_files:
        print(f"Skipped (no non-train test files): {', '.join(skipped_no_test_files)}")
    print(f"\nTotal files updated: {updated_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy attack email into test cases: cache(defense) else cache(none) else original train."
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        help="Defense type for which to choose attack (default: none). Fallback: cache(none) then original.",
    )
    parser.add_argument(
        "--attack-bench",
        type=Path,
        default=DEFAULT_ATTACK_BENCH,
        help="Base directory for attack_bench (default: data/benchmark/attack_bench)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Base directory for train cache (default: data/benchmark/attack_bench/train_cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be updated, do not write files",
    )
    parser.add_argument(
        "--stealth",
        action="store_true",
        help="Prefer attack from cached _stealth file (from adaptive --stealth run). If not found, fall back to normal cache then original train.",
    )
    args = parser.parse_args()

    attack_bench_base = args.attack_bench.resolve()
    cache_base = args.cache_dir.resolve()
    print(f"Attack bench: {attack_bench_base}")
    print(f"Cache dir:    {cache_base}")
    print(f"Defense:     {args.defense} (fallback: cache(none) then original train)")
    if args.stealth:
        print("Stealth:     enabled (prefer _stealth cache)")
    if args.dry_run:
        print("Dry run: yes")
    print()
    return run(attack_bench_base, cache_base, defense=args.defense, dry_run=args.dry_run, stealth=args.stealth)


if __name__ == "__main__":
    sys.exit(main())
