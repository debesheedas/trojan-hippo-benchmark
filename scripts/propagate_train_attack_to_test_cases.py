#!/usr/bin/env python3
"""
Propagate successful attack email from cached train test case to all test cases.

Workflow:
1. Run dataset generation (1 train + 4 test cases per attack class). Outputs go to
   attack_bench/train/{backend}/{suite}/ and attack_bench/test/{backend}/{suite}/.
2. Run run_benchmark.py with --adaptive on the TRAIN folder for each class, e.g.:
   --test data/benchmark/attack_bench/train/rag/persistent_exfiltrate_tax
   When openevolve finds a successful attack, it is cached under attack_bench/train_cache/
   with structure train_cache/{backend}/{suite}/00_*_train.json.
3. Run this script: for each (backend, suite), it finds the cached train file in
   train_cache/{backend}/{suite}/*_train.json, reads the attack email from step 1, and
   copies it into step 1 of every test case in attack_bench/test/{backend}/{suite}/.
   For memory_backend=none there is typically no cache; the script uses the attack email
   from attack_bench/train/{backend}/{suite}/*_train.json instead and prints a warning.
4. Run run_benchmark.py (non-adaptive) on the test folder, e.g.:
   --test data/benchmark/attack_bench/test/rag/persistent_exfiltrate_tax

Usage:
    python scripts/propagate_train_attack_to_test_cases.py
    python scripts/propagate_train_attack_to_test_cases.py --attack-bench data/benchmark/attack_bench --cache-dir data/benchmark/attack_bench/train_cache
    python scripts/propagate_train_attack_to_test_cases.py --dry-run
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


def find_cached_train_file(cache_base: Path, backend: str, suite_name: str) -> Optional[Path]:
    """Return path to the cached train file for this suite, or None. Cache layout: train_cache/<backend>/<suite>/."""
    cache_suite_dir = cache_base / backend / suite_name
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if TRAIN_STEM_MARKER in p.stem:
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
    dry_run: bool = False,
) -> int:
    updated_count = 0
    skipped_no_cache = []
    skipped_no_test_files = []

    for backend, suite_dir in discover_suites(attack_bench_base):
        suite_name = suite_dir.name
        cached_train = find_cached_train_file(cache_base, backend, suite_name)
        attack_email = None
        source_label = None

        if cached_train:
            attack_email = get_attack_email_from_cached(cached_train)
            source_label = f"cached train {cached_train.name}"
        else:
            # No cache: for memory_backend=none, adaptive attack cannot succeed (no memory to exploit),
            # so use the attack email from the attack_bench train file instead.
            if backend == "none":
                bench_train = find_train_file_in_bench(attack_bench_base, backend, suite_name)
                if bench_train:
                    attack_email = get_attack_email_from_file(bench_train)
                    source_label = f"attack_bench train {bench_train.name} (no cache)"
                    print(
                        f"  WARNING: No cache for {backend}/{suite_name} — adaptive attack did not produce a successful attack "
                        f"(expected for memory_backend=none). Using attack email from attack_bench train file instead.",
                        file=sys.stderr,
                    )
                else:
                    skipped_no_cache.append(f"{backend}/{suite_name}")
                    continue
            else:
                skipped_no_cache.append(f"{backend}/{suite_name}")
                continue

        if not attack_email:
            print(f"  WARNING: No insert_attack_email step in source for {backend}/{suite_name}", file=sys.stderr)
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

    if skipped_no_cache:
        print(f"\nSkipped (no cached train file): {', '.join(skipped_no_cache)}")
    if skipped_no_test_files:
        print(f"Skipped (no non-train test files): {', '.join(skipped_no_test_files)}")
    print(f"\nTotal files updated: {updated_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy attack email from cached train test case into all test cases for each attack class."
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
    args = parser.parse_args()

    attack_bench_base = args.attack_bench.resolve()
    cache_base = args.cache_dir.resolve()
    print(f"Attack bench: {attack_bench_base}")
    print(f"Cache dir:    {cache_base}")
    if args.dry_run:
        print("Dry run: yes")
    print()
    return run(attack_bench_base, cache_base, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
