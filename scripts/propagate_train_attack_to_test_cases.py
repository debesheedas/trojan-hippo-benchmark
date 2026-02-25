#!/usr/bin/env python3
"""
Propagate attack email into test cases. For each (topic, backend, defense) in test*:
  attack = cache(same backend, same defense) else cache(same backend, none) else original train file.
Writes chosen attack into attack_bench/test*/<topic>/<backend>/<defense>/*.json.

Cache layout: train_cache/<model>/<topic>/<backend>/<defense>/... or train_cache_10/<topic>/<backend>/<defense>/...

Usage:
    python scripts/propagate_train_attack_to_test_cases.py --model gemini-3.1-pro-preview
    python scripts/propagate_train_attack_to_test_cases.py --model gemini-3.1-pro-preview --stealth
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Default paths (attack_bench contains train/, test*, train_cache/)
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


def _discover_combinations_in_split(
    attack_bench_base: Path, split_name: str
) -> List[Tuple[str, str, str, Path]]:
    """
    Discover (topic, backend, defense, dir) for each combination under
    attack_bench/<split>/<topic>/<backend>/<defense>/.
    """
    combos: List[Tuple[str, str, str, Path]] = []
    base = attack_bench_base / split_name
    if not base.exists():
        return combos
    for topic_dir in base.iterdir():
        if not topic_dir.is_dir():
            continue
        topic = topic_dir.name
        for backend_dir in topic_dir.iterdir():
            if not backend_dir.is_dir():
                continue
            backend = backend_dir.name
            for defense_dir in backend_dir.iterdir():
                if not defense_dir.is_dir():
                    continue
                defense = defense_dir.name
                if any(defense_dir.glob("*.json")):
                    combos.append((topic, backend, defense, defense_dir))
    return combos


def _split_from_defense_dir(defense_dir: Path, attack_bench_base: Path) -> str:
    """Return test split name from defense_dir path, e.g. test_10 or test."""
    try:
        rel = defense_dir.relative_to(attack_bench_base)
        parts = rel.parts
        if parts:
            return parts[0]
    except ValueError:
        pass
    return "test"


def _train_split_and_cache_dir(test_split: str) -> Tuple[str, str]:
    """Return (train_split, cache_dir_name) for a test split. test_10 -> (train_10, train_cache_10)."""
    if test_split != "test" and test_split.startswith("test_"):
        suffix = test_split[4:]  # "test_10" -> "_10", "test_4" -> "_4"
        return f"train{suffix}", f"train_cache{suffix}"
    return "train", "train_cache"


def find_cached_train_file(
    cache_base: Path, model: str, topic: str, backend: str, defense: str,
    layout_without_model: bool = False,
) -> Optional[Path]:
    """Return path to the cached train file, or None.
    If layout_without_model=True (train_cache_10 style): cache_base/<topic>/<backend>/<defense>/*.json.
    Else (train_cache style): cache_base/<model>/<topic>/<backend>/<defense>/*.json.
    """
    if layout_without_model:
        cache_suite_dir = cache_base / topic / backend / defense
    else:
        cache_suite_dir = cache_base / model / topic / backend / defense
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if "_stealth" in p.stem:
            continue
        if TRAIN_STEM_MARKER in p.stem or layout_without_model:
            return p
    return None


def find_cached_train_file_stealth(
    cache_base: Path, model: str, topic: str, backend: str, defense: str,
    layout_without_model: bool = False,
) -> Optional[Path]:
    """Return path to the cached train file with _stealth (saved when adaptive --stealth achieved both attack and stealth)."""
    if layout_without_model:
        cache_suite_dir = cache_base / topic / backend / defense
    else:
        cache_suite_dir = cache_base / model / topic / backend / defense
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if "_stealth" not in p.stem:
            continue
        if TRAIN_STEM_MARKER in p.stem or layout_without_model:
            return p
    return None


def find_train_file_in_bench(
    attack_bench_base: Path, topic: str, backend: str, defense: str,
    train_split: str = "train",
) -> Optional[Path]:
    """Return path to the train file in attack_bench/<train_split>/<topic>/<backend>/<defense>/, or None."""
    train_suite_dir = attack_bench_base / train_split / topic / backend / defense
    if not train_suite_dir.exists():
        return None
    for p in train_suite_dir.glob("*.json"):
        if TRAIN_STEM_MARKER in p.stem:
            return p
    # Persistence train files are named 01.json, not *_train.json; accept any json as train for that dir
    for p in train_suite_dir.glob("*.json"):
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
    model: str,
    stealth: bool = False,
) -> int:
    """
    For each (topic, backend, defense): use attack from cache(same defense) else cache(none)
    else original train file; write into test cases. Fallback to defense=none is automatic.
    If stealth=True, prefer _stealth cache first.
    """
    updated_count = 0
    skipped_no_attack = []
    skipped_no_test_files = []

    # Discover (topic, backend, defense) under all test* subfolders (e.g., test, test_10, ...)
    combo_dirs: List[Tuple[str, str, str, Path]] = []
    for split_dir in sorted(d for d in attack_bench_base.iterdir() if d.is_dir() and d.name.startswith("test")):
        combo_dirs.extend(_discover_combinations_in_split(attack_bench_base, split_dir.name))

    for topic, backend, defense, defense_dir in combo_dirs:
        suite_name = topic
        attack_email = None
        source_label = None

        # Per-split cache and train: test_10 -> train_cache_10 + train_10 (no model in path)
        test_split = _split_from_defense_dir(defense_dir, attack_bench_base)
        train_split, cache_dir_name = _train_split_and_cache_dir(test_split)
        if cache_dir_name == "train_cache":
            split_cache_base = cache_base  # use --cache-dir (default attack_bench/train_cache)
        else:
            split_cache_base = attack_bench_base / cache_dir_name  # e.g. attack_bench/train_cache_10
        layout_without_model = cache_dir_name != "train_cache"

        if stealth:
            # 1a) Prefer stealth cache for this defense
            cached_stealth = find_cached_train_file_stealth(
                split_cache_base, model, topic, backend, defense, layout_without_model=layout_without_model
            )
            if cached_stealth:
                attack_email = get_attack_email_from_cached(cached_stealth)
                if attack_email:
                    source_label = f"cached (stealth) {backend}/{defense}/{suite_name}"
            # 1b) Fallback: stealth cache for defense=none
            if not attack_email and defense != "none":
                cached_stealth_none = find_cached_train_file_stealth(
                    split_cache_base, model, topic, backend, "none", layout_without_model=layout_without_model
                )
                if cached_stealth_none:
                    attack_email = get_attack_email_from_cached(cached_stealth_none)
                    if attack_email:
                        source_label = f"cached (stealth) {backend}/none/{suite_name} (fallback)"

        # 2) Normal cache for this defense (or first step when not stealth)
        if not attack_email:
            cached_train = find_cached_train_file(
                split_cache_base, model, topic, backend, defense, layout_without_model=layout_without_model
            )
            if cached_train:
                attack_email = get_attack_email_from_cached(cached_train)
                if attack_email:
                    source_label = source_label or f"cached {backend}/{defense}/{suite_name}"
        # 3) Fallback: cache for defense=none (unless we already used it)
        if not attack_email and defense != "none":
            cached_none = find_cached_train_file(
                split_cache_base, model, topic, backend, "none", layout_without_model=layout_without_model
            )
            if cached_none:
                attack_email = get_attack_email_from_cached(cached_none)
                if attack_email:
                    source_label = source_label or f"cached {backend}/none/{suite_name} (fallback)"
        # 4) Fallback: original train file from same split (train_10 for test_10, train for test)
        if not attack_email:
            bench_train = find_train_file_in_bench(
                attack_bench_base, topic, backend, defense, train_split=train_split
            )
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

        test_files = [p for p in defense_dir.glob("*.json") if TRAIN_STEM_MARKER not in p.stem]
        if not test_files:
            skipped_no_test_files.append(f"{backend}/{suite_name}")
            continue

        split_label = f" [{test_split}]" if test_split != "test" else ""
        print(f"  {topic}/{backend}/{defense}{split_label}: {source_label} → {len(test_files)} test file(s)")
        for test_path in sorted(test_files):
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
        description="Copy attack email into test cases: for each (backend, defense), use cache(same defense) else cache(none) else original train."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name whose train_cache should be used (e.g. gemini-3.1-pro-preview).",
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
        "--stealth",
        action="store_true",
        help="Prefer attack from cached _stealth file (from adaptive --stealth run). If not found, fall back to normal cache then original train.",
    )
    args = parser.parse_args()

    attack_bench_base = args.attack_bench.resolve()
    cache_base = args.cache_dir.resolve()
    print(f"Attack bench: {attack_bench_base}")
    print(f"Cache dir:    {cache_base}")
    print(f"Model:       {args.model}")
    if args.stealth:
        print("Stealth:     enabled (prefer _stealth cache)")
    print()
    return run(
        attack_bench_base,
        cache_base,
        model=args.model,
        stealth=args.stealth,
    )


if __name__ == "__main__":
    sys.exit(main())
