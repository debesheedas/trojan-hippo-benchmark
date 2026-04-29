#!/usr/bin/env python3
"""
One-time migration: move existing train_cache and train_cache_* contents into a model subfolder.

Current layout (old):
  train_cache/topic/backend/defense/*.json
  train_cache_10/topic/backend/defense/*.json

New layout (same as attack_logs / attack_results):
  train_cache/<model>/topic/backend/defense/*.json
  train_cache_10/<model>/topic/backend/defense/*.json

Existing cache files were produced with target model gemini-3.1-pro-preview, so we move
everything into train_cache*/gemini-3.1-pro-preview/.

Usage:
  python scripts/migrate_train_cache_to_model_subfolder.py
  python scripts/migrate_train_cache_to_model_subfolder.py --model gpt-5-mini  # if your cache was from another model
  python scripts/migrate_train_cache_to_model_subfolder.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ATTACK_BENCH = BASE_DIR / "data" / "benchmark" / "attack_bench"
ATTACK_BENCH_STEALTH = BASE_DIR / "data" / "benchmark" / "attack_bench_stealth"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def is_likely_model_subfolder(name: str) -> bool:
    """Heuristic: names that look like model IDs (contain dots or known prefixes)."""
    if not name or name.startswith("."):
        return False
    # Common patterns: gemini-3.1-pro-preview, gpt-5-mini, gpt-4o-mini, etc.
    if "." in name or "gemini" in name.lower() or "gpt" in name.lower() or "claude" in name.lower():
        return True
    return False


def migrate_one_cache_dir(cache_dir: Path, model: str, dry_run: bool) -> tuple[int, int]:
    """
    If cache_dir has top-level dirs that are NOT the model name, move them under cache_dir/model/.
    Returns (moved_count, skipped_count).
    """
    if not cache_dir.is_dir():
        return 0, 0
    children = [d for d in cache_dir.iterdir() if d.is_dir()]
    if not children:
        return 0, 0
    # Already migrated if model subfolder exists and has content
    model_sub = cache_dir / model
    if model_sub.is_dir() and any(model_sub.iterdir()):
        return 0, len(children)
    # If any child looks like a model name, assume already migrated
    if any(is_likely_model_subfolder(d.name) for d in children):
        return 0, len(children)
    if not dry_run:
        model_sub.mkdir(parents=True, exist_ok=True)
    moved = 0
    for d in children:
        dest = model_sub / d.name
        if dest.exists():
            continue
        if dry_run:
            print(f"  [dry-run] would move {d.relative_to(cache_dir.parent)} -> {dest.relative_to(cache_dir.parent)}")
            moved += 1
            continue
        shutil.move(str(d), str(dest))
        print(f"  moved {d.name} -> {model}/{d.name}")
        moved += 1
    return moved, len(children) - moved


def main():
    parser = argparse.ArgumentParser(description="Move train_cache* contents under <model>/ subfolder.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use as subfolder (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be moved",
    )
    parser.add_argument(
        "--stealth",
        action="store_true",
        help="Also migrate attack_bench_stealth/train_cache*",
    )
    args = parser.parse_args()
    model = args.model

    # Discover train_cache and train_cache_* under attack_bench
    cache_roots = []
    if ATTACK_BENCH.is_dir():
        for d in ATTACK_BENCH.iterdir():
            if d.is_dir() and (d.name == "train_cache" or (d.name.startswith("train_cache_") and d.name != "train_cache")):
                cache_roots.append(d)
    if args.stealth and ATTACK_BENCH_STEALTH.is_dir():
        for d in ATTACK_BENCH_STEALTH.iterdir():
            if d.is_dir() and (d.name == "train_cache" or (d.name.startswith("train_cache_") and d.name != "train_cache")):
                cache_roots.append(d)

    if not cache_roots:
        print("No train_cache or train_cache_* directories found.")
        return 0

    print(f"Model subfolder: {model}")
    if args.dry_run:
        print("Dry run (no changes)")
    print()
    total_moved = 0
    for cache_dir in sorted(cache_roots):
        rel = cache_dir.relative_to(BASE_DIR)
        print(f"{rel}:")
        moved, skipped = migrate_one_cache_dir(cache_dir, model, args.dry_run)
        total_moved += moved
        if moved == 0 and skipped == 0:
            print("  (empty or already migrated)")
        print()
    print(f"Total top-level dirs moved under '<model>/': {total_moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
