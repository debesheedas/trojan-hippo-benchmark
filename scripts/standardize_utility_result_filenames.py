#!/usr/bin/env python3
"""
Standardize utility suite result filenames to canonical numeric form.

Some runs saved results as legacy filenames like:
  <suite>_001.json
while newer code/tests save as:
  001.json

Consolidation aggregates *all* *.json in a folder, so mixing both naming
schemes doubles (or skews) metrics. This script renames/moves legacy files
so each suite folder contains only:
  001.json .. 004.json

Safety:
  - Keeps a backup of any file it needs to remove/overwrite.
  - Uses mtime to decide which of {NN.json, <suite>_NN.json} is newer.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UTILITY_SUITES = [
    "memory_only",
    "assistant_responses",
    "untrusted_probe",
    "untrusted_send",
    "disable_send",
    "memory_tools",
    "long_memory",
]


def _extract_indices_from_test_dir(test_dir: Path) -> List[str]:
    """
    Extract indices like 001..004 from filenames in data/benchmark/tests/<suite>/.
    Accepts both numeric and legacy patterns: 001.json or <prefix>_001.json.
    """
    idxs: List[str] = []
    for p in test_dir.glob("*.json"):
        m = re.match(r"^(?:\d{3}|.+_(\d{3}))\.json$", p.name)
        if m:
            if p.stem.isdigit():
                idxs.append(p.stem)
            else:
                idxs.append(m.group(1))
    return sorted(set(idxs))


@dataclass(frozen=True)
class PlanAction:
    action: str  # "keep", "delete", "rename", "move"
    suite: str
    idx: str
    from_path: Optional[Path]
    to_path: Optional[Path]
    detail: str


def plan_for_suite_dir(
    suite_dir: Path,
    suite: str,
    expected_idxs: List[str],
    backup_root: Path,
) -> Tuple[List[PlanAction], List[str]]:
    """
    Plan how to canonicalize one suite result directory.
    """
    actions: List[PlanAction] = []
    issues: List[str] = []

    for idx in expected_idxs:
        numeric_path = suite_dir / f"{idx}.json"
        legacy_path = suite_dir / f"{suite}_{idx}.json"

        has_numeric = numeric_path.exists()
        has_legacy = legacy_path.exists()

        if not has_numeric and not has_legacy:
            issues.append(f"Missing both {idx}.json and {suite}_{idx}.json")
            continue

        if has_numeric and has_legacy:
            num_mtime = numeric_path.stat().st_mtime
            leg_mtime = legacy_path.stat().st_mtime
            if num_mtime >= leg_mtime:
                # Keep numeric, remove legacy
                actions.append(
                    PlanAction(
                        action="delete_legacy",
                        suite=suite,
                        idx=idx,
                        from_path=legacy_path,
                        to_path=None,
                        detail=f"numeric newer/equal (num={num_mtime}, legacy={leg_mtime}); move legacy to backup and remove from results",
                    )
                )
            else:
                # Keep legacy content but write into numeric filename
                actions.append(
                    PlanAction(
                        action="replace_numeric_with_legacy",
                        suite=suite,
                        idx=idx,
                        from_path=numeric_path,
                        to_path=numeric_path,  # placeholder; actual target is numeric_path
                        detail=f"legacy newer (num={num_mtime}, legacy={leg_mtime}); backup numeric and move legacy -> {idx}.json",
                    )
                )
        elif has_numeric:
            actions.append(
                PlanAction(
                    action="keep_numeric",
                    suite=suite,
                    idx=idx,
                    from_path=numeric_path,
                    to_path=numeric_path,
                    detail="numeric exists; ok",
                )
            )
        else:
            # Rename legacy -> numeric
            actions.append(
                PlanAction(
                    action="rename_legacy_to_numeric",
                    suite=suite,
                    idx=idx,
                    from_path=legacy_path,
                    to_path=numeric_path,
                    detail="numeric missing; rename legacy -> numeric",
                )
            )

    return actions, issues


def apply_plan(
    plan: List[PlanAction],
    dry_run: bool,
    backup_root: Path,
    results_base_dir: Path,
) -> Tuple[int, int]:
    """
    Apply actions to the filesystem.
    Returns: (files_changed, files_removed_backed_up)
    """
    changed = 0
    backed_up = 0

    for a in plan:
        if a.action == "keep_numeric":
            continue

        # Compute backup paths by rebuilding from detail where needed isn't robust;
        # instead, back up deterministically based on where the file lives.
        assert a.from_path is not None, f"{a.action} requires from_path"

        if a.action == "delete_legacy":
            legacy_path = a.from_path
            backup_path = backup_root / legacy_path.relative_to(results_base_dir)
            # Make backup path unique and end with .bak
            backup_path = backup_path.with_name(backup_path.name + ".bak")
            if not dry_run:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy_path), str(backup_path))
            changed += 1
            backed_up += 1
        elif a.action == "rename_legacy_to_numeric":
            legacy_path = a.from_path
            numeric_path = a.to_path
            if numeric_path is None:
                raise RuntimeError("Missing to_path for rename_legacy_to_numeric")
            if not dry_run:
                numeric_path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path.rename(numeric_path)
            changed += 1
        elif a.action == "replace_numeric_with_legacy":
            # Here from_path is the current numeric file; we also need the legacy sibling.
            numeric_path = a.from_path
            if numeric_path is None:
                raise RuntimeError("Missing from_path for replace_numeric_with_legacy")
            suite_dir = numeric_path.parent
            legacy_path = suite_dir / f"{a.suite}_{a.idx}.json"

            if not legacy_path.exists():
                raise RuntimeError(f"Expected legacy file not found: {legacy_path}")

            backup_numeric_path = backup_root / numeric_path.relative_to(numeric_path.parents[2])
            backup_numeric_path = backup_root / numeric_path.relative_to(results_base_dir)
            backup_numeric_path = backup_numeric_path.with_suffix(backup_numeric_path.suffix + ".bak")

            if not dry_run:
                backup_numeric_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(numeric_path), str(backup_numeric_path))
                numeric_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy_path), str(numeric_path))
            changed += 1
            backed_up += 1
        else:
            raise RuntimeError(f"Unknown action: {a.action}")

    return changed, backed_up


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize utility result filenames to 001.json..")
    parser.add_argument("--results-base-dir", type=str, default="data/benchmark/results")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Limit to these model directories.")
    parser.add_argument("--suites", type=str, nargs="*", default=UTILITY_SUITES, help="Utility suite folders to standardize.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without applying changes.")
    parser.add_argument("--backup-dir", type=str, default="data/benchmark/results_filename_backups")
    args = parser.parse_args()

    results_base_dir = Path(args.results_base_dir)
    tests_base_dir = Path("data/benchmark/tests")

    expected_by_suite: Dict[str, List[str]] = {}
    for suite in args.suites:
        test_dir = tests_base_dir / suite
        if not test_dir.exists():
            raise SystemExit(f"Test dir does not exist: {test_dir}")
        expected_idxs = _extract_indices_from_test_dir(test_dir)
        if not expected_idxs:
            raise SystemExit(f"Could not extract expected indices from: {test_dir}")
        expected_by_suite[suite] = expected_idxs

    models: List[Path] = []
    if args.models:
        for m in args.models:
            p = results_base_dir / m
            if not p.exists():
                raise SystemExit(f"Model dir does not exist: {p}")
            models.append(p)
    else:
        models = [p for p in results_base_dir.iterdir() if p.is_dir()]

    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_root = Path(args.backup_dir) / ts

    global_plan: List[PlanAction] = []
    global_issues: List[str] = []

    for model_dir in sorted(models):
        for suite in args.suites:
            exp_idxs = expected_by_suite[suite]
            # Search for suite folders under this model.
            for suite_dir in model_dir.rglob(suite):
                if not suite_dir.is_dir() or suite_dir.name != suite:
                    continue

                plan, issues = plan_for_suite_dir(suite_dir, suite, exp_idxs, backup_root)
                global_plan.extend(plan)
                for iss in issues:
                    global_issues.append(f"{suite_dir}: {iss}")

    # Print a concise summary
    deletions = sum(1 for a in global_plan if a.action == "delete_legacy")
    renames = sum(1 for a in global_plan if a.action in {"rename_legacy_to_numeric", "replace_numeric_with_legacy"})

    print("==== Standardize Results Filename Plan ====")
    print(f"models: {[p.name for p in models]}")
    print(f"suites: {args.suites}")
    print(f"expected indices per suite: {expected_by_suite}")
    print(f"planned remove legacy files: {deletions}")
    print(f"planned renames/replacements: {renames}")
    if global_issues:
        print(f"issues (missing both numeric and legacy): {len(global_issues)}")
        print("Example issues:")
        for x in global_issues[:20]:
            print(" -", x)
    print("=============================================")

    if args.dry_run:
        print("DRY RUN mode: no filesystem changes.")
        return

    files_changed, backed = apply_plan(
        global_plan,
        dry_run=False,
        backup_root=backup_root,
        results_base_dir=results_base_dir,
    )
    print(f"Done. files_changed={files_changed}, backed_up={backed}, backup_root={backup_root}")


if __name__ == "__main__":
    main()

