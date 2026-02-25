"""
Build memory snapshots for persistence benchmarking.

Runs the single benign test case (persistence_unrelated_20) for each valid
(memory_backend, defense) combination. At the end of every session, the test
runner saves a snapshot to data/benchmark/snapshots/memory_snapshots/{backend}/{defense}/session_{n}.json.

Usage:
  python scripts/build_persistence_snapshots.py [--backend explicit] [--num-workers 4]
  python scripts/build_persistence_snapshots.py   # all backends and defenses, serial

Snapshots are written only for backends that implement save_snapshot (currently: explicit, none).
Other backends will log "Snapshot save not implemented" and continue.
"""

import argparse
import os
import sys
from pathlib import Path

# Fix OpenMP initialization error on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))  # for scripts.run_benchmark

from benchmark.benchmark_utils import (
    MEMORY_BACKENDS,
    UNIFIED_DEFENSE_TYPES,
    is_valid_combination,
)
from scripts.run_benchmark import run_all_combinations

# Snapshot layout:
#   data/benchmark/snapshots/unrelated_sessions.json
#   data/benchmark/snapshots/memory_snapshots/{backend}/{defense}/session_{n}.json
#   data/benchmark/snapshots/results/...
#   data/benchmark/snapshots/logs/...
SNAPSHOTS_ROOT = Path("data/benchmark/snapshots")
PERSISTENCE_TEST_PATH_LEGACY = str(SNAPSHOTS_ROOT / "unrelated_sessions.json")
PERSISTENCE_TEST_PATH_TRAIN = str(SNAPSHOTS_ROOT / "unrelated_sessions_train.json")
PERSISTENCE_TEST_PATH_TEST = str(SNAPSHOTS_ROOT / "unrelated_sessions_test.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run persistence_unrelated_20 for each (backend, defense) and save memory snapshots at end of each session."
    )
    parser.add_argument(
        "--backend",
        type=str,
        action="append",
        dest="backends",
        default=None,
        help="Memory backend to run (can repeat). Default: all.",
    )
    parser.add_argument(
        "--defense",
        type=str,
        action="append",
        dest="defenses",
        default=None,
        help="Defense type to run (can repeat). Default: all.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers (default 1 = serial).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Benchmark config path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing snapshot files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which benign snapshot split to run (train, test, or both). Default: both.",
    )
    args = parser.parse_args()

    memory_backends = args.backends if args.backends else list(MEMORY_BACKENDS)
    defense_types = args.defenses if args.defenses else list(UNIFIED_DEFENSE_TYPES)

    # Helper to resolve a test path relative to BASE_DIR if needed
    def _resolve_test_path(path_str: str) -> str:
        p = Path(path_str)
        if p.exists():
            return str(p)
        alt = BASE_DIR / path_str
        if alt.exists():
            return str(alt)
        return ""

    overall_ok = True

    # Decide which splits to run
    splits_to_run = []
    if args.split in ("train", "both"):
        splits_to_run.append(("train", PERSISTENCE_TEST_PATH_TRAIN))
    if args.split in ("test", "both"):
        splits_to_run.append(("test", PERSISTENCE_TEST_PATH_TEST))

    # Fallback: if split files do not exist yet, try legacy single-file layout
    if not splits_to_run:
        resolved_legacy = _resolve_test_path(PERSISTENCE_TEST_PATH_LEGACY)
        if not resolved_legacy:
            print(
                "ERROR: No persistence test files found. "
                "Generate them via generate_persistence_unrelated_snapshots.py."
            )
            sys.exit(1)
        splits_to_run.append(("legacy", resolved_legacy))

    for split_name, path_str in splits_to_run:
        resolved = _resolve_test_path(path_str)
        if not resolved:
            print(f"ERROR: Persistence test not found for split '{split_name}': {path_str}")
            overall_ok = False
            continue

        # Per-split results/logs layout:
        #   data/benchmark/snapshots/results/<split>/<model_name>/...
        #   data/benchmark/snapshots/logs/<split>/<model_name>/...
        snapshot_results_base = SNAPSHOTS_ROOT / "results" / split_name
        snapshot_logs_base = SNAPSHOTS_ROOT / "logs" / split_name
        snapshot_results_base.mkdir(parents=True, exist_ok=True)
        snapshot_logs_base.mkdir(parents=True, exist_ok=True)

        print(f"Persistence test ({split_name}): {resolved}")
        print(
            "Snapshots will be written to: "
            "data/benchmark/snapshots/memory_snapshots_<split>/<backend>/<defense>/session_<n>.json "
            "(split=train or split=test, inferred from snapshot_set_id)."
        )
        print(f"Benign run results will be written to: {snapshot_results_base}")
        print(f"Benign run logs will be written to: {snapshot_logs_base}")
        print()

        result = run_all_combinations(
            memory_backends=memory_backends,
            defense_types=defense_types,
            test_path=resolved,
            config_path=args.config,
            force=args.force,
            adaptive=False,
            stealth=False,
            num_workers=args.num_workers,
            results_base_dir=snapshot_results_base,
            logs_base_dir=snapshot_logs_base,
            target_model_name=None,
        )

        ok = result.get("failed_combinations", 0) == 0 and not result.get("combinations_with_errors")
        overall_ok = overall_ok and ok
        print()
        if ok:
            print(f"Snapshot build for split '{split_name}' completed. Check {SNAPSHOTS_ROOT}")
        else:
            print(f"Some combinations for split '{split_name}' had errors. Check logs in {snapshot_logs_base}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
