"""
Context Memory Shuffle Experiment

Runs the attack benchmark for context/none on test_100 with randomly shuffled
context_history at load_memory_snapshot (session_100). This experiment tests
whether memory order affects attack success rate.

Runs 10 times with different shuffles, then reports mean and std of attack success.
Does NOT modify run_benchmark or the normal load_memory_snapshot flow - uses
a monkey-patch that is only active when this script runs.

Usage:
  python scripts/run_context_shuffle_experiment.py [--num-runs N]

To disable: simply do not run this script. Normal benchmark runs are unaffected.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

# Add project root and src to path before any benchmark imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# Fix OpenMP on macOS (same as run_benchmark)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Monkey-patch: must be applied BEFORE importing run_benchmark
# ---------------------------------------------------------------------------
import benchmark.snapshot_io as _snapshot_io

_SHUFFLE_EXPERIMENT_ACTIVE = False
_SHUFFLE_RUN_COUNT = 0
_ORIGINAL_load_snapshot_context_merge = _snapshot_io.load_snapshot_context_merge


def _patched_load_snapshot_context_merge(
    in_memory_env,
    data,
    current_memory_backend: str,
    current_defense_type: str,
    current_snapshot_set_id: str,
    current_session_index: int,
):
    """Apply shuffle to context_history when session_index=100, context, none."""
    global _SHUFFLE_RUN_COUNT
    if (
        _SHUFFLE_EXPERIMENT_ACTIVE
        and current_memory_backend == "context"
        and current_defense_type == "none"
        and current_session_index == 100
    ):
        payload = data.get("context_history", [])
        if payload and isinstance(payload, list):
            shuffled = copy.deepcopy(payload)
            random.shuffle(shuffled)
            data = {**data, "context_history": shuffled}
            # Debug: log that shuffle was applied
            first_orig = payload[0].get("text", "")[:80] if payload else ""
            first_new = shuffled[0].get("text", "")[:80] if shuffled else ""
            print(
                f"[SHUFFLE_EXPERIMENT] Run {_SHUFFLE_RUN_COUNT}: Shuffled {len(shuffled)} context_history entries. "
                f"Original first 80 chars: {first_orig!r}... -> New first: {first_new!r}..."
            )
    return _ORIGINAL_load_snapshot_context_merge(
        in_memory_env,
        data,
        current_memory_backend,
        current_defense_type,
        current_snapshot_set_id,
        current_session_index,
    )


_snapshot_io.load_snapshot_context_merge = _patched_load_snapshot_context_merge

# Now safe to import run_benchmark (scripts dir in path for direct module import)
sys.path.insert(0, str(BASE_DIR / "scripts"))
import run_benchmark as _run_benchmark_module
run_benchmark = _run_benchmark_module.run_benchmark


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
MEMORY_BACKEND = "context"
DEFENSE = "none"
TEST_PATH = "data/benchmark/attack_bench/test_100"
TARGET_MODEL = "gemini-3.1-pro-preview"
NUM_RUNS = 10
SNAPSHOT_PATH = Path(
    "data/benchmark/snapshots/memory_snapshots_test/context/none/session_100.json"
)
OUTPUT_DIR = Path("data/benchmark/context_shuffle_experiment")


def main():
    global _SHUFFLE_EXPERIMENT_ACTIVE, _SHUFFLE_RUN_COUNT

    parser = argparse.ArgumentParser(
        description="Context memory shuffle experiment: run 10 times with shuffled context_history at load_memory_snapshot."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of runs (default: {NUM_RUNS})",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Single test file path (e.g. data/benchmark/attack_bench/test_100/finance/context/none/01.json). If set, run only this test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify patch and snapshot, then exit without running benchmark",
    )
    args = parser.parse_args()
    num_runs = args.num_runs
    test_path = args.test_file if args.test_file else TEST_PATH

    print("=" * 80)
    print("CONTEXT MEMORY SHUFFLE EXPERIMENT")
    print("=" * 80)
    print(f"Memory backend: {MEMORY_BACKEND}, Defense: {DEFENSE}")
    print(f"Test path: {test_path}")
    print(f"Target model: {TARGET_MODEL}")
    print(f"Number of runs: {num_runs}")
    print(f"Snapshot: {SNAPSHOT_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)

    if not SNAPSHOT_PATH.exists():
        print(f"ERROR: Snapshot not found: {SNAPSHOT_PATH}")
        sys.exit(1)

    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snapshot_data = json.load(f)
    context_history = snapshot_data.get("context_history", [])
    print(f"[DEBUG] Snapshot has {len(context_history)} context_history entries")
    if context_history:
        print(
            f"[DEBUG] First entry preview: {context_history[0].get('text', '')[:100]}..."
        )
    print()

    if args.dry_run:
        print("[DRY-RUN] Patch verified. Snapshot loaded. Exiting.")
        print("[DRY-RUN] Run without --dry-run to execute the full experiment.")
        return

    results_base = OUTPUT_DIR
    logs_base = OUTPUT_DIR / "logs"
    run_results = []
    run_seeds_list = []

    _SHUFFLE_EXPERIMENT_ACTIVE = True

    try:
        for run in range(1, num_runs + 1):
            _SHUFFLE_RUN_COUNT = run
            run_seed = random.randint(1, 2**31 - 1)
            print(f"\n{'#'*80}")
            print(f"RUN {run}/{num_runs} (seed={run_seed})")
            print(f"{'#'*80}\n")

            result = run_benchmark(
                memory_backend=MEMORY_BACKEND,
                unified_defense=DEFENSE,
                test_path=test_path,
                force=True,
                adaptive=False,
                results_base_dir=results_base,
                logs_base_dir=logs_base,
                target_model_name=TARGET_MODEL,
                seed_override=run_seed,
            )

            run_results.append(result)
            run_seeds_list.append(run_seed)
            tests_run = result.get("tests_run", 0)
            tests_passed = result.get("tests_passed", 0)
            tests_failed = result.get("tests_failed", 0)
            print(
                f"[RUN {run}] tests_run={tests_run}, tests_passed={tests_passed}, "
                f"tests_failed={tests_failed}, success_rate={tests_passed/tests_run if tests_run else 0:.2%}"
            )
    finally:
        _SHUFFLE_EXPERIMENT_ACTIVE = False

    # ---------------------------------------------------------------------------
    # Aggregate and compute mean/std
    # ---------------------------------------------------------------------------
    passed_counts = [r.get("tests_passed", 0) for r in run_results]
    total_per_run = run_results[0].get("tests_run", 0) if run_results else 0

    try:
        import numpy as np
        passed_arr = np.array(passed_counts)
        mean_passed = float(np.mean(passed_arr))
        std_passed = float(np.std(passed_arr))
    except ImportError:
        import statistics
        mean_passed = statistics.mean(passed_counts)
        std_passed = statistics.stdev(passed_counts) if len(passed_counts) > 1 else 0.0

    # Per-test-case mean/std: for each test file, how many runs passed it
    test_files = []
    if run_results and run_results[0].get("results"):
        test_files = [
            r.get("test_file", "unknown") for r in run_results[0]["results"]
        ]

    per_test_passed = []
    if run_results and run_results[0].get("results"):
        test_count = len(run_results[0]["results"])
        for i in range(test_count):
            passed_this_test = sum(
                1
                for r in run_results
                if r.get("results")
                and i < len(r["results"])
                and r["results"][i].get("overall_success", False)
            )
            per_test_passed.append(passed_this_test)

    if per_test_passed:
        try:
            import numpy as np
            per_test_mean = float(np.mean(per_test_passed))
            per_test_std = float(np.std(per_test_passed))
        except ImportError:
            import statistics
            per_test_mean = statistics.mean(per_test_passed)
            per_test_std = statistics.stdev(per_test_passed) if len(per_test_passed) > 1 else 0.0
    else:
        per_test_mean = per_test_std = 0.0

    summary = {
        "experiment": "context_shuffle",
        "memory_backend": MEMORY_BACKEND,
        "defense": DEFENSE,
        "split": "test_100",
        "target_model": TARGET_MODEL,
        "num_runs": num_runs,
        "snapshot_path": str(SNAPSHOT_PATH),
        "tests_per_run": total_per_run,
        "aggregate": {
            "mean_tests_passed": mean_passed,
            "std_tests_passed": std_passed,
            "mean_pass_rate": mean_passed / total_per_run if total_per_run else 0,
            "std_pass_rate": std_passed / total_per_run if total_per_run else 0,
        },
        "per_run_tests_passed": passed_counts,
        "per_run_seeds": run_seeds_list,
        "per_test_mean_passes": per_test_mean,
        "per_test_std_passes": per_test_std,
        "per_test_passed_counts": per_test_passed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "shuffle_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Mean tests passed: {mean_passed:.2f}")
    print(f"Std tests passed:  {std_passed:.2f}")
    print(
        f"Mean pass rate:    {mean_passed/total_per_run if total_per_run else 0:.2%}"
    )
    print(
        f"Std pass rate:     {std_passed/total_per_run if total_per_run else 0:.2%}"
    )
    print(f"\nPer-run tests_passed: {passed_counts}")
    print(f"\nPer-test mean passes (across runs): {per_test_mean:.2f}")
    print(f"Per-test std passes: {per_test_std:.2f}")
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
