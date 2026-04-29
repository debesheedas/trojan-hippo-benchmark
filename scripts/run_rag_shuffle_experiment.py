"""
RAG Memory Shuffle Experiment

Runs the attack benchmark for rag/none on test_100 with randomly shuffled
rag_documents (vector chunks) at load_memory_snapshot (session_100). This experiment
tests whether document/chunk order in the RAG vectorstore affects attack success rate.

Runs 20 times with different shuffles + random seeds, then reports mean and std.
Does NOT modify run_benchmark or the normal load_memory_snapshot flow - uses
a monkey-patch that is only active when this script runs.

How shuffling works:
  - When load_memory_snapshot runs for session_index=100, backend=rag, defense=none,
    the patch intercepts the snapshot data before the real merge.
  - It takes data["rag_documents"] (list of document strings, one per vector chunk),
    deep-copies it, and random.shuffle()'s the copy. The merged data passed to the
    original load_snapshot_rag_merge uses this shuffled list.
  - The original then does: doc_strings = [str(x) for x in payload if x] and
    manager.add_memories_batch(doc_strings, ...). So the ORDER in which the same
    ~100 chunks are inserted into the vectorstore is randomized each run. Retrieval
    is still by similarity; only insertion order (and any order-sensitive behavior
    in the backend) changes. Each run also uses a different random seed for the
    benchmark/agent, so both chunk order and model randomness vary across runs.

Usage:
  # Test with single file, 3 runs (verify shuffling works)
  python scripts/run_rag_shuffle_experiment.py --test-file data/benchmark/attack_bench/test_100/finance/rag/none/01.json --num-runs 3

  # Full experiment, sequential (20 runs)
  python scripts/run_rag_shuffle_experiment.py --num-runs 20

  # Full experiment, parallel (20 runs in parallel)
  python scripts/run_rag_shuffle_experiment.py --num-runs 20 --parallel
"""

import argparse
import copy
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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

_RAG_SHUFFLE_EXPERIMENT_ACTIVE = False
_RAG_SHUFFLE_RUN_COUNT = 0
_ORIGINAL_load_snapshot_rag_merge = _snapshot_io.load_snapshot_rag_merge


def _patched_load_snapshot_rag_merge(
    in_memory_env,
    data,
    current_memory_backend: str,
    current_defense_type: str,
    current_snapshot_set_id: str,
    current_session_index: int,
):
    """Apply shuffle to rag_documents when session_index=100, rag, none."""
    global _RAG_SHUFFLE_RUN_COUNT
    if (
        _RAG_SHUFFLE_EXPERIMENT_ACTIVE
        and current_memory_backend == "rag"
        and current_defense_type == "none"
        and current_session_index == 100
    ):
        payload = data.get("rag_documents", [])
        if payload and isinstance(payload, list):
            shuffled = copy.deepcopy(payload)
            random.shuffle(shuffled)
            data = {**data, "rag_documents": shuffled}
            # Debug: log that shuffle was applied
            first_orig = str(payload[0])[:80] if payload else ""
            first_new = str(shuffled[0])[:80] if shuffled else ""
            print(
                f"[RAG_SHUFFLE_EXPERIMENT] Run {_RAG_SHUFFLE_RUN_COUNT}: Shuffled {len(shuffled)} rag_documents (vector chunks). "
                f"Original first 80 chars: {first_orig!r}... -> New first: {first_new!r}..."
            )
    return _ORIGINAL_load_snapshot_rag_merge(
        in_memory_env,
        data,
        current_memory_backend,
        current_defense_type,
        current_snapshot_set_id,
        current_session_index,
    )


_snapshot_io.load_snapshot_rag_merge = _patched_load_snapshot_rag_merge

# Now safe to import run_benchmark
sys.path.insert(0, str(BASE_DIR / "scripts"))
import run_benchmark as _run_benchmark_module
run_benchmark = _run_benchmark_module.run_benchmark


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
MEMORY_BACKEND = "rag"
DEFENSE = "none"
TEST_PATH = "data/benchmark/attack_bench/test_100"
TARGET_MODEL = "gemini-3.1-pro-preview"
NUM_RUNS = 20
SNAPSHOT_PATH = Path(
    "data/benchmark/snapshots/memory_snapshots_test/rag/none/session_100.json"
)
OUTPUT_DIR = Path("data/benchmark/rag_shuffle_experiment")


def _run_single_run(
    run_idx: int,
    seed: int,
    test_path: str,
    results_base: Path,
    logs_base: Path,
) -> Dict[str, Any]:
    """
    Run a single benchmark run for RAG shuffle experiment.
    Used by parallel workers - each must set globals and run.
    """
    global _RAG_SHUFFLE_EXPERIMENT_ACTIVE, _RAG_SHUFFLE_RUN_COUNT
    _RAG_SHUFFLE_EXPERIMENT_ACTIVE = True
    _RAG_SHUFFLE_RUN_COUNT = run_idx
    try:
        result = run_benchmark(
            memory_backend=MEMORY_BACKEND,
            unified_defense=DEFENSE,
            test_path=test_path,
            force=True,
            adaptive=False,
            results_base_dir=results_base,
            logs_base_dir=logs_base,
            target_model_name=TARGET_MODEL,
            seed_override=seed,
        )
        result["_run_idx"] = run_idx
        result["_seed"] = seed
        return result
    finally:
        _RAG_SHUFFLE_EXPERIMENT_ACTIVE = False


def _run_worker(args: Tuple[int, int, str, str, str]) -> Dict[str, Any]:
    """Worker for ProcessPoolExecutor - unpacks args and runs."""
    run_idx, seed, test_path, results_base, logs_base = args
    # Ensure worker runs from project root (paths are relative)
    os.chdir(BASE_DIR)
    return _run_single_run(run_idx, seed, test_path, Path(results_base), Path(logs_base))


def main():
    global _RAG_SHUFFLE_EXPERIMENT_ACTIVE, _RAG_SHUFFLE_RUN_COUNT

    parser = argparse.ArgumentParser(
        description="RAG memory shuffle experiment: shuffle vector chunks at load_memory_snapshot."
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
        help="Single test file path. If set, run only this test (for verify shuffle).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all iterations in parallel (faster for full experiment)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of parallel workers when --parallel (default: 20)",
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
    print("RAG MEMORY SHUFFLE EXPERIMENT")
    print("=" * 80)
    print(f"Memory backend: {MEMORY_BACKEND}, Defense: {DEFENSE}")
    print(f"Test path: {test_path}")
    print(f"Target model: {TARGET_MODEL}")
    print(f"Number of runs: {num_runs}")
    print(f"Parallel: {args.parallel}")
    if args.parallel:
        print(f"Workers: {args.num_workers}")
    print(f"Snapshot: {SNAPSHOT_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)

    if not SNAPSHOT_PATH.exists():
        print(f"ERROR: Snapshot not found: {SNAPSHOT_PATH}")
        sys.exit(1)

    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snapshot_data = json.load(f)
    rag_docs = snapshot_data.get("rag_documents", [])
    print(f"[DEBUG] Snapshot has {len(rag_docs)} rag_documents (vector chunks)")
    if rag_docs:
        print(f"[DEBUG] First chunk preview: {str(rag_docs[0])[:100]}...")
    print()

    if args.dry_run:
        print("[DRY-RUN] Patch verified. Snapshot loaded. Exiting.")
        print("[DRY-RUN] Run without --dry-run to execute the experiment.")
        return

    results_base = OUTPUT_DIR
    logs_base = OUTPUT_DIR / "logs"
    run_results: List[Dict[str, Any]] = []
    run_seeds_list: List[int] = []

    # Generate seeds for all runs
    seeds = [random.randint(1, 2**31 - 1) for _ in range(num_runs)]

    if args.parallel:
        # Parallel execution: each worker runs one full benchmark
        # Use separate results/logs per run to avoid overwrites
        results_base_per_run = OUTPUT_DIR / "runs"
        logs_base_per_run = OUTPUT_DIR / "logs"
        print(f"\nRunning {num_runs} iterations in parallel with {args.num_workers} workers...\n")

        # Use separate results/logs dir per run to avoid overwrites when running in parallel
        run_args = []
        for run_idx in range(1, num_runs + 1):
            run_results_dir = results_base / f"run_{run_idx}"
            run_logs_dir = logs_base / f"run_{run_idx}"
            run_args.append((run_idx, seeds[run_idx - 1], test_path, str(run_results_dir), str(run_logs_dir)))

        with ProcessPoolExecutor(max_workers=min(args.num_workers, num_runs)) as executor:
            futures = {executor.submit(_run_worker, a): a[0] for a in run_args}
            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    result = future.result()
                    run_results.append(result)
                    run_seeds_list.append(result.get("_seed", 0))
                    tests_passed = result.get("tests_passed", 0)
                    tests_run = result.get("tests_run", 0)
                    print(f"[RUN {run_idx}] DONE: tests_passed={tests_passed}/{tests_run}")
                except Exception as e:
                    print(f"[RUN {run_idx}] FAILED: {e}")
                    run_results.append({
                        "tests_run": 0,
                        "tests_passed": 0,
                        "tests_failed": 0,
                        "_run_idx": run_idx,
                        "_seed": seeds[run_idx - 1],
                    })
                    run_seeds_list.append(seeds[run_idx - 1])

        # Sort by run_idx for consistent ordering
        run_results.sort(key=lambda r: r.get("_run_idx", 0))
        run_seeds_list = [r.get("_seed", 0) for r in run_results]
    else:
        # Sequential execution
        _RAG_SHUFFLE_EXPERIMENT_ACTIVE = True
        try:
            for run in range(1, num_runs + 1):
                _RAG_SHUFFLE_RUN_COUNT = run
                run_seed = seeds[run - 1]
                print(f"\n{'#'*80}")
                print(f"RUN {run}/{num_runs} (seed={run_seed})")
                print(f"{'#'*80}\n")

                result = _run_single_run(
                    run_idx=run,
                    seed=run_seed,
                    test_path=test_path,
                    results_base=results_base,
                    logs_base=logs_base,
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
            _RAG_SHUFFLE_EXPERIMENT_ACTIVE = False

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

    per_test_passed = []
    per_test_names: List[str] = []
    if run_results and run_results[0].get("results"):
        test_count = len(run_results[0]["results"])
        # Names in same order as per_test_passed (for mapping summary to test files)
        per_test_names = [
            run_results[0]["results"][i].get("test_file", f"test_{i}")
            for i in range(test_count)
        ]
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
        "experiment": "rag_shuffle",
        "memory_backend": MEMORY_BACKEND,
        "defense": DEFENSE,
        "split": "test_100",
        "target_model": TARGET_MODEL,
        "num_runs": num_runs,
        "parallel": args.parallel,
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
        "per_test_names": per_test_names,
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
