#!/usr/bin/env python3
"""
Update CI-tests ground truth files from actual benchmark results.

Ground truth is tied to a single model (see CI_MODEL). Run the benchmark
with that model first, then run this script to refresh ground truth.

  python scripts/run_benchmark.py --test CI-tests/testcases --results-dir CI-tests/results --model gpt-5-mini --force
  python CI-tests/update_ground_truth_from_results.py
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.benchmark_utils import (
    get_result_path,
    determine_attack_type,
    MEMORY_BACKENDS,
    UNIFIED_DEFENSE_TYPES,
    is_valid_combination,
)

RESULTS_DIR = BASE_DIR / "CI-tests" / "results"
GROUND_TRUTH_DIR = BASE_DIR / "CI-tests" / "ground_truth"
TESTCASES_DIR = BASE_DIR / "CI-tests" / "testcases"
# CI and ground truth are fixed to this model (see CI-tests/README.md).
CI_MODEL = "gpt-5-mini"
MODEL = CI_MODEL


def build_ground_truth_for_test(test_file: Path) -> dict:
    """Build ground truth dict from result files for one test."""
    with open(test_file, "r", encoding="utf-8") as f:
        test_def = json.load(f)
    attack_type = determine_attack_type(test_file, test_def)

    out = {}
    for backend in MEMORY_BACKENDS:
        out[backend] = {}
        for defense in UNIFIED_DEFENSE_TYPES:
            if not is_valid_combination(backend, defense):
                continue
            result_path = get_result_path(
                memory_backend=backend,
                unified_defense=defense,
                model_name=MODEL,
                attack_type=attack_type,
                test_file=test_file,
                results_base_dir=RESULTS_DIR,
            )
            if result_path.exists():
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                out[backend][defense] = result.get("overall_success", False)
            else:
                out[backend][defense] = False
    return out


def main():
    for test_file in sorted(TESTCASES_DIR.glob("*.json")):
        name = test_file.stem
        gt = build_ground_truth_for_test(test_file)
        out_path = GROUND_TRUTH_DIR / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=None)
        print(f"Updated {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
