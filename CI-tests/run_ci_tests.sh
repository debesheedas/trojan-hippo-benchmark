#!/usr/bin/env bash
# CI tests: benchmark + compare against ground truth.
# Usage: from repo root, run:  bash CI-tests/run_ci_tests.sh
# Optional: pass --skip-benchmark to only run comparison (use existing results).
#
# FIXED MODEL: CI and ground truth use gpt-5-mini only. Do not change the model
# without updating run_benchmark, compare_results, update_ground_truth, and
# the workflow to the same model; gold files are tied to this model.

set -e
cd "$(dirname "$0")/.."
SKIP_BENCHMARK=""
for arg in "$@"; do
  if [ "$arg" = "--skip-benchmark" ]; then
    SKIP_BENCHMARK=1
    break
  fi
done

RESULTS_DIR="CI-tests/results"
TESTCASES_DIR="CI-tests/testcases"
# CI and ground truth are fixed to this model (see CI-tests/README.md).
MODEL="gpt-5-mini"

if [ -z "$SKIP_BENCHMARK" ]; then
  echo "Running benchmark with model=$MODEL, results_dir=$RESULTS_DIR ..."
  python scripts/run_benchmark.py \
    --test "$TESTCASES_DIR" \
    --results-dir "$RESULTS_DIR" \
    --model "$MODEL" \
    --force \
    --num-workers 1
fi

echo ""
echo "Comparing results against ground truth (model=$MODEL) ..."
FAILED=0
for test in test1 test2 test3; do
  if [ -f "$TESTCASES_DIR/${test}.json" ]; then
    if ! python CI-tests/compare_results.py \
      --test-file "$TESTCASES_DIR/${test}.json" \
      --results-dir "$RESULTS_DIR"; then
      FAILED=1
    fi
    echo ""
  fi
done
exit $FAILED
