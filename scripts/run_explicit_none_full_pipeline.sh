#!/usr/bin/env bash
#
# Run the TEST part of the benchmark for:
#   Backend: explicit only
#   Defense: none only
#   Split:   test_20 only (override with TEST_SPLIT)
#
# A single run_benchmark invocation is used so that --num-workers applies to all
# tests in that split (test_20/<topic>/explicit/none for all topics). run_benchmark
# discovers all JSONs under the split and filters to explicit/none.
#
# Environment variables you can override:
#   MODEL            (default: gemini-3.1-pro-preview)
#   TEST_SPLIT       (default: test_20) - which test split to run
#   NUM_WORKERS      (default: 8) - parallelism for running tests
#
# Usage:
#   chmod +x scripts/run_explicit_none_full_pipeline.sh
#   ./scripts/run_explicit_none_full_pipeline.sh
#   TEST_SPLIT=test_10 NUM_WORKERS=4 ./scripts/run_explicit_none_full_pipeline.sh
#
set -euo pipefail

# Repo root
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

MODEL="${MODEL:-gemini-3.1-pro-preview}"
TEST_SPLIT="${TEST_SPLIT:-test_20}"
NUM_WORKERS="${NUM_WORKERS:-8}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

BACKEND="explicit"
DEFENSE="none"

# Single test path: attack_bench/test_20 (run_benchmark will discover all topic/explicit/none under it)
TEST_PATH="$ATTACK_BENCH/$TEST_SPLIT"

if [[ ! -d "$TEST_PATH" ]]; then
  echo "Error: Test split directory not found: $TEST_PATH"
  exit 1
fi

echo "=============================================="
echo "TEST pipeline: explicit + defense=none"
echo "Test split:       $TEST_SPLIT"
echo "Test path:        $TEST_PATH"
echo "Model:            $MODEL"
echo "Num workers:      $NUM_WORKERS"
echo "Repo root:        $REPO_ROOT"
echo "=============================================="
echo

cd "$REPO_ROOT"
python "$RUN_BENCHMARK" \
  --memory-backend "$BACKEND" \
  --defense-type "$DEFENSE" \
  --test "$TEST_PATH" \
  --model "$MODEL" \
  --num-workers "$NUM_WORKERS" \
  --try-all \
  --force

echo "=============================================="
echo "Test run completed."
echo "=============================================="

