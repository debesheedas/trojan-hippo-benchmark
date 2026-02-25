#!/usr/bin/env bash
#
# Run only the TEST part of the benchmark for:
#   Backend: mem0 only
#   Defense: none only
#   All topics: finance, health, legal, identity, tax
#
# Test paths include backend and defense: attack_bench/<test_split>/<topic>/mem0/none
# All (split, topic) combinations are run in parallel.
#
# Environment variables you can override:
#   MODEL               (default: gemini-3.1-pro-preview)
#   MAX_PARALLEL_TESTS  (default: 0 = unlimited) - max concurrent test jobs
#   NUM_WORKERS_TEST    (default: 8) - num-workers per run_benchmark invocation
#
# Usage:
#   chmod +x scripts/run_mem0_none_full_pipeline.sh
#   MODEL=gemini-3.1-pro-preview ./scripts/run_mem0_none_full_pipeline.sh
#
set -euo pipefail

# Repo root
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

MODEL="${MODEL:-gemini-3.1-pro-preview}"
MAX_PARALLEL_TESTS="${MAX_PARALLEL_TESTS:-0}"
NUM_WORKERS_TEST="${NUM_WORKERS_TEST:-8}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

BACKEND="mem0"
DEFENSE="none"
TOPICS=(finance health legal identity tax)

# Discover test splits (test_0, test_10, test_20, ...)
shopt -s nullglob
TEST_SPLITS=("$ATTACK_BENCH"/test*)
shopt -u nullglob
# Keep only basenames
TEST_SPLITS=("${TEST_SPLITS[@]##*/}")

echo "=============================================="
echo "TEST-only pipeline: mem0 + defense=none"
echo "Topics:            ${TOPICS[*]}"
echo "Test splits:       ${TEST_SPLITS[*]}"
echo "Model:             $MODEL"
echo "Max parallel:      ${MAX_PARALLEL_TESTS:-unlimited}"
echo "Num-workers/job:   $NUM_WORKERS_TEST"
echo "Repo root:         $REPO_ROOT"
echo "=============================================="
echo

# Build list of test dirs: each path = attack_bench/<split>/<topic>/mem0/none
declare -a TEST_DIRS=()
for split in "${TEST_SPLITS[@]}"; do
  for topic in "${TOPICS[@]}"; do
    test_dir="$ATTACK_BENCH/$split/$topic/$BACKEND/$DEFENSE"
    if [[ -d "$test_dir" ]] && [[ -n "$(find "$test_dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
      TEST_DIRS+=("$test_dir")
    fi
  done
done

echo "Found ${#TEST_DIRS[@]} test directory(ies) to run (path includes backend and defense)."
if [[ ${#TEST_DIRS[@]} -eq 0 ]]; then
  echo "No test dirs found under $ATTACK_BENCH/<split>/<topic>/$BACKEND/$DEFENSE – exiting."
  exit 1
fi
echo

# Run all test dirs in parallel (single backend+defense per run, so --test <path> has backend/defense in path)
if [[ "${MAX_PARALLEL_TESTS:-0}" -gt 0 ]]; then
  running=0
  for test_path in "${TEST_DIRS[@]}"; do
    while [[ $running -ge $MAX_PARALLEL_TESTS ]]; do
      wait -n 2>/dev/null || wait
      ((running--)) || true
    done
    echo "Starting test: $test_path"
    ( cd "$REPO_ROOT" && python "$RUN_BENCHMARK" \
        --memory-backend "$BACKEND" \
        --defense-type "$DEFENSE" \
        --test "$test_path" \
        --model "$MODEL" \
        --num-workers "$NUM_WORKERS_TEST" \
        --try-all \
        --force ) &
    ((running++)) || true
  done
  wait
else
  for test_path in "${TEST_DIRS[@]}"; do
    echo "Starting test: $test_path"
    ( cd "$REPO_ROOT" && python "$RUN_BENCHMARK" \
        --memory-backend "$BACKEND" \
        --defense-type "$DEFENSE" \
        --test "$test_path" \
        --model "$MODEL" \
        --num-workers "$NUM_WORKERS_TEST" \
        --try-all \
        --force ) &
  done
  wait
fi

echo "=============================================="
echo "All test runs completed."
echo "=============================================="
