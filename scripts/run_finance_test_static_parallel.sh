#!/usr/bin/env bash
#
# Run STATIC finance attack_bench tests (no --adaptive)
# for all test* splits, for:
#   memory-backend: explicit, rag, context, mem0
#   defense-type:  none
#
# Parallelism is handled by run_benchmark.py via --num-workers.
#
# Usage:
#   chmod +x scripts/run_finance_test_static_parallel.sh
#   MODEL=gemini-3.1-pro-preview ./scripts/run_finance_test_static_parallel.sh
#
set -euo pipefail

# Repo root
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Model to benchmark (default can be overridden with MODEL=...)
MODEL="${MODEL:-gemini-3.1-pro-preview}"

# Number of workers for run_benchmark (per split)
NUM_WORKERS="${NUM_WORKERS:-8}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

# Finance topic only
TOPIC="finance"

# 4 memory backends to evaluate
BACKENDS=(explicit rag context mem0)
DEFENSE="none"

echo "=============================================="
echo "Static finance tests (attack_bench/test*/$TOPIC)"
echo "Model:        $MODEL"
echo "Backends:     ${BACKENDS[*]}"
echo "Defense:      $DEFENSE"
echo "Num workers:  $NUM_WORKERS (per run_benchmark call)"
echo "Repo root:    $REPO_ROOT"
echo "Attack bench: $ATTACK_BENCH"
echo "=============================================="
echo

# Loop over all test* splits (e.g., test, test_0, test_10, test_20, ...)
shopt -s nullglob
test_splits=("$ATTACK_BENCH"/test*)
shopt -u nullglob

if [[ ${#test_splits[@]} -eq 0 ]]; then
  echo "No test* splits found under $ATTACK_BENCH. Did you run propagate_train_attack_to_test_cases.py?"
  exit 0
fi

for split_dir in "${test_splits[@]}"; do
  split_name="$(basename "$split_dir")"
  finance_dir="$split_dir/$TOPIC"

  if [[ ! -d "$finance_dir" ]]; then
    echo "Skip $split_name: no $TOPIC directory at $finance_dir"
    continue
  fi

  echo "----------------------------------------------"
  echo "Running finance tests for split: $split_name"
  echo "Test path: $finance_dir"
  echo "----------------------------------------------"

  # Single run_benchmark call with all four backends and defense=none.
  # This uses run_benchmark's internal ProcessPoolExecutor with --num-workers.
  python "$RUN_BENCHMARK" \
    --memory-backend "${BACKENDS[@]}" \
    --defense-type "$DEFENSE" \
    --test "$finance_dir" \
    --model "$MODEL" \
    --num-workers "$NUM_WORKERS" \
    --try-all

  echo
done

echo "=============================================="
echo "All finance test splits finished."
echo "=============================================="

