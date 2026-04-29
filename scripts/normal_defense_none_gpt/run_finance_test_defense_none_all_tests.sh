#!/usr/bin/env bash
#
# Run STATIC benchmark (no --adaptive) for FINANCE topic only, DEFENSE=NONE only,
# for all test splits test_0 through test_100, and all 4 memory backends
# (explicit, rag, context, mem0). Uses num_workers=8 for maximum parallelism.
#
# Test paths: data/benchmark/attack_bench/test_<N>/finance/<backend>/none
# Logs:       data/benchmark/attack_logs/<split>/<model>/finance/<backend>/none/
# Results:    data/benchmark/attack_results/<split>/<model>/finance/<backend>/none/
#
# No --adaptive, --force.
#
# Usage:
#   ./scripts/run_finance_test_defense_none_all_tests.sh
#
# Optional env:
#   REPO_ROOT   - repo root (default: script dir/..)
#   NUM_WORKERS - parallel jobs (default: 8)
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NUM_WORKERS="${NUM_WORKERS:-8}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

TOPIC="finance"
DEFENSE="none"
BACKENDS=(explicit rag context mem0)

# test_0 .. test_100 (101 splits)
declare -a TEST_SPLITS=()
for i in {0..100}; do
  TEST_SPLITS+=("test_$i")
done

# Build one command per (split, backend); only defense=none; no --adaptive, --force
declare -a COMMANDS=()
for split in "${TEST_SPLITS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    dir="$ATTACK_BENCH/$split/$TOPIC/$backend/$DEFENSE"
    if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
      cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$dir" --model gpt-5-mini --force)
      COMMANDS+=("${cmd[*]}")
    fi
  done
done

echo "=============================================="
echo "Finance static test benchmark (defense=none only)"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Topic:          $TOPIC"
echo "Defense:        $DEFENSE"
echo "Backends:       ${BACKENDS[*]}"
echo "Test splits:    test_0 .. test_100 (${#TEST_SPLITS[@]} splits)"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Num workers:    $NUM_WORKERS"
echo "Flags:          no --adaptive, --force"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No (split, backend) directories with JSONs found under $ATTACK_BENCH/<split>/$TOPIC/<backend>/$DEFENSE"
  exit 0
fi

# On Ctrl+C (SIGINT) or SIGTERM: kill all spawned run_benchmark processes and exit
declare -a PIDS=()
_cleanup() {
  echo ""
  echo "Interrupted - killing all spawned jobs..."
  kill -9 "${PIDS[@]}" 2>/dev/null || true
  exit 130
}
trap _cleanup INT TERM

# Run in batches of NUM_WORKERS (use exec so $! is the python PID, not the subshell)
i=0
while [[ $i -lt ${#COMMANDS[@]} ]]; do
  batch=0
  while [[ $batch -lt $NUM_WORKERS && $i -lt ${#COMMANDS[@]} ]]; do
    ( cd "$REPO_ROOT" && eval "exec ${COMMANDS[$i]}" ) &
    PIDS+=($!)
    ((i++)) || true
    ((batch++)) || true
  done
  wait
done

echo "=============================================="
echo "All jobs finished."
echo "=============================================="
