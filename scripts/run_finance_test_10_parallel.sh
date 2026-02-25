#!/usr/bin/env bash
#
# Run run_benchmark.py in parallel on all test_10/finance combinations.
# Each (memory_backend, defense_type) directory is run with the correct
# --memory-backend and --defense-type. Uses up to MAX_PARALLEL jobs at a time.
#
# Usage:
#   ./scripts/run_finance_test_10_parallel.sh [MAX_PARALLEL]
#   MAX_PARALLEL=16 ./scripts/run_finance_test_10_parallel.sh
#
set -euo pipefail

# Repo root (directory containing scripts/ and data/)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FINANCE_DIR="$REPO_ROOT/data/benchmark/attack_bench/test_10/finance"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

# How many runs at once (default 8)
MAX_PARALLEL="${1:-${MAX_PARALLEL:-8}}"

# Memory backends and defense types (must match folder names under finance/)
BACKENDS=(context explicit mem0 none rag)
DEFENSES=(none user_prompt_only no_untrusted_tools limit_memory_length provable_policy)

# Build list of commands: one per (backend, defense) directory that has JSONs
declare -a COMMANDS
for backend in "${BACKENDS[@]}"; do
  for defense in "${DEFENSES[@]}"; do
    dir="$FINANCE_DIR/$backend/$defense"
    if [[ -d "$dir" ]]; then
      if [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
        cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$defense" --test "$dir")
        COMMANDS+=("${cmd[*]}")
      fi
    fi
  done
done

echo "=============================================="
echo "Parallel finance test_10 benchmark"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Finance dir:    $FINANCE_DIR"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Max parallel:   $MAX_PARALLEL"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No (backend, defense) directories with JSON files found under $FINANCE_DIR"
  exit 0
fi

# Run jobs in parallel: batches of up to MAX_PARALLEL (portable, no wait -n)
i=0
while [[ $i -lt ${#COMMANDS[@]} ]]; do
  batch=0
  while [[ $batch -lt $MAX_PARALLEL && $i -lt ${#COMMANDS[@]} ]]; do
    ( cd "$REPO_ROOT" && eval "${COMMANDS[$i]}" ) &
    ((i++)) || true
    ((batch++)) || true
  done
  wait
done

echo "=============================================="
echo "All jobs finished."
echo "=============================================="
