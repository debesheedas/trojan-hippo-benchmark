#!/usr/bin/env bash
#
# Run adaptive benchmark (--adaptive) for FINANCE topic only, DEFENSE=NONE only,
# for all train splits train_0 through train_100, and all 4 memory backends
# (explicit, rag, context, mem0). Uses num_workers=8 for maximum parallelism.
#
# Test paths: data/benchmark/attack_bench/train_<N>/finance/<backend>/none
# Logs:       data/benchmark/attack_logs/<split>/<model>/finance/<backend>/none/
# Results:    data/benchmark/attack_results/<split>/<model>/finance/<backend>/none/
#
# For explicit backend only: --early-stop-patience 25 (other backends use default).
#
# Usage:
#   ./scripts/run_finance_adaptive_defense_none_all_trains.sh
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

# train_0 .. train_100 (101 splits)
declare -a TRAIN_SPLITS=()
for i in {0..100}; do
  TRAIN_SPLITS+=("train_$i")
done

# Build one command per (split, backend); only defense=none
declare -a COMMANDS=()
for split in "${TRAIN_SPLITS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    dir="$ATTACK_BENCH/$split/$TOPIC/$backend/$DEFENSE"
    if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
      cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$dir" --adaptive)
      if [[ "$backend" == "explicit" ]]; then
        cmd+=(--early-stop-patience 25)
      fi
      COMMANDS+=("${cmd[*]}")
    fi
  done
done

echo "=============================================="
echo "Finance adaptive benchmark (defense=none only)"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Topic:          $TOPIC"
echo "Defense:        $DEFENSE"
echo "Backends:       ${BACKENDS[*]}"
echo "Train splits:   train_0 .. train_100 (${#TRAIN_SPLITS[@]} splits)"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Num workers:    $NUM_WORKERS"
echo "Flags:          --adaptive, no --force; explicit: --early-stop-patience 25"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No (split, backend) directories with JSONs found under $ATTACK_BENCH/<split>/$TOPIC/<backend>/$DEFENSE"
  exit 0
fi

# Run in batches of NUM_WORKERS
i=0
while [[ $i -lt ${#COMMANDS[@]} ]]; do
  batch=0
  while [[ $batch -lt $NUM_WORKERS && $i -lt ${#COMMANDS[@]} ]]; do
    ( cd "$REPO_ROOT" && eval "${COMMANDS[$i]}" ) &
    ((i++)) || true
    ((batch++)) || true
  done
  wait
done

echo "=============================================="
echo "All jobs finished."
echo "=============================================="
