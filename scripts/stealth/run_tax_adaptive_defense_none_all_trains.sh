#!/usr/bin/env bash
#
# STEALTH: Run adaptive benchmark (--adaptive --stealth) for TAX topic only,
# DEFENSE=NONE only, for all train splits train_0 through train_100, and all 4
# memory backends (explicit, rag, context, mem0). No --force (resume-friendly).
#
# Test paths: data/benchmark/attack_bench/train_<N>/tax/<backend>/none
# Logs:       data/benchmark/attack_logs_stealth/<split>/<model>/tax/<backend>/none/
# Results:    data/benchmark/attack_results_stealth/<split>/<model>/tax/<backend>/none/
#
# For explicit backend only: --early-stop-patience 25.
#
# Usage:
#   ./scripts/stealth/run_tax_adaptive_defense_none_all_trains.sh
#
# Optional env:
#   REPO_ROOT   - repo root (default: script dir/../..)
#   NUM_WORKERS - parallel jobs (default: 8)
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NUM_WORKERS="${NUM_WORKERS:-8}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

TOPIC="tax"
DEFENSE="none"
BACKENDS=(explicit rag context mem0)

# train_0 .. train_100 (101 splits)
declare -a TRAIN_SPLITS=()
for i in {0..100}; do
  TRAIN_SPLITS+=("train_$i")
done

# Build one command per (split, backend); --adaptive --stealth, no --force
declare -a COMMANDS=()
for split in "${TRAIN_SPLITS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    dir="$ATTACK_BENCH/$split/$TOPIC/$backend/$DEFENSE"
    if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
      cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$dir" --adaptive --stealth)
      if [[ "$backend" == "explicit" ]]; then
        cmd+=(--early-stop-patience 25)
      fi
      COMMANDS+=("${cmd[*]}")
    fi
  done
done

echo "=============================================="
echo "Tax STEALTH adaptive train (defense=none)"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Topic:          $TOPIC"
echo "Defense:        $DEFENSE"
echo "Backends:       ${BACKENDS[*]}"
echo "Train splits:   train_0 .. train_100 (${#TRAIN_SPLITS[@]} splits)"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Num workers:    $NUM_WORKERS"
echo "Flags:          --adaptive, --stealth (no --force)"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No (split, backend) directories with JSONs found under $ATTACK_BENCH/<split>/$TOPIC/<backend>/$DEFENSE"
  exit 0
fi

declare -a PIDS=()
_cleanup() {
  echo ""
  echo "Interrupted - killing all spawned jobs..."
  kill -9 "${PIDS[@]}" 2>/dev/null || true
  exit 130
}
trap _cleanup INT TERM

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
