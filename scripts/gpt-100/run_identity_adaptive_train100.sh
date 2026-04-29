#!/usr/bin/env bash
#
# Run adaptive benchmark (--adaptive) for IDENTITY topic, train_100 only,
# all 4 memory backends (rag, mem0, context, explicit), defense=none.
# explicit and context: --early-stop-patience 20. Model: gpt-5-mini.
# Parallelism: 4 workers (one per backend).
#
# Usage:
#   ./scripts/gpt-100/run_identity_adaptive_train100.sh
#
# Optional env:
#   REPO_ROOT   - repo root (default: script dir/../..)
#   NUM_WORKERS - parallel jobs (default: 4)
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NUM_WORKERS="${NUM_WORKERS:-4}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

TOPIC="identity"
SPLIT="train_100"
DEFENSE="none"
BACKENDS=(rag mem0 context explicit)

# Build one command per backend for train_100
declare -a COMMANDS=()
for backend in "${BACKENDS[@]}"; do
  dir="$ATTACK_BENCH/$SPLIT/$TOPIC/$backend/$DEFENSE"
  if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
    cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$dir" --model gpt-5-mini --adaptive)
    if [[ "$backend" == "explicit" ]] || [[ "$backend" == "context" ]]; then
      cmd+=(--early-stop-patience 20)
    fi
    COMMANDS+=("${cmd[*]}")
  fi
done

echo "=============================================="
echo "Identity adaptive (train_100, defense=none)"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Topic:          $TOPIC"
echo "Split:          $SPLIT"
echo "Defense:        $DEFENSE"
echo "Backends:       ${BACKENDS[*]}"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Num workers:    $NUM_WORKERS"
echo "Flags:          --adaptive; explicit/context: --early-stop-patience 20"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No backend directories with JSONs found under $ATTACK_BENCH/$SPLIT/$TOPIC/<backend>/$DEFENSE"
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
