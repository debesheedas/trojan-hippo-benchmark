#!/usr/bin/env bash
#
# Run adaptive benchmark (--adaptive) on all <split>/finance (backend, defense)
# combinations with num_workers=4: at most 4 run_benchmark.py processes at a time.
# Does not use --force (existing results are skipped).
#
# Usage:
#   ./scripts/run_train_0_finance_adaptive_parallel.sh           # default split: train_0
#   ./scripts/run_train_0_finance_adaptive_parallel.sh train_20  # run on train_20
#
set -euo pipefail

# Repo root
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Train/test split (e.g., train_0, train_10, train_20). Default: train_0
SPLIT="${1:-train_0}"

TRAIN_FINANCE_DIR="$REPO_ROOT/data/benchmark/attack_bench/${SPLIT}/finance"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

# At most 4 jobs at a time (num_workers=4)
MAX_PARALLEL=4

# Same structure as <split>/finance: <backend>/<defense>/
BACKENDS=(context explicit mem0 none rag)
DEFENSES=(none user_prompt_only no_untrusted_tools limit_memory_length provable_policy)

# Build list of commands: one per (backend, defense) directory that has JSONs
# Each run: --adaptive, no --force. For explicit memory only: --early-stop-patience 20 (others use default 5).
declare -a COMMANDS
for backend in "${BACKENDS[@]}"; do
  for defense in "${DEFENSES[@]}"; do
    dir="$TRAIN_FINANCE_DIR/$backend/$defense"
    if [[ -d "$dir" ]]; then
      if [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
        cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$defense" --test "$dir" --adaptive)
        if [[ "$backend" == "explicit" ]]; then
          cmd+=(--early-stop-patience 20)
        fi
        COMMANDS+=("${cmd[*]}")
      fi
    fi
  done
done

echo "=============================================="
echo "Adaptive benchmark: ${SPLIT}/finance (num_workers=4)"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Train finance:  $TRAIN_FINANCE_DIR"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Max parallel:  $MAX_PARALLEL (4 at a time)"
echo "Flags:          --adaptive, no --force; explicit backend: --early-stop-patience 20"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No (backend, defense) directories with JSON files found under $TRAIN_FINANCE_DIR"
  exit 0
fi

# Run in batches of up to MAX_PARALLEL (portable: no wait -n)
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
