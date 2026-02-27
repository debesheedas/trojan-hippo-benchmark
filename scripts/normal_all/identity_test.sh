#!/usr/bin/env bash
#
# Run static benchmark (no --adaptive) for IDENTITY, all defenses: (none,none) +
# (all 5 backends × 4 defenses). No --stealth, no --force. 16 workers.
#
# Usage: ./scripts/normal_all/identity_test.sh
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NUM_WORKERS="${NUM_WORKERS:-16}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

TOPIC="identity"
BACKENDS=(none explicit rag mem0 context)
DEFENSES=(user_prompt_only limit_memory_length no_untrusted_tools provable_policy)

declare -a TEST_SPLITS=()
for i in {0..100}; do
  TEST_SPLITS+=("test_$i")
done

declare -a COMMANDS=()
for split in "${TEST_SPLITS[@]}"; do
  dir="$ATTACK_BENCH/$split/$TOPIC/none/none"
  if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
    cmd=(python "$RUN_BENCHMARK" --memory-backend none --defense-type none --test "$dir")
    COMMANDS+=("${cmd[*]}")
  fi
  for backend in "${BACKENDS[@]}"; do
    for defense in "${DEFENSES[@]}"; do
      dir="$ATTACK_BENCH/$split/$TOPIC/$backend/$defense"
      if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
        cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$defense" --test "$dir")
        if [[ "$backend" == "explicit" ]]; then
          cmd+=(--early-stop-patience 25)
        fi
        COMMANDS+=("${cmd[*]}")
      fi
    done
  done
done

echo "=============================================="
echo "Identity static test (all defenses, no stealth/force)"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Topic:          $TOPIC"
echo "Backends:       ${BACKENDS[*]}"
echo "Defenses:       none + ${DEFENSES[*]}"
echo "Test splits:    test_0 .. test_100 (${#TEST_SPLITS[@]} splits)"
echo "Total jobs:     ${#COMMANDS[@]}"
echo "Num workers:    $NUM_WORKERS"
echo "=============================================="

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No (split, backend, defense) dirs with JSONs found."
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
