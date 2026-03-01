#!/usr/bin/env bash
#
# Full STEALTH pipeline for all topics, DEFENSE=NONE only.
#
# 1. Adaptive TRAIN runs (with --stealth) for all topics:
#      finance, health, identity, legal, tax
#    for all train splits train_0 .. train_100 and all 4 memory backends
#      (explicit, rag, context, mem0).
#    Uses the stealth-aware adaptive scoring and writes to:
#      attack_results_stealth/train_<N>/<model>/<topic>/<backend>/none/
#      attack_logs_stealth/train_<N>/<model>/<topic>/<backend>/none/
#      attack_bench_stealth/train_cache[_N]/...  (same names as normal, inside attack_bench_stealth)
#
# 2. Propagate best stealth attacks from train -> test:
#      scripts/propagate_train_attack_to_test_cases.py (no args: does normal + stealth)
#      Stealth reads cache from attack_bench_stealth/train_cache[_N], writes to attack_bench_stealth/test_N/...
#
# 3. STATIC TEST runs (with --stealth flag for mirrored outputs) for all topics:
#      finance, health, identity, legal, tax
#    for all test splits test_0 .. test_100 and all 4 memory backends
#      (explicit, rag, context, mem0).
#    Writes to:
#      attack_results_stealth/test_<N>/<model>/<topic>/<backend>/none/
#      attack_logs_stealth/test_<N>/<model>/<topic>/<backend>/none/
#
# Parallelism:
#   - NUM_WORKERS controls how many run_benchmark processes are run in parallel.
#   - Default NUM_WORKERS=24 (can be overridden via env).
#
# Usage:
#   ./scripts/run_all_topics_stealth_defense_none_full_pipeline.sh
#
# Optional env:
#   REPO_ROOT   - repo root (default: script dir/..)
#   NUM_WORKERS - parallel jobs (default: 24)
#   MODEL       - override model name used by propagate_*; if unset, read from agent_config.yaml
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NUM_WORKERS="${NUM_WORKERS:-24}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"
PROPAGATE="$REPO_ROOT/scripts/propagate_train_attack_to_test_cases.py"

TOPICS=(finance health identity legal tax)
DEFENSE="none"
BACKENDS=(explicit rag context mem0)

# Optional startup delay: sleep for 2 hours before starting phases.
sleep 7200

# train_0 .. train_100 (101 splits)
declare -a TRAIN_SPLITS=()
for i in {0..100}; do
  TRAIN_SPLITS+=("train_$i")
done

# test_0 .. test_100 (101 splits)
declare -a TEST_SPLITS=()
for i in {0..100}; do
  TEST_SPLITS+=("test_$i")
done

echo "====================================================================="
echo "ALL TOPICS STEALTH PIPELINE (DEFENSE=none, NUM_WORKERS=$NUM_WORKERS)"
echo "====================================================================="
echo "Repo root:  $REPO_ROOT"
echo "Attack bench: $ATTACK_BENCH"
echo "Topics:     ${TOPICS[*]}"
echo "Backends:   ${BACKENDS[*]}"
echo "Train splits: ${TRAIN_SPLITS[0]} .. ${TRAIN_SPLITS[${#TRAIN_SPLITS[@]}-1]}"
echo "Test splits:  ${TEST_SPLITS[0]} .. ${TEST_SPLITS[${#TEST_SPLITS[@]}-1]}"
echo

declare -a PIDS=()

_cleanup() {
  echo ""
  echo "Interrupted - killing all spawned jobs..."
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill -9 "${PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap _cleanup INT TERM

run_command_batch() {
  # Bash on macOS (3.2) doesn't support nameref (local -n), so we pass the
  # array name and expand it via eval into a local CMDS array.
  local arr_name="$1"
  eval "local CMDS=(\"\${${arr_name}[@]}\")"

  if [[ ${#CMDS[@]} -eq 0 ]]; then
    echo "No commands to run for this phase."
    return 0
  fi

  echo "Total jobs in this phase: ${#CMDS[@]}"
  echo "Running with NUM_WORKERS=$NUM_WORKERS"
  echo "---------------------------------------------------------------------"

  PIDS=()
  local i=0
  while [[ $i -lt ${#CMDS[@]} ]]; do
    local batch=0
    while [[ $batch -lt $NUM_WORKERS && $i -lt ${#CMDS[@]} ]]; do
      # Use exec so $! is the python PID, not the subshell
      ( cd "$REPO_ROOT" && eval "exec ${CMDS[$i]}" ) &
      PIDS+=($!)
      ((i++)) || true
      ((batch++)) || true
    done
    wait
  done

  echo "Phase completed: ${#CMDS[@]} jobs finished."
  echo "---------------------------------------------------------------------"
}

###############################################################################
# 1. Adaptive TRAIN runs with STEALTH
###############################################################################

declare -a TRAIN_COMMANDS=()

for split in "${TRAIN_SPLITS[@]}"; do
  for topic in "${TOPICS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      dir="$ATTACK_BENCH/$split/$topic/$backend/$DEFENSE"
      if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
        cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$dir" --adaptive --stealth)
        if [[ "$backend" == "explicit" ]]; then
          cmd+=(--early-stop-patience 25)
        fi
        TRAIN_COMMANDS+=("${cmd[*]}")
      fi
    done
  done
done

echo "====================================================================="
echo "PHASE 1: Adaptive TRAIN runs with STEALTH (all topics, defense=none)"
echo "====================================================================="
run_command_batch TRAIN_COMMANDS

###############################################################################
# 2. Propagate best stealth attacks from TRAIN → TEST
###############################################################################

echo
echo "====================================================================="
echo "PHASE 2: Propagate attacks from TRAIN → TEST (normal + stealth mirrors)"
echo "====================================================================="

# Let propagate_train_attack_to_test_cases.py infer the model from agent_config.yaml
# and propagate for BOTH:
#   - attack_bench  + attack_bench/train_cache[_N]         (stealth=False)
#   - attack_bench_stealth + attack_bench_stealth/train_cache[_N] (stealth=True)
( cd "$REPO_ROOT" && python "$PROPAGATE" )

echo "Propagation completed (normal + stealth)."

###############################################################################
# 3. STATIC TEST runs with STEALTH outputs (mirrored folders)
###############################################################################

declare -a TEST_COMMANDS=()

for split in "${TEST_SPLITS[@]}"; do
  for topic in "${TOPICS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      dir="$ATTACK_BENCH/$split/$topic/$backend/$DEFENSE"
      if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
        # Static benchmark (no --adaptive) but with --stealth so that results/logs
        # go to the mirrored *_stealth trees.
        cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$dir" --stealth)
        TEST_COMMANDS+=("${cmd[*]}")
      fi
    done
  done
done

echo
echo "====================================================================="
echo "PHASE 3: Static TEST runs with STEALTH outputs (all topics, defense=none)"
echo "====================================================================="
run_command_batch TEST_COMMANDS

echo
echo "====================================================================="
echo "ALL STEALTH PIPELINE PHASES COMPLETED."
echo "====================================================================="

