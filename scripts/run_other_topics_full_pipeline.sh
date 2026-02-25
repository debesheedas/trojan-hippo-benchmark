#!/usr/bin/env bash
#
# Full pipeline for non-finance topics:
#   Topics: health, legal, identity, tax
#   Defense: none only
#   Backends: explicit, mem0, rag, context
#
# For each topic (one by one) this script will:
#   1) Run ADAPTIVE TRAINING on train_0, train_10, train_20
#      for all 4 backends (defense=none), in parallel (up to MAX_PARALLEL_TRAIN).
#   2) Run propagate_train_attack_to_test_cases.py to push train attacks into test cases.
#   3) Run STATIC TESTS on all attack_bench/test* splits for that topic,
#      with defense=none, all 4 backends at once, --try-all, and --num-workers=NUM_WORKERS_TEST.
#
# Environment variables you can override:
#   MODEL              (default: gemini-3.1-pro-preview)
#   MAX_PARALLEL_TRAIN (default: 16)  - max concurrent adaptive train jobs
#   NUM_WORKERS_TEST   (default: 16)  - num-workers passed to run_benchmark.py for tests
#
# Usage:
#   chmod +x scripts/run_other_topics_full_pipeline.sh
#   MODEL=gemini-3.1-pro-preview ./scripts/run_other_topics_full_pipeline.sh
#
set -euo pipefail

# Repo root
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

MODEL="${MODEL:-gemini-3.1-pro-preview}"
MAX_PARALLEL_TRAIN="${MAX_PARALLEL_TRAIN:-16}"
NUM_WORKERS_TEST="${NUM_WORKERS_TEST:-16}"

ATTACK_BENCH="$REPO_ROOT/data/benchmark/attack_bench"
RUN_BENCHMARK="$REPO_ROOT/scripts/run_benchmark.py"

TOPICS=(health legal identity tax)
TRAIN_SPLITS=(train_0 train_10 train_20)
BACKENDS=(explicit mem0 rag context)
DEFENSE="none"

echo "=============================================="
echo "Full pipeline for topics: ${TOPICS[*]}"
echo "Model:              $MODEL"
echo "Backends:           ${BACKENDS[*]}"
echo "Defense:            $DEFENSE"
echo "Train splits:       ${TRAIN_SPLITS[*]}"
echo "Max parallel train: $MAX_PARALLEL_TRAIN"
echo "Test num-workers:   $NUM_WORKERS_TEST"
echo "Repo root:          $REPO_ROOT"
echo "Attack bench:       $ATTACK_BENCH"
echo "=============================================="
echo

for topic in "${TOPICS[@]}"; do
  echo "##############################################"
  echo "TOPIC: $topic"
  echo "##############################################"
  echo

  # -----------------------------
  # 1) ADAPTIVE TRAINING
  # -----------------------------
  echo ">>> ADAPTIVE TRAINING for topic: $topic"

  declare -a TRAIN_COMMANDS=()
  for split in "${TRAIN_SPLITS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      train_dir="$ATTACK_BENCH/$split/$topic/$backend/$DEFENSE"
      if [[ -d "$train_dir" ]] && [[ -n "$(find "$train_dir" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]]; then
        cmd=(python "$RUN_BENCHMARK" --memory-backend "$backend" --defense-type "$DEFENSE" --test "$train_dir" --adaptive)
        # Match finance behavior: give explicit extra patience
        if [[ "$backend" == "explicit" ]]; then
          cmd+=(--early-stop-patience 20)
        fi
        TRAIN_COMMANDS+=("${cmd[*]}")
      fi
    done
  done

  echo "Found ${#TRAIN_COMMANDS[@]} adaptive train job(s) for topic $topic."
  if [[ ${#TRAIN_COMMANDS[@]} -eq 0 ]]; then
    echo "No train JSONs found for $topic under ${TRAIN_SPLITS[*]} – skipping training for this topic."
  else
    i=0
    while [[ $i -lt ${#TRAIN_COMMANDS[@]} ]]; do
      batch=0
      while [[ $batch -lt $MAX_PARALLEL_TRAIN && $i -lt ${#TRAIN_COMMANDS[@]} ]]; do
        echo "Starting TRAIN job $((i + 1))/${#TRAIN_COMMANDS[@]} for topic $topic"
        ( cd "$REPO_ROOT" && eval "${TRAIN_COMMANDS[$i]}" ) &
        ((i++)) || true
        ((batch++)) || true
      done
      wait
    done
  fi

  echo
  echo ">>> TRAINING COMPLETE for topic: $topic"
  echo

  # -----------------------------
  # 2) PROPAGATE TRAIN -> TEST
  # -----------------------------
  echo ">>> PROPAGATING TRAIN ATTACKS TO TEST for topic: $topic"
  # This script updates all topics; calling it once per topic is safe.
  ( cd "$REPO_ROOT" && python scripts/propagate_train_attack_to_test_cases.py --model "$MODEL" )
  echo "Propagation done."
  echo

  # -----------------------------
  # 3) STATIC TESTS WITH --try-all
  # -----------------------------
  echo ">>> STATIC TESTS for topic: $topic (defense=$DEFENSE, backends=${BACKENDS[*]}, --try-all)"

  shopt -s nullglob
  test_splits=("$ATTACK_BENCH"/test*)
  shopt -u nullglob

  if [[ ${#test_splits[@]} -eq 0 ]]; then
    echo "No test* splits found under $ATTACK_BENCH – skipping tests for topic $topic."
  else
    for split_dir in "${test_splits[@]}"; do
      split_name="$(basename "$split_dir")"
      topic_test_dir="$split_dir/$topic"

      if [[ ! -d "$topic_test_dir" ]]; then
        echo "Skip split $split_name for topic $topic: no directory $topic_test_dir"
        continue
      fi

      echo "Running tests for topic $topic in split $split_name:"
      echo "  Test path: $topic_test_dir"

      ( cd "$REPO_ROOT" && python "$RUN_BENCHMARK" \
          --memory-backend "${BACKENDS[@]}" \
          --defense-type "$DEFENSE" \
          --test "$topic_test_dir" \
          --model "$MODEL" \
          --num-workers "$NUM_WORKERS_TEST" \
          --try-all )
      echo
    done
  fi

  echo "<<< COMPLETED TOPIC: $topic"
  echo
done

echo "=============================================="
echo "All topics completed: ${TOPICS[*]}"
echo "=============================================="

