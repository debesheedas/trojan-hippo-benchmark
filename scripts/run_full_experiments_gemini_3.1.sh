#!/usr/bin/env bash
# Run adaptive benchmark on all train cases only (train phase of attack bench).
# For each memory backend and each defense type (except those in DEFENSES_TO_SKIP),
# runs adaptive on attack_bench/train/{backend}/. Successful attacks are cached
# under train_cache/{backend}/{defense}/{suite}/.
#
# Defenses to skip: add to DEFENSES_TO_SKIP (leave empty to run all).
#
# After this:
#   - Run tests with run_benchmark.py (attack email is resolved at runtime:
#     cache(backend, defense) else cache(backend, none) else original).
#   - Optionally run propagate_train_attack_to_test_cases.py --defense D to bake
#     a chosen attack into test files before running benchmarks.
#
# Usage:
#   ./scripts/run_full_experiments_gemini_3.1.sh
#   ./scripts/run_full_experiments_gemini_3.1.sh --force

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="gemini-3.1-pro-preview"
BACKENDS=(none explicit mem0 rag context)
DEFENSES=(none user_prompt_only no_untrusted_tools limit_memory_length provable_policy)

# Defenses to skip (e.g. "limit_memory_length" "provable_policy"). Leave empty to run all.
DEFENSES_TO_SKIP=()

BENCH_DIR="data/benchmark/attack_bench"
TRAIN_BASE="${BENCH_DIR}/train"
FORCE_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --force) FORCE_FLAG="--force" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

RUN_DEFENSES=()
for d in "${DEFENSES[@]}"; do
  skip=false
  for s in "${DEFENSES_TO_SKIP[@]}"; do
    if [[ "$d" == "$s" ]]; then skip=true; break; fi
  done
  if [[ "$skip" == false ]]; then
    RUN_DEFENSES+=("$d")
  fi
done

echo "================================================================================"
echo "Attack bench TRAIN (adaptive): model=$MODEL"
echo "Backends: ${BACKENDS[*]}"
echo "Defenses to run: ${RUN_DEFENSES[*]}"
echo "Defenses skipped: ${DEFENSES_TO_SKIP[*]:-(none)}"
echo "================================================================================"

for backend in "${BACKENDS[@]}"; do
  train_dir="${TRAIN_BASE}/${backend}"
  if [[ ! -d "$train_dir" ]]; then
    echo "Skip backend $backend: train dir not found ($train_dir)"
    continue
  fi
  for defense in "${RUN_DEFENSES[@]}"; do
    echo ""
    echo "--- Adaptive train: backend=$backend defense=$defense ---"
    python scripts/run_benchmark.py \
      --memory-backend "$backend" \
      --defense-type "$defense" \
      --test "$train_dir" \
      --model "$MODEL" \
      --adaptive \
      $FORCE_FLAG
  done
done

echo ""
echo "Done. Cache: ${BENCH_DIR}/train_cache/{backend}/{defense}/{suite}/"
echo "Run tests: python scripts/run_benchmark.py --test ${BENCH_DIR}/test/<backend> --defense-type <D> --model $MODEL ..."
