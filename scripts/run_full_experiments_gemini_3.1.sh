#!/usr/bin/env bash
# Use unbuffered Python so progress appears in terminal (avoids "stuck" appearance after long adaptive runs)
export PYTHONUNBUFFERED=1
#
# Run full experiment pipeline for gemini-3.1-pro-preview only:
# - For each test suite (health, legal, tax), for each memory backend:
#   1. Run adaptive benchmark with defense=none on the train test case only.
#   2. Run propagate_train_attack_to_test_cases.py to copy attack email to test cases.
#   3. Run non-adaptive benchmark for all defenses and all test cases for that backend.
# - Finally run consolidate_attack_results.py to generate heatmaps and CSVs.
#
# Memory backends: none (baseline), explicit, mem0, rag, context.
# Defenses: none + user_prompt_only, no_untrusted_tools, limit_memory_length, provable_policy.
#
# Usage:
#   ./scripts/run_full_experiments_gemini_3.1.sh
#   ./scripts/run_full_experiments_gemini_3.1.sh --no-consolidate   # skip final consolidation
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="gemini-3.1-pro-preview"
SUITES=(persistent_exfiltrate_health persistent_exfiltrate_legal persistent_exfiltrate_tax)
BACKENDS=(none explicit mem0 rag context)
# All defenses: none + 4 (user_prompt_only, no_untrusted_tools, limit_memory_length, provable_policy)
DEFENSES=(none user_prompt_only no_untrusted_tools limit_memory_length provable_policy)
BENCH_DIR="data/benchmark/attack_bench"
RUN_CONSOLIDATE=true

for arg in "$@"; do
  case "$arg" in
    --no-consolidate) RUN_CONSOLIDATE=false ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

echo "================================================================================"
echo "Full experiments: model=$MODEL, suites=${SUITES[*]}, backends=${BACKENDS[*]}"
echo "================================================================================"

for suite in "${SUITES[@]}"; do
  echo ""
  echo "========== Suite: $suite =========="

  for backend in "${BACKENDS[@]}"; do
    suite_dir="${BENCH_DIR}/${backend}/${suite}"
    train_json="${suite_dir}/00_${suite}_train.json"

    if [[ ! -d "$suite_dir" ]]; then
      echo "Skip backend $backend suite $suite: directory not found ($suite_dir)"
      continue
    fi
    if [[ ! -f "$train_json" ]]; then
      echo "Skip backend $backend suite $suite: train file not found ($train_json)"
      continue
    fi

    echo ""
    echo "--- Adaptive (defense=none, train only): $backend / $suite ---"
    python scripts/run_benchmark.py \
      --memory-backend "$backend" \
      --defense none \
      --test "$train_json" \
      --model "$MODEL" \
      --adaptive \
      # --force

    echo ""
    echo "--- Propagate train attack to test cases ---"
    python scripts/propagate_train_attack_to_test_cases.py

    echo ""
    echo "--- Non-adaptive (all defenses, all test cases): $backend / $suite ---"
    python scripts/run_benchmark.py \
      --memory-backend "$backend" \
      --defense-type "${DEFENSES[@]}" \
      --test "$suite_dir" \
      --model "$MODEL" \
      --num-workers 8
      # --force \

  done
done

if [[ "$RUN_CONSOLIDATE" == true ]]; then
  echo ""
  echo "================================================================================"
  echo "Consolidating attack results and generating heatmaps/CSVs"
  echo "================================================================================"
  python scripts/consolidate_attack_results.py --model "$MODEL"
  echo "Done. Outputs in data/benchmark/consolidated_attack_results/"
else
  echo ""
  echo "Skipped consolidation (--no-consolidate). Run manually:"
  echo "  python scripts/consolidate_attack_results.py --model $MODEL"
fi
