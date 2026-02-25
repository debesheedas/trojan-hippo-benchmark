#!/usr/bin/env bash
#
# Generate persistent attack bench for session checkpoints 0, 10, 20, ..., 100.
# Creates train_N and test_N under data/benchmark/attack_bench for each N.
#
# Prerequisites:
#   - Session snapshots must exist for sessions 10, 20, ..., 100 in both
#     data/benchmark/snapshots/memory_snapshots_train and memory_snapshots_test
#     (e.g. explicit/none/session_50.json). Session 0 does not load a snapshot.
#
# Usage:
#   From repo root:
#     PYTHONPATH=src ./scripts/generate_persistence_attack_bench.sh
#   Or with custom attack_bench dir:
#     ATTACK_BENCH_DIR=data/benchmark/attack_bench PYTHONPATH=src ./scripts/generate_persistence_attack_bench.sh
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ATTACK_BENCH_DIR="${ATTACK_BENCH_DIR:-$REPO_ROOT/data/benchmark/attack_bench}"
# Checkpoints: 0 (no snapshot), then every 10 up to 100
SESSION_CHECKPOINTS=(0 10 20 30 40 50 60 70 80 90 100)

cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

echo "=============================================="
echo "Generate persistence attack bench"
echo "Checkpoints: ${SESSION_CHECKPOINTS[*]}"
echo "Attack bench: $ATTACK_BENCH_DIR"
echo "=============================================="

for n in "${SESSION_CHECKPOINTS[@]}"; do
  echo "--- Session checkpoint $n ---"
  # Train split: persistence_unrelated_20_train
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests \
    --n "$n" \
    --defense all \
    --split train \
    --attack-bench-dir "$ATTACK_BENCH_DIR" \
    --snapshot-set-id persistence_unrelated_20_train

  # Test split: persistence_unrelated_20_test
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests \
    --n "$n" \
    --defense all \
    --split test \
    --attack-bench-dir "$ATTACK_BENCH_DIR" \
    --snapshot-set-id persistence_unrelated_20_test
done

echo "=============================================="
echo "Done. train_0..train_100 and test_0..test_100 written under $ATTACK_BENCH_DIR"
echo "=============================================="
