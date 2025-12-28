#!/bin/bash
# Run memory_only test suite with memory disabled (backend = "none")
#
# When memory is disabled, defense types don't matter, so we:
# 1. Run the benchmark once with --memory-backend none --defense-type none
# 2. The consolidate_results.py script will use these results for all defense types
#
# Usage:
#   bash scripts/run_no_memory_benchmark.sh [--model MODEL_NAME] [--force]
#
# Example:
#   bash scripts/run_no_memory_benchmark.sh --model gpt-5-mini --force

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
MODEL="gpt-5-mini"  # Default model
FORCE_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_NAME] [--force]"
            exit 1
            ;;
    esac
done

# Test suite
TEST_SUITE="memory_only"

echo "================================================================"
echo "Running No-Memory Benchmark for memory_only Suite"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Test Suite: $TEST_SUITE"
echo "  Memory Backend: none (memory disabled)"
echo "  Defense Type: none"
echo "  Force: ${FORCE_FLAG:-false}"
echo ""
echo -e "${CYAN}Note: When memory is disabled, all defense types produce identical results.${NC}"
echo -e "${CYAN}The consolidate_results.py script will use these results for all defense types.${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run the benchmark once with memory disabled
echo "================================================================"
echo "Running benchmark with memory disabled"
echo "================================================================"
echo ""

if python scripts/run_benchmark.py \
    --memory-backend none \
    --defense-type none \
    --suite "$TEST_SUITE" \
    --model "$MODEL" \
    $FORCE_FLAG; then
    echo ""
    echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Benchmark failed${NC}"
    exit 1
fi

echo ""
echo "================================================================"
echo "Summary"
echo "================================================================"
echo ""
echo -e "${GREEN}✓ Benchmark run completed${NC}"
echo ""
echo "Results location:"
echo "  data/benchmark/results/$MODEL/none/none/$TEST_SUITE/"
echo ""
echo -e "${CYAN}The consolidate_results.py script will use these results for all defense types${NC}"
echo -e "${CYAN}when generating consolidated tables (since memory is disabled).${NC}"
echo ""

