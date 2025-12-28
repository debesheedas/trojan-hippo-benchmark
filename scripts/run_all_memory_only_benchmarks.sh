#!/bin/bash
# Run all memory backends and defense combinations on memory_only test suite
#
# This script runs the memory_only test suite with:
# - All memory backends: explicit, mem0, rag, context
# - All defense types: none, user_prompt_only, no_untrusted_tools, limit_memory_length, provable_policy

#
# Usage:
#   bash scripts/run_all_memory_only_benchmarks.sh [--model MODEL_NAME] [--num-workers N] [--force] [--defense-type DEFENSE_TYPE] [--memory-backend BACKEND]
#
# Example:
#   bash scripts/run_all_memory_only_benchmarks.sh --model gpt-5-mini --num-workers 8
#   bash scripts/run_all_memory_only_benchmarks.sh --model gpt-5-mini --num-workers 1  # Serial execution
#   bash scripts/run_all_memory_only_benchmarks.sh --model gpt-5-mini --force  # Force re-run all benchmarks
#   bash scripts/run_all_memory_only_benchmarks.sh --model gpt-5-mini --defense-type none  # Run all backends only for no-defense
#   bash scripts/run_all_memory_only_benchmarks.sh --model gpt-5-mini --memory-backend mem0  # Run all defenses for mem0 only

# Don't exit on error - we want to continue with other combinations even if one fails

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
MODEL="gpt-5-mini"  # Default model
NUM_WORKERS=8  # Default number of parallel workers
USE_FORCE=""  # Default: don't use --force flag
SELECTED_DEFENSE_TYPE=""  # Default: run all defense types
SELECTED_MEMORY_BACKEND=""  # Default: run all memory backends
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --force)
            USE_FORCE="--force"
            shift
            ;;
        --defense-type)
            SELECTED_DEFENSE_TYPE="$2"
            shift 2
            ;;
        --memory-backend)
            SELECTED_MEMORY_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_NAME] [--num-workers N] [--force] [--defense-type DEFENSE_TYPE] [--memory-backend BACKEND]"
            exit 1
            ;;
    esac
done

# Validate num_workers
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [ "$NUM_WORKERS" -lt 1 ]; then
    echo "Error: --num-workers must be a positive integer"
    exit 1
fi

# Memory backends to test (excluding "none" which disables memory)
ALL_MEMORY_BACKENDS=("explicit" "mem0" "rag" "context")

# Filter memory backends if a specific one was selected
if [ -n "$SELECTED_MEMORY_BACKEND" ]; then
    # Validate that the selected memory backend is valid
    VALID_BACKEND=false
    for backend in "${ALL_MEMORY_BACKENDS[@]}"; do
        if [ "$backend" = "$SELECTED_MEMORY_BACKEND" ]; then
            VALID_BACKEND=true
            break
        fi
    done
    
    if [ "$VALID_BACKEND" = false ]; then
        echo "Error: Invalid memory backend '$SELECTED_MEMORY_BACKEND'"
        echo "Valid memory backends: ${ALL_MEMORY_BACKENDS[*]}"
        exit 1
    fi
    
    # Set MEMORY_BACKENDS to only include the selected one
    MEMORY_BACKENDS=("$SELECTED_MEMORY_BACKEND")
else
    # Use all memory backends
    MEMORY_BACKENDS=("${ALL_MEMORY_BACKENDS[@]}")
fi

# Defense types to test
ALL_DEFENSE_TYPES=("none" "user_prompt_only" "no_untrusted_tools" "limit_memory_length" "provable_policy")

# Filter defense types if a specific one was selected
if [ -n "$SELECTED_DEFENSE_TYPE" ]; then
    # Validate that the selected defense type is valid
    VALID_DEFENSE=false
    for defense in "${ALL_DEFENSE_TYPES[@]}"; do
        if [ "$defense" = "$SELECTED_DEFENSE_TYPE" ]; then
            VALID_DEFENSE=true
            break
        fi
    done
    
    if [ "$VALID_DEFENSE" = false ]; then
        echo "Error: Invalid defense type '$SELECTED_DEFENSE_TYPE'"
        echo "Valid defense types: ${ALL_DEFENSE_TYPES[*]}"
        exit 1
    fi
    
    # Set DEFENSE_TYPES to only include the selected one
    DEFENSE_TYPES=("$SELECTED_DEFENSE_TYPE")
else
    # Use all defense types
    DEFENSE_TYPES=("${ALL_DEFENSE_TYPES[@]}")
fi

# Test suite
TEST_SUITE="memory_only"

echo "================================================================"
if [ -n "$SELECTED_MEMORY_BACKEND" ]; then
    echo "Running All Defenses for $SELECTED_MEMORY_BACKEND Backend on memory_only Suite"
else
    echo "Running All Memory Backends and Defenses on memory_only Suite"
fi
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Test Suite: $TEST_SUITE"
echo "  Memory Backends: ${MEMORY_BACKENDS[*]}"
echo "  Defense Types: ${DEFENSE_TYPES[*]}"
echo "  Number of Workers: $NUM_WORKERS"
if [ -n "$USE_FORCE" ]; then
    echo "  Force: Enabled (will re-run all benchmarks)"
else
    echo "  Force: Disabled (will skip existing results)"
fi
echo ""
echo "Total combinations: $((${#MEMORY_BACKENDS[@]} * ${#DEFENSE_TYPES[@]}))"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Create error log file (append mode, will be created if doesn't exist)
ERROR_LOG="$PROJECT_ROOT/error.log"
echo "================================================================" >> "$ERROR_LOG"
echo "Benchmark run started: $(date)" >> "$ERROR_LOG"
echo "Model: $MODEL, Test Suite: $TEST_SUITE" >> "$ERROR_LOG"
echo "================================================================" >> "$ERROR_LOG"
echo "" >> "$ERROR_LOG"

# Track results
TOTAL_COMBINATIONS=$((${#MEMORY_BACKENDS[@]} * ${#DEFENSE_TYPES[@]}))
SUCCESSFUL=0
FAILED=0

# Start time
START_TIME=$(date +%s)

# Function to extract errors from log (exclude test failures, include actual errors)
extract_errors() {
    local log_file=$1
    local backend=$2
    local defense=$3
    
    # Filter for actual errors (not test failures)
    # Look for: ValueError, TypeError, ImportError, AttributeError, KeyError, etc.
    # Exclude: "FAILED", "✗ FAILED" (test failures are normal)
    # Include: Traceback, Error, Exception (but not test result messages)
    if [ -f "$log_file" ]; then
        # Extract Python errors and exceptions - be more inclusive
        # Capture any line with error indicators (case-insensitive)
        grep -iE "(traceback|valueerror|typeerror|importerror|attributeerror|keyerror|indexerror|runtimeerror|exception|error:|warning:.*error|file.*line.*in|raise |assertionerror|oserror|ioerror|cannot import)" "$log_file" 2>/dev/null | \
        grep -vE "(✗ FAILED|✓ PASSED|Test Result.*FAILED|Steps Summary|✓ Success|✗ Failed|completed successfully|PASSED|FAILED.*test)" | \
        while IFS= read -r line; do
            # Only add non-empty lines
            if [ -n "$line" ]; then
                echo "[$backend+$defense] $line" >> "$ERROR_LOG"
            fi
        done
        
        # Also capture the last few lines if they contain error patterns (for cases where error is at the end)
        # This helps catch errors that might not match the main pattern
        if tail -5 "$log_file" 2>/dev/null | grep -qiE "(error|exception|traceback|failed|cannot|import)"; then
            tail -5 "$log_file" 2>/dev/null | grep -iE "(error|exception|traceback|failed|cannot|import)" | \
            grep -vE "(✗ FAILED|✓ PASSED|Test Result|Steps Summary|✓ Success|✗ Failed|completed successfully|PASSED|FAILED.*test)" | \
            while IFS= read -r line; do
                if [ -n "$line" ]; then
                    echo "[$backend+$defense] $line" >> "$ERROR_LOG"
                fi
            done
        fi
    else
        # Log file doesn't exist - this is an error itself
        echo "[$backend+$defense] ERROR: Log file not found: $log_file" >> "$ERROR_LOG"
    fi
}

# Function to run a single benchmark job
run_benchmark_job() {
    local backend=$1
    local defense=$2
    local job_num=$3
    local total=$4
    local result_file=$5
    local force_flag=$6
    
    local prefix="[${job_num}/${total}] ${backend}+${defense}"
    local log_file="/tmp/benchmark_${backend}_${defense}_$$.log"
    
    # Run benchmark and capture result
    if [ -n "$force_flag" ]; then
        # With force flag
        if python scripts/run_benchmark.py \
            --memory-backend "$backend" \
            --defense-type "$defense" \
            --suite "$TEST_SUITE" \
            --model "$MODEL" \
            $force_flag > "$log_file" 2>&1; then
            echo "SUCCESS" > "$result_file"
            echo -e "${GREEN}✓${NC} $prefix completed successfully"
            # Extract any errors from log (even if job succeeded, there might be warnings)
            extract_errors "$log_file" "$backend" "$defense"
        else
            echo "FAILURE" > "$result_file"
            echo -e "${RED}✗${NC} $prefix failed"
            # Extract errors from failed job
            extract_errors "$log_file" "$backend" "$defense"
        fi
    else
        # Without force flag
        if python scripts/run_benchmark.py \
            --memory-backend "$backend" \
            --defense-type "$defense" \
            --suite "$TEST_SUITE" \
            --model "$MODEL" > "$log_file" 2>&1; then
            echo "SUCCESS" > "$result_file"
            echo -e "${GREEN}✓${NC} $prefix completed successfully"
            # Extract any errors from log (even if job succeeded, there might be warnings)
            extract_errors "$log_file" "$backend" "$defense"
        else
            echo "FAILURE" > "$result_file"
            echo -e "${RED}✗${NC} $prefix failed"
            # Extract errors from failed job
            extract_errors "$log_file" "$backend" "$defense"
        fi
    fi
}

# Function to wait for a job slot to become available
wait_for_job_slot() {
    while [ $(jobs -r | wc -l) -ge "$NUM_WORKERS" ]; do
        sleep 0.1
    done
}

# Create arrays to store job info
declare -a JOB_IDS=()
declare -a RESULT_FILES=()
declare -a JOB_INFO=()

# Run each combination in parallel (with worker limit)
CURRENT=0
for backend in "${MEMORY_BACKENDS[@]}"; do
    for defense in "${DEFENSE_TYPES[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # Wait for a job slot if we're at the worker limit
        if [ "$NUM_WORKERS" -gt 1 ]; then
            wait_for_job_slot
        fi
        
        echo ""
        echo "================================================================"
        echo -e "${CYAN}[$CURRENT/$TOTAL_COMBINATIONS]${NC} Launching: ${BLUE}$backend${NC} + ${MAGENTA}$defense${NC}"
        echo "================================================================"
        
        # Create temp file for result
        result_file="/tmp/benchmark_result_${backend}_${defense}_$$.txt"
        RESULT_FILES+=("$result_file")
        JOB_INFO+=("$backend|$defense|$CURRENT")
        
        # Run benchmark in background or foreground
        if [ "$NUM_WORKERS" -gt 1 ]; then
            # Parallel execution
            (run_benchmark_job "$backend" "$defense" "$CURRENT" "$TOTAL_COMBINATIONS" "$result_file" "$USE_FORCE") &
            JOB_IDS+=($!)
        else
            # Serial execution
            echo ""
            if [ -n "$USE_FORCE" ]; then
                # With force flag
                local log_file="/tmp/benchmark_${backend}_${defense}_$$.log"
                if python scripts/run_benchmark.py \
                    --memory-backend "$backend" \
                    --defense-type "$defense" \
                    --suite "$TEST_SUITE" \
                    --model "$MODEL" \
                    $USE_FORCE > "$log_file" 2>&1; then
                    echo ""
                    echo -e "${GREEN}✓ Success: $backend + $defense${NC}"
                    SUCCESSFUL=$((SUCCESSFUL + 1))
                    # Extract any errors from log (even if job succeeded, there might be warnings)
                    extract_errors "$log_file" "$backend" "$defense"
                else
                    echo ""
                    echo -e "${RED}✗ Failed: $backend + $defense${NC}"
                    FAILED=$((FAILED + 1))
                    # Extract errors to error log
                    extract_errors "$log_file" "$backend" "$defense"
                fi
                rm -f "$log_file"
            else
                # Without force flag
                local log_file="/tmp/benchmark_${backend}_${defense}_$$.log"
                if python scripts/run_benchmark.py \
                    --memory-backend "$backend" \
                    --defense-type "$defense" \
                    --suite "$TEST_SUITE" \
                    --model "$MODEL" > "$log_file" 2>&1; then
                    echo ""
                    echo -e "${GREEN}✓ Success: $backend + $defense${NC}"
                    SUCCESSFUL=$((SUCCESSFUL + 1))
                    # Extract any errors from log (even if job succeeded, there might be warnings)
                    extract_errors "$log_file" "$backend" "$defense"
                else
                    echo ""
                    echo -e "${RED}✗ Failed: $backend + $defense${NC}"
                    FAILED=$((FAILED + 1))
                    # Extract errors to error log
                    extract_errors "$log_file" "$backend" "$defense"
                fi
                rm -f "$log_file"
            fi
            echo ""
            echo "---"
        fi
    done
done

# Wait for all background jobs to complete and collect results
if [ "$NUM_WORKERS" -gt 1 ]; then
    echo ""
    echo "================================================================"
    echo "Waiting for all jobs to complete..."
    echo "================================================================"
    echo ""
    echo "Note: Jobs run in parallel. Results appear as each job completes."
    echo ""
    
    # Wait for all jobs and collect results
    for i in "${!JOB_IDS[@]}"; do
        job_id="${JOB_IDS[$i]}"
        result_file="${RESULT_FILES[$i]}"
        job_info="${JOB_INFO[$i]}"
        
        IFS='|' read -r backend defense job_num <<< "$job_info"
        
        # Show which job we're waiting for
        echo -e "${YELLOW}Waiting for job [$job_num/$TOTAL_COMBINATIONS]: $backend + $defense...${NC}"
        
        # Wait for this specific job
        wait $job_id
        
        # Read result
        result=$(cat "$result_file" 2>/dev/null || echo "FAILURE")
        rm -f "$result_file"
        
        # Update counters
        if [ "$result" = "SUCCESS" ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
            echo -e "${GREEN}  ✓ Job [$job_num/$TOTAL_COMBINATIONS] completed${NC}"
        else
            FAILED=$((FAILED + 1))
            echo -e "${RED}  ✗ Job [$job_num/$TOTAL_COMBINATIONS] failed${NC}"
            # Show log file for failed jobs
            log_file="/tmp/benchmark_${backend}_${defense}_$$.log"
            if [ -f "$log_file" ]; then
                echo ""
                echo -e "${RED}Error output for $backend + $defense:${NC}"
                tail -20 "$log_file"
                echo ""
                # Extract errors to error log BEFORE cleaning up
                extract_errors "$log_file" "$backend" "$defense"
                # Clean up log file after extracting errors
                rm -f "$log_file"
            else
                # Log file doesn't exist - this is an error itself
                echo "[$backend+$defense] ERROR: Log file not found for failed job: $log_file" >> "$ERROR_LOG"
            fi
        fi
    done
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Print summary
echo ""
echo "================================================================"
echo "Summary"
echo "================================================================"
echo ""
echo "Total combinations: $TOTAL_COMBINATIONS"
echo -e "  ${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"
echo ""
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results location: data/benchmark/results/$MODEL/"
echo "Error log: $ERROR_LOG"
echo ""

# Check if error log has any entries (excluding header)
if [ -f "$ERROR_LOG" ] && [ $(wc -l < "$ERROR_LOG") -gt 5 ]; then
    echo -e "${YELLOW}⚠️  Errors detected during benchmark run. Check $ERROR_LOG for details.${NC}"
    echo ""
fi

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All benchmarks completed successfully!${NC}"
    # Add summary to error log
    echo "" >> "$ERROR_LOG"
    echo "Benchmark run completed: $(date)" >> "$ERROR_LOG"
    echo "Status: All benchmarks passed" >> "$ERROR_LOG"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some benchmarks failed. Check the output above for details.${NC}"
    # Add summary to error log
    echo "" >> "$ERROR_LOG"
    echo "Benchmark run completed: $(date)" >> "$ERROR_LOG"
    echo "Status: $FAILED benchmark(s) failed" >> "$ERROR_LOG"
    exit 1
fi

