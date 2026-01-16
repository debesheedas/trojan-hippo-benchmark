#!/usr/bin/env python3
"""
Unified Benchmark Runner

Replaces the three duplicate scripts:
- run_explicit_defense_benchmarks.py
- run_mem0_defense_benchmarks.py
- run_rag_defense_benchmarks.py

This script runs benchmarks for any memory backend with any defense type.
Uses unified abstractions and result structure.
"""

import os
import sys

# Fix OpenMP initialization error on macOS
# This MUST be set before ANY imports that might use OpenMP (e.g., FAISS, numpy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import multiprocessing
import sys
import signal

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.utils import load_config
from benchmark.defense_backend import get_defense_backend_registry, UNIFIED_DEFENSE_TYPES
from benchmark.benchmark_utils import (
    should_skip_test,
    determine_attack_type,
    discover_test_files
)
from benchmark.test_bench import TestBench


def check_results_exist(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    test_files: List[Path],
    results_base_dir: Path,
    force: bool
) -> tuple[bool, List[str]]:
    """
    Check if all results exist for given tests.
    
    Returns:
        (all_exist, list_of_missing_tests)
    """
    if force:
        return False, []
    
    missing = []
    for test_file in test_files:
        # Get attack type from test file (handles memory_only tests)
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            attack_type = determine_attack_type(test_file, test_data)
        except Exception:
            attack_type = determine_attack_type(test_file, None)
        
        if not should_skip_test(
            memory_backend,
            unified_defense,
            model_name,
            attack_type,
            test_file,
            force,
            results_base_dir
        ):
            missing.append(test_file.name)
    
    return len(missing) == 0, missing


def run_benchmark(
    memory_backend: str,
    unified_defense: str,
    test_path: str,
    config_path: str = "benchmark_config.yaml",
    force: bool = False,
    results_base_dir: Optional[Path] = None,
    target_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run benchmark for a specific memory backend and defense type.
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", "context", or "none" for disable_memory)
        unified_defense: Unified defense name (e.g., "none", "user_prompt_only")
        test_path: Path to test file, directory, or suite name
        config_path: Path to config file
        force: Force overwrite existing results
        results_base_dir: Base directory for results (defaults to data/benchmark/results)
        
    Returns:
        Dictionary with results summary
    """
    # Handle no memory backend
    if memory_backend == "none":
        print(f"\n{'='*80}")
        print(f"Running benchmark with NO MEMORY (memory_backend: none)")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Running {memory_backend.upper()} benchmark with defense: {unified_defense}")
        print(f"{'='*80}\n")
    
    # Load benchmark config and make a deep copy to avoid modifying the original
    # This is important when multiple processes might be running in parallel
    import copy
    config = copy.deepcopy(load_config(config_path))
    
    # Load agent config from agent_config.yaml and merge it into benchmark config
    # Agent settings (target_model_name, etc.) and memory settings are only in agent_config.yaml to avoid duplication
    try:
        agent_config = load_config("agent_config.yaml")
        # Merge agent section
        if "agent" in agent_config:
            config["agent"] = agent_config["agent"].copy()
        # Merge memory section (memory settings are only in agent_config.yaml)
        if "memory" in agent_config:
            config["memory"] = agent_config["memory"].copy()
        # Merge seed if present
        if "seed" in agent_config:
            config["seed"] = agent_config["seed"]
    except FileNotFoundError:
        # If agent_config.yaml doesn't exist, that's okay - agent section will be missing
        # and will be handled by the target_model_name check below
        pass
    
    # Set target model name (command line arg takes precedence over config)
    if target_model_name:
        if "agent" not in config:
            config["agent"] = {}
        config["agent"]["target_model_name"] = target_model_name
    elif "agent" not in config or "target_model_name" not in config.get("agent", {}):
        # If no agent config was loaded and no target_model_name provided, raise error
        raise ValueError(
            "Agent configuration missing. Please ensure agent_config.yaml exists with an 'agent' section, "
            "or provide --model argument to specify target_model_name."
        )
    
    # Handle no memory backend: disable all backends
    if memory_backend == "none":
        if "memory" not in config:
            config["memory"] = {}
        # Disable all memory backends
        for backend_name in ["explicit", "mem0", "rag", "context"]:
            if backend_name not in config["memory"]:
                config["memory"][backend_name] = {}
            config["memory"][backend_name]["enabled"] = False
        # Set backend to "none" in config
        config["memory"]["backend"] = "none"
        # Store defense_type at top level for "none" backend (needed for provable_policy defense)
        # This allows defenses to work even when memory is disabled
        config["memory"]["defense_type"] = unified_defense
    else:
        # Set memory backend in config
        if "memory" not in config:
            config["memory"] = {}
        config["memory"]["backend"] = memory_backend
        
        # Enable the specified backend and disable others
        for backend_name in ["explicit", "mem0", "rag", "context"]:
            # Ensure nested structure exists
            if backend_name not in config["memory"]:
                config["memory"][backend_name] = {}
            elif config["memory"][backend_name] is None:
                config["memory"][backend_name] = {}
            
            # Handle backend-specific naming (explicit_memory vs explicit, mem0_memory vs mem0, etc.)
            backend_config_key = backend_name
            if backend_name == "explicit":
                backend_config_key = "explicit_memory"
            elif backend_name == "mem0":
                backend_config_key = "mem0_memory"
            elif backend_name == "rag":
                backend_config_key = "rag_memory"
            elif backend_name == "context":
                backend_config_key = "context_memory"
            
            # Ensure backend-specific config exists and is a dict (not None)
            if backend_config_key not in config["memory"]:
                config["memory"][backend_config_key] = {}
            elif config["memory"][backend_config_key] is None:
                config["memory"][backend_config_key] = {}
            
            if backend_name == memory_backend:
                # Enable both the generic key and the specific key
                config["memory"][backend_name]["enabled"] = True
                config["memory"][backend_config_key]["enabled"] = True
                # Set defense type (map unified to backend-specific)
                defense_registry = get_defense_backend_registry()
                backend_defense = defense_registry.map_defense(memory_backend, unified_defense)
                config["memory"][backend_name]["defense_type"] = backend_defense
                config["memory"][backend_config_key]["defense_type"] = backend_defense
            else:
                # Disable both keys
                config["memory"][backend_name]["enabled"] = False
                config["memory"][backend_config_key]["enabled"] = False
    
    # Set results directory (command line arg takes precedence, default if not provided)
    if results_base_dir is None:
        results_base_dir = Path("data/benchmark/results")  # Hardcoded default
    
    # Set results_dir in config for TestBench (it reads from config)
    if "benchmark" not in config:
        config["benchmark"] = {}
    config["benchmark"]["results_dir"] = str(results_base_dir)
    
    # Set test directory
    benchmark_config = config.get("benchmark", {})
    test_dir = Path(benchmark_config.get("test_dir", "data/benchmark/tests"))
    
    # Get test files
    test_files = discover_test_files(
        test_path=test_path,
        test_dir=test_dir,
        verbose=False
    )
    if not test_files:
        print(f"WARNING: No test files found for path: {test_path}")
        return {
            "success": False,
            "error": "No test files found",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    
    print(f"Found {len(test_files)} test file(s)")
    
    # Check if all results exist
    model_name = config.get("agent", {}).get("target_model_name", "unknown")
    all_exist, missing = check_results_exist(
        memory_backend,
        unified_defense,
        model_name,
        test_files,
        results_base_dir,
        force
    )
    
    if all_exist and not force:
        print(f"All results already exist for {memory_backend}/{unified_defense}")
        print(f"   Use --force to overwrite")
        return {
            "success": True,
            "skipped": True,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    
    if missing and not force:
        print(f"SKIPPED: {len(missing)} test(s) already have results, {len(test_files) - len(missing)} will run")
    
    # Initialize TestBench with config dict directly (no temp file needed)
    bench = TestBench(config=config, defense_type_override=unified_defense, force=force)
    
    try:
        # Run tests - TestBench.run_all_tests expects a path string
        # It will handle file/directory/suite discovery internally
        results = bench.run_all_tests(test_path)
        
        # Calculate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get("overall_success", False))
        failed_tests = total_tests - passed_tests
        
        return {
            "success": True,
            "memory_backend": memory_backend,
            "defense_type": unified_defense,
            "tests_run": total_tests,
            "tests_passed": passed_tests,
            "tests_failed": failed_tests,
            "results": results
        }
    finally:
        # Ensure cleanup happens even if there was an error
        # Only clean up test environments created by THIS benchmark run
        # Do NOT clean up old test environments here - that should be done
        # manually or at the end of all parallel runs to avoid interference
        bench.cleanup_all_test_environments()


def run_all_defenses(
    memory_backend: str,
    test_path: str,
    config_path: str = "benchmark_config.yaml",
    force: bool = False,
    defense_types: Optional[List[str]] = None,
    target_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run benchmark for all defense types for a memory backend.
    
    Args:
        memory_backend: Memory backend name
        test_path: Path to test file, directory, or suite name
        config_path: Path to config file
        force: Force overwrite existing results
        defense_types: List of defense types to run (defaults to all)
        
    Returns:
        Dictionary with results summary for all defenses
    """
    if defense_types is None:
        defense_types = UNIFIED_DEFENSE_TYPES
    
    print(f"\n{'#'*80}")
    print(f"Running all defenses for {memory_backend.upper()}")
    print(f"{'#'*80}\n")
    
    all_results = {}
    summary = {
        "memory_backend": memory_backend,
        "defenses": {},
        "total_tests": 0,
        "total_passed": 0,
        "total_failed": 0
    }
    
    for defense_type in defense_types:
        result = run_benchmark(
            memory_backend=memory_backend,
            unified_defense=defense_type,
            test_path=test_path,
            config_path=config_path,
            force=force,
            target_model_name=target_model_name
        )
        
        all_results[defense_type] = result
        
        if result.get("success"):
            if not result.get("skipped"):
                summary["defenses"][defense_type] = {
                    "tests_run": result.get("tests_run", 0),
                    "tests_passed": result.get("tests_passed", 0),
                    "tests_failed": result.get("tests_failed", 0)
                }
                summary["total_tests"] += result.get("tests_run", 0)
                summary["total_passed"] += result.get("tests_passed", 0)
                summary["total_failed"] += result.get("tests_failed", 0)
            else:
                summary["defenses"][defense_type] = {"skipped": True}
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {memory_backend.upper()}")
    print(f"{'='*80}")
    for defense_type, stats in summary["defenses"].items():
        if stats.get("skipped"):
            status = "SKIPPED"
        elif stats.get("tests_passed", 0) == stats.get("tests_run", 0):
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"{defense_type:20s}: {status}", end="")
        if not stats.get("skipped"):
            print(f" ({stats.get('tests_passed', 0)}/{stats.get('tests_run', 0)} tests)")
        else:
            print()
    
    if summary["total_tests"] > 0:
        print(f"\nTotal: {summary['total_passed']}/{summary['total_tests']} tests passed")
    
    return summary


def _check_result_for_errors(
    result: Dict[str, Any],
    backend_key: str,
    defense_key: str,
    summary: Dict[str, Any]
) -> Tuple[bool, bool, bool]:
    """
    Check a result dictionary for execution errors and update summary.
    
    Returns:
        (has_error, has_connection_error, has_rate_limit_error) tuple
    """
    error_msg = result.get("error", "")
    has_error = bool(error_msg)
    has_connection_error = False
    has_rate_limit_error = False
    
    if error_msg:
        error_lower = error_msg.lower()
        if "rate limit" in error_lower or "429" in error_lower or "quota" in error_lower:
            has_rate_limit_error = True
            if (backend_key, defense_key) not in summary["combinations_with_rate_limit_errors"]:
                summary["combinations_with_rate_limit_errors"].append((backend_key, defense_key))
        elif "connection" in error_lower or "timeout" in error_lower:
            has_connection_error = True
            if (backend_key, defense_key) not in summary["combinations_with_connection_errors"]:
                summary["combinations_with_connection_errors"].append((backend_key, defense_key))
    
    # Check for execution errors in test results (execution_success=False)
    test_results = result.get("results", [])
    for test_result in test_results:
        # Check execution_success flag
        if not test_result.get("execution_success", True):
            has_error = True
            if (backend_key, defense_key) not in summary["combinations_with_errors"]:
                summary["combinations_with_errors"].append((backend_key, defense_key))
            
            # Check execution_errors list for specific error types
            execution_errors = test_result.get("execution_errors")
            if execution_errors:
                for err_msg in execution_errors:
                    err_lower = str(err_msg).lower()
                    if "rate limit" in err_lower or "429" in err_lower:
                        has_rate_limit_error = True
                        if (backend_key, defense_key) not in summary["combinations_with_rate_limit_errors"]:
                            summary["combinations_with_rate_limit_errors"].append((backend_key, defense_key))
                    elif "connection" in err_lower or "timeout" in err_lower:
                        has_connection_error = True
                        if (backend_key, defense_key) not in summary["combinations_with_connection_errors"]:
                            summary["combinations_with_connection_errors"].append((backend_key, defense_key))
        
        # Also check step results for connection errors (backward compatibility)
        steps = test_result.get("steps", [])
        for step in steps:
            step_error = step.get("error", "")
            if step_error and ("connection" in step_error.lower() or "timeout" in step_error.lower()):
                has_connection_error = True
                if (backend_key, defense_key) not in summary["combinations_with_connection_errors"]:
                    summary["combinations_with_connection_errors"].append((backend_key, defense_key))
    
    # Track combinations with any errors
    if has_error or has_connection_error or has_rate_limit_error:
        if (backend_key, defense_key) not in summary["combinations_with_errors"]:
            summary["combinations_with_errors"].append((backend_key, defense_key))
    
    return (has_error, has_connection_error, has_rate_limit_error)


def _run_single_combination(
    args_tuple: Tuple[str, str, str, str, bool, Optional[Path], Optional[str]]
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Wrapper function to run a single backend+defense combination.
    This is used by ProcessPoolExecutor - it must be a top-level function
    (not a method) and must be picklable.
    
    IMPORTANT: Each call runs in a completely separate Python process with
    isolated memory space. Global variables, caches, and file handles are
    NOT shared between processes.
    
    All output is redirected to a log file specific to this combination.
    
    Args:
        args_tuple: (memory_backend, unified_defense, test_path, config_path, force, results_base_dir, target_model_name)
    
    Returns:
        (memory_backend, unified_defense, result_dict)
    """
    # determine_attack_type is already imported at module level
    
    # Set process name for debugging
    process_id = os.getpid()
    memory_backend, unified_defense, test_path, config_path, force, results_base_dir, target_model_name = args_tuple
    
    # Determine attack type from test_path (needed for result paths)
    test_dir = Path("data/benchmark/tests")
    if isinstance(test_path, str):
        if test_path in ["benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"]:
            attack_type = test_path
        else:
            # Try to determine from path
            test_file = test_dir / test_path
            if test_file.exists():
                attack_type = determine_attack_type(test_file, {})
            else:
                attack_type = "unknown"
    else:
        attack_type = determine_attack_type(test_path, {})
    
    # Note: We no longer create combination.log since each test case has its own log file
    # Individual test log files are created in TestBench.run_test_from_file()
    
    result = None
    try:
        # Print start message to terminal (not redirected to file)
        print(f"[PID {process_id}] Starting: {memory_backend} + {unified_defense}")
        print(f"Individual test logs will be written to: data/benchmark/logs/{target_model_name or 'unknown'}/{memory_backend}/{unified_defense}/{attack_type}/")
        print(f"{'='*80}\n", flush=True)
        
        result = run_benchmark(
            memory_backend=memory_backend,
            unified_defense=unified_defense,
            test_path=test_path,
            config_path=config_path,
            force=force,
            results_base_dir=results_base_dir,
            target_model_name=target_model_name
        )
        
        # Print completion summary to terminal
        print(f"\n{'='*80}", flush=True)
        print(f"[PID {process_id}] Completed: {memory_backend} + {unified_defense}", flush=True)
        print(f"Tests run: {result.get('tests_run', 0)}, Passed: {result.get('tests_passed', 0)}, Failed: {result.get('tests_failed', 0)}", flush=True)
        print(f"{'='*80}", flush=True)
        
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt so it propagates to the main process
        # This allows the signal handler to properly clean up all processes
        print(f"\n{'='*80}", flush=True)
        print(f"[PID {process_id}] INTERRUPTED in {memory_backend} + {unified_defense}", flush=True)
        print(f"{'='*80}", flush=True)
        raise  # Re-raise to propagate to main process
    except Exception as e:
        # Catch any exceptions and return error result
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"\n{'='*80}", flush=True)
        print(f"[PID {process_id}] ERROR in {memory_backend} + {unified_defense}: {error_msg}", flush=True)
        print(f"{'='*80}", flush=True)
        result = {
            "success": False,
            "error": str(e),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    finally:
        # Print completion status to terminal
        if result and result.get("success"):
            status = "OK"
        else:
            status = "ERROR"
        print(f"{status} {memory_backend.upper()} + {unified_defense} - See individual test logs in data/benchmark/logs/{target_model_name or 'unknown'}/{memory_backend}/{unified_defense}/{attack_type}/", flush=True)
    
    # Ensure result is never None
    if result is None:
        result = {
            "success": False,
            "error": "Unknown error - result is None",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    
    return (memory_backend, unified_defense, result)


# Global flag to track if we've been interrupted
_interrupted = False
_executor_ref = None  # Reference to executor for cleanup on interrupt
_futures_ref = None  # Reference to all futures for cancellation


def _signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM signals."""
    global _interrupted, _executor_ref, _futures_ref
    _interrupted = True
    print(f"\n\n{'='*80}", flush=True)
    print(f"INTERRUPTED: Received signal {signum}. Shutting down gracefully...", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Cancel all pending futures if they exist
    if _futures_ref is not None:
        print("Cancelling pending tasks...", flush=True)
        cancelled_count = 0
        for future in _futures_ref:
            if not future.done():
                future.cancel()
                cancelled_count += 1
        if cancelled_count > 0:
            print(f"Cancelled {cancelled_count} pending task(s).", flush=True)
    
    # Shutdown executor immediately if it exists
    if _executor_ref is not None:
        print("Terminating worker processes...", flush=True)
        _executor_ref.shutdown(wait=False, cancel_futures=True)
    
    # Re-raise KeyboardInterrupt so the script exits properly
    raise KeyboardInterrupt("Benchmark interrupted by user")


def run_all_combinations(
    memory_backends: List[str],
    defense_types: List[str],
    test_path: str,
    config_path: str = "benchmark_config.yaml",
    force: bool = False,
    num_workers: int = 1,
    results_base_dir: Optional[Path] = None,
    target_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run benchmarks for all combinations of memory backends and defense types in parallel.
    
    Each combination runs in a separate process with its own log file.
    All output is redirected to individual log files per combination.
    
    Args:
        memory_backends: List of memory backend names
        defense_types: List of defense type names
        test_path: Path to test file, directory, or suite name
        config_path: Path to config file
        force: Force overwrite existing results
        num_workers: Number of parallel workers (1 = serial execution)
        results_base_dir: Base directory for results
        target_model_name: Target model name
        
    Returns:
        Dictionary with results summary for all combinations
    """
    global _interrupted, _executor_ref
    # Generate all combinations
    # Treat "none" backend as a regular backend - run all defense types
    combinations = []
    for backend in memory_backends:
        for defense in defense_types:
            combinations.append((backend, defense))
    
    total_combinations = len(combinations)
    print(f"\n{'#'*80}")
    print(f"Running {total_combinations} combination(s) with {num_workers} worker(s)")
    print(f"Memory backends: {memory_backends}")
    print(f"Defense types: {defense_types}")
    print(f"{'#'*80}\n")
    
    # Note: Error logging is now done per-combination in individual log files
    # No shared error.log file needed - each combination has its own log
    
    rate_limit_count = 0
    api_error_count = 0
    
    all_results = {}
    summary = {
        "total_combinations": total_combinations,
        "combinations": {},
        "total_tests": 0,
        "total_passed": 0,
        "total_failed": 0,
        "successful_combinations": 0,
        "failed_combinations": 0,
        "combinations_with_errors": [],  # List of (backend, defense) tuples that had errors
        "combinations_with_connection_errors": [],  # List of (backend, defense) tuples with connection errors
        "combinations_with_rate_limit_errors": []  # List of (backend, defense) tuples with rate limit errors
    }
    
    # Prepare arguments for each combination
    args_list = [
        (backend, defense, test_path, config_path, force, results_base_dir, target_model_name)
        for backend, defense in combinations
    ]
    
    # Set up signal handlers for graceful shutdown (works for both serial and parallel)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Run combinations in parallel or serial
    if num_workers == 1:
        # Serial execution (easier debugging, no multiprocessing overhead)
        print("Running combinations serially...\n")
        print(f"TIP: Press Ctrl+C to interrupt and exit gracefully.\n")
        try:
            for i, args_tuple in enumerate(args_list, 1):
                # Check if we've been interrupted
                if _interrupted:
                    print("\nWARNING: Interrupt detected. Stopping execution...", flush=True)
                    break
                
                backend, defense = args_tuple[0], args_tuple[1]
                print(f"[{i}/{total_combinations}] {backend.upper()} + {defense}")
                print("-" * 80)
                
                try:
                    backend_key, defense_key, result = _run_single_combination(args_tuple)
                    
                    # Check for errors in result
                    has_error, has_connection_error, has_rate_limit_error = _check_result_for_errors(
                        result, backend_key, defense_key, summary
                    )
                    
                    all_results[f"{backend_key}_{defense_key}"] = result
                    
                    if result.get("success"):
                        if not result.get("skipped"):
                            summary["combinations"][f"{backend_key}_{defense_key}"] = {
                                "tests_run": result.get("tests_run", 0),
                                "tests_passed": result.get("tests_passed", 0),
                                "tests_failed": result.get("tests_failed", 0),
                                "has_errors": has_error or has_connection_error or has_rate_limit_error
                            }
                            summary["total_tests"] += result.get("tests_run", 0)
                            summary["total_passed"] += result.get("tests_passed", 0)
                            summary["total_failed"] += result.get("tests_failed", 0)
                            summary["successful_combinations"] += 1
                        else:
                            summary["combinations"][f"{backend_key}_{defense_key}"] = {"skipped": True}
                            summary["successful_combinations"] += 1
                    else:
                        summary["combinations"][f"{backend_key}_{defense_key}"] = {
                            "error": result.get("error", "Unknown error"),
                            "tests_run": result.get("tests_run", 0),
                            "has_errors": True
                        }
                        summary["failed_combinations"] += 1
                    
                    print()  # Blank line between combinations
                except KeyboardInterrupt:
                    print("\nWARNING: KeyboardInterrupt received. Stopping execution...", flush=True)
                    raise
        except KeyboardInterrupt:
            print("\nWARNING: Benchmark interrupted by user. Exiting...", flush=True)
            raise
    else:
        # Parallel execution using ProcessPoolExecutor
        print(f"Running combinations in parallel with {num_workers} workers...\n")
        print(f"NOTE: Each combination runs in a separate Python process for complete isolation.\n")
        print(f"WARNING: With {num_workers} workers, ensure you have sufficient API rate limits.\n")
        print(f"         If you see rate limit errors, reduce --num-workers.\n")
        print(f"TIP: Press Ctrl+C once to gracefully shutdown all processes.\n")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        
        executor = ProcessPoolExecutor(max_workers=num_workers)
        _executor_ref = executor  # Store reference for signal handler
        
        try:
            # Submit all jobs
            future_to_combo = {
                executor.submit(_run_single_combination, args_tuple): (args_tuple[0], args_tuple[1])
                for args_tuple in args_list
            }
            _futures_ref = list(future_to_combo.keys())  # Store reference for signal handler
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_combo):
                # Check if we've been interrupted
                if _interrupted:
                    print("\nWARNING: Interrupt detected. Cancelling remaining tasks...", flush=True)
                    # Cancel all remaining futures
                    for remaining_future in future_to_combo:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
                
                backend, defense = future_to_combo[future]
                completed += 1
                
                try:
                    # Add timeout to prevent hanging (10 minutes per combination should be enough)
                    # This prevents the entire benchmark from hanging if one process gets stuck
                    backend_key, defense_key, result = future.result(timeout=600)
                    
                    # Check for errors in result
                    has_error, has_connection_error, has_rate_limit_error = _check_result_for_errors(
                        result, backend_key, defense_key, summary
                    )
                    
                    # Update error counts for reporting
                    if has_rate_limit_error:
                        rate_limit_count += 1
                    if has_connection_error:
                        api_error_count += 1
                    if has_error:
                        api_error_count += 1
                    
                    # Print result with error indicators
                    status = "OK" if result.get("success") else "ERROR"
                    if result.get("skipped"):
                        status = "SKIPPED"
                    
                    error_indicator = ""
                    if has_rate_limit_error:
                        error_indicator = " [RATE LIMIT]"
                    elif has_connection_error:
                        error_indicator = " [CONNECTION ERROR]"
                    elif has_error:
                        error_indicator = " [ERROR]"
                    
                    print(f"[{completed}/{total_combinations}] {status} {backend_key.upper()} + {defense_key}{error_indicator}", flush=True)
                    
                    all_results[f"{backend_key}_{defense_key}"] = result
                    
                    if result.get("success"):
                        if not result.get("skipped"):
                            summary["combinations"][f"{backend_key}_{defense_key}"] = {
                                "tests_run": result.get("tests_run", 0),
                                "tests_passed": result.get("tests_passed", 0),
                                "tests_failed": result.get("tests_failed", 0),
                                "has_errors": has_error or has_connection_error or has_rate_limit_error
                            }
                            summary["total_tests"] += result.get("tests_run", 0)
                            summary["total_passed"] += result.get("tests_passed", 0)
                            summary["total_failed"] += result.get("tests_failed", 0)
                            summary["successful_combinations"] += 1
                        else:
                            summary["combinations"][f"{backend_key}_{defense_key}"] = {"skipped": True}
                            summary["successful_combinations"] += 1
                    else:
                        error_msg = result.get("error", "Unknown error")
                        print(f"   Error: {error_msg}")
                        summary["combinations"][f"{backend_key}_{defense_key}"] = {
                            "error": error_msg,
                            "tests_run": result.get("tests_run", 0),
                            "has_errors": True
                        }
                        summary["failed_combinations"] += 1
                except FutureTimeoutError:
                    error_str = f"Process timed out after 10 minutes - may be stuck"
                    print(f"[{completed}/{total_combinations}] ERROR: {backend.upper()} + {defense} - TIMEOUT: {error_str}")
                    # Try to cancel the future
                    future.cancel()
                    summary["combinations"][f"{backend}_{defense}"] = {
                        "error": error_str,
                        "tests_run": 0,
                        "has_errors": True
                    }
                    summary["failed_combinations"] += 1
                    if (backend, defense) not in summary["combinations_with_errors"]:
                        summary["combinations_with_errors"].append((backend, defense))
                    # Error is logged to individual log file - no need for shared error.log
                except KeyboardInterrupt:
                    # Re-raise KeyboardInterrupt to propagate it up
                    print(f"\nWARNING: KeyboardInterrupt received. Shutting down...", flush=True)
                    raise
                except Exception as e:
                    error_str = str(e)
                    print(f"[{completed}/{total_combinations}] ERROR: {backend.upper()} + {defense} - Exception: {error_str}")
                    
                    # Track this as a failed combination with error
                    summary["combinations"][f"{backend}_{defense}"] = {
                        "error": error_str,
                        "tests_run": 0,
                        "has_errors": True
                    }
                    summary["failed_combinations"] += 1
                    
                    # Add to error tracking lists
                    if (backend, defense) not in summary["combinations_with_errors"]:
                        summary["combinations_with_errors"].append((backend, defense))
                    
                    # Check if it's a process crash (fork issue on macOS)
                    if "terminated abruptly" in error_str.lower() or "fork" in error_str.lower():
                        # Error is already logged to individual log file - no need for shared error.log
                        pass
            
            # Explicitly shutdown executor to ensure all processes are cleaned up
            # If interrupted, cancel pending futures and don't wait
            if _interrupted:
                print("Terminating executor and all worker processes...", flush=True)
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                # Normal shutdown - wait for all processes to finish
                executor.shutdown(wait=True, cancel_futures=False)
        except KeyboardInterrupt:
            # Ensure executor is shut down even if we catch KeyboardInterrupt
            print("\nWARNING: KeyboardInterrupt caught. Forcing shutdown of all processes...", flush=True)
            executor.shutdown(wait=False, cancel_futures=True)
            _executor_ref = None
            raise  # Re-raise to exit the script
        finally:
            # Clear executor and futures references
            _executor_ref = None
            _futures_ref = None
    
    # Print final summary
    print(f"\n{'='*80}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total combinations: {total_combinations}")
    print(f"Successful: {summary['successful_combinations']}")
    print(f"Failed: {summary['failed_combinations']}")
    if summary["total_tests"] > 0:
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
    
    # Report combinations with errors
    if summary["combinations_with_errors"]:
        print(f"\n{'WARNING' * 40}")
        print("WARNING: COMBINATIONS WITH ERRORS - RESULTS MAY BE INCORRECT")
        print(f"{'WARNING' * 40}")
        print(f"\nThe following {len(summary['combinations_with_errors'])} combination(s) had errors:")
        print("You should rerun these to get reliable results:\n")
        
        for backend, defense in summary["combinations_with_errors"]:
            print(f"  - {backend.upper()} + {defense}")
        
        # Generate rerun command
        failed_backends = sorted(set(b for b, d in summary['combinations_with_errors']))
        failed_defenses = sorted(set(d for b, d in summary['combinations_with_errors']))
        model_arg = f"--model {target_model_name}" if target_model_name else ""
        
        print(f"\nTo rerun only the failed combinations, use:")
        if "suite" in test_path or test_path in ["benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"]:
            print(f"  python scripts/run_benchmark.py --suite {test_path} {model_arg} \\")
        else:
            print(f"  python scripts/run_benchmark.py --test {test_path} {model_arg} \\")
        print(f"    --memory-backend {' '.join(failed_backends)} \\")
        print(f"    --defense-type {' '.join(failed_defenses)} \\")
        print(f"    --num-workers 8 --force")
        print()
    
    # Report specific error types
    if summary["combinations_with_connection_errors"]:
        print(f"WARNING: Connection errors detected in {len(summary['combinations_with_connection_errors'])} combination(s)")
        print(f"   These may have incomplete results due to network issues")
        print(f"   Affected: {', '.join([f'{b}+{d}' for b, d in summary['combinations_with_connection_errors']])}")
        print()
    
    if summary["combinations_with_rate_limit_errors"]:
        print(f"WARNING: Rate limit errors detected in {len(summary['combinations_with_rate_limit_errors'])} combination(s)")
        print(f"   Consider reducing --num-workers or adding delays")
        print(f"   Affected: {', '.join([f'{b}+{d}' for b, d in summary['combinations_with_rate_limit_errors']])}")
        print()
    
    # Report general API issues
    if rate_limit_count > 0 or api_error_count > 0:
        print(f"WARNING: API ISSUES DETECTED:")
        if rate_limit_count > 0:
            print(f"   Rate limit errors: {rate_limit_count}")
        if api_error_count > 0:
            print(f"   API/Connection errors: {api_error_count}")
        print(f"   Check individual log files in data/benchmark/logs/ for details")
        print()
    
    # Final status
    if summary["combinations_with_errors"] or summary["failed_combinations"] > 0:
        print(f"{'ERROR' * 40}")
        print("ERROR: WARNING: Some combinations had EXECUTION ERRORS. Results may be unreliable!")
        print(f"{'ERROR' * 40}")
        print(f"\nWARNING: Remember: Test failures (some tests passing, some failing) are EXPECTED")
        print(f"   and are what we're measuring. Execution errors (API failures, connection")
        print(f"   errors, exceptions) are NOT expected and indicate problems running the benchmark.")
        if summary["combinations_with_errors"]:
            print(f"\nRerun the combinations with execution errors listed above to get correct results.")
        if summary["failed_combinations"] > 0:
            print(f"\n{summary['failed_combinations']} combination(s) failed completely (could not run).")
    else:
        print(f"{'SUCCESS' * 40}")
        print("SUCCESS: All combinations completed without EXECUTION ERRORS. Results are reliable!")
        print(f"{'SUCCESS' * 40}")
        print(f"\nNote: Test failures (some tests passing, some failing) are expected and normal.")
        print(f"      Only execution errors (API failures, connection errors, etc.) are reported here.")
    
    # Note: Detailed errors are in individual log files per combination
    # No need to write to shared error.log file
    
    print()
    
    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for all memory backends and defense types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ALL combinations (default behavior - all backends × all defenses)
  python scripts/run_benchmark.py --suite memory_only --model gpt-5-mini --num-workers 16 --force
  
  # Run specific memory backends with all defenses
  python scripts/run_benchmark.py --memory-backend explicit mem0 --suite memory_only --num-workers 8
  
  # Run all backends with specific defenses
  python scripts/run_benchmark.py --defense-type none user_prompt_only --suite memory_only --num-workers 8
  
  # Run specific backends and defenses
  python scripts/run_benchmark.py --memory-backend explicit mem0 --defense-type none user_prompt_only --suite memory_only --num-workers 4
  
  # Single combination (one backend, one defense)
  python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign
  
  # Run mem0 with all defenses (serial)
  python scripts/run_benchmark.py --memory-backend mem0 --suite benign --num-workers 1
  
  # Run specific test file
  python scripts/run_benchmark.py --memory-backend rag --defense-type user_prompt_only --test data/benchmark/tests/benign/00_email_tools.json --model gpt-5-mini
  
  # Force overwrite existing results
  python scripts/run_benchmark.py --suite memory_only --model gpt-5-mini --force --num-workers 16
        """
    )
    
    parser.add_argument(
        "--memory-backend",
        type=str,
        nargs="+",
        choices=["explicit", "mem0", "rag", "context", "none"],
        help="Memory backend(s) to use. Can specify multiple (e.g., --memory-backend explicit mem0). "
             "If not specified, all backends are used. Use 'none' to disable all memory backends."
    )
    
    parser.add_argument(
        "--defense-type",
        type=str,
        nargs="+",
        choices=UNIFIED_DEFENSE_TYPES,
        help="Defense type(s) to run. Can specify multiple (e.g., --defense-type none user_prompt_only). "
             "If not specified, all defense types are used."
    )
    
    parser.add_argument(
        "--defense",
        type=str,
        nargs="+",
        choices=UNIFIED_DEFENSE_TYPES,
        dest="defense_type",  # Use same dest as --defense-type
        help="Alias for --defense-type. Defense type(s) to run. Can specify multiple (e.g., --defense none user_prompt_only). "
             "If not specified, all defense types are used."
    )
    
    parser.add_argument(
        "--all-defenses",
        action="store_true",
        help="DEPRECATED: Use --defense-type without arguments or omit it to run all defenses. "
             "This flag is kept for backward compatibility but has no effect."
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="Path to specific test file"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        choices=["benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"],
        help="Test suite to run (benign, direct, indirect, memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, or long_memory)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to benchmark config file (default: benchmark_config.yaml)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Base directory for results (defaults to data/benchmark/results)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        dest="target_model_name",
        help="Target model name (e.g., 'gpt-5-mini', 'gpt-4o'). Overrides config value."
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for running multiple combinations (default: 1 = serial). "
             "When multiple backends/defenses are specified or defaults are used, combinations run in parallel."
    )
    
    parser.add_argument(
        "--debug-level",
        type=str,
        choices=["INFO", "DEBUG"],
        default="INFO",
        help="Debug verbosity level: INFO (standard messages) or DEBUG (detailed debug messages). Default: INFO"
    )
    
    args = parser.parse_args()
    
    # Set debug level globally
    from agent.utils import set_debug_level, DebugLevel
    debug_level = DebugLevel.DEBUG if args.debug_level == "DEBUG" else DebugLevel.INFO
    set_debug_level(debug_level)
    
    # Validate arguments
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    
    if not args.test and not args.suite:
        parser.error("Must specify either --test or --suite")
    
    if args.test and args.suite:
        parser.error("--test and --suite are mutually exclusive")
    
    # Determine test path
    test_path = args.test if args.test else args.suite
    
    # Determine results directory
    results_base_dir = Path(args.results_dir) if args.results_dir else None
    
    # Determine if memory backends and defense types were specified
    memory_backends_specified = args.memory_backend is not None
    defense_types_specified = args.defense_type is not None
    
    # Determine memory backends to use
    if memory_backends_specified:
        memory_backends = args.memory_backend
    else:
        # Default: all backends (including "none" as a regular backend)
        memory_backends = ["none", "explicit", "mem0", "rag", "context"]
    
    # Determine defense types to use
    if defense_types_specified:
        defense_types = args.defense_type
    else:
        # Default: all defense types
        defense_types = UNIFIED_DEFENSE_TYPES
    
    # Debug output: show what will be run
    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Memory backends: {memory_backends}")
    print(f"  Defense types: {defense_types}")
    print(f"  Total combinations: {len(memory_backends) * len(defense_types)}")
    print(f"{'='*80}\n")
    
    # Determine if we're running multiple combinations
    # Multiple combinations if:
    # - Multiple backends (specified or defaulted), OR
    # - Multiple defenses (specified or defaulted), OR
    # - Neither specified (defaults to all = multiple)
    # Single combination only if BOTH are single values
    total_combinations = len(memory_backends) * len(defense_types)
    
    if total_combinations > 1:
        # Multiple combinations - use parallel execution
        running_multiple_combinations = True
    else:
        # Single combination mode (1 backend × 1 defense = 1 combination)
        running_multiple_combinations = False
    
    # Set debug level in config for TestBench
    # This ensures debug level is available in all subprocesses
    import os
    os.environ["DEBUG_LEVEL"] = args.debug_level
    
    # Run benchmark(s)
    if running_multiple_combinations:
        # Multiple combinations mode - use parallel execution
        # Each combination writes to its own log file - no shared error.log needed
        result = run_all_combinations(
            memory_backends=memory_backends,
            defense_types=defense_types,
            test_path=test_path,
            config_path=args.config,
            force=args.force,
            num_workers=args.num_workers,
            results_base_dir=results_base_dir,
            target_model_name=args.target_model_name
        )
    else:
        # Single combination mode
        memory_backend = memory_backends[0]
        
        # Use provided defense type or default to "none"
        # Treat "none" backend as a regular backend - allow all defense types
        defense_type = defense_types[0] if defense_types_specified else "none"
        
        # Set debug level in environment for subprocesses
        import os
        os.environ["DEBUG_LEVEL"] = args.debug_level
        
        result = run_benchmark(
            memory_backend=memory_backend,
            unified_defense=defense_type,
            test_path=test_path,
            config_path=args.config,
            force=args.force,
            results_base_dir=results_base_dir,
            target_model_name=args.target_model_name
        )
    
    # Exit with appropriate code
    if result.get("success") and result.get("tests_failed", 0) == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    # Set start method for multiprocessing (required on some platforms)
    # On macOS, 'fork' causes crashes with Objective-C runtime (objc[PID]: fork() errors)
    # Use 'spawn' on macOS for safety, 'fork' on Linux for speed
    import platform
    if platform.system() == "Darwin":  # macOS
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, ignore
            pass
    elif platform.system() != "Windows":  # Linux/Unix
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            # Already set, ignore
            pass
    # Windows defaults to 'spawn' automatically
    
    main()
