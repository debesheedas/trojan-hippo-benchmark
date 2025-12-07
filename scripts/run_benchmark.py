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

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.utils import load_config
from benchmark.defense_backend import get_defense_backend_registry, UNIFIED_DEFENSE_TYPES
from benchmark.benchmark_utils import (
    get_result_path,
    should_skip_test,
    get_memory_backend_from_config,
    get_unified_defense_from_config
)
from benchmark.test_bench import TestBench


def get_all_test_files(test_path: str, test_dir: Path) -> List[Path]:
    """
    Get all test files from a path.
    
    Args:
        test_path: Path string (file, directory, or suite name)
        test_dir: Base test directory
        
    Returns:
        List of test file paths
    """
    # Handle suite keywords
    if test_path in {"benign", "direct", "indirect"}:
        test_path_obj = test_dir / test_path
    else:
        test_path_obj = Path(test_path)
    
    # Fallback to old structure if not found
    if not test_path_obj.exists():
        if test_path in {"benign", "direct", "indirect"}:
            old_paths = [
                Path(f"data/benchmark/attack_bench_explicit/{test_path}"),
                Path(f"data/benchmark/attack_bench_mem0/{test_path}"),
                Path(f"data/benchmark/attack_bench_rag/{test_path}"),
            ]
            for old_path in old_paths:
                if old_path.exists():
                    test_path_obj = old_path
                    break
    
    if not test_path_obj.exists():
        return []
    
    if test_path_obj.is_file():
        return [test_path_obj] if test_path_obj.suffix.lower() == '.json' else []
    
    # Directory - find all JSON files recursively
    return sorted(test_path_obj.rglob("*.json"))


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
        # Get attack type from test file
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            attack_type = test_data.get("attack_type", "benign")
        except Exception:
            attack_type = "benign"
        
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
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none" for disable_memory)
        unified_defense: Unified defense name (e.g., "none", "user_only")
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
    
    # Load config
    config = load_config(config_path)
    
    # Set target model name (command line arg takes precedence over config)
    if target_model_name:
        if "agent" not in config:
            config["agent"] = {}
        config["agent"]["target_model_name"] = target_model_name
    
    # Handle no memory backend: disable all backends
    if memory_backend == "none":
        if "memory" not in config:
            config["memory"] = {}
        # Disable all memory backends
        for backend_name in ["explicit", "mem0", "rag"]:
            if backend_name not in config["memory"]:
                config["memory"][backend_name] = {}
            config["memory"][backend_name]["enabled"] = False
        # Set backend to "none" in config
        config["memory"]["backend"] = "none"
    else:
        # Set memory backend in config
        if "memory" not in config:
            config["memory"] = {}
        config["memory"]["backend"] = memory_backend
        
        # Enable the specified backend and disable others
        for backend_name in ["explicit", "mem0", "rag"]:
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
    test_files = get_all_test_files(test_path, test_dir)
    if not test_files:
        print(f"⚠️  No test files found for path: {test_path}")
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
        print(f"✅ All results already exist for {memory_backend}/{unified_defense}")
        print(f"   Use --force to overwrite")
        return {
            "success": True,
            "skipped": True,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    
    if missing and not force:
        print(f"⏭️  {len(missing)} test(s) already have results, {len(test_files) - len(missing)} will run")
    
    # Initialize TestBench with config dict directly (no temp file needed)
    bench = TestBench(config=config, defense_type_override=unified_defense, force=force)
    
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
            status = "⏭️  SKIPPED"
        elif stats.get("tests_passed", 0) == stats.get("tests_run", 0):
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{defense_type:20s}: {status}", end="")
        if not stats.get("skipped"):
            print(f" ({stats.get('tests_passed', 0)}/{stats.get('tests_run', 0)} tests)")
        else:
            print()
    
    if summary["total_tests"] > 0:
        print(f"\nTotal: {summary['total_passed']}/{summary['total_tests']} tests passed")
    
    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for all memory backends and defense types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run explicit memory with no defense
  python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign
  
  # Run mem0 with all defenses
  python scripts/run_benchmark.py --memory-backend mem0 --all-defenses --suite benign
  
  # Run specific test file with custom model
  python scripts/run_benchmark.py --memory-backend rag --defense-type user_only --test data/benchmark/tests/benign/00_email_tools.json --model gpt-5-mini
  
  # Force overwrite existing results
  python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign --force --model gpt-5-mini
        """
    )
    
    parser.add_argument(
        "--memory-backend",
        type=str,
        choices=["explicit", "mem0", "rag", "none"],
        required=True,
        help="Memory backend to use (use 'none' to disable all memory backends)"
    )
    
    parser.add_argument(
        "--defense-type",
        type=str,
        choices=UNIFIED_DEFENSE_TYPES,
        help="Specific defense type to run (mutually exclusive with --all-defenses). Not used when --memory-backend is 'none'."
    )
    
    parser.add_argument(
        "--all-defenses",
        action="store_true",
        help="Run all defense types (mutually exclusive with --defense-type)"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="Path to specific test file"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        choices=["benign", "direct", "indirect"],
        help="Test suite to run (benign, direct, or indirect)"
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.defense_type and args.all_defenses:
        parser.error("--defense-type and --all-defenses are mutually exclusive")
    
    if not args.test and not args.suite:
        parser.error("Must specify either --test or --suite")
    
    if args.test and args.suite:
        parser.error("--test and --suite are mutually exclusive")
    
    # Determine test path
    test_path = args.test if args.test else args.suite
    
    # Determine results directory
    results_base_dir = Path(args.results_dir) if args.results_dir else None
    
    # Set memory backend from args
    memory_backend = args.memory_backend
    
    # If backend is "none", automatically set defense to "none" (no memory = no defense)
    if memory_backend == "none":
        if args.defense_type and args.defense_type != "none":
            parser.error("When using --memory-backend none, --defense-type must be 'none' (or omitted)")
        defense_type = "none"
    else:
        # For other backends, use provided defense type or default to "none"
        defense_type = args.defense_type or "none"
    
    # Run benchmark(s)
    if args.all_defenses:
        result = run_all_defenses(
            memory_backend=memory_backend,
            test_path=test_path,
            config_path=args.config,
            force=args.force,
            target_model_name=args.target_model_name
        )
    else:
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
    main()
