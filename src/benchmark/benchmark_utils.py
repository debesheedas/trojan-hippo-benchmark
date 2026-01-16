"""
Benchmark Utilities

Helper functions for result path generation, caching, and test isolation.
"""

import os
import time
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from benchmark.defense_backend import get_defense_backend_registry


def _get_result_path_components(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str
) -> tuple[str, str]:
    """
    Get the backend and defense folder names for result path construction.
    
    This is shared logic used by both get_result_path() and get_results_dir().
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none")
        unified_defense: Unified defense name (e.g., "none", "disable_memory")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type ("benign", "direct", "indirect")
        
    Returns:
        Tuple of (backend_for_path, defense_folder)
    """
    # Treat "none" memory backend like any other backend - use the actual defense type
    backend_for_path = memory_backend
    
    # Map unified defense to backend-specific defense type
    # For "none" backend, defenses still work (e.g., provable_policy), so we need to map them
    # Since "none" backend doesn't have a registered defense backend, we'll use the unified name directly
    if memory_backend == "none":
        # For "none" backend, use unified defense name directly (defenses like provable_policy work without memory)
        defense_folder = unified_defense
    else:
        defense_registry = get_defense_backend_registry()
        backend_defense = defense_registry.map_defense(memory_backend, unified_defense)
        
        # Normalize defense folder name to unified name for consistency
        # All backends should use "none" for no defense, regardless of backend-specific name
        # (mem0 internally uses "no_defense", but folder should be "none" for consistency)
        if unified_defense == "none" or backend_defense in ["none", "no_defense"]:
            defense_folder = "none"
        else:
            defense_folder = backend_defense
    
    return backend_for_path, defense_folder


def get_result_path(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    test_file: Path,
    results_base_dir: Path = Path("data/benchmark/results")
) -> Path:
    """
    Generate result file path using unified structure.
    
    Path structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/{test_file}.json
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none" for disable_memory)
        unified_defense: Unified defense name (e.g., "none", "disable_memory")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type ("benign", "direct", "indirect")
        test_file: Path to test file
        results_base_dir: Base directory for results
        
    Returns:
        Path to result file
    """
    backend_for_path, defense_folder = _get_result_path_components(
        memory_backend, unified_defense, model_name, attack_type
    )
    
    # Construct path: results/{model_name}/{memory_backend}/{defense_folder}/{attack_type}/{test_file_name}.json
    result_path = (
        results_base_dir /
        model_name /
        backend_for_path /
        defense_folder /
        attack_type /
        test_file.name
    )
    
    return result_path


def get_log_path(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    test_file: Path,
    logs_base_dir: Path = Path("data/benchmark/logs")
) -> Path:
    """
    Generate log file path using the same structure as results.
    
    Path structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/{test_file}.log
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none" for disable_memory)
        unified_defense: Unified defense name (e.g., "none", "disable_memory")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type ("benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools")
        test_file: Path to test file
        logs_base_dir: Base directory for logs
        
    Returns:
        Path to log file
    """
    backend_for_path, defense_folder = _get_result_path_components(
        memory_backend, unified_defense, model_name, attack_type
    )
    
    # Construct path: logs/{model_name}/{memory_backend}/{defense_folder}/{attack_type}/{test_file_name}.log
    log_path = (
        logs_base_dir /
        model_name /
        backend_for_path /
        defense_folder /
        attack_type /
        test_file.with_suffix('.log').name
    )
    
    return log_path


def get_combination_log_path(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    logs_base_dir: Path = Path("data/benchmark/logs")
) -> Path:
    """
    Generate log file path for an entire combination (all tests in a suite).
    
    Path structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/combination.log
    
    Args:
        memory_backend: Memory backend name
        unified_defense: Unified defense name
        model_name: Model name
        attack_type: Attack type ("benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools")
        logs_base_dir: Base directory for logs
        
    Returns:
        Path to combination log file
    """
    backend_for_path, defense_folder = _get_result_path_components(
        memory_backend, unified_defense, model_name, attack_type
    )
    
    # Construct path: logs/{model_name}/{memory_backend}/{defense_folder}/{attack_type}/combination.log
    log_path = (
        logs_base_dir /
        model_name /
        backend_for_path /
        defense_folder /
        attack_type /
        "combination.log"
    )
    
    return log_path


def get_results_dir(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    results_base_dir: Path = Path("data/benchmark/results")
) -> Path:
    """
    Get the results directory for a memory backend, defense type, model, and attack type.
    
    Path structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/
    
    This is the directory path (without filename) used by consolidate_results.py.
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none")
        unified_defense: Unified defense name (e.g., "none", "disable_memory")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type ("benign", "direct", "indirect")
        results_base_dir: Base directory for results
        
    Returns:
        Path to results directory
    """
    backend_for_path, defense_folder = _get_result_path_components(
        memory_backend, unified_defense, model_name, attack_type
    )
    
    return results_base_dir / model_name / backend_for_path / defense_folder / attack_type


def check_result_exists(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    test_file: Path,
    results_base_dir: Path = Path("data/benchmark/results")
) -> bool:
    """
    Check if a result file already exists.
    
    Args:
        memory_backend: Memory backend name
        unified_defense: Unified defense name
        model_name: Model name
        attack_type: Attack type
        test_file: Path to test file
        results_base_dir: Base directory for results
        
    Returns:
        True if result exists, False otherwise
    """
    result_path = get_result_path(
        memory_backend,
        unified_defense,
        model_name,
        attack_type,
        test_file,
        results_base_dir
    )
    return result_path.exists()


def should_skip_test(
    memory_backend: str,
    unified_defense: str,
    model_name: str,
    attack_type: str,
    test_file: Path,
    force: bool,
    results_base_dir: Path = Path("data/benchmark/results")
) -> bool:
    """
    Determine if a test should be skipped (result already exists and force=False).
    
    Args:
        memory_backend: Memory backend name
        unified_defense: Unified defense name
        model_name: Model name
        attack_type: Attack type
        test_file: Path to test file
        force: If True, don't skip even if result exists
        results_base_dir: Base directory for results
        
    Returns:
        True if test should be skipped, False otherwise
    """
    if force:
        return False
    
    return check_result_exists(
        memory_backend,
        unified_defense,
        model_name,
        attack_type,
        test_file,
        results_base_dir
    )


def create_isolated_test_dir(test_name: str, base_dir: Path = Path("data/benchmark/test_envs")) -> Path:
    """
    Create an isolated test directory with unique name.
    
    Uses process ID + timestamp for uniqueness to avoid race conditions
    in parallel execution.
    
    Args:
        test_name: Test name (for readability)
        base_dir: Base directory for test environments
        
    Returns:
        Path to isolated test directory
    """
    # Generate unique identifier
    pid = os.getpid()
    timestamp = int(time.time() * 1000000)  # microseconds
    unique_id = f"{test_name}_{pid}_{timestamp}"
    
    test_dir = base_dir / unique_id
    test_dir.mkdir(parents=True, exist_ok=True)
    
    return test_dir


def get_memory_backend_from_config(config: Dict[str, Any]) -> str:
    """
    Determine memory backend from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Memory backend name ("explicit", "mem0", "rag", "context", or "none" if all disabled)
    """
    memory_config = config.get("memory", {})
    
    # Check for explicit backend setting
    backend = memory_config.get("backend")
    if backend:
        # If backend is explicitly set to "none", return it
        if backend == "none":
            return "none"
        return backend
    
    # Fallback: detect from enabled flags
    if memory_config.get("explicit_memory", {}).get("enabled", False):
        return "explicit"
    elif memory_config.get("mem0_memory", {}).get("enabled", False):
        return "mem0"
    elif memory_config.get("rag_memory", {}).get("enabled", False):
        return "rag"
    elif memory_config.get("context_memory", {}).get("enabled", False):
        return "context"
    
    # Check if all backends are explicitly disabled
    explicit_disabled = memory_config.get("explicit_memory", {}).get("enabled") is False
    mem0_disabled = memory_config.get("mem0_memory", {}).get("enabled") is False
    rag_disabled = memory_config.get("rag_memory", {}).get("enabled") is False
    context_disabled = memory_config.get("context_memory", {}).get("enabled") is False
    
    if explicit_disabled and mem0_disabled and rag_disabled and context_disabled:
        return "none"
    
    # Default to explicit if nothing is explicitly set
    return "explicit"


def get_unified_defense_from_config(config: Dict[str, Any], memory_backend: str) -> str:
    """
    Get unified defense type from config.
    
    Args:
        config: Configuration dictionary
        memory_backend: Memory backend name
        
    Returns:
        Unified defense name
    """
    memory_config = config.get("memory", {})
    
    # Get defense from backend-specific config
    if memory_backend == "none":
        # For "none" backend, defense_type is stored at top level of memory config
        defense = memory_config.get("defense_type", "none")
    elif memory_backend == "explicit":
        defense = memory_config.get("explicit_memory", {}).get("defense_type", "none")
    elif memory_backend == "mem0":
        defense = memory_config.get("mem0_memory", {}).get("defense_type", "none")
    elif memory_backend == "rag":
        defense = memory_config.get("rag_memory", {}).get("defense_type", "none")
    elif memory_backend == "context":
        defense = memory_config.get("context_memory", {}).get("defense_type", "none")
    else:
        defense = "none"
    
    # Normalize: mem0 uses "no_defense" but we want unified "none"
    if memory_backend == "mem0" and defense == "no_defense":
        return "none"
    
    return defense


def cleanup_old_test_environments(
    base_dir: Path = Path("data/benchmark/test_envs"),
    max_age_hours: int = 24,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Clean up old test environment directories that are no longer needed.
    
    Test environments are created for each test run and should be cleaned up
    immediately after the test completes. This function removes any that were
    left behind (e.g., from crashed processes or interrupted runs).
    
    Args:
        base_dir: Base directory containing test environments
        max_age_hours: Maximum age in hours before cleanup (default: 24 hours)
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Tuple of (deleted_count, failed_count)
    """
    if not base_dir.exists():
        return 0, 0
    
    import time
    from datetime import datetime, timedelta
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    failed_count = 0
    
    # Get all test environment directories
    test_envs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not test_envs:
        return 0, 0
    
    print(f"\n🧹 Checking {len(test_envs)} test environment(s) for cleanup...")
    
    for test_env in test_envs:
        try:
            # Get directory modification time (when it was last modified)
            # This is a good proxy for when the test finished
            mtime = test_env.stat().st_mtime
            age_seconds = current_time - mtime
            age_hours = age_seconds / 3600
            
            if age_seconds > max_age_seconds:
                if dry_run:
                    print(f"  [DRY RUN] Would delete: {test_env.name} (age: {age_hours:.1f} hours)")
                    deleted_count += 1
                else:
                    try:
                        shutil.rmtree(test_env)
                        print(f"  Deleted: {test_env.name} (age: {age_hours:.1f} hours)")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  WARNING: Failed to delete {test_env.name}: {e}")
                        failed_count += 1
        except Exception as e:
            print(f"  WARNING: Error checking {test_env.name}: {e}")
            failed_count += 1
    
    if not dry_run and deleted_count > 0:
        print(f"Cleaned up {deleted_count} old test environment(s)")
    elif dry_run:
        print(f"  [DRY RUN] Would clean up {deleted_count} old test environment(s)")
    
    if failed_count > 0:
        print(f"WARNING: Failed to clean up {failed_count} test environment(s)")
    
    return deleted_count, failed_count


def discover_test_files(
    test_path: str,
    test_dir: Path,
    verbose: bool = False
) -> List[Path]:
    """
    Intelligently discover test files from a path.
    
    Handles:
    - Suite keywords (benign, direct, indirect, memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, long_memory) -> maps to test_dir/{suite}/
    - File paths -> returns single file if JSON
    - Directory paths -> finds all JSON files recursively
    
    Uses unified test directory structure: data/benchmark/tests/{suite}/
    
    Args:
        test_path: Path string (file, directory, or suite name)
        test_dir: Base test directory (e.g., Path("data/benchmark/tests"))
        verbose: If True, print warnings when paths not found
        
    Returns:
        List of test file paths (sorted)
    """
    
    # Handle suite keywords
    suite_keywords = {"benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"}
    if test_path in suite_keywords:
        test_path_obj = test_dir / test_path
    else:
        test_path_obj = Path(test_path)
    
    # Check if path exists
    if not test_path_obj.exists():
        if verbose:
            print(f"WARNING: Test path not found: {test_path}")
            if test_path in suite_keywords:
                print(f"   Expected location: {test_path_obj}")
                print(f"   Unified test directory: {test_dir}")
        return []
    
    # Handle file path
    if test_path_obj.is_file():
        if test_path_obj.suffix.lower() == '.json':
            return [test_path_obj]
        else:
            if verbose:
                print(f"File is not a JSON test file: {test_path}")
            return []
    
    # Directory - find all JSON files recursively
    test_files = list(test_path_obj.rglob("*.json"))
    if not test_files and verbose:
        print(f"No JSON test files found in {test_path}")
    
    return sorted(test_files)


def determine_attack_type(test_file: Path, test_def: Optional[Dict[str, Any]] = None) -> str:
    """
    Determine the attack_type for a test file, handling memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, and long_memory tests specially.

    Memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, and long_memory tests should be saved to their respective folders even if they have
    attack_type="benign" in their JSON. This function detects these tests by:
    1. Checking if "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", or "long_memory" is in the test file path
    2. Checking if the filename starts with "memory_only_", "assistant_responses_", "untrusted_probe_", "untrusted_send_", "disable_send_", "memory_tools_", or "long_memory_"
    
    Args:
        test_file: Path to test file
        test_def: Optional test definition dict (if already loaded)
        
    Returns:
        Attack type string ("benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", or "long_memory")
    """
    # Check if this is a memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, or long_memory test by path or filename
    test_file_str = str(test_file)
    if "memory_only" in test_file_str or test_file.name.startswith("memory_only_"):
        return "memory_only"
    if "assistant_responses" in test_file_str or test_file.name.startswith("assistant_responses_"):
        return "assistant_responses"
    if "untrusted_probe" in test_file_str or test_file.name.startswith("untrusted_probe_"):
        return "untrusted_probe"
    if "untrusted_send" in test_file_str or test_file.name.startswith("untrusted_send_"):
        return "untrusted_send"
    if "disable_send" in test_file_str or test_file.name.startswith("disable_send_"):
        return "disable_send"
    if "memory_tools" in test_file_str or test_file.name.startswith("memory_tools_"):
        return "memory_tools"
    if "long_memory" in test_file_str or test_file.name.startswith("long_memory_"):
        return "long_memory"
    
    # Otherwise, use attack_type from test definition if available
    if test_def:
        return test_def.get("attack_type", "benign")
    
    # Fallback: try to read from file
    try:
        import json
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        return test_data.get("attack_type", "benign")
    except Exception:
        return "benign"

