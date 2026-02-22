"""
Benchmark Utilities

Helper functions for result path generation, caching, and test isolation.
"""

import os
import time
import hashlib
import shutil
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
from agent.utils import load_config, detect_provider

# Hardcoded test directory (unified test location)
# This is a project structure constant - not configurable via config files
TEST_DIR = Path("data/benchmark/tests")

# Unified defense names (used in config and CLI)
# Note: To disable memory, use memory_backend="none" (no defense type needed)
# When memory_backend="none", defense_type is automatically "none"
UNIFIED_DEFENSE_TYPES = [
    "none",
    "user_prompt_only",
    "no_untrusted_tools",
    "limit_memory_length",
    "provable_policy",
]

# Memory backends (used by benchmark runner, CI, and consolidation scripts)
MEMORY_BACKENDS = ["none", "explicit", "mem0", "rag", "context"]


def is_valid_combination(memory_backend: str, unified_defense: str) -> bool:
    """
    Check if a memory backend and defense type combination is valid.
    
    Invalid combinations:
    - context + limit_memory_length (not applicable for context backend)
    - explicit + user_prompt_only (not applicable for explicit backend)
    
    Args:
        memory_backend: Memory backend name
        unified_defense: Unified defense name
        
    Returns:
        True if combination is valid, False otherwise
    """
    # Skip limit_memory_length for context backend (not applicable)
    if memory_backend == "context" and unified_defense == "limit_memory_length":
        return False
    
    # Skip user_prompt_only for explicit backend (not applicable)
    if memory_backend == "explicit" and unified_defense == "user_prompt_only":
        return False
    
    return True


def ensure_email_unread(email: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deep copy of the email with metadata.read = False.
    Use whenever adding an attack email to the mailbox so that
    "read unread emails" steps see it regardless of prior state.
    """
    out = copy.deepcopy(email)
    if "metadata" not in out:
        out["metadata"] = {}
    out["metadata"]["read"] = False
    return out


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
        memory_backend: Memory backend name ("explicit", "mem0", "rag", "context", or "none")
        unified_defense: Unified defense name (e.g., "none", "user_prompt_only")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type (utility suite name or attack_type from test_def for attack_bench)
        
    Returns:
        Tuple of (backend_for_path, defense_folder)
    """
    # Use unified defense name directly - no mapping needed
    return memory_backend, unified_defense


# Base segment for attack_bench in paths (used to detect and parse suite subfolder)
ATTACK_BENCH_SEGMENT = "attack_bench"
# Subfolders under attack_bench: train (train cases), test (test cases), train_cache (cached optimized train)
ATTACK_BENCH_TRAIN = "train"
ATTACK_BENCH_TEST = "test"
ATTACK_BENCH_TRAIN_CACHE = "train_cache"


def get_attack_bench_suite_subfolder(test_file: Path) -> Optional[str]:
    """
    If the test file lives under attack_bench/.../<suite>/file.json, return <suite> (the test case suite name).
    Supports:
      - attack_bench/<backend>/<suite>/file.json (legacy flat)
      - attack_bench/<train|test>/<backend>/<suite>/file.json (new layout)
    Returns None if path is flat (file directly under backend).
    """
    parts = test_file.resolve().parts
    if ATTACK_BENCH_SEGMENT not in parts:
        return None
    idx = parts.index(ATTACK_BENCH_SEGMENT)
    rest = parts[idx + 1:]  # after "attack_bench": [train|test|backend, ...]
    if len(rest) <= 1:
        return None  # file directly under backend (flat)
    # New layout: attack_bench/train|test/backend/suite/file -> rest[0]=train|test, rest[2]=suite
    if rest[0] in (ATTACK_BENCH_TRAIN, ATTACK_BENCH_TEST) and len(rest) >= 4:
        return rest[2]  # suite name
    # Legacy: attack_bench/backend/suite/file -> rest = [backend, suite, filename]
    if len(rest) >= 2:
        return rest[1]  # suite name
    return None


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
    For attack_bench tests: {model_name}/{memory_backend}/{defense_type}/{test_file}.json (no attack_type folder)
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none" for disable_memory)
        unified_defense: Unified defense name (e.g., "none", "disable_memory")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type (utility suite name or attack_type from test_def for attack_bench)
        test_file: Path to test file
        results_base_dir: Base directory for results
        
    Returns:
        Path to result file
    """
    backend_for_path, defense_folder = _get_result_path_components(
        memory_backend, unified_defense, model_name, attack_type
    )
    
    # Check if test is in attack_bench (skip attack_type folder for attack_bench tests)
    # Also check if results_base_dir is attack_results (never use attack_type folder for attack_results)
    test_file_str = str(test_file)
    results_base_dir_str = str(results_base_dir)
    is_attack_bench = "attack_bench" in test_file_str
    is_attack_results = "attack_results" in results_base_dir_str
    
    if is_attack_bench or is_attack_results:
        # For attack_bench tests or attack_results: add optional suite subfolder when test is under attack_bench/<backend>/<suite>/
        suite_subfolder = get_attack_bench_suite_subfolder(test_file)
        if suite_subfolder:
            result_path = (
                results_base_dir /
                model_name /
                backend_for_path /
                defense_folder /
                suite_subfolder /
                test_file.name
            )
        else:
            result_path = (
                results_base_dir /
                model_name /
                backend_for_path /
                defense_folder /
                test_file.name
            )
    else:
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
    attack_type: Optional[str],
    test_file: Path,
    logs_base_dir: Path = Path("data/benchmark/logs")
) -> Path:
    """
    Generate log file path using the same structure as results.
    
    Path structure: {model_name}/{memory_backend}/{defense_type}/{attack_type}/{test_file}.log
    For attack_bench tests: {model_name}/{memory_backend}/{defense_type}/{test_file}.log (no attack_type folder)
    
    Note: For attack_bench tests, logs go to attack_logs/ instead of logs/ (similar to attack_results vs results).
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", or "none" for disable_memory)
        unified_defense: Unified defense name (e.g., "none", "disable_memory")
        model_name: Model name (e.g., "gpt-5-mini")
        attack_type: Attack type ("benign", "direct", "indirect", "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", or None/unknown)
        test_file: Path to test file
        logs_base_dir: Base directory for logs
        
    Returns:
        Path to log file
    """
    # Handle None attack_type by determining it from test_file
    if attack_type is None:
        attack_type = determine_attack_type(test_file)
    
    backend_for_path, defense_folder = _get_result_path_components(
        memory_backend, unified_defense, model_name, attack_type
    )
    
    # Check if test is in attack_bench (skip attack_type folder for attack_bench tests)
    # Also check if logs_base_dir is attack_logs (never use attack_type folder for attack_logs)
    test_file_str = str(test_file)
    logs_base_dir_str = str(logs_base_dir)
    is_attack_bench = "attack_bench" in test_file_str
    is_attack_logs = "attack_logs" in logs_base_dir_str
    
    if is_attack_bench or is_attack_logs:
        # For attack_bench tests: add optional suite subfolder when test is under attack_bench/<backend>/<suite>/
        suite_subfolder = get_attack_bench_suite_subfolder(test_file)
        if suite_subfolder:
            log_path = (
                logs_base_dir /
                model_name /
                backend_for_path /
                defense_folder /
                suite_subfolder /
                test_file.with_suffix('.log').name
            )
        else:
            log_path = (
                logs_base_dir /
                model_name /
                backend_for_path /
                defense_folder /
                test_file.with_suffix('.log').name
            )
    else:
        # Construct path: logs/{model_name}/{memory_backend}/{defense_folder}/{attack_type}/{test_file_name}.log
        # Use "unknown" if attack_type is still None or empty
        attack_type_folder = attack_type if attack_type else "unknown"
        log_path = (
            logs_base_dir /
            model_name /
            backend_for_path /
            defense_folder /
            attack_type_folder /
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
        attack_type: Attack type (utility suite name or attack_type from test_def for attack_bench)
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
        attack_type: Attack type (utility suite name or attack_type from test_def for attack_bench)
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


def get_memory_backend_from_config(config: Dict[str, Any]) -> str:
    """
    Get memory backend from config.
    
    The backend is always set by get_run_config() based on CLI --memory-backend arg.
    
    Args:
        config: Configuration dictionary (with memory.backend set by get_run_config)
        
    Returns:
        Memory backend name ("explicit", "mem0", "rag", "context", or "none")
    """
    return config.get("memory", {}).get("backend", "explicit")


def get_unified_defense_from_config(config: Dict[str, Any], memory_backend: str) -> str:
    """
    Get unified defense type from config.
    
    The defense type is always set by get_run_config() based on CLI --defense-type arg.
    
    Args:
        config: Configuration dictionary (with defense_type set by get_run_config)
        memory_backend: Memory backend name
        
    Returns:
        Unified defense name
    """
    memory_config = config.get("memory", {})
    
    # Backend-specific config key mapping
    backend_config_key = {
        "none": None,  # defense_type stored at top level for "none" backend
        "explicit": "explicit_memory",
        "mem0": "mem0_memory",
        "rag": "rag_memory",
        "context": "context_memory",
    }.get(memory_backend)
    
    # Get defense from appropriate location
    if backend_config_key is None:
        defense = memory_config.get("defense_type", "none")
    else:
        defense = memory_config.get(backend_config_key, {}).get("defense_type", "none")
    
    return defense


def discover_test_files(
    test_path: str,
    test_dir: Optional[Path] = None,
    verbose: bool = False
) -> List[Path]:
    """
    Intelligently discover test files from a path.
    
    Handles:
    - Suite keywords (memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, long_memory) -> maps to TEST_DIR/{suite}/
    - File paths -> returns single file if JSON
    - Directory paths -> finds all JSON files recursively
    
    Uses unified test directory structure: data/benchmark/tests/{suite}/
    For attack bench tests, use data/benchmark/attack_bench/ directly (no suites).
    
    Args:
        test_path: Path string (file, directory, or suite name)
        test_dir: Base test directory (defaults to TEST_DIR constant)
        verbose: If True, print warnings when paths not found
        
    Returns:
        List of test file paths (sorted)
    """
    if test_dir is None:
        test_dir = TEST_DIR
    
    # Handle suite keywords
    suite_keywords = {"memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"}
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
    Determine the attack_type for a test file.
    
    For utility tests (in data/benchmark/tests/), detects suite by path/filename:
    1. Checking if "memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", or "long_memory" is in the test file path
    2. Checking if the filename starts with "memory_only_", "assistant_responses_", "untrusted_probe_", "untrusted_send_", "disable_send_", "memory_tools_", or "long_memory_"
    
    Args:
        test_file: Path to test file
        test_def: Optional test definition dict (if already loaded)
        
    Returns:
        Attack type string ("memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory", or from test_def if available)
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
    
    # Valid utility suite types (used for path construction and validation)
    valid_utility_suites = {"memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"}
    
    # Check if this is an attack_bench test (no suite structure)
    if "attack_bench" in test_file_str:
        # Attack bench tests don't have suites - return attack_type from test_def or "indirect" as default
        if test_def:
            return test_def.get("attack_type", "indirect")
        # Fallback: try to read from file
        try:
            import json
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            return test_data.get("attack_type", "indirect")
        except Exception:
            return "indirect"
    
    # Otherwise, use attack_type from test definition if available
    if test_def:
        attack_type = test_def.get("attack_type")
        if attack_type in valid_utility_suites:
            return attack_type
        # Invalid or missing attack_type: return "unknown" for path construction
        return "unknown"
    
    # Fallback: try to read from file
    try:
        import json
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        attack_type = test_data.get("attack_type")
        if attack_type in valid_utility_suites:
            return attack_type
        return "unknown"
    except Exception:
        return "unknown"


def prepare_benchmark_config(
    memory_backend: str,
    unified_defense: str,
    config_path: str = "benchmark_config.yaml",
    target_model_name: Optional[str] = None,
    results_base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Prepare configuration dictionary for a benchmark run.
    
    This function handles all the config manipulation needed to set up a benchmark:
    - Loads and merges config files
    - Sets memory backend and defense type
    - Configures results directory
    
    Args:
        memory_backend: Memory backend name ("explicit", "mem0", "rag", "context", or "none")
        unified_defense: Unified defense name (e.g., "none", "user_prompt_only")
        config_path: Path to benchmark config file
        target_model_name: Optional model name override
        results_base_dir: Optional results directory override
        
    Returns:
        Prepared configuration dictionary
        
    Raises:
        ValueError: If agent configuration is missing
    """
    # Load benchmark config and make a deep copy to avoid modifying the original
    # This is important when multiple processes might be running in parallel
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
    # When overriding model, also set provider so the correct API is used (e.g. gpt-4o-mini -> OpenAI, gemini-* -> Gemini)
    if target_model_name:
        if "agent" not in config:
            config["agent"] = {}
        config["agent"]["target_model_name"] = target_model_name
        config["agent"]["provider"] = detect_provider(target_model_name)
    elif "agent" not in config or "target_model_name" not in config.get("agent", {}):
        # If no agent config was loaded and no target_model_name provided, raise error
        raise ValueError(
            "Agent configuration missing. Please ensure agent_config.yaml exists with an 'agent' section, "
            "or provide target_model_name argument to specify target_model_name."
        )
    
    # Set memory backend and defense type in config
    if "memory" not in config:
        config["memory"] = {}
    config["memory"]["backend"] = memory_backend
    
    # Set defense type in the appropriate location
    if memory_backend == "none":
        # Store defense_type at top level for "none" backend (needed for provable_policy defense)
        config["memory"]["defense_type"] = unified_defense
    else:
        # Map unified defense to backend-specific defense and store in backend config
        backend_config_key = {
            "explicit": "explicit_memory",
            "mem0": "mem0_memory",
            "rag": "rag_memory",
            "context": "context_memory",
        }.get(memory_backend, f"{memory_backend}_memory")
        
        # Ensure backend-specific config exists
        if backend_config_key not in config["memory"]:
            config["memory"][backend_config_key] = {}
        elif config["memory"][backend_config_key] is None:
            config["memory"][backend_config_key] = {}
        
        config["memory"][backend_config_key]["defense_type"] = unified_defense
    
    # Set results directory (command line arg takes precedence, default if not provided)
    if results_base_dir is None:
        results_base_dir = Path("data/benchmark/results")  # Hardcoded default
    
    # Set results_dir in config for TestBench (it reads from config)
    if "benchmark" not in config:
        config["benchmark"] = {}
    config["benchmark"]["results_dir"] = str(results_base_dir)
    
    return config

