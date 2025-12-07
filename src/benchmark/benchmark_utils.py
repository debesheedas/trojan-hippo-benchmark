"""
Benchmark Utilities

Helper functions for result path generation, caching, and test isolation.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from benchmark.defense_backend import get_defense_backend_registry


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
    # Handle no memory backend: use "none" as backend name and "none" as defense
    if memory_backend == "none":
        backend_for_path = "none"
        defense_folder = "none"
    else:
        backend_for_path = memory_backend
        # Map unified defense to backend-specific defense type
        defense_registry = get_defense_backend_registry()
        backend_defense = defense_registry.map_defense(memory_backend, unified_defense)
        
        # Normalize defense folder name
        # For mem0, "no_defense" is used instead of "none"
        if memory_backend == "mem0" and backend_defense == "no_defense":
            defense_folder = "no_defense"
        else:
            defense_folder = backend_defense
    
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
        Memory backend name ("explicit", "mem0", "rag", or "none" if all disabled)
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
    
    # Check if all backends are explicitly disabled
    explicit_disabled = memory_config.get("explicit_memory", {}).get("enabled") is False
    mem0_disabled = memory_config.get("mem0_memory", {}).get("enabled") is False
    rag_disabled = memory_config.get("rag_memory", {}).get("enabled") is False
    
    if explicit_disabled and mem0_disabled and rag_disabled:
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
    if memory_backend == "explicit":
        defense = memory_config.get("explicit_memory", {}).get("defense_type", "none")
    elif memory_backend == "mem0":
        defense = memory_config.get("mem0_memory", {}).get("defense_type", "none")
    elif memory_backend == "rag":
        defense = memory_config.get("rag_memory", {}).get("defense_type", "none")
    else:
        defense = "none"
    
    # Normalize: mem0 uses "no_defense" but we want unified "none"
    if memory_backend == "mem0" and defense == "no_defense":
        return "none"
    
    return defense

