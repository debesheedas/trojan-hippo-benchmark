"""
Utility functions for the email agent MVP.
Provides helpers for ID generation, timestamps, trace logging, and config loading.
"""

import json
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Global constants
USER_EMAIL = "vince.j.kaminski@enron.com"  # User's email address - change this to update user email globally


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def get_timestamp() -> str:
    """Get fixed timestamp in ISO 8601 format with timezone (for reproducibility: November 3, 2025)."""
    # Fixed date: November 3, 2025, 12:00:00 UTC for reproducibility
    fixed_date = datetime(2025, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
    return fixed_date.isoformat()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def append_trace_event(
    trace_file: str,
    event_type: str,
    session_id: str,
    payload: dict,
    event_id: Optional[str] = None
) -> str:
    """
    Append a trace event to the session-specific trace file.
    
    Args:
        trace_file: Base path to the trace JSONL file (will be modified for session-specific storage)
        event_type: Type of event (user_input, tool_call, tool_result, agent_response)
        session_id: Session identifier
        payload: Event-specific data
        event_id: Optional event ID (generated if not provided)
    
    Returns:
        The event_id used for this event
    """
    if event_id is None:
        event_id = generate_id()
    
    event = {
        "ts": get_timestamp(),
        "event_id": event_id,
        "session_id": session_id,
        "event_type": event_type,
        "payload": payload
    }
    
    # Create session-specific trace file path
    trace_path = Path(trace_file)
    session_trace_dir = trace_path.parent / "traces"
    session_trace_file = session_trace_dir / f"{session_id}.jsonl"
    
    # Ensure trace file directory exists
    session_trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Append event to session-specific file
    with open(session_trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    
    return event_id


def read_trace_events(trace_file: str, session_id: Optional[str] = None) -> list:
    """
    Read trace events from the session-specific trace file.
    
    Args:
        trace_file: Base path to the trace JSONL file (will be modified for session-specific storage)
        session_id: Session ID to read events for (required for session-specific storage)
    
    Returns:
        List of trace events for the specified session
    """
    if session_id is None:
        return []
    
    # Create session-specific trace file path
    trace_path = Path(trace_file)
    session_trace_dir = trace_path.parent / "traces"
    session_trace_file = session_trace_dir / f"{session_id}.jsonl"
    
    if not session_trace_file.exists():
        return []
    
    events = []
    with open(session_trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                events.append(event)
    
    return events


def ensure_data_directories(config: dict) -> None:
    """Ensure all required data directories exist."""
    data_config = config.get("data", {})
    
    # Create mailbox directory
    mailbox_dir = Path(data_config.get("mailbox_dir", "data/mailbox"))
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Create drafts directory
    drafts_dir = Path(data_config.get("drafts_dir", "data/drafts"))
    drafts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create outbox directory
    outbox_dir = Path(data_config.get("outbox_dir", "data/outbox"))
    outbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trace file parent directory
    trace_file = Path(data_config.get("trace_file", "data/trace.jsonl"))
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize trace file if it doesn't exist
    if not trace_file.exists():
        trace_file.touch()


def compare_attack_bench_files(original_file: Path, cached_file: Path) -> Dict[str, Any]:
    """
    Compare attack benchmark files to ensure only attack_emails differ.
    
    This function verifies that cached files only differ from original files
    in the attack_emails attribute, ensuring the caching system works correctly.
    
    Args:
        original_file: Path to the original attack benchmark file
        cached_file: Path to the cached attack benchmark file
    
    Returns:
        Dictionary with comparison results:
        - valid: bool - True if files are valid (only attack_emails differ)
        - attack_emails_differ: bool - True if attack_emails are different
        - differences: List[str] - List of field names that differ
        - summary: str - Human-readable summary
    """
    result = {
        "valid": True,
        "attack_emails_differ": False,
        "differences": [],
        "summary": ""
    }
    
    try:
        # Load both files
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        with open(cached_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        
        # Compare all keys and values except attack_emails
        for key in original_data:
            if key == "attack_emails":
                # Check if attack_emails differ
                if original_data[key] != cached_data.get(key):
                    result["attack_emails_differ"] = True
                continue
            
            # For all other keys, they must be identical
            if key not in cached_data:
                result["valid"] = False
                result["differences"].append(f"{key} (missing in cached)")
            elif original_data[key] != cached_data[key]:
                result["valid"] = False
                result["differences"].append(key)
        
        # Check for extra keys in cached file (excluding optimization_metadata)
        for key in cached_data:
            if key not in original_data and key != "optimization_metadata":
                result["valid"] = False
                result["differences"].append(f"{key} (extra in cached)")
        
        # Generate summary
        if result["valid"] and result["attack_emails_differ"]:
            result["summary"] = "✅ Files are identical except for attack_emails (as expected)"
        elif result["valid"] and not result["attack_emails_differ"]:
            result["summary"] = "✅ Files are completely identical"
        else:
            result["summary"] = f"❌ Files differ in fields: {', '.join(result['differences'])}"
        
        return result
        
    except Exception as e:
        return {
            "valid": False,
            "attack_emails_differ": False,
            "differences": [f"Error: {e}"],
            "summary": f"❌ Error comparing files: {e}"
        }


def validate_cache_integrity(cache_dir: str = "attack_bench_cache", 
                           original_dir: str = "attack_bench") -> Dict[str, Any]:
    """
    Validate the integrity of all cached attack benchmark files.
    
    This function compares all cached files with their original counterparts
    to ensure the caching system is working correctly.
    
    Args:
        cache_dir: Directory containing cached files
        original_dir: Directory containing original files
    
    Returns:
        Dictionary with validation results:
        - total_files: int - Total number of files compared
        - valid_files: int - Number of files that are correctly cached
        - invalid_files: int - Number of files with issues
        - results: List[Dict] - Detailed results for each file
        - summary: str - Overall validation summary
    """
    cache_path = Path(cache_dir)
    original_path = Path(original_dir)
    
    if not cache_path.exists():
        return {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "results": [],
            "summary": "❌ Cache directory does not exist"
        }
    
    # Find all cached files
    cached_files = list(cache_path.rglob("*.json"))
    results = []
    valid_files = 0
    invalid_files = 0
    
    for cached_file in cached_files:
        # Find corresponding original file
        relative_path = cached_file.relative_to(cache_path)
        original_file = original_path / relative_path
        
        if not original_file.exists():
            results.append({
                "cached_file": str(cached_file),
                "original_file": str(original_file),
                "status": "missing_original",
                "result": {
                    "identical": False,
                    "summary": "❌ Original file not found"
                }
            })
            invalid_files += 1
            continue
        
        # Compare files
        comparison_result = compare_attack_bench_files(original_file, cached_file)
        
        # Use the simplified valid flag
        is_valid = comparison_result["valid"]
        
        if is_valid:
            valid_files += 1
        else:
            invalid_files += 1
        
        results.append({
            "cached_file": str(cached_file),
            "original_file": str(original_file),
            "status": "valid" if is_valid else "invalid",
            "result": comparison_result
        })
    
    total_files = len(cached_files)
    
    # Generate summary
    if invalid_files == 0:
        summary = f"✅ All {total_files} cached files are valid"
    else:
        summary = f"⚠️ {valid_files}/{total_files} cached files are valid, {invalid_files} have issues"
    
    return {
        "total_files": total_files,
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "results": results,
        "summary": summary
    }


def print_cache_validation_report(validation_result: Dict[str, Any]) -> None:
    """
    Print a formatted validation report for cache integrity.
    
    Args:
        validation_result: Result from validate_cache_integrity()
    """
    print("=" * 80)
    print("CACHE INTEGRITY VALIDATION REPORT")
    print("=" * 80)
    print(f"Total files: {validation_result['total_files']}")
    print(f"Valid files: {validation_result['valid_files']}")
    print(f"Invalid files: {validation_result['invalid_files']}")
    print(f"Summary: {validation_result['summary']}")
    print()
    
    if validation_result['invalid_files'] > 0:
        print("INVALID FILES:")
        print("-" * 40)
        for result in validation_result['results']:
            if result['status'] == 'invalid':
                print(f"❌ {result['cached_file']}")
                print(f"   Original: {result['original_file']}")
                print(f"   Issue: {result['result']['summary']}")
                if result['result']['differences']:
                    print("   Differences:")
                    for diff in result['result']['differences']:
                        print(f"     - {diff}")
                print()
    
    print("VALID FILES:")
    print("-" * 40)
    for result in validation_result['results']:
        if result['status'] == 'valid':
            print(f"✅ {result['cached_file']}")
            if result['result']['attack_emails_differ']:
                print("   (attack_emails differ as expected)")
            else:
                print("   (completely identical)")
    print("=" * 80)

