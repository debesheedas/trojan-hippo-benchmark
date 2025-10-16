"""
Utility functions for the email agent MVP.
Provides helpers for ID generation, timestamps, trace logging, and config loading.
"""

import json
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Global constants
USER_EMAIL = "vince.j.kaminski@enron.com"  # User's email address - change this to update user email globally


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def get_timestamp() -> str:
    """Get current timestamp in ISO 8601 format with timezone."""
    return datetime.now(timezone.utc).isoformat()


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

