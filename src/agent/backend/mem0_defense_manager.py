"""
Mem0 Defense Manager

Manages defense mechanisms for mem0 memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. user_only: Only index user messages (filter out assistant messages)
3. no_untrusted_tools: Only index when no untrusted tools have been called in the session
"""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from agent.utils import read_trace_events


# Define trusted and untrusted tools
# Untrusted tools are those that read from untrusted data sources (inbox)
UNTRUSTED_TOOLS: Set[str] = {
    "read_all_emails",
    "search_emails", 
    "reply_to_email",
    "forward_to_email"
}

# Trusted tools are those that don't read from untrusted sources
TRUSTED_TOOLS: Set[str] = {
    "compose_email",
    "draft_email",
    "update_memory"
}


class Mem0DefenseManager:
    """
    Manages defense mechanisms for mem0 memory indexing.
    """
    
    def __init__(
        self,
        defense_type: str = "none",
        trace_file: Optional[str] = None
    ):
        """
        Initialize the defense manager.
        
        Args:
            defense_type: Type of defense to apply. Options:
                - "none": No defense (default behavior)
                - "disable_memory": Disable all memory indexing
                - "user_only": Only index user messages
                - "no_untrusted_tools": Only index when no untrusted tools called
            trace_file: Path to trace file for checking tool calls
        """
        self.defense_type = defense_type
        self.trace_file = trace_file
        # Session trust variable: tracks whether untrusted tools have been called
        # - If session_id not in dict: session is TRUSTED (no untrusted tools called yet)
        # - If session_id in dict with value True: session is NOT TRUSTED (untrusted tool was called)
        # This acts as a session-level trust variable that starts as True (trusted) and
        # gets set to False (not trusted) when an untrusted tool is detected
        self._session_untrusted_tool_called: Dict[str, bool] = {}  # Track per session
        
    def should_index_memory(
        self,
        session_id: str,
        messages: List[Dict[str, str]]
    ) -> bool:
        """
        Determine if memory should be indexed based on defense type.
        
        Args:
            session_id: Session identifier
            messages: List of messages to potentially index
            
        Returns:
            True if memory should be indexed, False otherwise
        """
        # Defense 1: Disable memory (baseline)
        if self.defense_type == "disable_memory":
            return False
        
        # Defense 2: User-only indexing
        if self.defense_type == "user_only":
            # This is handled by filtering messages, not by returning False
            # So we return True here and filter in filter_messages
            return True
        
        # Defense 3: No untrusted tools
        if self.defense_type == "no_untrusted_tools":
            # Check if any untrusted tool has been called in this session
            if self._has_untrusted_tool_been_called(session_id):
                return False
            return True
        
        # No defense: allow indexing
        if self.defense_type == "none":
            return True
        
        # Unknown defense type: default to allowing
        return True
    
    def filter_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Filter messages based on defense type.
        
        Args:
            messages: List of messages to filter
            
        Returns:
            Filtered list of messages
        """
        # Defense 2: User-only indexing
        if self.defense_type == "user_only":
            # Filter to only include user messages
            return [msg for msg in messages if msg.get("role") == "user"]
        
        # Other defenses don't filter messages
        return messages
    
    def _has_untrusted_tool_been_called(self, session_id: str) -> bool:
        """
        Check if any untrusted tool has been called in this session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if an untrusted tool has been called, False otherwise
        """
        # Check cache first
        if session_id in self._session_untrusted_tool_called:
            return self._session_untrusted_tool_called[session_id]
        
        # If no trace file, assume no untrusted tools called
        if not self.trace_file:
            return False
        
        # Read trace events for this session
        try:
            trace_events = read_trace_events(self.trace_file, session_id)
            
            # Check for any tool_call events with untrusted tools
            for event in trace_events:
                if event.get("event_type") == "tool_call":
                    payload = event.get("payload", {})
                    tool_name = payload.get("tool_name", "")
                    
                    # Check if this is an untrusted tool
                    if tool_name in UNTRUSTED_TOOLS:
                        # Cache the result
                        self._session_untrusted_tool_called[session_id] = True
                        return True
        except Exception as e:
            print(f"Warning: Could not check trace events for untrusted tools: {e}")
        
        # Cache negative result
        self._session_untrusted_tool_called[session_id] = False
        return False
    
    def record_tool_call(self, session_id: str, tool_name: str):
        """
        Record a tool call for defense tracking.
        This can be called directly when a tool is invoked to avoid reading trace files.
        Sets the session trust variable to False (not trusted) when an untrusted tool is called.
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool that was called
        """
        if tool_name in UNTRUSTED_TOOLS:
            # Set session trust to False (not trusted) when untrusted tool is detected
            self._session_untrusted_tool_called[session_id] = True
    
    def reset_session(self, session_id: str):
        """
        Reset defense state for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._session_untrusted_tool_called:
            del self._session_untrusted_tool_called[session_id]


# Global defense manager cache (per session)
_defense_manager_cache: Dict[str, Mem0DefenseManager] = {}


def get_defense_manager(
    defense_type: str = "none",
    trace_file: Optional[str] = None,
    session_id: Optional[str] = None,
    force_new: bool = False
) -> Mem0DefenseManager:
    """
    Get or create a defense manager instance.
    
    Args:
        defense_type: Type of defense to apply
        trace_file: Path to trace file
        session_id: Optional session ID for caching
        force_new: If True, create a new instance
        
    Returns:
        Mem0DefenseManager instance
    """
    global _defense_manager_cache
    
    # Use session_id + defense_type as cache key
    cache_key = f"{session_id or 'default'}_{defense_type}"
    
    if force_new or cache_key not in _defense_manager_cache:
        _defense_manager_cache[cache_key] = Mem0DefenseManager(
            defense_type=defense_type,
            trace_file=trace_file
        )
    
    return _defense_manager_cache[cache_key]

