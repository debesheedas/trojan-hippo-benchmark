"""
Mem0 Defense Manager

Manages defense mechanisms for mem0 memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. user_only: Only index user messages (filter out assistant messages)
3. no_untrusted_tools: Only index when no untrusted tools have been called in the session
"""

from typing import List, Dict, Any, Optional


class Mem0DefenseManager:
    """
    Manages defense mechanisms for mem0 memory indexing.
    """
    
    def __init__(
        self,
        defense_type: str = "none",
        trace_file: Optional[str] = None  # Kept for backward compatibility, but not used
    ):
        """
        Initialize the defense manager.
        
        Args:
            defense_type: Type of defense to apply. Options:
                - "none": No defense (default behavior)
                - "disable_memory": Disable all memory indexing
                - "user_only": Only index user messages
                - "no_untrusted_tools": Only index when no untrusted tools called
                - "limit_memory_length": Truncate extracted mem0 memories to a fixed length
            trace_file: Deprecated - kept for backward compatibility only
        """
        self.defense_type = defense_type
        
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
            # Check session trust status using SessionTrustManager
            from agent.agent_core import SessionTrustManager
            if not SessionTrustManager.is_trusted(session_id):
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

