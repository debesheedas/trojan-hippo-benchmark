"""
Mem0 Defense Manager

Manages defense mechanisms for mem0 memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. user_prompt_only: Only index user messages (filter out assistant messages)
3. no_untrusted_tools: Only index when no untrusted tools have been called in the session
"""

from typing import List, Dict, Any, Optional


class Mem0DefenseManager:
    """
    Manages defense mechanisms for mem0 memory indexing.
    """
    
    def __init__(
        self,
        defense_type: str = "none"
    ):
        """
        Initialize the defense manager.
        
        Args:
            defense_type: Type of defense to apply. Options:
                - "none": No defense (default behavior)
                - "disable_memory": Disable all memory indexing
                - "user_prompt_only": Only index user messages
                - "no_untrusted_tools": Only index when no untrusted tools called
                - "limit_memory_length": Truncate extracted mem0 memories to a fixed length
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
        if self.defense_type == "user_prompt_only":
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
        if self.defense_type == "user_prompt_only":
            # Filter to only include user messages
            return [msg for msg in messages if msg.get("role") == "user"]
        
        # Other defenses don't filter messages
        return messages
    


def get_defense_manager(
    defense_type: str = "none",
    session_id: Optional[str] = None,
) -> Mem0DefenseManager:
    """
    Create a new defense manager instance.
    
    Args:
        defense_type: Type of defense to apply
        session_id: Optional session ID (kept for API compatibility, not used)
        
    Returns:
        A new Mem0DefenseManager instance
    """
    return Mem0DefenseManager(
        defense_type=defense_type
    )

