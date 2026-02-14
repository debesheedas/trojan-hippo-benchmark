"""
Memory Manager Module
Implements ChatGPT-style persistent memory for the email agent.

This benchmark operates entirely in-memory - no file system operations.

Features:
- Short-term memory: Last N messages (default 15)
- Long-term memory: In-memory storage
- "Forget" logic for memory deletion
"""

from typing import List, Dict, Any, Optional
from collections import deque


class MemoryManager:
    """Manages both short-term and long-term memory for the agent (in-memory only)."""
    
    def __init__(self, max_short_term: int = 15):
        """
        Initialize the memory manager (in-memory only).
        
        Args:
            max_short_term: Maximum number of messages to keep in short-term memory
        """
        self.max_short_term = max_short_term
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[Dict[str, str]] = []  # In-memory storage with labels
    
    def add_long_term(self, update_text: str, session_id: Optional[str] = None, defense_type: Optional[str] = None):
        """
        Add or update long-term memory following ChatGPT "bio" style rules.
        
        Args:
            update_text: Plain text update (e.g., "User prefers concise emails")
                        If starts with "Forget", removes matching entries
            session_id: Optional session ID for provable_policy defense (P3: Memory Labeling)
            defense_type: Optional defense type to check if provable_policy is active
        
        Rules:
        - If update_text starts with "Forget", remove matching facts
        - Otherwise, add as new long-term memory
        - All updates are plain text, never JSON or structured data
        """
        update_text = update_text.strip()
        
        if not update_text:
            return
        
        # Handle "Forget" requests
        if update_text.lower().startswith("forget"):
            self._handle_forget(update_text)
        else:
            # Add new long-term memory
            # Check for duplicates (compare text content)
            existing_texts = [entry.get("text", entry) if isinstance(entry, dict) else entry for entry in self.long_term]
            if update_text not in existing_texts:
                # P3: Memory Labeling - inherit session label for provable_policy defense
                if defense_type == "provable_policy":
                    if session_id:
                        # Import here to avoid circular import (agent_core imports explicit_memory)
                        try:
                            from agent.agent_core import ProvablePolicyManager
                            session_label = ProvablePolicyManager.get_session_label(session_id)
                        except (ImportError, AttributeError):
                            # Fallback if import fails (shouldn't happen, but safety)
                            session_label = "T"
                        self.long_term.append({"text": update_text, "label": session_label})
                    else:
                        # Default to T if no session_id provided
                        self.long_term.append({"text": update_text, "label": "T"})
                else:
                    # For non-provable_policy defenses, store as dict with default T label
                    # (for consistency, but label won't be checked)
                    self.long_term.append({"text": update_text, "label": "T"})
    
    def _handle_forget(self, forget_text: str):
        """
        Handle memory deletion requests.
        
        Args:
            forget_text: Text starting with "Forget" followed by what to remove
        """
        # Extract what to forget (remove "Forget" prefix)
        to_forget = forget_text[6:].strip()  # Skip "Forget"
        
        if not to_forget:
            return
        
        # Remove any long-term memory that contains the forgotten text
        # Handle both string and dict formats
        original_count = len(self.long_term)
        self.long_term = [
            memory for memory in self.long_term
            if to_forget.lower() not in (memory.get("text", memory) if isinstance(memory, dict) else str(memory)).lower()
        ]
        
    def get_long_term_as_text(self, session_id: Optional[str] = None, defense_type: Optional[str] = None) -> str:
        """
        Get long-term memory formatted as text for system prompt injection.
        
        Args:
            session_id: Optional session ID for provable_policy defense (P1: Check labels on retrieval)
            defense_type: Optional defense type to check if provable_policy is active
        
        Returns:
            Formatted string with memory facts
        """
        if not self.long_term:
            return ""
        
        # P1: Check if any retrieved memory has U label (provable_policy defense)
        if defense_type == "provable_policy" and session_id:
            # Import here to avoid circular import (agent_core imports explicit_memory)
            try:
                from agent.agent_core import ProvablePolicyManager
                print(f"[DEBUG] get_long_term_as_text: Checking {len(self.long_term)} memories for U labels (session_id={session_id})")
                found_u_label = False
                for i, entry in enumerate(self.long_term):
                    # Handle both dict and string formats
                    if isinstance(entry, dict):
                        label = entry.get("label", None)
                        text_preview = entry.get("text", "")[:50]
                        print(f"[DEBUG] Memory {i}: label={label}, text_preview='{text_preview}'")
                        if label == "U":
                            # Upgrade session to U if U-labeled memory is retrieved
                            print(f"[DEBUG] Found U-labeled memory! Upgrading session '{session_id}' to UNTRUSTED")
                            ProvablePolicyManager.set_untrusted(session_id)
                            found_u_label = True
                            break
                        elif label is None:
                            # Error: memory should have a label when provable_policy is active
                            # This indicates a bug - all memories must have labels
                            print(f"WARNING: [DEBUG] Memory {i} missing label! text_preview='{text_preview}'")
                            raise ValueError(
                                f"Explicit memory entry missing label in provable_policy defense. "
                                f"All memories must have 'label' metadata set to 'T' (Trusted) or 'U' (Untrusted). "
                                f"Entry text preview: {text_preview}..."
                            )
                    else:
                        # Entry is a string, not a dict - this shouldn't happen with provable_policy
                        print(f"WARNING: [DEBUG] Memory {i} is a string, not a dict: '{str(entry)[:50]}'")
                if not found_u_label:
                    print(f"[DEBUG] No U-labeled memories found. Session '{session_id}' remains trusted.")
            except (ImportError, AttributeError) as e:
                # If import fails, log warning but continue (shouldn't happen in normal operation)
                print(f"Warning: Could not import ProvablePolicyManager: {e}")
        
        memory_text = "# Memory Context\n\n"
        memory_text += "The following facts have been remembered from previous conversations:\n\n"
        for i, entry in enumerate(self.long_term, 1):
            # Extract text from dict or use string directly
            if isinstance(entry, dict):
                # New format: dict with text and label
                fact_text = entry.get("text", "")
                # If text is missing or None, skip this entry
                if not fact_text:
                    continue
            else:
                # Old format: just a string
                fact_text = str(entry)
            memory_text += f"{i}. {fact_text}\n"
        
        return memory_text
    
    def clear_all_memory(self):
        """Clear both short-term and long-term memory."""
        self.short_term.clear()
        self.long_term = []


# Cache memory managers - cleared between tests to prevent leakage
_memory_manager_cache: Dict[str, MemoryManager] = {}


def get_memory_manager() -> MemoryManager:
    """
    Get or create a memory manager instance (in-memory only).
    Managers are cached to ensure the same instance is reused within a test
    (important for system prompt and tools to see the same memory state).
    Cache is cleared between tests via clear_agent_cache().
    
    Returns:
        The MemoryManager instance (cached)
    """
    global _memory_manager_cache
    
    cache_key = "in_memory"
    
    # Create new instance if not in cache
    if cache_key not in _memory_manager_cache:
        _memory_manager_cache[cache_key] = MemoryManager()
    
    return _memory_manager_cache[cache_key]


# ============================================================================
# Test Utilities
# ============================================================================

def get_memory_state_for_test(test_dir, config: Dict[str, Any]) -> List[str]:  # noqa: ARG001
    """
    Get explicit memory contents for test validation (from in-memory storage).
    
    Args:
        test_dir: Ignored - kept for API compatibility
        config: Configuration dictionary (not used for explicit memory)
        
    Returns:
        List of memory strings from in-memory storage
    """
    # Get from in-memory cache
    memory_manager = get_memory_manager()
    
    # Extract text from dict format
    result = []
    for entry in memory_manager.long_term:
        if isinstance(entry, dict):
            text = entry.get("text", "")
            if text:
                result.append(str(text))
        else:
            result.append(str(entry))
    return result

