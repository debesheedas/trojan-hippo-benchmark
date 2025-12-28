"""
Memory Manager Module
Implements ChatGPT-style persistent memory for the email agent.

Features:
- Short-term memory: Last N messages (default 15)
- Long-term memory: Persistent JSON storage
- Auto-load/save functionality
- "Forget" logic for memory deletion
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime, timezone
# Removed threading import - locks not needed since each test gets isolated instance


class MemoryManager:
    """Manages both short-term and long-term memory for the agent."""
    
    def __init__(self, memory_file: str = "data/interactive_agent/agent_memory.json", max_short_term: int = 15):
        """
        Initialize the memory manager.
        
        Args:
            memory_file: Path to the JSON file for long-term memory
            max_short_term: Maximum number of messages to keep in short-term memory
        """
        self.memory_file = Path(memory_file)
        self.max_short_term = max_short_term
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[Dict[str, str]] = []  # Changed from List[str] to support labels
        # Removed _lock - not needed since each test gets isolated instance
        self._subscribers = []  # For SSE updates (interactive agent only)
        
        # Ensure data directory exists
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing long-term memory
        self._load_long_term()
    
    def _load_long_term(self):
        """Load long-term memory from JSON file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_long_term = data.get("long_term", [])
                    # Convert old string format to dict format for backward compatibility
                    self.long_term = []
                    for entry in raw_long_term:
                        if isinstance(entry, str):
                            # Old format: just a string, default to T (Trusted)
                            self.long_term.append({"text": entry, "label": "T"})
                        elif isinstance(entry, dict):
                            # New format: dict with text and label
                            self.long_term.append(entry)
                        else:
                            # Skip invalid entries
                            continue
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load memory file: {e}")
                self.long_term = []
        else:
            self.long_term = []
    
    def _save_long_term(self):
        """Save long-term memory to JSON file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "long_term": self.long_term,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving memory file: {e}")
    
    def add_short_term(self, message: str):
        """
        Add a message to short-term memory.
        
        Args:
            message: The message to add (can be user or assistant message)
        """
        self.short_term.append(message)
        self._notify_subscribers()
    
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
                        # Import here to avoid circular import (agent_core imports memory_manager)
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
                self._save_long_term()
                self._notify_subscribers()
    
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
        
        # Save if anything was removed
        if len(self.long_term) < original_count:
            self._save_long_term()
            self._notify_subscribers()
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get current memory context.
        
        Returns:
            Dictionary with short_term and long_term memory lists
        """
        return {
            "short_term": list(self.short_term),
            "long_term": self.long_term.copy()
        }
    
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
            # Import here to avoid circular import (agent_core imports memory_manager)
            try:
                from agent.agent_core import ProvablePolicyManager
                for entry in self.long_term:
                    # Handle both dict and string formats
                    if isinstance(entry, dict):
                        label = entry.get("label", None)
                        if label == "U":
                            # Upgrade session to U if U-labeled memory is retrieved
                            ProvablePolicyManager.set_untrusted(session_id)
                            break
                        elif label is None:
                            # Error: memory should have a label when provable_policy is active
                            # This indicates a bug - all memories must have labels
                            raise ValueError(
                                f"Explicit memory entry missing label in provable_policy defense. "
                                f"All memories must have 'label' metadata set to 'T' (Trusted) or 'U' (Untrusted). "
                                f"Entry text preview: {entry.get('text', '')[:50]}..."
                            )
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
        self._save_long_term()
        self._notify_subscribers()
    
    def clear_short_term(self):
        """Clear only short-term memory."""
        self.short_term.clear()
        self._notify_subscribers()
    
    def subscribe(self, callback):
        """
        Subscribe to memory updates for SSE.
        
        Args:
            callback: Function to call when memory changes
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """
        Unsubscribe from memory updates.
        
        Args:
            callback: The callback to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self):
        """Notify all subscribers of memory changes."""
        context = self.get_context()
        for callback in self._subscribers:
            try:
                callback(context)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")


# Cache memory managers by memory_file path - cleared between tests to prevent leakage
_memory_manager_cache: Dict[str, MemoryManager] = {}


def get_memory_manager(memory_file: str = "data/interactive_agent/agent_memory.json") -> MemoryManager:
    """
    Get or create a memory manager instance.
    Managers are cached by memory_file path to ensure the same instance is reused
    within a test (important for system prompt and tools to see the same memory state).
    Cache is cleared between tests via clear_agent_cache().
    
    Args:
        memory_file: Path to the memory file (used as cache key)
    
    Returns:
        The MemoryManager instance (cached per memory_file path)
    """
    global _memory_manager_cache
    
    # Use memory_file as cache key
    cache_key = str(memory_file)
    
    # Create new instance if not in cache
    if cache_key not in _memory_manager_cache:
        _memory_manager_cache[cache_key] = MemoryManager(memory_file=memory_file)
    
    return _memory_manager_cache[cache_key]

