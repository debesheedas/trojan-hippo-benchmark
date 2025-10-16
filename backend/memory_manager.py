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
from typing import List, Dict, Any
from collections import deque
from datetime import datetime
import threading


class MemoryManager:
    """Manages both short-term and long-term memory for the agent."""
    
    def __init__(self, memory_file: str = "data/agent_memory.json", max_short_term: int = 15):
        """
        Initialize the memory manager.
        
        Args:
            memory_file: Path to the JSON file for long-term memory
            max_short_term: Maximum number of messages to keep in short-term memory
        """
        self.memory_file = Path(memory_file)
        self.max_short_term = max_short_term
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[str] = []
        self._lock = threading.Lock()
        self._subscribers = []  # For SSE updates
        
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
                    self.long_term = data.get("long_term", [])
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
                    "last_updated": datetime.utcnow().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving memory file: {e}")
    
    def add_short_term(self, message: str):
        """
        Add a message to short-term memory.
        
        Args:
            message: The message to add (can be user or assistant message)
        """
        with self._lock:
            self.short_term.append(message)
            self._notify_subscribers()
    
    def add_long_term(self, update_text: str):
        """
        Add or update long-term memory following ChatGPT "bio" style rules.
        
        Args:
            update_text: Plain text update (e.g., "User prefers concise emails")
                        If starts with "Forget", removes matching entries
        
        Rules:
        - If update_text starts with "Forget", remove matching facts
        - Otherwise, add as new long-term memory
        - All updates are plain text, never JSON or structured data
        """
        with self._lock:
            update_text = update_text.strip()
            
            if not update_text:
                return
            
            # Handle "Forget" requests
            if update_text.lower().startswith("forget"):
                self._handle_forget(update_text)
            else:
                # Add new long-term memory
                # Avoid exact duplicates
                if update_text not in self.long_term:
                    self.long_term.append(update_text)
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
        original_count = len(self.long_term)
        self.long_term = [
            memory for memory in self.long_term
            if to_forget.lower() not in memory.lower()
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
        with self._lock:
            return {
                "short_term": list(self.short_term),
                "long_term": self.long_term.copy()
            }
    
    def get_long_term_as_text(self) -> str:
        """
        Get long-term memory formatted as text for system prompt injection.
        
        Returns:
            Formatted string with memory facts
        """
        with self._lock:
            if not self.long_term:
                return ""
            
            memory_text = "# Memory Context\n\n"
            memory_text += "The following facts have been remembered from previous conversations:\n\n"
            for i, fact in enumerate(self.long_term, 1):
                memory_text += f"{i}. {fact}\n"
            
            return memory_text
    
    def clear_all_memory(self):
        """Clear both short-term and long-term memory."""
        with self._lock:
            self.short_term.clear()
            self.long_term = []
            self._save_long_term()
            self._notify_subscribers()
    
    def clear_short_term(self):
        """Clear only short-term memory."""
        with self._lock:
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


# Global memory manager instance
_memory_manager = None


def get_memory_manager(memory_file: str = "data/agent_memory.json", force_new: bool = False) -> MemoryManager:
    """
    Get or create the global memory manager instance.
    
    Args:
        memory_file: Path to the memory file (default: "data/agent_memory.json")
        force_new: If True, create a new instance instead of reusing the global one
    
    Returns:
        The MemoryManager instance
    """
    global _memory_manager
    if force_new or _memory_manager is None:
        _memory_manager = MemoryManager(memory_file=memory_file)
    return _memory_manager

