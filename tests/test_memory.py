"""
Unit tests for the Memory Manager functionality.

Tests cover:
- Short-term memory (rolling buffer)
- Long-term memory (persistent storage)
- Forget functionality
- Memory persistence across instances
"""

import unittest
import json
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.memory_file = self.temp_file.name
        
        # Create memory manager with temp file
        self.memory = MemoryManager(memory_file=self.memory_file, max_short_term=5)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary file
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
    
    def test_short_term_memory_add(self):
        """Test adding messages to short-term memory."""
        self.memory.add_short_term("Message 1")
        self.memory.add_short_term("Message 2")
        
        context = self.memory.get_context()
        self.assertEqual(len(context["short_term"]), 2)
        self.assertIn("Message 1", context["short_term"])
        self.assertIn("Message 2", context["short_term"])
    
    def test_short_term_memory_rolling_buffer(self):
        """Test that short-term memory only keeps last N messages."""
        # Add 7 messages (max is 5)
        for i in range(7):
            self.memory.add_short_term(f"Message {i}")
        
        context = self.memory.get_context()
        
        # Should only have 5 messages (the most recent)
        self.assertEqual(len(context["short_term"]), 5)
        
        # Should have messages 2-6 (0 and 1 were dropped)
        self.assertNotIn("Message 0", context["short_term"])
        self.assertNotIn("Message 1", context["short_term"])
        self.assertIn("Message 6", context["short_term"])
    
    def test_long_term_memory_add(self):
        """Test adding facts to long-term memory."""
        self.memory.add_long_term("User prefers concise emails")
        self.memory.add_long_term("User works at Acme Corp")
        
        context = self.memory.get_context()
        self.assertEqual(len(context["long_term"]), 2)
        self.assertIn("User prefers concise emails", context["long_term"])
        self.assertIn("User works at Acme Corp", context["long_term"])
    
    def test_long_term_memory_no_duplicates(self):
        """Test that long-term memory avoids exact duplicates."""
        self.memory.add_long_term("User likes apples")
        self.memory.add_long_term("User likes apples")  # Duplicate
        
        context = self.memory.get_context()
        self.assertEqual(len(context["long_term"]), 1)
    
    def test_forget_functionality(self):
        """Test that Forget command removes matching entries."""
        self.memory.add_long_term("User likes apples")
        self.memory.add_long_term("User likes bananas")
        self.memory.add_long_term("User prefers detailed reports")
        
        # Forget apples
        self.memory.add_long_term("Forget apples")
        
        context = self.memory.get_context()
        
        # Should only have 2 items (bananas and reports)
        self.assertEqual(len(context["long_term"]), 2)
        self.assertNotIn("User likes apples", context["long_term"])
        self.assertIn("User likes bananas", context["long_term"])
        self.assertIn("User prefers detailed reports", context["long_term"])
    
    def test_forget_case_insensitive(self):
        """Test that Forget is case-insensitive."""
        self.memory.add_long_term("User prefers VERBOSE output")
        
        # Forget with different case
        self.memory.add_long_term("Forget verbose")
        
        context = self.memory.get_context()
        self.assertEqual(len(context["long_term"]), 0)
    
    def test_memory_persistence(self):
        """Test that long-term memory persists between instances."""
        # Add facts to first instance
        self.memory.add_long_term("User prefers concise emails")
        self.memory.add_long_term("User works remotely")
        
        # Create new instance with same file
        memory2 = MemoryManager(memory_file=self.memory_file)
        
        context = memory2.get_context()
        
        # Should load the same facts
        self.assertEqual(len(context["long_term"]), 2)
        self.assertIn("User prefers concise emails", context["long_term"])
        self.assertIn("User works remotely", context["long_term"])
    
    def test_clear_all_memory(self):
        """Test clearing all memory."""
        self.memory.add_short_term("Short message")
        self.memory.add_long_term("Long term fact")
        
        self.memory.clear_all_memory()
        
        context = self.memory.get_context()
        self.assertEqual(len(context["short_term"]), 0)
        self.assertEqual(len(context["long_term"]), 0)
    
    def test_clear_short_term_only(self):
        """Test clearing only short-term memory."""
        self.memory.add_short_term("Short message")
        self.memory.add_long_term("Long term fact")
        
        self.memory.clear_short_term()
        
        context = self.memory.get_context()
        self.assertEqual(len(context["short_term"]), 0)
        self.assertEqual(len(context["long_term"]), 1)
    
    def test_get_long_term_as_text(self):
        """Test formatting long-term memory as text for system prompt."""
        self.memory.add_long_term("User prefers detailed explanations")
        self.memory.add_long_term("User is learning Python")
        
        text = self.memory.get_long_term_as_text()
        
        # Should contain header and facts
        self.assertIn("# Memory Context", text)
        self.assertIn("User prefers detailed explanations", text)
        self.assertIn("User is learning Python", text)
    
    def test_empty_long_term_text(self):
        """Test that empty long-term memory returns empty string."""
        text = self.memory.get_long_term_as_text()
        self.assertEqual(text, "")
    
    def test_json_file_structure(self):
        """Test that JSON file has correct structure."""
        self.memory.add_long_term("Test fact")
        
        # Read JSON file directly
        with open(self.memory_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn("long_term", data)
        self.assertIn("last_updated", data)
        self.assertEqual(len(data["long_term"]), 1)
        self.assertEqual(data["long_term"][0], "Test fact")
    
    def test_corrupted_json_file(self):
        """Test that corrupted JSON file is handled gracefully."""
        # Write corrupted JSON
        with open(self.memory_file, 'w') as f:
            f.write("{ corrupted json }")
        
        # Should not crash, just start with empty memory
        memory = MemoryManager(memory_file=self.memory_file)
        context = memory.get_context()
        
        self.assertEqual(len(context["long_term"]), 0)
    
    def test_multiple_forget_patterns(self):
        """Test forgetting with partial matches."""
        self.memory.add_long_term("User prefers Gmail")
        self.memory.add_long_term("User uses Gmail API v1")
        self.memory.add_long_term("User works at Google")
        
        # Forget anything with "Gmail"
        self.memory.add_long_term("Forget Gmail")
        
        context = self.memory.get_context()
        
        # Should only have Google fact remaining
        self.assertEqual(len(context["long_term"]), 1)
        self.assertIn("User works at Google", context["long_term"])


class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for memory system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.memory_file = self.temp_file.name
        self.memory = MemoryManager(memory_file=self.memory_file, max_short_term=15)
    
    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
    
    def test_conversation_flow(self):
        """Test realistic conversation flow with memory."""
        # User asks to remember something
        self.memory.add_short_term("User: Remember that I prefer concise emails")
        self.memory.add_long_term("User prefers concise emails")
        
        # Some conversation happens
        self.memory.add_short_term("Assistant: I'll remember that you prefer concise emails")
        self.memory.add_short_term("User: Thanks")
        
        # Later in conversation, user asks
        self.memory.add_short_term("User: What did I tell you to remember?")
        
        context = self.memory.get_context()
        
        # Should have conversation in short-term
        self.assertEqual(len(context["short_term"]), 4)
        
        # Should have preference in long-term
        self.assertEqual(len(context["long_term"]), 1)
        self.assertIn("User prefers concise emails", context["long_term"])
    
    def test_session_restart(self):
        """Test memory persistence across session restart."""
        # Session 1: User shares preferences
        memory1 = MemoryManager(memory_file=self.memory_file)
        memory1.add_long_term("User prefers detailed emails")
        memory1.add_short_term("User: Remember I like details")
        
        # Session 2: New instance (simulates restart)
        memory2 = MemoryManager(memory_file=self.memory_file)
        
        context = memory2.get_context()
        
        # Long-term memory should persist
        self.assertIn("User prefers detailed emails", context["long_term"])
        
        # Short-term memory should be empty (new session)
        self.assertEqual(len(context["short_term"]), 0)


if __name__ == "__main__":
    unittest.main()

