#!/usr/bin/env python3
"""
Test script to verify LangChain's trim_messages works correctly as a sliding window.

This script tests:
1. That trim_messages correctly keeps the most recent messages (sliding window)
2. That it respects token limits
3. That it doesn't break normal agent operation
4. That message format is preserved

Run this script after installing dependencies:
    pip install langchain-core tiktoken

Or activate your virtual environment that has these dependencies.
"""

import sys
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Check dependencies first
try:
    from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage
    import tiktoken
    print("✅ Dependencies available")
except ImportError as e:
    print(f"❌ ERROR: Required dependencies not available: {e}")
    print("\nInstall with:")
    print("  pip install langchain-core tiktoken")
    print("\nOr activate your virtual environment that has these dependencies.")
    sys.exit(1)

# Import the function (this will fail if dependencies aren't available)
try:
    from agent.agent_core import _truncate_session_messages
    print("✅ Function imported successfully\n")
except Exception as e:
    print(f"❌ ERROR: Failed to import _truncate_session_messages: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_sliding_window_behavior():
    """Test that trim_messages correctly implements sliding window (keeps most recent messages)."""
    print("=" * 80)
    print("Test 1: Sliding Window Behavior")
    print("=" * 80)
    
    # Create messages with varying sizes
    messages = [
        {"role": "user", "content": "Message 1: " + "x" * 100},  # ~25 tokens
        {"role": "assistant", "content": "Response 1: " + "y" * 100},  # ~25 tokens
        {"role": "user", "content": "Message 2: " + "x" * 100},  # ~25 tokens
        {"role": "assistant", "content": "Response 2: " + "y" * 100},  # ~25 tokens
        {"role": "user", "content": "Message 3: " + "x" * 100},  # ~25 tokens
        {"role": "assistant", "content": "Response 3: " + "y" * 100},  # ~25 tokens
        {"role": "user", "content": "Message 4: " + "x" * 100},  # ~25 tokens
        {"role": "assistant", "content": "Response 4: " + "y" * 100},  # ~25 tokens
    ]
    
    # Total: ~200 tokens
    # Set max_tokens to 100 - should keep last 4 messages (~100 tokens)
    max_tokens = 100
    
    result = _truncate_session_messages(
        messages=messages,
        model_name="gpt-4o-mini",
        max_tokens=max_tokens
    )
    
    print(f"Original messages: {len(messages)}")
    print(f"Truncated messages: {len(result)}")
    print(f"Max tokens: {max_tokens}")
    print()
    
    # Verify sliding window: should keep most recent messages
    print("Original messages:")
    for i, msg in enumerate(messages):
        print(f"  {i}: {msg['role']} - {msg['content'][:50]}...")
    
    print("\nTruncated messages (should be last N messages):")
    for i, msg in enumerate(result):
        print(f"  {i}: {msg['role']} - {msg['content'][:50]}...")
    
    # Check: result should be a suffix of messages
    assert result == messages[-len(result):], "Sliding window failed - result is not the most recent messages!"
    print("\n✅ PASSED: Sliding window correctly keeps most recent messages")
    
    # Check: should have kept at least some messages
    assert len(result) > 0, "Should keep at least one message"
    assert len(result) < len(messages), "Should have truncated some messages"
    print(f"✅ PASSED: Kept {len(result)} out of {len(messages)} messages")


def test_large_message_handling():
    """Test that a single large message is handled correctly."""
    print("\n" + "=" * 80)
    print("Test 2: Large Message Handling")
    print("=" * 80)
    
    # Create a single very large message
    large_content = "x" * 10000  # ~2500 tokens
    messages = [
        {"role": "user", "content": large_content},
    ]
    
    # Set max_tokens to 100 - should still keep the message (even if it exceeds limit)
    max_tokens = 100
    
    result = _truncate_session_messages(
        messages=messages,
        model_name="gpt-4o-mini",
        max_tokens=max_tokens
    )
    
    print(f"Original messages: {len(messages)}")
    print(f"Truncated messages: {len(result)}")
    print(f"Max tokens: {max_tokens}")
    print()
    
    # Should keep at least the last message (even if it exceeds limit)
    assert len(result) == 1, "Should keep at least the last message"
    assert result[0]["content"] == large_content, "Content should be preserved"
    print("✅ PASSED: Large message is preserved (even if exceeds limit)")


def test_normal_operation():
    """Test that normal operation (messages within limit) is not affected."""
    print("\n" + "=" * 80)
    print("Test 3: Normal Operation (No Truncation Needed)")
    print("=" * 80)
    
    # Create messages that fit within limit
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
    ]
    
    # Set max_tokens very high - should keep all messages
    max_tokens = 10000
    
    result = _truncate_session_messages(
        messages=messages,
        model_name="gpt-4o-mini",
        max_tokens=max_tokens
    )
    
    print(f"Original messages: {len(messages)}")
    print(f"Truncated messages: {len(result)}")
    print(f"Max tokens: {max_tokens}")
    print()
    
    # Should keep all messages when within limit
    assert result == messages, "Should keep all messages when within token limit"
    print("✅ PASSED: Normal operation preserved - all messages kept when within limit")


def test_message_format_preservation():
    """Test that message format (role, content) is preserved correctly."""
    print("\n" + "=" * 80)
    print("Test 4: Message Format Preservation")
    print("=" * 80)
    
    messages = [
        {"role": "user", "content": "Test user message"},
        {"role": "assistant", "content": "Test assistant message"},
        {"role": "system", "content": "Test system message"},
    ]
    
    result = _truncate_session_messages(
        messages=messages,
        model_name="gpt-4o-mini",
        max_tokens=1000
    )
    
    print(f"Original messages: {len(messages)}")
    print(f"Truncated messages: {len(result)}")
    print()
    
    # Check format preservation
    for i, msg in enumerate(result):
        assert "role" in msg, f"Message {i} missing 'role'"
        assert "content" in msg, f"Message {i} missing 'content'"
        assert msg["role"] in ["user", "assistant", "system"], f"Invalid role: {msg['role']}"
        print(f"  {i}: {msg['role']} - {msg['content']}")
    
    print("✅ PASSED: Message format (role, content) preserved correctly")


if __name__ == "__main__":
    print("Testing LangChain's trim_messages implementation")
    print("=" * 80)
    print()
    
    try:
        test_sliding_window_behavior()
        test_large_message_handling()
        test_normal_operation()
        test_message_format_preservation()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nLangChain's trim_messages is working correctly:")
        print("  - Sliding window keeps most recent messages")
        print("  - Large messages are handled correctly")
        print("  - Normal operation is not affected")
        print("  - Message format is preserved")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

