#!/usr/bin/env python3
"""
Quick test to verify the memory tool works correctly.
"""

from memory_tools import create_memory_tools
from backend.memory_manager import get_memory_manager

def test_memory_tool():
    """Test the memory tool."""
    
    # Create memory tool
    memory_tools = create_memory_tools()
    update_tool = memory_tools[0]
    
    print("Testing Memory Tool...")
    print("-" * 50)
    
    # Test 1: Add memory
    print("\n1. Testing memory addition:")
    result = update_tool._run("User wants all emails signed with 'Love from Tina'")
    print(f"   Result: {result}")
    
    # Test 2: Check it was saved
    memory_manager = get_memory_manager()
    context = memory_manager.get_context()
    print(f"\n2. Current long-term memory:")
    for item in context['long_term']:
        print(f"   - {item}")
    
    # Test 3: Forget
    print("\n3. Testing forget:")
    result = update_tool._run("Forget email signature")
    print(f"   Result: {result}")
    
    # Test 4: Check it was removed
    context = memory_manager.get_context()
    print(f"\n4. After forget, long-term memory:")
    if context['long_term']:
        for item in context['long_term']:
            print(f"   - {item}")
    else:
        print("   (empty)")
    
    print("\n" + "-" * 50)
    print("✓ Memory tool test complete!")

if __name__ == "__main__":
    test_memory_tool()

