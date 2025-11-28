#!/usr/bin/env python3
"""
Inspect why persona memory test cases failed by checking:
1. Was crucial information indexed in mem0?
2. Was it retrieved into context?
3. Was it presented to agent but not utilized?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.backend.mem0_memory_manager import get_mem0_memory_manager
from agent.utils import load_config

def main():
    config = load_config()
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    
    # Load the memory set
    mem0_manager = get_mem0_memory_manager(
        llm_provider=mem0_config.get("llm_provider", "openai"),
        llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
        llm_temperature=mem0_config.get("llm_temperature", 0.0),
        embedding_provider=mem0_config.get("embedding_provider", "openai"),
        embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
        vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
        vectorstore_path="data/benchmark/initial_mem0_memory/mem0_memory_set_1",
        top_k=15,
        user_id="vince",
        agent_id=None,
        force_new=False
    )
    
    print("="*80)
    print("INSPECTING PERSONA MEMORY TEST FAILURES")
    print("="*80)
    print()
    
    # Get all memories
    all_memories = mem0_manager.get_all_memories(user_id="vince", limit=1000)
    print(f"Total memories in mem0_memory_set_1: {len(all_memories)}\n")
    
    # Test Case 1: Art project with cardboard cut-outs
    print("="*80)
    print("TEST CASE 1: Art Project (Cardboard Cut-outs)")
    print("="*80)
    print()
    print("Expected correct answer mentions: 'cardboard cut-outs', 'painted cardboard', 'collage'")
    print()
    
    # Check if indexed
    cardboard_found = False
    for mem in all_memories:
        mem_text = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
        if "cardboard" in mem_text.lower() or "cut-out" in mem_text.lower() or ("painted" in mem_text.lower() and "collage" in mem_text.lower()):
            cardboard_found = True
            print(f"✅ FOUND IN MEMORIES: {mem_text}")
            break
    
    if not cardboard_found:
        print("❌ NOT FOUND IN MEMORIES: No memory mentions cardboard/cut-outs/collage")
        print("   This suggests the information was either:")
        print("   1. Never in the original chat history")
        print("   2. In chat history but mem0 didn't extract it")
        print("   3. The correct answer is based on inference, not explicit memory")
    
    print()
    # Check if retrieved
    query1 = "How can I organize a collaborative art project for a community festival that lets young children contribute their own clay creations while making sure the process stays enjoyable and free from safety hazards?"
    context1 = mem0_manager.get_context(query1, user_id="vince")
    print(f"Query: {query1[:100]}...")
    print()
    print("Retrieved context:")
    if context1:
        print(context1)
        if "cardboard" in context1.lower() or "cut-out" in context1.lower():
            print("✅ Cardboard information IS in retrieved context")
        else:
            print("❌ Cardboard information NOT in retrieved context")
    else:
        print("❌ No context retrieved")
    
    print()
    print("="*80)
    print("TEST CASE 3: Sleep/Nighttime (House Check)")
    print("="*80)
    print()
    print("Expected correct answer mentions: 'house check', 'locked tight', 'mom or dad'")
    print()
    
    # Check if indexed
    house_check_found = False
    for mem in all_memories:
        mem_text = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
        if ("house" in mem_text.lower() and "check" in mem_text.lower()) or "locked tight" in mem_text.lower():
            house_check_found = True
            print(f"✅ FOUND IN MEMORIES: {mem_text}")
            break
    
    if not house_check_found:
        print("❌ NOT FOUND IN MEMORIES: No memory mentions house check/locked tight")
        print("   This suggests the information was either:")
        print("   1. Never in the original chat history")
        print("   2. In chat history but mem0 didn't extract it")
        print("   3. The correct answer is based on inference, not explicit memory")
    
    print()
    # Check if retrieved
    query3 = "How can I stop feeling on edge at night and relax enough to get a good night's sleep?"
    context3 = mem0_manager.get_context(query3, user_id="vince")
    print(f"Query: {query3}")
    print()
    print("Retrieved context:")
    if context3:
        print(context3)
        if "house check" in context3.lower() or "locked tight" in context3.lower():
            print("✅ House check information IS in retrieved context")
        else:
            print("❌ House check information NOT in retrieved context")
    else:
        print("❌ No context retrieved")
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    if not cardboard_found and not house_check_found:
        print("❌ ROOT CAUSE: The crucial information was NEVER INDEXED in mem0 memories")
        print("   This means the information either:")
        print("   - Wasn't in the original persona chat history")
        print("   - Was in chat history but mem0's extraction didn't capture it")
        print("   - The correct answers are based on expected inference, not explicit facts")
    elif cardboard_found or house_check_found:
        if not context1 or "cardboard" not in context1.lower():
            print("⚠️ PARTIAL ISSUE: Information indexed but NOT RETRIEVED")
        else:
            print("⚠️ PARTIAL ISSUE: Information indexed and retrieved but agent didn't use it")

if __name__ == "__main__":
    main()

