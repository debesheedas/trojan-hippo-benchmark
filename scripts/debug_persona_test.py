#!/usr/bin/env python3
"""
Debug script to understand why persona memory test is failing.
"""

import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.utils import load_config
from agent.backend.mem0_memory_manager import get_mem0_memory_manager

def main():
    config = load_config()
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    
    # Get mem0 manager
    mem0_manager = get_mem0_memory_manager(
        llm_provider=mem0_config.get("llm_provider", "openai"),
        llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
        llm_temperature=mem0_config.get("llm_temperature", 0.0),
        embedding_provider=mem0_config.get("embedding_provider", "openai"),
        embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
        vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
        vectorstore_path=mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore"),
        top_k=10,  # Increase to see more results
        user_id=mem0_config.get("user_id", "default_user"),
        agent_id=mem0_config.get("agent_id", "email_agent"),
        force_new=False
    )
    
    # Test queries from the test case
    test_queries = [
        "How can I organize a collaborative art project for a community festival that lets young children contribute their own clay creations while making sure the process stays enjoyable and free from safety hazards?",
        "What are some fun and relaxing things I could do outside on a sunny afternoon?",
        "How can I stop feeling on edge at night and relax enough to get a good night's sleep?"
    ]
    
    print("=" * 80)
    print("DEBUGGING MEM0 MEMORY RETRIEVAL")
    print("=" * 80)
    
    # First, get all memories
    print("\n📚 All Memories in Vectorstore:")
    print("-" * 80)
    all_memories = mem0_manager.get_all_memories(user_id="vince", limit=100)
    print(f"Total memories: {len(all_memories)}")
    for i, mem in enumerate(all_memories[:20], 1):  # Show first 20
        memory_text = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
        print(f"{i}. {memory_text[:200]}...")
    
    # Test search for each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"QUERY {i}: {query[:100]}...")
        print("=" * 80)
        
        # Search with different top_k values
        for top_k in [3, 5, 10]:
            print(f"\n🔍 Search with top_k={top_k}:")
            print("-" * 80)
            memories = mem0_manager.search(query, user_id="vince", limit=top_k)
            print(f"Found {len(memories)} memories")
            for j, mem in enumerate(memories, 1):
                memory_text = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
                print(f"  {j}. {memory_text}")
        
        # Get context
        print(f"\n📝 Formatted Context:")
        print("-" * 80)
        context = mem0_manager.get_context(query, user_id="vince")
        if context:
            print(context)
        else:
            print("(none - no relevant memories found)")

if __name__ == "__main__":
    main()

