#!/usr/bin/env python3
"""
Simple script to view mem0 memories using the get_all API.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.backend.mem0_memory_manager import get_mem0_memory_manager
from agent.utils import load_config

def main():
    # Load config
    config = load_config()
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    
    # Initialize mem0 memory manager
    mem0_manager = get_mem0_memory_manager(
        llm_provider=mem0_config.get("llm_provider", "openai"),
        llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
        llm_temperature=mem0_config.get("llm_temperature", 0.0),
        embedding_provider=mem0_config.get("embedding_provider", "openai"),
        embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
        vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
        vectorstore_path=mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore"),
        top_k=mem0_config.get("top_k", 3),
        user_id="vince",  # Hardcoded
        agent_id=mem0_config.get("agent_id", "email_agent"),
        force_new=False
    )
    
    # Get all memories for user "vince"
    print("=" * 80)
    print("MEM0 MEMORIES FOR USER: vince")
    print("=" * 80)
    print()
    
    memories = mem0_manager.get_all_memories(user_id="vince", limit=1000)
    
    if not memories:
        print("No memories found.")
        return
    
    print(f"Total memories: {len(memories)}\n")
    
    for i, memory in enumerate(memories, 1):
        print("-" * 80)
        print(f"Memory #{i}")
        print("-" * 80)
        
        memory_id = memory.get("id", "N/A")
        memory_text = memory.get("memory", "N/A")
        created_at = memory.get("created_at", "N/A")
        updated_at = memory.get("updated_at", "N/A")
        metadata = memory.get("metadata", {})
        
        print(f"ID: {memory_id}")
        print(f"Content: {memory_text}")
        print(f"Created: {created_at}")
        if updated_at:
            print(f"Updated: {updated_at}")
        if metadata:
            import json
            print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        print()
    
    print("=" * 80)
    print(f"Summary: {len(memories)} memories found")
    print("=" * 80)

if __name__ == "__main__":
    main()
