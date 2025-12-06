"""
Mem0 Memory Loader

Loads initial mem0 memory sets (similar to initial_memory for simple memory).
Initializes mem0 vectorstore with conversation messages.

Each test case gets a unique directory, so we simply copy from source to the test directory.
No clearing needed - each test is completely isolated.
"""

import json
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.backend.mem0_memory_manager import get_mem0_memory_manager


def load_mem0_memory_set(
    mem0_memory_set: str,
    config: Dict[str, Any],
    vectorstore_path: Optional[str] = None,
    force_new: bool = True  # Kept for API compatibility, but not used
) -> int:
    """
    Load an initial mem0 memory set from a pre-existing vectorstore.
    
    Since each test case gets a unique directory, we simply copy from source.
    No clearing needed - each test is completely isolated.
    
    Args:
        mem0_memory_set: Name of the mem0 memory set (e.g., "mem0_memory_set_1")
        config: Configuration dictionary
        vectorstore_path: Path to the test-specific vectorstore directory (should be unique per test)
        force_new: Kept for API compatibility, but not used (each test is already isolated)
    
    Returns:
        Number of memories loaded (0 if vectorstore doesn't exist or is empty)
    """
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    if not mem0_config.get("enabled", False):
        raise ValueError("mem0_memory must be enabled in config")
    
    # Determine vectorstore path
    if vectorstore_path is None:
        base_path = mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore")
        vectorstore_path = f"{base_path}_{mem0_memory_set}"
    
    vectorstore_path_obj = Path(vectorstore_path)
    
    # Source vectorstore location
    source_vectorstore = Path("data/benchmark/initial_mem0_memory") / mem0_memory_set
    
    # Since each test gets a unique directory (UUID-based), the vectorstore path should not exist yet
    # If it does exist, it could be:
    # 1. A UUID collision (extremely rare with 8 hex chars = 4.3B possibilities)
    # 2. A previous failed cleanup from an interrupted test run
    # 3. A bug where the directory was created before this function was called
    # In any case, we clear it to ensure a clean state
    if vectorstore_path_obj.exists():
        print(f"⚠️ Warning: Vectorstore path already exists: {vectorstore_path}")
        print(f"   This is unexpected with unique test directories. Clearing it to ensure clean state...")
        shutil.rmtree(vectorstore_path_obj)
    
    # Copy from source if it exists
    if source_vectorstore.exists() and source_vectorstore.is_dir():
        print(f"📦 Loading mem0 memory set '{mem0_memory_set}' from {source_vectorstore}...")
        shutil.copytree(source_vectorstore, vectorstore_path_obj)
        print(f"✅ Copied mem0 memory set vectorstore to: {vectorstore_path}")
    else:
        # No source vectorstore - create empty one
        print(f"⚠️ Source vectorstore not found: {source_vectorstore}")
        print(f"   Creating empty vectorstore at: {vectorstore_path}")
        vectorstore_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Initialize manager with the vectorstore
    mem0_manager = get_mem0_memory_manager(
        llm_provider=mem0_config.get("llm_provider", "openai"),
        llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
        llm_temperature=mem0_config.get("llm_temperature", 0.0),
        embedding_provider=mem0_config.get("embedding_provider", "openai"),
        embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
        vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
        vectorstore_path=vectorstore_path,
        top_k=mem0_config.get("top_k", 3),
        user_id="vince",  # Hardcoded
        agent_id=None,  # Memories are stored with agent_id=None
        force_new=False  # Use the copied vectorstore
    )
    
    # Verify memories are accessible
    try:
        memories = mem0_manager.get_all_memories(user_id="vince", agent_id=None, limit=1000)
        if memories:
            print(f"✅ Loaded {len(memories)} memories from {mem0_memory_set}")
            if mem0_config.get("mem0_print", False):
                print(f"\n📝 Sample memories:")
                for i, memory in enumerate(memories[:5], 1):
                    memory_text = memory.get("memory", "")[:100] if isinstance(memory, dict) else str(memory)[:100]
                    print(f"   {i}. {memory_text}...")
                if len(memories) > 5:
                    print(f"   ... and {len(memories) - 5} more memories")
            return len(memories)
        else:
            # Check if FAISS files exist
            faiss_files = list(vectorstore_path_obj.glob("*.faiss"))
            if faiss_files:
                print(f"⚠️ Vectorstore has files but no memories found (may be empty or corrupted)")
            else:
                print(f"⚠️ Empty vectorstore created (no source vectorstore found)")
            return 0
    except Exception as e:
        print(f"⚠️ Could not verify vectorstore contents: {e}")
        import traceback
        traceback.print_exc()
        # If FAISS files exist, assume it's valid
        faiss_files = list(vectorstore_path_obj.glob("*.faiss"))
        if faiss_files:
            print(f"   (FAISS files exist, assuming vectorstore is valid)")
            return 1
        return 0
