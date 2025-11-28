"""
Mem0 Memory Loader

Loads initial mem0 memory sets (similar to initial_memory for simple memory).
Initializes mem0 vectorstore with conversation messages.
"""

import json
import sys
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
    force_new: bool = True
) -> int:
    """
    Load an initial mem0 memory set from a pre-existing vectorstore.
    
    Mem0 stores memories in vectorstores (FAISS, etc.), not JSON files.
    This function checks if a vectorstore exists for the mem0_memory_set and uses it.
    
    Args:
        mem0_memory_set: Name of the mem0 memory set (e.g., "mem0_memory_set_1")
        config: Configuration dictionary
        vectorstore_path: Optional custom vectorstore path (uses config default if not provided)
        force_new: If True, create a new vectorstore (default: True for test isolation)
                    If False and vectorstore exists, use existing vectorstore
    
    Returns:
        Number of memories loaded (0 if vectorstore doesn't exist or is empty)
    """
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    if not mem0_config.get("enabled", False):
        raise ValueError("mem0_memory must be enabled in config")
    
    # Determine vectorstore path
    # If not provided, use the test directory's vectorstore location
    # The source vectorstore is in: data/benchmark/initial_mem0_memory/{mem0_memory_set}/
    if vectorstore_path is None:
        # Default to test directory structure (will be set by test_bench)
        base_path = mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore")
        vectorstore_path = f"{base_path}_{mem0_memory_set}"
    
    vectorstore_path_obj = Path(vectorstore_path)
    
    # Check if vectorstore already exists
    if vectorstore_path_obj.exists() and not force_new:
        # Check if vectorstore has files (FAISS creates .faiss and .pkl files)
        faiss_files = list(vectorstore_path_obj.glob("*.faiss"))
        
        # If vectorstore exists but is empty, try to copy from source
        if not faiss_files:
            source_vectorstore = Path("data/benchmark/initial_mem0_memory") / mem0_memory_set
            if source_vectorstore.exists() and source_vectorstore.is_dir():
                print(f"⚠️ Existing vectorstore is empty, copying from source: {source_vectorstore}")
                import shutil
                if vectorstore_path_obj.exists():
                    shutil.rmtree(vectorstore_path_obj)
                shutil.copytree(source_vectorstore, vectorstore_path_obj)
                print(f"✅ Copied mem0 memory set vectorstore: {mem0_memory_set}")
            else:
                print(f"Using existing mem0 vectorstore: {vectorstore_path} (appears empty, no source to copy from)")
        else:
            print(f"Using existing mem0 vectorstore: {vectorstore_path}")
        
        # Initialize manager to use the vectorstore
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
            force_new=False  # Use existing vectorstore
        )
        
        # Verify memories are actually accessible
        try:
            memories = mem0_manager.get_all_memories(user_id="vince", agent_id=None, limit=10)
            if memories:
                print(f"✅ Loaded mem0 memory set from vectorstore: {mem0_memory_set} ({len(memories)} memories found)")
                return len(memories)
            else:
                # Check if FAISS files exist but no memories found
                faiss_files_after = list(vectorstore_path_obj.glob("*.faiss"))
                if faiss_files_after:
                    print(f"⚠️ Vectorstore has files but no memories found (may be empty or corrupted)")
                else:
                    print(f"⚠️ Vectorstore directory exists but appears empty: {vectorstore_path}")
                return 0
        except Exception as e:
            print(f"⚠️ Could not verify vectorstore contents: {e}")
            # If FAISS files exist, assume it's valid
            if faiss_files:
                return 1
            return 0
    
    # Vectorstore doesn't exist - check if there's a source vectorstore to copy from
    # Look for pre-processed vectorstore in initial_mem0_memory directory
    # The vectorstore is stored directly in: data/benchmark/initial_mem0_memory/{mem0_memory_set}/
    source_vectorstore = Path("data/benchmark/initial_mem0_memory") / mem0_memory_set
    if source_vectorstore.exists() and source_vectorstore.is_dir():
        print(f"Copying pre-processed vectorstore from {source_vectorstore} to {vectorstore_path}...")
        import shutil
        if vectorstore_path_obj.exists():
            shutil.rmtree(vectorstore_path_obj)
        shutil.copytree(source_vectorstore, vectorstore_path_obj)
        print(f"✅ Copied mem0 memory set vectorstore: {mem0_memory_set}")
        
        # Initialize manager with the copied vectorstore
        mem0_manager = get_mem0_memory_manager(
            llm_provider=mem0_config.get("llm_provider", "openai"),
            llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
            llm_temperature=mem0_config.get("llm_temperature", 0.0),
            embedding_provider=mem0_config.get("embedding_provider", "openai"),
            embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
            vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
            vectorstore_path=vectorstore_path,
            top_k=mem0_config.get("top_k", 3),
            user_id="vince",
            agent_id=None,  # Memories are stored with agent_id=None
            force_new=False  # Use the copied vectorstore
        )
        return 1
    
    # No existing vectorstore and no source to copy from
    if force_new:
        # Create empty vectorstore
        print(f"Creating new empty mem0 vectorstore: {vectorstore_path}")
        mem0_manager = get_mem0_memory_manager(
            llm_provider=mem0_config.get("llm_provider", "openai"),
            llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
            llm_temperature=mem0_config.get("llm_temperature", 0.0),
            embedding_provider=mem0_config.get("embedding_provider", "openai"),
            embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
            vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
            vectorstore_path=vectorstore_path,
            top_k=mem0_config.get("top_k", 3),
            user_id="vince",
            agent_id=None,  # Memories are stored with agent_id=None
            force_new=True
        )
        print(f"✅ Created empty mem0 memory set: {mem0_memory_set}")
        return 0
    else:
        print(f"⚠️ Mem0 memory set '{mem0_memory_set}' vectorstore not found at {vectorstore_path}")
        return 0

