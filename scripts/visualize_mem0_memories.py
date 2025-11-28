#!/usr/bin/env python3
"""
Script to visualize mem0 memories stored in the vectorstore.

Usage:
    python scripts/visualize_mem0_memories.py [--user-id USER_ID] [--agent-id AGENT_ID] [--session-id SESSION_ID] [--json] [--all]
"""

import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.backend.mem0_memory_manager import get_mem0_memory_manager
from agent.utils import load_config


def read_faiss_vectorstore(vectorstore_path: Path) -> Dict[str, Any]:
    """Read the FAISS vectorstore files directly to inspect what's stored."""
    info = {
        "faiss_file": None,
        "pkl_file": None,
        "docstore_size": 0,
        "index_size": 0,
        "sample_ids": []
    }
    
    faiss_file = vectorstore_path / "mem0_memories.faiss"
    pkl_file = vectorstore_path / "mem0_memories.pkl"
    
    if pkl_file.exists():
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                info["pkl_file"] = str(pkl_file)
                if isinstance(data, dict):
                    info["docstore"] = data
                    info["docstore_size"] = len(data.get("docstore", {}))
                    # Try to extract some sample IDs
                    docstore = data.get("docstore", {})
                    if docstore:
                        info["sample_ids"] = list(docstore.keys())[:10]
                elif isinstance(data, tuple) and len(data) >= 2:
                    # FAISS format: (docstore, index_to_id)
                    docstore, index_to_id = data[0], data[1]
                    info["docstore"] = docstore
                    info["docstore_size"] = len(docstore) if isinstance(docstore, dict) else 0
                    if isinstance(docstore, dict):
                        info["sample_ids"] = list(docstore.keys())[:10]
        except Exception as e:
            info["pkl_error"] = str(e)
    
    if faiss_file.exists():
        info["faiss_file"] = str(faiss_file)
        info["faiss_exists"] = True
        try:
            import faiss
            index = faiss.read_index(str(faiss_file))
            info["index_size"] = index.ntotal
        except Exception as e:
            info["faiss_error"] = str(e)
    
    return info


def format_memory(memory: dict, index: int) -> str:
    """Format a single memory for display."""
    memory_id = memory.get("id", "N/A")
    memory_text = memory.get("memory", "")
    score = memory.get("score")
    created_at = memory.get("created_at", "")
    updated_at = memory.get("updated_at", "")
    metadata = memory.get("metadata", {})
    owner = memory.get("owner", "")
    
    lines = [
        f"  Memory #{index}",
        f"  ID: {memory_id}",
        f"  Content: {memory_text}",
    ]
    
    if owner:
        lines.append(f"  Owner: {owner}")
    
    if score is not None:
        lines.append(f"  Score: {score:.4f}")
    
    if created_at:
        lines.append(f"  Created: {created_at}")
    
    if updated_at:
        lines.append(f"  Updated: {updated_at}")
    
    if metadata:
        lines.append(f"  Metadata: {json.dumps(metadata, indent=4)}")
    
    return "\n".join(lines)


def visualize_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    output_json: bool = False,
    show_all: bool = False
):
    """Visualize all mem0 memories."""
    # Load config
    config = load_config()
    mem0_config = config.get("memory", {}).get("mem0_memory", {})
    
    if not mem0_config.get("enabled", False):
        print("Warning: mem0_memory is not enabled in config.yaml")
        print("Attempting to load memories anyway...\n")
    
    # Check vectorstore files directly
    vectorstore_path = Path(mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore"))
    print("=" * 80)
    print("VECTORSTORE INSPECTION")
    print("=" * 80)
    vectorstore_info = read_faiss_vectorstore(vectorstore_path)
    print(f"FAISS file: {vectorstore_info.get('faiss_file', 'Not found')}")
    print(f"PKL file: {vectorstore_info.get('pkl_file', 'Not found')}")
    print(f"Index size: {vectorstore_info.get('index_size', 0)} vectors")
    print(f"Docstore size: {vectorstore_info.get('docstore_size', 0)} documents")
    if vectorstore_info.get("sample_ids"):
        print(f"Sample IDs: {vectorstore_info['sample_ids'][:5]}")
    print()
    
    # Initialize mem0 memory manager
    try:
        mem0_manager = get_mem0_memory_manager(
            llm_provider=mem0_config.get("llm_provider", "openai"),
            llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
            llm_temperature=mem0_config.get("llm_temperature", 0.0),
            embedding_provider=mem0_config.get("embedding_provider", "openai"),
            embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
            vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
            vectorstore_path=str(vectorstore_path),
            top_k=mem0_config.get("top_k", 3),
            user_id=user_id or mem0_config.get("user_id", "default_user"),
            agent_id=agent_id or mem0_config.get("agent_id", "email_agent"),
            force_new=False
        )
    except Exception as e:
        print(f"Error initializing mem0 memory manager: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Determine which IDs to use
    # If session_id is provided, use it as user_id
    if session_id:
        user_id = session_id
        print(f"Using session_id '{session_id}' as user_id\n")
    else:
        user_id = user_id or mem0_config.get("user_id", "default_user")
        agent_id = agent_id or mem0_config.get("agent_id", "email_agent")
    
    # Get all memories
    all_memories = []
    
    if show_all:
        # Try to find all possible user_ids/agent_ids by searching common patterns
        print("Searching for memories across different IDs...\n")
        
        # Try default IDs
        for uid in [user_id, "default_user", "session_cli"]:
            for aid in [agent_id, "email_agent", None]:
                try:
                    if aid:
                        memories = mem0_manager.get_all_memories(user_id=uid, agent_id=aid)
                    else:
                        memories = mem0_manager.get_all_memories(user_id=uid, agent_id=None)
                    if memories:
                        print(f"Found {len(memories)} memories for user_id='{uid}', agent_id='{aid}'")
                        all_memories.extend(memories)
                except Exception as e:
                    pass
        
        # Try with just agent_id
        for aid in [agent_id, "email_agent"]:
            try:
                memories = mem0_manager.get_all_memories(user_id=None, agent_id=aid)
                if memories:
                    print(f"Found {len(memories)} memories for agent_id='{aid}'")
                    all_memories.extend(memories)
            except Exception as e:
                pass
    else:
        # Try specific IDs
        print(f"Retrieving memories for user_id='{user_id}', agent_id='{agent_id}'...\n")
        
        try:
            all_memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=agent_id)
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            # Try with just user_id
            try:
                all_memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=None)
            except Exception as e2:
                print(f"Error with user_id only: {e2}")
                all_memories = []
    
    # Remove duplicates based on memory ID
    seen_ids = set()
    unique_memories = []
    for mem in all_memories:
        mem_id = mem.get("id") if isinstance(mem, dict) else None
        if mem_id and mem_id not in seen_ids:
            seen_ids.add(mem_id)
            unique_memories.append(mem)
        elif not mem_id:
            unique_memories.append(mem)
    
    if not unique_memories:
        print("No memories found.")
        print("\nTips:")
        print("  - Try using --session-id with a specific session ID")
        print("  - Try using --all to search across all IDs")
        print("  - Check if memories were actually stored (vectorstore shows 0 documents)")
        return
    
    # Display memories
    if output_json:
        # Output as JSON
        print(json.dumps(unique_memories, indent=2, ensure_ascii=False))
    else:
        # Pretty print
        print("=" * 80)
        print(f"MEM0 MEMORIES ({len(unique_memories)} total, {len(all_memories) - len(unique_memories)} duplicates removed)")
        print("=" * 80)
        print()
        
        for i, memory in enumerate(unique_memories, 1):
            print(format_memory(memory, i))
            print("-" * 80)
            print()
        
        # Summary
        print("=" * 80)
        print(f"Summary: {len(unique_memories)} unique memories found")
        print("=" * 80)
        
        # Get stats
        try:
            stats = mem0_manager.get_stats()
            print("\nMemory System Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"\nCould not retrieve stats: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mem0 memories stored in the vectorstore"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID to filter memories (default: from config)"
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        default=None,
        help="Agent ID to filter memories (default: from config)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID to use as user_id (overrides --user-id)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Search across all common user_id/agent_id combinations"
    )
    
    args = parser.parse_args()
    
    visualize_memories(
        user_id=args.user_id,
        agent_id=args.agent_id,
        session_id=args.session_id,
        output_json=args.json,
        show_all=args.all
    )


if __name__ == "__main__":
    main()
