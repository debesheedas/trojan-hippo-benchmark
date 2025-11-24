"""
RAG Memory Loader

Loads initial RAG memory sets (similar to initial inbox/outbox sets).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.backend.rag_memory_manager import get_rag_memory_manager


def load_rag_memory_set(
    rag_memory_set: str,
    config: Dict[str, Any],
    vectorstore_path: Optional[str] = None,
    force_new: bool = True
) -> int:
    """
    Load an initial RAG memory set into the vector store.
    
    Supports multiple formats:
    1. JSON file with pre-chunked data: `rag_memory_set_X.json`
    2. Raw text file: `rag_memory_set_X.txt` (auto-chunked)
    3. Directory of text files: `rag_memory_set_X/` (each file = one chunk)
    
    Args:
        rag_memory_set: Name of the RAG memory set (e.g., "rag_memory_set_0")
        config: Configuration dictionary
        vectorstore_path: Optional custom vectorstore path (uses config default if not provided)
        force_new: If True, create a new vectorstore (default: True for test isolation)
    
    Returns:
        Number of chunks loaded
    """
    rag_config = config.get("memory", {}).get("rag_memory", {})
    if not rag_config.get("enabled", False):
        raise ValueError("RAG memory must be enabled in config")
    
    # Determine vectorstore path
    if vectorstore_path is None:
        base_path = rag_config.get("vectorstore_path", "data/interactive_agent/rag_vectorstore")
        # Use test-specific path to avoid contamination
        vectorstore_path = f"{base_path}_{rag_memory_set}"
    
    # Initialize RAG memory manager
    rag_memory_manager = get_rag_memory_manager(
        embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
        top_k=rag_config.get("top_k", 3),
        chunk_size=rag_config.get("chunk_size", 512),
        vectorstore_path=vectorstore_path,
        force_new=force_new
    )
    
    # Find the RAG memory set file/directory
    base_dir = Path("data/benchmark/initial_rag_memory")
    chunks_loaded = 0
    
    # Try JSON format first (pre-chunked)
    json_file = base_dir / f"{rag_memory_set}.json"
    if json_file.exists():
        print(f"Loading RAG memory set from JSON: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        if not chunks:
            # Fallback: if "chunks" key doesn't exist, treat whole file as single chunk
            text_content = data.get("text", "") or json.dumps(data, indent=2)
            chunks = [{"text": text_content}]
        
        for chunk_data in chunks:
            if isinstance(chunk_data, str):
                # Direct string chunk
                text = chunk_data
                metadata = {"source": rag_memory_set, "type": "initial_memory"}
            elif isinstance(chunk_data, dict):
                # Chunk with metadata
                text = chunk_data.get("text", "")
                metadata = chunk_data.get("metadata", {})
                metadata["source"] = rag_memory_set
                metadata["type"] = "initial_memory"
            else:
                continue
            
            if text and text.strip():
                rag_memory_manager.add_memory(text, metadata=metadata)
                chunks_loaded += 1
        
        print(f"  Loaded {chunks_loaded} chunks from JSON file")
        return chunks_loaded
    
    # Try raw text file (auto-chunk)
    txt_file = base_dir / f"{rag_memory_set}.txt"
    if txt_file.exists():
        print(f"Loading RAG memory set from text file: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Chunk the text
        from benchmark.memory_benchmark_utils import chunk_context_for_memory
        chunk_size = rag_config.get("chunk_size", 512)
        chunks = chunk_context_for_memory(text_content, chunk_size=chunk_size)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = {
                "source": rag_memory_set,
                "type": "initial_memory",
                "chunk_index": chunk_idx,
                "auto_chunked": True
            }
            rag_memory_manager.add_memory(chunk_text, metadata=metadata)
            chunks_loaded += 1
        
        print(f"  Loaded and chunked {chunks_loaded} chunks from text file")
        return chunks_loaded
    
    # Try directory format (each file = one chunk)
    dir_path = base_dir / rag_memory_set
    if dir_path.exists() and dir_path.is_dir():
        print(f"Loading RAG memory set from directory: {dir_path}")
        text_files = sorted(dir_path.glob("*.txt"))
        
        for file_idx, text_file in enumerate(text_files):
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if text_content.strip():
                metadata = {
                    "source": rag_memory_set,
                    "type": "initial_memory",
                    "file_name": text_file.name,
                    "file_index": file_idx
                }
                rag_memory_manager.add_memory(text_content, metadata=metadata)
                chunks_loaded += 1
        
        print(f"  Loaded {chunks_loaded} chunks from directory")
        return chunks_loaded
    
    # Not found
    print(f"Warning: RAG memory set '{rag_memory_set}' not found at:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")
    print(f"  - {dir_path}")
    return 0


def create_rag_memory_set_from_text(
    text: str,
    output_path: str,
    chunk_size: int = 512,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Create a RAG memory set JSON file from raw text.
    
    Args:
        text: Raw text to chunk and save
        output_path: Path to output JSON file
        chunk_size: Chunk size for splitting
        metadata: Optional metadata to include
    """
    from benchmark.memory_benchmark_utils import chunk_context_for_memory
    
    chunks = chunk_context_for_memory(text, chunk_size=chunk_size)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    chunks_data = []
    for chunk_idx, chunk_text in enumerate(chunks):
        chunks_data.append({
            "text": chunk_text,
            "metadata": {
                "chunk_index": chunk_idx,
                **(metadata or {})
            }
        })
    
    data = {
        "chunks": chunks_data,
        "metadata": {
            "total_chunks": len(chunks_data),
            "chunk_size": chunk_size,
            **(metadata or {})
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created RAG memory set: {output_file}")
    print(f"  Total chunks: {len(chunks_data)}")
    return output_file

