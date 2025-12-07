"""
Memory Backend Abstraction

Provides a unified interface for all memory systems (explicit, mem0, rag).
Uses protocol-based design inspired by prompt-siren's architecture.
"""

import json
import shutil
from pathlib import Path
from typing import Protocol, Dict, Any, List, Optional, Type
from datetime import datetime, timezone
import sys

# LangChain imports for RAG (optional, only used when RAG backend is used)
try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None
    FAISS = None

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


class MemoryBackend(Protocol):
    """Protocol defining the interface for memory backends."""
    
    @property
    def name(self) -> str:
        """Backend name: 'explicit', 'mem0', or 'rag'."""
        ...
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """
        Initialize memory from a memory set.
        
        Args:
            memory_set: Name of the memory set (e.g., "memory_set_0")
            test_dir: Test-specific directory for isolated memory storage
            config: Configuration dictionary
        """
        ...
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """
        Get current memory contents as list of strings.
        
        Args:
            test_dir: Test-specific directory
            config: Configuration dictionary
            
        Returns:
            List of memory strings
        """
        ...
    
    def clear_memory(self, test_dir: Path) -> None:
        """
        Clear memory for test isolation.
        
        Args:
            test_dir: Test-specific directory
        """
        ...


class ExplicitMemoryBackend:
    """Backend for explicit (JSON-based) memory."""
    
    @property
    def name(self) -> str:
        return "explicit"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize explicit memory from a memory set.
        
        Args:
            memory_set: Memory set number (e.g., "0", "1", "2") or old format name
        """
        memory_file = test_dir / "agent_memory.json"
        
        # New unified structure: initial_memory/{set_number}/explicit.json
        # Try new structure first
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        source_memory = base_dir / memory_set / "explicit.json"
        
        # Fallback to old structure for backward compatibility
        if not source_memory.exists():
            # Old structure: initial_explicit_memory/{memory_set}.json
            old_base_dir = Path("data/benchmark/initial_environment/initial_explicit_memory")
            source_memory = old_base_dir / f"{memory_set}.json"
        
        # Another fallback: old initial_memory/{memory_set}.json
        if not source_memory.exists():
            source_memory = base_dir / f"{memory_set}.json"
        
        if source_memory.exists():
            shutil.copy2(source_memory, memory_file)
            print(f"✅ Loaded explicit memory set: {memory_set}")
            
            # Print memory contents
            try:
                with open(source_memory, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                
                long_term_memory = memory_data.get("long_term", [])
                num_memories = len(long_term_memory)
                print(f"   📝 Total memories: {num_memories}")
                
                if num_memories > 0:
                    print(f"   Sample memories:")
                    for i, mem in enumerate(long_term_memory[:3], 1):
                        mem_str = str(mem)
                        if len(mem_str) > 100:
                            mem_str = mem_str[:100] + "..."
                        print(f"     {i}. {mem_str}")
                    if num_memories > 3:
                        print(f"     ... and {num_memories - 3} more")
            except Exception as e:
                print(f"   ⚠️ Could not read memory contents: {e}")
        else:
            # Create empty memory file
            empty_memory = {"long_term": []}
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(empty_memory, f, indent=2)
            print(f"⚠️ Memory set '{memory_set}' not found, created empty memory")
            print(f"   📝 Total memories: 0")
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get explicit memory contents."""
        memory_file = test_dir / "agent_memory.json"
        
        if not memory_file.exists():
            return []
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            long_term_memory = memory_data.get("long_term", [])
            return [str(entry) for entry in long_term_memory]
        except Exception as e:
            print(f"Warning: Could not read memory file {memory_file}: {e}")
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear explicit memory."""
        memory_file = test_dir / "agent_memory.json"
        if memory_file.exists():
            empty_memory = {"long_term": []}
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(empty_memory, f, indent=2)


class Mem0MemoryBackend:
    """Backend for mem0 (vectorstore-based) memory."""
    
    @property
    def name(self) -> str:
        return "mem0"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize mem0 memory from a memory set.
        
        Args:
            memory_set: Memory set number (e.g., "0", "1", "2") or old format name
        """
        from agent.backend.mem0_memory_manager import get_mem0_memory_manager
        
        mem0_config = config.get("memory", {}).get("mem0_memory", {})
        if not mem0_config.get("enabled", False):
            raise ValueError("mem0_memory must be enabled in config")
        
        # Use test-specific vectorstore path
        vectorstore_path = str(test_dir / "mem0_vectorstore")
        
        # New unified structure: initial_memory/{set_number}/mem0/
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        source_vectorstore = base_dir / memory_set / "mem0"
        
        # Fallback to old structure for backward compatibility
        if not source_vectorstore.exists() or not source_vectorstore.is_dir():
            # Old structure: initial_mem0_memory/{memory_set}/
            old_base_dir = Path("data/benchmark/initial_environment/initial_mem0_memory")
            source_vectorstore = old_base_dir / memory_set
        
        # Clear if exists (shouldn't happen with unique test dirs, but safety check)
        vectorstore_path_obj = Path(vectorstore_path)
        if vectorstore_path_obj.exists():
            shutil.rmtree(vectorstore_path_obj)
        
        # Copy from source if it exists
        if source_vectorstore.exists() and source_vectorstore.is_dir():
            print(f"📦 Loading mem0 memory set '{memory_set}' from {source_vectorstore}...")
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
            user_id=mem0_config.get("user_id", "vince"),
            agent_id=mem0_config.get("agent_id", None),
            force_new=False  # Use the copied vectorstore
        )
        
        # Verify memories are accessible and print contents
        try:
            user_id = mem0_config.get("user_id", "vince")
            agent_id = mem0_config.get("agent_id", None)
            memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=agent_id, limit=1000)
            if memories:
                num_memories = len(memories)
                print(f"✅ Loaded {num_memories} mem0 memories from {memory_set}")
                print(f"   📝 Total memories: {num_memories}")
                
                if num_memories > 0:
                    print(f"   Sample memories:")
                    for i, mem in enumerate(memories[:3], 1):
                        if isinstance(mem, dict):
                            memory_text = mem.get("memory", "")
                            if not memory_text:
                                memory_text = str(mem)
                        else:
                            memory_text = str(mem)
                        
                        if len(memory_text) > 100:
                            memory_text = memory_text[:100] + "..."
                        print(f"     {i}. {memory_text}")
                    if num_memories > 3:
                        print(f"     ... and {num_memories - 3} more")
            else:
                print(f"⚠️ Empty mem0 vectorstore (no source vectorstore found)")
                print(f"   📝 Total memories: 0")
        except Exception as e:
            print(f"⚠️ Could not verify mem0 vectorstore contents: {e}")
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get mem0 memory contents.
        
        For efficiency, this method first checks for recent memories added in the current step.
        If no recent memories file exists, it falls back to checking all memories from the vectorstore.
        """
        import json
        
        # First, try to get recent memories (much faster - only checks what was added this step)
        recent_memories_file = test_dir / "mem0_recent_memories.json"
        if recent_memories_file.exists():
            try:
                with open(recent_memories_file, 'r', encoding='utf-8') as f:
                    recent_memories = json.load(f)
                if recent_memories:
                    # Return recent memories - these are what were extracted and added in the current turn
                    return recent_memories
            except Exception:
                # If we can't read recent memories, fall back to full retrieval
                pass
        
        # Fallback: Get all memories from the vectorstore
        # This is slower but ensures we check everything if recent memories aren't available
        from agent.backend.mem0_memory_manager import get_mem0_memory_manager
        
        mem0_config = config.get("memory", {}).get("mem0_memory", {})
        vectorstore_path = str(test_dir / "mem0_vectorstore")
        user_id = mem0_config.get("user_id", "vince")
        agent_id = mem0_config.get("agent_id", None)
        
        try:
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                vectorstore_path=vectorstore_path,
                top_k=mem0_config.get("top_k", 3),
                user_id=user_id,
                agent_id=agent_id,
                force_new=False
            )
            
            memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=agent_id, limit=1000)
            
            memory_texts = []
            for memory in memories:
                if isinstance(memory, dict):
                    # Try multiple possible keys for mem0 memory content
                    # mem0 may return memories with different key names depending on version
                    memory_text = (
                        memory.get("memory") or
                        memory.get("memories") or
                        memory.get("text") or
                        memory.get("content") or
                        memory.get("fact") or
                        ""
                    )
                    # If still empty, try to get the first string value
                    if not memory_text:
                        for value in memory.values():
                            if isinstance(value, str) and value.strip():
                                memory_text = value
                                break
                    if memory_text:
                        memory_texts.append(str(memory_text))
                else:
                    memory_texts.append(str(memory))
            
            return memory_texts
        except Exception as e:
            print(f"Warning: Could not read mem0 memory: {e}")
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear mem0 memory."""
        vectorstore_path = test_dir / "mem0_vectorstore"
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)


class RAGMemoryBackend:
    """Backend for RAG (vectorstore-based) memory."""
    
    @property
    def name(self) -> str:
        return "rag"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize RAG memory from a memory set.
        
        First tries to copy a pre-computed vectorstore (fast, no API calls).
        Falls back to generating from JSON if pre-computed vectorstore doesn't exist.
        """
        from agent.backend.rag_memory_manager import get_rag_memory_manager
        
        rag_config = config.get("memory", {}).get("rag_memory", {})
        if not rag_config.get("enabled", False):
            raise ValueError("rag_memory must be enabled in config")
        
        # Use test-specific vectorstore path
        vectorstore_path = test_dir / "rag_vectorstore"
        vectorstore_path_str = str(vectorstore_path)
        
        # New unified structure: initial_memory/{set_number}/
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        old_base_dir = Path("data/benchmark/initial_environment/initial_rag_memory")
        
        # Check for pre-computed vectorstore first (fast path - no API calls!)
        source_vectorstore = base_dir / memory_set / "rag_vectorstore"
        old_source_vectorstore = old_base_dir / f"{memory_set}_vectorstore"
        
        # Try new unified structure first
        if source_vectorstore.exists() and (source_vectorstore / "index.faiss").exists():
            print(f"📦 Loading pre-computed RAG vectorstore from: {source_vectorstore}")
            # Copy pre-computed vectorstore (instant, no API calls)
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
            shutil.copytree(source_vectorstore, vectorstore_path)
            print(f"✅ Copied RAG vectorstore to: {vectorstore_path}")
            
            # Initialize manager with copied vectorstore
            rag_memory_manager = get_rag_memory_manager(
                embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
                top_k=rag_config.get("top_k", 3),
                chunk_size=rag_config.get("chunk_size", 512),
                vectorstore_path=vectorstore_path_str,
                force_new=False  # Load existing vectorstore
            )
            
            # Verify and print stats
            try:
                if rag_memory_manager.vectorstore:
                    num_docs = len(rag_memory_manager.documents) if rag_memory_manager.documents else 0
                    print(f"   📝 Total chunks: {num_docs}")
                    if num_docs > 0:
                        print(f"   Sample chunks:")
                        for i, doc_text in enumerate(rag_memory_manager.documents[:3], 1):
                            sample_text = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                            print(f"     {i}. {sample_text}")
                        if num_docs > 3:
                            print(f"     ... and {num_docs - 3} more")
            except Exception as e:
                print(f"⚠️ Could not verify vectorstore contents: {e}")
            
            return  # Successfully loaded from pre-computed vectorstore
        
        # Fallback: Try old structure
        if old_source_vectorstore.exists() and (old_source_vectorstore / "index.faiss").exists():
            print(f"📦 Loading pre-computed RAG vectorstore from: {old_source_vectorstore}")
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
            shutil.copytree(old_source_vectorstore, vectorstore_path)
            print(f"✅ Copied RAG vectorstore to: {vectorstore_path}")
            
            rag_memory_manager = get_rag_memory_manager(
                embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
                top_k=rag_config.get("top_k", 3),
                chunk_size=rag_config.get("chunk_size", 512),
                vectorstore_path=vectorstore_path_str,
                force_new=False
            )
            return
        
        # No pre-computed vectorstore found - generate from JSON (slower, but works)
        print(f"⚠️ No pre-computed vectorstore found. Generating from JSON (this may take a while)...")
        
        # Initialize RAG memory manager
        rag_memory_manager = get_rag_memory_manager(
            embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_config.get("top_k", 3),
            chunk_size=rag_config.get("chunk_size", 512),
            vectorstore_path=vectorstore_path_str,
            force_new=True  # Create new vectorstore for test isolation
        )
        
        chunks_loaded = 0
        
        # Try new unified structure first
        json_file = base_dir / memory_set / "rag.json"
        
        # Fallback to old structure for backward compatibility
        if not json_file.exists():
            json_file = old_base_dir / f"{memory_set}.json"
        
        if json_file.exists():
            print(f"Loading RAG memory set from JSON: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get("chunks", [])
            if not chunks:
                # Fallback: if "chunks" key doesn't exist, treat whole file as single chunk
                text_content = data.get("text", "") or json.dumps(data, indent=2)
                chunks = [{"text": text_content}]
            
            # Prepare all documents at once for batch processing (much faster!)
            if not LANGCHAIN_AVAILABLE:
                raise ImportError("langchain packages required for RAG memory")
            
            documents = []
            for chunk_idx, chunk_data in enumerate(chunks):
                if isinstance(chunk_data, str):
                    text = chunk_data
                    metadata = {"source": memory_set, "type": "initial_memory"}
                elif isinstance(chunk_data, dict):
                    text = chunk_data.get("text", "")
                    metadata = chunk_data.get("metadata", {})
                    metadata["source"] = memory_set
                    metadata["type"] = "initial_memory"
                else:
                    continue
                
                if text and text.strip():
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = chunk_idx
                    doc_metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
                    doc = Document(page_content=text.strip(), metadata=doc_metadata)
                    documents.append(doc)
                    chunks_loaded += 1
            
            # Batch add all documents at once (much faster than one-by-one)
            # This generates embeddings in batch and saves only once
            if documents:
                print(f"   Adding {chunks_loaded} chunks to vectorstore (batch mode)...")
                with rag_memory_manager._lock:
                    if rag_memory_manager.vectorstore is None:
                        # Initialize vector store with all documents at once
                        rag_memory_manager.vectorstore = FAISS.from_documents(
                            documents, rag_memory_manager.embeddings
                        )
                    else:
                        # Add all documents to existing vector store at once
                        rag_memory_manager.vectorstore.add_documents(documents)
                    
                    # Update internal state
                    rag_memory_manager.documents.extend([doc.page_content for doc in documents])
                    rag_memory_manager._chunk_counter += chunks_loaded
                    
                    # Save once at the end (instead of after each chunk)
                    if rag_memory_manager.vectorstore_path:
                        rag_memory_manager._save_vectorstore()
            
            print(f"✅ Generated {chunks_loaded} chunks from JSON file")
            print(f"   📝 Total chunks: {chunks_loaded}")
            print(f"   💡 Tip: Pre-compute vectorstore to speed up future loads using RAGMemoryBackend.precompute_vectorstore()")
            
            # Print sample chunks
            if chunks_loaded > 0:
                print(f"   Sample chunks:")
                sample_docs = documents[:3]
                for i, doc in enumerate(sample_docs, 1):
                    chunk_text = doc.page_content
                    if len(chunk_text) > 100:
                        chunk_text = chunk_text[:100] + "..."
                    print(f"     {i}. {chunk_text}")
                if chunks_loaded > 3:
                    print(f"     ... and {chunks_loaded - 3} more")
            return
        
        # Try raw text file (auto-chunk) - old structure only
        txt_file = old_base_dir / f"{memory_set}.txt"
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
                    "source": memory_set,
                    "type": "initial_memory",
                    "chunk_index": chunk_idx,
                    "auto_chunked": True
                }
                rag_memory_manager.add_memory(chunk_text, metadata=metadata)
                chunks_loaded += 1
            
            print(f"✅ Loaded and chunked {chunks_loaded} chunks from text file")
            print(f"   📝 Total chunks: {chunks_loaded}")
            
            # Print sample chunks
            if chunks_loaded > 0:
                print(f"   Sample chunks:")
                for i, chunk_text in enumerate(chunks[:3], 1):
                    display_text = chunk_text
                    if len(display_text) > 100:
                        display_text = display_text[:100] + "..."
                    print(f"     {i}. {display_text}")
                if chunks_loaded > 3:
                    print(f"     ... and {chunks_loaded - 3} more")
            return
        
        # Try directory format (each file = one chunk) - old structure only
        dir_path = old_base_dir / memory_set
        if dir_path.exists() and dir_path.is_dir():
            print(f"Loading RAG memory set from directory: {dir_path}")
            text_files = sorted(dir_path.glob("*.txt"))
            
            for file_idx, text_file in enumerate(text_files):
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                if text_content.strip():
                    metadata = {
                        "source": memory_set,
                        "type": "initial_memory",
                        "file_name": text_file.name,
                        "file_index": file_idx
                    }
                    rag_memory_manager.add_memory(text_content, metadata=metadata)
                    chunks_loaded += 1
            
            print(f"✅ Loaded {chunks_loaded} chunks from directory")
            print(f"   📝 Total chunks: {chunks_loaded}")
            
            # Print sample chunks
            if chunks_loaded > 0:
                print(f"   Sample chunks:")
                sample_files = text_files[:3]
                for i, text_file in enumerate(sample_files, 1):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        chunk_text = f.read()
                    display_text = chunk_text
                    if len(display_text) > 100:
                        display_text = display_text[:100] + "..."
                    print(f"     {i}. {display_text}")
                if chunks_loaded > 3:
                    print(f"     ... and {chunks_loaded - 3} more")
            return
        
        # Not found
        print(f"⚠️ RAG memory set '{memory_set}' not found, created empty vectorstore")
        print(f"   📝 Total chunks: 0")
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get RAG memory contents.
        
        For efficiency, this method first checks for recent chunks added in the current step.
        If no recent chunks file exists, it falls back to checking all documents.
        """
        import json
        
        # First, try to get recent chunks (much faster - only checks what was added this step)
        recent_chunks_file = test_dir / "rag_recent_chunks.json"
        if recent_chunks_file.exists():
            try:
                with open(recent_chunks_file, 'r', encoding='utf-8') as f:
                    recent_chunks = json.load(f)
                if recent_chunks:
                    # Return recent chunks - these are what were added in the current turn
                    return recent_chunks
            except Exception:
                # If we can't read recent chunks, fall back to full retrieval
                pass
        
        # Fallback: Get all memory chunks from the documents list
        # This is slower but ensures we check everything if recent chunks aren't available
        from agent.backend.rag_memory_manager import get_rag_memory_manager
        
        rag_config = config.get("memory", {}).get("rag_memory", {})
        vectorstore_path = str(test_dir / "rag_vectorstore")
        
        try:
            rag_memory_manager = get_rag_memory_manager(
                embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
                top_k=rag_config.get("top_k", 3),
                chunk_size=rag_config.get("chunk_size", 512),
                vectorstore_path=vectorstore_path,
                force_new=False
            )
            
            # Get all memory chunks from the documents list
            # RAGMemoryManager stores all chunks in self.documents
            if hasattr(rag_memory_manager, 'documents') and rag_memory_manager.documents:
                # documents is a list of strings (chunk texts)
                return list(rag_memory_manager.documents)
            
            # Fallback: Try to retrieve from vectorstore using a broad query
            if rag_memory_manager.vectorstore:
                try:
                    results = rag_memory_manager.vectorstore.similarity_search("", k=1000)
                    return [doc.page_content for doc in results]
                except Exception:
                    pass
            
            return []
        except Exception as e:
            print(f"Warning: Could not read RAG memory: {e}")
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear RAG memory."""
        vectorstore_path = test_dir / "rag_vectorstore"
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)
    
    @staticmethod
    def precompute_vectorstore(memory_set: str, config: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Pre-compute FAISS vectorstore for a RAG memory set.
        
        This generates embeddings and creates a vectorstore that can be quickly copied
        during benchmark initialization, avoiding slow API calls during test runs.
        
        Args:
            memory_set: Memory set identifier (e.g., "0", "1")
            config: Configuration dictionary with RAG memory settings
            overwrite: If True, overwrite existing pre-computed vectorstore
        
        Returns:
            True if successful, False otherwise
        """
        from agent.backend.rag_memory_manager import get_rag_memory_manager
        
        rag_config = config.get("memory", {}).get("rag_memory", {})
        if not rag_config.get("enabled", False):
            print(f"⚠️ RAG memory not enabled in config. Enabling temporarily...")
            rag_config["enabled"] = True
        
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        old_base_dir = Path("data/benchmark/initial_environment/initial_rag_memory")
        
        # Determine source JSON file
        json_file = base_dir / memory_set / "rag.json"
        if not json_file.exists():
            json_file = old_base_dir / f"{memory_set}.json"
            if not json_file.exists():
                json_file = old_base_dir / f"rag_memory_set_{memory_set}.json"
        
        if not json_file.exists():
            print(f"❌ JSON file not found for memory set '{memory_set}'")
            return False
        
        # Determine output vectorstore path (new unified structure)
        output_vectorstore = base_dir / memory_set / "rag_vectorstore"
        
        # Check if already exists
        if output_vectorstore.exists() and (output_vectorstore / "index.faiss").exists():
            if not overwrite:
                print(f"⏭️  Pre-computed vectorstore already exists for '{memory_set}' at {output_vectorstore}")
                print(f"   Use overwrite=True to regenerate")
                return True
            else:
                print(f"🔄 Overwriting existing vectorstore for '{memory_set}'...")
        
        print(f"\n{'='*80}")
        print(f"Pre-computing vectorstore for memory set: {memory_set}")
        print(f"{'='*80}")
        print(f"Source JSON: {json_file}")
        print(f"Output vectorstore: {output_vectorstore}")
        
        # Load JSON
        print(f"\n📖 Loading JSON file...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        if not chunks:
            text_content = data.get("text", "") or json.dumps(data, indent=2)
            chunks = [{"text": text_content}]
        
        print(f"   Found {len(chunks)} chunks")
        
        # Prepare documents
        print(f"📝 Preparing documents...")
        documents = []
        for chunk_idx, chunk_data in enumerate(chunks):
            if isinstance(chunk_data, str):
                text = chunk_data
                metadata = {"source": memory_set, "type": "initial_memory"}
            elif isinstance(chunk_data, dict):
                text = chunk_data.get("text", "")
                metadata = chunk_data.get("metadata", {})
                metadata["source"] = memory_set
                metadata["type"] = "initial_memory"
            else:
                continue
            
            if text and text.strip():
                doc_metadata = metadata.copy()
                doc_metadata["chunk_id"] = chunk_idx
                doc_metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
                doc = Document(page_content=text.strip(), metadata=doc_metadata)
                documents.append(doc)
        
        print(f"   Prepared {len(documents)} documents")
        
        # Initialize RAG memory manager
        print(f"🔧 Initializing RAG memory manager...")
        vectorstore_path_str = str(output_vectorstore)
        rag_memory_manager = get_rag_memory_manager(
            embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_config.get("top_k", 3),
            chunk_size=rag_config.get("chunk_size", 512),
            vectorstore_path=vectorstore_path_str,
            force_new=True
        )
        
        # Generate embeddings and create vectorstore (batch mode)
        print(f"🚀 Generating embeddings and creating vectorstore (batch mode)...")
        print(f"   This may take a while for large memory sets...")
        
        with rag_memory_manager._lock:
            rag_memory_manager.vectorstore = FAISS.from_documents(
                documents, rag_memory_manager.embeddings
            )
            
            # Update internal state
            rag_memory_manager.documents = [doc.page_content for doc in documents]
            rag_memory_manager._chunk_counter = len(documents)
            
            # Save vectorstore
            output_vectorstore.mkdir(parents=True, exist_ok=True)
            rag_memory_manager._save_vectorstore()
        
        print(f"✅ Successfully pre-computed vectorstore!")
        print(f"   📁 Location: {output_vectorstore}")
        print(f"   📝 Chunks: {len(documents)}")
        
        # Calculate size
        if output_vectorstore.exists():
            total_size = sum(f.stat().st_size for f in output_vectorstore.rglob('*') if f.is_file())
            print(f"   💾 Size: {total_size / 1024 / 1024:.2f} MB")
        
        return True
    
    @staticmethod
    def find_rag_memory_sets() -> List[str]:
        """Find all available RAG memory sets."""
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        old_base_dir = Path("data/benchmark/initial_environment/initial_rag_memory")
        
        memory_sets = set()
        
        # Check new unified structure
        if base_dir.exists():
            for memory_set_dir in base_dir.iterdir():
                if memory_set_dir.is_dir():
                    json_file = memory_set_dir / "rag.json"
                    if json_file.exists():
                        memory_sets.add(memory_set_dir.name)
        
        # Check old structure
        if old_base_dir.exists():
            for json_file in old_base_dir.glob("*.json"):
                # Extract memory set name (e.g., "rag_memory_set_1.json" -> "1")
                name = json_file.stem
                if name.startswith("rag_memory_set_"):
                    memory_sets.add(name.replace("rag_memory_set_", ""))
                else:
                    memory_sets.add(name)
        
        return sorted(memory_sets)


class MemoryBackendRegistry:
    """Registry for memory backends (inspired by prompt-siren's registry pattern)."""
    
    def __init__(self):
        self._registry: Dict[str, Type[MemoryBackend]] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default memory backends."""
        self.register("explicit", ExplicitMemoryBackend)
        self.register("mem0", Mem0MemoryBackend)
        self.register("rag", RAGMemoryBackend)
    
    def register(self, name: str, backend_class: Type[MemoryBackend]) -> None:
        """
        Register a memory backend.
        
        Args:
            name: Backend name (e.g., "explicit", "mem0", "rag")
            backend_class: Backend class implementing MemoryBackend protocol
        """
        if name in self._registry:
            raise ValueError(f"Memory backend '{name}' is already registered")
        self._registry[name] = backend_class
    
    def create(self, name: str, config: Dict[str, Any]) -> MemoryBackend:
        """
        Create a memory backend instance.
        
        Args:
            name: Backend name
            config: Configuration dictionary
            
        Returns:
            MemoryBackend instance
            
        Raises:
            ValueError: If backend name is not registered
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Memory backend '{name}' is not registered. "
                f"Available backends: {available}"
            )
        
        backend_class = self._registry[name]
        return backend_class()
    
    def get_registered_backends(self) -> List[str]:
        """Get list of registered backend names."""
        return list(self._registry.keys())


# Global registry instance
_global_registry: Optional[MemoryBackendRegistry] = None


def get_memory_backend_registry() -> MemoryBackendRegistry:
    """Get the global memory backend registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MemoryBackendRegistry()
    return _global_registry

