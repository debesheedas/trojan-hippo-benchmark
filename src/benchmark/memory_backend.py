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
        
        # Get defense type from config
        explicit_memory_config = config.get("memory", {}).get("explicit_memory", {})
        defense_type = explicit_memory_config.get("defense_type", "none")
        
        # Unified structure: initial_memory/{set_number}/{backend}/{defense}/explicit.json
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        source_memory = base_dir / memory_set / "explicit" / defense_type / "explicit.json"
        
        if source_memory.exists():
            # Validate labels for provable_policy defense
            is_provable_policy = defense_type == "provable_policy"
            if is_provable_policy:
                try:
                    with open(source_memory, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                    
                    long_term_memory = memory_data.get("long_term", [])
                    for mem_idx, mem_entry in enumerate(long_term_memory):
                        # Handle both string and dict formats
                        if isinstance(mem_entry, str):
                            # Old format: string without label - error for provable_policy
                            raise ValueError(
                                f"Initial explicit memory entry {mem_idx} is missing required 'label' metadata. "
                                f"All initial memories must have 'label' set to 'T' (Trusted) or 'U' (Untrusted) "
                                f"when using provable_policy defense. "
                                f"Memory set: {memory_set}, Entry text preview: {mem_entry[:50]}..."
                            )
                        elif isinstance(mem_entry, dict):
                            if "label" not in mem_entry:
                                raise ValueError(
                                    f"Initial explicit memory entry {mem_idx} is missing required 'label' metadata. "
                                    f"All initial memories must have 'label' set to 'T' (Trusted) or 'U' (Untrusted) "
                                    f"when using provable_policy defense. "
                                    f"Memory set: {memory_set}, Entry text preview: {mem_entry.get('text', '')[:50]}..."
                                )
                            label = mem_entry.get("label")
                            if label not in ["T", "U"]:
                                raise ValueError(
                                    f"Initial explicit memory entry {mem_idx} has invalid label '{label}'. "
                                    f"Label must be 'T' (Trusted) or 'U' (Untrusted). "
                                    f"Memory set: {memory_set}, Entry text preview: {mem_entry.get('text', '')[:50]}..."
                                )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in explicit memory file {source_memory}: {e}")
            
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
                        # Extract text from dict or use string directly
                        mem_text = mem.get("text", mem) if isinstance(mem, dict) else mem
                        mem_str = str(mem_text)
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
        
        # Get defense type from config
        defense_type = mem0_config.get("defense_type", "none")
        
        # Unified structure: initial_memory/{set_number}/{backend}/{defense}/mem0/
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        source_vectorstore = base_dir / memory_set / "mem0" / defense_type / "mem0"
        
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
  # Use the copied vectorstore
        )
        
        # Validate labels for provable_policy defense
        is_provable_policy = defense_type == "provable_policy"
        if is_provable_policy and source_vectorstore.exists() and source_vectorstore.is_dir():
            try:
                user_id = mem0_config.get("user_id", "vince")
                agent_id = mem0_config.get("agent_id", None)
                memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=agent_id, limit=1000)
                for mem_idx, memory_item in enumerate(memories):
                    if isinstance(memory_item, dict):
                        metadata = memory_item.get("metadata", {})
                        label = metadata.get("label", None)
                        if label is None:
                            memory_text = memory_item.get("memory", "")[:50]
                            raise ValueError(
                                f"Initial mem0 memory entry {mem_idx} is missing required 'label' metadata. "
                                f"All initial memories must have 'label' set to 'T' (Trusted) or 'U' (Untrusted) "
                                f"when using provable_policy defense. "
                                f"Memory set: {memory_set}, Memory text preview: {memory_text}..."
                            )
                        if label not in ["T", "U"]:
                            memory_text = memory_item.get("memory", "")[:50]
                            raise ValueError(
                                f"Initial mem0 memory entry {mem_idx} has invalid label '{label}'. "
                                f"Label must be 'T' (Trusted) or 'U' (Untrusted). "
                                f"Memory set: {memory_set}, Memory text preview: {memory_text}..."
                            )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise  # Re-raise validation errors
                # For other errors (e.g., vectorstore not accessible), just warn
                print(f"⚠️ Could not validate mem0 memory labels: {e}")
        
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
        
        # Get defense type from config
        defense_type = rag_config.get("defense_type", "none")
        
        # Unified structure: initial_memory/{set_number}/{backend}/{defense}/rag_vectorstore/
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        
        # Check for pre-computed vectorstore first (fast path - no API calls!)
        source_vectorstore = base_dir / memory_set / "rag" / defense_type / "rag_vectorstore"
        
        # Try unified structure
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
  # Load existing vectorstore
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
        
        # No pre-computed vectorstore found - generate from JSON (slower, but works)
        print(f"⚠️ No pre-computed vectorstore found. Generating from JSON (this may take a while)...")
        
        # Initialize RAG memory manager
        rag_memory_manager = get_rag_memory_manager(
            embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_config.get("top_k", 3),
            chunk_size=rag_config.get("chunk_size", 512),
            vectorstore_path=vectorstore_path_str,
  # Create new vectorstore for test isolation
        )
        
        chunks_loaded = 0
        
        # Unified structure: initial_memory/{set_number}/{backend}/{defense}/rag.json
        json_file = base_dir / memory_set / "rag" / defense_type / "rag.json"
        
        if json_file.exists():
            print(f"Loading RAG memory set from JSON: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get("chunks", [])
            
            # Check if provable_policy defense is active
            rag_defense_type = config.get("memory", {}).get("rag_memory", {}).get("defense_type", "none")
            is_provable_policy = rag_defense_type == "provable_policy"
            
            if not chunks:
                # Fallback: if "chunks" key doesn't exist, treat whole file as single chunk
                # BUT: Skip this fallback for provable_policy defense (empty memory sets are allowed)
                if is_provable_policy:
                    # Empty memory set is valid for provable_policy - just skip chunk creation
                    chunks = []
                else:
                    # For other defenses, use fallback for backward compatibility
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
                    
                    # Validate label for provable_policy defense
                    if is_provable_policy:
                        if "label" not in doc_metadata:
                            raise ValueError(
                                f"Initial memory chunk {chunk_idx} is missing required 'label' metadata. "
                                f"All initial memories must have 'label' set to 'T' (Trusted) or 'U' (Untrusted) "
                                f"when using provable_policy defense. "
                                f"Memory set: {memory_set}, Chunk text preview: {text[:50]}..."
                            )
                        label = doc_metadata["label"]
                        if label not in ["T", "U"]:
                            raise ValueError(
                                f"Initial memory chunk {chunk_idx} has invalid label '{label}'. "
                                f"Label must be 'T' (Trusted) or 'U' (Untrusted). "
                                f"Memory set: {memory_set}, Chunk text preview: {text[:50]}..."
                            )
                    
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
        
        # Determine source JSON file (unified structure)
        # Try with defense type "none" first (most common)
        defense_type = rag_config.get("defense_type", "none")
        json_file = base_dir / memory_set / "rag" / defense_type / "rag.json"
        
        # If not found, try without defense type (for backward compatibility with unified structure)
        if not json_file.exists():
            json_file = base_dir / memory_set / "rag.json"
        
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
        
        memory_sets = set()
        
        # Check unified structure
        if base_dir.exists():
            for memory_set_dir in base_dir.iterdir():
                if memory_set_dir.is_dir():
                    # Check for rag.json in unified structure: {set}/{backend}/{defense}/rag.json
                    rag_dir = memory_set_dir / "rag"
                    if rag_dir.exists() and rag_dir.is_dir():
                        # Check any defense type subdirectory
                        for defense_dir in rag_dir.iterdir():
                            if defense_dir.is_dir():
                                json_file = defense_dir / "rag.json"
                                if json_file.exists():
                                    memory_sets.add(memory_set_dir.name)
                                    break
                        # Also check for rag.json directly under rag/ (unified without defense)
                        json_file = rag_dir / "rag.json"
                        if json_file.exists():
                            memory_sets.add(memory_set_dir.name)
        
        return sorted(memory_sets)


class ContextMemoryBackend:
    """Backend for context (simple list-based) memory."""
    
    @property
    def name(self) -> str:
        return "context"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize context memory from a memory set.
        
        Args:
            memory_set: Memory set number (e.g., "0", "1", "2") or old format name
        """
        from agent.backend.context_memory_manager import get_context_memory_manager
        
        context_config = config.get("memory", {}).get("context_memory", {})
        if not context_config.get("enabled", False):
            raise ValueError("context_memory must be enabled in config")
        
        # Use test-specific context path
        context_path = test_dir / "context_memory.json"
        context_path_str = str(context_path)
        
        # Get defense type from config
        defense_type = context_config.get("defense_type", "none")
        
        # Unified structure: initial_memory/{set_number}/{backend}/{defense}/context.json
        base_dir = Path("data/benchmark/initial_environment/initial_memory")
        source_context = base_dir / memory_set / "context" / defense_type / "context.json"
        
        # Clear if exists (shouldn't happen with unique test dirs, but safety check)
        if context_path.exists():
            context_path.unlink()
        
        # Copy from source if it exists
        if source_context.exists():
            print(f"📦 Loading context memory set '{memory_set}' from {source_context}...")
            
            # Validate labels for provable_policy defense
            is_provable_policy = defense_type == "provable_policy"
            if is_provable_policy:
                try:
                    with open(source_context, 'r', encoding='utf-8') as f:
                        context_data = json.load(f)
                    
                    history = context_data.get("history", [])
                    for entry_idx, entry in enumerate(history):
                        # Handle both string and dict formats
                        if isinstance(entry, str):
                            # Old format: string without label - error for provable_policy
                            raise ValueError(
                                f"Initial context memory entry {entry_idx} is missing required 'label' metadata. "
                                f"All initial memories must have 'label' set to 'T' (Trusted) or 'U' (Untrusted) "
                                f"when using provable_policy defense. "
                                f"Memory set: {memory_set}, Entry text preview: {entry[:50]}..."
                            )
                        elif isinstance(entry, dict):
                            if "label" not in entry:
                                raise ValueError(
                                    f"Initial context memory entry {entry_idx} is missing required 'label' metadata. "
                                    f"All initial memories must have 'label' set to 'T' (Trusted) or 'U' (Untrusted) "
                                    f"when using provable_policy defense. "
                                    f"Memory set: {memory_set}, Entry text preview: {entry.get('text', '')[:50]}..."
                                )
                            label = entry.get("label")
                            if label not in ["T", "U"]:
                                raise ValueError(
                                    f"Initial context memory entry {entry_idx} has invalid label '{label}'. "
                                    f"Label must be 'T' (Trusted) or 'U' (Untrusted). "
                                    f"Memory set: {memory_set}, Entry text preview: {entry.get('text', '')[:50]}..."
                                )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in context memory file {source_context}: {e}")
            
            shutil.copy2(source_context, context_path)
            print(f"✅ Copied context memory set to: {context_path}")
        else:
            # No source context - create empty one
            print(f"⚠️ Source context not found: {source_context}")
            print(f"   Creating empty context at: {context_path}")
            empty_context = {"history": []}
            with open(context_path, 'w', encoding='utf-8') as f:
                json.dump(empty_context, f, indent=2)
        
        # Initialize manager with the context path
        # Get max_context_length from config if specified
        max_context_length = context_config.get("max_context_length")
        model_name = config.get("agent", {}).get("target_model_name", "gpt-5-mini")
        
        context_manager = get_context_memory_manager(
            context_path=context_path_str,
            max_context_length=max_context_length,
            model_name=model_name,
        )
        
        # Verify and print stats
        try:
            history = context_manager.history
            num_entries = len(history)
            print(f"✅ Loaded {num_entries} context entries from {memory_set}")
            print(f"   📝 Total entries: {num_entries}")
            
            if num_entries > 0:
                print(f"   Sample entries:")
                for i, entry in enumerate(history[:3], 1):
                    # Extract text from dict or use string directly
                    entry_text = entry.get("text", entry) if isinstance(entry, dict) else entry
                    entry_str = str(entry_text)
                    if len(entry_str) > 100:
                        entry_str = entry_str[:100] + "..."
                    print(f"     {i}. {entry_str}")
                if num_entries > 3:
                    print(f"     ... and {num_entries - 3} more")
            else:
                print(f"⚠️ Empty context (no source context found)")
                print(f"   📝 Total entries: 0")
        except Exception as e:
            print(f"⚠️ Could not verify context contents: {e}")
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get context memory contents.
        
        For efficiency, this method first checks for recent entries added in the current step.
        If no recent entries file exists, it falls back to checking all history.
        """
        import json
        
        # First, try to get recent entries (much faster - only checks what was added this step)
        recent_entries_file = test_dir / "context_recent_chunks.json"
        if recent_entries_file.exists():
            try:
                with open(recent_entries_file, 'r', encoding='utf-8') as f:
                    recent_entries = json.load(f)
                if recent_entries:
                    # Return recent entries - these are what were added in the current turn
                    return recent_entries
            except Exception:
                # If we can't read recent entries, fall back to full retrieval
                pass
        
        # Fallback: Get all memory entries from the context file
        # This is slower but ensures we check everything if recent entries aren't available
        from agent.backend.context_memory_manager import get_context_memory_manager
        
        context_config = config.get("memory", {}).get("context_memory", {})
        context_path = str(test_dir / "context_memory.json")
        
        try:
            # Get max_context_length from config if specified
            max_context_length = context_config.get("max_context_length")
            model_name = config.get("agent", {}).get("target_model_name", "gpt-5-mini")
            
            context_manager = get_context_memory_manager(
                context_path=context_path,
                max_context_length=max_context_length,
                model_name=model_name,
            )
            
            # Get all history entries
            if hasattr(context_manager, 'history') and context_manager.history:
                return list(context_manager.history)
            
            return []
        except Exception as e:
            print(f"Warning: Could not read context memory: {e}")
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear context memory."""
        context_path = test_dir / "context_memory.json"
        if context_path.exists():
            context_path.unlink()
        
        # Also clear recent chunks file if it exists
        recent_chunks_file = test_dir / "context_recent_chunks.json"
        if recent_chunks_file.exists():
            recent_chunks_file.unlink()


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
        self.register("context", ContextMemoryBackend)
    
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

