"""
RAG Memory Module

Combines RAG Defense Manager and RAG Memory Manager functionality.

RAG Defense Manager:
Manages defense mechanisms for RAG memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. none: No defense (normal RAG execution - all messages indexed)
3. user_prompt_only: Only index user messages (filter out assistant messages)
4. limit_chunk_size: Limit chunk sizes to 8 tokens each
5. no_untrusted_tools: Disable RAG indexing for the rest of the session once an
   untrusted tool has been used

RAG Memory Manager:
Embedding-based RAG Memory Manager Module
Implements vector-based retrieval-augmented memory for the email agent.

Features:
- Vector store for semantic search over conversation history
- Incremental memory updates (chunks added over time)
- Top-K retrieval for relevant context
- FAISS-based vector storage for efficient similarity search
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import json
from datetime import datetime, timezone
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from agent.utils import debug_info, debug_debug, debug_print_exception


class RAGDefenseManager:
    """
    Manages defense mechanisms for RAG memory indexing.
    """

    def __init__(
        self,
        defense_type: str = "none"
    ):
        """
        Initialize the RAG defense manager.

        Args:
            defense_type: Type of defense to apply. Options:
                - "none": No defense (default behavior - all messages indexed)
                - "disable_memory": Disable all memory indexing
                - "user_prompt_only": Only index user messages
                - "limit_chunk_size": Limit chunk sizes to 8 tokens
                - "no_untrusted_tools": Disable indexing once an untrusted tool
                  has been used in the session
        """
        self.defense_type = defense_type

    def should_index_memory(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        """
        Determine if memory should be indexed based on defense type.

        Args:
            session_id: Session identifier
            user_message: User message text
            assistant_message: Assistant message text

        Returns:
            True if memory should be indexed, False otherwise
        """
        # Defense 1: Disable memory (baseline)
        if self.defense_type == "disable_memory":
            return False

        # Defense 5: No untrusted tools – once any untrusted tool is used in
        # the session, disable RAG indexing for the rest of the session.
        if self.defense_type == "no_untrusted_tools":
            # Check session trust status using SessionTrustManager
            from agent.agent_core import SessionTrustManager
            if not SessionTrustManager.is_trusted(session_id):
                return False

        # Other defenses allow indexing (they filter/modify instead)
        return True

    def filter_conversation_turn(
        self,
        user_message: str,
        assistant_message: str,
    ) -> str:
        """
        Filter or modify conversation turn based on defense type.

        Args:
            user_message: User message text
            assistant_message: Assistant message text

        Returns:
            Filtered conversation turn text to be indexed
        """
        # Defense 2: User-only indexing
        if self.defense_type == "user_prompt_only":
            # Only include user message, ignore assistant message
            return f"User: {user_message}"

        # No defense or other defenses: include both messages
        return f"User: {user_message}\nAssistant: {assistant_message}"

    def get_chunk_size(
        self,
        default_chunk_size: int = 512,
        limit_memory_size: int = 80,
    ) -> int:
        """
        Get the chunk size to use based on defense type.

        Args:
            default_chunk_size: Default chunk size from config
            limit_memory_size: Maximum characters for limit_memory_length defense (from config)

        Returns:
            Chunk size to use (in characters)
        """
        # Defense: limit_chunk_size - Limit chunk size to configured limit
        if self.defense_type == "limit_chunk_size":
            return limit_memory_size

        # Other defenses use default chunk size
        return default_chunk_size


class RAGMemoryManager:
    """
    Manages embedding-based RAG memory using vector stores for semantic retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 8,
        chunk_size: int = 512,
        vectorstore_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the RAG memory manager.
        
        Args:
            embedding_model: Name of the embedding model to use
            top_k: Number of top documents to retrieve
            chunk_size: Size of text chunks in tokens
            vectorstore_path: Optional path to persist vector store
            api_key: Optional OpenAI API key (uses env var if not provided)
        """
        
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.vectorstore_path = Path(vectorstore_path) if vectorstore_path else None
        self._lock = threading.Lock()
        
        # Initialize embedding model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=SecretStr(api_key)
        )
        
        # Initialize vector store
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[str] = []  # Store raw documents for reference
        self._chunk_counter = 0
        
        # Load existing vector store if path provided
        if self.vectorstore_path and self.vectorstore_path.exists():
            self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load vector store from disk if it exists."""
        try:
            if self.vectorstore_path and (self.vectorstore_path / "index.faiss").exists():
                self.vectorstore = FAISS.load_local(
                    str(self.vectorstore_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Load document metadata
                metadata_file = self.vectorstore_path / "documents.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.documents = data.get("documents", [])
                        self._chunk_counter = data.get("chunk_counter", 0)
                debug_debug(f"Loaded existing vector store from {self.vectorstore_path}")
        except Exception as e:
            debug_info(f"Could not load vector store from {self.vectorstore_path} (will create new one)")
            debug_print_exception(e, context=f"Loading RAG vectorstore from {self.vectorstore_path}", include_traceback=True)
            self.vectorstore = None
    
    def _save_vectorstore(self):
        """Save vector store to disk if path is configured."""
        if self.vectorstore and self.vectorstore_path:
            try:
                self.vectorstore_path.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(self.vectorstore_path))
                # Save document metadata
                metadata_file = self.vectorstore_path / "documents.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "documents": self.documents,
                        "chunk_counter": self._chunk_counter,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }, f, indent=2)
            except Exception as e:
                debug_info(f"Could not save vector store to {self.vectorstore_path}")
                debug_print_exception(e, context=f"Saving RAG vectorstore to {self.vectorstore_path}", include_traceback=True)
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None):
        """
        Add text to the RAG memory system.
        
        Args:
            text: Text content to add to memory
            metadata: Optional metadata dictionary for the document
            session_id: Optional session ID for provable_policy defense (P3: Memory Labeling)
            defense_type: Optional defense type to check if provable_policy is active
        """
        # Use batch method for efficiency (single document batch)
        self.add_memories_batch([text], metadata, session_id, defense_type)
    
    def add_memories_batch(self, texts: List[str], metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None):
        """
        Add multiple texts to the RAG memory system in a single batch.
        This is more efficient than calling add_memory multiple times.
        
        Args:
            texts: List of text contents to add to memory
            metadata: Optional metadata dictionary (applied to all documents)
            session_id: Optional session ID for provable_policy defense (P3: Memory Labeling)
            defense_type: Optional defense type to check if provable_policy is active
        """
        with self._lock:
            # Filter out empty texts
            valid_texts = [t.strip() for t in texts if t and t.strip()]
            if not valid_texts:
                return
            
            # Create documents with metadata
            docs = []
            start_chunk_id = self._chunk_counter
            for i, text in enumerate(valid_texts):
                doc_metadata = (metadata or {}).copy()
                doc_metadata["chunk_id"] = start_chunk_id + i
                doc_metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
                
                # P3: Memory Labeling - inherit session label for provable_policy defense
                if defense_type == "provable_policy":
                    if session_id:
                        from agent.agent_core import ProvablePolicyManager
                        session_label = ProvablePolicyManager.get_session_label(session_id)
                        doc_metadata["label"] = session_label
                    else:
                        # Default to T if no session_id provided
                        doc_metadata["label"] = "T"
                elif "label" not in doc_metadata:
                    # For non-provable_policy defenses, don't require label
                    # But if label is explicitly provided, keep it
                    pass
                
                doc = Document(page_content=text, metadata=doc_metadata)
                docs.append(doc)
            
            # Add to vector store (batch operation - more efficient)
            if self.vectorstore is None:
                # Initialize vector store with first batch of documents
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            else:
                # Add to existing vector store (batched - single embedding API call for all docs)
                self.vectorstore.add_documents(docs)
            
            # Store document references
            for text in valid_texts:
                self.documents.append(text)
                self._chunk_counter += 1
            
            # Track recent chunks for efficient validation
            # Store in a file in the test directory (if vectorstore_path is in a test directory)
            if self.vectorstore_path:
                try:
                    vectorstore_path_obj = Path(self.vectorstore_path)
                    # Check if this is a test directory (contains "test_env" or "rag_vectorstore" in test_envs)
                    if "test_env" in str(vectorstore_path_obj) or "test_envs" in str(vectorstore_path_obj):
                        # Get the test directory (parent of rag_vectorstore)
                        test_dir = vectorstore_path_obj.parent
                        recent_chunks_file = test_dir / "rag_recent_chunks.json"
                        
                        # Load existing recent chunks
                        recent_chunks = []
                        if recent_chunks_file.exists():
                            try:
                                with open(recent_chunks_file, 'r', encoding='utf-8') as f:
                                    recent_chunks = json.load(f)
                            except Exception:
                                recent_chunks = []
                        
                        # Add new chunks to recent chunks list
                        for i, text in enumerate(valid_texts):
                            recent_chunks.append({
                                "text": text,
                                "chunk_id": start_chunk_id + i,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })
                        
                        # Keep only recent chunks, limit to last 100 to avoid file bloat
                        recent_chunks = recent_chunks[-100:]
                        
                        # Save updated recent chunks
                        with open(recent_chunks_file, 'w', encoding='utf-8') as f:
                            json.dump(recent_chunks, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    # Fail gracefully if we can't track recent chunks (non-critical)
                    debug_debug("Could not track recent chunks for validation (non-critical)")
                    debug_print_exception(e, context="Tracking recent chunks in RAG memory", include_traceback=True)
            
            # Save if path is configured
            if self.vectorstore_path:
                self._save_vectorstore()
    
    def retrieve(self, query: str, top_k: Optional[int] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None) -> List[str]:
        """
        Retrieve relevant memory chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (defaults to self.top_k)
            session_id: Optional session ID for provable_policy defense
            defense_type: Optional defense type to check if provable_policy is active
            
        Returns:
            List of retrieved document texts
        """
        with self._lock:
            if self.vectorstore is None:
                return []
            
            k = top_k or self.top_k
            
            try:
                # Perform similarity search
                results = self.vectorstore.similarity_search(query, k=k)
                
                # P1: Check if any retrieved memory has U label (provable_policy defense)
                if defense_type == "provable_policy" and session_id:
                    from agent.agent_core import ProvablePolicyManager
                    for doc in results:
                        label = doc.metadata.get("label", None)
                        if label == "U":
                            # Upgrade session to U if U-labeled memory is retrieved
                            ProvablePolicyManager.set_untrusted(session_id)
                            break
                        elif label is None:
                            # Error: memory should have a label
                            raise ValueError(
                                f"Memory chunk missing label in provable_policy defense. "
                                f"Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}, "
                                f"All memories must have 'label' metadata set to 'T' or 'U'."
                            )
                
                return [doc.page_content for doc in results]
            except Exception as e:
                debug_info(f"Error during retrieval: {e}")
                debug_print_exception(e, context="RAG memory retrieval", include_traceback=True)
                return []
    
    def get_context(self, query: str, top_k: Optional[int] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None) -> str:
        """
        Get formatted context string for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            session_id: Optional session ID for provable_policy defense
            defense_type: Optional defense type to check if provable_policy is active
            
        Returns:
            Formatted context string with retrieved memories
        """
        retrieved = self.retrieve(query, top_k, session_id=session_id, defense_type=defense_type)
        
        if not retrieved:
            return ""
        
        context_parts = []
        for i, memory in enumerate(retrieved, 1):
            context_parts.append(f"Memory {i}:\n{memory}")
        
        return "\n\n".join(context_parts)
    
    def clear_all_memory(self):
        """Clear all memory from the vector store."""
        with self._lock:
            self.vectorstore = None
            self.documents = []
            self._chunk_counter = 0
            
            # Delete persisted vector store if it exists
            if self.vectorstore_path and self.vectorstore_path.exists():
                import shutil
                try:
                    shutil.rmtree(self.vectorstore_path)
                except Exception as e:
                    debug_info(f"Could not delete vector store at {self.vectorstore_path}")
                    debug_print_exception(e, context=f"Deleting RAG vectorstore at {self.vectorstore_path}", include_traceback=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            return {
                "total_chunks": len(self.documents),
                "chunk_counter": self._chunk_counter,
                "vectorstore_initialized": self.vectorstore is not None,
                "embedding_model": self.embedding_model,
                "top_k": self.top_k,
                "chunk_size": self.chunk_size
            }


def get_rag_defense_manager(
    defense_type: str = "none",
    session_id: Optional[str] = None,
) -> RAGDefenseManager:
    """
    Create a new RAG defense manager instance.

    Args:
        defense_type: Type of defense to apply
        session_id: Optional session ID (kept for API compatibility, not used)

    Returns:
        A new RAGDefenseManager instance
    """
    return RAGDefenseManager(
        defense_type=defense_type
    )


def get_rag_memory_manager(
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 10,
    chunk_size: int = 512,
    vectorstore_path: Optional[str] = None,
) -> RAGMemoryManager:
    """
    Create a new RAG memory manager instance.
    
    Args:
        embedding_model: Name of the embedding model
        top_k: Number of top documents to retrieve
        chunk_size: Size of text chunks
        vectorstore_path: Optional path to persist vector store
    
    Returns:
        A new RAGMemoryManager instance
    """
    return RAGMemoryManager(
        embedding_model=embedding_model,
        top_k=top_k,
        chunk_size=chunk_size,
        vectorstore_path=vectorstore_path
    )


def get_rag_memory_context(text: str, session_id: str, memory_config: dict, memory_backend: str) -> str:
    """Retrieve RAG memory context if enabled."""
    rag_memory_config = memory_config.get("rag_memory", {})
    rag_memory_enabled = rag_memory_config.get("enabled", False) or (memory_backend == "rag")
    rag_defense_type = rag_memory_config.get("defense_type", "none")
    
    if not rag_memory_enabled or rag_defense_type == "disable_memory":
        return ""
    
    try:
        rag_memory_manager = get_rag_memory_manager(
            embedding_model=rag_memory_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_memory_config.get("top_k", 3),
            chunk_size=rag_memory_config.get("chunk_size", 512),
            vectorstore_path=rag_memory_config.get("vectorstore_path", "data/agent/rag_vectorstore"),
        )
        rag_context = rag_memory_manager.get_context(text, session_id=session_id, defense_type=rag_defense_type)
        return "\n\n# Relevant Memory Context\n" + rag_context + "\n" if rag_context else ""
    except (OSError, IOError, ValueError, RuntimeError) as e:
        debug_debug(f"Could not retrieve RAG memory context: {e}")
        return ""


def index_rag_memory(text: str, response_text: str, session_id: str, rag_memory_config: dict, rag_defense_type: str, config: dict) -> None:
    """Index conversation into RAG memory if enabled and defense allows."""
    try:
        defense_manager = get_rag_defense_manager(defense_type=rag_defense_type, session_id=session_id)
        
        if not defense_manager.should_index_memory(session_id, text, response_text):
            return
        
        default_chunk_size = rag_memory_config.get("chunk_size", 512)
        limit_memory_size = config.get("benchmark", {}).get("limit_memory_size_defense", 80)
        effective_chunk_size = defense_manager.get_chunk_size(default_chunk_size, limit_memory_size=limit_memory_size)
        conversation_turn = defense_manager.filter_conversation_turn(text, response_text)
        
        rag_memory_manager = get_rag_memory_manager(
            embedding_model=rag_memory_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_memory_config.get("top_k", 3),
            chunk_size=effective_chunk_size,
            vectorstore_path=rag_memory_config.get("vectorstore_path", "data/agent/rag_vectorstore"),
        )
        
        # Chunk conversation turn
        chunks = []
        chunk_size = effective_chunk_size
        if len(conversation_turn) > 100000:
            print(f"Chunking large conversation turn ({len(conversation_turn)} chars) into {chunk_size}-char chunks...", flush=True)
        
        for i in range(0, len(conversation_turn), chunk_size):
            chunk = conversation_turn[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        if not chunks:
            chunks = [""]
        
        if len(chunks) > 1000:
            print(f"WARNING: Creating {len(chunks)} chunks for RAG storage - this may take a while...", flush=True)
        
        if len(chunks) > 100:
            print(f"Storing {len(chunks)} chunks in RAG memory (batching to reduce API calls)...", flush=True)
        
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        
        if valid_chunks:
            max_batch_size = 1000
            batch_size = min(max_batch_size, len(valid_chunks))
            total_batches = (len(valid_chunks) + batch_size - 1) // batch_size
            
            if len(valid_chunks) > 100:
                print(f"   Using batch size: {batch_size} chunks per batch", flush=True)
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(valid_chunks))
                batch_chunks = valid_chunks[start_idx:end_idx]
                
                if len(valid_chunks) > 100:
                    print(f"   Progress: Batch {batch_idx + 1}/{total_batches} ({start_idx + 1}-{end_idx}/{len(valid_chunks)} chunks)...", flush=True)
                
                try:
                    rag_memory_manager.add_memories_batch(
                        batch_chunks,
                        metadata={"session_id": session_id, "type": "conversation", "defense_type": rag_defense_type},
                        session_id=session_id,
                        defense_type=rag_defense_type
                    )
                except (OSError, IOError, ValueError, RuntimeError) as e:
                    debug_debug(f"Failed to store RAG memory batch {batch_idx + 1}: {e}")
            
            if len(valid_chunks) > 100:
                print(f"Stored {len(valid_chunks)} chunks in RAG memory ({total_batches} batches)", flush=True)
    except (OSError, IOError, ValueError, RuntimeError) as e:
        debug_debug(f"Could not index RAG memory: {e}")
