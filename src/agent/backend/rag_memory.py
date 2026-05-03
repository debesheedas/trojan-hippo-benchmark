"""
RAG Memory Module

Combines RAG Defense Manager and RAG Memory Manager functionality.

RAG Defense Manager:
Manages defense mechanisms for RAG memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. none: No defense (normal RAG execution - all messages indexed)
3. user_prompt_only: Only index user messages (filter out assistant messages)
4. limit_memory_length: Limit chunk sizes to 8 tokens each
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
from typing import List, Dict, Any, Optional
# Threading removed - each test runs in isolation, no locks needed
from datetime import datetime, timezone
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from agent.utils import debug_info, debug_debug, debug_print_exception

# Import in-memory storage (optional - for backward compatibility)
try:
    from benchmark.in_memory_storage import InMemoryVectorstore
except ImportError:
    InMemoryVectorstore = None  # Type: ignore


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
                - "limit_memory_length": Limit chunk sizes to configured limit
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
        # Defense: limit_memory_length - Limit chunk size to configured limit
        if self.defense_type == "limit_memory_length":
            return limit_memory_size

        # Other defenses use default chunk size
        return default_chunk_size


class RAGMemoryManager:
    """
    Manages embedding-based RAG memory using vector stores for semantic retrieval (in-memory only).
    """
    
    def __init__(
        self,
        vectorstore: 'InMemoryVectorstore',
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 8,
        chunk_size: int = 512,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the RAG memory manager (in-memory only).
        
        Args:
            vectorstore: In-memory vectorstore (required)
            embedding_model: Name of the embedding model to use
            top_k: Number of top documents to retrieve
            chunk_size: Size of text chunks in tokens
            api_key: Optional OpenAI API key (uses env var if not provided)
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.vectorstore_storage = vectorstore  # Renamed to avoid conflict with self.vectorstore
        
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
        
        # Load existing vector store from in-memory storage
        self._load_from_storage()

    def reload_from_storage(self) -> None:
        """Reload vectorstore, documents, and chunk_counter from storage. Use before merging snapshot so manager state matches current storage (e.g. after agent indexed step 2)."""
        self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Load existing vector store from in-memory storage into self."""
        vectorstore, documents, chunk_counter = self.vectorstore_storage.load()
        if vectorstore is not None:
            self.vectorstore = vectorstore
            self.documents = list(documents) if documents else []
            self._chunk_counter = chunk_counter if chunk_counter is not None else len(self.documents)
        else:
            self.vectorstore = None
            self.documents = []
            self._chunk_counter = 0

    def _save_vectorstore(self):
        """Save vector store to in-memory storage."""
        if self.vectorstore and self.vectorstore_storage:
            self.vectorstore_storage.save(self.vectorstore, self.documents, self._chunk_counter)
    
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
        # No lock needed - each test runs in isolation
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
        
        # Save to in-memory vectorstore
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
        # No lock needed - each test runs in isolation
        if self.vectorstore is None:
            return []
        
        k = top_k or self.top_k
        
        try:
            # Perform similarity search
            debug_debug(f"RAG Memory: Performing similarity search with k={k}, query='{query[:100]}...'", truncate=False, max_length=500)
            results = self.vectorstore.similarity_search(query, k=k)
            debug_debug(f"RAG Memory: Similarity search returned {len(results)} results (requested k={k})", truncate=False)
            
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
            
            retrieved_chunks = [doc.page_content for doc in results]
            # Log each retrieved chunk (without truncation for full visibility)
            debug_debug(f"RAG Memory: Retrieved {len(retrieved_chunks)} chunks", truncate=False)
            for i, chunk in enumerate(retrieved_chunks, 1):
                debug_debug(f"RAG Memory Chunk {i}/{len(retrieved_chunks)} ({len(chunk)} chars):\n{chunk}", truncate=False)
            
            return retrieved_chunks
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
        
        # Debug logging: Log retrieval details (without truncation to see full chunks)
        k = top_k or self.top_k
        debug_debug(f"RAG Memory get_context: Query='{query[:100]}...', top_k={k}, retrieved {len(retrieved)} chunks", truncate=False, max_length=500)
        
        if not retrieved:
            debug_debug("RAG Memory get_context: No chunks retrieved (vectorstore may be empty or query didn't match)", truncate=False)
            return ""
        
        # Log each retrieved chunk in full (without truncation)
        for i, memory in enumerate(retrieved, 1):
            debug_debug(f"RAG Memory Chunk {i}/{len(retrieved)} ({len(memory)} chars):\n{memory}", truncate=False)
        
        context_parts = []
        for i, memory in enumerate(retrieved, 1):
            context_parts.append(f"Memory {i}:\n{memory}")
        
        formatted_context = "\n\n".join(context_parts)
        debug_debug(f"RAG Memory get_context: Formatted context ({len(formatted_context)} total chars):\n{formatted_context}", truncate=False)
        
        return formatted_context
    
    def clear_all_memory(self):
        """Clear all memory from the vector store."""
        # No lock needed - each test runs in isolation
        self.vectorstore = None
        self.documents = []
        self._chunk_counter = 0
        
        # Clear in-memory vectorstore
        if self.vectorstore_storage:
            self.vectorstore_storage.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        # No lock needed - each test runs in isolation
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
    vectorstore: 'InMemoryVectorstore',
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 10,
    chunk_size: int = 512,
) -> RAGMemoryManager:
    """
    Create a new RAG memory manager instance (in-memory only).
    
    Args:
        vectorstore: In-memory vectorstore (required)
        embedding_model: Name of the embedding model
        top_k: Number of top documents to retrieve
        chunk_size: Size of text chunks
    
    Returns:
        A new RAGMemoryManager instance
    """
    return RAGMemoryManager(
        vectorstore=vectorstore,
        embedding_model=embedding_model,
        top_k=top_k,
        chunk_size=chunk_size,
    )


def get_rag_memory_context(text: str, session_id: str, memory_config: dict, memory_backend: str, in_memory_env = None) -> str:
    """Retrieve RAG memory context if RAG backend is active.
    
    When in_memory_env is provided, the exact list of retrieved chunk texts is stored in
    in_memory_env.last_rag_retrieved_chunks (in order) so the benchmark/scorer can use
    it for RAG chunk analysis feedback instead of simulating chunking.
    """
    # RAG is enabled when using the "rag" backend
    if memory_backend != "rag":
        return ""
    
    rag_memory_config = memory_config.get("rag_memory", {})
    rag_defense_type = rag_memory_config.get("defense_type", "none")
    
    if rag_defense_type == "disable_memory":
        return ""
    
    try:
        # Get vectorstore from in_memory_env (preferred) or from config (backward compatibility)
        vectorstore = None
        if in_memory_env:
            vectorstore = in_memory_env.rag_vectorstore
        else:
            vectorstore = rag_memory_config.get("vectorstore")
        
        if not vectorstore:
            debug_debug("No vectorstore available, cannot retrieve RAG memory")
            return ""
        
        debug_debug(f"get_rag_memory_context called: text='{text[:100]}...', session_id={session_id}, defense_type={rag_defense_type}", truncate=False, max_length=500)
        rag_memory_manager = get_rag_memory_manager(
            vectorstore=vectorstore,
            embedding_model=rag_memory_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_memory_config.get("top_k", 8),  # Default to 8 for RAG backend
            chunk_size=rag_memory_config.get("chunk_size", 512),
        )
        debug_debug(f"RAG memory manager created with top_k={rag_memory_manager.top_k}", truncate=False)
        # Retrieve raw chunks so we can store them for exact RAG analysis (and format context ourselves)
        retrieved_chunks = rag_memory_manager.retrieve(
            text, top_k=rag_memory_manager.top_k, session_id=session_id, defense_type=rag_defense_type
        )
        if in_memory_env is not None and hasattr(in_memory_env, "last_rag_retrieved_chunks"):
            in_memory_env.last_rag_retrieved_chunks = list(retrieved_chunks) if retrieved_chunks else []
        if not retrieved_chunks:
            debug_debug("No RAG context retrieved (empty result)", truncate=False)
            return ""
        context_parts = [f"Memory {i}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks, 1)]
        rag_context = "\n\n".join(context_parts)
        debug_debug(f"RAG context retrieved ({len(rag_context)} chars), adding to prompt", truncate=False)
        return "\n\n# Relevant Memory Context\n" + rag_context + "\n"
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
        
        vectorstore = rag_memory_config.get("vectorstore")
        if not vectorstore:
            debug_debug("No vectorstore in config, cannot index RAG memory")
            return
        
        rag_memory_manager = get_rag_memory_manager(
            vectorstore=vectorstore,
            embedding_model=rag_memory_config.get("embedding_model", "text-embedding-3-small"),
            top_k=rag_memory_config.get("top_k", 3),
            chunk_size=effective_chunk_size,
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


# ============================================================================
# Test Utilities
# ============================================================================

def get_memory_state_for_test(test_dir, config: Dict[str, Any]) -> List[str]:
    """
    Get RAG memory contents for test validation (from in-memory storage).
    
    Args:
        test_dir: Ignored - kept for API compatibility
        config: Configuration dictionary (should contain vectorstore)
    
    Returns:
        List of memory chunk strings
    """
    # Get vectorstore from config
    rag_memory_config = config.get("memory", {}).get("rag_memory", {})
    vectorstore = rag_memory_config.get("vectorstore")
    
    if vectorstore:
        # Get documents directly from in-memory vectorstore
        _, documents, _ = vectorstore.load()
        return [str(doc) for doc in documents] if documents else []
    
    return []
