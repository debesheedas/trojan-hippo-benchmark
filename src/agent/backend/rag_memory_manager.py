"""
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

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        from langchain_core.documents import Document
    except ImportError:
        # Fallback for older langchain versions
        from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: langchain packages not available. RAG memory will not work. Error: {e}")
    print("Install with: pip install langchain langchain-openai langchain-community langchain-core faiss-cpu")


class RAGMemoryManager:
    """
    Manages embedding-based RAG memory using vector stores for semantic retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 3,
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
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain packages required for RAG memory. "
                "Install with: pip install langchain langchain-openai langchain-community faiss-cpu"
            )
        
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
            api_key=api_key
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
                print(f"Loaded existing vector store from {self.vectorstore_path}")
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
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
                print(f"Warning: Could not save vector store: {e}")
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add text to the RAG memory system.
        
        Args:
            text: Text content to add to memory
            metadata: Optional metadata dictionary for the document
        """
        with self._lock:
            if not text or not text.strip():
                return
            
            # Create document with metadata
            doc_metadata = metadata or {}
            doc_metadata["chunk_id"] = self._chunk_counter
            doc_metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            doc = Document(page_content=text.strip(), metadata=doc_metadata)
            
            # Add to vector store
            if self.vectorstore is None:
                # Initialize vector store with first document
                self.vectorstore = FAISS.from_documents([doc], self.embeddings)
            else:
                # Add to existing vector store
                self.vectorstore.add_documents([doc])
            
            # Store document reference
            self.documents.append(text.strip())
            self._chunk_counter += 1
            
            # Save if path is configured
            if self.vectorstore_path:
                self._save_vectorstore()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant memory chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (defaults to self.top_k)
            
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
                return [doc.page_content for doc in results]
            except Exception as e:
                print(f"Error during retrieval: {e}")
                return []
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get formatted context string for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string with retrieved memories
        """
        retrieved = self.retrieve(query, top_k)
        
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
                    print(f"Warning: Could not delete vector store: {e}")
    
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


# Global RAG memory manager instance
_rag_memory_manager = None


def get_rag_memory_manager(
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 3,
    chunk_size: int = 512,
    vectorstore_path: Optional[str] = None,
    force_new: bool = False
) -> RAGMemoryManager:
    """
    Get or create the global RAG memory manager instance.
    
    Args:
        embedding_model: Name of the embedding model
        top_k: Number of top documents to retrieve
        chunk_size: Size of text chunks
        vectorstore_path: Optional path to persist vector store
        force_new: If True, create a new instance instead of reusing
    
    Returns:
        The RAGMemoryManager instance
    """
    global _rag_memory_manager
    if force_new or _rag_memory_manager is None:
        _rag_memory_manager = RAGMemoryManager(
            embedding_model=embedding_model,
            top_k=top_k,
            chunk_size=chunk_size,
            vectorstore_path=vectorstore_path
        )
    return _rag_memory_manager

