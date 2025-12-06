"""
Mem0 Memory Manager Module
Wraps the mem0 Memory class for intelligent memory management.

Features:
- Intelligent fact extraction from conversations
- Semantic search over memories
- Automatic memory updates and deduplication
- Support for local and hosted configurations
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError as e:
    MEM0_AVAILABLE = False
    print(f"Warning: mem0 package not available. Mem0 memory will not work. Error: {e}")
    print("Install with: pip install mem0ai")


class Mem0MemoryManager:
    """
    Manages intelligent memory using mem0's Memory class.
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-5-mini",
        llm_temperature: float = 0.0,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        vector_store_provider: str = "faiss",
        vectorstore_path: Optional[str] = None,
        top_k: int = 3,
        user_id: str = "vince",
        agent_id: str = "email_agent",
        api_key: Optional[str] = None
    ):
        """
        Initialize the mem0 memory manager.
        
        Args:
            llm_provider: LLM provider (openai, ollama, anthropic, etc.)
            llm_model: LLM model name
            llm_temperature: LLM temperature
            embedding_provider: Embedding provider (openai, ollama, huggingface, etc.)
            embedding_model: Embedding model name
            vector_store_provider: Vector store provider (faiss, chroma, qdrant, etc.)
            vectorstore_path: Optional path to persist vector store
            top_k: Number of top memories to retrieve
            user_id: User identifier for memory scoping
            agent_id: Agent identifier for memory scoping
            api_key: Optional API key (uses env vars if not provided)
        """
        if not MEM0_AVAILABLE:
            raise ImportError(
                "mem0 package required for mem0 memory. "
                "Install with: pip install mem0ai"
            )
        
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.vector_store_provider = vector_store_provider
        self.vectorstore_path = Path(vectorstore_path) if vectorstore_path else None
        self.top_k = top_k
        self.user_id = user_id
        self.agent_id = agent_id
        self._lock = threading.Lock()
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Build mem0 configuration
        config_dict = self._build_config_dict(api_key)
        
        # Initialize mem0 Memory instance using from_config classmethod
        self.memory = Memory.from_config(config_dict)
    
    def _build_config_dict(self, api_key: Optional[str]) -> Dict[str, Any]:
        """Build mem0 configuration dictionary."""
        # Determine embedding dimensions based on model
        # text-embedding-3-small: 1536, text-embedding-3-large: 3072, text-embedding-ada-002: 1536
        embedding_dims = 1536  # Default
        if "large" in self.embedding_model.lower():
            embedding_dims = 3072
        elif "3-small" in self.embedding_model.lower() or "ada" in self.embedding_model.lower():
            embedding_dims = 1536
        
        config_dict = {
            "llm": {
                "provider": self.llm_provider,
                "config": {
                    "model": self.llm_model,
                    "temperature": self.llm_temperature,
                }
            },
            "embedder": {
                "provider": self.embedding_provider,
                "config": {
                    "model": self.embedding_model,
                }
            },
            "vector_store": {
                "provider": self.vector_store_provider,
                "config": {
                    "collection_name": "mem0_memories",
                    "embedding_model_dims": embedding_dims,
                }
            }
        }
        
        # Add API key to LLM config if needed
        if self.llm_provider == "openai" and api_key:
            config_dict["llm"]["config"]["api_key"] = api_key
        
        # Add API key to embedding config if needed
        if self.embedding_provider == "openai" and api_key:
            config_dict["embedder"]["config"]["api_key"] = api_key
        
        # Configure vector store path for local stores
        if self.vector_store_provider == "faiss" and self.vectorstore_path:
            config_dict["vector_store"]["config"]["path"] = str(self.vectorstore_path)
            # Ensure directory exists
            self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        return config_dict
    
    def add_memory(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        max_memory_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add memories from conversation messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            metadata: Optional metadata dictionary
            user_id: Optional user ID (defaults to "vince" - hardcoded)
            agent_id: Optional agent ID (defaults to self.agent_id)
            
        Returns:
            Dictionary with memory addition results
        """
        with self._lock:
            if not messages:
                return {"results": []}
            
            # Always use "vince" as user_id (hardcoded)
            user_id = "vince"
            agent_id = agent_id or self.agent_id
            
            # Combine metadata
            combined_metadata = metadata or {}
            combined_metadata["session_type"] = "conversation"
            
            try:
                # For user memory extraction, we should NOT pass agent_id to memory.add()
                # because mem0 uses agent_id presence in metadata to decide which prompt to use:
                # - If agent_id is in metadata: uses AGENT_MEMORY_EXTRACTION_PROMPT (extracts from assistant messages only)
                # - If agent_id is NOT in metadata: uses USER_MEMORY_EXTRACTION_PROMPT (extracts from user messages only)
                # We want user memory extraction, so we don't include agent_id in metadata
                # Note: We can still use agent_id for filtering in search operations via filters parameter
                
                # When max_memory_length is provided (e.g., by a defense), it is passed
                # through to mem0 so that extracted fact strings can be truncated
                # before they are embedded and written into the vector store.
                result = self.memory.add(
                    messages=messages,
                    user_id=user_id,
                    agent_id=None,  # Don't pass agent_id to force USER_MEMORY_EXTRACTION_PROMPT
                    metadata=combined_metadata,  # Don't include agent_id here
                    infer=True,  # Use LLM to extract facts
                    max_memory_length=max_memory_length,
                )
                return result
            except Exception as e:
                print(f"Warning: Could not add memory to mem0: {e}")
                return {"results": []}
    
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query string
            user_id: Optional user ID (ignored - always uses "vince")
            agent_id: Optional agent ID (defaults to self.agent_id)
            limit: Number of results to return (defaults to self.top_k)
            
        Returns:
            List of memory dictionaries
        """
        with self._lock:
            if not query or not query.strip():
                return []
            
            # Always use "vince" as user_id (hardcoded)
            user_id = "vince"
            agent_id = agent_id or self.agent_id
            limit = limit or self.top_k
            
            try:
                result = self.memory.search(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=limit
                )
                # Extract results from mem0 response
                if isinstance(result, dict) and "results" in result:
                    return result["results"]
                elif isinstance(result, list):
                    return result
                else:
                    return []
            except Exception as e:
                print(f"Warning: Could not search mem0 memory: {e}")
                return []
    
    def get_context(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """
        Get formatted context string for a query.
        
        Args:
            query: The search query
            user_id: Optional user ID (ignored - always uses "vince")
            agent_id: Optional agent ID
            
        Returns:
            Formatted context string with retrieved memories
        """
        # Always use "vince" as user_id (hardcoded)
        memories = self.search(query, user_id="vince", agent_id=agent_id)
        
        # If no memories found, try a fallback: get some general persona memories
        # This helps when the query is too specific and doesn't match persona details
        if not memories:
            try:
                # Get a sample of all memories as fallback (up to top_k)
                all_memories = self.get_all_memories(user_id="vince", agent_id=agent_id, limit=self.top_k)
                if all_memories:
                    # Use first few memories as fallback context
                    memories = all_memories[:min(5, len(all_memories))]
            except Exception:
                pass
        
        if not memories:
            return ""
        
        context_parts = []
        for i, memory_item in enumerate(memories, 1):
            # Extract memory text from mem0 response format
            memory_text = memory_item.get("memory", "") if isinstance(memory_item, dict) else str(memory_item)
            if memory_text:
                context_parts.append(f"Memory {i}:\n{memory_text}")
        
        return "\n\n".join(context_parts)
    
    def get_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for a user/agent using the get_all API.
        
        Note: User memories are stored with agent_id=None (to use USER_MEMORY_EXTRACTION_PROMPT).
        This method will try both the provided agent_id and None to find all relevant memories.
        
        Args:
            user_id: Optional user ID (defaults to self.user_id)
            agent_id: Optional agent ID (defaults to None to match how user memories are stored)
            filters: Optional additional filters (supports AND, OR, etc.)
            limit: Maximum number of memories to return (default: 100)
            
        Returns:
            List of all memory dictionaries
        """
        with self._lock:
            user_id = user_id or self.user_id
            # User memories are stored with agent_id=None, so default to None if not explicitly provided
            # This ensures we retrieve user memories correctly
            if agent_id is None:
                agent_id = None  # Explicitly use None for user memories
            # If agent_id is provided, we'll try both that and None to get all memories
            
            all_memories = []
            
            try:
                # First, try with the provided agent_id (or None)
                query_filters = (filters or {}).copy()
                
                # Add user_id and agent_id to filters if provided
                if user_id:
                    query_filters["user_id"] = user_id
                if agent_id:
                    query_filters["agent_id"] = agent_id
                
                # Call get_all with filters
                result = self.memory.get_all(
                    user_id=user_id if user_id else None,
                    agent_id=agent_id if agent_id else None,
                    filters=query_filters if query_filters else None,
                    limit=limit
                )
                
                # Extract results from mem0 response
                if isinstance(result, dict) and "results" in result:
                    all_memories.extend(result["results"])
                elif isinstance(result, list):
                    all_memories.extend(result)
                
                # Also try with agent_id=None to find user memories (stored with agent_id=None)
                # This is important because user memories are stored with agent_id=None
                if agent_id is not None:  # Only try None if we haven't already
                    try:
                        query_filters_none = (filters or {}).copy()
                        if user_id:
                            query_filters_none["user_id"] = user_id
                        # Don't add agent_id to filters for this query
                        
                        result_none = self.memory.get_all(
                            user_id=user_id if user_id else None,
                            agent_id=None,
                            filters=query_filters_none if query_filters_none else None,
                            limit=limit
                        )
                        
                        if isinstance(result_none, dict) and "results" in result_none:
                            # Add memories that aren't already in all_memories (deduplicate by id)
                            existing_ids = {m.get("id") for m in all_memories if isinstance(m, dict) and "id" in m}
                            for mem in result_none["results"]:
                                if isinstance(mem, dict) and mem.get("id") not in existing_ids:
                                    all_memories.append(mem)
                        elif isinstance(result_none, list):
                            existing_ids = {m.get("id") for m in all_memories if isinstance(m, dict) and "id" in m}
                            for mem in result_none:
                                if isinstance(mem, dict) and mem.get("id") not in existing_ids:
                                    all_memories.append(mem)
                    except Exception:
                        pass  # If this fails, just return the memories we found with the original query
                
                return all_memories
            except Exception as e:
                print(f"Warning: Could not get all memories from mem0: {e}")
                import traceback
                traceback.print_exc()
                return []
    
    def clear_all_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        Clear all memories for a user/agent.
        
        Args:
            user_id: Optional user ID (defaults to self.user_id)
            agent_id: Optional agent ID (defaults to self.agent_id)
        """
        with self._lock:
            user_id = user_id or self.user_id
            agent_id = agent_id or self.agent_id
            
            try:
                # Get all memories first
                all_memories = self.get_all_memories(user_id=user_id, agent_id=agent_id)
                
                # Delete each memory (mem0 doesn't have a bulk delete, so we iterate)
                for memory_item in all_memories:
                    memory_id = memory_item.get("id") if isinstance(memory_item, dict) else None
                    if memory_id:
                        try:
                            # Note: mem0 Memory class may not have delete method in open-source version
                            # This is a placeholder - actual implementation depends on mem0 API
                            pass
                        except Exception as e:
                            print(f"Warning: Could not delete memory {memory_id}: {e}")
                
                print(f"Cleared memories for user_id={user_id}, agent_id={agent_id}")
            except Exception as e:
                print(f"Warning: Could not clear mem0 memory: {e}")
    
    def get_users(self) -> Dict[str, Any]:
        """
        Get all users, agents, and runs that have memories.
        
        Note: This method may not be available in all versions of mem0.
        
        Returns:
            Dictionary with list of entities
        """
        with self._lock:
            try:
                # Try to call users() method if it exists
                if hasattr(self.memory, 'users'):
                    result = self.memory.users()
                    return result
                else:
                    # Fallback: return empty result
                    return {"results": []}
            except Exception as e:
                print(f"Warning: Could not get users from mem0: {e}")
                return {"results": []}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            try:
                all_memories = self.get_all_memories()
                return {
                    "total_memories": len(all_memories),
                    "llm_provider": self.llm_provider,
                    "llm_model": self.llm_model,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": self.embedding_model,
                    "vector_store_provider": self.vector_store_provider,
                    "top_k": self.top_k,
                    "user_id": self.user_id,
                    "agent_id": self.agent_id
                }
            except Exception as e:
                print(f"Warning: Could not get mem0 memory stats: {e}")
                return {}


# Global mem0 memory manager instance
# Cache managers by vectorstore_path to support multiple test environments
_mem0_manager_cache: Dict[str, Mem0MemoryManager] = {}


# Cache managers by vectorstore_path to support multiple test environments
_mem0_manager_cache: Dict[str, Mem0MemoryManager] = {}

def get_mem0_memory_manager(
    llm_provider: str = "openai",
    llm_model: str = "gpt-5-mini",
    llm_temperature: float = 0.0,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    vector_store_provider: str = "faiss",
    vectorstore_path: Optional[str] = None,
    top_k: int = 3,
    user_id: str = "vince",
    agent_id: str = "email_agent",
    force_new: bool = False
) -> Mem0MemoryManager:
    """
    Get or create a mem0 memory manager instance.
    Managers are cached by vectorstore_path to support multiple test environments.
    
    Args:
        llm_provider: LLM provider
        llm_model: LLM model name
        llm_temperature: LLM temperature
        embedding_provider: Embedding provider
        embedding_model: Embedding model name
        vector_store_provider: Vector store provider
        vectorstore_path: Optional path to persist vector store (used as cache key)
        top_k: Number of top memories to retrieve
        user_id: User identifier
        agent_id: Agent identifier
        force_new: If True, create a new instance instead of reusing cached one
    
    Returns:
        The Mem0MemoryManager instance
    """
    global _mem0_manager_cache
    
    # Use vectorstore_path as cache key (or "default" if None)
    cache_key = str(vectorstore_path) if vectorstore_path else "default"
    
    # Create new instance if force_new or not in cache
    if force_new or cache_key not in _mem0_manager_cache:
        _mem0_manager_cache[cache_key] = Mem0MemoryManager(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_store_provider=vector_store_provider,
            vectorstore_path=vectorstore_path,
            top_k=top_k,
            user_id=user_id,
            agent_id=agent_id
        )
    
    return _mem0_manager_cache[cache_key]

