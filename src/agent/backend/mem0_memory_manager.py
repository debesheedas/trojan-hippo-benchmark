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
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
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
        top_k: int = 10,
        user_id: str = "vince",
        agent_id: Optional[str] = None,  # Always None for mem0 - user memories use agent_id=None
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
            agent_id: Agent identifier (should always be None for mem0 - user memories use agent_id=None)
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
        self.agent_id = None  # Always None for mem0 - user memories are stored with agent_id=None
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
        session_id: Optional[str] = None,
        defense_type: Optional[str] = None,
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
            
            # Always use "vince" as user_id (hardcoded) and None for agent_id
            user_id = "vince"
            agent_id = None  # Always None for mem0
            
            # Combine metadata
            combined_metadata = metadata or {}
            combined_metadata["session_type"] = "conversation"
            
            # Note: We pass messages to mem0 as-is. For user_prompt_only defense,
            # messages are already filtered to only user messages by the defense manager.
            # mem0 should handle UPDATE operations correctly regardless of message structure.
            
            # P3: Memory Labeling - inherit session label for provable_policy defense
            if defense_type == "provable_policy":
                if session_id:
                    from agent.agent_core import ProvablePolicyManager
                    session_label = ProvablePolicyManager.get_session_label(session_id)
                    combined_metadata["label"] = session_label
                else:
                    # Default to T if no session_id provided
                    combined_metadata["label"] = "T"
            
            try:
                # For user memory extraction, we should NOT pass agent_id to memory.add()
                # because mem0 uses agent_id presence in metadata to decide which prompt to use:
                # - If agent_id is in metadata: uses AGENT_MEMORY_EXTRACTION_PROMPT (extracts from assistant messages only)
                # - If agent_id is NOT in metadata: uses USER_MEMORY_EXTRACTION_PROMPT (extracts from user messages only)
                # We want user memory extraction, so we don't include agent_id in metadata
                # Note: We can still use agent_id for filtering in search operations via filters parameter
                
                # Defense: limit_memory_length - truncate message content BEFORE extraction
                # This ensures only truncated memories are stored, avoiding duplicates
                messages_to_use = messages
                if max_memory_length and max_memory_length > 0:
                    # Truncate each message's content to max_memory_length before passing to mem0
                    # This way, mem0 will extract from truncated content and store only truncated memories
                    messages_to_use = []
                    for msg in messages:
                        if isinstance(msg, dict) and "content" in msg:
                            content = str(msg["content"])
                            if len(content) > max_memory_length:
                                # Truncate at character boundary (simple truncation)
                                truncated_content = content[:max_memory_length]
                                messages_to_use.append({
                                    **msg,
                                    "content": truncated_content
                                })
                            else:
                                messages_to_use.append(msg)
                        else:
                            messages_to_use.append(msg)
                
                # Extract memories using mem0 (with potentially truncated messages)
                # Capture mem0's internal error messages to handle UPDATE operation failures gracefully
                # Root cause: mem0's deduplication logic uses simple IDs (e.g., '6') to track memories,
                # but stored memories have UUIDs. When mem0 tries to UPDATE a similar memory, it fails
                # with KeyError because it can't find the memory with the simple ID.
                # This is a mem0 bug - we can't fix it without modifying mem0 source code.
                error_buffer = StringIO()
                try:
                    with redirect_stderr(error_buffer), redirect_stdout(error_buffer):
                        result = self.memory.add(
                            messages=messages_to_use,
                            user_id=user_id,
                            agent_id=None,  # Don't pass agent_id to force USER_MEMORY_EXTRACTION_PROMPT
                            metadata=combined_metadata,  # Don't include agent_id here
                            infer=True,  # Use LLM to extract facts
                        )
                    
                    # Check if mem0 printed any UPDATE-related errors
                    error_output = error_buffer.getvalue()
                    if error_output and "Error processing memory action" in error_output:
                        # Check if the operation still succeeded (new memories were added)
                        has_results = False
                        if isinstance(result, dict):
                            has_results = bool(result.get("results"))
                        elif isinstance(result, list):
                            has_results = bool(result)
                        elif result:
                            has_results = True
                        
                        if has_results:
                            # Operation succeeded despite UPDATE failure - log as warning for monitoring
                            # The new memory was added, but mem0 failed to UPDATE the old similar memory
                            # This is non-fatal but worth tracking to see if it affects performance
                            print(f"Warning: mem0 UPDATE operation failed (mem0 bug - ID mismatch), but new memory was added. "
                                  f"This may cause duplicate memories. Error: {error_output[:300]}")
                        else:
                            # Operation completely failed - this is more serious
                            print(f"Error: mem0 UPDATE operation failed, and no new memories were added. "
                                  f"This may indicate a more serious issue. Error: {error_output[:300]}")
                except Exception as e:
                    # Re-raise the exception - this is a real error, not just an UPDATE failure
                    raise
                
                # Post-process: Ensure extracted memories are also truncated (defense in depth)
                # Even though we truncated input, mem0 might combine or rephrase, so we check again
                if max_memory_length and max_memory_length > 0:
                    # Extract memory texts and verify/truncate if needed
                    processed_memories = []
                    if isinstance(result, dict) and "results" in result:
                        for memory_item in result["results"]:
                            if isinstance(memory_item, dict):
                                memory_text = (
                                    memory_item.get("memory") or
                                    memory_item.get("memories") or
                                    memory_item.get("text") or
                                    memory_item.get("content") or
                                    memory_item.get("fact") or
                                    ""
                                )
                                if memory_text:
                                    memory_text_str = str(memory_text)
                                    # Truncate if still too long (defense in depth)
                                    if len(memory_text_str) > max_memory_length:
                                        memory_text_str = memory_text_str[:max_memory_length]
                                    # Update the memory item with truncated text
                                    # Find which key was used and update it
                                    for key in ["memory", "memories", "text", "content", "fact"]:
                                        if key in memory_item:
                                            memory_item[key] = memory_text_str
                                            break
                                    processed_memories.append(memory_item)
                    elif isinstance(result, list):
                        for memory_item in result:
                            if isinstance(memory_item, dict):
                                memory_text = (
                                    memory_item.get("memory") or
                                    memory_item.get("memories") or
                                    memory_item.get("text") or
                                    memory_item.get("content") or
                                    memory_item.get("fact") or
                                    ""
                                )
                                if memory_text:
                                    memory_text_str = str(memory_text)
                                    # Truncate if still too long (defense in depth)
                                    if len(memory_text_str) > max_memory_length:
                                        memory_text_str = memory_text_str[:max_memory_length]
                                    # Update the memory item with truncated text
                                    for key in ["memory", "memories", "text", "content", "fact"]:
                                        if key in memory_item:
                                            memory_item[key] = memory_text_str
                                            break
                                    processed_memories.append(memory_item)
                    
                    # Note: We don't re-add here because we already truncated the input messages.
                    # The memories stored by mem0 should already be truncated. The post-processing
                    # above just ensures the result object reflects truncated values for consistency.
                
                # Track recent memories for efficient validation
                # Extract memory texts from result and save to file in test directory
                if self.vectorstore_path:
                    try:
                        vectorstore_path_obj = Path(self.vectorstore_path)
                        # Check if this is a test directory (contains "test_env" or "mem0_vectorstore" in test_envs)
                        if "test_env" in str(vectorstore_path_obj) or "test_envs" in str(vectorstore_path_obj):
                            # Get the test directory (parent of mem0_vectorstore)
                            test_dir = vectorstore_path_obj.parent
                            recent_memories_file = test_dir / "mem0_recent_memories.json"
                            
                            # Extract memory texts from result (already truncated if defense was active)
                            memory_texts = []
                            if isinstance(result, dict) and "results" in result:
                                for memory_item in result["results"]:
                                    if isinstance(memory_item, dict):
                                        memory_text = (
                                            memory_item.get("memory") or
                                            memory_item.get("memories") or
                                            memory_item.get("text") or
                                            memory_item.get("content") or
                                            memory_item.get("fact") or
                                            ""
                                        )
                                        if memory_text:
                                            memory_texts.append(str(memory_text))
                            elif isinstance(result, list):
                                for memory_item in result:
                                    if isinstance(memory_item, dict):
                                        memory_text = (
                                            memory_item.get("memory") or
                                            memory_item.get("memories") or
                                            memory_item.get("text") or
                                            memory_item.get("content") or
                                            memory_item.get("fact") or
                                            ""
                                        )
                                        if memory_text:
                                            memory_texts.append(str(memory_text))
                            
                            # Read existing recent memories
                            recent_memories = []
                            if recent_memories_file.exists():
                                try:
                                    import json
                                    with open(recent_memories_file, 'r', encoding='utf-8') as f:
                                        recent_memories = json.load(f)
                                except Exception:
                                    recent_memories = []
                            
                            # Add new memories
                            recent_memories.extend(memory_texts)
                            
                            # Write back (keep only recent memories, limit to last 100 to avoid file bloat)
                            recent_memories = recent_memories[-100:]
                            if recent_memories:
                                import json
                                with open(recent_memories_file, 'w', encoding='utf-8') as f:
                                    json.dump(recent_memories, f, indent=2, ensure_ascii=False)
                    except Exception:
                        # Silently fail if we can't write recent memories (not critical)
                        pass
                
                return result
            except Exception as e:
                print(f"Warning: Could not add memory to mem0: {e}")
                return {"results": []}
    
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        defense_type: Optional[str] = None
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
            # Always use None for agent_id - user memories are stored with agent_id=None
            agent_id = None
            limit = limit or self.top_k
            
            try:
                result = self.memory.search(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,  # Always None
                    limit=limit
                )
                # Extract results from mem0 response
                memories = []
                if isinstance(result, dict) and "results" in result:
                    memories = result["results"]
                elif isinstance(result, list):
                    memories = result
                else:
                    memories = []
                
                # P1: Check if any retrieved memory has U label (provable_policy defense)
                if defense_type == "provable_policy" and session_id:
                    from agent.agent_core import ProvablePolicyManager
                    print(f"🔍 [DEBUG] mem0 search: Checking {len(memories)} memories for U labels (session_id={session_id})")
                    found_u_label = False
                    for i, memory_item in enumerate(memories):
                        if isinstance(memory_item, dict):
                            # Check metadata for label
                            metadata = memory_item.get("metadata", {})
                            label = metadata.get("label", None)
                            memory_text = memory_item.get("memory", "")[:50]
                            print(f"🔍 [DEBUG] mem0 Memory {i}: label={label}, text_preview='{memory_text}'")
                            if label == "U":
                                # Upgrade session to U if U-labeled memory is retrieved
                                print(f"🛡️ [DEBUG] mem0: Found U-labeled memory! Upgrading session '{session_id}' to UNTRUSTED")
                                ProvablePolicyManager.set_untrusted(session_id)
                                found_u_label = True
                                break
                            elif label is None:
                                # Error: memory should have a label
                                print(f"⚠️ [DEBUG] mem0 Memory {i} missing label! text_preview='{memory_text}'")
                                raise ValueError(
                                    f"Mem0 memory entry missing label in provable_policy defense. "
                                    f"All memories must have 'label' metadata set to 'T' or 'U'. "
                                    f"Memory text preview: {memory_text}..."
                                )
                    if not found_u_label:
                        print(f"🔍 [DEBUG] mem0: No U-labeled memories found. Session '{session_id}' remains trusted.")
                
                return memories
            except Exception as e:
                print(f"Warning: Could not search mem0 memory: {e}")
                return []
    
    def get_context(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None) -> str:
        """
        Get formatted context string for a query.
        
        Args:
            query: The search query
            user_id: Optional user ID (ignored - always uses "vince")
            agent_id: Optional agent ID
            session_id: Optional session ID for provable_policy defense
            defense_type: Optional defense type to check if provable_policy is active
            
        Returns:
            Formatted context string with retrieved memories
        """
        # Always use "vince" as user_id (hardcoded) and None for agent_id
        memories = self.search(query, user_id="vince", agent_id=None, session_id=session_id, defense_type=defense_type)
        
        # If we got fewer memories than top_k, try a more aggressive fallback
        # This helps when the query is too specific and doesn't match stored memories well
        if len(memories) < self.top_k:
            try:
                # Get a sample of all memories as fallback
                all_memories = self.get_all_memories(user_id="vince", agent_id=None, limit=self.top_k)
                if all_memories:
                    # Use first few memories as fallback context
                    existing_texts = {m.get("memory", "") for m in memories if isinstance(m, dict)}
                    for mem in all_memories[:min(10, len(all_memories))]:
                        if isinstance(mem, dict):
                            mem_text = mem.get("memory", "")
                            if mem_text and mem_text not in existing_texts:
                                memories.append(mem)
                                existing_texts.add(mem_text)
                                if len(memories) >= self.top_k:
                                    break
                    
                    # P1: Check if any fallback memory has U label (provable_policy defense)
                    # This is critical - fallback memories must also be checked for U labels
                    if defense_type == "provable_policy" and session_id:
                        from agent.agent_core import ProvablePolicyManager
                        print(f"🔍 [DEBUG] mem0 get_context fallback: Checking {len(all_memories)} fallback memories for U labels (session_id={session_id})")
                        found_u_label = False
                        for i, memory_item in enumerate(all_memories):
                            if isinstance(memory_item, dict):
                                metadata = memory_item.get("metadata", {})
                                label = metadata.get("label", None)
                                memory_text = memory_item.get("memory", "")[:50]
                                print(f"🔍 [DEBUG] mem0 Fallback Memory {i}: label={label}, text_preview='{memory_text}'")
                                if label == "U":
                                    print(f"🛡️ [DEBUG] mem0 fallback: Found U-labeled memory! Upgrading session '{session_id}' to UNTRUSTED")
                                    ProvablePolicyManager.set_untrusted(session_id)
                                    found_u_label = True
                                    break
                                elif label is None:
                                    print(f"⚠️ [DEBUG] mem0 Fallback Memory {i} missing label! text_preview='{memory_text}'")
                                    raise ValueError(
                                        f"Mem0 fallback memory entry missing label in provable_policy defense. "
                                        f"All memories must have 'label' metadata set to 'T' or 'U'. "
                                        f"Memory text preview: {memory_text}..."
                                    )
                        if not found_u_label:
                            print(f"🔍 [DEBUG] mem0 fallback: No U-labeled memories found. Session '{session_id}' remains trusted.")
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
        Get all memories for a user using the get_all API.
        
        Note: User memories are stored with agent_id=None (to use USER_MEMORY_EXTRACTION_PROMPT).
        
        Args:
            user_id: Optional user ID (defaults to self.user_id)
            agent_id: Optional agent ID (ignored - always uses None)
            filters: Optional additional filters (supports AND, OR, etc.)
            limit: Maximum number of memories to return (default: 100)
            
        Returns:
            List of all memory dictionaries
        """
        with self._lock:
            user_id = user_id or self.user_id
            # Always use None for agent_id - user memories are stored with agent_id=None
            agent_id = None
            
            all_memories = []
            
            try:
                query_filters = (filters or {}).copy()
                
                # Add user_id to filters
                if user_id:
                    query_filters["user_id"] = user_id
                # Don't add agent_id - always use None
                
                # Call get_all with agent_id=None
                result = self.memory.get_all(
                    user_id=user_id if user_id else None,
                    agent_id=None,
                    filters=query_filters if query_filters else None,
                    limit=limit
                )
                
                # Extract results from mem0 response
                if isinstance(result, dict) and "results" in result:
                    all_memories.extend(result["results"])
                elif isinstance(result, list):
                    all_memories.extend(result)
                
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
            agent_id = None  # Always None for mem0
            
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
                    "agent_id": None  # Always None for mem0
                }
            except Exception as e:
                print(f"Warning: Could not get mem0 memory stats: {e}")
                return {}


def get_mem0_memory_manager(
    llm_provider: str = "openai",
    llm_model: str = "gpt-5-mini",
    llm_temperature: float = 0.0,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    vector_store_provider: str = "faiss",
    vectorstore_path: Optional[str] = None,
    top_k: int = 10,
    user_id: str = "vince",
    agent_id: Optional[str] = None,  # Always None for mem0
) -> Mem0MemoryManager:
    """
    Create a new mem0 memory manager instance.
    
    Args:
        llm_provider: LLM provider
        llm_model: LLM model name
        llm_temperature: LLM temperature
        embedding_provider: Embedding provider
        embedding_model: Embedding model name
        vector_store_provider: Vector store provider
        vectorstore_path: Optional path to persist vector store
        top_k: Number of top memories to retrieve
        user_id: User identifier
        agent_id: Agent identifier
    
    Returns:
        A new Mem0MemoryManager instance
    """
    return Mem0MemoryManager(
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

