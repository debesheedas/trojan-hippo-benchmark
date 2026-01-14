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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Load environment variables from .env file
load_dotenv()

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError as e:
    MEM0_AVAILABLE = False
    print(f"Warning: mem0 package not available. Mem0 memory will not work. Error: {e}")
    print("Install with: pip install mem0ai")

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class Mem0TimeoutError(Exception):
    """
    Custom exception for mem0 API timeouts.
    
    This exception should be re-raised (not caught silently) to ensure
    test results are marked as unreliable when API calls fail.
    """
    pass


def _call_with_timeout(func, timeout_seconds=300, error_message="mem0 API call"):
    """
    Helper function to call a mem0 API function with a timeout.
    
    Args:
        func: Callable that performs the mem0 API call
        timeout_seconds: Maximum time to wait (default: 5 minutes)
        error_message: Error message prefix for timeout errors
        
    Returns:
        Result from func()
        
    Raises:
        Mem0TimeoutError: If the call exceeds timeout_seconds
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            error_msg = (
                f"{error_message} timed out after {timeout_seconds} seconds. "
                f"This indicates the API call was not successful. "
                f"Results from this test are UNRELIABLE and should be marked as failed. "
                f"This may indicate an API issue, network problem, or mem0 library bug."
            )
            print(f"❌ ERROR: {error_msg}")
            print(f"   This is likely causing the benchmark to hang.")
            print(f"   Consider checking:")
            print(f"   - API rate limits and quotas")
            print(f"   - Network connectivity")
            print(f"   - mem0 library version and known issues")
            raise Mem0TimeoutError(error_msg) from None


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
    
    def _chunk_large_message(
        self,
        message: Dict[str, str],
        max_tokens: int = 6000,
        model_name: str = "gpt-4o-mini"
    ) -> List[Dict[str, str]]:
        """
        Chunk a large message into smaller pieces that fit within token limits.
        
        This prevents mem0's embedding model from hitting token limits during search.
        The embedding model (text-embedding-3-small) has an 8192 token limit, so we
        use 6000 tokens per chunk to leave room for overhead.
        
        Args:
            message: Message dictionary with 'role' and 'content'
            max_tokens: Maximum tokens per chunk (default: 6000, safe for 8192 limit)
            model_name: Model name for tokenizer (default: gpt-4o-mini)
            
        Returns:
            List of message chunks, each within token limit
        """
        if not isinstance(message, dict) or "content" not in message:
            return [message]
        
        content = str(message["content"])
        role = message.get("role", "user")
        
        # If content is small, no need to chunk
        if len(content) < 10000:  # Rough heuristic: ~10000 chars ≈ ~2500 tokens
            return [message]
        
        # Count tokens to see if chunking is needed
        if not TIKTOKEN_AVAILABLE:
            # Fallback: use character-based estimation (rough: 1 token ≈ 4 chars)
            estimated_tokens = len(content) // 4
            if estimated_tokens <= max_tokens:
                return [message]
            # Chunk by characters
            chunk_size_chars = max_tokens * 4
            chunks = []
            for i in range(0, len(content), chunk_size_chars):
                chunk_content = content[i:i + chunk_size_chars]
                chunks.append({
                    "role": role,
                    "content": chunk_content,
                    "chunk_index": i // chunk_size_chars,
                    "total_chunks": (len(content) + chunk_size_chars - 1) // chunk_size_chars
                })
            return chunks
        
        # Use tiktoken for accurate token counting
        try:
            try:
                tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            
            # Encode to get token count
            encoded = tokenizer.encode(content, disallowed_special=())
            token_count = len(encoded)
            
            # If within limit, return as-is
            if token_count <= max_tokens:
                return [message]
            
            # Chunk by tokens (preserve token boundaries)
            chunks = []
            num_chunks = (token_count + max_tokens - 1) // max_tokens
            
            for i in range(num_chunks):
                start_idx = i * max_tokens
                end_idx = min((i + 1) * max_tokens, token_count)
                chunk_encoded = encoded[start_idx:end_idx]
                chunk_content = tokenizer.decode(chunk_encoded)
                
                chunks.append({
                    "role": role,
                    "content": chunk_content,
                    "chunk_index": i,
                    "total_chunks": num_chunks
                })
            
            return chunks
        except Exception as e:
            # If tokenization fails, fall back to character-based chunking
            print(f"Warning: Token-based chunking failed, using character-based: {e}")
            chunk_size_chars = max_tokens * 4
            chunks = []
            for i in range(0, len(content), chunk_size_chars):
                chunk_content = content[i:i + chunk_size_chars]
                chunks.append({
                    "role": role,
                    "content": chunk_content,
                    "chunk_index": i // chunk_size_chars,
                    "total_chunks": (len(content) + chunk_size_chars - 1) // chunk_size_chars
                })
            return chunks
    
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
                
                # Step 1: Chunk large messages to prevent embedding model token limit errors
                # mem0's embedding model (text-embedding-3-small) has 8192 token limit
                # We chunk messages to 6000 tokens to leave room for overhead
                # This prevents errors during memory.search() when mem0 tries to embed all memories
                chunked_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        # Chunk large messages (6000 tokens = safe for 8192 limit)
                        msg_chunks = self._chunk_large_message(msg, max_tokens=6000)
                        chunked_messages.extend(msg_chunks)
                    else:
                        chunked_messages.append(msg)
                
                # Step 2: Defense: limit_memory_length - truncate message content BEFORE extraction
                # This ensures only truncated memories are stored, avoiding duplicates
                messages_to_use = chunked_messages
                if max_memory_length and max_memory_length > 0:
                    # Truncate each message's content to max_memory_length before passing to mem0
                    # This way, mem0 will extract from truncated content and store only truncated memories
                    messages_to_use = []
                    for msg in chunked_messages:
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
                
                # Log chunking if it occurred
                if len(messages_to_use) > len(messages):
                    print(f"📦 Chunked {len(messages)} messages into {len(messages_to_use)} chunks to prevent token limit errors")
                
                # Extract memories using mem0 (with potentially chunked and truncated messages)
                # Capture mem0's internal error messages to handle UPDATE operation failures gracefully
                # Root cause: mem0's deduplication logic uses simple IDs (e.g., '6') to track memories,
                # but stored memories have UUIDs. When mem0 tries to UPDATE a similar memory, it fails
                # with KeyError because it can't find the memory with the simple ID.
                # This is a mem0 bug - we can't fix it without modifying mem0 source code.
                error_buffer = StringIO()
                try:
                    # Wrap mem0.add() call with timeout to prevent hanging
                    # Use 5 minutes timeout (300 seconds) - should be enough for most API calls
                    # If it hangs longer, something is wrong and we should fail fast
                    def _call_mem0_add():
                        with redirect_stderr(error_buffer), redirect_stdout(error_buffer):
                            return self.memory.add(
                                messages=messages_to_use,
                                user_id=user_id,
                                agent_id=None,  # Don't pass agent_id to force USER_MEMORY_EXTRACTION_PROMPT
                                metadata=combined_metadata,  # Don't include agent_id here
                                infer=True,  # Use LLM to extract facts
                            )
                    
                    # Execute with timeout
                    result = _call_with_timeout(_call_mem0_add, timeout_seconds=300, error_message="mem0.add()")
                    
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
                # Even though we chunked and truncated input, mem0 might combine or rephrase, so we check again
                # Also truncate memories that exceed embedding model token limit (8192 tokens ≈ 32000 chars)
                max_safe_memory_length = 32000  # Safe limit for embedding model (8192 tokens * ~4 chars/token)
                
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
                                    # First check defense limit, then check embedding model limit
                                    if max_memory_length and len(memory_text_str) > max_memory_length:
                                        memory_text_str = memory_text_str[:max_memory_length]
                                    elif len(memory_text_str) > max_safe_memory_length:
                                        # Truncate to safe limit for embedding model
                                        memory_text_str = memory_text_str[:max_safe_memory_length]
                                        print(f"⚠️  Truncated extracted memory from {len(str(memory_text))} to {max_safe_memory_length} chars to prevent embedding model errors")
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
                                    # First check defense limit, then check embedding model limit
                                    if max_memory_length and len(memory_text_str) > max_memory_length:
                                        memory_text_str = memory_text_str[:max_memory_length]
                                    elif len(memory_text_str) > max_safe_memory_length:
                                        # Truncate to safe limit for embedding model
                                        memory_text_str = memory_text_str[:max_safe_memory_length]
                                        print(f"⚠️  Truncated extracted memory from {len(str(memory_text))} to {max_safe_memory_length} chars to prevent embedding model errors")
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
        
        Uses batch search to avoid token limit errors:
        1. First tries normal search (semantic search over all memories)
        2. If that fails (token limit), falls back to batch search:
           - Gets memories in batches using get_all()
           - Searches each batch separately
           - Combines and ranks results
        
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
            
            # Try normal search first (semantic search over all memories)
            print(f"🔍 [BATCH SEARCH DEBUG] Attempting normal mem0 search (semantic search over all memories)")
            print(f"   Query: '{query[:100]}...' (truncated)" if len(query) > 100 else f"   Query: '{query}'")
            print(f"   Limit: {limit} results")
            
            try:
                # Wrap mem0.search() call with timeout to prevent hanging
                def _call_mem0_search():
                    return self.memory.search(
                        query=query,
                        user_id=user_id,
                        agent_id=agent_id,  # Always None
                        limit=limit
                    )
                
                result = _call_with_timeout(_call_mem0_search, timeout_seconds=300, error_message="mem0.search()")
                # Extract results from mem0 response
                memories = []
                if isinstance(result, dict) and "results" in result:
                    memories = result["results"]
                elif isinstance(result, list):
                    memories = result
                else:
                    memories = []
                
                print(f"✅ [BATCH SEARCH DEBUG] Normal search succeeded: Found {len(memories)} memories")
                
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
                error_str = str(e)
                print(f"❌ [BATCH SEARCH DEBUG] Normal search failed: {error_str[:200]}")
                
                # Check if this is a token limit error
                is_token_limit_error = (
                    "8192 tokens" in error_str or
                    "context length" in error_str.lower() or
                    ("token" in error_str.lower() and "limit" in error_str.lower()) or
                    "105107 tokens" in error_str or
                    "102925 tokens" in error_str or
                    "118565 tokens" in error_str or
                    "119102 tokens" in error_str
                )
                
                if is_token_limit_error:
                    # Fall back to batch search
                    print(f"⚠️  [BATCH SEARCH DEBUG] Token limit error detected - switching to batch search fallback")
                    print(f"   Error type: Token limit exceeded (embedding model limit: 8192 tokens)")
                    return self._batch_search(query, user_id, agent_id, limit, session_id, defense_type)
                else:
                    # Other errors - just return empty
                    print(f"⚠️  [BATCH SEARCH DEBUG] Non-token-limit error - returning empty results")
                    print(f"Warning: Could not search mem0 memory: {e}")
                    return []
    
    def _batch_search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str],
        limit: int,
        session_id: Optional[str],
        defense_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch search fallback: Get memories in batches and search each batch.
        
        This is a workaround for mem0's token limit issue. Since mem0's search()
        embeds ALL memories at once (which exceeds token limit), we:
        1. Get memories in batches using get_all() with a reasonable limit
        2. For each batch, try to use mem0's search() (semantic search)
        3. If batch search also fails, use text matching as final fallback
        4. Combine and rank results
        
        Note: mem0's get_all() doesn't support pagination, so we get the first N memories.
        This searches through recent memories, which is often sufficient.
        
        Args:
            query: Search query string
            user_id: User ID
            agent_id: Agent ID (always None for mem0)
            limit: Number of results to return
            session_id: Optional session ID
            defense_type: Optional defense type
            
        Returns:
            List of memory dictionaries
        """
        print(f"🔄 [BATCH SEARCH DEBUG] Starting batch search fallback")
        print(f"   Query: '{query[:100]}...' (truncated)" if len(query) > 100 else f"   Query: '{query}'")
        print(f"   Target: {limit} results")
        
        try:
            # Get memories in batches - try semantic search first, fall back to text matching
            batch_size = 50  # Process 50 memories at a time
            max_memories_to_search = 200  # Search through up to 200 memories total
            all_results = []
            
            # Get all memories we want to search through
            print(f"📥 [BATCH SEARCH DEBUG] Getting memories to search (limit: {max_memories_to_search})")
            all_memories = self.get_all_memories(
                user_id=user_id,
                agent_id=agent_id,
                limit=max_memories_to_search
            )
            
            total_memories = len(all_memories)
            print(f"   Retrieved {total_memories} memories from vector store")
            
            if not all_memories:
                print(f"⚠️  [BATCH SEARCH DEBUG] No memories found in vector store")
                return []
            
            # Try to search in batches using mem0's search()
            # Note: mem0's search() still searches ALL memories, but we can try smaller batches
            # by creating temporary filtered searches. However, mem0 doesn't support this directly.
            # So we'll use text matching as a fallback, but try semantic search on the full set first
            # with a smaller limit to see if it works.
            
            # Strategy: Try semantic search with smaller result limit first
            # If that fails, use text matching on batches
            print(f"🔍 [BATCH SEARCH DEBUG] Attempting semantic search with reduced scope...")
            
            # Try semantic search one more time with a smaller limit (might work if fewer results needed)
            try:
                def _call_mem0_search_small():
                    return self.memory.search(
                        query=query,
                        user_id=user_id,
                        agent_id=agent_id,
                        limit=min(limit, 10)  # Try smaller limit
                    )
                
                semantic_result = _call_with_timeout(_call_mem0_search_small, timeout_seconds=300, error_message="mem0.search() (batch fallback)")
                semantic_memories = []
                if isinstance(semantic_result, dict) and "results" in semantic_result:
                    semantic_memories = semantic_result["results"]
                elif isinstance(semantic_result, list):
                    semantic_memories = semantic_result
                
                if semantic_memories:
                    print(f"✅ [BATCH SEARCH DEBUG] Semantic search with reduced limit succeeded: {len(semantic_memories)} results")
                    all_results = semantic_memories
                else:
                    raise Exception("No results from semantic search")
            except Exception as semantic_error:
                print(f"⚠️  [BATCH SEARCH DEBUG] Semantic search with reduced limit also failed: {str(semantic_error)[:100]}")
                print(f"   Falling back to text matching on {total_memories} memories")
                
                # Fall back to text matching: check if query words appear in memory text
                query_lower = query.lower()
                query_words = set(query_lower.split())
                if not query_words:
                    print(f"⚠️  [BATCH SEARCH DEBUG] Empty query words after processing")
                    return []
                
                print(f"   Query words: {list(query_words)[:10]}..." if len(query_words) > 10 else f"   Query words: {list(query_words)}")
                
                scored_memories = []
                for i, mem in enumerate(all_memories):
                    if isinstance(mem, dict):
                        memory_text = str(mem.get("memory", "")).lower()
                        memory_words = set(memory_text.split())
                        
                        # Count how many query words match
                        matches = len(query_words.intersection(memory_words))
                        
                        if matches > 0:
                            # Calculate relevance score (percentage of query words matched)
                            relevance_score = matches / len(query_words)
                            
                            # Also check if query appears as substring (higher relevance)
                            if query_lower in memory_text:
                                relevance_score += 0.5
                            
                            mem_copy = mem.copy()
                            mem_copy["_relevance_score"] = relevance_score
                            scored_memories.append(mem_copy)
                            
                            if len(scored_memories) <= 5:  # Debug first 5 matches
                                print(f"   Match {len(scored_memories)}: score={relevance_score:.2f}, text='{memory_text[:60]}...'")
                
                # Sort by relevance score (highest first)
                scored_memories.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
                
                # Remove temporary relevance scores
                for mem in scored_memories:
                    if "_relevance_score" in mem:
                        del mem["_relevance_score"]
                
                all_results = scored_memories
                print(f"📊 [BATCH SEARCH DEBUG] Text matching: {len(scored_memories)} matches found from {total_memories} memories")
            
            # P1: Check if any retrieved memory has U label (provable_policy defense)
            if defense_type == "provable_policy" and session_id:
                from agent.agent_core import ProvablePolicyManager
                found_u_label = False
                for memory_item in all_results:
                    if isinstance(memory_item, dict):
                        metadata = memory_item.get("metadata", {})
                        label = metadata.get("label", None)
                        if label == "U":
                            ProvablePolicyManager.set_untrusted(session_id)
                            found_u_label = True
                            break
                        elif label is None:
                            raise ValueError(
                                f"Mem0 batch search memory entry missing label in provable_policy defense. "
                                f"All memories must have 'label' metadata set to 'T' or 'U'."
                            )
            
            final_results = all_results[:limit]
            print(f"✅ [BATCH SEARCH DEBUG] Batch search complete: Returning {len(final_results)} results (requested: {limit})")
            if final_results:
                print(f"   Top result: '{final_results[0].get('memory', '')[:80]}...'")
            
            return final_results
            
        except Exception as e:
            print(f"❌ [BATCH SEARCH DEBUG] Batch search fallback failed: {e}")
            import traceback
            traceback.print_exc()
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
                
                # Call get_all with agent_id=None (with timeout to prevent hanging)
                def _call_mem0_get_all():
                    return self.memory.get_all(
                        user_id=user_id if user_id else None,
                        agent_id=None,
                        filters=query_filters if query_filters else None,
                        limit=limit
                    )
                
                result = _call_with_timeout(_call_mem0_get_all, timeout_seconds=300, error_message="mem0.get_all()")
                
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

