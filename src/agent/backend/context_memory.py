"""
Context Memory Module
Implements simple context-based memory that stores all conversation history.
No embeddings or retrieval - just accumulates all messages in order.

This benchmark operates entirely in-memory - no file system operations.

Features:
- Simple list-based storage for all conversation history
- No embeddings or vector stores needed
- All history retrieved for every query (no filtering)
- Sliding window truncation to handle context length limits

Defense Mechanisms:
1. disable_memory: Baseline defense that disables all memory indexing
2. none: No defense (normal context execution - all messages indexed)
3. user_prompt_only: Only index user messages (filter out assistant messages)
4. no_untrusted_tools: Disable context indexing for the rest of the session once an
   untrusted tool has been used
"""

from typing import List, Dict, Any, Optional
import threading
import tiktoken
from agent.utils import debug_info, debug_debug, debug_print_exception


class ContextDefenseManager:
    """
    Manages defense mechanisms for context memory indexing.
    """

    def __init__(
        self,
        defense_type: str = "none"
    ):
        """
        Initialize the context defense manager.

        Args:
            defense_type: Type of defense to apply. Options:
                - "none": No defense (default behavior - all messages indexed)
                - "disable_memory": Disable all memory indexing
                - "user_prompt_only": Only index user messages
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

        # Defense: No untrusted tools – once any untrusted tool is used in
        # the session, disable context indexing for the rest of the session.
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
        # Defense: User-only indexing
        if self.defense_type == "user_prompt_only":
            # Only include user message, ignore assistant message
            return f"User: {user_message}"

        # No defense or other defenses: include both messages
        return f"User: {user_message}\nAssistant: {assistant_message}"


class ContextMemoryManager:
    """
    Manages context-based memory by storing all conversation history (in-memory only).
    This is a simplified version of RAG - no embeddings, just store everything.
    """
    
    def __init__(
        self,
        max_context_length: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the context memory manager (in-memory only).
        
        Args:
            max_context_length: Maximum number of tokens to keep in context (None = no limit)
            model_name: Model name for tokenizer (defaults to gpt-4o-mini if tiktoken available)
        """
        self._lock = threading.Lock()
        
        # Store all conversation history as a list of dicts with text and label
        # Format: [{"text": "...", "label": "T" or "U"}, ...]
        self.history: List[Dict[str, str]] = []
        
        # Context length limits (for sliding window truncation)
        self.max_context_length = max_context_length
        
        # Initialize tokenizer for token counting
        self.tokenizer = None
        if max_context_length is not None:
            try:
                # Use provided model_name or default to gpt-4o-mini
                tokenizer_model = model_name if model_name else "gpt-4o-mini"
                # Try to get encoding for the model, fallback to gpt-4o-mini if not found
                try:
                    self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
                except KeyError:
                    # Fallback to gpt-4o-mini if model not found
                    self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            except Exception as e:
                debug_info("Could not initialize tokenizer for context memory (will use character-based estimation)")
                debug_print_exception(e, context="Initializing tokenizer for context memory", include_traceback=True)
                self.tokenizer = None
        # Start with empty history (in-memory only)
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None):
        """
        Add text to the context memory system (in-memory only).
        
        Args:
            text: Text content to add to memory
            metadata: Optional metadata dictionary (not used, kept for compatibility)
            session_id: Optional session ID for provable_policy defense (P3: Memory Labeling)
            defense_type: Optional defense type to check if provable_policy is active
        """
        with self._lock:
            if not text or not text.strip():
                return
            
            # Simply append to history (no chunking, no embeddings)
            chunk_text = text.strip()
            
            # P3: Memory Labeling - inherit session label for provable_policy defense
            if defense_type == "provable_policy":
                if session_id:
                    from agent.agent_core import ProvablePolicyManager
                    session_label = ProvablePolicyManager.get_session_label(session_id)
                    self.history.append({"text": chunk_text, "label": session_label})
                else:
                    # Default to T if no session_id provided
                    self.history.append({"text": chunk_text, "label": "T"})
            else:
                # For non-provable_policy defenses, store as dict with default T label
                self.history.append({"text": chunk_text, "label": "T"})
    
    def retrieve(self, query: str = "", top_k: Optional[int] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None) -> List[str]:  # noqa: ARG002
        """
        Retrieve all context history (no filtering - returns everything).
        
        Args:
            query: The search query (ignored - we return everything)
            top_k: Number of documents to retrieve (ignored - we return everything)
            session_id: Optional session ID for provable_policy defense (P1: Check labels on retrieval)
            defense_type: Optional defense type to check if provable_policy is active
            
        Returns:
            List of all stored context entry texts
        """
        with self._lock:
            # P1: Check if any retrieved memory has U label (provable_policy defense)
            # This is critical: if U-labeled memories from previous sessions are loaded into a new session,
            # the new session must be marked as untrusted immediately
            if defense_type == "provable_policy" and session_id:
                from agent.agent_core import ProvablePolicyManager
                debug_debug(f"[DEBUG] context retrieve: Checking {len(self.history)} history entries for U labels (session_id={session_id})")
                found_u_label = False
                for i, entry in enumerate(self.history):
                    # Handle both dict and string formats
                    if isinstance(entry, dict):
                        label = entry.get("label", None)
                        text_preview = entry.get("text", "")[:50]
                        debug_debug(f"[DEBUG] context History {i}: label={label}, text_preview='{text_preview}'")
                        if label == "U":
                            # Upgrade session to U if U-labeled memory is retrieved
                            debug_info(f"[DEBUG] context: Found U-labeled memory! Upgrading session '{session_id}' to UNTRUSTED")
                            ProvablePolicyManager.set_untrusted(session_id)
                            found_u_label = True
                            break
                        elif label is None:
                            # Error: memory should have a label
                            debug_info(f"WARNING: [DEBUG] context History {i} missing label! text_preview='{text_preview}'")
                            raise ValueError(
                                f"Context memory entry missing label in provable_policy defense. "
                                f"All memories must have 'label' metadata set to 'T' or 'U'. "
                                f"Entry text preview: {text_preview}..."
                            )
                if not found_u_label:
                    debug_debug(f"[DEBUG] context: No U-labeled memories found. Session '{session_id}' remains trusted.")
            
            # Return text content from history entries
            result: List[str] = []
            for entry in self.history:
                if isinstance(entry, dict):
                    text = entry.get("text", "")
                    result.append(text if isinstance(text, str) else str(text))
                else:
                    result.append(str(entry))
            return result
    
    def get_context(self, query: str = "", top_k: Optional[int] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None) -> str:  # noqa: ARG002
        """
        Get formatted context string with all history.
        Applies sliding window truncation if max_context_length is set.
        
        Args:
            query: The search query (ignored)
            top_k: Number of documents to retrieve (ignored)
            session_id: Optional session ID for provable_policy defense
            defense_type: Optional defense type to check if provable_policy is active
            
        Returns:
            Formatted context string with all stored history (truncated if needed)
        """
        retrieved = self.retrieve(query, top_k, session_id=session_id, defense_type=defense_type)
        
        if not retrieved:
            return ""
        
        # Format context
        context_parts = []
        for i, memory_text in enumerate(retrieved, 1):
            context_parts.append(f"Memory {i}:\n{memory_text}")
        
        context_string = "\n\n".join(context_parts)
        
        # Apply sliding window truncation if max_context_length is set
        if self.max_context_length is not None and self.max_context_length > 0:
            # Add debug output for large contexts
            original_length = len(context_string)
            if len(context_string) > 100000:  # If context is > 100k chars, warn
                debug_info(f"WARNING: Large context detected ({len(context_string)} chars), checking truncation...")
            context_string = self._truncate_context(context_string)
            if len(context_string) != original_length:
                debug_info(f"WARNING: Context memory truncated: {original_length} → {len(context_string)} chars")
            elif len(context_string) > 100000:
                debug_debug(f"Context memory: {len(context_string)} chars (no truncation needed)")
        
        return context_string
    
    def _truncate_context(self, context: str) -> str:
        """
        Truncate context using sliding window approach (keep most recent tokens).
        Similar to MemoryAgentBench's approach.
        
        Args:
            context: The full context string
            
        Returns:
            Truncated context string (most recent tokens)
        """
        if not context or not self.max_context_length:
            return context
        
        # Count tokens if tokenizer available, otherwise use character-based estimation
        if self.tokenizer:
            try:
                # For very large contexts, encoding can be slow - add progress indicator
                if len(context) > 100000:
                    debug_debug(f"Encoding large context ({len(context)} chars) for truncation...")
                
                encoded = self.tokenizer.encode(context, disallowed_special=())
                token_count = len(encoded)
                
                # If within limit, return as-is
                if token_count <= self.max_context_length:
                    if len(context) > 100000:
                        debug_debug(f"Context memory: {token_count} tokens (within limit of {self.max_context_length} tokens, no truncation needed)")
                    return context
                
                # Truncate to keep most recent tokens (sliding window from end)
                if len(context) > 100000:
                    debug_info(f"Truncating context memory from {token_count} to {self.max_context_length} tokens (keeping most recent, evicting oldest)...")
                truncated_encoded = encoded[-self.max_context_length:]
                result = self.tokenizer.decode(truncated_encoded)
                if len(context) > 100000:
                    result_tokens = len(self.tokenizer.encode(result, disallowed_special=()))
                    debug_info(f"Truncation complete: {len(result)} chars ({result_tokens} tokens), evicted {token_count - result_tokens} tokens from beginning")
                return result
            except Exception as e:
                debug_info(f"Warning: Error during token-based truncation: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to character-based truncation
                return self._truncate_context_by_chars(context)
        else:
            # Fallback: use character-based estimation (rough approximation: 4 chars per token)
            estimated_tokens = len(context) // 4
            if estimated_tokens <= self.max_context_length:
                return context
            
            # Truncate characters (rough approximation)
            max_chars = self.max_context_length * 4
            return context[-max_chars:]
    
    def _truncate_context_by_chars(self, context: str) -> str:
        """
        Fallback truncation using character count (rough approximation).
        
        Args:
            context: The full context string
            
        Returns:
            Truncated context string
        """
        # Rough approximation: 4 characters per token
        if self.max_context_length is None:
            return context
        max_chars = self.max_context_length * 4
        if len(context) <= max_chars:
            return context
        return context[-max_chars:]
    
    def clear_all_memory(self):
        """Clear all memory from the context (in-memory only)."""
        with self._lock:
            self.history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            return {
                "total_entries": len(self.history),
                "context_initialized": len(self.history) > 0,
            }


def get_context_memory_manager(
    max_context_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> ContextMemoryManager:
    """
    Create a new context memory manager instance (in-memory only).
    
    Args:
        max_context_length: Maximum number of tokens to keep in context (None = no limit)
        model_name: Model name for tokenizer (defaults to gpt-4o-mini if tiktoken available)
    
    Returns:
        A new ContextMemoryManager instance
    """
    return ContextMemoryManager(
        max_context_length=max_context_length,
        model_name=model_name,
    )


def get_context_defense_manager(
    defense_type: str = "none",
    session_id: Optional[str] = None,
) -> ContextDefenseManager:
    """
    Create a new context defense manager instance.

    Args:
        defense_type: Type of defense to apply
        session_id: Optional session ID (kept for API compatibility, not used)

    Returns:
        A new ContextDefenseManager instance
    """
    return ContextDefenseManager(
        defense_type=defense_type
    )


def get_context_memory_context(text: str, session_id: str, model_name: str, memory_config: dict, memory_backend: str) -> str:
    """Retrieve context memory context if enabled."""
    context_memory_config = memory_config.get("context_memory", {})
    context_memory_enabled = context_memory_config.get("enabled", False) or (memory_backend == "context")
    unified_defense_type = context_memory_config.get("defense_type", "none")
    # Use local mapper function (avoid circular import)
    context_defense_type = map_unified_defense(unified_defense_type)
    
    if not context_memory_enabled or context_defense_type == "disable_memory":
        return ""
    
    try:
        # Use shared manager from config if available (persists memories across invocations)
        context_memory_manager = context_memory_config.get("manager")
        if not context_memory_manager:
            # Fallback: create new manager (memories won't persist across calls)
            max_context_length = context_memory_config.get("max_context_length")
            context_memory_manager = get_context_memory_manager(
                max_context_length=max_context_length,
                model_name=model_name,
            )
        context_memory_context = context_memory_manager.get_context(
            text, session_id=session_id, defense_type=context_defense_type
        )
        return "\n\n# Relevant Context Memory\n" + context_memory_context + "\n" if context_memory_context else ""
    except (OSError, IOError, ValueError, RuntimeError) as e:
        debug_debug(f"Could not retrieve context memory: {e}")
        return ""


def index_context_memory(text: str, response_text: str, session_id: str, model_name: str, context_memory_config: dict, context_defense_type: str) -> None:
    """Index conversation into context memory if enabled and defense allows."""
    try:
        defense_manager = get_context_defense_manager(defense_type=context_defense_type, session_id=session_id)
        
        if not defense_manager.should_index_memory(session_id, text, response_text):
            return
        
        conversation_turn = defense_manager.filter_conversation_turn(text, response_text)
        
        # Use shared manager from config if available (persists memories across invocations)
        context_memory_manager = context_memory_config.get("manager")
        if not context_memory_manager:
            # Fallback: create new manager (memories won't persist across calls)
            max_context_length = context_memory_config.get("max_context_length")
            context_memory_manager = get_context_memory_manager(
                max_context_length=max_context_length,
                model_name=model_name,
            )
        
        if conversation_turn.strip():
            context_memory_manager.add_memory(
                conversation_turn,
                metadata={"session_id": session_id, "type": "conversation", "defense_type": context_defense_type},
                session_id=session_id,
                defense_type=context_defense_type
            )
    except (OSError, IOError, ValueError, RuntimeError) as e:
        debug_debug(f"Could not index context memory: {e}")


# ============================================================================
# Defense Mapping and Test Utilities
# ============================================================================

def map_unified_defense(unified_defense: str) -> str:
    """
    Map unified defense name to context backend-specific defense type.
    
    Args:
        unified_defense: Unified defense name (e.g., "none", "user_prompt_only")
        
    Returns:
        Backend-specific defense type string
        
    Note:
        limit_memory_length is NOT applicable for context backend.
    """
    # Context uses the same names as unified defenses, except limit_memory_length is not applicable
    DEFENSE_MAP = {
        "disable_memory": "disable_memory",
        "none": "none",
        "user_prompt_only": "user_prompt_only",
        "no_untrusted_tools": "no_untrusted_tools",
        "provable_policy": "provable_policy",
        # limit_memory_length is NOT included - not applicable for context
    }
    return DEFENSE_MAP.get(unified_defense, unified_defense)


def get_memory_state_for_test(test_dir, config: Dict[str, Any]) -> List[str]:
    """
    Get context memory contents for test validation (from in-memory storage).
    
    Args:
        test_dir: Ignored - kept for API compatibility
        config: Configuration dictionary
        
    Returns:
        List of memory entry strings from in-memory storage
    """
    # Get from in-memory manager
    context_config = config.get("memory", {}).get("context_memory", {})
    max_context_length = context_config.get("max_context_length")
    model_name = config.get("agent", {}).get("target_model_name", "gpt-5-mini")
    
    context_manager = get_context_memory_manager(
        max_context_length=max_context_length,
        model_name=model_name,
    )
    
    # Extract text from dict format
    result = []
    for entry in context_manager.history:
        if isinstance(entry, dict):
            text = entry.get("text", "")
            if text:
                result.append(str(text))
        else:
            result.append(str(entry))
    return result
