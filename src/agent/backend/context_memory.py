"""
Context Memory Module
Implements simple context-based memory that stores all conversation history.
No embeddings or retrieval - just accumulates all messages in order.

Features:
- Simple list-based storage for all conversation history
- No embeddings or vector stores needed
- All history retrieved for every query (no filtering)
- Continues growing across sessions
- Sliding window truncation to handle context length limits

Defense Mechanisms:
1. disable_memory: Baseline defense that disables all memory indexing
2. none: No defense (normal context execution - all messages indexed)
3. user_prompt_only: Only index user messages (filter out assistant messages)
4. no_untrusted_tools: Disable context indexing for the rest of the session once an
   untrusted tool has been used
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import json
from datetime import datetime, timezone
from agent.utils import debug_info, debug_debug, debug_print_exception

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    debug_info("Warning: tiktoken not available. Context truncation will use character-based estimation.")


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
    Manages context-based memory by storing all conversation history.
    This is a simplified version of RAG - no embeddings, just store everything.
    """
    
    def __init__(
        self,
        context_path: Optional[str] = None,
        max_context_length: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the context memory manager.
        
        Args:
            context_path: Optional path to persist context history
            max_context_length: Maximum number of tokens to keep in context (None = no limit)
            model_name: Model name for tokenizer (defaults to gpt-4o-mini if tiktoken available)
        """
        self.context_path = Path(context_path) if context_path else None
        self._lock = threading.Lock()
        
        # Store all conversation history as a list of dicts with text and label
        # Format: [{"text": "...", "label": "T" or "U"}, ...]
        # For backward compatibility, can also handle strings (default to T)
        self.history: List[Dict[str, str]] = []
        
        # Context length limits (for sliding window truncation)
        self.max_context_length = max_context_length
        
        # Initialize tokenizer for token counting
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE and max_context_length is not None:
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
        
        # Load existing context if path provided
        if self.context_path and self.context_path.exists():
            self._load_context()
    
    def _load_context(self):
        """Load context history from disk if it exists."""
        try:
            if self.context_path and self.context_path.exists():
                with open(self.context_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_history = data.get("history", [])
                    # Convert old string format to dict format for backward compatibility
                    self.history = []
                    for entry in raw_history:
                        if isinstance(entry, str):
                            # Old format: just a string, default to T (Trusted)
                            self.history.append({"text": entry, "label": "T"})
                        elif isinstance(entry, dict):
                            # New format: dict with text and label
                            self.history.append(entry)
                        else:
                            # Skip invalid entries
                            continue
                debug_debug(f"Loaded existing context from {self.context_path} ({len(self.history)} entries)")
        except Exception as e:
            debug_info(f"Could not load context from {self.context_path} (will start with empty context)")
            debug_print_exception(e, context=f"Loading context memory from {self.context_path}", include_traceback=True)
            self.history = []
    
    def _save_context(self):
        """Save context history to disk if path is configured."""
        if self.context_path:
            try:
                self.context_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.context_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "history": self.history,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                debug_info(f"Could not save context to {self.context_path}")
                debug_print_exception(e, context=f"Saving context memory to {self.context_path}", include_traceback=True)
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None, defense_type: Optional[str] = None):
        """
        Add text to the context memory system.
        
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
                # (for consistency, but label won't be checked)
                self.history.append({"text": chunk_text, "label": "T"})
            
            # Track recent additions for efficient validation
            # Store in a file in the test directory (if context_path is in a test directory)
            if self.context_path:
                try:
                    context_path_obj = Path(self.context_path)
                    # Check if this is a test directory (contains "test_env" or "context_memory" in test_envs)
                    if "test_env" in str(context_path_obj) or "test_envs" in str(context_path_obj):
                        # Get the test directory (parent of context_memory file)
                        test_dir = context_path_obj.parent
                        recent_chunks_file = test_dir / "context_recent_chunks.json"
                        
                        # Read existing recent chunks
                        recent_chunks = []
                        if recent_chunks_file.exists():
                            try:
                                with open(recent_chunks_file, 'r', encoding='utf-8') as f:
                                    recent_chunks = json.load(f)
                            except Exception as e:
                                debug_debug(f"Could not read recent chunks file {recent_chunks_file}, starting with empty list")
                                debug_print_exception(e, context=f"Reading recent chunks from {recent_chunks_file}", include_traceback=True)
                                recent_chunks = []
                        
                        # Add new chunk (extract text from dict if needed)
                        chunk_text_for_file = chunk_text if isinstance(chunk_text, str) else chunk_text.get("text", "")
                        recent_chunks.append(chunk_text_for_file)
                        
                        # Write back (keep only recent chunks, limit to last 100 to avoid file bloat)
                        recent_chunks = recent_chunks[-100:]
                        with open(recent_chunks_file, 'w', encoding='utf-8') as f:
                            json.dump(recent_chunks, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    # Fail gracefully if we can't write recent chunks (non-critical)
                    debug_debug("Could not write recent chunks for validation (non-critical)")
                    debug_print_exception(e, context="Writing recent chunks in context memory", include_traceback=True)
            
            # Save if path is configured
            if self.context_path:
                self._save_context()
    
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
                debug_debug(f"🔍 [DEBUG] context retrieve: Checking {len(self.history)} history entries for U labels (session_id={session_id})")
                found_u_label = False
                for i, entry in enumerate(self.history):
                    # Handle both dict and string formats
                    if isinstance(entry, dict):
                        label = entry.get("label", None)
                        text_preview = entry.get("text", "")[:50]
                        debug_debug(f"🔍 [DEBUG] context History {i}: label={label}, text_preview='{text_preview}'")
                        if label == "U":
                            # Upgrade session to U if U-labeled memory is retrieved
                            debug_info(f"🛡️ [DEBUG] context: Found U-labeled memory! Upgrading session '{session_id}' to UNTRUSTED")
                            ProvablePolicyManager.set_untrusted(session_id)
                            found_u_label = True
                            break
                        elif label is None:
                            # Error: memory should have a label
                            debug_info(f"⚠️ [DEBUG] context History {i} missing label! text_preview='{text_preview}'")
                            raise ValueError(
                                f"Context memory entry missing label in provable_policy defense. "
                                f"All memories must have 'label' metadata set to 'T' or 'U'. "
                                f"Entry text preview: {text_preview}..."
                            )
                if not found_u_label:
                    debug_debug(f"🔍 [DEBUG] context: No U-labeled memories found. Session '{session_id}' remains trusted.")
            
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
                debug_info(f"⚠️  Large context detected ({len(context_string)} chars), checking truncation...")
            context_string = self._truncate_context(context_string)
            if len(context_string) != original_length:
                debug_info(f"⚠️  Context memory truncated: {original_length} → {len(context_string)} chars")
            elif len(context_string) > 100000:
                debug_debug(f"ℹ️  Context memory: {len(context_string)} chars (no truncation needed)")
        
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
                    debug_debug(f"🔄 Encoding large context ({len(context)} chars) for truncation...")
                
                encoded = self.tokenizer.encode(context, disallowed_special=())
                token_count = len(encoded)
                
                # If within limit, return as-is
                if token_count <= self.max_context_length:
                    if len(context) > 100000:
                        debug_debug(f"ℹ️  Context memory: {token_count} tokens (within limit of {self.max_context_length} tokens, no truncation needed)")
                    return context
                
                # Truncate to keep most recent tokens (sliding window from end)
                if len(context) > 100000:
                    debug_info(f"🔄 Truncating context memory from {token_count} to {self.max_context_length} tokens (keeping most recent, evicting oldest)...")
                truncated_encoded = encoded[-self.max_context_length:]
                result = self.tokenizer.decode(truncated_encoded)
                if len(context) > 100000:
                    result_tokens = len(self.tokenizer.encode(result, disallowed_special=()))
                    debug_info(f"✅ Truncation complete: {len(result)} chars ({result_tokens} tokens), evicted {token_count - result_tokens} tokens from beginning")
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
        """Clear all memory from the context."""
        with self._lock:
            self.history = []
            
            # Delete persisted context if it exists
            if self.context_path and self.context_path.exists():
                try:
                    self.context_path.unlink()
                except Exception as e:
                    debug_info(f"Could not delete context file at {self.context_path}")
                    debug_print_exception(e, context=f"Deleting context file at {self.context_path}", include_traceback=True)
    
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
    context_path: Optional[str] = None,
    max_context_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> ContextMemoryManager:
    """
    Create a new context memory manager instance.
    
    Args:
        context_path: Optional path to persist context history
        max_context_length: Maximum number of tokens to keep in context (None = no limit)
        model_name: Model name for tokenizer (defaults to gpt-4o-mini if tiktoken available)
    
    Returns:
        A new ContextMemoryManager instance
    """
    return ContextMemoryManager(
        context_path=context_path,
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
