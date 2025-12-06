"""
RAG Defense Manager

Manages defense mechanisms for RAG memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. none: No defense (normal RAG execution - all messages indexed)
3. user_only: Only index user messages (filter out assistant messages)
4. limit_chunk_size: Limit chunk sizes to 8 tokens each
5. no_untrusted_tools: Disable RAG indexing for the rest of the session once an
   untrusted tool has been used
"""

from typing import Dict, Optional, Set

from agent.utils import read_trace_events


# Untrusted tools are those that read from untrusted data sources (inbox)
UNTRUSTED_TOOLS: Set[str] = {
    "read_all_emails",
    "search_emails",
    "reply_to_email",
    "forward_to_email",
}


class RAGDefenseManager:
    """
    Manages defense mechanisms for RAG memory indexing.
    """

    def __init__(
        self,
        defense_type: str = "none",
        trace_file: Optional[str] = None,
    ):
        """
        Initialize the RAG defense manager.

        Args:
            defense_type: Type of defense to apply. Options:
                - "none": No defense (default behavior - all messages indexed)
                - "disable_memory": Disable all memory indexing
                - "user_only": Only index user messages
                - "limit_chunk_size": Limit chunk sizes to 8 tokens
                - "no_untrusted_tools": Disable indexing once an untrusted tool
                  has been used in the session
            trace_file: Path to trace file for checking tool calls (needed for
                'no_untrusted_tools' behavior)
        """
        self.defense_type = defense_type
        self.trace_file = trace_file
        # Cache of whether an untrusted tool has been seen per session
        self._session_untrusted_tool_called: Dict[str, bool] = {}

    def _has_untrusted_tool_been_called(self, session_id: str) -> bool:
        """
        Check if any untrusted tool has been called in this session.

        This mirrors the mem0 'no_untrusted_tools' behavior: once an untrusted
        tool is detected in the trace for a session, we treat the rest of the
        session as untrusted and block RAG indexing.
        """
        # Check cache first
        if session_id in self._session_untrusted_tool_called:
            return self._session_untrusted_tool_called[session_id]

        # If no trace file, assume no untrusted tools called
        if not self.trace_file:
            self._session_untrusted_tool_called[session_id] = False
            return False

        try:
            trace_events = read_trace_events(self.trace_file, session_id)
            for event in trace_events:
                if event.get("event_type") == "tool_call":
                    payload = event.get("payload", {})
                    tool_name = payload.get("tool_name", "")
                    if tool_name in UNTRUSTED_TOOLS:
                        self._session_untrusted_tool_called[session_id] = True
                        return True
        except Exception as e:
            print(f"Warning: Could not check trace events for RAG defense: {e}")

        self._session_untrusted_tool_called[session_id] = False
        return False

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
            if self._has_untrusted_tool_been_called(session_id):
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
        if self.defense_type == "user_only":
            # Only include user message, ignore assistant message
            return f"User: {user_message}"

        # No defense or other defenses: include both messages
        return f"User: {user_message}\nAssistant: {assistant_message}"

    def get_chunk_size(
        self,
        default_chunk_size: int = 512,
    ) -> int:
        """
        Get the chunk size to use based on defense type.

        Args:
            default_chunk_size: Default chunk size from config

        Returns:
            Chunk size to use
        """
        # Defense 4: Limit chunk size to 8 tokens
        if self.defense_type == "limit_chunk_size":
            return 8

        # Other defenses use default chunk size
        return default_chunk_size


# Global defense manager cache (per session)
_rag_defense_manager_cache: Dict[str, RAGDefenseManager] = {}


def get_rag_defense_manager(
    defense_type: str = "none",
    session_id: Optional[str] = None,
    force_new: bool = False,
    trace_file: Optional[str] = None,
) -> RAGDefenseManager:
    """
    Get or create a RAG defense manager instance.

    Args:
        defense_type: Type of defense to apply
        session_id: Optional session ID for caching
        force_new: If True, create a new instance
        trace_file: Optional trace file path (needed for 'no_untrusted_tools')

    Returns:
        RAGDefenseManager instance
    """
    global _rag_defense_manager_cache

    # Use session_id + defense_type as cache key
    cache_key = f"{session_id or 'default'}_{defense_type}"

    if force_new or cache_key not in _rag_defense_manager_cache:
        _rag_defense_manager_cache[cache_key] = RAGDefenseManager(
            defense_type=defense_type,
            trace_file=trace_file,
        )

    return _rag_defense_manager_cache[cache_key]

