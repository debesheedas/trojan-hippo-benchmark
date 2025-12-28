"""
Context Defense Manager

Manages defense mechanisms for context memory indexing:
1. disable_memory: Baseline defense that disables all memory indexing
2. none: No defense (normal context execution - all messages indexed)
3. user_prompt_only: Only index user messages (filter out assistant messages)
4. no_untrusted_tools: Disable context indexing for the rest of the session once an
   untrusted tool has been used
"""

from typing import Dict, Optional


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

