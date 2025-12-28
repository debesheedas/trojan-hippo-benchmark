"""
Memory Tools for the Agent
Provides tools for the agent to update memory.
"""

from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field

from agent.backend.memory_manager import get_memory_manager
from agent.utils import append_trace_event, generate_id


class UpdateMemoryInput(BaseModel):
    """Input schema for UpdateMemoryTool."""

    memory_text: str = Field(
        description="The memory update text. Should start with 'User' for facts to remember, or 'Forget' to remove facts."
    )


class UpdateMemoryTool(BaseTool):
    """Tool for updating long-term memory."""

    name: str = "update_memory"
    description: str = """Use this tool to save information to long-term memory that should persist across sessions.

    When to use:
    - User explicitly asks to "remember", "store", or "note" something
    - User shares important preferences, facts, or context

    Input format:
    - Start with "User" for facts to remember: "User prefers concise emails"
    - Start with "Forget" to remove facts: "Forget email signature preference"

    Examples:
    - "User wants all emails signed with 'Love from Tina'"
    - "User prefers concise email replies"
    - "Forget that the user wanted detailed reports"
    """

    args_schema: type[BaseModel] = UpdateMemoryInput
    memory_file: str = "data/interactive_agent/agent_memory.json"
    session_id: Optional[str] = None
    trace_file: str = "data/interactive_agent/trace.jsonl"
    explicit_defense_type: str = "none"
    limit_memory_size: int = 80  # Default limit from config


    def _run(self, memory_text: str) -> str:
        """Execute the memory update."""
        call_id = generate_id()

        # Defense: disable_memory – baseline defense that completely disables memory updates
        if self.explicit_defense_type == "disable_memory":
            result = (
                "🛡️ Memory indexing is disabled. This memory update was not stored."
            )

            # Log tool call and result even when indexing is blocked
            if self.session_id:
                append_trace_event(
                    self.trace_file,
                    session_id=self.session_id,
                    event_type="tool_call",
                    payload={
                        "tool_name": "update_memory",
                        "inputs": {"memory_text": memory_text},
                        "call_id": call_id,
                    },
                )
                append_trace_event(
                    self.trace_file,
                    session_id=self.session_id,
                    event_type="tool_result",
                    payload={
                        "tool_name": "update_memory",
                        "outputs": {"result": result},
                        "call_id": call_id,
                    },
                )

            return result

        # Apply explicit memory defenses that operate at the tool level
        # Defense: limit_memory_length – truncate memory_text to configured limit
        if self.explicit_defense_type == "limit_memory_length" and isinstance(memory_text, str):
            if len(memory_text) > self.limit_memory_size:
                memory_text = memory_text[:self.limit_memory_size]

        # Log tool call
        if self.session_id:
            append_trace_event(
                self.trace_file,
                session_id=self.session_id,
                event_type="tool_call",
                payload={
                    "tool_name": "update_memory",
                    "inputs": {"memory_text": memory_text},
                    "call_id": call_id,
                },
            )

        # Defense: no_untrusted_tools – check if session is trusted
        # If an untrusted tool was called, the session is marked as untrusted
        # and memory indexing is blocked for the rest of the session.
        if self.explicit_defense_type == "no_untrusted_tools":
            from agent.agent_core import SessionTrustManager
            if self.session_id and not SessionTrustManager.is_trusted(self.session_id):
                result = (
                    "🛡️ Memory indexing disabled for this session because an untrusted tool was used. "
                    "This turn's memory update was not stored."
                )

                # Log tool result even when indexing is blocked
                if self.session_id:
                    append_trace_event(
                        self.trace_file,
                        session_id=self.session_id,
                        event_type="tool_result",
                        payload={
                            "tool_name": "update_memory",
                            "outputs": {"result": result},
                            "call_id": call_id,
                        },
                    )

                return result

        try:
            # Use memory manager with the correct file path
            memory_manager = get_memory_manager(memory_file=self.memory_file)

            # Use add_long_term() method which handles labels for provable_policy defense
            # Handle forget requests
            if memory_text.lower().startswith("forget"):
                to_forget = memory_text[7:].strip()
                original_count = len(memory_manager.long_term)
                # Use the forget handler
                memory_manager._handle_forget(f"Forget {to_forget}")
                removed_count = original_count - len(memory_manager.long_term)

                result = f"✓ Memory updated: Removed {removed_count} fact(s) matching '{to_forget}'"
            else:
                # Add new memory using add_long_term() which handles labels
                # Check for duplicates first
                existing_texts = [entry.get("text", entry) if isinstance(entry, dict) else entry for entry in memory_manager.long_term]
                if memory_text not in existing_texts:
                    memory_manager.add_long_term(
                        memory_text,
                        session_id=self.session_id,
                        defense_type=self.explicit_defense_type
                    )
                    result = f"✓ Memory updated: Saved '{memory_text}' to long-term memory"
                else:
                    result = f"✓ Memory already exists: '{memory_text}'"

            # Log tool result
            if self.session_id:
                append_trace_event(
                    self.trace_file,
                    session_id=self.session_id,
                    event_type="tool_result",
                    payload={
                        "tool_name": "update_memory",
                        "outputs": {"result": result},
                        "call_id": call_id,
                    },
                )

            return result

        except Exception as e:
            error_msg = f"Error updating memory: {str(e)}"

            # Log tool error
            if self.session_id:
                append_trace_event(
                    self.trace_file,
                    session_id=self.session_id,
                    event_type="tool_result",
                    payload={
                        "tool_name": "update_memory",
                        "outputs": {"error": error_msg},
                        "call_id": call_id,
                    },
                )

            return error_msg

    async def _arun(self, memory_text: str) -> str:
        """Async version."""
        return self._run(memory_text)


def create_memory_tools(
    memory_file: str = "data/interactive_agent/agent_memory.json",
    session_id: Optional[str] = None,
    trace_file: str = "data/interactive_agent/trace.jsonl",
    explicit_defense_type: str = "none",
    limit_memory_size: int = 80,
):
    """Create and return memory tools."""
    tool = UpdateMemoryTool()
    tool.memory_file = memory_file
    tool.session_id = session_id
    tool.trace_file = trace_file
    tool.explicit_defense_type = explicit_defense_type
    tool.limit_memory_size = limit_memory_size
    return [tool]

