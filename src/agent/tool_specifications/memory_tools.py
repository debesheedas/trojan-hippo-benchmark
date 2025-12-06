"""
Memory Tools for the Agent
Provides tools for the agent to update memory.
"""

from langchain.tools import BaseTool
from typing import Optional, Set
from pydantic import BaseModel, Field

from agent.backend.memory_manager import get_memory_manager
from agent.utils import append_trace_event, generate_id, read_trace_events


# Untrusted tools are those that read from untrusted data sources (inbox)
# We duplicate this set here (also defined in mem0_defense_manager and tools_registry)
# to avoid circular imports.
UNTRUSTED_TOOLS: Set[str] = {
    "read_all_emails",
    "search_emails",
    "reply_to_email",
    "forward_to_email",
}


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

    def _has_untrusted_tool_been_called(self) -> bool:
        """
        Check if any untrusted tool has been called in this session.
        Mirrors the behavior of the mem0 'no_untrusted_tools' defense.
        """
        if not self.session_id or not self.trace_file:
            return False

        try:
            trace_events = read_trace_events(self.trace_file, self.session_id)
            for event in trace_events:
                if event.get("event_type") == "tool_call":
                    payload = event.get("payload", {})
                    tool_name = payload.get("tool_name", "")
                    if tool_name in UNTRUSTED_TOOLS:
                        return True
        except Exception as e:
            print(f"Warning: Could not check trace events for explicit memory defense: {e}")

        return False

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
        # Defense: limit_memory_length – truncate memory_text to 80 characters
        if self.explicit_defense_type == "limit_memory_length" and isinstance(memory_text, str):
            if len(memory_text) > 80:
                memory_text = memory_text[:80]

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

        # Defense: no_untrusted_tools – once any untrusted tool is used in this
        # session, disable explicit memory indexing for the rest of the session.
        if self.explicit_defense_type == "no_untrusted_tools":
            if self._has_untrusted_tool_been_called():
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
            memory_manager = get_memory_manager(memory_file=self.memory_file, force_new=True)

            # Handle forget requests
            if memory_text.lower().startswith("forget"):
                to_forget = memory_text[7:].strip()
                original_count = len(memory_manager.long_term)
                memory_manager.long_term = [
                    item for item in memory_manager.long_term if to_forget.lower() not in item.lower()
                ]
                removed_count = original_count - len(memory_manager.long_term)

                # Save updated memory
                memory_manager._save_long_term()

                result = f"✓ Memory updated: Removed {removed_count} fact(s) matching '{to_forget}'"
            else:
                # Add new memory (avoid duplicates)
                if memory_text not in memory_manager.long_term:
                    memory_manager.long_term.append(memory_text)

                    # Save updated memory
                    memory_manager._save_long_term()

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
):
    """Create and return memory tools."""
    tool = UpdateMemoryTool()
    tool.memory_file = memory_file
    tool.session_id = session_id
    tool.trace_file = trace_file
    tool.explicit_defense_type = explicit_defense_type
    return [tool]

