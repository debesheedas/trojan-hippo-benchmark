"""
Memory Tools for the Agent
Provides tools for the agent to update memory.
"""

from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field

from backend.memory_manager import get_memory_manager


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
    
    def _run(self, memory_text: str) -> str:
        """Execute the memory update."""
        try:
            memory_manager = get_memory_manager()
            memory_manager.add_long_term(memory_text)
            
            if memory_text.lower().startswith("forget"):
                return f"✓ Memory updated: Removed facts matching '{memory_text[7:].strip()}'"
            else:
                return f"✓ Memory updated: Saved '{memory_text}' to long-term memory"
        except Exception as e:
            return f"Error updating memory: {str(e)}"
    
    async def _arun(self, memory_text: str) -> str:
        """Async version."""
        return self._run(memory_text)


def create_memory_tools():
    """Create and return memory tools."""
    return [UpdateMemoryTool()]

