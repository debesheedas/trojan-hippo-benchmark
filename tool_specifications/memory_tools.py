"""
Memory Tools for the Agent
Provides tools for the agent to update memory.
"""

from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field

from backend.memory_manager import get_memory_manager
from utils import append_trace_event, get_timestamp, generate_id


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
    memory_file: str = "data/agent_memory.json"
    session_id: Optional[str] = None
    trace_file: str = "data/trace.jsonl"
    
    def _run(self, memory_text: str) -> str:
        """Execute the memory update."""
        call_id = generate_id()
        
        # Log tool call
        if self.session_id:
            append_trace_event(
                self.trace_file,
                session_id=self.session_id,
                event_type="tool_call",
                payload={
                    "tool_name": "update_memory",
                    "inputs": {"memory_text": memory_text},
                    "call_id": call_id
                }
            )
        
        try:
            # Use memory manager with the correct file path
            memory_manager = get_memory_manager(memory_file=self.memory_file, force_new=True)
            
            # Handle forget requests
            if memory_text.lower().startswith("forget"):
                to_forget = memory_text[7:].strip()
                original_count = len(memory_manager.long_term)
                memory_manager.long_term = [item for item in memory_manager.long_term if to_forget.lower() not in item.lower()]
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
                        "call_id": call_id
                    }
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
                        "call_id": call_id
                    }
                )
            
            return error_msg
    
    async def _arun(self, memory_text: str) -> str:
        """Async version."""
        return self._run(memory_text)


def create_memory_tools(memory_file: str = "data/agent_memory.json", session_id: Optional[str] = None, trace_file: str = "data/trace.jsonl"):
    """Create and return memory tools."""
    tool = UpdateMemoryTool()
    tool.memory_file = memory_file
    tool.session_id = session_id
    tool.trace_file = trace_file
    return [tool]

