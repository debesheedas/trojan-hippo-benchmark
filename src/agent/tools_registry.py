"""
Unified tools registry for the memory agent.
Provides a single entry point for all agent tools (email and memory).
"""

from typing import List, Optional, Set
from langchain.tools import BaseTool

from agent.tool_specifications.email_tools import EmailToolsConfig, create_tools
from agent.tool_specifications.memory_tools import create_memory_tools

# Tool trust classification for defense mechanisms
# Untrusted tools are those that read from untrusted data sources (inbox)
UNTRUSTED_TOOLS: Set[str] = {
    "read_all_emails",
    "search_emails", 
    "reply_to_email",
    "forward_to_email"
}

# Trusted tools are those that don't read from untrusted sources
TRUSTED_TOOLS: Set[str] = {
    "compose_email",
    "draft_email",
    "update_memory"
}


def is_untrusted_tool(tool_name: str) -> bool:
    """
    Check if a tool is untrusted.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        True if the tool is untrusted, False otherwise
    """
    return tool_name in UNTRUSTED_TOOLS


def is_trusted_tool(tool_name: str) -> bool:
    """
    Check if a tool is trusted.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        True if the tool is trusted, False otherwise
    """
    return tool_name in TRUSTED_TOOLS


def create_all_tools(
    email_config: EmailToolsConfig,
    memory_file: str = "data/interactive_agent/agent_memory.json",
    session_id: Optional[str] = None,
    trace_file: str = "data/interactive_agent/trace.jsonl"
) -> List[BaseTool]:
    """
    Create and return all available tools for the agent.
    
    Args:
        email_config: EmailToolsConfig instance with paths and settings
        memory_file: Path to the memory file
        session_id: Optional session ID for tracing
        trace_file: Path to the trace file
    
    Returns:
        List of all configured tools (email + memory)
    """
    # Create email tools
    email_tools = create_tools(email_config)
    
    # Create memory tools
    memory_tools = create_memory_tools(memory_file, session_id, trace_file)
    
    # Combine all tools
    all_tools = email_tools + memory_tools
    
    return all_tools


def create_email_tools(email_config: EmailToolsConfig) -> List[BaseTool]:
    """Create only email tools."""
    return create_tools(email_config)


