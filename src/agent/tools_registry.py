"""
Unified tools registry for the memory agent.
Provides a single entry point for all agent tools (email and memory).
"""

from typing import Any, List, Optional, Set
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
    "send_email",
    "draft_email",
    "update_memory"
}

# Provable Policy Defense: Taint Axis (Source)
# Tools that introduce untrusted data (read from adversary-controlled inbox)
TAINT_TOOLS: Set[str] = {
    "read_all_emails",
    "search_emails",
}

# Provable Policy Defense: Leakage Axis (Sink)
# Tools that can exfiltrate data outside the system
EXFILTRATION_TOOLS: Set[str] = {
    "send_email",
    "reply_to_email",
    "forward_to_email"
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


def is_taint_tool(tool_name: str) -> bool:
    """
    Check if a tool is a taint source (introduces untrusted data).
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        True if the tool is a taint source, False otherwise
    """
    return tool_name in TAINT_TOOLS


def is_exfiltration_tool(tool_name: str) -> bool:
    """
    Check if a tool is an exfiltration sink (can leak data).
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        True if the tool is an exfiltration sink, False otherwise
    """
    return tool_name in EXFILTRATION_TOOLS


def is_both_taint_and_exfil(tool_name: str) -> bool:
    """
    Check if a tool is both a taint source and an exfiltration sink.
    Such tools should always be blocked when provable_policy defense is enabled.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        True if the tool is both taint and exfil, False otherwise
    """
    return tool_name in TAINT_TOOLS and tool_name in EXFILTRATION_TOOLS


def create_all_tools(
    email_config: EmailToolsConfig,
    session_id: Optional[str] = None,
    explicit_defense_type: str = "none",
    limit_memory_size: int = 80,
    explicit_memory_manager: Optional[Any] = None,
) -> List[BaseTool]:
    """
    Create and return all available tools for the agent (in-memory only).

    Args:
        email_config: EmailToolsConfig instance with in-memory mailbox and trace_store
        session_id: Optional session ID for tracing
        explicit_defense_type: Defense type for explicit memory
        limit_memory_size: Maximum characters for limit_memory_length defense
        explicit_memory_manager: Optional per-run MemoryManager (benchmark); if None, tools use global

    Returns:
        List of all configured tools (email + memory)
    """
    # Create email tools
    email_tools = create_tools(email_config)

    # Create memory tools (pass trace_store from email_config for consistency)
    memory_tools = create_memory_tools(
        session_id=session_id,
        trace_store=email_config.trace_store,
        explicit_defense_type=explicit_defense_type,
        limit_memory_size=limit_memory_size,
        explicit_memory_manager=explicit_memory_manager,
    )
    
    # Combine all tools
    all_tools = email_tools + memory_tools
    
    return all_tools


def create_email_tools(email_config: EmailToolsConfig) -> List[BaseTool]:
    """Create only email tools."""
    return create_tools(email_config)


