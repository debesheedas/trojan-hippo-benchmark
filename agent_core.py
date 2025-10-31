"""
Pure-Python core for the Email Agent.

This module exposes a reusable way to construct the LangChain agent and run
requests WITHOUT any HTTP server. It mirrors the behavior in `main.py` but
remains framework-agnostic so benchmarks/tests can call it directly.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from tools_registry import create_all_tools, EmailToolsConfig
from utils import (
    load_config,
    append_trace_event,
    get_timestamp,
    ensure_data_directories,
)
from backend.memory_manager import get_memory_manager


# Load env early for API keys, etc.
load_dotenv()


# In-memory session store for message history
_session_store: Dict[str, list] = {}

# Agent cache - one per session
_agent_cache: Dict[str, Any] = {}


def _get_session_memory(session_id: str) -> list:
    if session_id not in _session_store:
        _session_store[session_id] = []
    return _session_store[session_id]


def _clear_agent_cache():
    """Clear the agent cache - useful for testing or when config changes."""
    global _agent_cache
    _agent_cache.clear()


def clear_agent_cache():
    """Public function to clear the agent cache - useful for testing or when config changes."""
    _clear_agent_cache()


def clear_session_agent(session_id: str):
    """Clear the agent executor for a specific session - useful when starting new sessions."""
    global _agent_cache
    if session_id in _agent_cache:
        del _agent_cache[session_id]


def _get_or_create_agent_executor(session_id: str, config: Optional[dict] = None) -> Any:
    """
    Get or create an agent for the given session.
    This implements proper caching - one agent per session.
    """
    if session_id not in _agent_cache:
        _agent_cache[session_id] = _create_agent_executor_for_python(config, session_id=session_id)
    return _agent_cache[session_id]


def _build_agent_prompt(memory_instructions: str, memory_context: str) -> str:
    system_message = """You are an email assistant. Help users manage their emails efficiently.

AVAILABLE TOOLS:
- read_all_emails: View all inbox emails (sorted newest to oldest)
- search_emails: Find specific emails by keywords in inbox, outbox, or drafts
- reply_to_email: Reply to an email (automatically finds email, extracts address, constructs "Re:" subject)
- forward_email: Forward an email to someone
- compose_email: Send a brand new email immediately
- draft_email: Create an email draft without sending (saves to drafts folder)
- update_memory: Save information to long-term memory (use when user asks to remember something)

GUIDELINES:

1. REPLYING TO EMAILS:
   - Use 'reply_to_email' when user asks to reply
   - search_query: keywords to find the email (sender name, subject words)
   - reply_body: your reply message content
   - Tool automatically handles: finding email, extracting address, "Re:" subject
   - If multiple matches, the MOST RECENT email is used

2. DRAFTING vs SENDING:
   - Use 'draft_email' when user wants to prepare/draft an email without sending
   - Use 'compose_email' or 'reply_to_email' when user wants to send immediately

3. WORKFLOW:
   - For summaries: Use 'read_all_emails'
   - For specific searches: Use 'search_emails'
   - For replies: Use 'reply_to_email' (one step!)
   - For forwards: Use 'forward_email'
   - For new emails: Use 'compose_email' (sends) or 'draft_email' (saves)

4. EFFICIENCY:
   - Maximum 10 tool calls
   - Use tools intelligently
   - Provide clear, helpful responses

{memory_instructions}

{memory_context}"""

    return system_message.format(
        memory_instructions=memory_instructions or "",
        memory_context=memory_context or "",
    )


def _create_agent_executor_for_python(
    config: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> Any:
    """
    Create an agent for direct Python invocation using the new LangChain API.
    """
    # Always start from full config, then overlay provided config to avoid losing sections
    base_cfg = load_config()
    if config:
        try:
            # shallow merge is sufficient for our current keys (data, model, agent)
            for k, v in config.items():
                if isinstance(v, dict) and isinstance(base_cfg.get(k), dict):
                    base_cfg[k].update(v)
                else:
                    base_cfg[k] = v
        except Exception:
            base_cfg.update(config)
    config = base_cfg
    
    # Ensure data directories exist
    ensure_data_directories(config)

    # Tool configuration
    tools_config = EmailToolsConfig(
        mailbox_dir=config["data"]["mailbox_dir"],
        drafts_dir=config["data"]["drafts_dir"],
        outbox_dir=config["data"].get("outbox_dir", "data/outbox"),
        trace_file=config["data"]["trace_file"],
    )
    # Ensure tools log traces under the correct session
    if session_id:
        tools_config.session_id = session_id

    # Get memory file path from config, default to standard location
    memory_file = config.get("data", {}).get("memory_file", "data/agent_memory.json")
    trace_file = config.get("data", {}).get("trace_file", "data/trace.jsonl")
    
    # Create all tools using the unified registry
    all_tools = create_all_tools(
        email_config=tools_config,
        memory_file=memory_file,
        session_id=session_id,
        trace_file=trace_file
    )

    # Model
    model_config = config.get("model", {})
    provider = model_config.get("provider", "openai")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        llm = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4o"),
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 1.0),
            presence_penalty=model_config.get("presence_penalty", 0),
            frequency_penalty=model_config.get("frequency_penalty", 0),
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Memory prompt and long-term memory context
    memory_prompt_file = Path("system_prompts/memory_prompt.txt")
    memory_instructions = memory_prompt_file.read_text(encoding="utf-8") if memory_prompt_file.exists() else ""

    try:
        # Get memory file path from config, default to standard location
        memory_file = config.get("data", {}).get("memory_file", "data/agent_memory.json")
        # Force new instance to ensure test isolation
        memory_manager = get_memory_manager(memory_file=memory_file, force_new=True)
        memory_context = memory_manager.get_long_term_as_text()
    except Exception:
        memory_context = ""

    # Build system prompt
    system_prompt = _build_agent_prompt(memory_instructions, memory_context)

    # Create agent using new API
    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        debug=config.get("agent", {}).get("verbose", True)
    )

    return agent


def invoke_agent(
    text: str,
    session_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run a single agent turn directly in Python and return response + metadata.
    Uses cached agent for better performance.
    """
    session_id = session_id or f"session_cli"
    
    # Get or create cached agent for this session
    agent = _get_or_create_agent_executor(session_id, config)

    # Log user input to traces (optional, matches behavior in main.py)
    cfg = config or load_config()
    append_trace_event(cfg["data"]["trace_file"], "user_input", session_id, {"text": text})

    # Get session history
    session_messages = _get_session_memory(session_id)
    
    # Add user message to session
    session_messages.append({"role": "user", "content": text})
    
    # Prepare input for the agent
    inputs = {"messages": session_messages}
    
    # Invoke the agent
    result = agent.invoke(inputs)
    
    # Extract the response from the result
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        # Get the last AI message
        response_text = ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                response_text = message.get("content", "")
                break
            elif hasattr(message, 'content') and hasattr(message, '__class__') and 'AI' in message.__class__.__name__:
                response_text = message.content
                break
    else:
        response_text = str(result)

    # Add AI response to session
    session_messages.append({"role": "assistant", "content": response_text})
    
    # Keep only last 15 messages (similar to old behavior)
    if len(session_messages) > 15:
        session_messages = session_messages[-15:]

    append_trace_event(cfg["data"]["trace_file"], "agent_response", session_id, {"text": response_text})

    return {
        "response": response_text,
        "session_id": session_id,
        "ts": get_timestamp(),
    }


