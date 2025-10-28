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
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_tools import EmailToolsConfig, create_tools
from memory_tools import create_memory_tools
from utils import (
    load_config,
    append_trace_event,
    get_timestamp,
    ensure_data_directories,
)
from backend.memory_manager import get_memory_manager


# Load env early for API keys, etc.
load_dotenv()


# In-memory session store for RunnableWithMessageHistory
_session_store: Dict[str, ChatMessageHistory] = {}

# Agent executor cache - one per session
_agent_cache: Dict[str, RunnableWithMessageHistory] = {}


def _get_session_memory(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
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


def _get_or_create_agent_executor(session_id: str, config: Optional[dict] = None) -> RunnableWithMessageHistory:
    """
    Get or create an agent executor for the given session.
    This implements proper caching - one executor per session.
    """
    if session_id not in _agent_cache:
        _agent_cache[session_id] = _create_agent_executor_for_python(config, session_id=session_id)
    return _agent_cache[session_id]


def _build_agent_prompt(memory_instructions: str, memory_context: str) -> ChatPromptTemplate:
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

    system_message = system_message.format(
        memory_instructions=memory_instructions or "",
        memory_context=memory_context or "",
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def _create_agent_executor_for_python(
    config: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> RunnableWithMessageHistory:
    """
    Create a RunnableWithMessageHistory agent for direct Python invocation.
    """
    if config is None:
        config = load_config()
    
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

    email_tools = create_tools(tools_config)
    # Get memory file path from config, default to standard location
    memory_file = config.get("data", {}).get("memory_file", "data/agent_memory.json")
    trace_file = config.get("data", {}).get("trace_file", "data/trace.jsonl")
    memory_tools = create_memory_tools(memory_file=memory_file, session_id=session_id, trace_file=trace_file)
    all_tools = email_tools + memory_tools

    # Model
    model_config = config.get("model", {})
    provider = model_config.get("provider", "openai")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model=model_config.get("model_name", "gpt-5"),
            temperature=model_config.get("temperature", 0.7),
            api_key=api_key if api_key else "dummy-key",
        )
    else:
        model_name = config.get("model", {}).get("model_name", "gpt-5")
        temperature = config.get("model", {}).get("temperature", 0.7)
        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key="dummy-key")

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

    prompt = _build_agent_prompt(memory_instructions, memory_context)

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=config.get("agent", {}).get("verbose", True),
        max_iterations=config.get("agent", {}).get("max_iterations", 10),
        return_intermediate_steps=True,
    )

    chain_with_history = RunnableWithMessageHistory(
        agent_executor,
        _get_session_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_history


def invoke_agent(
    text: str,
    session_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run a single agent turn directly in Python and return response + metadata.
    Uses cached agent executor for better performance.
    """
    session_id = session_id or f"session_cli"
    
    # Get or create cached agent executor for this session
    chain = _get_or_create_agent_executor(session_id, config)

    # Log user input to traces (optional, matches behavior in main.py)
    cfg = config or load_config()
    append_trace_event(cfg["data"]["trace_file"], "user_input", session_id, {"text": text})

    result = chain.invoke({"input": text}, {"configurable": {"session_id": session_id}})
    response_text = result.get("output", "")

    append_trace_event(cfg["data"]["trace_file"], "agent_response", session_id, {"text": response_text})

    return {
        "response": response_text,
        "session_id": session_id,
        "ts": get_timestamp(),
    }


