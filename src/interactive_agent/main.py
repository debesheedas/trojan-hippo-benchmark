"""
Email Agent FastAPI Application
Main entry point for the email agent MVP.

Uses LangChain's Tool Calling Agent (not ReAct):
- Leverages native function calling (OpenAI Functions)
- No JSON parsing errors
- More reliable than text-based ReAct
- Compatible with OpenAI, Anthropic, Gemini
"""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent.tools_registry import create_all_tools
from agent.tool_specifications.email_tools import EmailToolsConfig
from agent.utils import (
    load_config,
    ensure_data_directories,
    append_trace_event,
    read_trace_events,
    get_timestamp,
    generate_id,
    set_global_seeds,
)
from agent.backend.memory_manager import get_memory_manager
from interactive_agent.backend.routes.memory_routes import router as memory_router

# Load environment variables
load_dotenv()

# Load configuration
config = load_config()

# Set global seed for reproducibility
global_seed = config.get("seed", 42)
set_global_seeds(global_seed)

# Ensure data directories exist
ensure_data_directories(config)

# Initialize FastAPI app
app = FastAPI(
    title="Email Agent MVP",
    description="A minimal email agent with LangChain tools",
    version="1.0.0"
)

# Include memory router
app.include_router(memory_router)

# Hard-coded data paths for interactive agent
INTERACTIVE_DATA_ROOT = Path("data/interactive_agent")
INTERACTIVE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

MAILBOX_DIR = INTERACTIVE_DATA_ROOT / "mailbox"
DRAFTS_DIR = INTERACTIVE_DATA_ROOT / "drafts"
OUTBOX_DIR = INTERACTIVE_DATA_ROOT / "outbox"
TRACE_FILE = INTERACTIVE_DATA_ROOT / "trace.jsonl"
MEMORY_FILE = INTERACTIVE_DATA_ROOT / "agent_memory.json"
SESSIONS_DIR = INTERACTIVE_DATA_ROOT / "sessions"

# Global agent configuration
tools_config = EmailToolsConfig(
    mailbox_dir=str(MAILBOX_DIR),
    drafts_dir=str(DRAFTS_DIR),
    outbox_dir=str(OUTBOX_DIR),
    trace_file=str(TRACE_FILE)
)

# Session store for message history
session_store = {}

def get_session_memory(session_id: str) -> list:
    """Get or create session memory for a given session ID."""
    if session_id not in session_store:
        session_store[session_id] = []
    return session_store[session_id]

def load_sessions_from_disk():
    """Load sessions from disk on startup."""
    import json
    from pathlib import Path
    
    sessions_dir = SESSIONS_DIR
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    for session_file in sessions_dir.glob("*.json"):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                session_id = session_data["session_id"]
                
                # Recreate message history from stored messages
                messages = []
                for msg_data in session_data.get("messages", []):
                    if msg_data["type"] == "human":
                        messages.append({"role": "user", "content": msg_data["content"]})
                    elif msg_data["type"] == "ai":
                        messages.append({"role": "assistant", "content": msg_data["content"]})
                
                session_store[session_id] = messages
                print(f"Loaded session {session_id} with {len(messages)} messages")
        except Exception as e:
            print(f"Error loading session from {session_file}: {e}")

def save_session_to_disk(session_id: str):
    """Save session to disk."""
    import json
    from pathlib import Path
    
    if session_id not in session_store:
        return
    
    sessions_dir = SESSIONS_DIR
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    session_file = sessions_dir / f"{session_id}.json"
    messages = session_store[session_id]
    
    # Convert messages to serializable format
    serializable_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            msg_type = "human" if msg.get("role") == "user" else "ai"
            content = msg.get("content", "")
        else:
            # Handle old format if any
            msg_type = "human" if hasattr(msg, '__class__') and 'Human' in msg.__class__.__name__ else "ai"
            content = getattr(msg, 'content', str(msg))
        
        serializable_messages.append({
            "type": msg_type,
            "content": content
        })
    
    session_data = {
        "session_id": session_id,
        "messages": serializable_messages,
        "last_updated": get_timestamp()
    }
    
    try:
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")

# Create tools using the unified registry
memory_file = str(MEMORY_FILE)
trace_file = str(TRACE_FILE)
all_tools = create_all_tools(
    email_config=tools_config,
    memory_file=memory_file,
    session_id=None,  # Will be set per session
    trace_file=trace_file
)

# Create separate email tools for direct API endpoints
from agent.tools_registry import create_email_tools
email_tools = create_email_tools(tools_config)


# Note: Callback handling is now built into the new LangChain API


def create_agent_executor():
    """Create and configure the LangChain agent with session management."""
    
    # Initialize LLM based on config
    model_config = config.get("agent", {})
    # Auto-detect provider from model name if not explicitly set
    model_name = model_config.get("target_model_name", model_config.get("model_name", "gpt-4o"))
    provider = model_config.get("provider")
    
    # Auto-detect provider from model name
    from agent.utils import detect_provider
    if provider is None:
        provider = detect_provider(model_name)
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Use a mock response if no API key
            print("Warning: OPENAI_API_KEY not set. Agent will have limited functionality.")
        
        # Build LLM with error handling for unsupported parameters
        temperature = model_config.get("temperature", 0.7)
        
        try:
            # Try with temperature parameter
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key if api_key else "dummy-key"
            )
        except (TypeError, ValueError) as e:
            # If model doesn't support temperature, use minimal config
            print(f"Warning: Model {model_name} may not support temperature parameter. Using minimal configuration. Error: {e}")
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key if api_key else "dummy-key"
            )
    elif provider == "gemini":
        # Note: langchain-google-genai has version conflicts with langchain 1.0+
        # For now, we'll use a workaround or fallback
        raise NotImplementedError(
            "Gemini models are not yet supported with LangChain integration due to version conflicts. "
            "Direct API calls via google-generativeai work (used in adaptive attacks). "
            "For LangChain agent, please use OpenAI models or install langchain-google-genai manually "
            "with compatible versions."
        )
    else:
        # Mock LLM for testing (fallback to OpenAI mock)
        try:
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key="dummy-key"
            )
        except (TypeError, ValueError):
            llm = ChatOpenAI(
                model="gpt-4o",
                api_key="dummy-key"
            )
    
    # Load memory prompt
    memory_prompt_file = BASE_DIR / "src" / "agent" / "memory_prompt.txt"
    memory_instructions = ""
    if memory_prompt_file.exists():
        with open(memory_prompt_file, 'r', encoding='utf-8') as f:
            memory_instructions = f.read()
    
    # Get long-term memory context
    try:
        memory_manager = get_memory_manager()
        memory_context = memory_manager.get_long_term_as_text()
        print(f"Loaded memory context: {len(memory_context)} characters")
        if memory_context:
            print(f"Memory context preview: {memory_context[:200]}...")
    except Exception as e:
        print(f"Warning: Could not load memory context: {e}")
        memory_context = ""
    
    # Create system prompt
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
    
    # Format with memory content
    system_message = system_message.format(
        memory_instructions=memory_instructions,
        memory_context=memory_context
    )
    
    # Create agent using new API
    # Note: debug mode can be enabled for development, but verbose output is noisy
    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_message,
        debug=config.get("agent", {}).get("verbose", True)  # Keep verbose for main.py (web interface)
    )
    
    return agent


# Pydantic models for API
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: str


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ReadEmailRequest(BaseModel):
    email_id: str


class SearchInboxRequest(BaseModel):
    query: str


class DraftEmailRequest(BaseModel):
    to: str
    subject: str
    body: str


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main UI."""
    # Static files are now in interactive_agent/static relative to project root
    index_file = BASE_DIR / "src" / "interactive_agent" / "static" / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            content="<h1>UI not found</h1><p>Please ensure static/index.html exists.</p>",
            status_code=404
        )
    
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Processes user input through the agent with session management.
    """
    try:
        # Resolve or create session id
        session_id = request.session_id or f"session_{generate_id()[:8]}"

        # Ensure session memory exists
        _ = get_session_memory(session_id)

        # Log user input
        append_trace_event(
            str(TRACE_FILE),
            "user_input",
            session_id,
            {"text": request.text}
        )
        
        # Set session ID in tools config
        tools_config.session_id = session_id
        
        # Create agent
        agent = create_agent_executor()
        
        # Get session history
        session_messages = get_session_memory(session_id)
        
        # Add user message to session
        session_messages.append({"role": "user", "content": request.text})
        
        # Prepare input for the agent
        inputs = {"messages": session_messages}
        
        # Run agent
        try:
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
                
        except Exception as e:
            error_msg = str(e)
            response_text = f"I encountered an error: {error_msg}"
            print(f"Agent Error: {error_msg}")
        
        # Log agent response
        append_trace_event(
            str(TRACE_FILE),
            "agent_response",
            session_id,
            {"text": response_text}
        )
        
        # Save session to disk after each interaction
        save_session_to_disk(session_id)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Memory updates are now handled by the update_memory tool

# Session Management API Endpoints
@app.get("/sessions")
async def get_sessions():
    """Get all sessions."""
    try:
        sessions = []
        for session_id, messages in session_store.items():
            sessions.append({
                "session_id": session_id,
                "title": f"Session {session_id}",
                "created_at": "2025-01-08T00:00:00Z",
                "last_updated": "2025-01-08T00:00:00Z",
                "message_count": len(messages)
            })
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def create_session(request: dict = None):
    """Create a new session."""
    try:
        session_id = f"session_{len(session_store) + 1:03d}"
        session_store[session_id] = []
        
        # Save session to disk
        save_session_to_disk(session_id)
        
        return {
            "session_id": session_id,
            "title": "New Chat",
            "created_at": "2025-01-08T00:00:00Z",
            "last_updated": "2025-01-08T00:00:00Z",
            "message_count": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    try:
        if session_id not in session_store:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = session_store[session_id]
        conversation_history = []
        
        for message in messages:
            if isinstance(message, dict):
                conversation_history.append({
                    "message": message.get("content", ""),
                    "is_user": message.get("role") == "user",
                    "timestamp": "2025-01-08T00:00:00Z"
                })
            else:
                # Handle old format if any
                conversation_history.append({
                    "message": getattr(message, 'content', str(message)),
                    "is_user": hasattr(message, '__class__') and 'Human' in message.__class__.__name__,
                    "timestamp": "2025-01-08T00:00:00Z"
                })
        
        return {
            "session_id": session_id,
            "title": f"Session {session_id}",
            "created_at": "2025-01-08T00:00:00Z",
            "last_updated": "2025-01-08T00:00:00Z",
            "message_count": len(messages),
            "short_term_memory": [msg.get("content", "") if isinstance(msg, dict) else getattr(msg, 'content', str(msg)) for msg in messages[-15:]],
            "conversation_history": conversation_history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/traces/{session_id}")
async def get_session_traces(session_id: str):
    """Get trace events for a specific session."""
    try:
        trace_events = read_trace_events(str(TRACE_FILE), session_id)
        return trace_events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/tool/read")
async def read_email_endpoint(request: ReadEmailRequest):
    """Direct endpoint for reading an email."""
    try:
        read_tool = email_tools[0]  # ReadEmailTool
        result = read_tool.run(request.email_id)  # Use public run method
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/tool/search")
async def search_inbox_endpoint(request: SearchInboxRequest):
    """Direct endpoint for searching emails."""
    try:
        search_tool = email_tools[1]  # SearchInboxTool
        result = search_tool.run(request.query)  # Use public run method
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/tool/draft")
async def draft_email_endpoint(request: DraftEmailRequest):
    """Direct endpoint for creating a draft."""
    try:
        draft_tool = email_tools[2]  # DraftEmailTool
        result = draft_tool.run({"to": request.to, "subject": request.subject, "body": request.body})  # Use public run method
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/emails")
async def get_all_emails():
    """
    Get all emails from the mailbox.
    """
    try:
        import json
        mailbox_dir = MAILBOX_DIR
        emails = []
        
        for email_file in sorted(mailbox_dir.glob("*.json")):
            with open(email_file, "r", encoding="utf-8") as f:
                email_data = json.load(f)
                emails.append(email_data)
        
        # Sort by received timestamp (newest first)
        emails.sort(key=lambda x: x.get("received_ts", ""), reverse=True)
        
        return {"emails": emails}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/session/{session_id}/trace")
async def get_session_trace(session_id: str):
    """Get trace events for a specific session."""
    try:
        events = read_trace_events(str(TRACE_FILE), session_id)
        return {"session_id": session_id, "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "config": {
            "model": config.get("agent", {}).get("target_model_name"),
            "mailbox_dir": str(MAILBOX_DIR),
            "drafts_dir": str(DRAFTS_DIR)
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load sessions from disk on startup
    load_sessions_from_disk()
    
    # Hard-coded server configuration for interactive agent
    uvicorn.run(
        "interactive_agent.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

