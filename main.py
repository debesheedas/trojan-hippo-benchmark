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
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

from agent_tools import EmailToolsConfig, create_tools
from memory_tools import create_memory_tools
from utils import (
    load_config,
    ensure_data_directories,
    append_trace_event,
    read_trace_events,
    get_timestamp,
    generate_id,
)
from backend.memory_manager import get_memory_manager
from backend.routes.memory_routes import router as memory_router
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Load configuration
config = load_config()

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

# Global agent configuration
tools_config = EmailToolsConfig(
    mailbox_dir=config["data"]["mailbox_dir"],
    drafts_dir=config["data"]["drafts_dir"],
    outbox_dir=config["data"].get("outbox_dir", "data/outbox"),
    trace_file=config["data"]["trace_file"]
)

# Session store for LangChain's RunnableWithMessageHistory
session_store = {}

def get_session_memory(session_id: str) -> ChatMessageHistory:
    """Get or create session memory for a given session ID."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def load_sessions_from_disk():
    """Load sessions from disk on startup."""
    import json
    from pathlib import Path
    
    sessions_dir = Path("data/sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    for session_file in sessions_dir.glob("*.json"):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                session_id = session_data["session_id"]
                
                # Recreate ChatMessageHistory from stored messages
                chat_history = ChatMessageHistory()
                for msg_data in session_data.get("messages", []):
                    if msg_data["type"] == "human":
                        from langchain_core.messages import HumanMessage
                        chat_history.add_message(HumanMessage(content=msg_data["content"]))
                    elif msg_data["type"] == "ai":
                        from langchain_core.messages import AIMessage
                        chat_history.add_message(AIMessage(content=msg_data["content"]))
                
                session_store[session_id] = chat_history
                print(f"Loaded session {session_id} with {len(chat_history.messages)} messages")
        except Exception as e:
            print(f"Error loading session from {session_file}: {e}")

def save_session_to_disk(session_id: str):
    """Save session to disk."""
    import json
    from pathlib import Path
    
    if session_id not in session_store:
        return
    
    sessions_dir = Path("data/sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    session_file = sessions_dir / f"{session_id}.json"
    chat_history = session_store[session_id]
    
    # Convert messages to serializable format
    messages = []
    for msg in chat_history.messages:
        messages.append({
            "type": "human" if msg.__class__.__name__ == "HumanMessage" else "ai",
            "content": msg.content
        })
    
    session_data = {
        "session_id": session_id,
        "messages": messages,
        "last_updated": get_timestamp()
    }
    
    try:
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")

# Create tools
email_tools = create_tools(tools_config)
# Get memory file path from config, default to standard location
memory_file = config.get("data", {}).get("memory_file", "data/agent_memory.json")
memory_tools = create_memory_tools(memory_file=memory_file)
all_tools = email_tools + memory_tools


class ErrorHandlingCallback(BaseCallbackHandler):
    """Callback to handle and log tool errors gracefully."""
    
    def on_tool_start(self, serialized, input_str: str, **kwargs) -> None:
        """Log when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")
        print(f"\n{'='*60}")
        print(f"TOOL START: {tool_name}")
        print(f"RAW INPUT: {input_str}")
        print(f"{'='*60}\n")
        
        # Log to trace file as well
        if hasattr(tools_config, 'session_id') and tools_config.session_id:
            append_trace_event(
                config["data"]["trace_file"],
                "debug_tool_start",
                tools_config.session_id,
                {
                    "tool_name": tool_name,
                    "raw_input": input_str,
                    "serialized": str(serialized)
                }
            )
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Handle tool execution errors."""
        error_msg = str(error)
        print(f"\n{'!'*60}")
        print(f"TOOL ERROR: {error_msg}")
        print(f"{'!'*60}\n")
        
        # Log to trace file
        if hasattr(tools_config, 'session_id') and tools_config.session_id:
            append_trace_event(
                config["data"]["trace_file"],
                "debug_tool_error",
                tools_config.session_id,
                {"error": error_msg}
            )
    
    def on_agent_action(self, action, **kwargs) -> None:
        """Log agent actions for debugging."""
        print(f"\n{'~'*60}")
        print(f"AGENT ACTION:")
        print(f"  Tool: {action.tool}")
        print(f"  Tool Input (type={type(action.tool_input).__name__}): {action.tool_input}")
        print(f"  Log: {action.log}")
        print(f"{'~'*60}\n")
        
        # Log to trace file
        if hasattr(tools_config, 'session_id') and tools_config.session_id:
            append_trace_event(
                config["data"]["trace_file"],
                "debug_agent_action",
                tools_config.session_id,
                {
                    "tool": action.tool,
                    "tool_input": str(action.tool_input),
                    "tool_input_type": type(action.tool_input).__name__,
                    "log": action.log
                }
            )


def create_agent_executor() -> RunnableWithMessageHistory:
    """Create and configure the LangChain agent with session management."""
    
    # Initialize LLM based on config
    model_config = config.get("model", {})
    provider = model_config.get("provider", "openai")
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Use a mock response if no API key
            print("Warning: OPENAI_API_KEY not set. Agent will have limited functionality.")
        
        llm = ChatOpenAI(
            model=model_config.get("model_name", "gpt-4"),
            temperature=model_config.get("temperature", 0.7),
            api_key=api_key if api_key else "dummy-key"
        )
    else:
        # Mock LLM for testing
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key="dummy-key"
        )
    
    # Load memory prompt
    memory_prompt_file = Path("system_prompts/memory_prompt.txt")
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
    
    # Create agent prompt for tool calling (native function calling)
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create tool calling agent (uses native function calling - no parsing errors!)
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=config.get("agent", {}).get("verbose", True),
        max_iterations=config.get("agent", {}).get("max_iterations", 10),
        return_intermediate_steps=True,  # For debugging
        callbacks=[ErrorHandlingCallback()]  # Debug logging
    )
    
    # Wrap with session management
    chain_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return chain_with_history


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
    index_file = Path("static/index.html")
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
            config["data"]["trace_file"],
            "user_input",
            session_id,
            {"text": request.text}
        )
        
        # Set session ID in tools config
        tools_config.session_id = session_id
        
        # Create agent executor with session management
        agent_executor = create_agent_executor()
        
        # Run agent with session management
        try:
            result = agent_executor.invoke(
                {"input": request.text},
                {"configurable": {"session_id": session_id}}
            )
            response_text = result.get("output", "I encountered an error processing your request.")
            
            # Check if max iterations was reached
            if "Agent stopped due to iteration limit" in response_text or \
               "Agent stopped due to max iterations" in response_text:
                response_text += "\n\n(Note: I've reached the maximum number of tool calls allowed. Please provide more specific instructions or break this into smaller tasks.)"
                
        except Exception as e:
            error_msg = str(e)
            response_text = f"I encountered an error: {error_msg}"
            print(f"Agent Error: {error_msg}")
        
        # Log agent response
        append_trace_event(
            config["data"]["trace_file"],
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
        for session_id, chat_history in session_store.items():
            sessions.append({
                "session_id": session_id,
                "title": f"Session {session_id}",
                "created_at": "2025-01-08T00:00:00Z",
                "last_updated": "2025-01-08T00:00:00Z",
                "message_count": len(chat_history.messages)
            })
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def create_session(request: dict = None):
    """Create a new session."""
    try:
        session_id = f"session_{len(session_store) + 1:03d}"
        session_store[session_id] = ChatMessageHistory()
        
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
        
        chat_history = session_store[session_id]
        conversation_history = []
        
        for message in chat_history.messages:
            conversation_history.append({
                "message": message.content,
                "is_user": message.__class__.__name__ == "HumanMessage",
                "timestamp": "2025-01-08T00:00:00Z"
            })
        
        return {
            "session_id": session_id,
            "title": f"Session {session_id}",
            "created_at": "2025-01-08T00:00:00Z",
            "last_updated": "2025-01-08T00:00:00Z",
            "message_count": len(chat_history.messages),
            "short_term_memory": [msg.content for msg in chat_history.messages[-15:]],
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
        trace_events = read_trace_events(config["data"]["trace_file"], session_id)
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
        mailbox_dir = Path(config["data"]["mailbox_dir"])
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
        events = read_trace_events(config["data"]["trace_file"], session_id)
        return {"session_id": session_id, "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "config": {
            "model": config.get("model", {}).get("model_name"),
            "mailbox_dir": config["data"]["mailbox_dir"],
            "drafts_dir": config["data"]["drafts_dir"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load sessions from disk on startup
    load_sessions_from_disk()
    
    server_config = config.get("server", {})
    uvicorn.run(
        "main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=True
    )

