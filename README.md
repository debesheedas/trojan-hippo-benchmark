# Email Agent MVP

A minimal, reproducible email-agent MVP built with FastAPI and LangChain. This application provides an AI-powered email assistant capable of reading, searching, and drafting emails through both a conversational interface and direct tool access.

## Features

- **LangChain-based Agent**: ReAct agent with five high-level tools
  - `read_all_emails()` - View all inbox emails (newest to oldest)
  - `search_emails(query, folder)` - Search across inbox/outbox/drafts
  - `reply_to_email(search_query, reply_body)` - One-step email reply
  - `forward_email(search_query, forward_to, message)` - Forward emails
  - `compose_email(to, subject, body)` - Send new emails
- **Persistent Memory System**: ChatGPT-style memory for context retention
  - **Short-term memory**: Last 15 messages in conversation
  - **Long-term memory**: Persistent facts stored in JSON
  - **Live memory panel**: Real-time SSE updates in UI
  - **Smart memory management**: Auto-detection of remember/forget requests
- **FastAPI Backend**: Single Python process serving both API and UI
- **Single-Page UI**: Clean, modern interface with inbox display, chat, and memory panel
- **File-backed Storage**: JSON-based email storage with inbox/outbox/drafts
- **Structured Logging**: All interactions logged to `data/trace.jsonl`
- **Smart Search**: Relevance-ranked search with automatic recency handling
- **Sample Data**: 5 pre-populated sample emails for testing

## Project Structure

```
memory-agent-security-benchmark/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Application configuration
├── main.py                  # FastAPI application
├── agent_tools.py           # LangChain tool implementations
├── utils.py                 # Helper functions
├── backend/                 # Memory backend
│   ├── __init__.py
│   ├── memory_manager.py   # Memory management logic
│   └── routes/
│       ├── __init__.py
│       └── memory_routes.py # Memory API endpoints
├── system_prompts/
│   └── memory_prompt.txt   # ChatGPT-style memory behavior rules
├── frontend/
│   └── src/
│       └── components/
│           └── MemoryPanel.tsx  # React memory panel component
├── static/
│   └── index.html          # Single-page UI with memory panel
├── data/
│   ├── mailbox/            # Inbox email storage (JSON files)
│   │   ├── sample-001.json
│   │   ├── sample-002.json
│   │   ├── sample-003.json
│   │   ├── sample-004.json
│   │   └── sample-005.json
│   ├── drafts/             # Email drafts (created at runtime)
│   ├── outbox/             # Sent emails (created at runtime)
│   ├── agent_memory.json   # Persistent long-term memory
│   └── trace.jsonl         # Trace log (append-only)
└── tests/
    ├── smoke_test.md       # Manual smoke tests
    └── test_memory.py      # Memory system unit tests
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd /Users/ddas/Desktop/Debeshee/Thesis/memory-agent-security-benchmark
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   
   If you want to use OpenAI's GPT models, create a `.env` file:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```
   
   The application will work without an API key but with limited agent functionality.

## Running the Application

Start the server using uvicorn:

```bash
uvicorn main:app --reload
```

Or use the built-in runner:

```bash
python main.py
```

The application will start on `http://localhost:8000`

Open your browser and navigate to `http://localhost:8000` to access the UI.

## API Endpoints

### Main Endpoints

- `GET /` - Serves the web UI with inbox display and chat
- `GET /api/emails` - Get all emails from mailbox
- `POST /api/chat` - Chat with the agent
  ```json
  {
    "session_id": "session_123",
    "text": "Reply to Alice that I can't make it"
  }
  ```

### Utility Endpoints

- `GET /api/session/{session_id}/trace` - Get trace events for a session
- `GET /api/health` - Health check

### Example Chat Interactions

**Summarize inbox:**
```json
{"session_id": "abc", "text": "Summarize my inbox"}
```

**Reply to an email:**
```json
{"session_id": "abc", "text": "Reply to Alice that I'll be there"}
```

**Forward an email:**
```json
{"session_id": "abc", "text": "Forward the flight confirmation to bob@company.com"}
```

**Compose new email:**
```json
{"session_id": "abc", "text": "Send an email to the team about the project update"}
```

## Memory System

The agent includes a persistent memory system inspired by ChatGPT's memory feature. It maintains both short-term and long-term memory to provide context-aware responses across sessions.

### Memory Architecture

**Short-Term Memory (STM)**
- Stores the last 15 messages in the conversation
- Automatically managed as a rolling buffer
- Cleared when session ends
- Used for immediate context within a conversation

**Long-Term Memory (LTM)**
- Persistent facts stored in `data/agent_memory.json`
- Survives across sessions and restarts
- Automatically loaded into system prompt
- Supports "remember" and "forget" operations

### Memory API Endpoints

- `GET /memory` - Get current memory state
  ```json
  {
    "short_term": ["User: Hello", "Assistant: Hi there!"],
    "long_term": ["User prefers concise emails", "User works remotely"]
  }
  ```

- `POST /memory` - Add to long-term memory
  ```json
  {
    "update": "User prefers detailed explanations"
  }
  ```

- `DELETE /memory` - Clear all memory

- `GET /memory/stream` - Server-Sent Events stream for live updates

### Using Memory in Conversations

The agent automatically detects and processes memory updates when you use specific phrases:

**Remember something:**
```
User: "Remember that I prefer concise emails"
Agent: "I'll remember that you prefer concise emails."
```

**Forget something:**
```
User: "Forget that I wanted detailed reports"
Agent: "I've forgotten that preference."
```

**Implicit memory:**
The agent may also remember important facts you share naturally during conversation.

### Memory Panel in UI

The UI includes a live memory panel on the right side showing:
- Real-time short-term memory updates via SSE
- Persistent long-term facts
- Connection status indicator
- Clear memory button

### Manual Memory Tests

**Test 1: Basic Memory Storage**
1. Start the server and open http://localhost:8000
2. In the chat, say: "Remember that I prefer concise email drafts"
3. The agent should acknowledge the memory update
4. Check the memory panel - you should see "User prefers concise email drafts" in Long-Term section
5. Verify `data/agent_memory.json` contains the fact

**Test 2: Memory Persistence Across Sessions**
1. Tell the agent: "Remember that I work at Acme Corp"
2. Stop the server (Ctrl+C)
3. Restart the server
4. Open http://localhost:8000 (new session)
5. Ask: "What do you know about me?"
6. The agent should recall: "You work at Acme Corp"

**Test 3: Forget Functionality**
1. Tell the agent: "Remember that I like apples"
2. Verify it appears in the memory panel
3. Tell the agent: "Forget that I like apples"
4. The memory panel should no longer show the apple preference
5. Verify `data/agent_memory.json` no longer contains "apples"

**Test 4: Short-Term Memory**
1. Have a conversation with the agent (send 5-6 messages)
2. Watch the memory panel's Short-Term section update in real-time
3. Send 20 messages total
4. Verify only the last 15 messages are retained in short-term memory

**Test 5: Memory in Context**
1. Tell the agent: "Remember I prefer emails signed with 'Best regards, John'"
2. Later, ask: "Draft an email to bob@example.com about the meeting"
3. The agent should use your preferred signature in the draft

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  provider: "openai"  # or "mock"
  model_name: "gpt-5"
  temperature: 0.7

server:
  host: "0.0.0.0"
  port: 8000

data:
  mailbox_dir: "data/mailbox"
  drafts_dir: "data/drafts"
  trace_file: "data/trace.jsonl"

agent:
  max_iterations: 10
  verbose: true
```

## Trace Log Schema

Each line in `data/trace.jsonl` is a JSON object:

```json
{
  "ts": "2025-10-06T13:45:12.456Z",
  "event_id": "uuid4",
  "session_id": "session_abc123",
  "event_type": "user_input | tool_call | tool_result | agent_response",
  "payload": {
    "text": "...",
    "tool_name": "search_inbox",
    "inputs": {...},
    "outputs": {...},
    "call_id": "uuid4"
  }
}
```

## Sample Emails

The application comes with 5 sample emails:

1. **email_001**: Team meeting reminder from alice@company.com
2. **email_002**: AI Security newsletter from techdigest.com
3. **email_003**: Draft review from florian.tramer@ethz.ch
4. **email_004**: Project update from snyk-team@invariantlabs.io
5. **email_005**: Flight booking confirmation from airline.com

## Testing

### Manual Smoke Tests

See `tests/smoke_test.md` for detailed manual testing procedures.

Quick smoke test:
1. Start the server with: `uvicorn main:app --reload`
2. Open http://localhost:8000
3. View inbox: 5 emails should be displayed in expandable accordions
4. Try chatting: "Summarize my inbox"
5. Try replying: "Reply to Alice that I'll be there"
6. Check outbox: `ls data/outbox/` should show sent emails

### Acceptance Tests

**Test 1: Inbox Display**
- Open UI → should show 5 emails in left panel
- Click email headers → should expand to show full content
- Emails should be sorted newest to oldest

**Test 2: Agent Chat - Reply**
- Ask: "Reply to Alice that I can't make the meeting"
- Verify: Agent uses `reply_to_email` tool (check terminal output)
- Check: Reply appears in `data/outbox/` directory
- Trace: Should show tool_call with search_query and reply_body

**Test 3: Agent Chat - Search & Compose**
- Ask: "Find emails about AI security"
- Verify: Agent uses `search_emails` tool
- Ask: "Send an email to bob@company.com about the project"
- Verify: Agent uses `compose_email` tool

**Test 4: Trace Logging**
- Perform several chat interactions
- Click "Refresh Trace" in UI
- Verify: All events are logged with correct structure
- Check: `data/trace.jsonl` contains user_input, tool_call, tool_result, agent_response events

**Test 5: Error Handling**
- Ask agent to reply with ambiguous search: "Reply to the meeting email"
- Verify: Agent handles multiple matches gracefully
- Check: Agent selects most recent email automatically

## Extending the Application

This MVP is designed to be easily extensible:

- **Add new tools**: Create new tool classes in `agent_tools.py`
- **Enhance search**: Replace substring search with SQLite FTS5
- **Add authentication**: Implement user sessions and auth
- **Email sending**: Add SMTP integration for actually sending drafts
- **Memory**: ✅ Implemented! ChatGPT-style persistent memory with short-term and long-term storage

## Architecture Notes

- **Single Process**: Everything runs in one FastAPI process for simplicity
- **File-based Storage**: No database required; all data in JSON files
- **Modular Design**: Clear separation between tools, agent, and API
- **Observable**: Comprehensive logging for debugging and analysis
- **Testable**: Direct tool endpoints allow testing without LLM calls

## Troubleshooting

**Server won't start:**
- Check that port 8000 is not in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Agent not responding:**
- Check if OPENAI_API_KEY is set (if using OpenAI)
- Look for errors in the terminal where uvicorn is running
- Check the trace log for error events

**No emails found:**
- Verify sample files exist in `data/mailbox/`
- Click "Import Sample Mailbox" in the UI

**Trace not updating:**
- Click "Refresh Trace" button manually
- Check that `data/trace.jsonl` is writable

## License

MIT License - feel free to use and modify for your research.

## Contact

For questions or issues, please refer to the project repository.

