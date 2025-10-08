# Smoke Test Suite

Manual smoke tests to verify the Email Agent MVP is working correctly.

## Prerequisites

- Server is running on http://localhost:8000
- Browser is open to the application UI
- Sample emails are loaded in `data/mailbox/`

## Test 1: Server Health Check

**Objective**: Verify the server is running and configured correctly.

**Steps**:
1. Open terminal
2. Run: `curl http://localhost:8000/api/health`

**Expected Result**:
```json
{
  "status": "healthy",
  "config": {
    "model": "gpt-4",
    "mailbox_dir": "data/mailbox",
    "drafts_dir": "data/drafts"
  }
}
```

**Pass Criteria**:
- ✅ Status is "healthy"
- ✅ Configuration values are present
- ✅ Response time < 1 second

---

## Test 2: Mailbox Import and Direct Tool Access

**Objective**: Verify sample emails are loaded and direct tool endpoints work.

**Steps**:
1. Open UI at http://localhost:8000
2. Click "Import Sample Mailbox" button
3. Verify message shows "Mailbox contains 5 emails"
4. In "Read Email" section, enter `email_001` and click "Read"
5. In "Search Inbox" section, enter `meeting` and click "Search"
6. In "Draft Email" section:
   - To: `test@example.com`
   - Subject: `Test Draft`
   - Body: `This is a test draft email.`
   - Click "Create Draft"

**Expected Results**:

*Step 2-3*: Import confirmation message appears

*Step 4*: Email content displays:
```
Email ID: email_001
From: alice@company.com
To: you@example.com
Subject: Team Meeting Tomorrow
...
```

*Step 5*: Search results show:
```
Found 1 email(s) matching 'meeting':

1. [email_001] From: alice@company.com
   Subject: Team Meeting Tomorrow
   ...
```

*Step 6*: Draft confirmation:
```
Draft created successfully!
Draft ID: draft_...
To: test@example.com
Subject: Test Draft
```

**Pass Criteria**:
- ✅ All 5 sample emails are confirmed
- ✅ Read tool returns full email content
- ✅ Search tool finds correct email(s)
- ✅ Draft tool creates file in `data/drafts/`
- ✅ All results appear in the UI chat/tool output area

---

## Test 3: Agent Conversation

**Objective**: Verify the LangChain agent can understand requests and use tools.

**Steps**:
1. In the chat input box, type: `Search for emails about meetings`
2. Click "Send" or press Enter
3. Wait for agent response
4. Next, type: `Read email email_003`
5. Click "Send"
6. Finally, type: `Draft a reply to alice@company.com thanking her for the reminder`
7. Click "Send"

**Expected Results**:

*Step 2-3*: Agent should:
- Acknowledge the search request
- Use the `search_inbox` tool with query "meetings"
- Return results showing email_001

*Step 4-5*: Agent should:
- Use the `read_email` tool with email_id "email_003"
- Display the email from florian.tramer@ethz.ch

*Step 6-7*: Agent should:
- Use the `draft_email` tool
- Create a draft to alice@company.com
- Confirm draft creation with draft ID

**Pass Criteria**:
- ✅ Agent correctly interprets user requests
- ✅ Agent selects appropriate tools
- ✅ Agent provides coherent responses
- ✅ Tool calls are visible in chat history
- ✅ No error messages in responses

**Note**: If OPENAI_API_KEY is not set, the agent may have limited functionality. This is expected and documented.

---

## Test 4: Trace Logging and Session Management

**Objective**: Verify that all interactions are logged to the trace file with correct structure.

**Steps**:
1. Perform a few actions (chat messages, tool calls) in the UI
2. Click the "Refresh Trace" button in the Trace Log panel
3. Verify trace events appear in the log
4. Check the session ID displayed at the bottom of the UI
5. Open terminal and run:
   ```bash
   cat data/trace.jsonl | tail -10
   ```
6. Open a new browser tab/window to http://localhost:8000
7. Note the different session ID
8. Perform an action in the new tab
9. In terminal, run:
   ```bash
   curl http://localhost:8000/api/session/{SESSION_ID_1}/trace
   curl http://localhost:8000/api/session/{SESSION_ID_2}/trace
   ```
   (Replace SESSION_ID_1 and SESSION_ID_2 with actual session IDs from each tab)

**Expected Results**:

*Step 2-3*: Trace log shows events like:
```json
{
  "ts": "2025-10-06T...",
  "event_id": "...",
  "session_id": "session_...",
  "event_type": "user_input",
  "payload": { "text": "..." }
}
```

*Step 5*: Terminal shows valid JSON lines, each with:
- `ts`: ISO 8601 timestamp
- `event_id`: UUID
- `session_id`: Session identifier
- `event_type`: One of user_input, tool_call, tool_result, agent_response
- `payload`: Event-specific data

*Step 9*: Each session returns only its own events

**Pass Criteria**:
- ✅ Trace events appear in UI after refresh
- ✅ data/trace.jsonl contains valid JSON lines
- ✅ Each event has all required fields
- ✅ Event types are correct (user_input, tool_call, tool_result, agent_response)
- ✅ tool_call and tool_result events have matching `call_id`
- ✅ Sessions are isolated (different session_ids for different tabs)
- ✅ Session-specific trace retrieval works correctly

---

## Additional Checks

### File System Verification

Run these commands to verify the file structure:

```bash
# Verify mailbox has 5 emails
ls -la data/mailbox/*.json | wc -l
# Should output: 5

# Verify drafts directory exists and contains created drafts
ls -la data/drafts/

# Verify trace file exists and has content
wc -l data/trace.jsonl
# Should show number of trace events
```

### Port and Process

```bash
# Verify server is listening on port 8000
lsof -i :8000

# Should show uvicorn/Python process
```

---

## Summary Checklist

After running all smoke tests, verify:

- [ ] Server starts without errors
- [ ] UI loads and displays correctly
- [ ] All 5 sample emails are accessible
- [ ] Read tool works for any email ID
- [ ] Search tool finds emails by keyword
- [ ] Draft tool creates JSON files in data/drafts/
- [ ] Agent responds to chat messages
- [ ] Agent uses appropriate tools
- [ ] Trace events are logged to data/trace.jsonl
- [ ] Trace log displays in UI
- [ ] Sessions are isolated
- [ ] No Python errors in terminal
- [ ] No JavaScript errors in browser console

---

## Troubleshooting

**If any test fails:**

1. Check terminal for Python errors
2. Check browser console for JavaScript errors
3. Verify file permissions on data/ directories
4. Ensure all dependencies are installed
5. Verify config.yaml is present and valid
6. Check that OPENAI_API_KEY is set (if using OpenAI)

**Common Issues:**

- **Agent not responding**: Check OPENAI_API_KEY or switch to mock mode
- **Trace not updating**: Click "Refresh Trace" manually
- **Tools not working**: Verify data/mailbox/ contains sample emails
- **Server won't start**: Check if port 8000 is already in use

---

## Success Criteria

All tests pass = MVP is working correctly and ready for extension! 🎉

