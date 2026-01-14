# Session Message Token-Based Truncation Implementation

## Problem

LangChain's `ChatOpenAI` does **NOT** automatically truncate conversation history when it exceeds token limits. It passes all messages directly to the OpenAI API, which causes `context_length_exceeded` errors when:
- Multiple large messages accumulate in a session
- Total tokens exceed the model's context window (e.g., 272k tokens for GPT-5 Mini with buffer)

Example error:
```
Input tokens exceed the configured limit of 272000 tokens. 
Your messages resulted in 326544 tokens.
```

## Solution

Implemented **LangChain's recommended approach** using `trim_messages` from `langchain_core.messages`, with fallback to custom token-based truncation if LangChain utilities aren't available.

### Key Implementation Details

1. **Full History Preservation**: We keep the complete conversation history in `_session_store` for all memory backends (explicit, RAG, mem0, context).

2. **Truncation Before Agent Invocation**: Before passing messages to LangChain's agent, we:
   - Create a copy of the session messages
   - Calculate token count for each message
   - Apply sliding window truncation (keep most recent messages that fit within token limit)
   - Pass the truncated copy to the agent

3. **Token Limit Calculation**: 
   - Model context window (e.g., 400k for GPT-5 Mini)
   - Minus buffer tokens (50k default) for system prompt, generation tokens, etc.
   - Minus generation max tokens (2k default)
   - Result: Maximum tokens available for conversation history

4. **Sliding Window Approach**: 
   - Keeps the most recent messages (from the end)
   - Ensures we always have the latest context
   - At minimum, keeps the last message (even if it exceeds limit)

### Code Changes

**File**: `src/agent/agent_core.py`

1. **Added `_truncate_session_messages()` function**:
   - **Primary**: Uses LangChain's `trim_messages()` function (recommended approach)
     - Converts message dicts to LangChain message objects (`HumanMessage`, `AIMessage`, `SystemMessage`)
     - Uses `strategy="last"` to keep most recent messages
     - Provides token counter function using `tiktoken` for accurate counting
   - **Fallback**: Custom token-based truncation if LangChain utilities unavailable
     - Uses `tiktoken` for accurate token counting
     - Falls back to character-based estimation if tiktoken unavailable
     - Implements sliding window truncation (keep most recent messages)

2. **Modified `invoke_agent()` function**:
   - Calculates max tokens based on model context window
   - Truncates a copy of session messages before passing to agent
   - Keeps full history in `_session_store` for future turns

### How It Works

```python
# 1. Get full session history
session_messages = _get_session_memory(session_id)

# 2. Add new user message
session_messages.append({"role": "user", "content": user_message})

# 3. Truncate COPY for agent (keep full history)
messages_for_agent = _truncate_session_messages(
    session_messages.copy(),
    model_name=model_name,
    max_tokens=max_session_tokens
)

# 4. Pass truncated version to agent
inputs = {"messages": messages_for_agent}
result = agent.invoke(inputs)

# 5. Append response to FULL history (not truncated)
session_messages.append({"role": "assistant", "content": response_text})
```

### Benefits

1. **Uses LangChain's Recommended Solution**: Leverages `trim_messages()` - the official LangChain approach
2. **Prevents Token Limit Errors**: Ensures we never exceed model context window
3. **Preserves Recent Context**: Sliding window keeps most recent messages
4. **Maintains Full History**: Complete conversation history preserved for memory systems
5. **Model-Aware**: Automatically adjusts based on model's context window
6. **Robust Fallback**: Falls back gracefully if LangChain utilities or tiktoken unavailable
7. **Clean Implementation**: Uses LangChain's built-in utilities instead of custom code

### Configuration

Token limits are calculated from:
- Model context window (auto-detected from model name)
- `buffer_length` in `context_memory` config (default: 50k tokens)
- `max_tokens` in benchmark config (default: 2k tokens)

Example for GPT-5 Mini:
- Context window: 400k tokens
- Buffer: 50k tokens
- Generation: 2k tokens
- **Max session tokens: 348k tokens**

### Testing

This fix addresses token limit errors for:
- ✅ Explicit memory backend
- ✅ RAG memory backend  
- ✅ Context memory backend
- ⚠️ mem0 memory backend (separate issue with embedding model limits)

### Comparison with Context Memory Backend

The context memory backend already had sliding window truncation for its own memory storage. This implementation applies the same approach to **session messages** passed to LangChain, ensuring consistency across the codebase.

