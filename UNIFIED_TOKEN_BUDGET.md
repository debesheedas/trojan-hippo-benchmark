# Unified Token Budget Management Using LangChain

## Problem

When using memory backends (especially `context` memory), large context memory is prepended to user messages before they're added to session history. This causes token limit errors because:

1. Context memory can be 100k+ tokens (e.g., 453k chars ≈ 113k tokens)
2. This is prepended to the user message BEFORE session message truncation
3. Even if we truncate session messages, the current user message already exceeds limits
4. Custom string truncation is brittle and doesn't leverage LangChain's built-in solutions

## Solution: Use LangChain's `trim_messages` for Everything

Instead of custom string truncation, we use **LangChain's `trim_messages` on the ENTIRE message list**. This leverages LangChain's built-in sliding window logic for all truncation.

### Token Allocation Strategy

```
Total Available Tokens (model context window)
├── Reserved Tokens (fixed)
│   ├── System prompt (~5k tokens)
│   ├── Generation tokens (configurable, ~2k default)
│   └── Buffer tokens (configurable, ~50k default)
└── All Messages (session history + current user message with context)
    └── Handled entirely by LangChain's trim_messages (sliding window)
```

### Implementation Flow

1. **Build Complete Message List**:
   - Get session history
   - Build current user message with all context (RAG + mem0 + context memory)
   - Add user message to session history

2. **Calculate Total Budget**:
   ```
   max_tokens_for_all_messages = context_window - reserved_tokens
   ```

3. **Apply LangChain's `trim_messages`**:
   - Pass ENTIRE message list (session history + current message with context)
   - Use `strategy="last"` to keep most recent messages
   - LangChain handles all truncation logic automatically
   - If a single message (e.g., user message with huge context) exceeds limit, LangChain still keeps it (most recent message)

4. **Result**: LangChain's built-in logic handles:
   - Truncation of session history (sliding window)
   - Handling of large individual messages
   - Token counting and optimization

## Benefits

1. **Uses LangChain's Built-in Functions**:
   - No custom string truncation logic
   - Leverages LangChain's `trim_messages` for everything
   - LangChain handles all edge cases (large messages, token counting, etc.)

2. **Works for All Memory Backends**: 
   - No backend-specific logic
   - LangChain handles all truncation uniformly
   - Works with explicit, RAG, mem0, context, and none backends

3. **Works with All Defenses**:
   - Defense logic is separate from token management
   - Token budget is calculated after defense filtering

4. **Robust and Clean**:
   - Single LangChain function handles everything
   - No brittle custom truncation code
   - Uses LangChain's recommended approach throughout

5. **Prevents Token Limit Errors**:
   - LangChain's `trim_messages` ensures we never exceed model limits
   - Handles large individual messages gracefully
   - Provides clear warnings when truncation occurs

## Code Location

**File**: `src/agent/agent_core.py`

**Function**: `invoke_agent()` (lines ~870-1000)

**Key Functions**:
- `_truncate_session_messages()`: Handles session history truncation using LangChain's `trim_messages`

## Example

For GPT-5 Mini (400k context window):
- Reserved: 5k (system) + 2k (generation) + 50k (buffer) = 57k tokens
- Max tokens for all messages: 400k - 57k = 343k tokens

**Scenario 1: Normal case**
- Session history: 50k tokens
- Current user message with context: 100k tokens
- Total: 150k tokens < 343k ✅
- LangChain keeps all messages

**Scenario 2: Large context memory**
- Session history: 200k tokens
- Current user message with context: 150k tokens
- Total: 350k tokens > 343k ❌
- LangChain's `trim_messages` applies sliding window:
  - Keeps current user message (most recent)
  - Truncates session history to fit remaining budget
  - Result: ~143k tokens for session history ✅

**Scenario 3: Huge single message**
- Session history: 50k tokens
- Current user message with context: 300k tokens
- Total: 350k tokens > 343k ❌
- LangChain's `trim_messages` keeps the most recent message (current user message)
  - May remove some or all session history to fit
  - Ensures we never exceed limits ✅

## Configuration

Token budget parameters are configured in `agent_config.yaml`:

```yaml
memory:
  context_memory:
    buffer_length: 50000  # Buffer tokens to reserve
    max_context_length: 270000  # Max context memory tokens (if set, overrides auto-calculation)
```

Benchmark generation tokens:
```yaml
benchmark:
  dspy:
    max_tokens: 2000  # Generation tokens
```

