# Benchmark Analysis: Long Memory Test Suite

## Summary of Findings

### 1. Explicit Memory Success Rate Discrepancy

**Observed Results:**
- `none` defense: **40% success** (2/5 probes pass)
- `user_prompt_only` defense: **100% success** (5/5 probes pass)
- `no_untrusted_tools` defense: **100% success** (5/5 probes pass)

**Root Cause Analysis:**

The discrepancy appears to be due to **agent behavior**, not a bug:

1. **Tool Availability Difference:**
   - `search_emails` is an **UNTRUSTED tool**
   - In `none` defense: Agent has access to `search_emails`
   - In `user_prompt_only`/`no_untrusted_tools`: Agent may be restricted or prefer explicit memory

2. **Agent Response Pattern:**
   - **`none` defense**: Agent says "I don't have any memory or record of that" and offers to use `search_emails`
   - **`user_prompt_only`/`no_untrusted_tools`**: Agent directly recalls from explicit memory

3. **Hypothesis:**
   - When `search_emails` is available, the agent may be **choosing to use it** instead of relying on explicit memory in the system prompt
   - This could be due to:
     - Agent's preference for tool-based retrieval
     - Non-deterministic behavior in tool selection
     - Explicit memory might not be prominently featured in the system prompt

**Verification Needed:**
- Check if explicit memory is actually loaded into system prompt for both cases
- Verify agent is seeing the same explicit memory content
- This may be expected behavior (agent choosing tools over system prompt memory)

### 2. Mem0 Token Limit Error

**Error Found:**
```
Warning: Could not search mem0 memory: Error code: 400
Message: "This model's maximum context length is 8192 tokens, 
         however you requested 105107 tokens"
```

**Root Cause:**
- mem0's embedding model (`text-embedding-3-small`) has an **8192 token limit**
- Session 2's large context queries (~110k tokens total) are being stored in mem0
- When mem0 searches, it tries to process all memories including these large ones
- This exceeds the embedding model's token limit

**Impact:**
- mem0 search **fails silently** (error is caught and returns empty list)
- Agent falls back to other methods or returns empty results
- This could be affecting mem0's 80% success rate

**Location:**
- Error occurs in `src/agent/backend/mem0_memory_manager.py` line 465
- Error is caught and handled gracefully, but search returns empty results

**Recommendation:**
- Need to truncate or filter large memories before storing in mem0
- Or filter large memories during search to avoid token limit errors

### 3. RAG Success Rate (100%)

**Observed:** 100% success across all defenses

**Status:** ✅ **Expected and Correct**

**Reason:**
- RAG uses semantic search over **chunked** documents
- Chunks are limited to 512 tokens (configurable)
- RAG retrieves relevant chunks, not full conversation history
- Not affected by context overload from Session 2
- This is the expected behavior for RAG memory backend

### 4. Context Memory Success Rate (0%)

**Observed:** 0% success across all defenses

**Status:** ✅ **Expected and Correct**

**Reason:**
- Context memory uses sliding window truncation
- Session 2's large messages (~110k tokens) exceed the limit
- LangChain's `trim_messages` evicts Session 1 memories (oldest)
- Session 3 probes fail because Session 1 facts are no longer in context
- This is the intended behavior - confirms truncation is working correctly

## Recommendations

### 1. Explicit Memory Investigation
- **Action:** Verify explicit memory is loaded correctly in system prompt for all defenses
- **Check:** Compare system prompts between `none` and `user_prompt_only` defenses
- **Possible Fix:** If agent is choosing `search_emails` over explicit memory, we may need to:
  - Make explicit memory more prominent in system prompt
  - Or accept this as expected non-deterministic behavior

### 2. Mem0 Token Limit Fix
- **Action:** Prevent large context queries from being stored in mem0
- **Options:**
  - Truncate messages before storing in mem0 (already done for `limit_memory_length` defense)
  - Filter out large memories during search
  - Use a different embedding model with higher token limits
  - Skip memory extraction for large context queries

### 3. No Action Needed
- **RAG:** Working as expected
- **Context Memory:** Working as expected (truncation confirmed)

## Conclusion

1. **Explicit Memory Discrepancy:** Likely due to agent's tool selection preference, not a bug. Needs verification.
2. **Mem0 Error:** Real bug - token limit errors are being silently handled. Needs fix.
3. **RAG:** Working correctly.
4. **Context Memory:** Working correctly (truncation confirmed).

