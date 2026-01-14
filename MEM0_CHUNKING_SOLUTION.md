# Mem0 Token Limit Error - Chunking Solution

## Problem

Mem0's embedding model (`text-embedding-3-small`) has an **8192 token limit**. When large context queries (~110k tokens) from Session 2 are stored in mem0, the extracted facts can be very large. During `memory.search()`, mem0 tries to embed all stored memories, and if some memories exceed the token limit, the embedding model fails with:

```
Error code: 400 - "This model's maximum context length is 8192 tokens, 
however you requested 105107 tokens"
```

## Root Cause

1. **Large messages stored**: Session 2's large context queries (~110k tokens) are passed to mem0
2. **Facts extracted**: mem0 extracts facts from these large messages
3. **Search fails**: When searching, mem0 tries to embed all stored memories, including large ones
4. **Token limit exceeded**: The embedding model's 8192 token limit is exceeded

## Solution: Chunking + Truncation

### 1. Message Chunking (Before Extraction)

**Location**: `src/agent/backend/mem0_memory_manager.py` - `_chunk_large_message()` method

**How it works**:
- Before passing messages to mem0, check if any message exceeds 6000 tokens
- If so, chunk the message into smaller pieces (6000 tokens each)
- Each chunk is processed separately by mem0
- This ensures extracted facts are smaller and won't cause embedding errors

**Token limit**: 6000 tokens per chunk (safe for 8192 limit, leaves room for overhead)

**Implementation**:
- Uses `tiktoken` for accurate token counting
- Falls back to character-based estimation if tiktoken unavailable
- Preserves message role and adds chunk metadata

### 2. Memory Truncation (After Extraction)

**Location**: `src/agent/backend/mem0_memory_manager.py` - post-processing in `add_memory()`

**How it works**:
- After mem0 extracts facts, check if any extracted memory exceeds safe limits
- Truncate memories that exceed 32,000 characters (~8000 tokens)
- This provides defense-in-depth in case mem0 combines chunks or creates large facts

**Safe limit**: 32,000 characters (≈8000 tokens, safe for 8192 limit)

## Inputs to Mem0

**Format**: List of message dictionaries
```python
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

**What happens**:
1. Messages are chunked if too large (>6000 tokens)
2. Chunked messages are passed to `memory.add()`
3. mem0 extracts facts using LLM (USER_MEMORY_EXTRACTION_PROMPT)
4. Extracted facts are stored in vector store
5. During search, mem0 embeds all memories and finds similar ones

## Benefits

1. **Prevents token limit errors**: Chunking ensures no single memory exceeds embedding model limits
2. **Preserves information**: Large messages are still processed, just in smaller pieces
3. **Backward compatible**: Small messages are unaffected
4. **Defense in depth**: Truncation provides additional safety

## Testing

To verify the fix works:
1. Run the `long_memory` test suite with mem0 backend
2. Check logs for chunking messages: `"📦 Chunked X messages into Y chunks"`
3. Verify no token limit errors during search
4. Confirm mem0 search returns results successfully

## Configuration

The chunking is automatic and requires no configuration. The limits are:
- **Chunk size**: 6000 tokens (hardcoded, safe for 8192 limit)
- **Memory truncation**: 32,000 characters (hardcoded, safe for 8192 limit)

These limits can be adjusted if needed, but should remain below the embedding model's 8192 token limit.

