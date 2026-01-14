# Mem0 Error Analysis - Search Token Limit Issue

## Summary

**Status**: ⚠️ **Errors are occurring but being silently handled**

The chunking solution **IS working** for preventing large memories from being stored, but **search errors still occur** because mem0's `search()` method internally tries to embed ALL stored memories at once, which can exceed the embedding model's 8192 token limit.

## Findings

### 1. Chunking Works ✅

**Evidence from logs:**
```
📦 Chunked 2 messages into 19 chunks to prevent token limit errors
📦 Chunked 2 messages into 18 chunks to prevent token limit errors
📦 Chunked 2 messages into 21 chunks to prevent token limit errors
```

- Messages are being chunked before storage
- Large messages (>6000 tokens) are split into smaller pieces
- Each chunk is processed separately by mem0

### 2. Search Errors Still Occur ⚠️

**Evidence from logs:**
```
Warning: Could not search mem0 memory: Error code: 400 - 
{'error': {'message': "This model's maximum context length is 8192 tokens, 
however you requested 105107 tokens (105107 in your prompt; 0 for the completion). 
Please reduce your prompt; or completion length.", ...}}
```

**When errors occur:**
- Step 10 (first large context query): Line 1032
- Step 10 (after chunking): Line 2843
- Step 11: Lines 4664, 4787
- Step 12: Lines 4874, 5062
- Step 13: Lines 5149, 5338

**Pattern:**
- Errors occur during `get_context()` → `search()` calls
- Happens BEFORE chunking is applied (Step 10, first error)
- Also happens AFTER chunking (Step 10, second error - after chunking was added)

### 3. Root Cause

**The Problem:**
1. **Chunking prevents large memories from being stored** ✅
2. **BUT mem0's `search()` internally tries to embed ALL stored memories at once** ❌
3. **If there are many memories OR some are still large, token limit exceeded** ❌

**Why chunking doesn't fully solve it:**
- Chunking happens during `add_memory()` - prevents large single memories
- But `search()` processes ALL stored memories together
- Even if each memory is small, many memories together can exceed 8192 tokens
- OR if some memories are still large (from before chunking was added), they cause errors

### 4. Current Error Handling

**What happens when search fails:**
1. Error is caught in `search()` method (line 600)
2. Returns empty list `[]`
3. `get_context()` falls back to `get_all_memories()` (line 625)
4. Fallback tries to get recent memories without embedding
5. May still work but is inefficient

**From code:**
```python
def search(...):
    try:
        result = self.memory.search(...)
        return memories
    except Exception as e:
        print(f"Warning: Could not search mem0 memory: {e}")
        return []  # Returns empty, triggers fallback

def get_context(...):
    memories = self.search(...)  # May return []
    if len(memories) < self.top_k:
        # Fallback to get_all_memories()
        all_memories = self.get_all_memories(...)
```

### 5. Impact

**Positive:**
- Errors are caught and handled gracefully
- System doesn't crash
- Fallback mechanism provides some results

**Negative:**
- Search fails silently (only warning printed)
- Falls back to less efficient `get_all_memories()`
- May return empty or limited results
- Performance impact (inefficient fallback)

## Test Results

**From logs (Step 15 - Session 3 probe):**
- Search error occurred (line 2843)
- BUT context was still retrieved (lines 2845-2880 show memories)
- This suggests the fallback (`get_all_memories()`) is working
- Agent was able to recall information (80% success rate)

**Conclusion:**
- Errors are being handled, but not optimally
- System still works due to fallback
- But search is failing, which is inefficient

## Recommendations

### Option 1: Limit Search Scope (Recommended)
- Modify `search()` to limit how many memories are processed
- Use `get_all_memories()` with a limit before searching
- Only search through recent N memories (e.g., last 50)

### Option 2: Batch Search
- Split memories into batches
- Search each batch separately
- Combine results

### Option 3: Accept Current Behavior
- Errors are handled gracefully
- Fallback works
- System still functions (80% success rate)
- Document the limitation

### Option 4: Skip Large Memories During Search
- Check memory size before including in search
- Skip memories that exceed token limit
- Only search through smaller memories

## Next Steps

1. **Immediate**: Document that search errors occur but are handled
2. **Short-term**: Implement Option 1 (limit search scope)
3. **Long-term**: Consider if mem0 has configuration options to limit search scope

## Code Locations

- **Chunking**: `src/agent/backend/mem0_memory_manager.py` - `_chunk_large_message()` (line ~142)
- **Search Error**: `src/agent/backend/mem0_memory_manager.py` - `search()` (line 600)
- **Fallback**: `src/agent/backend/mem0_memory_manager.py` - `get_context()` (line 625)

