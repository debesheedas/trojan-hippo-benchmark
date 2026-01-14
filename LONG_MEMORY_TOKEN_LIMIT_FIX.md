# Long Memory Test Case Token Limit Issues and Fixes

## Problem Summary

The `long_memory` test case is hitting token limit errors in two places:

1. **mem0 Memory Search Errors**: 
   - Error: `This model's maximum context length is 8192 tokens, however you requested 105107 tokens`
   - Occurs when mem0 tries to search through memories that include large context queries (~100k tokens each)
   - The embedding model (`text-embedding-3-small`) has an 8192 token limit

2. **Main Agent Context Errors**:
   - Error: `Input tokens exceed the configured limit of 272000 tokens. Your messages resulted in 326544 tokens`
   - Occurs when the conversation history accumulates 4 large context queries (~100k tokens each)
   - Even with keeping only 15 messages, 4 × 100k tokens = 400k tokens, exceeding the 272k limit

## Root Cause

The test case is designed to "overload context" with ~80k words per query (~100k tokens) in Session 2:
- These large contexts are being stored as memories in mem0
- When mem0 searches, it tries to process all memories including these large ones
- The main agent conversation history also accumulates these large queries

## Recommended Solutions

### Solution 1: Reduce Large Context Query Size (RECOMMENDED)

**Modify**: `src/benchmark/dataset_generation/07_long_memory/long_memory_config.yaml`

Change:
```yaml
target_words_per_query: 80000  # Target number of words per session 2 query (~100k words)
```

To:
```yaml
target_words_per_query: 20000  # Target number of words per session 2 query (~25k tokens)
```

**Rationale**: 
- 20k words ≈ 25k tokens (still large enough to "overload context")
- 4 queries × 25k tokens = 100k tokens total (fits within 272k limit)
- Reduces mem0 search errors while still testing memory recall under context overload

### Solution 2: Skip Memory Extraction for Large Context Queries

**Modify**: Test case generation to mark large context queries as "no memory extraction"

**Implementation**: Add a flag to large context queries indicating they shouldn't be stored as memories, only used to overload conversation context.

### Solution 3: Implement Truncation in mem0 Search

**Modify**: `src/agent/backend/mem0_memory_manager.py` to truncate memories before searching if they exceed a threshold.

**Implementation**: Add truncation logic in the `search()` method to limit memory text length before passing to mem0's search.

### Solution 4: Use Larger Context Window Embedding Model

**Modify**: `agent_config.yaml` to use a model with larger context window for embeddings.

**Note**: This may be expensive and doesn't solve the main agent context limit issue.

## Immediate Action

**Recommended**: Implement Solution 1 (reduce query size) as it's the simplest and most effective fix.

Change `target_words_per_query` from `80000` to `20000` in `long_memory_config.yaml` and regenerate test cases.

## Testing

After implementing the fix:
1. Regenerate test cases: `python src/benchmark/dataset_generation/07_long_memory/generate_long_memory_test_cases.py`
2. Run benchmark: `python scripts/run_benchmark.py --suite long_memory --model gpt-5-mini --defense none`
3. Verify no token limit errors occur
4. Verify memory recall still works correctly (the test should still pass/fail based on memory accuracy, not token limits)

