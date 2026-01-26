# Pure In-Memory Architecture

## Overview

This benchmark operates **entirely in-memory** with no file system dependencies for test execution environments. All test environments, mailboxes, memory backends, and vector stores are stored in RAM.

**IMPORTANT**: File system storage has been completely removed from the codebase. There are no legacy/fallback modes - the system is designed exclusively for in-memory operation.

## Clean API Design

### EmailToolsConfig

```python
class EmailToolsConfig:
    def __init__(self, mailbox: InMemoryMailbox, trace_store: InMemoryTraceStore, 
                 defense_type: str = None):
        """
        Args:
            mailbox: In-memory mailbox storage (required)
            trace_store: In-memory trace store for logging (required)
            defense_type: Defense type for tools
        """
    
    def append_trace_event(self, event_type: str, payload: dict) -> str:
        """Log event to trace_store (called by tools)."""
```

### MemoryManager (Explicit Memory)

```python
class MemoryManager:
    def __init__(self, max_short_term: int = 15):
        """In-memory storage for short-term and long-term memory."""

def get_memory_manager() -> MemoryManager:
    """Get cached in-memory memory manager."""
```

### ContextMemoryManager

```python
class ContextMemoryManager:
    def __init__(self, max_context_length: int = None, model_name: str = None):
        """In-memory context storage."""

def get_context_memory_manager(max_context_length: int = None, model_name: str = None):
    """Create in-memory context memory manager."""
```

### RAGMemoryManager

```python
class RAGMemoryManager:
    def __init__(self, vectorstore: InMemoryVectorstore, embedding_model: str = "text-embedding-3-small",
                 top_k: int = 8, chunk_size: int = 512, api_key: str = None):
        """
        Args:
            vectorstore: In-memory vectorstore (required)
            ...
        """

def get_rag_memory_manager(vectorstore: InMemoryVectorstore, ...):
    """Create RAG memory manager with required vectorstore."""
```

### Mem0MemoryManager

```python
class Mem0MemoryManager:
    def __init__(self, llm_provider: str = "openai", llm_model: str = "gpt-5-mini", ...):
        """Mem0 manages its own internal in-memory vectorstore."""

def get_mem0_memory_manager(...) -> Mem0MemoryManager:
    """Create mem0 memory manager (always in-memory)."""
```

### Trace Events

Trace logging is done directly through `InMemoryTestEnvironment`:

```python
# Log a trace event
in_memory_env.log_event(session_id, "user_input", {"text": "Hello"})

# Read trace events
traces = in_memory_env.get_traces(session_id)
```

Tools receive `trace_store` explicitly:
- `EmailToolsConfig` takes `trace_store` parameter
- `UpdateMemoryTool` has `trace_store` attribute
- Both have `append_trace_event()` helper methods

## Core In-Memory Components

| Component | Description | Location |
|-----------|-------------|----------|
| `InMemoryTestEnvironment` | Complete test environment (single source of truth) | `src/benchmark/in_memory_storage.py` |
| `InMemoryMailbox` | Email storage (inbox/outbox/drafts) | `src/benchmark/in_memory_storage.py` |
| `InMemoryVectorstore` | FAISS vectorstore wrapper | `src/benchmark/in_memory_storage.py` |
| `InMemoryTraceStore` | Trace event storage with `log_event()` and `get_traces()` | `src/benchmark/in_memory_storage.py` |
| `InMemorySessionStore` | Session history storage | `src/benchmark/in_memory_storage.py` |

## Test Config Structure

When running tests, the config contains:

```python
test_config = {
    "mailbox": InMemoryMailbox(),  # Required - shortcut to in_memory_environment.mailbox
    "in_memory_environment": InMemoryTestEnvironment(),  # Single source of truth
    "memory": {
        "backend": "rag",  # or "mem0", "context", "explicit", "none"
        "rag_memory": {
            "vectorstore": InMemoryVectorstore(),  # Required for RAG
            "manager": RAGMemoryManager(),  # Shared instance for persistence
            "defense_type": "none",
        },
        "mem0_memory": {
            "manager": Mem0MemoryManager(),  # Shared instance for persistence
            "defense_type": "none",
        },
        "context_memory": {
            "manager": ContextMemoryManager(),  # Shared instance for persistence
            "defense_type": "none",
        },
        # Memory managers persist across invocations within a test
    },
    "data": {},  # Empty - no file paths needed
}
```

**Important**: Memory managers are stored in config and reused across agent invocations within a test. This ensures memories persist across steps.

## Mem0 Library Modifications

The bundled `mem0/` library has been modified for pure in-memory operation:

| File | Changes |
|------|---------|
| `mem0/mem0/memory/setup.py` | Removed `~/.mem0` directory creation |
| `mem0/mem0/memory/main.py` | Always sets `path=None` for telemetry |
| `mem0/mem0/vector_stores/faiss.py` | `path` param ignored, `_save()`/`_load()` are no-ops |

## Benefits

1. **Fast Execution**: No file I/O overhead
2. **Complete Isolation**: Each test gets isolated in-memory storage
3. **Clean Architecture**: No temporary directories or files
4. **Portable**: No file system layout dependencies
5. **Simple Testing**: Easy to inspect in-memory state

## What Still Uses Files

The following still write to the file system (intentionally):
- **Benchmark log files** (`logs/` directory) - Execution logs for debugging
- **Result JSON files** (`data/benchmark/` directory) - Test results and metrics
- **Cache files** - Optimization cache for adaptive attacks

These are **output files**, not part of the test execution environment.
