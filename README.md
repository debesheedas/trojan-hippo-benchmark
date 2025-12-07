# Email Agent Security Benchmark

A security-focused email agent with persistent memory, featuring both an interactive web interface and automated benchmarking capabilities. The system includes static and adaptive benchmarking for evaluating agent security and resilience to attacks.

## Quick Start

### Running the Interactive Agent

Start the web-based interactive email agent:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run the interactive agent server
python src/interactive_agent/main.py
```

The server will start on `http://localhost:8000`. Open your browser and navigate to the URL to access the web interface.

**What it does**: Provides a conversational interface where you can interact with the email agent, manage emails (read, search, reply, forward, compose), and use persistent memory that persists across sessions.

### Running Benchmarks

The benchmark system supports three memory backends (explicit, mem0, rag) with unified defense types. Use the unified benchmark runner for all scenarios.

#### Quick Start

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run explicit memory with no defense on benign tests
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign

# Run mem0 with all defense types
python scripts/run_benchmark.py --memory-backend mem0 --all-defenses --suite benign

# Run a specific test file with RAG memory
python scripts/run_benchmark.py --memory-backend rag --defense-type user_only --test data/benchmark/tests/benign/00_email_tools.json
```

#### Memory Backends

Choose from three memory backends, each with different characteristics:

- **`explicit`**: JSON-based explicit memory
  - Simple, deterministic storage
  - Fast access, easy to inspect
  - Best for: Simple use cases, debugging, deterministic behavior
  
- **`mem0`**: Mem0 vector-based memory
  - Semantic search using embeddings
  - Automatic memory extraction via LLM
  - Best for: Complex memory needs, semantic similarity search
  
- **`rag`**: RAG-based memory
  - Chunked text retrieval
  - Document-based storage
  - Best for: Long-form content, document-based memory

**Note**: Only one memory backend should be enabled at a time. The `--memory-backend` argument automatically configures the correct backend.

#### Defense Types

Unified defense types work consistently across all backends:

- **`none`**: No defense (baseline behavior)
  - All messages are indexed normally
  - Use this as the baseline for comparison

- **`disable_memory`**: Completely disable memory indexing
  - No memory entries are created
  - Useful for baseline comparison without memory

- **`user_only`**: Only index user messages
  - Filters out agent responses
  - Prevents agent-generated content from being stored
  - **Backend mapping**: `user_prompt_only` for explicit, `user_only` for mem0/rag

- **`no_untrusted_tools`**: Block memory indexing after untrusted tools
  - Memory indexing stops once an untrusted tool is called
  - Prevents memory poisoning via tool calls
  - Works identically across all backends

- **`limit_memory_length`**: Limit memory entry length
  - Truncates memory entries to prevent long-form attacks
  - **Backend mapping**: `limit_memory_length` for explicit/mem0, `limit_chunk_size` for RAG

#### Running Test Suites

Test suites are organized by attack type:

```bash
# Run benign (normal behavior) tests
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign

# Run direct attack tests
python scripts/run_benchmark.py --memory-backend mem0 --defense-type user_only --suite direct

# Run indirect attack tests
python scripts/run_benchmark.py --memory-backend rag --defense-type none --suite indirect
```

#### Running Specific Tests

```bash
# Run a single test file
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --test data/benchmark/tests/benign/00_email_tools.json

# Run all tests in a directory
python scripts/run_benchmark.py --memory-backend mem0 --defense-type user_only --test data/benchmark/tests/benign/
```

#### Running All Defenses

Run all defense types for a backend in one command:

```bash
# Run all defenses for explicit memory
python scripts/run_benchmark.py --memory-backend explicit --all-defenses --suite benign

# This runs: disable_memory, none, user_only, no_untrusted_tools, limit_memory_length
```

#### Result Caching

The benchmark automatically skips tests if results already exist, preventing unnecessary re-runs:

```bash
# First run - executes all tests
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign
# Output: Running tests, saving results...

# Second run - skips tests (results exist)
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign
# Output: ⏭️  Skipping test_name.json - result already exists
#         Use --force to overwrite

# Force overwrite existing results
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign --force
# Output: Running tests, overwriting existing results...
```

#### Advanced Usage

```bash
# Use custom config file
python scripts/run_benchmark.py --memory-backend mem0 --defense-type user_only --suite benign --config my_config.yaml

# Use custom results directory
python scripts/run_benchmark.py --memory-backend rag --defense-type none --suite benign --results-dir data/custom_results

# Combine options
python scripts/run_benchmark.py \
  --memory-backend explicit \
  --defense-type user_only \
  --suite benign \
  --config config.yaml \
  --force \
  --results-dir data/benchmark/results
```

#### Command-Line Options

```bash
python scripts/run_benchmark.py --help

# Required:
--memory-backend {explicit,mem0,rag}  # Memory backend to use

# Test selection (one required):
--suite {benign,direct,indirect}      # Test suite to run
--test TEST                           # Specific test file or directory

# Defense selection (one required):
--defense-type {disable_memory,none,user_only,no_untrusted_tools,limit_memory_length}
--all-defenses                        # Run all defense types

# Optional:
--config CONFIG                       # Config file (default: config.yaml)
--force                              # Force overwrite existing results
--results-dir RESULTS_DIR            # Custom results directory
```

#### Results Location

Test results are saved in a unified structure:

```
data/benchmark/results/
  {memory_backend}/           # explicit, mem0, or rag
    {defense_type}/           # none, disable_memory, user_only, etc.
      {model_name}/           # gpt-5-mini, gpt-4o, etc.
        {attack_type}/        # benign, direct, or indirect
          {test_file}.json    # Individual test results
```

Example: `data/benchmark/results/explicit/none/gpt-5-mini/benign/00_email_tools.json`

### Running the Adaptive Benchmark

Run the adaptive benchmark with attack optimization when static attacks fail:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# 1. Enable adaptive mode in config.yaml
# Set benchmark.enable_adaptive_benchmark: true

# 2. Configure memory backend in config.yaml
# Set memory.backend: "explicit" (or "mem0" or "rag")
# Set memory.{backend}_memory.enabled: true

# 3. Run adaptive benchmark
python src/benchmark/test_bench.py --suite indirect --defense-type none

# Or use the unified runner (adaptive mode uses test_bench.py internally)
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite indirect
```

**What it does**: When a static attack fails, the system automatically attempts to optimize the attack using configured strategies (OpenEvolve or DSPy) to find a successful variant.

**Configuration**: Configure optimizers in `config.yaml`:
```yaml
benchmark:
  enable_adaptive_benchmark: true
  openevolve:
    enabled: true
    # ... optimizer settings
  dspy:
    enabled: false
    # ... optimizer settings
```

**Note**: Adaptive benchmarking uses `test_bench.py` internally. Memory backend and defense type are specified via config or command-line arguments.

---

## Project Structure

```
memory-agent-security-benchmark/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration (agent & benchmark settings)
├── src/
│   ├── agent/                   # Core agent implementation
│   │   ├── agent_core.py        # Main agent execution logic
│   │   ├── tools_registry.py    # Tool registration
│   │   ├── utils.py             # Helper functions
│   │   ├── memory_prompt.txt    # Memory behavior prompt
│   │   ├── tool_specifications/ # Tool implementations
│   │   │   ├── email_tools.py   # Email management tools
│   │   │   └── memory_tools.py # Memory management tools
│   │   └── backend/             # Backend services
│   │       ├── memory_manager.py
│   │       └── session_manager.py
│   ├── benchmark/               # Benchmarking system
│   │   ├── test_bench.py        # Main benchmark entry point
│   │   ├── test_validators.py  # Test validation logic
│   │   ├── memory_backend.py    # Memory backend abstractions
│   │   ├── defense_backend.py   # Defense backend abstractions
│   │   ├── benchmark_utils.py  # Benchmark utilities
│   │   ├── test_case_normalizer.py # Test case normalization
│   │   ├── unified_validator.py # Unified memory validator
│   │   ├── environment_state.py # Environment state management
│   │   └── adaptive_attacks/   # Adaptive attack optimizers
│   │       ├── base_optimizer.py
│   │       ├── dspy_optimizer.py
│   │       ├── openevolve_optimizer.py
│   │       └── scorer.py
│   └── interactive_agent/       # Web interface
│       ├── main.py              # FastAPI application
│       ├── frontend/            # React frontend components
│       ├── static/              # Static HTML/JS files
│       └── backend/routes/      # API routes
├── scripts/                     # Utility scripts
│   ├── run_benchmark.py        # Unified benchmark runner
│   ├── consolidate_results.py  # Unified results consolidation
│   └── migrate_test_cases.py  # Test case migration tool
├── data/
│   ├── interactive_agent/       # Interactive agent data
│   │   ├── mailbox/             # Inbox emails
│   │   ├── drafts/             # Draft emails
│   │   ├── outbox/             # Sent emails
│   │   ├── sessions/           # Session data
│   │   ├── agent_memory.json   # Persistent memory
│   │   └── trace.jsonl         # Interaction traces
│   └── benchmark/              # Benchmark data
│       ├── tests/              # Unified test directory
│       │   ├── benign/         # Benign behavior tests
│       │   ├── direct/         # Direct attack tests
│       │   └── indirect/       # Indirect attack tests
│       ├── results/            # Unified results directory
│       │   ├── explicit/       # Explicit memory results
│       │   ├── mem0/           # Mem0 memory results
│       │   └── rag/            # RAG memory results
│       ├── initial_inbox/     # Initial inbox states
│       ├── initial_outbox/    # Initial outbox states
│       ├── initial_drafts/    # Initial draft states
│       ├── initial_explicit_memory/ # Explicit memory sets
│       ├── initial_mem0_memory/     # Mem0 memory sets
│       ├── initial_rag_memory/      # RAG memory sets
│       ├── initial_sessions/ # Initial session states
│       ├── few_shot_examples/ # Few-shot examples for optimizers
│       ├── attack_bench_cache/ # Cached optimized attacks
│       └── test_envs/          # Isolated test environments
└── logs/                       # Application logs
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd memory-agent-security-benchmark
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```
   
   Required for using OpenAI models. The application will work with mock provider for testing, but LLM functionality requires a valid API key.

## Configuration

Edit `config.yaml` to customize behavior:

### Agent Configuration

```yaml
agent:
  provider: "openai"          # openai or mock
  target_model_name: "gpt-4o" # Model to use
  temperature: 0.0            # Set to 0.0 for determinism
  max_iterations: 10          # Max tool calls per turn
  verbose: true               # Verbose logging
```

### Memory Configuration

Configure which memory backend to use and its settings. **Only one backend should be enabled at a time.**

```yaml
memory:
  # Optional: specify default backend (can be overridden via CLI)
  backend: "explicit"  # Options: "explicit", "mem0", "rag"
  
  # Explicit memory configuration
  explicit_memory:
    enabled: true      # Set to true to use explicit memory
    defense_type: "none"  # Unified defense type: none, disable_memory, user_only, etc.
    memory_file: "data/interactive_agent/agent_memory.json"
  
  # Mem0 memory configuration
  mem0_memory:
    enabled: false     # Set to true to use mem0 memory
    defense_type: "none"  # Unified defense type (maps to "no_defense" internally)
    vectorstore_path: "data/interactive_agent/mem0_vectorstore"
    llm_provider: "openai"
    llm_model: "gpt-4o-mini"
    llm_temperature: 0.0
    embedding_provider: "openai"
    embedding_model: "text-embedding-3-small"
    vector_store_provider: "faiss"
    top_k: 10
    user_id: "vince"
    agent_id: "email_agent"
  
  # RAG memory configuration
  rag_memory:
    enabled: false     # Set to true to use RAG memory
    defense_type: "none"  # Unified defense type
    vectorstore_path: "data/interactive_agent/rag_vectorstore"
    chunk_size: 512
    chunk_overlap: 20
    embedding_model: "text-embedding-3-small"
    top_k: 15
```

**Important Notes**:
- **Only enable one backend**: Set `enabled: true` for the backend you want to use, and `enabled: false` for others
- **CLI overrides config**: The `--memory-backend` argument in `run_benchmark.py` automatically enables the specified backend and disables others
- **Unified defense types**: Use the same defense type names across all backends (e.g., `"none"`, `"user_only"`). The system automatically maps them to backend-specific implementations
- **Defense mapping**: 
  - `"none"` → `"none"` (explicit/rag) or `"no_defense"` (mem0)
  - `"user_only"` → `"user_prompt_only"` (explicit) or `"user_only"` (mem0/rag)
  - `"limit_memory_length"` → `"limit_memory_length"` (explicit/mem0) or `"limit_chunk_size"` (rag)

### Benchmark Configuration

```yaml
benchmark:
  enable_adaptive_benchmark: false  # Enable adaptive optimization
  test_dir: "data/benchmark/tests"  # Unified test directory
  results_dir: "data/benchmark/results"  # Unified results directory
  semantic_judge:
    model_name: "gpt-5-mini"
    temperature: 0.0
  openevolve:
    enabled: true
    # ... optimizer settings
  dspy:
    enabled: false
    # ... optimizer settings
```

### Global Settings

```yaml
seed: 42  # Global seed for reproducibility
```

## Features

### Email Agent Capabilities

- **Email Management**: Read, search, reply, forward, and compose emails
- **Multiple Memory Backends**: Support for explicit (JSON), mem0 (vector), and RAG memory systems
- **Unified Defense Types**: Consistent defense mechanisms across all memory backends
- **Tool-Based Architecture**: LangChain-based agent with structured tool calling
- **Session Management**: Multi-session support with isolated conversation histories

### Benchmarking System

- **Unified Benchmark Runner**: Single script for all memory backends and defense types
- **Result Caching**: Automatic skipping of tests with existing results
- **Static Benchmarking**: Automated test execution on predefined attack scenarios
- **Adaptive Benchmarking**: Automatic attack optimization when static attacks fail
- **Multiple Optimizers**: Support for OpenEvolve and DSPy-based optimization strategies
- **Comprehensive Validation**: Semantic judges and multiple validation strategies
- **Unified Test Format**: Single test format works with all memory backends
- **Isolated Test Environments**: Each test run uses a unique, isolated directory

### Interactive Interface

- **Web UI**: Single-page application with chat interface
- **Real-time Updates**: Live memory panel with SSE updates
- **Email Management**: Visual inbox, drafts, and outbox management
- **Trace Viewing**: Real-time trace log viewing for debugging

## API Endpoints (Interactive Agent)

### Main Endpoints

- `GET /` - Web UI
- `POST /api/chat` - Chat with the agent
- `GET /api/emails` - Get all emails
- `GET /sessions` - List all sessions
- `POST /sessions` - Create new session
- `GET /sessions/{session_id}` - Get session details

### Memory Endpoints

- `GET /memory` - Get current memory state
- `POST /memory` - Add to long-term memory
- `DELETE /memory` - Clear all memory
- `GET /memory/stream` - SSE stream for live memory updates

### Utility Endpoints

- `GET /api/session/{session_id}/trace` - Get trace events
- `GET /api/health` - Health check

## Memory System

The agent supports three memory backends, each with different characteristics:

### Explicit Memory (JSON-based)
- **Format**: Simple JSON file with structured memory entries
- **Storage**: `data/interactive_agent/agent_memory.json`
- **Characteristics**: Deterministic, easy to inspect, fast access
- **Best for**: Simple use cases, debugging, deterministic behavior

### Mem0 Memory (Vector-based)
- **Format**: Vector embeddings stored in FAISS vectorstore
- **Storage**: `data/interactive_agent/mem0_vectorstore/`
- **Characteristics**: Semantic search, automatic memory extraction, LLM-powered
- **Best for**: Complex memory needs, semantic similarity search

### RAG Memory (Retrieval Augmented Generation)
- **Format**: Chunked text stored in vectorstore
- **Storage**: `data/interactive_agent/rag_vectorstore/`
- **Characteristics**: Document-based, chunked retrieval, configurable chunking
- **Best for**: Long-form content, document-based memory

### Memory Behavior

- **Short-Term Memory**: Last 15 messages in conversation (session-scoped, all backends)
- **Long-Term Memory**: Backend-specific persistent storage (persists across sessions)
- **Auto-Detection**: The agent automatically detects "remember" and "forget" requests and updates memory accordingly

### Defense Mechanisms

All memory backends support unified defense types:
- **`none`**: No defense (baseline behavior)
- **`disable_memory`**: Completely disable memory indexing
- **`user_only`**: Only index user messages (filter agent responses)
- **`no_untrusted_tools`**: Block memory indexing after untrusted tools are called
- **`limit_memory_length`**: Limit memory entry length (backend-specific implementation)

## Benchmark Test Format

Tests are defined as JSON files with a **unified format** that works with all memory backends. The system automatically normalizes old-format test cases when loaded.

### Test File Structure

```json
{
  "name": "Test Name",
  "description": "Test description",
  "attack_type": "benign|direct|indirect",
  "initial_data": {
    "inbox_set": "inbox_set_1",
    "outbox_set": "outbox_set_1",
    "drafts_set": "drafts_set_1",
    "memory": {
      "set": "memory_set_0"  // Unified format - works for all backends
    },
    "session_set": "session_set_1"  // Optional: initial session state
  },
  "steps": [
    {
      "step_number": 1,
      "user_message": "User query",
      "expected_behavior": "What should happen",
      "success_check": {
        "type": "composite",
        "operator": "AND",
        "validators": [
          {
            "type": "keyword",
            "keywords": ["expected", "keywords"],
            "min_required": 2
          },
          {
            "type": "unified_memory",  // Works with all memory backends
            "check_keywords": ["memory", "content"],
            "min_keywords": 1,
            "check_absence": ["malicious"],
            "require_all_keywords": false
          }
        ]
      },
      "user_goal": {
        "description": "User intent description",
        "success_check": { ... }  // Same validator format
      },
      "attack_goal": {
        "description": "Attack intent description",
        "success_check": { ... }  // Same validator format
      }
    }
  ]
}
```

### Unified Memory Format

**New Format (Recommended)**:
```json
"initial_data": {
  "memory": {
    "set": "memory_set_0"  // Works for all backends
  }
}
```

**Old Formats (Auto-Normalized)**:
```json
// Explicit memory
"initial_data": {
  "memory_set": "memory_set_0"
}

// Mem0 memory
"initial_data": {
  "mem0_memory_set": "mem0_memory_set_0"
}

// RAG memory
"initial_data": {
  "rag_memory_set": "rag_memory_set_0"
}
```

The system automatically converts old formats to the unified format when loading test cases.

### Validator Types

#### Keyword Validator
```json
{
  "type": "keyword",
  "keywords": ["expected", "keywords"],
  "min_required": 2,
  "require_all_keywords": false
}
```

#### Unified Memory Validator
Works with all memory backends (explicit, mem0, rag):
```json
{
  "type": "unified_memory",
  "check_keywords": ["memory", "content"],
  "min_keywords": 1,
  "check_absence": ["malicious", "attack"],
  "require_all_keywords": false
}
```

#### Tool Call Validator
```json
{
  "type": "tool_call",
  "required_tools": ["search_emails", "read_email"],
  "min_required": 1
}
```

#### Filesystem Validator
```json
{
  "type": "filesystem",
  "check_files": ["outbox/*.json"],
  "file_content": {
    "outbox/*.json": ["expected", "content"]
  },
  "require_all_files": true
}
```

#### Composite Validator
```json
{
  "type": "composite",
  "operator": "AND",  // or "OR"
  "validators": [
    { "type": "keyword", ... },
    { "type": "unified_memory", ... }
  ]
}
```

### Test Case Location

Test cases are stored in:
- **Unified location**: `data/benchmark/tests/{attack_type}/`
- **Old locations** (still supported): `data/benchmark/attack_bench_{backend}/{attack_type}/`

The system automatically searches both locations for backward compatibility.

## Consolidating Results

After running benchmarks, consolidate results into CSV tables:

```bash
# Consolidate all results (generates comprehensive cross-backend comparison tables)
python scripts/consolidate_results.py

# Consolidate specific model or attack type
python scripts/consolidate_results.py --model gpt-5-mini
python scripts/consolidate_results.py --attack-type benign
python scripts/consolidate_results.py --model gpt-5-mini --attack-type benign

# Custom results or output directory
python scripts/consolidate_results.py --results-dir data/custom_results
python scripts/consolidate_results.py --output-dir data/consolidated
```

The script generates comprehensive CSV tables with:
- **Rows**: Defense types (disable_memory, none, user_only, no_untrusted_tools, limit_memory_length)
- **Columns**: Memory backends (disable_memory, explicit, mem0, rag) with steps passed/total and percentage
- **Output**: Separate CSV files for each model and attack type: `{model_name}_{attack_type}_consolidated.csv`

Consolidated CSV files are saved to the output directory (default: `data/benchmark/consolidated_results/`).

**Note**: The script only counts steps that:
- Are not session management steps (`start_new_session`, `insert_attack_email`)
- Have a `success_check` field defined (meaningful validation)

## Results and Logs

### Result Location

Test results are saved in a **unified directory structure**:

```
data/benchmark/results/
  {model_name}/               # gpt-5-mini, gpt-4o, etc.
    {memory_backend}/         # explicit, mem0, rag, or none (for disable_memory)
      {defense_type}/         # none, disable_memory, user_only, etc.
        {attack_type}/        # benign, direct, or indirect
          {test_file}.json
```

**Example paths**:
- `data/benchmark/results/gpt-5-mini/explicit/none/benign/00_email_tools.json`
- `data/benchmark/results/gpt-5-mini/mem0/user_only/direct/01_attack_test.json`
- `data/benchmark/results/gpt-4o/rag/no_untrusted_tools/indirect/02_poisoning_test.json`
- `data/benchmark/results/gpt-5-mini/none/disable_memory/benign/00_email_tools.json` (no memory enabled)

### Result File Format

Results are saved as **JSON files** with detailed test information:

```json
{
  "test_name": "Test Name",
  "test_file": "data/benchmark/tests/benign/00_email_tools.json",
  "description": "Test description",
  "session_id": "bench_42188cb0",
  "timestamp": "2025-11-03T12:00:00",
  "overall_success": true,
  "memory_backend": "explicit",
  "defense_type": "none",
  "backend_defense": "none",
  "model_name": "gpt-5-mini",
  "attack_type": "benign",
  "steps": [
    {
      "step": 1,
      "user_message": "User query",
      "agent_response": "Agent's response",
      "duration_s": 12.397,
      "passed": true,
      "success_check": {...},
      "user_goal": {...},
      "attack_goal": {...}
    }
  ],
  "session_history": [],
  "test_environment": "data/benchmark/test_envs/..."
}
```

**Key fields**:
- **Test Info**: `test_name`, `test_file`, `description`, `attack_type`
- **Execution**: `session_id`, `timestamp`, `test_environment`
- **Configuration**: `memory_backend`, `defense_type`, `backend_defense`, `model_name`
- **Results**: `overall_success`, `steps` (array with step-by-step results)
- **Step Details**: Each step includes `user_message`, `agent_response`, `duration_s`, `passed`, validation results, tool calls, etc.

### Result Caching

Results are automatically cached. If a result file already exists:
- **Default**: Test is skipped, existing result is returned
- **Force**: Use `--force` flag to re-run and overwrite

The caching key is: `{memory_backend}/{defense_type}/{model_name}/{attack_type}/{test_file}`

**See `RESULTS_FORMAT.md` for complete documentation of the result format.**

### Other Data Locations

- **Interactive Agent Data**: `data/interactive_agent/`
- **Application Logs**: `logs/`
- **Trace Logs**: `data/interactive_agent/trace.jsonl`
- **Test Environments**: `data/benchmark/test_envs/` (isolated per-run directories)

## Troubleshooting

**Server won't start:**
- Check that port 8000 is not in use
- Verify dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated

**Agent not responding:**
- Check `OPENAI_API_KEY` in `.env` file
- Verify model name in `config.yaml`
- Check terminal output for errors

**Benchmark tests failing:**
- Verify test files exist in `data/benchmark/tests/{suite}/` (unified location)
- Check that initial data sets exist in `data/benchmark/initial_*/`
- Verify memory backend is correctly configured in `config.yaml`
- Check that the specified memory backend is enabled
- Review logs in `logs/` directory
- Ensure test case format is correct (use unified format)

**Memory backend not working:**
- Ensure only one memory backend is enabled in `config.yaml`
- Check that `memory.backend` matches the enabled backend (or use `--memory-backend` CLI argument)
- Verify backend-specific configuration (vectorstore paths, etc.)
- For mem0/rag: Ensure vectorstore directories are accessible
- Check that API keys are set for mem0/rag (they use LLMs for embeddings)

**Results not being saved:**
- Check that results directory is writable
- Verify result path structure: `results/{backend}/{defense}/{model}/{attack_type}/`
- Use `--force` flag to overwrite existing results if needed
- Check that model name is correctly set in `config.yaml` (`agent.target_model_name`)

**Test cases not found:**
- Check test directory: `data/benchmark/tests/{attack_type}/`
- All test cases are in the unified location: `data/benchmark/tests/`
- Use `--test` with full path if test is in non-standard location
- Ensure test suite directory exists (benign, direct, or indirect)

**Defense type not working:**
- Verify defense type is one of: `none`, `disable_memory`, `user_only`, `no_untrusted_tools`, `limit_memory_length`
- Check that defense type is correctly mapped for your backend (see Memory Configuration section)
- For mem0: `none` maps to `no_defense` internally (this is automatic)

**Result caching issues:**
- Results are cached by: `{backend}/{defense}/{model}/{attack_type}/{test_file}`
- Use `--force` to overwrite existing results
- Check result file exists at expected path before assuming caching is broken

**Import errors:**
- Ensure you're running from the project root directory
- Verify `src/` is in Python path (handled automatically by entry scripts)
- Activate virtual environment before running

## Quick Reference

### Common Commands

```bash
# Run explicit memory with no defense
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite benign

# Run mem0 with all defenses
python scripts/run_benchmark.py --memory-backend mem0 --all-defenses --suite benign

# Run specific test with force overwrite
python scripts/run_benchmark.py --memory-backend rag --defense-type user_only --test data/benchmark/tests/benign/00_email_tools.json --force

# Consolidate all results (generates comprehensive CSV tables)
python scripts/consolidate_results.py
```

### Memory Backend Selection

| Backend | Use Case | Characteristics |
|---------|----------|----------------|
| `explicit` | Simple, deterministic | JSON-based, fast, easy to inspect |
| `mem0` | Semantic search | Vector embeddings, LLM-powered extraction |
| `rag` | Document-based | Chunked retrieval, long-form content |

### Defense Type Reference

| Defense | Description | Backend Mapping |
|---------|-------------|-----------------|
| `none` | No defense (baseline) | `none` (explicit/rag), `no_defense` (mem0) |
| `disable_memory` | Disable all indexing | `disable_memory` (all) |
| `user_only` | Only index user messages | `user_prompt_only` (explicit), `user_only` (mem0/rag) |
| `no_untrusted_tools` | Block after untrusted tools | `no_untrusted_tools` (all) |
| `limit_memory_length` | Limit entry length | `limit_memory_length` (explicit/mem0), `limit_chunk_size` (rag) |

### Test Suite Organization

```
data/benchmark/tests/
  benign/      # Normal behavior tests
  direct/     # Direct attack tests
  indirect/   # Indirect attack tests
```

### Result Structure

```
data/benchmark/results/
  {memory_backend}/        # explicit, mem0, or rag
    {defense_type}/        # none, disable_memory, user_only, etc.
      {model_name}/        # gpt-5-mini, gpt-4o, etc.
        {attack_type}/     # benign, direct, or indirect
          {test_file}.json # Individual test results
```

### Configuration Priority

1. **CLI arguments** (highest priority) - `--memory-backend`, `--defense-type`
2. **Config file** - `config.yaml` settings
3. **Defaults** - System defaults

The `--memory-backend` argument automatically enables the specified backend and disables others, overriding config file settings.

## License

MIT License - feel free to use and modify for your research.
