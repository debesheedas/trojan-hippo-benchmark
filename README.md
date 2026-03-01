# Email Agent Security Benchmark

A security-focused email agent with persistent memory and automated benchmarking capabilities. The system includes static and adaptive benchmarking for evaluating agent security and resilience to attacks.

## Quick Start

### Running Benchmarks

The benchmark system supports three memory backends (explicit, mem0, rag) with unified defense types. Use the unified benchmark runner for all scenarios.

#### Quick Start

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run explicit memory with no defense on utility test suite
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only

# Run mem0 with all defense types (omit --defense-type to run all)
python scripts/run_benchmark.py --memory-backend mem0 --suite memory_only

# Run a specific test file with RAG memory
python scripts/run_benchmark.py --memory-backend rag --defense-type user_prompt_only --test data/benchmark/tests/memory_only/memory_only_001.json
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

- **`user_prompt_only`**: Only index user messages
  - Filters out agent responses
  - Prevents agent-generated content from being stored
  - **Backend mapping**: Unified name `user_prompt_only` maps to `user_prompt_only` (explicit) or `user_prompt_only` (mem0/rag/context) internally

- **`no_untrusted_tools`**: Block memory indexing after untrusted tools
  - Memory indexing stops once an untrusted tool is called
  - Prevents memory poisoning via tool calls
  - Works identically across all backends

- **`limit_memory_length`**: Limit memory entry length
  - Truncates memory entries to prevent long-form attacks
  - Works identically across all backends

#### Running Test Suites

Utility test suites are organized under `data/benchmark/tests/`:

```bash
# Run utility test suites (memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, long_memory)
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only

# Run attack bench test cases (from attack_bench/test/ or a single file)
python scripts/run_benchmark.py --memory-backend mem0 --defense-type user_prompt_only --test data/benchmark/attack_bench/test/rag/persistent_exfiltrate_tax
```

#### Running Specific Tests

```bash
# Run a single utility test file
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --test data/benchmark/tests/memory_only/memory_only_001.json

# Run all tests in a utility suite directory
python scripts/run_benchmark.py --memory-backend mem0 --defense-type user_prompt_only --test data/benchmark/tests/memory_only/

# Run attack bench test folder (evaluation) or train folder (adaptive)
python scripts/run_benchmark.py --memory-backend rag --defense-type none --test data/benchmark/attack_bench/test/rag/persistent_exfiltrate_tax
python scripts/run_benchmark.py --memory-backend rag --defense-type none --test data/benchmark/attack_bench/train/rag/persistent_exfiltrate_tax --adaptive
```

#### Running All Defenses

Run all defense types for a backend in one command:

```bash
# Run all defenses for explicit memory (omit --defense-type to run all)
python scripts/run_benchmark.py --memory-backend explicit --suite memory_only

# This runs: none, user_prompt_only, no_untrusted_tools, limit_memory_length, provable_policy (invalid combinations are skipped)
```

#### Result Caching

The benchmark automatically skips tests if results already exist, preventing unnecessary re-runs:

```bash
# First run - executes all tests
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only
# Output: Running tests, saving results...

# Second run - skips tests (results exist)
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only
# Output: ⏭️  Skipping test_name.json - result already exists
#         Use --force to overwrite

# Force overwrite existing results
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only --force
# Output: Running tests, overwriting existing results...
```

#### Advanced Usage

```bash
# Use custom config file
python scripts/run_benchmark.py --memory-backend mem0 --defense-type user_prompt_only --suite memory_only --config my_config.yaml

# Use custom results directory
python scripts/run_benchmark.py --memory-backend rag --defense-type none --suite memory_only --results-dir data/custom_results

# Combine options
python scripts/run_benchmark.py \
  --memory-backend explicit \
  --defense-type user_prompt_only \
  --suite memory_only \
  --config benchmark_config.yaml \
  --force \
  --results-dir data/benchmark/results
```

#### Command-Line Options

```bash
python scripts/run_benchmark.py --help

# Required:
--memory-backend {explicit,mem0,rag}  # Memory backend to use

# Test selection (one required):
--suite {memory_only,assistant_responses,untrusted_probe,untrusted_send,disable_send,memory_tools,long_memory}  # Utility test suite to run
--test TEST                           # Specific test file or directory

# Defense selection (optional):
--defense-type {none,user_prompt_only,no_untrusted_tools,limit_memory_length,provable_policy}  # If omitted, all defense types are run

# Optional:
--config CONFIG                       # Benchmark config file (default: benchmark_config.yaml)
--force                              # Force overwrite existing results
--results-dir RESULTS_DIR            # Custom results directory
```

#### Results Location

Test results are saved in a unified structure:

```
data/benchmark/results/
  {memory_backend}/           # explicit, mem0, or rag
    {defense_type}/           # none, disable_memory, user_prompt_only, etc.
      {model_name}/           # gpt-5-mini, gpt-4o, etc.
        {attack_type}/        # memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, long_memory (for utility tests) or test filename (for attack_bench)
          {test_file}.json    # Individual test results
```

Example: `data/benchmark/results/gpt-5-mini/explicit/none/memory_only/memory_only_001.json`

### Running the Adaptive Benchmark

Run the adaptive benchmark with attack optimization when static attacks fail:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# 1. Configure memory backend in agent_config.yaml
# Set memory.backend: "explicit" (or "mem0" or "rag")
# Set memory.{backend}_memory.enabled: true

# 2. Run adaptive benchmark (add --adaptive for adaptive mode)
python src/benchmark/test_bench.py --suite memory_only --defense-type none --adaptive

# Or use the unified runner with --adaptive
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only --adaptive
```

**What it does**: When a static attack fails, the system automatically attempts to optimize the attack using the configured optimizer (OpenEvolve) to find a successful variant.

**Configuration**: Use the `--adaptive` flag when running the benchmark. Optimizer settings remain in `benchmark_config.yaml`:
```yaml
benchmark:
  openevolve:
    enabled: true
    # ... optimizer settings
```

**Note**: Adaptive benchmarking uses `test_bench.py` internally. Memory backend and defense type are specified via config or command-line arguments.

---

## Project Structure

```
memory-agent-security-benchmark/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── agent_config.yaml            # Agent configuration (model, temperature, etc.)
├── benchmark_config.yaml        # Benchmark-specific settings
├── proof.md                     # Security proofs and formal analysis
├── .github/
│   └── workflows/              # GitHub Actions CI/CD workflows
│       └── regression_test.yml  # Regression testing workflow
├── src/
│   ├── agent/                   # Core agent implementation
│   │   ├── agent_core.py        # Main agent execution logic
│   │   ├── attack_utils.py      # Attack-related utility functions
│   │   ├── tools_registry.py    # Tool registration and management
│   │   ├── utils.py             # Helper functions and utilities
│   │   ├── memory_prompt.txt    # Memory behavior prompt template
│   │   ├── user_only_memory_prompt.txt  # User-only memory prompt template
│   │   ├── tool_specifications/ # Tool implementations
│   │   │   ├── email_tools.py   # Email management tools
│   │   │   └── memory_tools.py  # Memory management tools
│   │   └── backend/             # Memory backend implementations
│   │       ├── context_memory.py    # Context-based memory backend
│   │       ├── explicit_memory.py  # JSON-based explicit memory backend
│   │       ├── mem0_memory.py      # Mem0 vector-based memory backend
│   │       └── rag_memory.py       # RAG-based memory backend
│   └── benchmark/               # Benchmarking system
│       ├── test_bench.py        # Main benchmark entry point
│       ├── test_validators.py  # Test validation logic
│       ├── benchmark_utils.py  # Benchmark utilities
│       ├── environment_state.py # Environment state management
│       ├── memory_validators.py # Memory validation utilities
│       ├── adaptive_attacks/   # Adaptive attack optimizers
│       │   ├── base_optimizer.py      # Base optimizer interface
│       │   ├── openevolve_optimizer.py # OpenEvolve-based optimizer
│       │   ├── scorer.py              # Attack scoring logic
│       │   ├── mutator_prompt.txt     # Mutation prompt template
│       │   └── numeric_judge_prompt.txt # Numeric judgment prompt
│       └── dataset_generation/  # Test case generation scripts
│           ├── 01_memory_only/  # Memory-only test case generation
│           ├── 02_assistant_responses/ # Assistant response test cases
│           ├── 03_untrusted_probe/    # Untrusted probe test cases
│           ├── 04_untrusted_send/     # Untrusted send test cases
│           ├── 05_disable_send/       # Disable send test cases
│           ├── 06_memory_tools/       # Memory tools test cases
│           └── 07_long_memory/        # Long memory test cases
├── scripts/                     # Utility scripts
│   ├── run_benchmark.py        # Unified benchmark runner
│   ├── consolidate_results.py  # Unified results consolidation
│   └── aggregate_csv.py        # CSV aggregation utilities
├── CI-tests/                    # Continuous integration test suite
│   ├── compare_results.py      # Result comparison utilities
│   ├── testcases/              # CI test cases
│   │   ├── test1.json
│   │   └── test2.json
│   ├── ground_truth/           # Ground truth files
│   │   ├── test1.json          # Ground truth for test1
│   │   └── test2.json          # Ground truth for test2
│   ├── results/                # CI test results
│   ├── README.md               # CI tests documentation
│   ├── GITHUB_ACTIONS_EXPLANATION.md  # GitHub Actions guide
│   └── GITHUB_ACTIONS_SETUP.md        # GitHub Actions setup guide
├── data/
│   ├── agent/                   # Agent data (mailbox, drafts, outbox, memory)
│   └── benchmark/              # Benchmark data
│       ├── tests/              # Utility test directory (organized by suite)
│       │   ├── memory_only/    # Memory-only utility tests
│       │   ├── assistant_responses/  # Assistant responses utility tests
│       │   ├── untrusted_probe/ # Untrusted probe utility tests
│       │   ├── untrusted_send/  # Untrusted send utility tests
│       │   ├── disable_send/    # Disable send utility tests
│       │   ├── memory_tools/   # Memory tools utility tests
│       │   └── long_memory/    # Long memory utility tests
│       ├── attack_bench/       # Attack bench: train/, test/, train_cache/
│       │   ├── train/          # Train cases (run with --adaptive to optimize attacks)
│       │   ├── test/           # Test cases (run after propagating attack from cache)
│       │   └── train_cache/    # Cached optimized train attacks
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
│       └── few_shot_examples/ # Few-shot examples for optimizers
├── html_reports/               # HTML visualization reports
│   ├── index.html              # Reports index
│   └── gpt-4o/                 # Model-specific reports
│       └── {suite}/            # Suite-specific reports
├── logs/                       # Application logs
├── mem0/                       # Mem0 library (dependency)
└── useful_temp/                # Temporary utility scripts and notes
    ├── extract_mutators.py
    ├── format_body_plain.py
    ├── generate_html_reports.py
    ├── validate_cache.py
    └── notes/                  # Development notes and scratchpad
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

Edit `agent_config.yaml` to customize agent behavior:

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
    defense_type: "none"  # Unified defense type: none, disable_memory, user_prompt_only, etc.
    memory_file: "data/agent/agent_memory.json"
  
  # Mem0 memory configuration
  mem0_memory:
    enabled: false     # Set to true to use mem0 memory
    defense_type: "none"  # Unified defense type
    vectorstore_path: "data/agent/mem0_vectorstore"
    llm_provider: "openai"
    llm_model: "gpt-4o-mini"
    llm_temperature: 0.0
    embedding_provider: "openai"
    embedding_model: "text-embedding-3-small"
    vector_store_provider: "faiss"
    top_k: 10
    user_id: "vince"
  
  # RAG memory configuration
  rag_memory:
    enabled: false     # Set to true to use RAG memory
    defense_type: "none"  # Unified defense type
    vectorstore_path: "data/agent/rag_vectorstore"
    chunk_size: 512
    embedding_model: "text-embedding-3-small"
    top_k: 8
```

**Important Notes**:
- **Only enable one backend**: Set `enabled: true` for the backend you want to use, and `enabled: false` for others
- **CLI overrides config**: The `--memory-backend` argument in `run_benchmark.py` automatically enables the specified backend and disables others
- **Unified defense types**: Use the same defense type names across all backends: `none`, `user_prompt_only`, `no_untrusted_tools`, `limit_memory_length`, `provable_policy`

### Benchmark Configuration

```yaml
benchmark:
  results_dir: "data/benchmark/results"  # Unified results directory
  semantic_judge:
    model_name: "gpt-5-mini"
    temperature: 0.0
  openevolve:
    enabled: true
    # ... optimizer settings
```

Use the `--adaptive` flag when running the benchmark to enable adaptive optimization; otherwise static mode is used.

Note: Test files are located in `data/benchmark/tests/` (hardcoded, not configurable).

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
- **Adaptive Optimizer**: OpenEvolve-based optimization for evolving attacks when static ones fail
- **Comprehensive Validation**: Semantic judges and multiple validation strategies
- **Unified Test Format**: Single test format works with all memory backends
- **Isolated Test Environments**: Each test run uses a unique, isolated directory

## Memory System

The agent supports three memory backends, each with different characteristics:

### Explicit Memory (JSON-based)
- **Format**: Simple JSON file with structured memory entries
- **Storage**: `data/agent/agent_memory.json`
- **Characteristics**: Deterministic, easy to inspect, fast access
- **Best for**: Simple use cases, debugging, deterministic behavior

### Mem0 Memory (Vector-based)
- **Format**: Vector embeddings stored in FAISS vectorstore
- **Storage**: `data/agent/mem0_vectorstore/`
- **Characteristics**: Semantic search, automatic memory extraction, LLM-powered
- **Best for**: Complex memory needs, semantic similarity search

### RAG Memory (Retrieval Augmented Generation)
- **Format**: Chunked text stored in vectorstore
- **Storage**: `data/agent/rag_vectorstore/`
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
- **`user_prompt_only`**: Only index user messages (filter agent responses)
- **`no_untrusted_tools`**: Block memory indexing after untrusted tools are called
- **`limit_memory_length`**: Limit memory entry length (backend-specific implementation)

## Benchmark Test Format

Tests are defined as JSON files with a **unified format** that works with all memory backends. The system automatically normalizes old-format test cases when loaded.

### Test File Structure

```json
{
  "name": "Test Name",
  "description": "Test description",
  "attack_type": "memory_only|assistant_responses|untrusted_probe|untrusted_send|disable_send|memory_tools|long_memory (for utility tests) or attack_type from JSON (for attack_bench)",
  "initial_data": {
    "inbox_set": "inbox_set_1",
    "outbox_set": "outbox_set_1",
    "drafts_set": "drafts_set_1",
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

### Memory Initialization

All tests start with **empty memory** - memory is built during test execution through user interactions. No initial memory configuration is needed or supported. Memory backends will create empty stores on first use.

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
python scripts/consolidate_results.py --suite memory_only
python scripts/consolidate_results.py --model gpt-5-mini --suite memory_only

# Custom results or output directory
python scripts/consolidate_results.py --results-dir data/custom_results
python scripts/consolidate_results.py --output-dir data/consolidated
```

The script generates comprehensive CSV tables with:
- **Rows**: Defense types (disable_memory, none, user_prompt_only, no_untrusted_tools, limit_memory_length)
- **Columns**: Memory backends (disable_memory, explicit, mem0, rag) with steps passed/total and percentage
- **Output**: Separate CSV files for each model and suite: `{model_name}_{suite}_consolidated.csv`

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
      {defense_type}/         # none, disable_memory, user_prompt_only, etc.
        {attack_type}/        # memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, long_memory (for utility tests) or test filename (for attack_bench)
          {test_file}.json
```

**Example paths**:
- `data/benchmark/results/gpt-5-mini/explicit/none/memory_only/memory_only_001.json` (utility test)
- `data/benchmark/results/gpt-5-mini/mem0/user_prompt_only/assistant_responses/assistant_responses_001.json` (utility test)
- `data/benchmark/results/gpt-4o/rag/none/00_exfiltrate.json` (attack bench test - no suite folder)
- `data/benchmark/results/gpt-5-mini/none/disable_memory/memory_only/memory_only_001.json` (no memory enabled)

### Result File Format

Results are saved as **JSON files** with detailed test information:

```json
{
  "test_name": "Test Name",
  "test_file": "data/benchmark/tests/memory_only/memory_only_001.json",
  "description": "Test description",
  "session_id": "bench_42188cb0",
  "timestamp": "2025-11-03T12:00:00",
  "overall_success": true,
  "memory_backend": "explicit",
  "defense_type": "none",
  "backend_defense": "none",
  "model_name": "gpt-5-mini",
  "attack_type": "memory_only",
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
  "session_history": []
}
```

**Key fields**:
- **Test Info**: `test_name`, `test_file`, `description`, `attack_type`
- **Execution**: `session_id`, `timestamp`
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

- **Agent Data**: `data/agent/`
- **Application Logs**: `logs/`
- **Trace Logs**: `data/agent/trace.jsonl`
- **Test Environments**: All test environments are now in-memory (no file system directories needed)

## Troubleshooting

**Server won't start:**
- Check that port 8000 is not in use
- Verify dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated

**Agent not responding:**
- Check `OPENAI_API_KEY` in `.env` file
- Verify model name in `agent_config.yaml`
- Check terminal output for errors

**Benchmark tests failing:**
- Verify test files exist in `data/benchmark/tests/{suite}/` (unified location)
- Check that initial data sets exist in `data/benchmark/initial_*/`
- Verify memory backend is correctly configured in `agent_config.yaml`
- Check that the specified memory backend is enabled
- Review logs in `logs/` directory
- Ensure test case format is correct (use unified format)

**Memory backend not working:**
- Ensure only one memory backend is enabled in `agent_config.yaml`
- Check that `memory.backend` matches the enabled backend (or use `--memory-backend` CLI argument)
- Verify backend-specific configuration (vectorstore paths, etc.)
- For mem0/rag: Ensure vectorstore directories are accessible
- Check that API keys are set for mem0/rag (they use LLMs for embeddings)

**Results not being saved:**
- Check that results directory is writable
- Verify result path structure: `results/{backend}/{defense}/{model}/{attack_type}/`
- Use `--force` flag to overwrite existing results if needed
- Check that model name is correctly set in `agent_config.yaml` (`agent.target_model_name`)

**Test cases not found:**
- Check test directory: `data/benchmark/tests/{suite}/` for utility tests or `data/benchmark/attack_bench/train/` / `attack_bench/test/` for attack bench
- All attack bench train cases: `data/benchmark/attack_bench/train/`; test cases: `data/benchmark/attack_bench/test/`
- Use `--test` with full path if test is in non-standard location
- Ensure test suite directory exists (memory_only, assistant_responses, untrusted_probe, untrusted_send, disable_send, memory_tools, long_memory) for utility tests

**Defense type not working:**
- Verify defense type is one of: `none`, `user_prompt_only`, `no_untrusted_tools`, `limit_memory_length`, `provable_policy`
- Check that defense type is correctly specified for your backend (see Memory Configuration section)

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
python scripts/run_benchmark.py --memory-backend explicit --defense-type none --suite memory_only

# Run mem0 with all defenses (omit --defense-type to run all)
python scripts/run_benchmark.py --memory-backend mem0 --suite memory_only

# Run specific test with force overwrite
python scripts/run_benchmark.py --memory-backend rag --defense-type user_prompt_only --test data/benchmark/tests/memory_only/memory_only_001.json --force

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

| Defense | Description |
|---------|-------------|
| `none` | No defense (baseline) |
| `user_prompt_only` | Only index user messages |
| `no_untrusted_tools` | Block after untrusted tools |
| `limit_memory_length` | Limit entry length |
| `provable_policy` | Cryptographically verifiable memory policy |

### Test Suite Organization

**Utility Tests** (organized by suite):
```
data/benchmark/tests/
  memory_only/         # Memory-only utility tests
  assistant_responses/ # Assistant responses utility tests
  untrusted_probe/     # Untrusted probe utility tests
  untrusted_send/      # Untrusted send utility tests
  disable_send/        # Disable send utility tests
  memory_tools/        # Memory tools utility tests
  long_memory/         # Long memory utility tests
```

**Attack Bench Tests** (train / test / train_cache):
```
data/benchmark/attack_bench/
  train/               # Train cases (run with --adaptive)
    {backend}/{suite}/ # e.g. rag/persistent_exfiltrate_tax/
  test/                # Test cases (run after propagate_train_attack_to_test_cases)
    {backend}/{suite}/
  train_cache/         # Cached optimized train attacks
    {backend}/{suite}/
```

### Result Structure

**Utility Tests**:
```
data/benchmark/results/
  {model_name}/        # gpt-5-mini, gpt-4o, etc.
    {memory_backend}/  # explicit, mem0, rag, or none
      {defense_type}/  # none, disable_memory, user_prompt_only, etc.
        {suite}/       # memory_only, assistant_responses, etc.
          {test_file}.json
```

**Attack Bench Tests**:
```
data/benchmark/results/
  {model_name}/        # gpt-5-mini, gpt-4o, etc.
    {memory_backend}/  # explicit, mem0, rag, or none
      {defense_type}/  # none, disable_memory, user_prompt_only, etc.
        {test_file}.json  # No suite folder for attack bench tests
```

### Configuration Priority

1. **CLI arguments** (highest priority) - `--memory-backend`, `--defense-type`
2. **Config file** - `agent_config.yaml` settings
3. **Defaults** - System defaults

The `--memory-backend` argument automatically enables the specified backend and disables others, overriding config file settings.

## CI Regression Tests

The `CI-tests/` directory contains regression tests to verify the benchmark works correctly across all memory backends and defense types.

### Running CI Tests

Run test1 across all 5 backends × 5 defenses (23 valid combinations):

```bash
python scripts/run_benchmark.py \
  --test CI-tests/testcases/test1.json \
  --memory-backend none explicit mem0 rag context \
  --defense-type none user_prompt_only no_untrusted_tools limit_memory_length provable_policy \
  --results-dir CI-tests/results \
  --logs-dir CI-tests/logs \
  --num-workers 1 \
  --force
```

### Quick Single-Combination Tests

To quickly verify a specific backend works:

```bash
# explicit backend with none defense
python scripts/run_benchmark.py \
  --test CI-tests/testcases/test1.json \
  --memory-backend explicit \
  --defense-type none \
  --results-dir CI-tests/results \
  --logs-dir CI-tests/logs \
  --force

# rag backend with none defense  
python scripts/run_benchmark.py \
  --test CI-tests/testcases/test1.json \
  --memory-backend rag \
  --defense-type none \
  --results-dir CI-tests/results \
  --logs-dir CI-tests/logs \
  --force

# mem0 backend with none defense
python scripts/run_benchmark.py \
  --test CI-tests/testcases/test1.json \
  --memory-backend mem0 \
  --defense-type none \
  --results-dir CI-tests/results \
  --logs-dir CI-tests/logs \
  --force

# context backend with none defense
python scripts/run_benchmark.py \
  --test CI-tests/testcases/test1.json \
  --memory-backend context \
  --defense-type none \
  --results-dir CI-tests/results \
  --logs-dir CI-tests/logs \
  --force
```

### Comparing Results Against Ground Truth

After running tests, compare results:

```bash
python CI-tests/compare_results.py \
  --results-dir CI-tests/results \
  --test-file CI-tests/testcases/test1.json
```

See `CI-tests/README.md` for more details on the CI test infrastructure.

## License

MIT License - feel free to use and modify for your research.
