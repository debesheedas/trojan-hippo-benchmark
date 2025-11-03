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

### Running the Static Benchmark

Run automated tests on the attack benchmark suite:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run all test suites
python src/benchmark/test_bench.py

# Or run a specific suite
python src/benchmark/test_bench.py --suite benign
python src/benchmark/test_bench.py --suite direct
python src/benchmark/test_bench.py --suite indirect

# Or run a specific test file
python src/benchmark/test_bench.py --test data/benchmark/attack_bench/benign/01_reply_to_alice.json
```

**Requirements**: Ensure `enable_adaptive_benchmark: false` in `config.yaml` (under `benchmark` section).

**Results**: Test results are saved to `data/benchmark/test_bench_results/{model_name}/{suite}/`.

### Running the Adaptive Benchmark

Run the adaptive benchmark with attack optimization when static attacks fail:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Enable adaptive mode in config.yaml
# Set benchmark.enable_adaptive_benchmark: true

# Run adaptive benchmark
python src/benchmark/test_bench.py --suite indirect
```

**What it does**: When a static attack fails, the system automatically attempts to optimize the attack using configured strategies (OpenEvolve or DSPy) to find a successful variant.

**Configuration**: Configure optimizers in `config.yaml` under `benchmark.openevolve` and `benchmark.dspy` sections.

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
├── data/
│   ├── interactive_agent/       # Interactive agent data
│   │   ├── mailbox/             # Inbox emails
│   │   ├── drafts/             # Draft emails
│   │   ├── outbox/             # Sent emails
│   │   ├── sessions/           # Session data
│   │   ├── agent_memory.json   # Persistent memory
│   │   └── trace.jsonl         # Interaction traces
│   └── benchmark/              # Benchmark data
│       ├── attack_bench/       # Test definitions
│       │   ├── benign/         # Benign behavior tests
│       │   ├── direct/        # Direct attack tests
│       │   └── indirect/       # Indirect attack tests
│       ├── initial_inbox/     # Initial inbox states
│       ├── initial_outbox/    # Initial outbox states
│       ├── initial_drafts/    # Initial draft states
│       ├── initial_memory/   # Initial memory states
│       ├── initial_sessions/ # Initial session states
│       ├── few_shot_examples/ # Few-shot examples for optimizers
│       ├── attack_bench_cache/ # Cached optimized attacks
│       └── test_bench_results/ # Benchmark results
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

### Benchmark Configuration

```yaml
benchmark:
  enable_adaptive_benchmark: false  # Enable adaptive optimization
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
- **Persistent Memory**: ChatGPT-style memory system with short-term and long-term storage
- **Tool-Based Architecture**: LangChain-based agent with structured tool calling
- **Session Management**: Multi-session support with isolated conversation histories

### Benchmarking System

- **Static Benchmarking**: Automated test execution on predefined attack scenarios
- **Adaptive Benchmarking**: Automatic attack optimization when static attacks fail
- **Multiple Optimizers**: Support for OpenEvolve and DSPy-based optimization strategies
- **Comprehensive Validation**: Semantic judges and multiple validation strategies

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

The agent maintains both short-term and long-term memory:

- **Short-Term Memory**: Last 15 messages in conversation (session-scoped)
- **Long-Term Memory**: Persistent facts stored in `data/interactive_agent/agent_memory.json` (persists across sessions)

The agent automatically detects "remember" and "forget" requests in conversation and updates memory accordingly.

## Benchmark Test Format

Tests are defined as JSON files with the following structure:

```json
{
  "name": "Test Name",
  "description": "Test description",
  "attack_type": "benign|direct|indirect",
  "initial_data": {
    "inbox_set": "inbox_set_1",
    "outbox_set": "outbox_set_1",
    "drafts_set": "drafts_set_1",
    "memory_set": "memory_set_0"
  },
  "steps": [
    {
      "user_message": "User query",
      "expected_behavior": "What should happen",
      "user_goal": {
        "description": "User intent",
        "success_check": { ... }
      },
      "attack_goal": {
        "description": "Attack intent",
        "success_check": { ... }
      }
    }
  ]
}
```

## Results and Logs

- **Benchmark Results**: `data/benchmark/test_bench_results/{model_name}/{suite}/`
- **Interactive Agent Data**: `data/interactive_agent/`
- **Application Logs**: `logs/`
- **Trace Logs**: `data/interactive_agent/trace.jsonl`

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
- Verify test files exist in `data/benchmark/attack_bench/`
- Check that initial data sets exist in `data/benchmark/initial_*/`
- Review logs in `logs/` directory

**Import errors:**
- Ensure you're running from the project root directory
- Verify `src/` is in Python path (handled automatically by entry scripts)
- Activate virtual environment before running

## License

MIT License - feel free to use and modify for your research.
