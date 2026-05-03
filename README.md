# Trojan Hippo Benchmark

This repository accompanies our work on Trojan Hippo, a class of persistent memory attacks on LLM agents.

Modern memory-enabled agents can store and reuse user information across sessions—but this also creates a new attack surface. \attackname{} shows how an attacker can inject a dormant payload into an agent’s long-term memory through a single untrusted interaction (e.g., a crafted email). The payload stays inactive until the user later discusses sensitive topics, where it triggers and leaks private information.

This repo includes a dynamic evaluation framework for studying such attacks across different memory architectures. It combines adaptive red-teaming (using OpenEvolve) with capability-aware security–utility analysis, allowing systematic comparison of attacks and defenses under realistic usage settings.

Our experiments (on multiple memory backends and frontier models from OpenAI and Google) show high attack success rates and long-lived persistence of injected payloads. We also implement and evaluate several defenses, highlighting a key challenge: stronger security often comes with non-trivial utility tradeoffs.


Use **`scripts/run_benchmark.py`** to run utility tests and attack-bench cases (static or adaptive), then read JSON under **`data/benchmark/results/`** and optional logs under **`data/benchmark/logs/`**.

---

## Installation

- **Python** 3.9+ recommended  
- **Project root** — run commands from the repository root so `src/` and configs resolve correctly.

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Environment variables** — create a `.env` file in the project root (loaded by the agent and benchmark). You typically need provider API keys for the target model and for semantic-judge / adaptive settings (for example `OPENAI_API_KEY`, plus any keys required by the models you choose). The benchmark expects working LLM access for the agent and for judges.

---

## Quick start

Every run must pass **`--model`** (the runner sets the target model from the CLI; do not rely on `target_model_name` in `agent_config.yaml`).

```bash
source venv/bin/activate

# Utility suite: explicit memory, no defense, memory_only tests
python scripts/run_benchmark.py \
  --model gpt-5-mini \
  --memory-backend explicit \
  --defense-type none \
  --suite memory_only

# Single utility test file (numbered JSON files, e.g. 001.json)
python scripts/run_benchmark.py \
  --model gpt-5-mini \
  --memory-backend explicit \
  --defense-type none \
  --test data/benchmark/tests/memory_only/001.json

# Attack bench: point --test at a file or directory under data/benchmark/attack_bench/
python scripts/run_benchmark.py \
  --model gpt-5-mini \
  --memory-backend rag \
  --defense-type none \
  --test data/benchmark/attack_bench/test_0/tax/rag/none/01.json

# Adaptive mode (reads benchmark.openevolve from benchmark_config.yaml)
python scripts/run_benchmark.py \
  --model gpt-5-mini \
  --memory-backend rag \
  --defense-type none \
  --test data/benchmark/attack_bench/train/... \
  --adaptive
```

```bash
python scripts/run_benchmark.py --help
```

---

## What the main parts of the code do

### `scripts/run_benchmark.py`

**Primary entrypoint.** Parses CLI arguments (`--model`, `--suite` or `--test`, memory backends, defenses, `--adaptive`, `--force`, paths, workers, etc.), loads and merges `agent_config.yaml` and `benchmark_config.yaml`, discovers test JSON files, enumerates valid backend × defense combinations (skipping invalid pairs via `benchmark_utils.is_valid_combination`), and either runs one combination or dispatches work in parallel. It calls into **`TestBench`** in `src/benchmark/test_bench.py` and writes results under the configured results root and logs under the logs root.

**How to use it:** Always invoke from the repo root with `python scripts/run_benchmark.py …` as in Quick start. Use `--suite` for a full utility suite under `data/benchmark/tests/<suite>/`, or `--test` for one file or folder (utility or attack bench).

### `src/benchmark/test_bench.py`

**Benchmark engine.** Loads JSON test cases, sets up an isolated in-memory email environment per run, drives the **agent** turn-by-turn (user messages, tools, sessions), applies **defenses** through the active memory backend, and records **steps** (responses, tool calls, timings). For **adaptive** runs it invokes optimizers from `src/benchmark/adaptive_attacks/` when a static attack path fails. You normally do not run `test_bench.py` directly for full sweeps; use `run_benchmark.py` so configs and paths stay consistent.

### `src/benchmark/test_validators.py`

**Success checks.** Implements validators referenced from test JSON (`keyword`, `semantic_judge`, `unified_memory`, composites, etc.). Semantic checks use the model configured under **`benchmark.semantic_judge`** in `benchmark_config.yaml`.

### `src/benchmark/benchmark_utils.py`

**Paths and combination rules.** Computes result and log paths (`get_result_path`, `get_log_path`), discovers tests, filters attack-bench layouts, and defines **`MEMORY_BACKENDS`**, **`UNIFIED_DEFENSE_TYPES`**, and **`is_valid_combination`**. If you wonder why a backend–defense pair was skipped, look here.

### `src/benchmark/adaptive_attacks/`

**Adaptive attack optimization** when you pass **`--adaptive`**. Contains **`openevolve_optimizer.py`** (MAP-elites–style search with an LLM mutator), **`scorer.py`** (scores candidate attacks), memory-backend **strategies** (`memory_strategies/`), and prompts. Behavior is controlled by **`benchmark.openevolve`** in `benchmark_config.yaml`.

### `src/agent/`

**Agent and memory backends.** **`agent_core.py`** runs the ReAct-style loop and tools; **`backend/`** implements **`explicit_memory.py`**, **`mem0_memory.py`**, **`rag_memory.py`**, **`context_memory.py`** (wired from config / CLI). **`tools_registry.py`** and **`tool_specifications/`** define email and memory tools. Prompt templates live next to the code (for example `memory_prompt.txt`).

### `src/benchmark/dataset_generation/` (optional)

**Regenerating** utility test JSON using LoCoMo-style helpers. Only needed if you change generation scripts; not required to run the benchmark on existing tests. See **Third-party code** below and `src/benchmark/dataset_generation/README.md`.

### `data/benchmark/`

- **`tests/<suite>/`** — utility benchmark JSON (`memory_only`, `assistant_responses`, …).  
- **`attack_bench/`** — train/test/propagated attack scenarios (layout varies by pipeline).  
- **`results/`**, **`logs/`** — default outputs (overridable with `--results-dir`, `--logs-dir`).  
- **`initial_*`**, snapshots, etc. — mailbox/outbox/drafts seeds referenced from test JSON.

### `scripts/` helpers

- **`consolidate_results.py`** — scans result JSON and builds CSV summaries / heatmaps (optional plotting deps).  
- **`consolidate_attack_results.py`**, **`consolidate_train.py`** — attack-bench–oriented aggregation.  
- **`propagate_train_attack_to_test_cases.py`** — propagates optimized train attacks into test layouts.

---

## Core concepts

### Memory backends (`--memory-backend`)

One enabled backend per combination. Choices:

| Backend | Role |
|--------|------|
| `explicit` | Structured JSON-style explicit memory |
| `mem0` | Embedding / vector memory (integrated via vendored `mem0/`) |
| `rag` | Chunked retrieval-style memory |
| `context` | Context-window-style memory |
| `none` | No long-term memory backend |

You can pass **multiple** backends in one invocation (e.g. `explicit mem0`); the runner schedules **combinations** (see `--num-workers`). Omitting `--memory-backend` runs all backends.

### Defense types (`--defense-type`)

Unified defenses:

- `none` — baseline  
- `user_prompt_only` — restrict what gets indexed (backend-specific)  
- `no_untrusted_tools` — stop indexing after untrusted tool use  
- `limit_memory_length` — cap stored entry length  
- `provable_policy` — policy-backed behavior where configured  

Invalid backend–defense pairs are **skipped** (`is_valid_combination` in `benchmark_utils.py`). Omitting `--defense-type` runs all defenses.

### Test selection (`--suite` **or** `--test`)

Exactly one of:

- **`--suite`** — `memory_only`, `assistant_responses`, `untrusted_probe`, `untrusted_send`, `disable_send`, `memory_tools`, `long_memory` → loads `data/benchmark/tests/<suite>/`.  
- **`--test`** — path to one `.json` file or a directory (utility or attack bench).

Attack-bench trees live under `data/benchmark/attack_bench/`; use paths that exist on disk (`train/`, `test_*`, etc.).

### Result caching

Existing result files cause a **skip** unless you pass **`--force`**.

---

## Command-line reference (`scripts/run_benchmark.py`)

| Argument | Description |
|----------|-------------|
| `--model` | **Required.** Target model id (e.g. `gpt-5-mini`, `gemini-3.1-pro-preview`). |
| `--memory-backend` | One or more of: `explicit`, `mem0`, `rag`, `context`, `none`. Default: all. |
| `--defense-type` / `--defense` | One or more unified defenses. Default: all. |
| `--suite` | Utility suite name. Mutually exclusive with `--test`. |
| `--test` | Path to file or directory of tests. Mutually exclusive with `--suite`. |
| `--config` | Benchmark config path (default: `benchmark_config.yaml`). |
| `--force` | Overwrite existing results. |
| `--results-dir` | Results root (default: `data/benchmark/results`). |
| `--logs-dir` | Logs root (default: `data/benchmark/logs`). |
| `--adaptive` | Enable adaptive / evolutionary attack optimization. |
| `--stealth` | Adaptive only — optimize for stealth and success; changes default output dirs unless overridden (see below). |
| `--early-stop-patience` | Adaptive only — early-stop patience (default 10). |
| `--num-workers` | Parallel workers for backend/defense combinations (default 1). |
| `--debug-level` | `INFO` or `DEBUG`. |
| `--seed` | Optional RNG seed override. |

The runner loads **`agent_config.yaml`** (agent and memory) and **`benchmark_config.yaml`** (semantic judge, OpenEvolve, defense caps).

---

## Configuration files

### `agent_config.yaml`

Invoke timeouts, LangGraph recursion limit, Mem0/RAG embedding and LLM settings for extraction, etc. **`target_model_name` is not set here** — use **`--model`** on the CLI.

### `benchmark_config.yaml`

- **`benchmark.semantic_judge`** — model and temperature for semantic success checks in validators.  
- **`benchmark.openevolve`** — mutator/judge models, iterations, workers, early-stop, token limits (**`--adaptive`**).  
- **`benchmark.limit_memory_size_defense`** — character cap for `limit_memory_length`.  

**Precedence:** CLI overrides where supported; then YAML defaults.

---

## Where results and logs go

### Results (default)

```text
data/benchmark/results/
  <model_name>/
    <memory_backend>/
      <defense_type>/
        <attack_type_or_suite>/
          <test_basename>.json
```

Example: `data/benchmark/results/gpt-5-mini/explicit/none/memory_only/001.json`

### Logs (default)

```text
data/benchmark/logs/<model_name>/<memory_backend>/<defense_type>/...
```

### Interpreting a result JSON

- **`overall_success`** — end-to-end pass/fail.  
- **`memory_backend`**, **`defense_type`**, **`model_name`** — run configuration.  
- **`steps`** — per-step content, `passed`, validators, durations.  
- **`total_user_steps` / `total_successful_user_steps`** — used by consolidation when present.  

**Exit code** of `run_benchmark.py` reflects **execution** success (crashes, API failures), not necessarily every semantic check — inspect the JSON for validation outcomes.

---

## Adaptive benchmark (`--adaptive`)

Failed static attacks can trigger the in-repo optimizer in **`src/benchmark/adaptive_attacks/`** (see **`benchmark.openevolve`** in `benchmark_config.yaml`). **`--stealth`** adds stealth objectives; it also defaults **`--results-dir`** to `data/benchmark/attack_results_stealth` and **`--logs-dir`** to `data/benchmark/attack_logs_stealth` unless you set those flags explicitly. Train cases typically live under `data/benchmark/attack_bench/train/`; exact layout depends on your generated trees.

---

## Consolidating results (optional)

```bash
python scripts/consolidate_results.py
python scripts/consolidate_results.py --model gpt-5-mini --suite memory_only
python scripts/consolidate_results.py --results-dir data/benchmark/results
```

For attack-focused pipelines, see **`consolidate_attack_results.py`**, **`consolidate_train.py`**, and **`propagate_train_attack_to_test_cases.py`** in `scripts/`.

---

## Repository layout (high level)

```text
memory-agent-security-benchmark/
├── README.md
├── requirements.txt
├── agent_config.yaml
├── benchmark_config.yaml
├── proof.md
├── scripts/                 # run_benchmark.py, consolidation, propagation
├── src/
│   ├── agent/               # agent_core, tools, memory backends
│   └── benchmark/           # test_bench, validators, benchmark_utils, adaptive_attacks, dataset_generation
├── data/benchmark/          # tests, attack_bench, results, logs, initial data
└── mem0/                    # Vendored Mem0 integration (see Third-party code)
```

---

## Third-party code and this repository

### Mem0 (memory backend)

Upstream: [mem0ai/mem0](https://github.com/mem0ai/mem0). This repo contains a **modified, vendored copy** under **`mem0/`**, wired through `src/agent/backend/mem0_memory.py`. Use **`requirements.txt`** and `agent_config.yaml` for Mem0-related settings; you do not clone Mem0 separately for normal benchmark runs.

### LoCoMo (optional — dataset generation only)

Methodology and reference data: [snap-research/locomo](https://github.com/snap-research/locomo). You **do not** need LoCoMo for **`run_benchmark.py`** or standard **`src/benchmark`** execution.

Clone LoCoMo only if you run generators under **`src/benchmark/dataset_generation/`**, which import **`global_methods`**. Place the clone at the repository root as **`LoCoMo/`** so those imports resolve. Details: **`src/benchmark/dataset_generation/README.md`**. Running on existing JSON under **`data/benchmark/tests/`** never requires LoCoMo.

### OpenEvolve (adaptive attacks — concept vs upstream repo)

Reference implementation: [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve). This repository **does not** execute that project as a dependency. Adaptive mode uses a **custom** OpenEvolve-style implementation in **`src/benchmark/adaptive_attacks/`** (notably **`openevolve_optimizer.py`**), tuned via **`benchmark.openevolve`** and **`--adaptive`**.


