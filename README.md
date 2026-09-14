# Trojan Hippo: Weaponizing Agent Memory for Data Exfiltration

**Debeshee Das, Julien Piet, Darya Kaviani, Luca Beurer-Kellner, Florian Tramèr, David Wagner**

[arXiv:2605.01970](https://arxiv.org/abs/2605.01970) · [PDF](https://arxiv.org/pdf/2605.01970) · [Supplementary material](appendix.pdf)

This repository contains the benchmark, attack generator, adaptive red-teaming optimizer, defenses, and raw results from the paper.

## Abstract

Memory systems enable otherwise-stateless LLM agents to persist user information across sessions, but also introduce a new attack surface. We characterize the Trojan Hippo attack, a class of persistent memory attacks that operates in a more realistic threat model than prior memory poisoning work: the attacker plants a dormant payload into an agent's long-term memory via a single untrusted tool call (e.g., a crafted email), which activates only when the user later discusses sensitive topics such as finance, health, or identity, and exfiltrates high-value personal data to the attacker. While anecdotal demonstrations of such attacks have appeared against deployed systems, no prior work systematically evaluates them across heterogeneous memory architectures and defenses. We introduce a dynamic evaluation framework comprising two components: (1) an OpenEvolve-based adaptive red-teaming benchmark that stress-tests defenses and memory backends against continuously refined attacks, and (2) the first capability-aware security/utility analysis for persistent memory systems, enabling principled reasoning about defense deployment across different usage profiles. Instantiated on an email assistant across four memory backends (explicit tool memory, agentic memory, RAG, and sliding-window context), Trojan Hippo achieves up to 85-100% ASR against current frontier models from OpenAI and Google, with planted memories successfully activating even after 100 benign sessions. We evaluate four memory-system defenses inspired by basic security principles, finding they substantially reduce attack success rates (to as low as 0-5%), though at utility costs that vary widely with task requirements. Because of this substantial security-utility tradeoff, the effective real-world deployment of defenses remains an open challenge, which our evaluation framework is specifically designed to address.

## What is in this repository

| Component | Location |
|-----------|----------|
| Email agent with four memory backends and five defenses | `src/agent/` |
| Benchmark engine, validators, snapshot I/O | `src/benchmark/` |
| Adaptive (OpenEvolve-style) attack optimizer | `src/benchmark/adaptive_attacks/` |
| Attack bench: 5 topics × 5 backends × 5 defenses, persistence splits 0–100 sessions | `data/benchmark/attack_bench/` |
| Utility suites: 7 user flows × 4 tests | `data/benchmark/tests/` |
| Memory snapshots after N benign sessions (per backend and defense) | `data/benchmark/snapshots/` |
| Raw attack results for `gpt-5`, `gpt-5-mini`, `gemini-3.1-pro-preview` | `data/benchmark/attack_results/` |
| Consolidated CSVs and plots | `data/benchmark/consolidated_*` |
| Patched Mem0 1.0.1 (in-memory only) | `mem0/`, diff in `patches/` |

## Installation

Python 3.9+. Run all commands from the repository root.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repository root:

```bash
OPENAI_API_KEY=...   # OpenAI target models, Mem0 extraction, embeddings
GEMINI_API_KEY=...   # Gemini target models; default semantic judge and adaptive mutator/judge
```

The defaults in `benchmark_config.yaml` use Gemini for the semantic judge (`gemini-3-flash-preview`) and for the adaptive mutator and judge (`gemini-3.1-pro-preview`), so both keys are needed for a full run.

## Quick start

`--model` is required on every run.

```bash
# Utility suite: explicit memory, no defense
python scripts/run_benchmark.py --model gpt-5-mini \
  --memory-backend explicit --defense-type none --suite memory_only

# One utility test file
python scripts/run_benchmark.py --model gpt-5-mini \
  --memory-backend explicit --defense-type none \
  --test data/benchmark/tests/memory_only/001.json

# Attack bench, no benign sessions between planting and trigger (test_0)
python scripts/run_benchmark.py --model gpt-5-mini \
  --memory-backend rag --defense-type none \
  --test data/benchmark/attack_bench/test_0/tax/rag/none/01.json

# Attack bench, 100 benign sessions between planting and trigger (test_100)
python scripts/run_benchmark.py --model gpt-5-mini \
  --memory-backend rag --defense-type none \
  --test data/benchmark/attack_bench/test_100/tax/rag/none

# All backends × all defenses for one topic, 4 processes
python scripts/run_benchmark.py --model gpt-5-mini \
  --test data/benchmark/attack_bench/test_0/tax --num-workers 4

# Adaptive red-teaming on a train case
python scripts/run_benchmark.py --model gpt-5-mini \
  --memory-backend rag --defense-type none \
  --test data/benchmark/attack_bench/train_0/tax/rag/none/01.json --adaptive
```

`python scripts/run_benchmark.py --help` lists every option.

## Core concepts

### Memory backends (`--memory-backend`)

| Backend | Description |
|---------|-------------|
| `explicit` | Structured tool memory (agent calls save/recall tools) |
| `mem0` | Agentic memory: LLM fact extraction + vector store (patched Mem0, see below) |
| `rag` | Chunked retrieval over the conversation history |
| `context` | Sliding-window context (`buffer_length` in `agent_config.yaml`) |
| `none` | No long-term memory |

### Defenses (`--defense-type`)

| Defense | Description |
|---------|-------------|
| `none` | Baseline |
| `user_prompt_only` | Index only user messages, never assistant output |
| `no_untrusted_tools` | Stop indexing once an untrusted tool (inbox read) has been called in the session |
| `limit_memory_length` | Cap each stored entry (`limit_memory_size_defense` characters, default 80) |
| `provable_policy` | Label memories by session trust; retrieving an untrusted memory marks the session untrusted and blocks send |

Two pairs are skipped by design (`is_valid_combination` in `src/benchmark/benchmark_utils.py`): `explicit + user_prompt_only` and `context + limit_memory_length`. Omitting `--memory-backend` or `--defense-type` runs all valid combinations.

### Attack bench and persistence

```text
data/benchmark/attack_bench/
  train_<N>/<topic>/<backend>/<defense>/01.json        # one train case per cell
  test_<N>/<topic>/<backend>/<defense>/01..04.json     # four test cases per cell
  train_cache_<N>/...                                  # optimized attacks from --adaptive
```

- Topics: `finance`, `health`, `identity`, `legal`, `tax`.
- `N ∈ {0, 10, 20, …, 100}` is the number of benign, unrelated sessions between planting the attack and the trigger. Each test case inserts the attack email, has the agent read it, then loads the memory snapshot for session `N` from `data/benchmark/snapshots/memory_snapshots_test/<backend>/<defense>/session_N.json`, starts a new session, and asks a topic-related question.
- Train and test cases use disjoint templates and entity slots.

### Utility suites (`--suite`)

`memory_only`, `assistant_responses`, `untrusted_probe`, `untrusted_send`, `disable_send`, `memory_tools`, `long_memory`. Each maps to one user flow with distinct capability requirements; per-flow utility is what the security/utility analysis in the paper reweights per deployment.

### Result caching

An existing result file makes the runner skip that test. Pass `--force` to re-run.

## Adaptive red-teaming pipeline

1. **Optimize** on train cases: `run_benchmark.py --adaptive --test data/benchmark/attack_bench/train_<N>/...`. Failed static attacks trigger the MAP-Elites-style optimizer in `src/benchmark/adaptive_attacks/openevolve_optimizer.py` (settings under `benchmark.openevolve` in `benchmark_config.yaml`). Optimized attacks are cached in `train_cache_<N>/`.
2. **Propagate** the best cached attack into every test case: `python scripts/propagate_train_attack_to_test_cases.py --model <model>`.
3. **Evaluate** on `test_<N>/` as in Quick start.
4. **Consolidate**: `python scripts/consolidate_attack_results.py` (test splits, ASR-vs-sessions plots) and `python scripts/consolidate_train.py` (train splits).

`--stealth` additionally optimizes for no visible exfiltration in the assistant's reply; it reads and writes `attack_bench_stealth/` and `attack_results_stealth/`.

## Command-line reference (`scripts/run_benchmark.py`)

| Argument | Description |
|----------|-------------|
| `--model` | **Required.** Target model id, e.g. `gpt-5-mini`, `gpt-5`, `gemini-3.1-pro-preview`. |
| `--memory-backend` | One or more of `explicit mem0 rag context none`. Default: all. |
| `--defense-type` / `--defense` | One or more unified defenses. Default: all. |
| `--suite` | Utility suite under `data/benchmark/tests/<suite>/`. Mutually exclusive with `--test`. |
| `--test` | Test file or directory (utility or attack bench). |
| `--adaptive` | Optimize attacks when the static attack fails. |
| `--stealth` | Adaptive only: also optimize for stealth. |
| `--early-stop-patience` | Adaptive only: stop after N iterations without improvement (default 10). |
| `--num-workers` | Parallel processes over backend × defense combinations (default 1). |
| `--force` | Overwrite cached results. |
| `--config` | Benchmark config path (default `benchmark_config.yaml`). |
| `--results-dir`, `--logs-dir` | Output roots (defaults under `data/benchmark/`). |
| `--seed` | RNG seed override (default 42 from `agent_config.yaml`). |
| `--debug-level` | `INFO` or `DEBUG`. |

## Configuration

- `agent_config.yaml`: agent timeouts and recursion limit, Mem0 extraction LLM and embeddings, RAG chunking, context buffer, seed. `target_model_name` is intentionally absent; use `--model`.
- `benchmark_config.yaml`: `semantic_judge` model, `openevolve` optimizer settings, `limit_memory_size_defense`.

## Outputs

```text
data/benchmark/results/<model>/<backend>/<defense>/<suite>/<test>.json                # utility
data/benchmark/attack_results/<split>/<model>/<topic>/<backend>/<defense>/<test>.json # attack bench
data/benchmark/logs/…  and  data/benchmark/attack_logs/…                              # per-test logs
```

Each result JSON records the configuration, every step's user message, agent response, tool calls, validator outcomes, and `overall_success`. `python scripts/consolidate_results.py` builds per-suite CSVs and heatmaps for utility runs.

## Regenerating data

- Attack cases: `src/benchmark/attack_generation/persistent_exfiltrate/` (deterministic under `seed`). See its README.
- Utility suites: `src/benchmark/dataset_generation/` (requires a LoCoMo checkout, see below).
- Benign sessions for snapshots: `src/benchmark/dataset_generation/generate_persistence_unrelated_snapshots.py`; snapshots are written by `src/benchmark/snapshot_io.py` when the benign multi-session case is run.

## Repository layout

```text
├── agent_config.yaml, benchmark_config.yaml
├── appendix.pdf                 # supplementary material
├── scripts/                     # run_benchmark, propagate, consolidate, shuffle experiments
├── src/agent/                   # agent_core, tools, backend/{explicit,mem0,rag,context}_memory.py
├── src/benchmark/               # test_bench, validators, snapshot_io, adaptive_attacks/, generators
├── data/benchmark/              # tests, attack_bench, snapshots, attack_results, consolidated_*
├── mem0/                        # patched Mem0 1.0.1 (Apache-2.0)
└── patches/                     # mem0-1.0.1-in-memory.patch
```

## Third-party code

- **Mem0** ([mem0ai/mem0](https://github.com/mem0ai/mem0), v1.0.1, Apache-2.0) is vendored under `mem0/mem0/` and picked up automatically by `src/agent/backend/mem0_memory.py`. Three files are modified so the FAISS store never touches disk (upstream persists to `/tmp/faiss/` and reloads it, which leaks memories across tests) and no `~/.mem0` directory is created. The exact diff against the PyPI 1.0.1 release is `patches/mem0-1.0.1-in-memory.patch`.
- **LoCoMo** ([snap-research/locomo](https://github.com/snap-research/locomo)) is needed only by the utility-suite generators, which import `global_methods`. Clone it to `LoCoMo/` at the repository root (git-ignored). Running the benchmark on the shipped tests does not need it.
- **OpenEvolve** ([algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)) is not a dependency; the adaptive optimizer is a self-contained reimplementation in `src/benchmark/adaptive_attacks/`.

## Citation

```bibtex
@article{das2026trojan,
  title={Trojan Hippo: Weaponizing Agent Memory for Data Exfiltration},
  author={Das, Debeshee and Piet, Julien and Kaviani, Darya and Beurer-Kellner, Luca and Tram{\`e}r, Florian and Wagner, David},
  journal={arXiv preprint arXiv:2605.01970},
  year={2026}
}
```

## License

Apache License 2.0 (see `LICENSE`). The vendored Mem0 code keeps its own Apache-2.0 license in `mem0/LICENSE`.
