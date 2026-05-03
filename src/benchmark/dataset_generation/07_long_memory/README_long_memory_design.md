# Long Memory Suite Design

## Purpose

The long_memory suite stress-tests whether memory backends can retain and recall information from **session 1** after **session 2** has overloaded the model’s context window. Session 2 is intentionally larger than the **largest** model context so that:

- **Context backend**: Cannot keep the full conversation in context; sliding-window/truncation will drop session-1 content, so recall must rely on whatever the context memory layer stores.
- **RAG/Mem0**: Must retrieve from stored memory, not from raw context.

If session 2 is smaller than a model’s context window, that model can “cheat” by holding everything in context and still answering probes correctly (e.g. Gemini 1M vs ~416k tokens previously).

## Model context limits (reference)

| Model                     | Context window | API input limit (approx) |
|---------------------------|----------------|---------------------------|
| gpt-5-mini                | 400k           | 272k                      |
| gemini-3.1-pro-preview    | 1M             | 1M                        |

(Defined in `src/agent/utils.py`: `_get_model_context_windows()`, `_get_api_token_limits()`.)

## Design requirement

**Session 2 total size must exceed the largest model’s context (1M tokens)** so that no model can fit the full conversation. Target: session 2 ≥ ~1.05M tokens (e.g. ~1.04M words at ~1.3 tokens/word).

- With `target_words_per_query: 80000` and 4 queries: 320k words ≈ 416k tokens → **does not** stress Gemini (fits in 1M).
- With **13** large-context queries × 80k words: 1.04M words ≈ 1.35M tokens → **does** stress both models.

So `num_large_context_queries_per_case` is set to **13** in the config.

## Why Gemini had high scores before

- Session 2 was ~416k tokens, below Gemini’s 1M limit.
- Full conversation (session 1 + session 2 + session 3) fit in context, so the Context backend did not need to “remember” via a sliding window; the model could recall from the in-context conversation.
- GPT-5-mini’s API limit (272k) was exceeded, so messages were truncated and session-1 content was dropped, leading to lower probe success.

After increasing session 2 to >1M tokens, both models are stressed and long_memory scores should reflect true memory/retrieval behavior.

## Regenerating test cases

From repo root:

```bash
python src/benchmark/dataset_generation/07_long_memory/generate_long_memory_test_cases.py \
  --config src/benchmark/dataset_generation/07_long_memory/long_memory_config.yaml
```

Output: `data/benchmark/tests/long_memory/long_memory_001.json` … `long_memory_004.json`. Remove any old `001.json`–`004.json` (no prefix) so only the new files are run.

## Rerunning the long_memory suite (both models, all defenses)

From repo root, run the suite for each model. Invalid combinations (e.g. context + limit_memory_length) are skipped automatically.

```bash
# Gemini 3.1 Pro Preview (all backends × all defenses for long_memory only)
python scripts/run_benchmark.py --suite long_memory --model gemini-3.1-pro-preview --force

# GPT-5-mini
python scripts/run_benchmark.py --suite long_memory --model gpt-5-mini --force
```

Then re-run consolidation so `all_suites_combined.csv` and thesis figures include the new long_memory results (see main README or `scripts/consolidate_results.py`).
