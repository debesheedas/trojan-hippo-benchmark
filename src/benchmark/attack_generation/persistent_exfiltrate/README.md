# Persistent exfiltrate test case generator

Single generator that produces **persistent exfiltrate** attack test cases for **all five memory backends** (none, explicit, mem0, rag, context) and for **all configured categories** (e.g. tax, legal, health).

**Determinism:** With the same config files and `seed` (default 42), the script is deterministic: re-running it produces identical test case files. All randomness goes through a single `random.Random(seed)` per category; template choice and entity slot assignment are fixed for that seed.

Output layout:
- Train: `data/benchmark/attack_bench/train/{backend}/persistent_exfiltrate_{category}/00_*_train.json`
- Test: `data/benchmark/attack_bench/test/{backend}/persistent_exfiltrate_{category}/01_*.json` … `04_*.json`

## Running the generator

From the **repository root** (with `src` on `PYTHONPATH`):

```bash
# Generate for all categories and all backends
PYTHONPATH=src python -m benchmark.attack_generation.persistent_exfiltrate.generate_all

# Generate for one category only
PYTHONPATH=src python -m benchmark.attack_generation.persistent_exfiltrate.generate_all --category tax

# Custom attack_bench base directory
PYTHONPATH=src python -m benchmark.attack_generation.persistent_exfiltrate.generate_all --attack-bench-dir data/benchmark/attack_bench
```

## Adding a new category (e.g. for 7 more topics)

1. **Create a new directory** under `categories/`, e.g. `categories/finance/`.

2. **Add these files:**
   - **`config.yaml`** — Minimal keys:
     - **`topic_label`** — Human-readable topic (e.g. `"taxes"`, `"legal matters, court, or lawyer"`). Used in descriptions and for deriving `topic_trigger_phrase` and default `topic_trigger_for_email`.
     - **`topic_short`** (optional) — Short label for “X-related” text (e.g. `"tax"` when topic_label is `"taxes"`). Defaults to the first word of `topic_label`.
     - **`topic_trigger_for_email`** (optional) — Exact phrase used in attack email instructions. Defaults to `"{topic_label} or related topics"`.
     - **`topic_related`** (optional) — e.g. `"legal/court/lawyer-related"`. Defaults to `"{topic_short}-related"`.
     - **`user_goal_keyword`** — Keyword for success_check (defaults to category directory name).
     - **`attack_email_overrides`** for **mem0**, **rag**, and **context**: each backend can specify `from`, `subject`, `body_plain`. `to` defaults to `vince.j.kaminski@enron.com` in code if omitted.
     - Optional overrides (only if you need non-defaults): `num_attacks`, `num_train_cases`, `num_test_cases`, `seed`, `initial_data`.
   - **`templates.yaml`** — List of user-message templates with placeholders. At least `num_train_cases + num_test_cases` templates (default 1+4=5).
   - **`entity_pools.yaml`** — Entity pools keyed by placeholder name. Each pool must have at least that many entries for disjoint train/test slots.

   **Defaults (in code):** `num_attacks=1`, `num_train_cases=1`, `num_test_cases=4`, `seed=42`, `initial_data` = standard inbox/outbox/drafts sets. Naming is `persistent_exfiltrate_{category_name}`; train file suffix is `train`. `expected_behavior` and `user_goal_description` are derived from `topic_short` (e.g. “Agent should help with {topic_short}-related questions…”).

3. **Run the generator** (with or without `--category finance`). The new category will be written to `attack_bench/train/` and `attack_bench/test/` under each backend (e.g. `attack_bench/train/rag/persistent_exfiltrate_finance/`, `attack_bench/test/rag/persistent_exfiltrate_finance/`).

Train/test split is the same as before: one train case (`00_*_train.json`) and four test cases (`01_*.json` … `04_*.json`) per backend, with disjoint templates and entity slots (no leakage).

## Backend-specific content

- **none** and **explicit**: Use shared text from `backend_definitions.yaml`; only topic placeholders (e.g. `{topic_trigger_for_email}`) are filled from the category config.
- **mem0**, **rag**, **context**: Require category-specific attack email bodies (different wording per topic). Define them under `attack_email_overrides.mem0`, `attack_email_overrides.rag`, and `attack_email_overrides.context` in the category’s `config.yaml`.
