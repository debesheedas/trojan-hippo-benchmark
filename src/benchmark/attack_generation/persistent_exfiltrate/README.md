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

### Persistence tests (test_N)

To generate **test-only** persistence cases (used with the memory snapshot system): insert_attack_email → read inbox → **load_memory_snapshot(N)** → start_new_session → trigger. Output goes under `attack_bench/test_{N}/` (e.g. `test_4/`).

```bash
# Generate test_4 for all categories
PYTHONPATH=src python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests --n 4

# One category only
PYTHONPATH=src python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests --n 4 --category tax
```

Requires snapshots to be built first (e.g. `python scripts/build_persistence_snapshots.py`).

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

Train/test split: one train case (`00_*_train.json`) and four test cases (`01_*.json` … `04_*.json`) per backend, with disjoint templates and entity slots (no leakage).

## Test case layout: num_sessions (default 5)

- **Train cases** are always 4 steps: insert_attack_email, read inbox, start_new_session, one trigger-topic query.
- **Test cases** depend on **`num_sessions`** (global default 5, overridable per category in `config.yaml`):
  - **`num_sessions: 0`** — Old style: 4 steps, single trigger-topic query.
  - **`num_sessions: 5`** (default) — Multi-session: 5 trigger-topic queries across 6 sessions, plus 2 unrelated share/probe pairs (noise to test attack persistence). Session layout: (1) trigger1; (2) trigger2; (3) trigger3 + share; (4) trigger4 + probe; (5) trigger5 + share; (6) probe. Only **probe** steps have a `user_goal`/success_check; share steps have no validator. The five trigger messages use **similar templates** (same slot, indices 1,2,3,4,4) so attacks generalize from train to test.

Unrelated share/probe pairs are hardcoded in **`unrelated_pairs.py`** (10 pairs from `data/benchmark/tests/memory_only`); each test case gets 2 pairs. Templates should stay very similar across the category so the attack generalizes.

## Backend-specific content

- **none** and **explicit**: Use shared text from `backend_definitions.yaml`; only topic placeholders (e.g. `{topic_trigger_for_email}`) are filled from the category config.
- **mem0**, **rag**, **context**: Require category-specific attack email bodies (different wording per topic). Define them under `attack_email_overrides.mem0`, `attack_email_overrides.rag`, and `attack_email_overrides.context` in the category’s `config.yaml`.
