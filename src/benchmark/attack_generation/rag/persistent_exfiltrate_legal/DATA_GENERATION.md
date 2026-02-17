# Data generation: train/test split and leakage control

Same procedure as persistent_exfiltrate_health and persistent_exfiltrate_tax: templates + entity pools, disjoint split by seed.

## Procedure (reproducible, no leakage)

1. **Templates.** Five similar templates for the step-4 user message (legal advice: case type, lawyer, court, timeline, opposing party, jurisdiction, key facts, question). Stored in `templates.yaml` with placeholders. Each message includes "legal" for the keyword check.

2. **Entity pools.** One pool per placeholder in `entity_pools.yaml` (case_type, lawyer_name, court, timeline, opposing_party, jurisdiction, key_facts, question_advice). At least five entries per category for five disjoint sets.

3. **Random split (seeded).** For a given run with seed \(s\): shuffle templates and assign one to train, four to test; shuffle each entity pool and partition into five disjoint subsets (0 = train, 1–4 = test).

4. **Filling.** Train instance: fill train template with entities from subset 0 only. Test instances: fill each test template with entities from the corresponding subset. No entity is shared between train and test.

5. **Reproducibility.** Same seed and same YAML files yield the same split and filled messages.

## Files

- `templates.yaml` – five template strings with placeholders.
- `entity_pools.yaml` – per-category lists of entities.
- `generate_persistent_exfiltrate_legal.py` – run with `--config config.yaml`.
