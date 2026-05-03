# Dataset generation scripts

These scripts generate benchmark test cases (queries, probes, and steps) for the utility test suites.

## External dependency: LoCoMo

The generators in the `0*_*/` directories use the **LoCoMo** methodology and depend on an external module `global_methods` (from a LoCoMo repository). Each script adds `BASE_DIR / "LoCoMo"` to `sys.path` and then runs:

```python
from global_methods import set_openai_key, run_chatgpt
```

- **BASE_DIR** is set per script; typically it is the repository root (e.g. `Path(__file__).resolve().parent...` up to the repo root).
- You must have the LoCoMo code available so that `global_methods` can be imported. Common options:
  1. Clone or copy the LoCoMo repository (or the part that provides `global_methods`) into a directory named `LoCoMo` at the **repository root**, so that `sys.path.insert(0, str(BASE_DIR / "LoCoMo"))` finds it.
  2. Or set `PYTHONPATH` to include the path to the directory that contains `global_methods` before running the script.

Without LoCoMo set up, running any of these generators will fail at import time with something like: `ModuleNotFoundError: No module named 'global_methods'`.

## Scripts and suites

- `01_memory_only/` – memory-only test cases  
- `02_assistant_responses/` – assistant response checks  
- `03_untrusted_probe/` – untrusted probe tests  
- `04_untrusted_send/` – untrusted send tests  
- `05_disable_send/` – disable-send tests  
- `06_memory_tools/` – memory tools tests  
- `07_long_memory/` – long-memory tests  

Each suite has a YAML config (e.g. `memory_only_config.yaml`) and a generator script; run the script from the repo root with the appropriate `--config` path.
