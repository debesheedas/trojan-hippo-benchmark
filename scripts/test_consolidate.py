#!/usr/bin/env python3
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import directly
import importlib.util
spec = importlib.util.spec_from_file_location("consolidate_defense_results", BASE_DIR / "scripts" / "consolidate_defense_results.py")
consolidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consolidate)

collect_results_for_defense = consolidate.collect_results_for_defense
get_defense_results_dir = consolidate.get_defense_results_dir

# Test disable_memory
defense_type = "disable_memory"
results_dir = get_defense_results_dir(defense_type)
print(f"Results dir for {defense_type}: {results_dir}")
print(f"Exists: {results_dir.exists()}")

if results_dir.exists():
    result_files = list(results_dir.rglob("*.json"))
    print(f"Found {len(result_files)} JSON files")
    for rf in result_files:
        print(f"  - {rf}")

results, passed, total = collect_results_for_defense(defense_type)
print(f"\nCollected: {len(results)} results, {passed}/{total} steps")

if results:
    print(f"\nFirst result keys: {list(results[0].keys())}")
    if 'steps' in results[0]:
        print(f"Number of steps: {len(results[0]['steps'])}")
        for i, step in enumerate(results[0]['steps'][:3]):
            print(f"  Step {i+1}: passed={step.get('passed')}, step_type={step.get('step_type')}")

