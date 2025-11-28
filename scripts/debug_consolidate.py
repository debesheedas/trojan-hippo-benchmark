#!/usr/bin/env python3
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import importlib.util
spec = importlib.util.spec_from_file_location("consolidate", BASE_DIR / "scripts" / "consolidate_defense_results.py")
consolidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consolidate)

# Check "none" defense
defense_type = "none"
results_dir = consolidate.get_defense_results_dir(defense_type)
print(f"Results dir: {results_dir}")
print(f"Exists: {results_dir.exists()}")

if results_dir.exists():
    result_files = list(results_dir.rglob("*.json"))
    print(f"\nFound {len(result_files)} JSON files:")
    for rf in result_files:
        print(f"  - {rf}")
        print(f"    Relative: {rf.relative_to(results_dir)}")
        print(f"    Parts: {rf.relative_to(results_dir).parts}")
        print(f"    Name: {rf.name}")
        print(f"    Starts with 00_: {rf.name.startswith('00_')}")

results, passed, total = consolidate.collect_results_for_defense(defense_type)
print(f"\nCollected: {len(results)} results, {passed}/{total} steps")

