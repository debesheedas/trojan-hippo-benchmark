#!/usr/bin/env python3
"""
Validate attack_logs, attack_results, and train_cache for train splits 0 to 100.
Reports JSON validity, required keys, and any suspicious/empty files.
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
ATTACK_RESULTS = BASE / "data" / "benchmark" / "attack_results"
ATTACK_LOGS = BASE / "data" / "benchmark" / "attack_logs"
ATTACK_BENCH = BASE / "data" / "benchmark" / "attack_bench"

# Train splits: 0, 10, 20, ..., 100
TRAIN_SPLITS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def validate_results_json(p: Path) -> tuple[bool, str]:
    """Validate a single attack result JSON. Return (ok, message)."""
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Read error: {e}"
    if not isinstance(data, dict):
        return False, "Root is not a dict"
    if "steps" not in data:
        return False, "Missing 'steps'"
    if not isinstance(data["steps"], list):
        return False, "'steps' is not a list"
    if "overall_success" not in data:
        return False, "Missing 'overall_success'"
    if "test_name" not in data and "test_file" not in data:
        return False, "Missing both 'test_name' and 'test_file'"
    return True, "OK"


def validate_cache_json(p: Path) -> tuple[bool, str]:
    """Validate a train_cache JSON: must have steps with insert_attack_email and attack_email."""
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Read error: {e}"
    if not isinstance(data, dict):
        return False, "Root is not a dict"
    steps = data.get("steps")
    if not isinstance(steps, list):
        return False, "Missing or invalid 'steps'"
    attack_step = None
    for s in steps:
        if isinstance(s, dict) and s.get("step_type") == "insert_attack_email":
            attack_step = s
            break
    if not attack_step:
        return False, "No step_type 'insert_attack_email'"
    if "attack_email" not in attack_step:
        return False, "insert_attack_email step missing 'attack_email'"
    email = attack_step["attack_email"]
    if not isinstance(email, dict):
        return False, "'attack_email' is not a dict"
    for k in ("from", "to", "subject", "body_plain"):
        if k not in email:
            return False, f"attack_email missing '{k}'"
    return True, "OK"


def main():
    errors = []
    stats = defaultdict(lambda: defaultdict(int))

    for split in TRAIN_SPLITS:
        split_name = f"train_{split}"

        # --- attack_results/train_N ---
        results_dir = ATTACK_RESULTS / split_name
        if not results_dir.exists():
            errors.append(f"[{split_name}] attack_results: directory missing")
            continue
        result_jsons = list(results_dir.rglob("*.json"))
        stats[split_name]["result_files"] = len(result_jsons)
        for p in result_jsons:
            ok, msg = validate_results_json(p)
            if not ok:
                errors.append(f"[{split_name}] result {p.relative_to(results_dir)}: {msg}")
            if p.stat().st_size == 0:
                errors.append(f"[{split_name}] result empty file: {p.relative_to(results_dir)}")

        # --- attack_logs/train_N ---
        logs_dir = ATTACK_LOGS / split_name
        if not logs_dir.exists():
            errors.append(f"[{split_name}] attack_logs: directory missing")
        else:
            log_files = list(logs_dir.rglob("*.log")) + list(logs_dir.rglob("*.json"))
            stats[split_name]["log_files"] = len(log_files)
            for p in log_files:
                if p.stat().st_size == 0:
                    errors.append(f"[{split_name}] log empty: {p.relative_to(logs_dir)}")

        # --- attack_bench/train_cache_N ---
        cache_name = f"train_cache_{split}"
        cache_dir = ATTACK_BENCH / cache_name
        if not cache_dir.exists():
            errors.append(f"[{split_name}] cache: directory missing ({cache_name})")
            continue
        cache_jsons = list(cache_dir.rglob("*.json"))
        stats[split_name]["cache_files"] = len(cache_jsons)
        for p in cache_jsons:
            ok, msg = validate_cache_json(p)
            if not ok:
                errors.append(f"[{split_name}] cache {p.relative_to(cache_dir)}: {msg}")
            if p.stat().st_size == 0:
                errors.append(f"[{split_name}] cache empty: {p.relative_to(cache_dir)}")

    # Print report
    print("=== Attack artifacts validation (train splits 0–100) ===\n")
    print("Counts per split:")
    print(f"{'Split':<12} {'results':>10} {'logs':>10} {'cache':>10}")
    print("-" * 46)
    for split in TRAIN_SPLITS:
        sn = f"train_{split}"
        r = stats[sn].get("result_files", 0)
        l = stats[sn].get("log_files", 0)
        c = stats[sn].get("cache_files", 0)
        print(f"{sn:<12} {r:>10} {l:>10} {c:>10}")
    print()
    if errors:
        print(f"=== Issues ({len(errors)} total) ===")
        for e in errors[:100]:
            print(e)
        if len(errors) > 100:
            print(f"... and {len(errors) - 100} more")
    else:
        print("No issues found.")
    return 1 if errors else 0


if __name__ == "__main__":
    exit(main())
