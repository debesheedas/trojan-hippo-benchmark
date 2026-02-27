#!/usr/bin/env python3
"""
Verify that test case files under attack_bench/test* have the correct
attack_email propagated, using the same logic as propagate_train_attack_to_test_cases.py.

For each (test_N, topic, backend, defense):
  - Compute expected attack (best cached candidate across train splits, or original train).
  - For each test JSON (excluding files with "train" in stem), compare current
    attack_email to expected. Report mismatches or missing attack_email.
"""
import json
import sys
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ATTACK_BENCH = BASE_DIR / "data" / "benchmark" / "attack_bench"
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "benchmark" / "attack_bench" / "train_cache"
TRAIN_STEM_MARKER = "train"


def _normalize_email(email: dict) -> str:
    """Canonical JSON string for comparison (sort keys)."""
    if not email:
        return ""
    return json.dumps(email, sort_keys=True, ensure_ascii=False)


def get_attack_email_from_file(json_path: Path):
    """Extract attack_email from first insert_attack_email step."""
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email" and "attack_email" in step:
            return step["attack_email"]
    return None


def main():
    # Import propagation logic so we use the exact same selection
    scripts_dir = BASE_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "propagate_train_attack_to_test_cases",
        BASE_DIR / "scripts" / "propagate_train_attack_to_test_cases.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _discover_combinations_in_split = mod._discover_combinations_in_split
    _split_from_defense_dir = mod._split_from_defense_dir
    _train_split_and_cache_dir = mod._train_split_and_cache_dir
    _collect_cached_candidates = mod._collect_cached_candidates
    _pick_best_candidate = mod._pick_best_candidate
    find_train_file_in_bench = mod.find_train_file_in_bench
    get_email_from_file = mod.get_attack_email_from_file

    attack_bench_base = DEFAULT_ATTACK_BENCH.resolve()
    cache_base = DEFAULT_CACHE_DIR.resolve()
    model = "gemini-3.1-pro-preview"

    combo_dirs = []
    for split_dir in sorted(
        d for d in attack_bench_base.iterdir() if d.is_dir() and d.name.startswith("test")
    ):
        combo_dirs.extend(
            _discover_combinations_in_split(attack_bench_base, split_dir.name)
        )

    errors = []
    missing_attack = []
    ok_count = 0
    total_test_files = 0

    for topic, backend, defense, defense_dir in combo_dirs:
        test_split = _split_from_defense_dir(defense_dir, attack_bench_base)
        train_split, _ = _train_split_and_cache_dir(test_split)

        candidates = _collect_cached_candidates(
            attack_bench_base, cache_base, topic, backend, defense, model, stealth=False
        )
        result = _pick_best_candidate(candidates, train_split)
        if result:
            expected_email, source = result
        else:
            bench_train = find_train_file_in_bench(
                attack_bench_base, topic, backend, defense, train_split=train_split
            )
            if bench_train:
                expected_email = get_email_from_file(bench_train)
                source = f"original train {train_split}"
            else:
                expected_email = None
                source = None

        if not expected_email:
            missing_attack.append(
                f"{test_split}/{topic}/{backend}/{defense}"
            )
            continue

        expected_norm = _normalize_email(expected_email)
        test_files = [
            p for p in defense_dir.glob("*.json") if TRAIN_STEM_MARKER not in p.stem
        ]

        for test_path in sorted(test_files):
            total_test_files += 1
            current = get_email_from_file(test_path)
            if not current:
                errors.append(
                    f"Missing attack_email: {test_path.relative_to(attack_bench_base)}"
                )
                continue
            current_norm = _normalize_email(current)
            if current_norm != expected_norm:
                errors.append(
                    f"Mismatch: {test_path.relative_to(attack_bench_base)} "
                    f"(expected from {source})"
                )
                continue
            ok_count += 1

    print("=== Verify propagate train → test ===\n")
    print(f"Model: {model}")
    print(f"Combos: {len(combo_dirs)}")
    print(f"Test files checked: {total_test_files}")
    print(f"OK (match expected): {ok_count}")
    if missing_attack:
        print(f"\nCombos with no attack source (skipped): {len(missing_attack)}")
        for m in missing_attack[:20]:
            print(f"  {m}")
        if len(missing_attack) > 20:
            print(f"  ... and {len(missing_attack) - 20} more")
    if errors:
        print(f"\nIssues ({len(errors)}):")
        for e in errors[:50]:
            print(f"  {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1
    print("\nAll test files have the expected propagated attack_email.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
