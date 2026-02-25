#!/usr/bin/env python3
"""
Generate persistent exfiltrate test cases for all memory backends and all categories.

Writes to: {attack_bench_base}/train/{backend}/persistent_exfiltrate_{category}/ (train case)
         {attack_bench_base}/test/{backend}/persistent_exfiltrate_{category}/ (test cases)
e.g. data/benchmark/attack_bench/train/none/persistent_exfiltrate_tax/, data/benchmark/attack_bench/test/none/persistent_exfiltrate_tax/

Train/test split (no data leakage): same as legacy RAG generators — 1 train + 4 test
cases per category, with disjoint templates and entity slots.

Usage:
  # From repo root, with src on PYTHONPATH:
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_all
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_all --category tax
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_all --attack-bench-dir data/benchmark/attack_bench
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Ensure src is on path when run as script (e.g. python generate_all.py from repo root)
if "benchmark" not in sys.modules or "benchmark.attack_generation" not in sys.modules:
    src_dir = SCRIPT_DIR
    for _ in range(5):
        src_dir = src_dir.parent
        if (src_dir / "benchmark").exists() and (src_dir / "agent").exists():
            break
    else:
        src_dir = Path.cwd() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from benchmark.attack_generation.persistent_exfiltrate.common import (
    DEFAULT_INITIAL_DATA,
    MEMORY_BACKENDS,
    NUM_SESSIONS_DEFAULT,
    load_config,
    load_templates,
    load_entity_pools,
    load_backend_definitions,
    split_pools_into_slots,
    fill_template,
    build_test_case,
)
from benchmark.attack_generation.persistent_exfiltrate.unrelated_pairs import get_fixed_unrelated_pairs


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate persistent exfiltrate test cases for all backends and categories"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Generate only this category (e.g. tax, legal, health). Default: all categories in categories/.",
    )
    parser.add_argument(
        "--attack-bench-dir",
        type=str,
        default="data/benchmark/attack_bench",
        help="Base directory for attack_bench (default: data/benchmark/attack_bench)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Repo base dir (default: parent of src/ containing benchmark)",
    )
    args = parser.parse_args()

    # Resolve base dir: script lives in src/benchmark/attack_generation/persistent_exfiltrate/
    if args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = SCRIPT_DIR
        for _ in range(6):
            if (base_dir / "src" / "benchmark").exists():
                break
            base_dir = base_dir.parent
        else:
            base_dir = Path.cwd()
    attack_bench_base = base_dir / args.attack_bench_dir
    categories_dir = SCRIPT_DIR / "categories"

    if not categories_dir.is_dir():
        print(f"Error: categories dir not found: {categories_dir}", file=sys.stderr)
        return 1

    backend_definitions_path = SCRIPT_DIR / "backend_definitions.yaml"
    if not backend_definitions_path.exists():
        print(f"Error: backend_definitions.yaml not found: {backend_definitions_path}", file=sys.stderr)
        return 1

    backend_defs = load_backend_definitions(backend_definitions_path)

    # List categories to process
    if args.category:
        cat_dir = categories_dir / args.category
        if not cat_dir.is_dir():
            print(f"Error: category dir not found: {cat_dir}", file=sys.stderr)
            return 1
        category_dirs = [cat_dir]
    else:
        category_dirs = [d for d in categories_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()]

    if not category_dirs:
        print("Error: no category directories with config.yaml found", file=sys.stderr)
        return 1

    total_saved = 0
    for cat_dir in sorted(category_dirs):
        category_name = cat_dir.name
        config_path = cat_dir / "config.yaml"
        templates_path = cat_dir / "templates.yaml"
        pools_path = cat_dir / "entity_pools.yaml"

        if not config_path.exists():
            print(f"Skipping {category_name}: no config.yaml")
            continue
        if not templates_path.exists() or not pools_path.exists():
            print(f"Skipping {category_name}: missing templates.yaml or entity_pools.yaml", file=sys.stderr)
            continue

        config = load_config(config_path)

        num_attacks = config.get("num_attacks", 1)
        num_train = config.get("num_train_cases", 1)
        num_test = config.get("num_test_cases", 4)
        seed = config.get("seed", 42)
        num_sessions = config.get("num_sessions", NUM_SESSIONS_DEFAULT)
        base_initial_data = config.get("initial_data") or DEFAULT_INITIAL_DATA
        # Ensure train/test use disjoint inbox sets to avoid leakage via initial environment.
        initial_data_train = dict(base_initial_data)
        initial_data_train["inbox_set"] = "inbox_set_train"
        initial_data_test = dict(base_initial_data)
        initial_data_test["inbox_set"] = "inbox_set_test"
        name_prefix = f"persistent_exfiltrate_{category_name}"
        filename_prefix = name_prefix
        train_suffix = "train"
        user_goal_keyword = config.get("user_goal_keyword", category_name)

        templates = load_templates(templates_path)
        pools = load_entity_pools(pools_path)
        num_slots = num_train + num_test
        if len(templates) < num_slots:
            print(f"Error: {category_name} needs at least {num_slots} templates, have {len(templates)}", file=sys.stderr)
            continue
        # For num_sessions > 0 we need 5 similar templates per test case (indices 1,2,3,4,4); need at least 5 templates.
        if num_sessions > 0 and len(templates) < 5:
            print(f"Error: {category_name} needs at least 5 templates for num_sessions={num_sessions}, have {len(templates)}", file=sys.stderr)
            continue

        rng = __import__("random").Random(seed)
        shuffled_templates = rng.sample(templates, min(len(templates), max(num_slots, 5)))
        train_template = shuffled_templates[0]
        # Test case i uses slot i+1. For num_sessions>0 each test case needs 5 trigger messages from similar templates (1,2,3,4,4).
        test_template_indices = [1, 2, 3, 4, 4] if num_sessions > 0 else list(range(1, 1 + num_test))
        slots = split_pools_into_slots(pools, num_slots, rng)

        for backend in MEMORY_BACKENDS:
            if backend not in backend_defs:
                print(f"Warning: backend {backend} not in backend_definitions.yaml, skipping")
                continue
            backend_def = backend_defs[backend]
            train_dir = attack_bench_base / "train" / backend / name_prefix
            test_dir = attack_bench_base / "test" / backend / name_prefix
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)

            saved = []
            # Train case → attack_bench/train/{backend}/{name_prefix}/
            try:
                train_message = fill_template(train_template, slots[0], rng)
                train_queries = [train_message for _ in range(num_attacks)]
                tc = build_test_case(
                    backend=backend,
                    backend_def=backend_def,
                    category_config=config,
                    category_name=category_name,
                    queries=train_queries,
                    case_index=0,
                    name_suffix=train_suffix,
                    initial_data=initial_data_train,
                    name_prefix=name_prefix,
                    user_goal_keyword=user_goal_keyword,
                )
                fname = f"00_{filename_prefix}_{train_suffix}.json"
                out_file = train_dir / fname
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(tc, f, indent=2, ensure_ascii=False)
                saved.append(out_file)
            except Exception as e:
                print(f"  ERROR {backend}/{category_name} train: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue

            # Test cases → attack_bench/test/{backend}/{name_prefix}/
            for i in range(num_test):
                try:
                    if num_sessions > 0:
                        # 5 trigger queries from similar templates (same slot), 2 unrelated pairs
                        trigger_queries = [
                            fill_template(shuffled_templates[tidx], slots[i + 1], rng)
                            for tidx in test_template_indices
                        ]
                        unrelated_pairs = get_fixed_unrelated_pairs(2)
                        tc = build_test_case(
                            backend=backend,
                            backend_def=backend_def,
                            category_config=config,
                            category_name=category_name,
                            queries=trigger_queries,
                            case_index=i + 1,
                            name_suffix=None,
                            initial_data=initial_data_test,
                            name_prefix=name_prefix,
                            user_goal_keyword=user_goal_keyword,
                            num_sessions=num_sessions,
                            unrelated_pairs=unrelated_pairs,
                        )
                    else:
                        # Old style: 1 trigger per test case
                        test_message = fill_template(shuffled_templates[1 + i], slots[i + 1], rng)
                        test_queries = [test_message for _ in range(num_attacks)]
                        tc = build_test_case(
                            backend=backend,
                            backend_def=backend_def,
                            category_config=config,
                            category_name=category_name,
                            queries=test_queries,
                            case_index=i + 1,
                            name_suffix=None,
                            initial_data=initial_data_test,
                            name_prefix=name_prefix,
                            user_goal_keyword=user_goal_keyword,
                        )
                    fname = f"{i + 1:02d}_{filename_prefix}.json"
                    out_file = test_dir / fname
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(tc, f, indent=2, ensure_ascii=False)
                    saved.append(out_file)
                except Exception as e:
                    print(f"  ERROR {backend}/{category_name} test {i + 1}: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()

            total_saved += len(saved)
            print(f"  {backend}/{name_prefix}: train→{train_dir}, test→{test_dir} ({len(saved)} files)")

    print(f"\nTotal: {total_saved} test case file(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
