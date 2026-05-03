#!/usr/bin/env python3
"""
Generate persistence test cases for persistence snapshot evaluation.

Each test case: insert_attack_email, read inbox, load_memory_snapshot(N), start_new_session, trigger.
Writes to:
  - {attack_bench_base}/test_{N}/{topic}/{backend}/{defense}/{NN}.json   (split=test)
  - {attack_bench_base}/train_{N}/{topic}/{backend}/{defense}/{NN}.json  (split=train)
e.g. data/benchmark/attack_bench/test_4/finance/explicit/none/01.json

Usage:
  # Test-side persistence (e.g. test_4, test_10), typically using *_test snapshot_set_id
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests --n 4 --defense none --split test
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests --n 10 --defense none --split test --snapshot-set-id persistence_unrelated_20_test

  # Train-side persistence (train_10), typically using *_train snapshot_set_id
  python -m benchmark.attack_generation.persistent_exfiltrate.generate_persistence_tests --n 10 --defense none --split train --snapshot-set-id persistence_unrelated_20_train
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
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
    load_config,
    load_templates,
    load_entity_pools,
    load_backend_definitions,
    split_pools_into_slots,
    fill_template,
    build_persistence_test_case,
)
from benchmark.benchmark_utils import UNIFIED_DEFENSE_TYPES


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate persistence test cases (test_N): insert, read inbox, load_memory_snapshot(N), start_new_session, trigger."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Session index for load_memory_snapshot (e.g. 20). Use 0 to skip load_memory_snapshot (train_0/test_0).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Generate only this topic/category (finance, health, identity, legal, tax). Default: all.",
    )
    parser.add_argument(
        "--attack-bench-dir",
        type=str,
        default="data/benchmark/attack_bench",
        help="Base directory for attack_bench.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Repo base dir.",
    )
    parser.add_argument(
        "--snapshot-set-id",
        type=str,
        default="persistence_unrelated_20",
        help="Snapshot set id used in load_memory_snapshot step.",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="all",
        help=(
            "Defense type this test set is intended for "
            "(none, user_prompt_only, no_untrusted_tools, limit_memory_length, provable_policy), "
            "or 'all' to generate for every defense. Tests themselves are model-agnostic but organized "
            "by defense in the path layout."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Whether to write to test_N (split=test) or train_N (split=train).",
    )
    args = parser.parse_args()

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
    # Mirrored stealth tree lives alongside the primary attack_bench directory.
    # Example: data/benchmark/attack_bench_stealth with identical layout.
    attack_bench_stealth_base = attack_bench_base.parent / f"{attack_bench_base.name}_stealth"
    categories_dir = SCRIPT_DIR / "categories"
    backend_definitions_path = SCRIPT_DIR / "backend_definitions.yaml"

    if not categories_dir.is_dir():
        print(f"Error: categories dir not found: {categories_dir}", file=sys.stderr)
        return 1
    if not backend_definitions_path.exists():
        print(f"Error: backend_definitions.yaml not found", file=sys.stderr)
        return 1

    backend_defs = load_backend_definitions(backend_definitions_path)
    session_n = args.n
    split = args.split
    split_subdir = f"{split}_{session_n}"
    snapshot_set_id = args.snapshot_set_id
    if args.defense == "all":
        defenses = list(UNIFIED_DEFENSE_TYPES)
    else:
        defenses = [args.defense]

    if args.category:
        cat_dir = categories_dir / args.category
        if not cat_dir.is_dir():
            print(f"Error: category dir not found: {cat_dir}", file=sys.stderr)
            return 1
        category_dirs = [cat_dir]
    else:
        category_dirs = [
            d for d in categories_dir.iterdir()
            if d.is_dir() and (d / "config.yaml").exists()
        ]

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
            continue
        if not templates_path.exists() or not pools_path.exists():
            print(f"Skipping {category_name}: missing templates or entity_pools", file=sys.stderr)
            continue

        config = load_config(config_path)
        # Number of test cases per (topic, backend, defense)
        # For persistence:
        #   - test_N splits (e.g. test_10) keep the full set from config (default 4)
        #   - train_N splits (e.g. train_10) use a SINGLE train case per (topic, backend, defense)
        #     so that there is exactly one train persistence test feeding multiple test cases.
        num_test_config = config.get("num_test_cases", 4)
        if split == "train":
            num_test = 1
        else:
            num_test = num_test_config
        seed = config.get("seed", 42)
        base_initial_data = config.get("initial_data") or DEFAULT_INITIAL_DATA
        # Use split-specific inbox sets so train_* uses train inbox and test_* uses test inbox.
        initial_data = dict(base_initial_data)
        if split == "train":
            initial_data["inbox_set"] = "inbox_set_train"
        else:
            initial_data["inbox_set"] = "inbox_set_test"
        # Topic name is just the category directory name (finance, health, identity, legal, tax)
        topic_name = category_name
        user_goal_keyword = config.get("user_goal_keyword", category_name)

        templates = load_templates(templates_path)
        pools = load_entity_pools(pools_path)
        num_slots = 1 + num_test
        if len(templates) < num_slots:
            print(f"Error: {category_name} needs at least {num_slots} templates", file=sys.stderr)
            continue

        rng = __import__("random").Random(seed)
        shuffled_templates = rng.sample(templates, min(len(templates), num_slots))
        slots = split_pools_into_slots(pools, num_slots, rng)

        for backend in MEMORY_BACKENDS:
            if backend not in backend_defs:
                continue
            backend_def = backend_defs[backend]
            for defense in defenses:
                # Layout: attack_bench/<split_N>/<topic>/<backend>/<defense>/<NN>.json
                out_dir = attack_bench_base / split_subdir / topic_name / backend / defense
                # Mirrored stealth layout: attack_bench_stealth/<split_N>/<topic>/<backend>/<defense>/<NN>.json
                out_dir_stealth = attack_bench_stealth_base / split_subdir / topic_name / backend / defense
                # For train_N splits we want exactly ONE train case per (topic, backend, defense).
                # Clean out any existing JSON files so we don't keep stale 02/03/04 from older runs.
                if split == "train":
                    if out_dir.exists():
                        for old in out_dir.glob("*.json"):
                            try:
                                old.unlink()
                            except OSError:
                                pass
                    if out_dir_stealth.exists():
                        for old in out_dir_stealth.glob("*.json"):
                            try:
                                old.unlink()
                            except OSError:
                                pass
                out_dir.mkdir(parents=True, exist_ok=True)
                out_dir_stealth.mkdir(parents=True, exist_ok=True)
                saved = []
                for i in range(num_test):
                    try:
                        trigger_message = fill_template(
                            shuffled_templates[1 + i], slots[i + 1], rng
                        )
                        tc = build_persistence_test_case(
                            backend=backend,
                            backend_def=backend_def,
                            category_config=config,
                            category_name=category_name,
                            trigger_query=trigger_message,
                            case_index=i + 1,
                            session_n=session_n,
                            initial_data=initial_data,
                            name_prefix=topic_name,
                            user_goal_keyword=user_goal_keyword,
                            snapshot_set_id=snapshot_set_id,
                        )
                        fname = f"{i + 1:02d}.json"
                        out_file = out_dir / fname
                        out_file_stealth = out_dir_stealth / fname
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(tc, f, indent=2, ensure_ascii=False)
                        with open(out_file_stealth, "w", encoding="utf-8") as f:
                            json.dump(tc, f, indent=2, ensure_ascii=False)
                        saved.append(out_file)
                    except Exception as e:
                        print(
                            f"  ERROR {backend}/{category_name}/{defense} persistence test {i + 1}: {e}",
                            file=sys.stderr,
                        )
                        import traceback
                        traceback.print_exc()
                total_saved += len(saved)
                print(f"  {backend}/{topic_name}/{defense}: {len(saved)} files -> {out_dir}")

    print(f"\nTotal: {total_saved} persistence test case(s) written to {split_subdir}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
