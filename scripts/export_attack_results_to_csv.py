#!/usr/bin/env python3
"""
Export attack_results to a single CSV (one row per result file) for multi-seed aggregation.

Reads from data/benchmark/attack_results/ (test* folders only). Writes
data/benchmark/aggregated_test_results/results_seed{N}.csv where N is read from
agent_config.yaml (top-level "seed"). Run this after each test run; use different
seeds and --force for each run to produce multiple CSVs, then use
aggregate_seed_runs.py to compute mean ± std.

Usage:
    python scripts/export_attack_results_to_csv.py
    python scripts/export_attack_results_to_csv.py --results-dir data/benchmark/attack_results
    python scripts/export_attack_results_to_csv.py --seed 42  # override config
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ATTACK_RESULTS_BASE = BASE_DIR / "data" / "benchmark" / "attack_results"
DEFAULT_AGGREGATED_DIR = BASE_DIR / "data" / "benchmark" / "aggregated_test_results"
TRAIN_STEM_MARKER = "train"

CSV_COLUMNS = [
    "split",
    "model",
    "topic",
    "memory_backend",
    "defense",
    "test_name",
    "result_file",
    "user_passed",
    "user_total",
    "user_rate",
    "attack_passed",
    "attack_total",
    "attack_rate",
    "max_attack_passed",
    "stealth_passed",
    "stealth_total",
    "stealth_rate",
    "seed",
]


def load_seed_from_config(agent_config_path: Optional[Path] = None) -> int:
    """Read seed from agent_config.yaml (top-level 'seed'). Default 42."""
    path = agent_config_path or BASE_DIR / "agent_config.yaml"
    if not path.exists():
        return 42
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if isinstance(config, dict) and "seed" in config:
            return int(config["seed"])
    except Exception:
        pass
    return 42


def parse_result_file(result_file: Path) -> Optional[Dict[str, Any]]:
    """Parse one result JSON; return dict with same keys as consolidate_attack_results.parse_result_file."""
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception as e:
        print(f"WARNING: Could not parse {result_file}: {e}", file=sys.stderr)
        return None

    steps = result.get("steps", [])
    excluded_types = {"start_new_session", "insert_attack_email"}

    if "total_user_steps" in result and "total_successful_user_steps" in result:
        user_total = result["total_user_steps"]
        user_passed = result["total_successful_user_steps"]
    else:
        user_steps = [
            s for s in steps
            if s.get("step_type") not in excluded_types
            and (s.get("user_goal") is not None or s.get("success_check") is not None)
        ]
        user_total = len(user_steps)
        user_passed = sum(
            1 for s in user_steps
            if s.get("user_goal", {}).get("passed") is True or s.get("passed") is True
        )

    attack_steps = [
        s for s in steps
        if s.get("step_type") not in excluded_types
        and s.get("attack_goal") is not None
        and isinstance(s.get("attack_goal"), dict)
        and s["attack_goal"].get("passed") is not None
    ]
    attack_total = len(attack_steps)
    attack_passed = sum(1 for s in attack_steps if s["attack_goal"].get("passed") is True)
    max_attack_passed = 1 if (attack_total > 0 and attack_passed >= 1) else 0

    stealth_steps = [
        s for s in steps
        if s.get("step_type") not in excluded_types
        and s.get("stealth_goal") is not None
        and isinstance(s.get("stealth_goal"), dict)
        and s["stealth_goal"].get("passed") is not None
    ]
    stealth_total = len(stealth_steps)
    stealth_passed = sum(1 for s in stealth_steps if s["stealth_goal"].get("passed") is True)

    user_rate = (user_passed / user_total * 100) if user_total > 0 else 0.0
    attack_rate = (attack_passed / attack_total * 100) if attack_total > 0 else 0.0
    stealth_rate = (stealth_passed / stealth_total * 100) if stealth_total > 0 else 0.0

    return {
        "test_name": result.get("test_name", "Unknown"),
        "user_passed": user_passed,
        "user_total": user_total,
        "user_rate": user_rate,
        "attack_passed": attack_passed,
        "attack_total": attack_total,
        "attack_rate": attack_rate,
        "max_attack_passed": max_attack_passed,
        "stealth_passed": stealth_passed,
        "stealth_total": stealth_total,
        "stealth_rate": stealth_rate,
    }


def discover_result_files(results_base: Path) -> List[tuple]:
    """
    Yield (split, model, topic, memory_backend, defense, result_file) for each test result JSON.
    Only test* folders; skip files whose stem contains TRAIN_STEM_MARKER.
    """
    if not results_base.is_dir():
        return []
    out = []
    for split_dir in sorted(results_base.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith("test"):
            continue
        split = split_dir.name
        for model_dir in split_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for topic_dir in model_dir.iterdir():
                if not topic_dir.is_dir():
                    continue
                topic = topic_dir.name
                for backend_dir in topic_dir.iterdir():
                    if not backend_dir.is_dir():
                        continue
                    memory_backend = backend_dir.name
                    for defense_dir in backend_dir.iterdir():
                        if not defense_dir.is_dir():
                            continue
                        defense = defense_dir.name
                        for result_file in defense_dir.glob("*.json"):
                            if TRAIN_STEM_MARKER in result_file.stem:
                                continue
                            out.append((split, model, topic, memory_backend, defense, result_file))
    return out


def run(
    results_base: Path,
    output_dir: Path,
    seed: Optional[int] = None,
) -> int:
    if seed is None:
        seed = load_seed_from_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"results_seed{seed}.csv"

    rows = []
    for split, model, topic, memory_backend, defense, result_file in discover_result_files(results_base):
        parsed = parse_result_file(result_file)
        if not parsed:
            continue
        rows.append({
            "split": split,
            "model": model,
            "topic": topic,
            "memory_backend": memory_backend,
            "defense": defense,
            "test_name": parsed["test_name"],
            "result_file": result_file.name,
            "user_passed": parsed["user_passed"],
            "user_total": parsed["user_total"],
            "user_rate": round(parsed["user_rate"], 2),
            "attack_passed": parsed["attack_passed"],
            "attack_total": parsed["attack_total"],
            "attack_rate": round(parsed["attack_rate"], 2),
            "max_attack_passed": parsed["max_attack_passed"],
            "stealth_passed": parsed["stealth_passed"],
            "stealth_total": parsed["stealth_total"],
            "stealth_rate": round(parsed["stealth_rate"], 2),
            "seed": seed,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {out_path} (seed={seed})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export attack_results to one CSV per run (seed in filename) for multi-seed aggregation."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_ATTACK_RESULTS_BASE,
        help="Attack results base (default: data/benchmark/attack_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AGGREGATED_DIR,
        help="Output directory for CSVs (default: data/benchmark/aggregated_test_results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed (default: read from agent_config.yaml)",
    )
    args = parser.parse_args()
    return run(
        results_base=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
