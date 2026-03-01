#!/usr/bin/env python3
"""
Propagate a single attack email into each test case under test*.

For each (test_N, topic, backend, defense):
  - Collect all cached candidates from all train splits (cache stores attack_score 0-1).
  - If any cache has attack_score == 1 (ASR success): choose among those, preferring
    same-split (train_N) then train_100, train_90, ..., train_0.
  - Else: choose the cached candidate with highest score (ties: same preference).
  - If no cache: use original train file from train_N.

Writes only that one attack into step 1 (insert_attack_email) of each test JSON.

Usage:
    python scripts/propagate_train_attack_to_test_cases.py --model gemini-3.1-pro-preview
    python scripts/propagate_train_attack_to_test_cases.py --model gemini-3.1-pro-preview --stealth
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Default paths (attack_bench contains train/, test*, train_cache/)
DEFAULT_ATTACK_BENCH = BASE_DIR / "data" / "benchmark" / "attack_bench"
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "benchmark" / "attack_bench" / "train_cache"

# Train files are identified by having "train" in the filename stem (e.g. 00_*_train.json)
TRAIN_STEM_MARKER = "train"


def get_attack_email_from_cached(cached_path: Path) -> Optional[dict]:
    """Extract the attack email from the first insert_attack_email step in a cached test JSON."""
    if not cached_path.exists():
        return None
    with open(cached_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email" and "attack_email" in step:
            return step["attack_email"]
    return None


def get_attack_and_score_from_cached(cached_path: Path) -> Optional[Tuple[dict, float]]:
    """
    Extract (attack_email, attack_score) from a cached test JSON.
    attack_score is 0-1: from optimization_metadata.attack_score if present,
    else 1.0 if adaptive_success else 0.0. Returns None if no attack email.
    """
    if not cached_path.exists():
        return None
    with open(cached_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    email = None
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email" and "attack_email" in step:
            email = step["attack_email"]
            break
    if not email:
        return None
    meta = data.get("optimization_metadata") or {}
    if "attack_score" in meta:
        score = float(meta["attack_score"])
        score = max(0.0, min(1.0, score))
    else:
        score = 1.0 if meta.get("adaptive_success") else 0.0
    return (email, score)


def propagate_attack_to_file(test_case_path: Path, attack_email: dict) -> bool:
    """Set step 1 (insert_attack_email) attack_email in test_case_path.
    Returns True if updated.
    """
    with open(test_case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    updated = False
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email":
            step["attack_email"] = attack_email
            updated = True
            break
    if updated:
        with open(test_case_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return updated


def _discover_combinations_in_split(
    attack_bench_base: Path, split_name: str
) -> List[Tuple[str, str, str, Path]]:
    """
    Discover (topic, backend, defense, dir) for each combination under
    attack_bench/<split>/<topic>/<backend>/<defense>/.
    """
    combos: List[Tuple[str, str, str, Path]] = []
    base = attack_bench_base / split_name
    if not base.exists():
        return combos
    for topic_dir in base.iterdir():
        if not topic_dir.is_dir():
            continue
        topic = topic_dir.name
        for backend_dir in topic_dir.iterdir():
            if not backend_dir.is_dir():
                continue
            backend = backend_dir.name
            for defense_dir in backend_dir.iterdir():
                if not defense_dir.is_dir():
                    continue
                defense = defense_dir.name
                if any(defense_dir.glob("*.json")):
                    combos.append((topic, backend, defense, defense_dir))
    return combos


def _split_from_defense_dir(defense_dir: Path, attack_bench_base: Path) -> str:
    """Return test split name from defense_dir path, e.g. test_10 or test."""
    try:
        rel = defense_dir.relative_to(attack_bench_base)
        parts = rel.parts
        if parts:
            return parts[0]
    except ValueError:
        pass
    return "test"


def _train_split_and_cache_dir(test_split: str) -> Tuple[str, str]:
    """Return (train_split, cache_dir_name) for a test split. test_10 -> (train_10, train_cache_10)."""
    if test_split != "test" and test_split.startswith("test_"):
        suffix = test_split[4:]  # "test_10" -> "_10", "test_4" -> "_4"
        return f"train{suffix}", f"train_cache{suffix}"
    return "train", "train_cache"


def find_cached_train_file(
    cache_base: Path, model: str, topic: str, backend: str, defense: str,
    layout_without_model: bool = False,
) -> Optional[Path]:
    """Return path to the cached train file, or None.
    If layout_without_model=True (train_cache_10 style): cache_base/<topic>/<backend>/<defense>/*.json.
    Else (train_cache style): cache_base/<model>/<topic>/<backend>/<defense>/*.json.
    """
    if layout_without_model:
        cache_suite_dir = cache_base / topic / backend / defense
    else:
        cache_suite_dir = cache_base / model / topic / backend / defense
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if "_stealth" in p.stem:
            continue
        if TRAIN_STEM_MARKER in p.stem or layout_without_model:
            return p
    return None


def find_cached_train_file_stealth(
    cache_base: Path, model: str, topic: str, backend: str, defense: str,
    layout_without_model: bool = False,
) -> Optional[Path]:
    """Return path to the cached train file with _stealth (saved when adaptive --stealth achieved both attack and stealth)."""
    if layout_without_model:
        cache_suite_dir = cache_base / topic / backend / defense
    else:
        cache_suite_dir = cache_base / model / topic / backend / defense
    if not cache_suite_dir.exists():
        return None
    for p in cache_suite_dir.glob("*.json"):
        if "_stealth" not in p.stem:
            continue
        if TRAIN_STEM_MARKER in p.stem or layout_without_model:
            return p
    return None


def find_train_file_in_bench(
    attack_bench_base: Path, topic: str, backend: str, defense: str,
    train_split: str = "train",
) -> Optional[Path]:
    """Return path to the train file in attack_bench/<train_split>/<topic>/<backend>/<defense>/, or None."""
    train_suite_dir = attack_bench_base / train_split / topic / backend / defense
    if not train_suite_dir.exists():
        return None
    for p in train_suite_dir.glob("*.json"):
        if TRAIN_STEM_MARKER in p.stem:
            return p
    # Persistence train files are named 01.json, not *_train.json; accept any json as train for that dir
    for p in train_suite_dir.glob("*.json"):
        return p
    return None


# Session checkpoints: 0, 10, 20, ..., 100 (same as generate_persistence_attack_bench.sh).
SESSION_CHECKPOINTS: List[int] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
NUM_TRAIN_SPLITS = len(SESSION_CHECKPOINTS)
# When same-split has no (or worse) candidate: prefer train_100, then train_90, ..., then train_0.
TRAIN_SPLITS_PREFERENCE_ORDER: List[str] = [f"train_{i}" for i in reversed(SESSION_CHECKPOINTS)]


def _get_best_attack_for_train_split(
    attack_bench_base: Path,
    cache_base: Path,
    train_split: str,
    topic: str,
    backend: str,
    defense: str,
    model: str,
    stealth: bool,
) -> Optional[Tuple[dict, str]]:
    """
    Return the single best attack (email, source) for the given train split.
    Prefers cached (optimized) attack for that split, then original train file.
    """
    if train_split == "train":
        split_cache_base = cache_base
        layout_without_model = False
        cache_label = "train_cache"
    else:
        # train_0 -> train_cache_0, train_10 -> train_cache_10, ..., train_100 -> train_cache_100
        suffix = train_split.replace("train", "", 1).lstrip("_") or "0"
        cache_dir_name = f"train_cache_{suffix}"
        split_cache_base = attack_bench_base / cache_dir_name
        if not split_cache_base.exists():
            split_cache_base = cache_base
            layout_without_model = False
            cache_label = "train_cache"
        else:
            layout_without_model = True
            cache_label = cache_dir_name

    if stealth:
        cached = find_cached_train_file_stealth(
            split_cache_base, model, topic, backend, defense,
            layout_without_model=layout_without_model,
        )
        if cached:
            email = get_attack_email_from_cached(cached)
            if email:
                return (email, f"cached (stealth) {backend}/{defense} [{cache_label}]")
        if defense != "none":
            cached = find_cached_train_file_stealth(
                split_cache_base, model, topic, backend, "none",
                layout_without_model=layout_without_model,
            )
            if cached:
                email = get_attack_email_from_cached(cached)
                if email:
                    return (email, f"cached (stealth) {backend}/none [{cache_label}]")

    cached = find_cached_train_file(
        split_cache_base, model, topic, backend, defense,
        layout_without_model=layout_without_model,
    )
    if cached:
        email = get_attack_email_from_cached(cached)
        if email:
            return (email, f"cached {backend}/{defense} [{cache_label}]")
    if defense != "none":
        cached = find_cached_train_file(
            split_cache_base, model, topic, backend, "none",
            layout_without_model=layout_without_model,
        )
        if cached:
            email = get_attack_email_from_cached(cached)
            if email:
                return (email, f"cached {backend}/none [{cache_label}]")

    bench_train = find_train_file_in_bench(
        attack_bench_base, topic, backend, defense, train_split=train_split
    )
    if bench_train:
        email = get_attack_email_from_file(bench_train)
        if email:
            return (email, f"original train {train_split} {backend}")
    return None


def _resolve_cache_base_for_split(
    attack_bench_base: Path, cache_base: Path, train_split: str
) -> Tuple[Path, bool, str]:
    """Return (split_cache_base, layout_without_model, cache_label) for a train split.
    Normal: attack_bench/train_cache, attack_bench/train_cache_10, ...
    Stealth: attack_bench_stealth/train_cache, attack_bench_stealth/train_cache_10, ... (same names; pass attack_bench_stealth as base and cache_base = attack_bench_stealth/train_cache).
    """
    if train_split == "train":
        return cache_base, False, "train_cache"
    suffix = train_split.replace("train", "", 1).lstrip("_") or "0"
    cache_dir_name = f"train_cache_{suffix}"
    split_cache_base = attack_bench_base / cache_dir_name
    if not split_cache_base.exists():
        return cache_base, False, "train_cache"
    return split_cache_base, True, cache_dir_name


def _find_one_cached_file(
    split_cache_base: Path,
    model: str,
    topic: str,
    backend: str,
    defense: str,
    layout_without_model: bool,
    stealth: bool,
) -> Optional[Path]:
    """Return path to one cached file (stealth first if stealth=True), or None."""
    if stealth:
        p = find_cached_train_file_stealth(
            split_cache_base, model, topic, backend, defense,
            layout_without_model=layout_without_model,
        )
        if p:
            return p
        if defense != "none":
            p = find_cached_train_file_stealth(
                split_cache_base, model, topic, backend, "none",
                layout_without_model=layout_without_model,
            )
            if p:
                return p
    p = find_cached_train_file(
        split_cache_base, model, topic, backend, defense,
        layout_without_model=layout_without_model,
    )
    if p:
        return p
    if defense != "none":
        p = find_cached_train_file(
            split_cache_base, model, topic, backend, "none",
            layout_without_model=layout_without_model,
        )
        if p:
            return p
    return None


def _collect_cached_candidates(
    attack_bench_base: Path,
    cache_base: Path,
    topic: str,
    backend: str,
    defense: str,
    model: str,
    stealth: bool,
) -> List[Tuple[dict, float, str, str]]:
    """
    Collect all cached (email, score, train_split, source_label) for this (topic, backend, defense)
    from all train splits. score is 0-1 (from optimization_metadata).
    """
    out: List[Tuple[dict, float, str, str]] = []
    splits_to_check: List[str] = ["train"] + [f"train_{i}" for i in SESSION_CHECKPOINTS]
    for train_split in splits_to_check:
        split_cache_base, layout, cache_label = _resolve_cache_base_for_split(
            attack_bench_base, cache_base, train_split
        )
        cached_path = _find_one_cached_file(
            split_cache_base, model, topic, backend, defense, layout, stealth
        )
        if not cached_path:
            continue
        pair = get_attack_and_score_from_cached(cached_path)
        if not pair:
            continue
        email, score = pair
        def_label = f"{backend}/{defense}" if defense != "none" else f"{backend}/none"
        source = f"cached {def_label} [{cache_label}] score={score:.2f}"
        out.append((email, score, train_split, source))
    return out


def get_attack_email_from_file(json_path: Path) -> Optional[dict]:
    """Extract the attack email from the first insert_attack_email step in a test JSON file."""
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for step in data.get("steps", []):
        if step.get("step_type") == "insert_attack_email" and "attack_email" in step:
            return step["attack_email"]
    return None


def _pick_best_candidate(
    candidates: List[Tuple[dict, float, str, str]],
    train_split: str,
) -> Optional[Tuple[dict, str]]:
    """
    From collected (email, score, train_split, source) list:
    - If any has score >= 1.0 (ASR success), pick among those by preference (same-split first, then train_100, ..., train_0).
    - Else pick the one with highest score; ties broken by same preference.
    Returns (email, source) or None.
    """
    if not candidates:
        return None
    # Preference order: same-split first, then train_100, train_90, ..., train_0
    def preference_rank(s: str) -> int:
        if s == train_split:
            return 0
        try:
            return 1 + TRAIN_SPLITS_PREFERENCE_ORDER.index(s)
        except ValueError:
            return 999
    success = [c for c in candidates if c[1] >= 1.0]
    pool = success if success else candidates
    # Sort by score desc, then by preference rank asc (same-split then train_100, ..., train_0)
    pool_sorted = sorted(pool, key=lambda c: (-c[1], preference_rank(c[2])))
    best = pool_sorted[0]
    return (best[0], best[3])


def run(
    attack_bench_base: Path,
    cache_base: Path,
    model: str,
    stealth: bool = False,
) -> int:
    """
    For each (test_N, topic, backend, defense): choose a single attack to write to test files.
    - Collect all cached candidates from all train splits (each has attack_score 0-1 in cache).
    - If any cache has attack_score == 1 (ASR success): pick among those, preferring same-split then train_100, ..., train_0
    - Else: pick the cached candidate with highest score (ties: same preference).
    - If no cache: use original train file from train_N.
    Writes only that one attack. If stealth=True, prefer _stealth cache first when resolving cache file.
    """
    updated_count = 0
    skipped_no_attack = []
    skipped_no_test_files = []

    combo_dirs: List[Tuple[str, str, str, Path]] = []
    for split_dir in sorted(d for d in attack_bench_base.iterdir() if d.is_dir() and d.name.startswith("test")):
        combo_dirs.extend(_discover_combinations_in_split(attack_bench_base, split_dir.name))

    for topic, backend, defense, defense_dir in combo_dirs:
        suite_name = topic
        test_split = _split_from_defense_dir(defense_dir, attack_bench_base)
        train_split, _ = _train_split_and_cache_dir(test_split)

        candidates = _collect_cached_candidates(
            attack_bench_base, cache_base, topic, backend, defense, model, stealth
        )
        result = _pick_best_candidate(candidates, train_split)
        if result:
            primary_email, primary_source = result
        else:
            bench_train = find_train_file_in_bench(
                attack_bench_base, topic, backend, defense, train_split=train_split
            )
            if bench_train:
                primary_email = get_attack_email_from_file(bench_train)
                primary_source = f"original train {train_split} {backend}/{suite_name}" if primary_email else None
            else:
                primary_email, primary_source = None, None
            if not primary_email:
                skipped_no_attack.append(f"{backend}/{suite_name}")
                continue

        test_files = [p for p in defense_dir.glob("*.json") if TRAIN_STEM_MARKER not in p.stem]
        if not test_files:
            skipped_no_test_files.append(f"{backend}/{suite_name}")
            continue

        split_label = f" [{test_split}]" if test_split != "test" else ""
        print(f"  {topic}/{backend}/{defense}{split_label}: {primary_source} → {len(test_files)} test file(s)")
        for test_path in sorted(test_files):
            if propagate_attack_to_file(test_path, primary_email):
                print(f"    updated {test_path.name}")
                updated_count += 1

    if skipped_no_attack:
        print(f"\nSkipped (no attack source): {', '.join(skipped_no_attack)}")
    if skipped_no_test_files:
        print(f"Skipped (no non-train test files): {', '.join(skipped_no_test_files)}")
    print(f"\nTotal files updated: {updated_count}")
    return 0


def _infer_model_from_agent_config() -> Optional[str]:
    """Best-effort helper to infer model name from agent_config.yaml."""
    try:
        import yaml  # type: ignore[import]
        from pathlib import Path as _Path

        cfg = yaml.safe_load(_Path("agent_config.yaml").read_text())
        return cfg.get("agent", {}).get("target_model_name")
    except Exception:
        return None


def main() -> int:
    # Special case: no CLI arguments (just the script name).
    # In this mode we automatically:
    #   - Infer model from agent_config.yaml
    #   - Propagate for BOTH normal and stealth (fully parallel; no data mixing):
    #       * Normal:  attack_bench + attack_bench/train_cache[_N] → write to attack_bench/test_N/...
    #       * Stealth: attack_bench_stealth + attack_bench_stealth/train_cache[_N] → write to attack_bench_stealth/test_N/...
    if len(sys.argv) == 1:
        model = _infer_model_from_agent_config()
        if not model:
            print(
                "ERROR: Could not infer model from agent_config.yaml. "
                "Please run with --model <name>."
            )
            return 1

        # Normal attack_bench
        attack_bench_base = DEFAULT_ATTACK_BENCH.resolve()
        cache_base = DEFAULT_CACHE_DIR.resolve()
        print(f"Attack bench (normal):  {attack_bench_base}")
        print(f"Cache dir (normal):     {cache_base}")
        print(f"Model:                  {model}")
        print("Stealth:                disabled (normal propagation)")
        print()
        run(
            attack_bench_base,
            cache_base,
            model=model,
            stealth=False,
        )

        # Stealth: cache lives inside attack_bench_stealth with same names (train_cache, train_cache_0, ...).
        attack_bench_stealth = (DEFAULT_ATTACK_BENCH.parent / f"{DEFAULT_ATTACK_BENCH.name}_stealth").resolve()
        cache_stealth_base = attack_bench_stealth / "train_cache"
        if attack_bench_stealth.exists():
            print(f"Attack bench (stealth): {attack_bench_stealth}")
            print(f"Cache dir (stealth):    {cache_stealth_base}")
            print(f"Model:                  {model}")
            print("Stealth:                enabled (prefer _stealth cache)")
            print()
            run(
                attack_bench_stealth,
                cache_stealth_base,
                model=model,
                stealth=True,
            )
        else:
            print(f"Stealth attack bench directory not found, skipping stealth propagation: {attack_bench_stealth}")
        return 0

    # CLI-override mode: respect explicit --model / --attack-bench / --cache-dir / --stealth.
    parser = argparse.ArgumentParser(
        description="Copy attack email into test cases: for each (backend, defense), use cache(same defense) else cache(none) else original train."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name whose train_cache should be used (e.g. gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--attack-bench",
        type=Path,
        default=DEFAULT_ATTACK_BENCH,
        help="Base directory for attack_bench (default: data/benchmark/attack_bench)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Base directory for train cache (default: data/benchmark/attack_bench/train_cache)",
    )
    parser.add_argument(
        "--stealth",
        action="store_true",
        help="Prefer attack from cached _stealth file (from adaptive --stealth run). If not found, fall back to normal cache then original train.",
    )
    args = parser.parse_args()

    attack_bench_base = args.attack_bench.resolve()
    cache_base = args.cache_dir.resolve()
    print(f"Attack bench: {attack_bench_base}")
    print(f"Cache dir:    {cache_base}")
    print(f"Model:       {args.model}")
    if args.stealth:
        print("Stealth:     enabled (prefer _stealth cache)")
    print()
    return run(
        attack_bench_base,
        cache_base,
        model=args.model,
        stealth=args.stealth,
    )


if __name__ == "__main__":
    sys.exit(main())
