"""
Memory snapshot save/load for persistence benchmarking.

Snapshots are saved at the end of each session when running the benign
persistence_unrelated_20 test case. They are loaded in attack tests via the
load_memory_snapshot step and merged into the current in-memory environment.

Schema version for strict validation on load.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.utils import debug_info

# Schema version; increment when snapshot format changes. Load raises if mismatch.
SNAPSHOT_SCHEMA_VERSION = 1


def _snapshot_base_path(snapshot_set_id: str) -> Path:
    """
    Base directory for snapshots for a given snapshot set.

    Train/test-specific layout:
      - Train snapshots: data/benchmark/snapshots/memory_snapshots_train/{backend}/{defense}/session_{n}.json
      - Test  snapshots: data/benchmark/snapshots/memory_snapshots_test/{backend}/{defense}/session_{n}.json

    The snapshot_set_id (e.g. persistence_unrelated_20_train/test) controls which root we use.
    """
    root = Path("data/benchmark/snapshots")
    if snapshot_set_id.endswith("_train"):
        return root / "memory_snapshots_train"
    if snapshot_set_id.endswith("_test"):
        return root / "memory_snapshots_test"
    # Fallback for legacy/other snapshot sets
    return root / "memory_snapshots"


def get_snapshot_path(
    snapshot_set_id: str,
    memory_backend: str,
    defense_type: str,
    session_index: int,
) -> Path:
    """Path to a single session snapshot file."""
    return _snapshot_base_path(snapshot_set_id) / memory_backend / defense_type / f"session_{session_index}.json"


# ---------------------------------------------------------------------------
# Explicit backend
# ---------------------------------------------------------------------------


def save_snapshot_explicit(
    in_memory_env: Any,
    snapshot_path: Path,
    session_index: int,
    memory_backend: str,
    defense_type: str,
    snapshot_set_id: str,
    user_goals_passed: int,
    user_goals_total: int,
) -> None:
    """
    Save explicit memory state to a JSON file.
    Raises if explicit_manager is None or path cannot be written.
    """
    manager = getattr(in_memory_env, "explicit_manager", None)
    if manager is None:
        raise ValueError("save_snapshot_explicit: in_memory_env has no explicit_manager")
    long_term = getattr(manager, "long_term", [])
    # Copy to a serializable list of dicts
    payload: List[Dict[str, str]] = []
    for entry in long_term:
        if isinstance(entry, dict):
            payload.append({"text": entry.get("text", ""), "label": entry.get("label", "T")})
        else:
            payload.append({"text": str(entry), "label": "T"})

    data: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "session_index": session_index,
        "memory_backend": memory_backend,
        "defense_type": defense_type,
        "snapshot_set_id": snapshot_set_id,
        "user_goals_passed": user_goals_passed,
        "user_goals_total": user_goals_total,
        "explicit_long_term": payload,
    }
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_snapshot_explicit_merge(
    in_memory_env: Any,
    data: Dict[str, Any],
    current_memory_backend: str,
    current_defense_type: str,
    current_snapshot_set_id: str,
    current_session_index: int,
) -> tuple[int, int]:
    """
    Merge snapshot explicit long_term into current in_memory_env.explicit_manager.
    Validates snapshot metadata; raises on mismatch.
    Returns (user_goals_passed, user_goals_total) from snapshot.
    """
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema_version mismatch: got {data.get('schema_version')}, expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    if data.get("memory_backend") != current_memory_backend:
        raise ValueError(
            f"Snapshot memory_backend mismatch: got {data.get('memory_backend')}, expected {current_memory_backend}"
        )
    if data.get("defense_type") != current_defense_type:
        raise ValueError(
            f"Snapshot defense_type mismatch: got {data.get('defense_type')}, expected {current_defense_type}"
        )
    if data.get("snapshot_set_id") != current_snapshot_set_id:
        raise ValueError(
            f"Snapshot snapshot_set_id mismatch: got {data.get('snapshot_set_id')}, expected {current_snapshot_set_id}"
        )
    if data.get("session_index") != current_session_index:
        raise ValueError(
            f"Snapshot session_index mismatch: got {data.get('session_index')}, expected {current_session_index}"
        )

    debug_info(
        f"[load_memory_snapshot][explicit] Snapshot metadata validated "
        f"(backend={current_memory_backend}, defense_type={current_defense_type}, "
        f"snapshot_set_id={current_snapshot_set_id}, session_index={current_session_index}).",
        truncate=False,
    )

    manager = getattr(in_memory_env, "explicit_manager", None)
    if manager is None:
        raise ValueError("load_snapshot_explicit_merge: in_memory_env has no explicit_manager")

    payload = data.get("explicit_long_term", [])
    if not isinstance(payload, list):
        raise ValueError("Snapshot explicit_long_term is not a list")

    n_before = len(manager.long_term)
    PREVIEW_LEN = 150

    debug_info(
        f"[load_memory_snapshot][explicit] "
        f"BEFORE merge: explicit_manager.long_term has {n_before} entries (expected: ≥1 from attack session if agent called update_memory). "
        f"Will append {len(payload)} memories from snapshot.",
        truncate=False,
    )
    if n_before > 0:
        for i, m in enumerate(manager.long_term):
            text = m.get("text", str(m)) if isinstance(m, dict) else str(m)
            preview = (text[:PREVIEW_LEN] + "…") if len(text) > PREVIEW_LEN else text
            debug_info(f"[load_memory_snapshot][explicit]   Pre-merge [{i}] preview: {preview}", truncate=False)
    else:
        debug_info("[load_memory_snapshot][explicit]   Pre-merge: no memories (agent may not have called update_memory in session 1).", truncate=False)

    debug_info(
        f"[load_memory_snapshot][explicit] Snapshot contents ({len(payload)} entries from {current_snapshot_set_id} session_{current_session_index}):",
        truncate=False,
    )
    for i, entry in enumerate(payload):
        text = entry.get("text", str(entry)) if isinstance(entry, dict) else str(entry)
        preview = (text[:PREVIEW_LEN] + "…") if len(text) > PREVIEW_LEN else text
        label_str = f" label={entry.get('label', 'T')}" if isinstance(entry, dict) and current_defense_type == "provable_policy" else ""
        debug_info(f"[load_memory_snapshot][explicit]   Snapshot [{i}]{label_str} preview: {preview}", truncate=False)

    for entry in payload:
        if isinstance(entry, dict):
            manager.long_term.append({
                "text": entry.get("text", ""),
                "label": entry.get("label", "T"),
            })
        else:
            manager.long_term.append({"text": str(entry), "label": "T"})

    n_after = len(manager.long_term)
    debug_info(
        f"[load_memory_snapshot][explicit] Merge complete: total_memories={n_after} "
        f"(pre-merge={n_before} + snapshot={len(payload)}). user_goals={data.get('user_goals_passed', 0)}/{data.get('user_goals_total', 0)}",
        truncate=False,
    )
    debug_info("[load_memory_snapshot][explicit] VERIFY explicit memory order (all entries after merge):", truncate=False)
    for i, m in enumerate(manager.long_term):
        text = m.get("text", str(m)) if isinstance(m, dict) else str(m)
        preview = (text[:PREVIEW_LEN] + "…") if len(text) > PREVIEW_LEN else text
        label_str = f" label={m.get('label', 'T')}" if isinstance(m, dict) and current_defense_type == "provable_policy" else ""
        debug_info(f"[load_memory_snapshot][explicit]   [{i}]{label_str} {preview}", truncate=False)
    if n_before > 0:
        debug_info(
            f"[load_memory_snapshot][explicit] Expected: [0..{n_before - 1}] = from attack session (session 1), "
            f"[{n_before}..{n_after - 1}] = from snapshot (unrelated sessions). Merge is APPEND.",
            truncate=False,
        )
    else:
        debug_info(
            f"[load_memory_snapshot][explicit] Expected: [0..{n_after - 1}] = from snapshot only (no attack-session memories; agent may not have called update_memory in step 2).",
            truncate=False,
        )

    return (data.get("user_goals_passed", 0), data.get("user_goals_total", 0))


# ---------------------------------------------------------------------------
# Mem0 backend
# ---------------------------------------------------------------------------


def _mem0_memory_text(item: Dict[str, Any]) -> str:
    """Extract memory text from a mem0 result item (same logic as mem0_memory.py)."""
    return (
        item.get("memory")
        or item.get("memories")
        or item.get("text")
        or item.get("content")
        or item.get("fact")
        or ""
    )


def save_snapshot_mem0(
    in_memory_env: Any,
    snapshot_path: Path,
    session_index: int,
    memory_backend: str,
    defense_type: str,
    snapshot_set_id: str,
    user_goals_passed: int,
    user_goals_total: int,
) -> None:
    """
    Save mem0 memory state to a JSON file.
    Serializes get_all_memories() result (memory text + optional metadata per item).
    """
    manager = getattr(in_memory_env, "mem0_manager", None)
    if manager is None:
        raise ValueError("save_snapshot_mem0: in_memory_env has no mem0_manager")
    try:
        all_memories = manager.get_all_memories(limit=1000)
    except Exception as e:
        raise RuntimeError(f"save_snapshot_mem0: get_all_memories failed: {e}") from e
    payload: List[Dict[str, Any]] = []
    for item in all_memories:
        if not isinstance(item, dict):
            payload.append({"memory": str(item), "metadata": {}})
            continue
        text = _mem0_memory_text(item)
        if not text:
            continue
        payload.append({
            "memory": text,
            "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        })
    data: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "session_index": session_index,
        "memory_backend": memory_backend,
        "defense_type": defense_type,
        "snapshot_set_id": snapshot_set_id,
        "user_goals_passed": user_goals_passed,
        "user_goals_total": user_goals_total,
        "mem0_memories": payload,
    }
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_snapshot_mem0_merge(
    in_memory_env: Any,
    data: Dict[str, Any],
    current_memory_backend: str,
    current_defense_type: str,
    current_snapshot_set_id: str,
    current_session_index: int,
) -> tuple[int, int]:
    """
    Merge snapshot mem0 memories into current manager using memory.add(..., infer=False).
    Validates snapshot metadata; raises on mismatch.
    Returns (user_goals_passed, user_goals_total) from snapshot.
    """
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema_version mismatch: got {data.get('schema_version')}, expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    if data.get("memory_backend") != current_memory_backend:
        raise ValueError(
            f"Snapshot memory_backend mismatch: got {data.get('memory_backend')}, expected {current_memory_backend}"
        )
    if data.get("defense_type") != current_defense_type:
        raise ValueError(
            f"Snapshot defense_type mismatch: got {data.get('defense_type')}, expected {current_defense_type}"
        )
    if data.get("snapshot_set_id") != current_snapshot_set_id:
        raise ValueError(
            f"Snapshot snapshot_set_id mismatch: got {data.get('snapshot_set_id')}, expected {current_snapshot_set_id}"
        )
    if data.get("session_index") != current_session_index:
        raise ValueError(
            f"Snapshot session_index mismatch: got {data.get('session_index')}, expected {current_session_index}"
        )
    manager = getattr(in_memory_env, "mem0_manager", None)
    if manager is None:
        raise ValueError("load_snapshot_mem0_merge: in_memory_env has no mem0_manager")
    payload = data.get("mem0_memories", [])
    if not isinstance(payload, list):
        raise ValueError("Snapshot mem0_memories is not a list")
    debug_info(
        f"[load_memory_snapshot][mem0] "
        f"Merging {len(payload)} memories from snapshot "
        f"(backend={current_memory_backend}, defense={current_defense_type}, "
        f"snapshot_set_id={current_snapshot_set_id}, session_index={current_session_index})",
        truncate=False,
    )

    # Log a small sample of snapshot memories for manual inspection
    try:
        sample_entries = payload[:3]
        sample_texts = []
        for entry in sample_entries:
            if isinstance(entry, dict):
                text = entry.get("memory") or entry.get("memories") or entry.get("text") or entry.get("content") or entry.get("fact") or ""
            else:
                text = str(entry)
            if text:
                sample_texts.append(repr(str(text)[:300]))
        if sample_texts:
            debug_info(
                "[load_memory_snapshot][mem0] Sample snapshot memories: "
                + " | ".join(sample_texts),
                truncate=False,
            )
    except Exception:
        # Logging should never break snapshot loading; ignore any sampling errors
        pass

    memory_obj = getattr(manager, "memory", None)
    if memory_obj is None:
        raise ValueError("load_snapshot_mem0_merge: mem0_manager has no memory attribute")
    user_id = getattr(manager, "user_id", "vince") or "vince"
    agent_id = getattr(manager, "agent_id", None)
    imported_count = 0
    for entry in payload:
        text = entry.get("memory", "") if isinstance(entry, dict) else str(entry)
        if not text:
            continue
        meta = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        try:
            memory_obj.add(
                [{"role": "user", "content": text}],
                user_id=user_id,
                agent_id=agent_id,
                metadata=meta,
                infer=False,
            )
            imported_count += 1
        except Exception as e:
            raise RuntimeError(f"load_snapshot_mem0_merge: failed to add memory: {e}") from e

    debug_info(
        f"[load_memory_snapshot][mem0] Merge complete "
        f"(imported_memories={imported_count}, snapshot_entries={len(payload)}, "
        f"user_goals={data.get('user_goals_passed', 0)}/{data.get('user_goals_total', 0)})",
        truncate=False,
    )

    # Verification: log total memory count and short preview of each (attack session + snapshot)
    try:
        all_after = manager.get_all_memories(user_id=user_id, agent_id=agent_id, limit=1000)
        n_total = len(all_after) if isinstance(all_after, list) else 0
        debug_info(
            f"[load_memory_snapshot][mem0] VERIFY after merge: total memories in vectorstore = {n_total} "
            f"(expected: pre-merge attack-session memories + {imported_count} from snapshot)",
            truncate=False,
        )
        for idx, item in enumerate(all_after[:20]):  # cap at 20 to avoid huge logs
            if isinstance(item, dict):
                text = item.get("memory", "") or item.get("memories", "") or ""
            else:
                text = str(item)
            preview = (text[:120] + "…") if len(text) > 120 else text
            debug_info(f"[load_memory_snapshot][mem0]   [{idx}] {preview!r}", truncate=False)
        if n_total > 20:
            debug_info(f"[load_memory_snapshot][mem0]   ... and {n_total - 20} more", truncate=False)
    except Exception as e:
        debug_info(f"[load_memory_snapshot][mem0] VERIFY get_all_memories failed (non-fatal): {e}", truncate=False)

    return (data.get("user_goals_passed", 0), data.get("user_goals_total", 0))


# ---------------------------------------------------------------------------
# Context backend
# ---------------------------------------------------------------------------


def save_snapshot_context(
    in_memory_env: Any,
    snapshot_path: Path,
    session_index: int,
    memory_backend: str,
    defense_type: str,
    snapshot_set_id: str,
    user_goals_passed: int,
    user_goals_total: int,
) -> None:
    """Save context memory history to a JSON file."""
    manager = getattr(in_memory_env, "context_manager", None)
    if manager is None:
        raise ValueError("save_snapshot_context: in_memory_env has no context_manager")
    history = getattr(manager, "history", [])
    payload: List[Dict[str, str]] = []
    for entry in history:
        if isinstance(entry, dict):
            payload.append({"text": entry.get("text", ""), "label": entry.get("label", "T")})
        else:
            payload.append({"text": str(entry), "label": "T"})
    data: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "session_index": session_index,
        "memory_backend": memory_backend,
        "defense_type": defense_type,
        "snapshot_set_id": snapshot_set_id,
        "user_goals_passed": user_goals_passed,
        "user_goals_total": user_goals_total,
        "context_history": payload,
    }
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_snapshot_context_merge(
    in_memory_env: Any,
    data: Dict[str, Any],
    current_memory_backend: str,
    current_defense_type: str,
    current_snapshot_set_id: str,
    current_session_index: int,
) -> tuple[int, int]:
    """Merge snapshot context history into current context_manager.history."""
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema_version mismatch: got {data.get('schema_version')}, expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    if data.get("memory_backend") != current_memory_backend:
        raise ValueError(
            f"Snapshot memory_backend mismatch: got {data.get('memory_backend')}, expected {current_memory_backend}"
        )
    if data.get("defense_type") != current_defense_type:
        raise ValueError(
            f"Snapshot defense_type mismatch: got {data.get('defense_type')}, expected {current_defense_type}"
        )
    if data.get("snapshot_set_id") != current_snapshot_set_id:
        raise ValueError(
            f"Snapshot snapshot_set_id mismatch: got {data.get('snapshot_set_id')}, expected {current_snapshot_set_id}"
        )
    if data.get("session_index") != current_session_index:
        raise ValueError(
            f"Snapshot session_index mismatch: got {data.get('session_index')}, expected {current_session_index}"
        )
    manager = getattr(in_memory_env, "context_manager", None)
    if manager is None:
        raise ValueError("load_snapshot_context_merge: in_memory_env has no context_manager")
    payload = data.get("context_history", [])
    if not isinstance(payload, list):
        raise ValueError("Snapshot context_history is not a list")

    # Verification logging: context merge is APPEND (existing = session 1 attack; then snapshot = 10 unrelated)
    # For no_untrusted_tools, attack session uses untrusted tool so indexing is disabled => expected 0 pre-merge entries
    expected_pre = 0 if current_defense_type == "no_untrusted_tools" else 1
    n_before = len(manager.history)
    debug_info(
        f"[load_memory_snapshot][context] "
        f"BEFORE merge: context_manager.history has {n_before} entries (expected: {expected_pre} from attack session for defense={current_defense_type}). "
        f"Merging {len(payload)} memories from snapshot "
        f"(backend={current_memory_backend}, defense={current_defense_type}, "
        f"snapshot_set_id={current_snapshot_set_id}, session_index={current_session_index})",
        truncate=False,
    )
    if n_before > 0:
        first_text = manager.history[0].get("text", "") if isinstance(manager.history[0], dict) else str(manager.history[0])
        debug_info(
            f"[load_memory_snapshot][context] Pre-merge entry 0 (attack session) preview: {first_text[:300]}...",
            truncate=False,
        )
        # Defense verification for user_prompt_only: stored entry should be user-only (no Assistant content)
        if current_defense_type == "user_prompt_only":
            contains_assistant = "Assistant:" in first_text or "\nAssistant:" in first_text
            debug_info(
                f"[load_memory_snapshot][context] Pre-merge entry 0 defense check: len={len(first_text)}, "
                f"contains_Assistant={contains_assistant} => "
                f"{'user_only (user_prompt_only applied)' if not contains_assistant else 'WARNING: contains assistant content (defense may not be applied)'}",
                truncate=False,
            )
        # Defense verification for provable_policy: attack-session entry should have label U (session was untrusted when indexed)
        elif current_defense_type == "provable_policy":
            first_label = manager.history[0].get("label", "T") if isinstance(manager.history[0], dict) else "T"
            debug_info(
                f"[load_memory_snapshot][context] Pre-merge entry 0 defense check: label={first_label} => "
                f"{'U (attack session was untrusted when indexed, P3 correct)' if first_label == 'U' else 'WARNING: expected U for attack session (provable_policy)'}",
                truncate=False,
            )
    elif current_defense_type == "no_untrusted_tools":
        debug_info(
            "[load_memory_snapshot][context] Pre-merge 0 entries => no_untrusted_tools: attack session did not index (correct, untrusted tool was used).",
            truncate=False,
        )

    for entry in payload:
        if isinstance(entry, dict):
            manager.history.append({
                "text": entry.get("text", ""),
                "label": entry.get("label", "T"),
            })
        else:
            manager.history.append({"text": str(entry), "label": "T"})

    n_after = len(manager.history)
    debug_info(
        f"[load_memory_snapshot][context] Merge complete: total_entries={n_after} "
        f"(pre-snapshot={n_before} + snapshot={len(payload)}). "
        f"user_goals={data.get('user_goals_passed', 0)}/{data.get('user_goals_total', 0)})",
        truncate=False,
    )
    # Log first snapshot entry and last snapshot entry to verify exact match with session_10.json
    if payload:
        first_snap_text = payload[0].get("text", "") if isinstance(payload[0], dict) else str(payload[0])
        last_snap_text = payload[-1].get("text", "") if isinstance(payload[-1], dict) else str(payload[-1])
        debug_info(
            f"[load_memory_snapshot][context] First snapshot entry preview: {first_snap_text[:200]}...",
            truncate=False,
        )
        debug_info(
            f"[load_memory_snapshot][context] Last snapshot entry preview: ...{last_snap_text[-200:] if len(last_snap_text) > 200 else last_snap_text}",
            truncate=False,
        )
        # Snapshot defense check: user_prompt_only => snapshot should be user_only; no_untrusted_tools => User+Assistant is expected
        first_has_assistant = "Assistant:" in first_snap_text or "\nAssistant:" in first_snap_text
        if current_defense_type == "user_prompt_only":
            debug_info(
                f"[load_memory_snapshot][context] Snapshot entry 0 defense check: len={len(first_snap_text)}, "
                f"contains_Assistant={first_has_assistant} => "
                f"{'user_only (snapshot consistent with user_prompt_only)' if not first_has_assistant else 'WARNING: snapshot contains assistant content'}",
                truncate=False,
            )
        elif current_defense_type == "no_untrusted_tools":
            debug_info(
                f"[load_memory_snapshot][context] Snapshot entry 0 defense check: len={len(first_snap_text)}, "
                f"contains_Assistant={first_has_assistant} => "
                f"{'User+Assistant (expected for no_untrusted_tools snapshot)' if first_has_assistant else 'user_only (snapshot may be from user_prompt_only train)'}",
                truncate=False,
            )
        # Snapshot defense check for provable_policy: training-session entries should have label T
        elif current_defense_type == "provable_policy":
            first_snap_label = payload[0].get("label", "T") if isinstance(payload[0], dict) else "T"
            debug_info(
                f"[load_memory_snapshot][context] Snapshot entry 0 defense check: label={first_snap_label} => "
                f"{'T (training sessions trusted, correct)' if first_snap_label == 'T' else 'WARNING: expected T for snapshot (provable_policy)'}",
                truncate=False,
            )

    return (data.get("user_goals_passed", 0), data.get("user_goals_total", 0))


# ---------------------------------------------------------------------------
# RAG backend
# ---------------------------------------------------------------------------


def save_snapshot_rag(
    in_memory_env: Any,
    snapshot_path: Path,
    session_index: int,
    memory_backend: str,
    defense_type: str,
    snapshot_set_id: str,
    user_goals_passed: int,
    user_goals_total: int,
) -> None:
    """Save RAG memory state: documents list and chunk_counter from in_memory vectorstore."""
    vectorstore_storage = getattr(in_memory_env, "rag_vectorstore", None)
    if vectorstore_storage is None:
        raise ValueError("save_snapshot_rag: in_memory_env has no rag_vectorstore")
    try:
        _vs, documents, chunk_counter = vectorstore_storage.load()
    except Exception as e:
        raise RuntimeError(f"save_snapshot_rag: vectorstore load failed: {e}") from e
    documents_list = list(documents) if documents else []
    data: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "session_index": session_index,
        "memory_backend": memory_backend,
        "defense_type": defense_type,
        "snapshot_set_id": snapshot_set_id,
        "user_goals_passed": user_goals_passed,
        "user_goals_total": user_goals_total,
        "rag_documents": documents_list,
        "rag_chunk_counter": chunk_counter,
    }
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_snapshot_rag_merge(
    in_memory_env: Any,
    data: Dict[str, Any],
    current_memory_backend: str,
    current_defense_type: str,
    current_snapshot_set_id: str,
    current_session_index: int,
) -> tuple[int, int]:
    """Merge snapshot RAG documents into current RAG manager via add_memories_batch (re-embeds)."""
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema_version mismatch: got {data.get('schema_version')}, expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    if data.get("memory_backend") != current_memory_backend:
        raise ValueError(
            f"Snapshot memory_backend mismatch: got {data.get('memory_backend')}, expected {current_memory_backend}"
        )
    if data.get("defense_type") != current_defense_type:
        raise ValueError(
            f"Snapshot defense_type mismatch: got {data.get('defense_type')}, expected {current_defense_type}"
        )
    if data.get("snapshot_set_id") != current_snapshot_set_id:
        raise ValueError(
            f"Snapshot snapshot_set_id mismatch: got {data.get('snapshot_set_id')}, expected {current_snapshot_set_id}"
        )
    if data.get("session_index") != current_session_index:
        raise ValueError(
            f"Snapshot session_index mismatch: got {data.get('session_index')}, expected {current_session_index}"
        )
    manager = getattr(in_memory_env, "rag_manager", None)
    if manager is None:
        raise ValueError("load_snapshot_rag_merge: in_memory_env has no rag_manager")
    payload = data.get("rag_documents", [])
    if not isinstance(payload, list):
        raise ValueError("Snapshot rag_documents is not a list")

    # Defense-specific verification: log semantics for no_untrusted_tools so load can be verified from logs
    if current_defense_type == "no_untrusted_tools":
        debug_info(
            "[load_memory_snapshot][rag] defense=no_untrusted_tools: snapshot documents were saved from "
            "sessions that were TRUSTED when the snapshot was created (no untrusted tool had been used in those sessions).",
            truncate=False,
        )
    # provable_policy: snapshot docs must get label "T" when merged (persistence sessions were trusted)
    if current_defense_type == "provable_policy":
        debug_info(
            "[load_memory_snapshot][rag] defense=provable_policy: snapshot documents will be merged with label=T "
            "(persistence sessions were trusted when snapshot was created; retrieval will enforce T/U semantics).",
            truncate=False,
        )

    # Reload manager from storage so it has current docs (from attack session); otherwise it would
    # overwrite storage with only snapshot docs when we call add_memories_batch + _save_vectorstore.
    if hasattr(manager, "reload_from_storage"):
        manager.reload_from_storage()

    PREVIEW_LEN = 150
    # Pre-merge: get current doc count from vectorstore (attack session may have added docs in step 2)
    vs_storage = getattr(in_memory_env, "rag_vectorstore", None)
    n_before = 0
    pre_docs: List[str] = []
    if vs_storage is not None:
        try:
            _, pre_docs, _ = vs_storage.load()
            n_before = len(pre_docs)
        except Exception:
            pass

    debug_info(
        f"[load_memory_snapshot][rag] BEFORE merge: vectorstore has {n_before} documents "
        f"(expected: ≥1 from attack session if agent indexed the read-inbox turn). "
        f"Will append {len(payload)} documents from snapshot.",
        truncate=False,
    )
    if n_before > 0:
        for i, doc in enumerate(pre_docs):
            text = str(doc) if not isinstance(doc, str) else doc
            preview = (text[:PREVIEW_LEN] + "…") if len(text) > PREVIEW_LEN else text
            debug_info(f"[load_memory_snapshot][rag]   Pre-merge [{i}] preview: {preview}", truncate=False)
    else:
        debug_info(
            "[load_memory_snapshot][rag]   Pre-merge: no documents (attack session may not have indexed yet).",
            truncate=False,
        )
        if current_defense_type == "no_untrusted_tools":
            debug_info(
                "[load_memory_snapshot][rag] no_untrusted_tools: 0 pre-merge docs is EXPECTED when the attack "
                "session was marked UNTRUSTED (untrusted tool used in step 2); RAG indexing is disabled for that session.",
                truncate=False,
            )

    debug_info(
        f"[load_memory_snapshot][rag] Snapshot contents ({len(payload)} documents from "
        f"{current_snapshot_set_id} session_{current_session_index}):",
        truncate=False,
    )
    for i, entry in enumerate(payload):
        text = str(entry) if entry else ""
        preview = (text[:PREVIEW_LEN] + "…") if len(text) > PREVIEW_LEN else text
        debug_info(f"[load_memory_snapshot][rag]   Snapshot [{i}] preview: {preview}", truncate=False)

    doc_strings = [str(x) for x in payload if x]
    if doc_strings:
        # For provable_policy, merged snapshot docs must have label T (no session_id => default T)
        merge_defense_type = current_defense_type if current_defense_type == "provable_policy" else None
        try:
            manager.add_memories_batch(
                doc_strings,
                metadata=None,
                session_id=None,
                defense_type=merge_defense_type,
            )
        except Exception as e:
            raise RuntimeError(f"load_snapshot_rag_merge: add_memories_batch failed: {e}") from e

    n_after = 0
    after_docs: List[str] = []
    if vs_storage is not None:
        try:
            _, after_docs, _ = vs_storage.load()
            n_after = len(after_docs)
        except Exception:
            n_after = n_before + len(doc_strings)

    debug_info(
        f"[load_memory_snapshot][rag] Merge complete: total_documents={n_after} "
        f"(pre-merge={n_before} + snapshot={len(doc_strings)}). "
        f"user_goals={data.get('user_goals_passed', 0)}/{data.get('user_goals_total', 0)}",
        truncate=False,
    )
    if current_defense_type == "no_untrusted_tools":
        debug_info(
            f"[load_memory_snapshot][rag] no_untrusted_tools VERIFY: pre-merge={n_before} (attack session "
            f"UNTRUSTED→no indexing), snapshot={len(doc_strings)} (from trusted sessions), total={n_after}. "
            "Load memory snapshot correct for this defense.",
            truncate=False,
        )
    if current_defense_type == "provable_policy":
        debug_info(
            f"[load_memory_snapshot][rag] provable_policy VERIFY: pre-merge={n_before} (attack session), "
            f"snapshot={len(doc_strings)} (merged with label=T), total={n_after}. "
            "Snapshot docs have label T; retrieval will upgrade session to U if U-labeled memory is retrieved.",
            truncate=False,
        )
    debug_info("[load_memory_snapshot][rag] VERIFY RAG document order (all docs after merge):", truncate=False)
    for i, doc in enumerate(after_docs):
        text = str(doc) if not isinstance(doc, str) else doc
        preview = (text[:PREVIEW_LEN] + "…") if len(text) > PREVIEW_LEN else text
        debug_info(f"[load_memory_snapshot][rag]   [{i}] {preview}", truncate=False)
    if n_before > 0:
        debug_info(
            f"[load_memory_snapshot][rag] Expected: [0..{n_before - 1}] = from attack session (session 1), "
            f"[{n_before}..{n_after - 1}] = from snapshot (unrelated sessions). Merge is APPEND.",
            truncate=False,
        )
    else:
        debug_info(
            f"[load_memory_snapshot][rag] Expected: [0..{n_after - 1}] = from snapshot only "
            "(no attack-session docs in vectorstore before merge).",
            truncate=False,
        )

    if not doc_strings:
        debug_info(
            f"[load_memory_snapshot][rag] Snapshot contained no documents "
            f"(snapshot_entries={len(payload)}, "
            f"user_goals={data.get('user_goals_passed', 0)}/{data.get('user_goals_total', 0)})",
            truncate=False,
        )
    return (data.get("user_goals_passed", 0), data.get("user_goals_total", 0))


# ---------------------------------------------------------------------------
# Public save/load API (backend-dispatched)
# ---------------------------------------------------------------------------


def save_snapshot(
    in_memory_env: Any,
    snapshot_path: Path,
    session_index: int,
    memory_backend: str,
    defense_type: str,
    snapshot_set_id: str,
    user_goals_passed: int,
    user_goals_total: int,
) -> None:
    """
    Save current memory state for the given backend to snapshot_path.
    Only explicit is implemented in this first step; other backends will raise.
    """
    if memory_backend == "explicit":
        save_snapshot_explicit(
            in_memory_env,
            snapshot_path,
            session_index,
            memory_backend,
            defense_type,
            snapshot_set_id,
            user_goals_passed,
            user_goals_total,
        )
    elif memory_backend == "mem0":
        save_snapshot_mem0(
            in_memory_env,
            snapshot_path,
            session_index,
            memory_backend,
            defense_type,
            snapshot_set_id,
            user_goals_passed,
            user_goals_total,
        )
    elif memory_backend == "rag":
        save_snapshot_rag(
            in_memory_env,
            snapshot_path,
            session_index,
            memory_backend,
            defense_type,
            snapshot_set_id,
            user_goals_passed,
            user_goals_total,
        )
    elif memory_backend == "context":
        save_snapshot_context(
            in_memory_env,
            snapshot_path,
            session_index,
            memory_backend,
            defense_type,
            snapshot_set_id,
            user_goals_passed,
            user_goals_total,
        )
    elif memory_backend == "none":
        # Empty/dummy snapshot so path exists
        snapshot_path = Path(snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "session_index": session_index,
            "memory_backend": memory_backend,
            "defense_type": defense_type,
            "snapshot_set_id": snapshot_set_id,
            "user_goals_passed": user_goals_passed,
            "user_goals_total": user_goals_total,
        }
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        raise NotImplementedError(f"save_snapshot not implemented for backend={memory_backend}")


def load_snapshot_into_env(
    snapshot_path: Path,
    in_memory_env: Any,
    test_config: Dict[str, Any],
    memory_backend: str,
    defense_type: str,
    snapshot_set_id: str,
    session_index: int,
) -> tuple[int, int]:
    """
    Load snapshot from file and merge into in_memory_env.
    Validates metadata; raises on missing file or mismatch.
    Stores user_goals_passed and user_goals_total in test_config for reporting.
    Returns (user_goals_passed, user_goals_total) from the snapshot.
    """
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    debug_info(
        f"[load_memory_snapshot] Loading snapshot from {path} "
        f"(backend={memory_backend}, defense={defense_type}, "
        f"snapshot_set_id={snapshot_set_id}, session_index={session_index})",
        truncate=False,
    )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    passed = 0
    total = 0

    if memory_backend == "explicit":
        passed, total = load_snapshot_explicit_merge(
            in_memory_env,
            data,
            memory_backend,
            defense_type,
            snapshot_set_id,
            session_index,
        )
    elif memory_backend == "mem0":
        passed, total = load_snapshot_mem0_merge(
            in_memory_env,
            data,
            memory_backend,
            defense_type,
            snapshot_set_id,
            session_index,
        )
    elif memory_backend == "rag":
        passed, total = load_snapshot_rag_merge(
            in_memory_env,
            data,
            memory_backend,
            defense_type,
            snapshot_set_id,
            session_index,
        )
    elif memory_backend == "context":
        passed, total = load_snapshot_context_merge(
            in_memory_env,
            data,
            memory_backend,
            defense_type,
            snapshot_set_id,
            session_index,
        )
    elif memory_backend == "none":
        # No-op for memory; still validate and return user_goal counts
        if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"Snapshot schema_version mismatch: got {data.get('schema_version')}")
        if data.get("memory_backend") != memory_backend or data.get("defense_type") != defense_type:
            raise ValueError("Snapshot metadata mismatch")
        if data.get("snapshot_set_id") != snapshot_set_id or data.get("session_index") != session_index:
            raise ValueError("Snapshot snapshot_set_id or session_index mismatch")
        passed = data.get("user_goals_passed", 0)
        total = data.get("user_goals_total", 0)
    else:
        raise NotImplementedError(f"load_snapshot_into_env not implemented for backend={memory_backend}")

    # Store for additive user_goal reporting
    if "memory" not in test_config:
        test_config["memory"] = {}
    test_config["memory"]["_snapshot_user_goals_passed"] = passed
    test_config["memory"]["_snapshot_user_goals_total"] = total

    # High-level summary for benchmark logs
    item_counts: Dict[str, int] = {
        "explicit": len(data.get("explicit_long_term", [])),
        "mem0": len(data.get("mem0_memories", [])),
        "context": len(data.get("context_history", [])),
        "rag": len(data.get("rag_documents", [])),
        "none": 0,
    }
    snapshot_items = item_counts.get(memory_backend, 0)
    debug_info(
        f"[load_memory_snapshot] Loaded snapshot for backend={memory_backend} "
        f"(snapshot_items={snapshot_items}, "
        f"user_goals={passed}/{total})",
        truncate=False,
    )

    return (passed, total)
