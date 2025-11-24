#!/usr/bin/env python3
"""
Trace Viewer for RAG Poisoning Attack Sessions

View complete traces for Session 1 and Session 2 of RAG poisoning attacks.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
load_dotenv()

from agent.utils import read_trace_events, load_config


def format_trace_event(event: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format a trace event for display.
    
    Args:
        event: Trace event dictionary
        verbose: If True, show full details including timestamps and IDs
    
    Returns:
        Formatted string representation
    """
    event_type = event.get("event_type", "unknown")
    payload = event.get("payload", {})
    ts = event.get("ts", "")
    
    lines = []
    
    if verbose:
        lines.append(f"[{ts}] Event ID: {event.get('event_id', 'N/A')}")
        lines.append(f"Type: {event_type}")
        lines.append("Payload:")
    
    if event_type == "user_input":
        text = payload.get("text", "")
        lines.append(f"👤 User: {text}")
        
    elif event_type == "agent_response":
        text = payload.get("text", "")
        lines.append(f"🤖 Agent: {text}")
        
    elif event_type == "tool_call":
        tool_name = payload.get("tool_name", "unknown")
        inputs = payload.get("inputs", {})
        call_id = payload.get("call_id", "N/A")
        
        lines.append(f"🔧 Tool Call: {tool_name}")
        if verbose:
            lines.append(f"  Call ID: {call_id}")
        lines.append(f"  Inputs: {json.dumps(inputs, indent=4)}")
        
    elif event_type == "tool_result":
        tool_name = payload.get("tool_name", "unknown")
        outputs = payload.get("outputs", {})
        call_id = payload.get("call_id", "N/A")
        
        lines.append(f"✅ Tool Result: {tool_name}")
        if verbose:
            lines.append(f"  Call ID: {call_id}")
        
        # Format outputs nicely
        if "result" in outputs:
            result = outputs["result"]
            # Truncate very long results
            if len(result) > 1000:
                result = result[:1000] + f"\n... [truncated, {len(result)} chars total]"
            lines.append(f"  Result: {result}")
        elif "error" in outputs:
            lines.append(f"  ❌ Error: {outputs['error']}")
        else:
            lines.append(f"  Outputs: {json.dumps(outputs, indent=4)}")
    
    else:
        lines.append(f"📋 {event_type}: {json.dumps(payload, indent=2)}")
    
    return "\n".join(lines)


def view_session_traces(
    session_id: str,
    trace_file: str = "data/interactive_agent/trace.jsonl",
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    View all trace events for a session.
    
    Args:
        session_id: Session ID to view traces for
        trace_file: Base trace file path
        verbose: If True, show full details
    
    Returns:
        List of trace events
    """
    events = read_trace_events(trace_file, session_id)
    
    if not events:
        print(f"No trace events found for session: {session_id}")
        return []
    
    print(f"\n{'='*80}")
    print(f"Trace Events for Session: {session_id}")
    print(f"Total Events: {len(events)}")
    print(f"{'='*80}\n")
    
    for i, event in enumerate(events, 1):
        print(f"\n--- Event {i}/{len(events)} ---")
        print(format_trace_event(event, verbose=verbose))
        print()
    
    return events


def view_rag_poisoning_traces(
    results_file: Optional[str] = None,
    session1_id: Optional[str] = None,
    session2_id: Optional[str] = None,
    trace_file: str = "data/interactive_agent/trace.jsonl",
    verbose: bool = False
):
    """
    View traces for both Session 1 and Session 2 of a RAG poisoning attack.
    
    Args:
        results_file: Path to results JSON file (will extract session IDs)
        session1_id: Session 1 ID (if not loading from results file)
        session2_id: Session 2 ID (if not loading from results file)
        trace_file: Base trace file path
        verbose: If True, show full details
    """
    # Load session IDs from results file if provided
    if results_file:
        results_path = Path(results_file)
        if not results_path.exists():
            print(f"Error: Results file not found: {results_file}")
            return
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"Loaded results from: {results_file}")
        print(f"Attack Objective: {results.get('attack_objective', 'N/A')}")
        print(f"Attack Success: {results.get('attack_success', False)}")
        
        # Try to get session IDs from results (if they were saved)
        session1_id_from_results = results.get("session1", {}).get("session_id")
        session2_id_from_results = results.get("session2", {}).get("session_id")
        
        if session1_id_from_results:
            session1_id = session1_id_from_results
            print(f"Session 1 ID from results: {session1_id}")
        if session2_id_from_results:
            session2_id = session2_id_from_results
            print(f"Session 2 ID from results: {session2_id}")
        
        # If not in results, try to find from trace files (fallback)
        if not session1_id or not session2_id:
            print(f"\nNote: Session IDs not in results file. Searching trace files...")
            trace_dir = Path(trace_file).parent / "traces"
            if trace_dir.exists():
                # Find most recent matching trace files
                session1_files = sorted(trace_dir.glob("rag_poisoning_session1_*.jsonl"), 
                                       key=lambda p: p.stat().st_mtime, reverse=True)
                session2_files = sorted(trace_dir.glob("rag_poisoning_session2_*.jsonl"),
                                       key=lambda p: p.stat().st_mtime, reverse=True)
                
                if not session1_id and session1_files:
                    session1_id = session1_files[0].stem  # Most recent
                    print(f"Found Session 1 (most recent): {session1_id}")
                if not session2_id and session2_files:
                    session2_id = session2_files[0].stem  # Most recent
                    print(f"Found Session 2 (most recent): {session2_id}")
    
    # View Session 1 traces
    if session1_id:
        print(f"\n{'#'*80}")
        print(f"# SESSION 1 TRACES")
        print(f"{'#'*80}")
        view_session_traces(session1_id, trace_file, verbose)
    else:
        print("\n⚠️  Session 1 ID not found. Use --session1-id or provide results file.")
    
    # View Session 2 traces
    if session2_id:
        print(f"\n{'#'*80}")
        print(f"# SESSION 2 TRACES")
        print(f"{'#'*80}")
        view_session_traces(session2_id, trace_file, verbose)
    else:
        print("\n⚠️  Session 2 ID not found. Use --session2-id or provide results file.")


def list_available_sessions(trace_file: str = "data/interactive_agent/trace.jsonl"):
    """List all available session trace files."""
    trace_dir = Path(trace_file).parent / "traces"
    
    if not trace_dir.exists():
        print(f"Trace directory not found: {trace_dir}")
        return
    
    trace_files = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    print(f"\nAvailable Session Traces ({len(trace_files)} total):\n")
    
    for trace_file_path in trace_files:
        session_id = trace_file_path.stem
        file_size = trace_file_path.stat().st_size
        mod_time = datetime.fromtimestamp(trace_file_path.stat().st_mtime)
        
        # Count events
        try:
            with open(trace_file_path, 'r', encoding='utf-8') as f:
                event_count = sum(1 for line in f if line.strip())
        except:
            event_count = 0
        
        print(f"  {session_id}")
        print(f"    Events: {event_count}, Size: {file_size/1024:.1f} KB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="View traces for RAG poisoning attack sessions")
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to results JSON file (will auto-detect session IDs)")
    parser.add_argument("--session1-id", type=str, default=None,
                       help="Session 1 ID (overrides results file)")
    parser.add_argument("--session2-id", type=str, default=None,
                       help="Session 2 ID (overrides results file)")
    parser.add_argument("--trace-file", type=str, default="data/interactive_agent/trace.jsonl",
                       help="Base trace file path")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose details (timestamps, event IDs, etc.)")
    parser.add_argument("--list", action="store_true",
                       help="List all available session traces")
    parser.add_argument("--session-id", type=str, default=None,
                       help="View traces for a specific session ID (single session)")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_sessions(args.trace_file)
        return
    
    if args.session_id:
        # View single session
        view_session_traces(args.session_id, args.trace_file, args.verbose)
        return
    
    # View both sessions
    view_rag_poisoning_traces(
        results_file=args.results_file,
        session1_id=args.session1_id,
        session2_id=args.session2_id,
        trace_file=args.trace_file,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

