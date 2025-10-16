#!/usr/bin/env python3
"""
CLI benchmark runner for the Email Agent using the pure-Python core.

Usage examples:
  python benchmark_runner.py --message "Summarize my inbox"
  python benchmark_runner.py --script examples/basic.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from agent_core import invoke_agent


def run_single(message: str, session_id: str) -> dict:
    start = time.time()
    result = invoke_agent(message, session_id=session_id)
    duration = time.time() - start
    return {
        "session_id": result["session_id"],
        "message": message,
        "response": result["response"],
        "duration_s": round(duration, 3),
    }


def run_script(script_path: Path, session_id: str) -> list:
    """
    Run a JSONL script of inputs. Each line: {"input": "..."}
    """
    outputs = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            msg = obj.get("input", "")
        except Exception:
            msg = line.strip()
        outputs.append(run_single(msg, session_id))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Email Agent Benchmark Runner (Pure Python)")
    parser.add_argument("--message", type=str, help="Single message to send", default="")
    parser.add_argument("--script", type=str, help="Path to JSONL script of inputs", default="")
    parser.add_argument("--session", type=str, help="Optional session id", default="benchmark")
    parser.add_argument("--out", type=str, help="Optional output file (JSON)", default="")
    args = parser.parse_args(argv)

    results = []
    if args.script:
        path = Path(args.script)
        if not path.exists():
            print(f"Script not found: {path}")
            return 1
        results = run_script(path, args.session)
    elif args.message:
        results = [run_single(args.message, args.session)]
    else:
        print("Provide --message or --script")
        return 1

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


