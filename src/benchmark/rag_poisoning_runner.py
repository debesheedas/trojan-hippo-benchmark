#!/usr/bin/env python3
"""
RAG Poisoning Attack Test Runner

Runs RAG poisoning attack test cases and evaluates results.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
load_dotenv()

from agent.utils import load_config
from benchmark.rag_poisoning_attack import run_rag_poisoning_test_case


def main():
    parser = argparse.ArgumentParser(description="Run RAG poisoning attack test case")
    parser.add_argument("--test-case-id", type=str, default=None,
                       help="Test case ID (e.g., 'test_case_1', 'test_case_2', 'test_case_3')")
    parser.add_argument("--attack-objective", type=str, default=None,
                       help="Attack objective string (default: from config or test case)")
    parser.add_argument("--max-emails", type=int, default=None,
                       help="Maximum number of attack emails (default: from config)")
    parser.add_argument("--max-email-length", type=int, default=None,
                       help="Maximum email length (default: from config)")
    parser.add_argument("--output-dir", type=str, default="data/benchmark/rag_poisoning_results",
                       help="Output directory for results")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode: use smaller numbers (5 emails, 1000 chars)")
    parser.add_argument("--list-test-cases", action="store_true",
                       help="List all available test cases and exit")
    
    args = parser.parse_args()
    
    # List test cases if requested
    if args.list_test_cases:
        from benchmark.rag_poisoning_test_cases import get_all_test_case_ids, get_test_case
        print("Available test cases:")
        for test_case_id in get_all_test_case_ids():
            test_case = get_test_case(test_case_id)
            print(f"\n{test_case_id}:")
            print(f"  Name: {test_case['name']}")
            print(f"  Description: {test_case['description']}")
            print(f"  Attack Objective: {test_case['attack_objective']}")
            print(f"  Validator Type: {test_case['validator_config']['type']}")
        return
    
    # Load config
    config = load_config()
    
    # Test mode overrides
    if args.test_mode:
        max_emails = 5
        max_email_length = 1000
        print("Running in TEST MODE (small numbers for quick testing)")
    else:
        max_emails = args.max_emails
        max_email_length = args.max_email_length
    
    # Run test case
    print(f"\n{'='*80}")
    if args.test_case_id:
        from benchmark.rag_poisoning_test_cases import get_test_case
        test_case = get_test_case(args.test_case_id)
        print(f"RAG Poisoning Attack Test Case: {test_case['name']}")
        print(f"Description: {test_case['description']}")
    else:
        print(f"RAG Poisoning Attack Test Case")
    print(f"{'='*80}\n")
    
    results = run_rag_poisoning_test_case(
        attack_objective=args.attack_objective,
        max_emails=max_emails,
        max_email_length=max_email_length,
        config=config,
        test_case_id=args.test_case_id
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"rag_poisoning_{timestamp_str}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print trace viewing instructions
    session1_id = results.get("session1", {}).get("session_id")
    session2_id = results.get("session2", {}).get("session_id")
    if session1_id or session2_id:
        print(f"\n{'='*80}")
        print(f"View Complete Traces")
        print(f"{'='*80}")
        print(f"\nTo view complete traces for both sessions:")
        print(f"  python src/benchmark/trace_viewer.py --results-file {results_file}")
        if session1_id:
            print(f"\nOr view individual sessions:")
            print(f"  python src/benchmark/trace_viewer.py --session-id {session1_id}")
            print(f"  python src/benchmark/trace_viewer.py --session-id {session2_id}")
        print(f"\nOr list all available sessions:")
        print(f"  python src/benchmark/trace_viewer.py --list")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Final Summary")
    print(f"{'='*80}")
    print(f"Attack Objective: {results['attack_objective']}")
    print(f"Emails Generated: {results['max_emails']}")
    print(f"Session 1 Chunks Stored: {results['session1']['chunks_stored']}")
    print(f"Session 2 Queries: {results['session2']['total_queries']}")
    print(f"Attack Detected: {results['session2']['attack_detected_count']}/{results['session2']['total_queries']}")
    print(f"Attack Success Rate: {results['attack_success_rate']:.2%}")
    print(f"\n{'⚠️  ATTACK SUCCESSFUL' if results['attack_success'] else '✅ Attack Failed (Good!)'}")
    
    return results


if __name__ == "__main__":
    main()

