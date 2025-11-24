#!/usr/bin/env python3
"""
Multi-Test-Case RAG Poisoning Attack Runner

Runs multiple RAG poisoning attack test cases and aggregates results.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
load_dotenv()

from agent.utils import load_config
from benchmark.rag_poisoning_attack import run_rag_poisoning_test_case
from benchmark.rag_poisoning_test_cases import get_all_test_case_ids, get_test_case


def run_multiple_test_cases(
    test_case_ids: List[str],
    max_emails: int = 5,
    max_email_length: int = 1000,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run multiple test cases and aggregate results.
    
    Args:
        test_case_ids: List of test case IDs to run
        max_emails: Maximum number of attack emails per test case
        max_email_length: Maximum email length per test case
        config: Configuration dictionary
    
    Returns:
        Dictionary with aggregated results
    """
    if config is None:
        config = load_config()
    
    all_results = {
        "test_cases": {},
        "summary": {
            "total_test_cases": len(test_case_ids),
            "successful_attacks": 0,
            "failed_attacks": 0,
            "total_queries": 0,
            "total_attack_detections": 0
        }
    }
    
    for test_case_id in test_case_ids:
        print(f"\n{'#'*80}")
        print(f"# Running Test Case: {test_case_id}")
        print(f"{'#'*80}\n")
        
        try:
            test_case = get_test_case(test_case_id)
            print(f"Test Case: {test_case['name']}")
            print(f"Description: {test_case['description']}")
            print(f"Attack Objective: {test_case['attack_objective']}\n")
            
            results = run_rag_poisoning_test_case(
                max_emails=max_emails,
                max_email_length=max_email_length,
                config=config,
                test_case_id=test_case_id
            )
            
            all_results["test_cases"][test_case_id] = results
            
            # Update summary
            if results.get("attack_success", False):
                all_results["summary"]["successful_attacks"] += 1
            else:
                all_results["summary"]["failed_attacks"] += 1
            
            session2_total = results.get("session2", {}).get("total_queries", 0)
            session2_detected = results.get("session2", {}).get("attack_detected_count", 0)
            
            all_results["summary"]["total_queries"] += session2_total
            all_results["summary"]["total_attack_detections"] += session2_detected
            
        except Exception as e:
            print(f"❌ Error running test case {test_case_id}: {e}")
            import traceback
            traceback.print_exc()
            all_results["test_cases"][test_case_id] = {
                "error": str(e),
                "attack_success": False
            }
            all_results["summary"]["failed_attacks"] += 1
    
    # Calculate overall success rate
    if all_results["summary"]["total_queries"] > 0:
        all_results["summary"]["overall_attack_success_rate"] = (
            all_results["summary"]["total_attack_detections"] / 
            all_results["summary"]["total_queries"]
        )
    else:
        all_results["summary"]["overall_attack_success_rate"] = 0.0
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run multiple RAG poisoning attack test cases")
    parser.add_argument("--test-cases", type=str, nargs="+", default=None,
                       help="Test case IDs to run (e.g., 'test_case_1 test_case_2 test_case_3')")
    parser.add_argument("--all", action="store_true",
                       help="Run all available test cases")
    parser.add_argument("--max-emails", type=int, default=5,
                       help="Maximum number of attack emails per test case")
    parser.add_argument("--max-email-length", type=int, default=1000,
                       help="Maximum email length per test case")
    parser.add_argument("--output-dir", type=str, default="data/benchmark/rag_poisoning_results",
                       help="Output directory for results")
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
    
    # Determine which test cases to run
    if args.all:
        test_case_ids = get_all_test_case_ids()
    elif args.test_cases:
        test_case_ids = args.test_cases
    else:
        # Default: run first 3 test cases
        all_ids = get_all_test_case_ids()
        test_case_ids = all_ids[:3] if len(all_ids) >= 3 else all_ids
        print(f"No test cases specified. Running default: {test_case_ids}")
    
    # Load config
    config = load_config()
    
    # Run test cases
    print(f"\n{'='*80}")
    print(f"Multi-Test-Case RAG Poisoning Attack Runner")
    print(f"{'='*80}")
    print(f"Test Cases: {', '.join(test_case_ids)}")
    print(f"Max Emails: {args.max_emails}")
    print(f"Max Email Length: {args.max_email_length}")
    print(f"{'='*80}\n")
    
    results = run_multiple_test_cases(
        test_case_ids=test_case_ids,
        max_emails=args.max_emails,
        max_email_length=args.max_email_length,
        config=config
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"rag_poisoning_multi_{timestamp_str}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")
    
    # Print summary
    print(f"{'='*80}")
    print(f"Overall Summary")
    print(f"{'='*80}")
    print(f"Total Test Cases: {results['summary']['total_test_cases']}")
    print(f"Successful Attacks: {results['summary']['successful_attacks']}")
    print(f"Failed Attacks: {results['summary']['failed_attacks']}")
    print(f"Total Queries: {results['summary']['total_queries']}")
    print(f"Total Attack Detections: {results['summary']['total_attack_detections']}")
    if results['summary']['total_queries'] > 0:
        print(f"Overall Attack Success Rate: {results['summary']['overall_attack_success_rate']:.2%}")
    
    print(f"\n{'='*80}")
    print(f"Per-Test-Case Results")
    print(f"{'='*80}")
    for test_case_id, test_results in results['test_cases'].items():
        if "error" in test_results:
            print(f"\n{test_case_id}: ❌ ERROR - {test_results['error']}")
        else:
            attack_obj = test_results.get('attack_objective', 'N/A')
            success = test_results.get('attack_success', False)
            detected = test_results.get('session2', {}).get('attack_detected_count', 0)
            total = test_results.get('session2', {}).get('total_queries', 0)
            rate = test_results.get('attack_success_rate', 0.0)
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"\n{test_case_id}: {status}")
            print(f"  Objective: {attack_obj}")
            print(f"  Detections: {detected}/{total} ({rate:.2%})")
    
    return results


if __name__ == "__main__":
    main()

