#!/usr/bin/env python3
"""
Memory Benchmark Runner
Tests the email agent's memory capabilities using MemoryAgentBench dataset.

Usage:
    python memory_benchmark_runner.py [--rag-memory] [--simple-memory] [--num-cases N]
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

from agent.agent_core import invoke_agent, clear_agent_cache, clear_session_agent
from agent.utils import load_config
from benchmark.memory_benchmark_utils import load_test_cases, save_test_cases
from benchmark.memory_metrics import evaluate_response, aggregate_metrics


def run_memory_benchmark(
    test_cases: List[Dict[str, Any]],
    config: Dict[str, Any],
    use_rag_memory: bool = True,
    use_simple_memory: bool = False,
    output_dir: str = "data/benchmark/memory_benchmark_results",
    max_queries_per_case: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run memory benchmark on test cases.
    
    Args:
        test_cases: List of test case dictionaries
        config: Agent configuration dictionary
        use_rag_memory: Whether to use RAG memory
        use_simple_memory: Whether to use simple memory
        output_dir: Directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    # Update config with memory settings
    if "memory" not in config:
        config["memory"] = {}
    
    config["memory"]["rag_memory"] = {
        "enabled": use_rag_memory,
        "embedding_model": "text-embedding-3-small",
        "top_k": 3,
        "chunk_size": 512,
        "vectorstore_path": "data/interactive_agent/rag_vectorstore"
    }
    
    config["memory"]["simple_memory"] = {
        "enabled": use_simple_memory,
        "memory_file": "data/interactive_agent/agent_memory.json",
        "max_short_term": 15
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_evaluations = []
    
    print(f"\n{'='*80}")
    print(f"Running Memory Benchmark")
    print(f"RAG Memory: {'Enabled' if use_rag_memory else 'Disabled'}")
    print(f"Simple Memory: {'Enabled' if use_simple_memory else 'Disabled'}")
    print(f"Test Cases: {len(test_cases)}")
    print(f"{'='*80}\n")
    
    for case_idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {case_idx}/{len(test_cases)}")
        print(f"{'='*80}")
        
        context_chunks = test_case["context_chunks"]
        queries = test_case["queries"]
        
        # Limit queries per case if specified
        if max_queries_per_case is not None and max_queries_per_case > 0:
            original_count = len(queries)
            queries = queries[:max_queries_per_case]
            if len(queries) < original_count:
                print(f"  Limited queries: {original_count} -> {len(queries)} (max_queries_per_case={max_queries_per_case})")
        
        # Create a unique session for this test case
        session_id = f"memory_benchmark_case_{case_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Clear agent cache for clean state
        clear_agent_cache()
        clear_session_agent(session_id, config=config, auto_store_rag=False)  # Don't auto-store for benchmark sessions
        
        # Phase 1: Memory Construction - Add context chunks to memory
        print(f"\n[Phase 1] Memory Construction: Adding {len(context_chunks)} chunks...")
        rag_memory_manager = None
        if use_rag_memory:
            try:
                from agent.backend.rag_memory_manager import get_rag_memory_manager
                rag_config = config["memory"]["rag_memory"]
                rag_memory_manager = get_rag_memory_manager(
                    embedding_model=rag_config["embedding_model"],
                    top_k=rag_config["top_k"],
                    chunk_size=rag_config["chunk_size"],
                    vectorstore_path=f"{rag_config['vectorstore_path']}_case_{case_idx}",
                    force_new=True
                )
                print(f"  Initialized RAG memory manager for case {case_idx}")
            except Exception as e:
                error_msg = f"ERROR: Could not initialize RAG memory: {e}"
                print(f"  {error_msg}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to initialize RAG memory manager. "
                    f"This is a critical error. {error_msg}"
                ) from e
        
        for chunk_idx, chunk in enumerate(context_chunks, 1):
            if use_rag_memory and rag_memory_manager:
                try:
                    rag_memory_manager.add_memory(chunk, metadata={
                        "chunk_id": chunk_idx,
                        "test_case": case_idx
                    })
                    if chunk_idx % 5 == 0 or chunk_idx == len(context_chunks):
                        print(f"  Added chunk {chunk_idx}/{len(context_chunks)} (length: {len(chunk)} chars)")
                except Exception as e:
                    error_msg = f"ERROR adding chunk {chunk_idx} to RAG memory: {e}"
                    print(f"  {error_msg}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"Failed to add context chunks to RAG memory. "
                        f"This is a critical error. {error_msg}"
                    ) from e
            
            if use_simple_memory:
                # For simple memory, we could add a summary or key facts
                # For now, we'll skip this as simple memory is more for user preferences
                pass
        
        print(f"[Phase 1] Complete: {len(context_chunks)} chunks added to memory")
        
        # Phase 2: Query Execution - Ask questions and evaluate responses
        print(f"\n[Phase 2] Query Execution: Processing {len(queries)} queries...")
        case_results = []
        case_evaluations = []
        
        for query_idx, (query, answer) in enumerate(queries, 1):
            print(f"\n  Query {query_idx}/{len(queries)}")
            print(f"    Query: {query[:150]}{'...' if len(query) > 150 else ''}")
            print(f"    Expected Answer: {str(answer)[:100]}{'...' if len(str(answer)) > 100 else ''}")
            
            try:
                # Invoke agent with the query
                print(f"    Invoking agent...")
                response = invoke_agent(
                    text=query,
                    session_id=session_id,
                    config=config
                )
                
                if not response:
                    raise ValueError("Agent returned None or empty response")
                
                agent_response = response.get("response", "")
                if not agent_response:
                    raise ValueError("Agent response is empty")
                
                print(f"    Agent Response: {agent_response[:200]}{'...' if len(agent_response) > 200 else ''}")
                
                # Evaluate the response
                evaluation = evaluate_response(
                    agent_response=agent_response,
                    ground_truth=answer,
                    query=query
                )
                
                case_evaluations.append(evaluation)
                
                # Store result
                result = {
                    "test_case": case_idx,
                    "query_idx": query_idx,
                    "query": query,
                    "ground_truth": answer,
                    "prediction": agent_response,
                    "metrics": evaluation["metrics"],
                    "session_id": session_id,
                    "parsed_output": evaluation.get("parsed_output")
                }
                case_results.append(result)
                
                # Print metrics
                metrics = evaluation["metrics"]
                print(f"    Metrics:")
                print(f"      Exact Match: {metrics.get('exact_match', 0):.4f}")
                print(f"      F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"      Substring Match: {metrics.get('substring_exact_match', 0):.4f}")
                if 'rougeL_f1' in metrics:
                    print(f"      ROUGE-L F1: {metrics.get('rougeL_f1', 0):.4f}")
                
            except Exception as e:
                print(f"    ERROR processing query: {e}")
                import traceback
                error_traceback = traceback.format_exc()
                print(f"    Full traceback:\n{error_traceback}")
                
                # Store error result
                result = {
                    "test_case": case_idx,
                    "query_idx": query_idx,
                    "query": query,
                    "ground_truth": answer,
                    "prediction": "",
                    "error": str(e),
                    "error_traceback": error_traceback,
                    "metrics": {}
                }
                case_results.append(result)
                
                # Fail fast: raise exception to stop benchmark
                raise RuntimeError(
                    f"Benchmark failed at test case {case_idx}, query {query_idx}. "
                    f"Error: {e}\n\nThis indicates a critical error that prevents the benchmark from continuing. "
                    f"Please fix the issue before re-running."
                ) from e
        
        all_results.extend(case_results)
        all_evaluations.extend(case_evaluations)
        
        # Calculate case-level metrics
        if case_evaluations:
            case_metrics = aggregate_metrics(case_evaluations)
            print(f"\n  Case {case_idx} Summary:")
            print(f"    Queries Processed: {len(case_evaluations)}/{len(queries)}")
            print(f"    Average Metrics:")
            for metric_name, metric_value in sorted(case_metrics.items()):
                print(f"      {metric_name}: {metric_value:.4f}")
        else:
            print(f"\n  Case {case_idx} Warning: No evaluations completed!")
    
    # Calculate overall metrics
    overall_metrics = aggregate_metrics(all_evaluations) if all_evaluations else {}
    
    # Prepare final results
    benchmark_results = {
        "config": {
            "use_rag_memory": use_rag_memory,
            "use_simple_memory": use_simple_memory,
            "num_test_cases": len(test_cases),
            "max_queries_per_case": max_queries_per_case,
            "total_queries": len(all_results)
        },
        "overall_metrics": overall_metrics,
        "results": all_results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_type = "rag" if use_rag_memory else "simple" if use_simple_memory else "none"
    results_file = output_path / f"memory_benchmark_{memory_type}_{timestamp_str}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Benchmark Complete!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Test Cases: {len(test_cases)}")
    print(f"  Total Queries: {len(all_results)}")
    print(f"  Successful Evaluations: {len(all_evaluations)}")
    print(f"\nOverall Metrics:")
    if overall_metrics:
        for metric_name, metric_value in sorted(overall_metrics.items()):
            print(f"  {metric_name}: {metric_value:.4f}")
    else:
        print("  No metrics calculated (no successful evaluations)")
    print(f"\nResults saved to: {results_file}")
    print(f"  File size: {results_file.stat().st_size / 1024:.2f} KB")
    
    return benchmark_results


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run memory benchmark on email agent")
    parser.add_argument("--rag-memory", action="store_true", help="Enable RAG memory")
    parser.add_argument("--simple-memory", action="store_true", help="Enable simple memory")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of test cases to run")
    parser.add_argument("--max-queries-per-case", type=int, default=None,
                       help="Maximum number of queries to run per test case (default: all queries)")
    parser.add_argument("--dataset", type=str, default="Accurate_Retrieval", 
                       help="Dataset name (Accurate_Retrieval, Test_Time_Learning, etc.)")
    parser.add_argument("--sub-dataset", type=str, default="ruler_qa1_197K",
                       help="Sub-dataset name (e.g., ruler_qa1_197K, eventqa_full, etc.)")
    parser.add_argument("--output-dir", type=str, default="data/benchmark/memory_benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--load-cached", type=str, default=None,
                       help="Load test cases from cached file instead of downloading")
    parser.add_argument("--attack-cases", type=str, default=None,
                       help="Path to JSON file with attack test cases (overrides dataset loading)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available sub-datasets and exit")
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("Loading dataset to discover available sub-datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset('ai-hyz/MemoryAgentBench', split=args.dataset)
            sources = set()
            for item in ds:
                source = item.get('metadata', {}).get('source', '')
                if source:
                    sources.add(source)
            print(f"\nAvailable sub-datasets for '{args.dataset}':")
            for source in sorted(sources):
                count = sum(1 for item in ds if item.get('metadata', {}).get('source', '') == source)
                print(f"  - {source} ({count} samples)")
            return None
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return None
    
    # Load configuration
    config = load_config()
    
    # Ensure config has required "data" section for trace file
    if "data" not in config:
        config["data"] = {}
    if "trace_file" not in config["data"]:
        config["data"]["trace_file"] = "data/interactive_agent/trace.jsonl"
    
    # Load test cases
    if args.attack_cases:
        print(f"Loading attack test cases from {args.attack_cases}...")
        try:
            from benchmark.attack_test_case_generator import load_attack_test_cases
            all_attack_cases = load_attack_test_cases(args.attack_cases)
            # Limit to requested number
            test_cases = all_attack_cases[:args.num_cases] if args.num_cases else all_attack_cases
            print(f"Loaded {len(test_cases)} attack test cases")
            if len(all_attack_cases) > len(test_cases):
                print(f"  (Limited from {len(all_attack_cases)} total cases)")
        except Exception as e:
            print(f"Error loading attack test cases: {e}")
            import traceback
            traceback.print_exc()
            return None
    elif args.load_cached:
        print(f"Loading test cases from {args.load_cached}...")
        from benchmark.memory_benchmark_utils import load_test_cases_from_file
        test_cases = load_test_cases_from_file(args.load_cached)
        if len(test_cases) < args.num_cases:
            print(f"Warning: Cached file only has {len(test_cases)} test cases, but {args.num_cases} requested.")
    else:
        print(f"Loading {args.num_cases} test cases from dataset...")
        try:
            test_cases = load_test_cases(
                dataset_name=args.dataset,
                sub_dataset=args.sub_dataset,
                num_cases=args.num_cases,
                chunk_size=512
            )
            
            if len(test_cases) == 0:
                print(f"\nERROR: No test cases found for sub-dataset '{args.sub_dataset}'")
                print(f"  Try running with --list-datasets to see available options")
                return None
            
            if len(test_cases) < args.num_cases:
                print(f"\nWarning: Only {len(test_cases)} test cases available for '{args.sub_dataset}'")
                print(f"  Requested {args.num_cases}, but will use {len(test_cases)} available cases")
                print(f"  Try a different sub-dataset or use --list-datasets to see options")
            
            # Save test cases for future use
            cache_file = Path(args.output_dir) / "test_cases_cache.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            save_test_cases(test_cases, str(cache_file))
            print(f"Test cases cached to {cache_file}")
        except Exception as e:
            print(f"Error loading test cases: {e}")
            print("\nTroubleshooting:")
            print("  1. Make sure MemoryAgentBench directory exists at the project root")
            print("  2. Check that datasets library is installed: pip install datasets")
            print("  3. Try listing available sub-datasets: --list-datasets")
            print(f"  4. Current sub-dataset: '{args.sub_dataset}'")
            print(f"  5. Current dataset: '{args.dataset}'")
            import traceback
            traceback.print_exc()
            return None
    
    # Run benchmark
    results = run_memory_benchmark(
        test_cases=test_cases,
        config=config,
        use_rag_memory=args.rag_memory,
        use_simple_memory=args.simple_memory,
        output_dir=args.output_dir,
        max_queries_per_case=args.max_queries_per_case
    )
    
    return results


if __name__ == "__main__":
    main()

