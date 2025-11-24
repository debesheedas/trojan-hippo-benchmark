#!/usr/bin/env python3
"""
Create a RAG bench test case for evaluating 5 queries from MemoryAgentBench context.

This creates a test case that uses rag_memory_set_1 (which contains the full context)
and evaluates 5 queries with proper MemoryAgentBench metrics.
"""

import sys
import json
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
load_dotenv()

from benchmark.memory_benchmark_utils import load_memory_benchmark_dataset


def main():
    print("Loading first test case from MemoryAgentBench dataset...")
    
    # Load the first test case
    samples = load_memory_benchmark_dataset(
        dataset_name="Accurate_Retrieval",
        sub_dataset="ruler_qa1_197K",
        max_samples=1,
        seed=42
    )
    
    if not samples:
        print("Error: No samples loaded from dataset")
        return 1
    
    sample = samples[0]
    questions = sample.get("questions", [])
    answers = sample.get("answers", [])
    
    if not questions or not answers:
        print("Error: No questions or answers found in sample")
        return 1
    
    # Take first 5 questions
    num_queries = 5
    selected_questions = questions[:num_queries]
    selected_answers = answers[:num_queries]
    
    print(f"Selected {len(selected_questions)} questions from {len(questions)} total")
    
    # Create test case structure
    test_case = {
        "name": "00_eval_memoryagentbench_queries.json",
        "attack_type": "evaluation",
        "description": "Evaluation test case: 5 queries from MemoryAgentBench context to evaluate RAG memory retrieval and response quality using standard metrics (EM, F1, ROUGE).",
        "initial_data": {
            "inbox_set": "inbox_set_1",
            "outbox_set": "outbox_set_1",
            "drafts_set": "drafts_set_1",
            "memory_set": "memory_set_0",
            "rag_memory_set": "rag_memory_set_1"
        },
        "rag_memory": {
            "enabled": True,
            "vectorstore_path": "data/interactive_agent/rag_vectorstore"
        },
        "evaluation": {
            "metrics": ["exact_match", "f1", "rouge_l", "rouge_lsum"],
            "ground_truth": []
        },
        "steps": []
    }
    
    # Add evaluation queries as steps
    for i, (question, answer) in enumerate(zip(selected_questions, selected_answers), 1):
        # Handle answer format (could be string or list)
        if isinstance(answer, list):
            ground_truth = answer
        else:
            ground_truth = [answer] if answer else []
        
        step = {
            "step_type": "evaluation_query",
            "description": f"Evaluation query {i}/{num_queries} from MemoryAgentBench",
            "user_message": question,
            "expected_behavior": "Agent should retrieve relevant context from RAG memory and provide accurate answer.",
            "evaluation": {
                "ground_truth": ground_truth,
                "metrics": ["exact_match", "f1", "rouge_l", "rouge_lsum"]
            }
        }
        test_case["steps"].append(step)
        test_case["evaluation"]["ground_truth"].append(ground_truth)
    
    # Save test case
    output_path = BASE_DIR / "data/benchmark/rag_bench/00_eval_memoryagentbench_queries.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully created test case: {output_path}")
    print(f"  Queries: {num_queries}")
    print(f"  Uses rag_memory_set_1 (MemoryAgentBench context)")
    print(f"  Evaluation metrics: EM, F1, ROUGE-L, ROUGE-Lsum")
    
    # Show sample query
    print(f"\nSample query:")
    print(f"  Q: {selected_questions[0][:100]}...")
    print(f"  A: {str(selected_answers[0])[:100]}...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

