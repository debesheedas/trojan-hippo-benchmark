"""
Utilities for loading and processing MemoryAgentBench dataset for memory evaluation.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

# Add MemoryAgentBench to path if needed
MEMORY_AGENT_BENCH_PATH = Path(__file__).parent.parent.parent / "MemoryAgentBench"
if str(MEMORY_AGENT_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(MEMORY_AGENT_BENCH_PATH))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import datasets library: {e}")
    print("Install with: pip install datasets")
    DATASETS_AVAILABLE = False

# Try to import MemoryAgentBench utilities
try:
    from utils.eval_data_utils import load_eval_data
    from utils.eval_other_utils import chunk_text_into_sentences
    MEMORY_BENCH_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MemoryAgentBench utilities: {e}")
    print("Make sure MemoryAgentBench directory exists and has utils/ subdirectory")
    MEMORY_BENCH_UTILS_AVAILABLE = False


def load_memory_benchmark_dataset(
    dataset_name: str = "Accurate_Retrieval",
    sub_dataset: str = "ruler_qa",
    max_samples: int = 5,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load a subset of the MemoryAgentBench dataset.
    
    Args:
        dataset_name: Main dataset name (Accurate_Retrieval, Test_Time_Learning, 
                     Long_Range_Understanding, Conflict_Resolution)
        sub_dataset: Sub-dataset name (e.g., "ruler_qa", "eventqa", etc.)
        max_samples: Maximum number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of dataset samples, each containing:
        - context: The full context text
        - questions: List of questions
        - answers: List of answers
        - metadata: Additional metadata
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library not available. "
            "Install with: pip install datasets"
        )
    
    if not MEMORY_BENCH_UTILS_AVAILABLE:
        raise ImportError(
            "MemoryAgentBench utilities not available. "
            "Make sure MemoryAgentBench directory exists at the project root."
        )
    
    # Create dataset config
    dataset_config = {
        "dataset": dataset_name,
        "sub_dataset": sub_dataset,
        "max_test_samples": max_samples,
        "seed": seed,
        "context_max_length": 100000,  # Large enough for most contexts
        "chunk_size": 512,
        "generation_max_length": 500,
        "debug": False
    }
    
    # Load the dataset
    loaded_data = load_eval_data(dataset_config)
    
    # Convert to list format
    if isinstance(loaded_data, dict) and "data" in loaded_data:
        dataset_list = list(loaded_data["data"])
    else:
        dataset_list = list(loaded_data)
    
    # Process and format the data
    processed_samples = []
    for item in dataset_list[:max_samples]:
        sample = {
            "context": item.get("context", ""),
            "questions": item.get("questions", []),
            "answers": item.get("answers", []),
            "metadata": item.get("metadata", {})
        }
        processed_samples.append(sample)
    
    return processed_samples


def chunk_context_for_memory(
    context: str,
    chunk_size: int = 512,
    model_name: str = "gpt-4o-mini"
) -> List[str]:
    """
    Chunk a context text into smaller pieces for memory storage.
    
    Args:
        context: The full context text
        chunk_size: Target chunk size in tokens
        model_name: Model name for tokenizer
        
    Returns:
        List of text chunks
    """
    if MEMORY_BENCH_UTILS_AVAILABLE:
        # Use MemoryAgentBench's chunking function, but catch NLTK errors
        try:
            return chunk_text_into_sentences(context, model_name=model_name, chunk_size=chunk_size)
        except LookupError as e:
            if 'punkt' in str(e).lower():
                print(f"Warning: NLTK punkt tokenizer not available. Using fallback chunking.")
                print(f"  To fix: python -c \"import nltk; nltk.download('punkt_tab')\"")
            else:
                raise
        except Exception as e:
            print(f"Warning: Error using MemoryAgentBench chunking: {e}")
            print("  Falling back to simple chunking method.")
    
    # Fallback: simple splitting by sentences
    # Use a more sophisticated approach: split by sentence endings
    import re
    # Split by sentence endings (., !, ?) followed by space or newline
    sentences = re.split(r'([.!?]\s+)', context)
    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Approximate token count (roughly 4 chars per token for English)
    approx_tokens_per_char = 0.25
    
    for sentence in combined_sentences:
        sentence_length = len(sentence) * approx_tokens_per_char
        if current_length + sentence_length > chunk_size * 0.75:  # Use 75% threshold
            if current_chunk:
                chunks.append(''.join(current_chunk).strip())
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(''.join(current_chunk).strip())
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks if chunks else [context]  # Return at least one chunk


def create_test_case(
    sample: Dict[str, Any],
    chunk_size: int = 512
) -> Dict[str, Any]:
    """
    Create a test case structure from a dataset sample.
    
    Args:
        sample: Dataset sample with context, questions, answers
        chunk_size: Chunk size for context splitting
        
    Returns:
        Test case dictionary with:
        - context_chunks: List of context chunks for memory construction
        - queries: List of (query, answer) tuples
        - metadata: Sample metadata
    """
    context = sample["context"]
    questions = sample.get("questions", [])
    answers = sample.get("answers", [])
    metadata = sample.get("metadata", {})
    
    # Chunk the context
    context_chunks = chunk_context_for_memory(context, chunk_size=chunk_size)
    
    # Create query-answer pairs
    queries = []
    if isinstance(questions, list) and isinstance(answers, list):
        # Handle multiple questions/answers
        if len(questions) == len(answers):
            for q, a in zip(questions, answers):
                queries.append((q, a))
        elif len(answers) > 0 and isinstance(answers[0], list):
            # Answers is list of lists
            for i, q in enumerate(questions):
                if i < len(answers):
                    queries.append((q, answers[i] if isinstance(answers[i], list) else [answers[i]]))
        else:
            # Single answer for all questions
            answer = answers[0] if answers else ""
            for q in questions:
                queries.append((q, answer))
    else:
        # Single question/answer
        queries.append((questions if isinstance(questions, str) else questions[0] if questions else "",
                       answers if isinstance(answers, str) else answers[0] if answers else ""))
    
    return {
        "context_chunks": context_chunks,
        "queries": queries,
        "metadata": metadata,
        "original_context": context
    }


def load_test_cases(
    dataset_name: str = "Accurate_Retrieval",
    sub_dataset: str = "ruler_qa",
    num_cases: int = 5,
    chunk_size: int = 512,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load and prepare test cases from MemoryAgentBench dataset.
    
    Args:
        dataset_name: Main dataset name
        sub_dataset: Sub-dataset name
        num_cases: Number of test cases to load
        chunk_size: Chunk size for context splitting
        seed: Random seed
        
    Returns:
        List of test case dictionaries
    """
    samples = load_memory_benchmark_dataset(
        dataset_name=dataset_name,
        sub_dataset=sub_dataset,
        max_samples=num_cases,
        seed=seed
    )
    
    test_cases = []
    for sample in samples:
        test_case = create_test_case(sample, chunk_size=chunk_size)
        test_cases.append(test_case)
    
    return test_cases


def save_test_cases(test_cases: List[Dict[str, Any]], output_path: str):
    """
    Save test cases to a JSON file.
    
    Args:
        test_cases: List of test case dictionaries
        output_path: Path to save the JSON file
    """
    # Convert to JSON-serializable format
    serializable_cases = []
    for case in test_cases:
        serializable_case = {
            "context_chunks": case["context_chunks"],
            "queries": [(q, a) for q, a in case["queries"]],
            "metadata": case["metadata"],
            "original_context": case["original_context"]
        }
        serializable_cases.append(serializable_case)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_cases, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(test_cases)} test cases to {output_path}")


def load_test_cases_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of test case dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

