#!/usr/bin/env python3
"""
Create rag_memory_set_1 from the first MemoryAgentBench test case context.

This script loads the first test case from MemoryAgentBench dataset and creates
a RAG memory set with its entire context.
"""

import sys
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
load_dotenv()

from benchmark.memory_benchmark_utils import load_memory_benchmark_dataset, chunk_context_for_memory
import json


def main():
    print("Loading first test case from MemoryAgentBench dataset...")
    
    # Load the first test case
    # Using a common dataset - you can change this if needed
    samples = load_memory_benchmark_dataset(
        dataset_name="Accurate_Retrieval",
        sub_dataset="ruler_qa1_197K",  # Using the default from config
        max_samples=1,
        seed=42
    )
    
    if not samples:
        print("Error: No samples loaded from dataset")
        return 1
    
    sample = samples[0]
    context = sample.get("context", "")
    
    if not context:
        print("Error: No context found in sample")
        print(f"Sample keys: {sample.keys()}")
        return 1
    
    print(f"Loaded context ({len(context)} characters)")
    print(f"Context preview: {context[:200]}...")
    
    # Get metadata about the sample
    metadata = sample.get("metadata", {})
    dataset_info = {
        "source": "MemoryAgentBench",
        "dataset_name": "Accurate_Retrieval",
        "sub_dataset": "ruler_qa1_197K",
        "sample_index": 0,
        "has_questions": "questions" in sample,
        "num_questions": len(sample.get("questions", [])) if "questions" in sample else 0
    }
    
    if metadata:
        dataset_info.update(metadata)
    
    # Chunk the context
    print(f"\nChunking context (chunk_size=512)...")
    chunks = chunk_context_for_memory(context, chunk_size=512)
    
    print(f"  Context length: {len(context)} characters")
    print(f"  Number of chunks: {len(chunks)}")
    if chunks:
        print(f"  Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")
    
    # Create RAG memory set JSON
    output_path = BASE_DIR / "data/benchmark/initial_rag_memory/rag_memory_set_1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunks_data = []
    for chunk_idx, chunk_text in enumerate(chunks):
        chunks_data.append({
            "text": chunk_text,
            "metadata": {
                "chunk_index": chunk_idx,
                "source": "MemoryAgentBench",
                **dataset_info
            }
        })
    
    data = {
        "chunks": chunks_data,
        "metadata": {
            "name": "rag_memory_set_1",
            "description": "Initial RAG memory from first MemoryAgentBench test case context",
            "total_chunks": len(chunks_data),
            "context_length": len(context),
            "chunk_size": 512,
            **dataset_info
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully created rag_memory_set_1.json")
    print(f"  Output: {output_path}")
    print(f"  Total chunks: {len(chunks_data)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

