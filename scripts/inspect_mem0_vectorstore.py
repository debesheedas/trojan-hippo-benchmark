#!/usr/bin/env python3
"""
Direct inspection of mem0 FAISS vectorstore files.
"""

import pickle
import json
from pathlib import Path

vectorstore_path = Path("data/interactive_agent/mem0_vectorstore")
pkl_file = vectorstore_path / "mem0_memories.pkl"

if pkl_file.exists():
    print(f"Reading {pkl_file}...\n")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print("=" * 80)
    print("PKL FILE CONTENTS")
    print("=" * 80)
    print(f"Type: {type(data)}")
    print()
    
    if isinstance(data, tuple):
        print(f"Tuple length: {len(data)}")
        docstore, index_to_id = data[0], data[1]
        print(f"\nDocstore type: {type(docstore)}")
        print(f"Docstore size: {len(docstore) if isinstance(docstore, dict) else 'N/A'}")
        print(f"Index to ID type: {type(index_to_id)}")
        print(f"Index to ID size: {len(index_to_id) if isinstance(index_to_id, dict) else 'N/A'}")
        
        if isinstance(docstore, dict):
            print("\n" + "=" * 80)
            print("DOCSTORE CONTENTS")
            print("=" * 80)
            for i, (mem_id, mem_data) in enumerate(list(docstore.items())[:10], 1):
                print(f"\nMemory {i}:")
                print(f"  ID: {mem_id}")
                print(f"  Type: {type(mem_data)}")
                if isinstance(mem_data, dict):
                    print(f"  Keys: {list(mem_data.keys())}")
                    for key, value in mem_data.items():
                        if key == "metadata" and isinstance(value, dict):
                            print(f"  {key}: {json.dumps(value, indent=4, ensure_ascii=False)}")
                        elif isinstance(value, str) and len(value) > 200:
                            print(f"  {key}: {value[:200]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Data: {str(mem_data)[:500]}")
        
        if isinstance(index_to_id, dict):
            print("\n" + "=" * 80)
            print("INDEX TO ID MAPPING (first 10)")
            print("=" * 80)
            for i, (idx, mem_id) in enumerate(list(index_to_id.items())[:10], 1):
                print(f"  Index {idx} -> {mem_id}")
    else:
        print("Data structure:")
        print(json.dumps(data, indent=2, default=str, ensure_ascii=False)[:2000])
else:
    print(f"File not found: {pkl_file}")

