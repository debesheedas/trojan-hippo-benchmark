#!/usr/bin/env python3
"""
Update the persona memory test case to use pre-processed memory.

This script:
1. Updates the test case to use mem0_memory_set_1 (pre-processed)
2. Removes all chat history messages (first 142 steps)
3. Keeps only the questions (after the session break)
"""

import json
from pathlib import Path


def update_test_case(
    test_case_path: Path,
    mem0_memory_set: str = "mem0_memory_set_1"
) -> None:
    """
    Update test case to use pre-processed memory and remove chat history.
    """
    # Load test case
    with open(test_case_path, 'r', encoding='utf-8') as f:
        test_case = json.load(f)
    
    print(f"Original test case: {len(test_case.get('steps', []))} steps")
    
    # Update initial_data to use pre-processed memory
    if "initial_data" not in test_case:
        test_case["initial_data"] = {}
    
    test_case["initial_data"]["mem0_memory_set"] = mem0_memory_set
    print(f"Updated initial_data to use: {mem0_memory_set}")
    
    # Find the session break and keep only steps after it
    steps = test_case.get("steps", [])
    new_steps = []
    
    found_session_break = False
    for step in steps:
        if step.get("step_type") == "start_new_session":
            found_session_break = True
            # Keep the session break
            new_steps.append(step)
            continue
        
        if found_session_break:
            # Keep all steps after session break (these are the questions)
            new_steps.append(step)
        # Skip all steps before session break (these are chat history)
    
    if not found_session_break:
        print("⚠️ Warning: No 'start_new_session' step found. Keeping all steps.")
        new_steps = steps
    
    test_case["steps"] = new_steps
    
    print(f"Updated test case: {len(new_steps)} steps")
    print(f"  Removed chat history messages")
    print(f"  Kept questions: {len([s for s in new_steps if 'user_message' in s and s.get('step_type') != 'start_new_session'])}")
    
    # Update description
    test_case["description"] = (
        "Test case created from ImplicitPersona dataset for persona_id 66. "
        "Uses pre-processed mem0 memory from persona chat history. "
        "Tests if agent can maintain persona-specific memory and provide personalized responses."
    )
    
    # Save updated test case
    with open(test_case_path, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Updated test case: {test_case_path}")


def main():
    """Main function."""
    test_case_path = Path("data/benchmark/attack_bench_mem0/benign/00_persona_memory.json")
    mem0_memory_set = "mem0_memory_set_1"
    
    print("="*80)
    print("Updating Persona Test Case for Pre-processed Memory")
    print("="*80)
    print(f"Test case: {test_case_path}")
    print(f"Memory set: {mem0_memory_set}")
    print("="*80)
    
    if not test_case_path.exists():
        print(f"✗ Error: Test case not found at {test_case_path}")
        return 1
    
    try:
        update_test_case(test_case_path, mem0_memory_set)
        
        print("\n" + "="*80)
        print("✓ Successfully updated test case!")
        print("="*80)
        print("\nThe test case now:")
        print("1. Uses pre-processed mem0_memory_set_1")
        print("2. Has no chat history messages (they're in the pre-processed memory)")
        print("3. Contains only the questions for evaluation")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

