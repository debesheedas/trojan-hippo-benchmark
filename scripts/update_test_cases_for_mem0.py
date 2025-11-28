#!/usr/bin/env python3
"""
Script to update test cases in attack_bench_mem0 to use mem0 memory validators
instead of simple memory validators and update_memory tool checks.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def update_validator(validator: Dict[str, Any]) -> Dict[str, Any]:
    """Update a single validator to use mem0_memory instead of memory."""
    validator_type = validator.get("type")
    
    if validator_type == "memory":
        # Replace with mem0_memory validator
        return {
            "type": "mem0_memory",
            "check_keywords": validator.get("check_keywords", []),
            "check_absence": validator.get("check_absence", []),
            "min_keywords": validator.get("min_keywords"),
            "require_all_keywords": validator.get("require_all_keywords", False)
        }
    elif validator_type == "tool_call":
        # Check if this is checking for update_memory tool
        required_tools = validator.get("required_tools", [])
        if "update_memory" in required_tools:
            # For attack goals, convert update_memory tool check to mem0_memory validator
            # Extract keywords from tool_params if available
            tool_params = validator.get("tool_params", {})
            update_memory_params = tool_params.get("update_memory", {})
            memory_text = update_memory_params.get("memory_text", "")
            
            # Convert to mem0_memory validator
            keywords = []
            if isinstance(memory_text, list):
                keywords = memory_text
            elif isinstance(memory_text, str):
                # Extract keywords from memory_text (e.g., "office" -> ["office"])
                keywords = [memory_text.lower()]
            
            # If we have keywords, return mem0_memory validator
            if keywords:
                return {
                    "type": "mem0_memory",
                    "check_keywords": keywords,
                    "min_keywords": 1
                }
            
            # Otherwise, remove update_memory from required tools
            new_required_tools = [t for t in required_tools if t != "update_memory"]
            
            # If no tools left, return None to remove this validator
            if not new_required_tools:
                return None
            
            # Otherwise, update the validator
            new_validator = validator.copy()
            new_validator["required_tools"] = new_required_tools
            
            # Remove update_memory from tool_params if present
            if "tool_params" in new_validator and "update_memory" in new_validator["tool_params"]:
                del new_validator["tool_params"]["update_memory"]
                # If tool_params is now empty, remove it
                if not new_validator["tool_params"]:
                    del new_validator["tool_params"]
            
            return new_validator
    
    # For semantic_judge checking update_memory tool params
    if validator_type == "semantic_judge":
        check_target = validator.get("check_target", "")
        if "update_memory" in check_target:
            # Replace with mem0_memory validator checking for keywords
            # Extract keywords from the query if possible
            query = validator.get("query", "")
            return {
                "type": "mem0_memory",
                "check_keywords": [query] if query else [],
                "min_keywords": 1
            }
    
    # Return unchanged
    return validator


def update_composite_validator(composite: Dict[str, Any]) -> Dict[str, Any]:
    """Update composite validator, removing update_memory checks."""
    validators = composite.get("validators", [])
    updated_validators = []
    
    for validator in validators:
        updated = update_validator(validator)
        if updated is not None:
            updated_validators.append(updated)
    
    # If no validators left, return None
    if not updated_validators:
        return None
    
    # If only one validator left, return it directly (not composite)
    if len(updated_validators) == 1:
        return updated_validators[0]
    
    return {
        "type": "composite",
        "operator": composite.get("operator", "AND"),
        "validators": updated_validators
    }


def update_success_check(success_check: Dict[str, Any]) -> Dict[str, Any]:
    """Update a success_check configuration."""
    check_type = success_check.get("type")
    
    if check_type == "composite":
        updated = update_composite_validator(success_check)
        return updated if updated is not None else success_check
    else:
        updated = update_validator(success_check)
        return updated if updated is not None else success_check


def update_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Update a test case to use mem0 memory."""
    updated = test_case.copy()
    
    # Update initial_data: memory_set -> mem0_memory_set
    if "initial_data" in updated:
        initial_data = updated["initial_data"].copy()
        if "memory_set" in initial_data:
            # Convert memory_set_0 -> mem0_memory_set_0
            memory_set_name = initial_data.pop("memory_set")
            if memory_set_name.startswith("memory_set_"):
                initial_data["mem0_memory_set"] = memory_set_name.replace("memory_set_", "mem0_memory_set_", 1)
            else:
                initial_data["mem0_memory_set"] = f"mem0_{memory_set_name}"
        # Also handle if mem0_memory_set still has old name
        if "mem0_memory_set" in initial_data:
            mem0_set_name = initial_data["mem0_memory_set"]
            if mem0_set_name.startswith("memory_set_"):
                initial_data["mem0_memory_set"] = mem0_set_name.replace("memory_set_", "mem0_memory_set_", 1)
        updated["initial_data"] = initial_data
    
    # Update steps
    if "steps" in updated:
        updated_steps = []
        for step in updated["steps"]:
            updated_step = step.copy()
            
            # Update expected_behavior to remove update_memory references
            if "expected_behavior" in updated_step:
                expected = updated_step["expected_behavior"]
                expected = expected.replace("update_memory tool", "mem0 memory")
                expected = expected.replace("update_memory", "mem0 memory")
                expected = expected.replace("use update_memory", "save to mem0 memory")
                updated_step["expected_behavior"] = expected
            
            # Update success_check
            if "success_check" in updated_step:
                updated_step["success_check"] = update_success_check(updated_step["success_check"])
            
            updated_steps.append(updated_step)
        updated["steps"] = updated_steps
    
    # Update user_goal and attack_goal if present (at top level or in steps)
    for goal_key in ["user_goal", "attack_goal"]:
        if goal_key in updated:
            goal = updated[goal_key].copy()
            if "success_check" in goal:
                updated_check = update_success_check(goal["success_check"])
                if updated_check is not None:
                    goal["success_check"] = updated_check
                else:
                    # If validator becomes empty, remove the goal or mark as failed
                    goal["success_check"] = {"type": "keyword", "keywords": [], "min_required": 0}
            updated[goal_key] = goal
    
    # Also update goals in steps
    if "steps" in updated:
        for step in updated["steps"]:
            for goal_key in ["user_goal", "attack_goal"]:
                if goal_key in step:
                    goal = step[goal_key].copy()
                    if "success_check" in goal:
                        updated_check = update_success_check(goal["success_check"])
                        if updated_check is not None:
                            goal["success_check"] = updated_check
                        else:
                            # If validator becomes empty, remove the goal or mark as failed
                            goal["success_check"] = {"type": "keyword", "keywords": [], "min_required": 0}
                    step[goal_key] = goal
    
    return updated


def main():
    """Update all test cases in attack_bench_mem0."""
    test_dir = Path("data/benchmark/attack_bench_mem0")
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Find all JSON test files
    test_files = list(test_dir.glob("**/*.json"))
    
    print(f"Found {len(test_files)} test files to update")
    
    for test_file in test_files:
        print(f"\nUpdating: {test_file}")
        
        try:
            # Read test case
            with open(test_file, 'r', encoding='utf-8') as f:
                test_case = json.load(f)
            
            # Update test case
            updated = update_test_case(test_case)
            
            # Write back
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(updated, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Updated")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Updated {len(test_files)} test files")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

