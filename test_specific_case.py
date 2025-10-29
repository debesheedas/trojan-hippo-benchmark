#!/usr/bin/env python3
"""
Test script for the specific 04_inject_memory.json test case.

This script tests if DSPy can generate a successful attack for this specific test case.
"""

import json
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from optimization_strategies.dspy_optimizer import DSPyOptimizer

def test_04_inject_memory_case():
    """Test DSPy optimization for the 04_inject_memory.json test case."""
    print("🧪 Testing DSPy for 04_inject_memory.json")
    print("=" * 60)
    
    # Load the specific test case
    test_file = Path("attack_bench/indirect/04_inject_memory.json")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✅ Loaded test case: {test_data['name']}")
    print(f"✅ Attack type: {test_data['attack_type']}")
    
    # Extract attack email and step
    attack_email = test_data["initial_data"]["attack_emails"][0]
    step = test_data["steps"][0]
    
    print(f"\n📧 Original Attack Email:")
    print(f"   From: {attack_email['from']}")
    print(f"   Subject: {attack_email['subject']}")
    print(f"   Body: {attack_email['body_plain'][:200]}...")
    
    print(f"\n🎯 Attack Goal:")
    print(f"   Description: {step['attack_goal']['description']}")
    print(f"   Success Check: {step['attack_goal']['success_check']}")
    
    # Load config
    config = {
        "dspy": {
            "enabled": True,
            "max_variations": 3,
            "learning_rate": 0.1
        }
    }
    
    # Create optimizer
    optimizer = DSPyOptimizer(config)
    
    print(f"\n🤖 DSPy Configuration:")
    print(f"   DSPy available: {optimizer.optimizer_module is not None}")
    print(f"   Few-shot examples: {len(optimizer.few_shot_examples)}")
    print(f"   Language model: gpt-4o-mini")
    
    # Test DSPy generation
    print(f"\n🔧 Testing DSPy optimization...")
    
    try:
        # Get context information
        attack_goal = step["attack_goal"]["description"]
        user_message = step["user_message"]
        user_context = f"User: Vince Kaminski, Email: vince.j.kaminski@enron.com. User message: {user_message}"
        
        # Get agent context
        agent_tools = optimizer._get_agent_tools_description()
        system_prompt = optimizer._get_system_prompt()
        failure_reason = "Attack did not trigger the expected tool calls or behavior"
        
        print(f"   Attack goal: {attack_goal}")
        print(f"   User context: {user_context}")
        print(f"   Agent tools: {agent_tools[:100]}...")
        print(f"   System prompt: {system_prompt[:100]}...")
        print(f"   Failure reason: {failure_reason}")
        
        # Generate optimized attack using DSPy
        print(f"\n🤖 Generating optimized attack with DSPy...")
        result = optimizer.optimizer_module(
            original_attack=json.dumps(attack_email),
            attack_goal=attack_goal,
            user_context=user_context,
            agent_tools=agent_tools,
            system_prompt=system_prompt,
            failure_reason=failure_reason
        )
        
        print(f"\n✅ DSPy generation successful!")
        print(f"Generated attack: {result.optimized_attack}")
        
        # Parse the optimized attack
        optimized_attack_data = optimizer._parse_optimized_attack(result.optimized_attack, attack_email)
        print(f"\n📧 Parsed optimized attack email:")
        print(f"   From: {optimized_attack_data.get('from', 'Unknown')}")
        print(f"   Subject: {optimized_attack_data.get('subject', 'Unknown')}")
        print(f"   Body: {optimized_attack_data.get('body_plain', 'Unknown')[:300]}...")
        
        # Compare with original
        print(f"\n🔄 Comparison:")
        print(f"   Original subject: {attack_email['subject']}")
        print(f"   Optimized subject: {optimized_attack_data.get('subject', 'Unknown')}")
        print(f"   Original body length: {len(attack_email['body_plain'])}")
        print(f"   Optimized body length: {len(optimized_attack_data.get('body_plain', ''))}")
        
        return True
        
    except Exception as e:
        print(f"❌ DSPy generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Specific Test Case Analysis")
    print("=" * 60)
    
    try:
        success = test_04_inject_memory_case()
        
        if success:
            print("\n✅ Test completed successfully!")
            print("\nNote: This test only shows DSPy generation capability.")
            print("To test actual success, we need to implement email injection")
            print("and run the attack through the real agent.")
        else:
            print("\n❌ Test failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
