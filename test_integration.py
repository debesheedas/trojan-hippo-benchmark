#!/usr/bin/env python3
"""
Test script for the integrated adaptive benchmark with DSPy optimization.

This script tests the full integration of:
- Environment state management
- DSPy optimization with real agent testing
- State restoration and attack injection
"""

import json
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from test_bench import TestBench

def test_integration():
    """Test the integrated adaptive benchmark."""
    print("🧪 Testing Integrated Adaptive Benchmark")
    print("=" * 60)
    
    # Create test bench with adaptive mode enabled
    bench = TestBench("config.yaml")
    
    if not bench.adaptive_enabled:
        print("❌ Adaptive benchmark not enabled in config")
        return False
    
    print(f"✅ Adaptive benchmark enabled")
    print(f"✅ Available optimizers: {list(bench.optimizers.keys())}")
    print(f"✅ State manager initialized: {bench.state_manager is not None}")
    
    # Test with a specific indirect attack test case
    test_file = Path("attack_bench/indirect/04_inject_memory.json")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"\n🔧 Running integrated test: {test_file.name}")
    
    try:
        # Run the test with full integration
        result = bench.run_test_from_file(test_file)
        
        print(f"\n📊 Test Results:")
        print(f"   Test name: {result['test_name']}")
        print(f"   Overall success: {result['overall_success']}")
        print(f"   Optimization used: {result.get('optimization_used', False)}")
        print(f"   Adaptive benchmark: {result.get('adaptive_benchmark', False)}")
        print(f"   Steps: {len(result.get('steps', []))}")
        
        # Check if optimization was used
        if result.get('optimization_used'):
            print(f"\n🔧 Optimization Details:")
            for i, step in enumerate(result.get('steps', []), 1):
                if 'optimization' in step:
                    opt = step['optimization']
                    print(f"   Step {i}: {opt['strategy']} ({opt['iterations']} iterations)")
                    print(f"     Feedback: {len(opt['feedback'])} messages")
        
        # Check attack goal results
        attack_goals = []
        for step in result.get('steps', []):
            if 'attack_goal' in step:
                attack_goals.append(step['attack_goal']['passed'])
        
        if attack_goals:
            passed_attacks = sum(1 for p in attack_goals if p is True)
            total_attacks = len(attack_goals)
            print(f"\n🎯 Attack Goal Results: {passed_attacks}/{total_attacks} passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        bench.cleanup_all_test_environments()

if __name__ == "__main__":
    print("🚀 Integrated Adaptive Benchmark Test")
    print("=" * 60)
    
    try:
        success = test_integration()
        
        if success:
            print("\n✅ Integration test completed successfully!")
            print("\nNext steps:")
            print("1. Check the detailed logs for optimization attempts")
            print("2. Verify that DSPy generated effective attacks")
            print("3. Check if attacks were cached for future runs")
        else:
            print("\n❌ Integration test failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
