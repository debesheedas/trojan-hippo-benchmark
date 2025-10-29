#!/usr/bin/env python3
"""
Pure-Python Test Bench for Email Agent

This test bench runs the same tests as test_suite.py but uses the pure-Python
agent core (no HTTP) for maximum speed and reliability.

Test Structure:
- Each test is a JSON file with a series of user queries
- Agent is loaded fresh for each test with clean session history
- Existing long-term memory is preserved across tests
- Results are automatically validated and reported
"""

import json
import time
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from agent_core import invoke_agent, clear_session_agent, clear_agent_cache
from utils import read_trace_events, load_config, ensure_data_directories
from test_validators import create_validator


class TestBench:
    """Pure-Python test bench for email agent."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        # Read model name strictly from config.yaml
        self.model_name = (self.config.get("model", {}) or {}).get("model_name")
        if not self.model_name:
            raise ValueError("Config missing model.model_name. Please set it in config.yaml.")
        self.results_dir = Path("test_bench_results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_dirs = []  # Track test directories for cleanup
    
    def create_test_environment(self, test_name: str, initial_data: Dict[str, str] = None) -> Dict[str, str]:
        """
        Create an isolated test environment with specified initial data.
        
        Args:
            test_name: Name of the test (for logging)
            initial_data: Dict with keys 'inbox_set', 'outbox_set', 'drafts_set', 'memory_set', 'session_set'
        
        Returns:
            Config dict with test-specific paths.
        """
        # Create unique test directory
        test_id = str(uuid.uuid4())[:8]
        test_dir = Path(f"test_env_{test_id}")
        test_dir.mkdir(exist_ok=True)
        self.test_dirs.append(test_dir)
        
        # Create subdirectories
        inbox_dir = test_dir / "inbox"
        outbox_dir = test_dir / "outbox"
        drafts_dir = test_dir / "drafts"
        sessions_dir = test_dir / "sessions"
        traces_dir = test_dir / "traces"
        
        for dir_path in [inbox_dir, outbox_dir, drafts_dir, sessions_dir, traces_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Copy initial data based on configuration
        if initial_data:
            # Copy inbox data
            if "inbox_set" in initial_data:
                inbox_set = initial_data["inbox_set"]
                source_inbox = Path(f"data/initial_inbox/{inbox_set}")
                if source_inbox.exists():
                    for email_file in source_inbox.glob("*.json"):
                        shutil.copy2(email_file, inbox_dir)
                else:
                    print(f"Warning: Inbox set '{inbox_set}' not found at {source_inbox}")
            
            # Copy outbox data
            if "outbox_set" in initial_data:
                outbox_set = initial_data["outbox_set"]
                source_outbox = Path(f"data/initial_outbox/{outbox_set}")
                if source_outbox.exists():
                    for email_file in source_outbox.glob("*.json"):
                        shutil.copy2(email_file, outbox_dir)
                else:
                    print(f"Warning: Outbox set '{outbox_set}' not found at {source_outbox}")
            
            # Copy drafts data
            if "drafts_set" in initial_data:
                drafts_set = initial_data["drafts_set"]
                source_drafts = Path(f"data/initial_drafts/{drafts_set}")
                if source_drafts.exists():
                    for email_file in source_drafts.glob("*.json"):
                        shutil.copy2(email_file, drafts_dir)
                else:
                    print(f"Warning: Drafts set '{drafts_set}' not found at {source_drafts}")
            
            # Add attack emails to inbox (for attack benchmarks)
            if "attack_emails" in initial_data:
                attack_emails = initial_data["attack_emails"]
                for attack_email in attack_emails:
                    # Generate unique filename for attack email
                    attack_email_id = str(uuid.uuid4())[:8]
                    attack_email_file = inbox_dir / f"attack_{attack_email_id}.json"
                    
                    # Write attack email to inbox
                    with open(attack_email_file, 'w', encoding='utf-8') as f:
                        json.dump(attack_email, f, indent=2, ensure_ascii=False)
                    
                    print(f"Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
        else:
            # Fallback to original behavior if no initial_data specified
            original_mailbox = Path("data/mailbox")
            if original_mailbox.exists():
                for email_file in original_mailbox.glob("*.json"):
                    shutil.copy2(email_file, inbox_dir)
        
        # Copy agent memory file to test environment
        if initial_data and "memory_set" in initial_data:
            # Copy from specified memory set
            memory_set = initial_data["memory_set"]
            source_memory = Path(f"data/initial_memory/{memory_set}.json")
            if source_memory.exists():
                shutil.copy2(source_memory, test_dir / "agent_memory.json")
            else:
                print(f"Warning: Memory set '{memory_set}' not found at {source_memory}")
                # Create empty memory file as fallback
                empty_memory = {
                    "long_term": []
                }
                with open(test_dir / "agent_memory.json", 'w', encoding='utf-8') as f:
                    json.dump(empty_memory, f, indent=2)
        else:
            # Fallback to original behavior
            original_memory = Path("data/agent_memory.json")
            if original_memory.exists():
                shutil.copy2(original_memory, test_dir / "agent_memory.json")
            else:
                # Create empty memory file if it doesn't exist
                empty_memory = {
                    "long_term": []
                }
                with open(test_dir / "agent_memory.json", 'w', encoding='utf-8') as f:
                    json.dump(empty_memory, f, indent=2)
        
        # Copy initial session data to test environment
        if initial_data and "session_set" in initial_data:
            session_set = initial_data["session_set"]
            source_sessions = Path(f"data/initial_sessions/{session_set}.json")
            if source_sessions.exists():
                shutil.copy2(source_sessions, sessions_dir / "session_index.json")
            else:
                print(f"Warning: Session set '{session_set}' not found at {source_sessions}")
                # Create empty session file as fallback
                empty_sessions = {
                    "sessions": []
                }
                with open(sessions_dir / "session_index.json", 'w', encoding='utf-8') as f:
                    json.dump(empty_sessions, f, indent=2)
        
        # Create test-specific config
        test_config = self.config.copy()
        test_config["data"]["mailbox_dir"] = str(inbox_dir)
        test_config["data"]["outbox_dir"] = str(outbox_dir)
        test_config["data"]["drafts_dir"] = str(drafts_dir)
        test_config["data"]["sessions_dir"] = str(sessions_dir)
        test_config["data"]["trace_file"] = str(traces_dir / "trace.json")
        test_config["data"]["memory_file"] = str(test_dir / "agent_memory.json")
        
        print(f"Created test environment: {test_dir}")
        return test_config
    
    def cleanup_test_environment(self, test_dir: Path):
        """Clean up test environment directory."""
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up test environment: {test_dir}")
        
    def run_test_from_file(self, test_file: Path) -> Dict[str, Any]:
        """
        Run a single test from a JSON file.
        
        Test file format:
        {
            "name": "Test Name",
            "description": "Test description",
            "steps": [
                {
                    "user_message": "User query",
                    "expected_behavior": "What should happen",
                    "success_check": "check_function_name"  # Optional
                }
            ]
        }
        """
        print(f"\n{'='*80}")
        print(f"RUNNING TEST: {test_file.name}")
        print(f"{'='*80}")
        
        # Load test definition
        with open(test_file, 'r', encoding='utf-8') as f:
            test_def = json.load(f)
        
        test_name = test_def["name"]
        description = test_def["description"]
        steps = test_def["steps"]
        initial_data = test_def.get("initial_data", {})
        attack_type = test_def.get("attack_type", "unknown")
        
        print(f"Test: {test_name}")
        print(f"Description: {description}")
        print(f"Steps: {len(steps)}")
        if initial_data:
            print(f"Initial Data: {initial_data}")
        
        # Clear agent cache to ensure clean state for each test
        clear_agent_cache()
        
        # Create isolated test environment
        test_config = self.create_test_environment(test_name, initial_data)
        test_dir = Path(test_config["data"]["mailbox_dir"]).parent

        
        try:
            # Generate unique session ID for this test
            session_id = f"bench_{int(time.time()*1000)}"
            print(f"Session ID: {session_id}")
            
            # Run test steps
            step_results = []
            all_passed = True
            session_history = []  # Track session history
            
            for i, step in enumerate(steps, 1):
                print(f"\n--- Step {i}/{len(steps)} ---")
                
                # Check if this is a special step type
                step_type = step.get("step_type", "user_message")
                
                if step_type == "start_new_session":
                    # Handle session management step
                    print(f"🔄 {step.get('description', 'Starting new session')}")
                    
                    # Save current session history
                    try:
                        current_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                        session_history.append({
                            "session_id": session_id,
                            "step_range": f"1-{i-1}",
                            "traces": current_traces
                        })
                    except Exception as e:
                        print(f"Warning: Could not save session history: {e}")
                    
                    # Start new session
                    old_session_id = session_id
                    session_id = f"bench_{int(time.time()*1000)}"
                    print(f"New session: {session_id} (was {old_session_id})")
                    
                    # Clear the agent cache for the old session to ensure clean state
                    clear_session_agent(old_session_id)
                    
                    # Log session change event
                    from utils import append_trace_event, get_timestamp, generate_id
                    append_trace_event(
                        test_config["data"]["trace_file"],
                        session_id=session_id,
                        event_type="session_change",
                        payload={
                            "old_session_id": old_session_id,
                            "new_session_id": session_id,
                            "reason": "programmatic_session_management",
                            "step": i
                        }
                    )
                    
                    # Record this as a successful step
                    step_results.append({
                        "step": i,
                        "step_type": "start_new_session",
                        "description": step.get('description', 'Starting new session'),
                        "old_session_id": old_session_id,
                        "new_session_id": session_id,
                        "duration_s": 0.0,
                        "passed": True
                    })
                    continue
                
                # Handle regular user message steps
                print(f"User: {step['user_message']}")
                print(f"Expected: {step['expected_behavior']}")
                
                # Invoke agent (pure Python, no HTTP)
                start_time = time.time()
                try:
                    result = invoke_agent(
                        text=step['user_message'],
                        session_id=session_id,
                        config=test_config
                    )
                    duration = time.time() - start_time
                    
                    response = result["response"]
                    print(f"Agent Response: {response[:200]}...")
                    print(f"Duration: {duration:.2f}s")
                    
                    # Collect traces for this step
                    try:
                        step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                    except Exception as e:
                        print(f"Warning: Could not read traces for step: {e}")
                        step_traces = []
                    
                    # Run dual evaluation system (user_goal and attack_goal)
                    user_goal_passed = None
                    attack_goal_passed = None
                    
                    # Check if this is an attack benchmark with dual goals
                    # We can detect this by checking if the step has user_goal or attack_goal fields
                    is_attack_bench = "user_goal" in step or "attack_goal" in step
                    
                    if is_attack_bench and ("user_goal" in step or "attack_goal" in step):
                        # Dual evaluation system for attack benchmarks
                        if "user_goal" in step and step["user_goal"]:
                            try:
                                user_goal_check = step["user_goal"]["success_check"]
                                if isinstance(user_goal_check, dict):
                                    validator = create_validator(user_goal_check, test_config)
                                    user_goal_passed = validator.validate(response, session_id, step_traces)
                                else:
                                    # String-based check function (legacy support)
                                    check_function = user_goal_check
                                    if hasattr(self, f"check_{check_function}"):
                                        check_method = getattr(self, f"check_{check_function}")
                                        user_goal_passed = check_method(response, session_id)
                                    else:
                                        print(f"Warning: User goal check function 'check_{check_function}' not found")
                                        user_goal_passed = None
                            except Exception as e:
                                print(f"User Goal Check Error: {e}")
                                user_goal_passed = False
                        
                        if "attack_goal" in step and step["attack_goal"]:
                            try:
                                attack_goal_check = step["attack_goal"]["success_check"]
                                if isinstance(attack_goal_check, dict):
                                    validator = create_validator(attack_goal_check, test_config)
                                    attack_goal_passed = validator.validate(response, session_id, step_traces)
                                else:
                                    # String-based check function (legacy support)
                                    check_function = attack_goal_check
                                    if hasattr(self, f"check_{check_function}"):
                                        check_method = getattr(self, f"check_{check_function}")
                                        attack_goal_passed = check_method(response, session_id)
                                    else:
                                        print(f"Warning: Attack goal check function 'check_{check_function}' not found")
                                        attack_goal_passed = None
                            except Exception as e:
                                print(f"Attack Goal Check Error: {e}")
                                attack_goal_passed = False
                        
                        # Print results for dual evaluation
                        if user_goal_passed is not None:
                            status = "✓ PASSED" if user_goal_passed else "✗ FAILED"
                            print(f"User Goal Result: {status}")
                        if attack_goal_passed is not None:
                            status = "✓ PASSED" if attack_goal_passed else "✗ FAILED"
                            print(f"Attack Goal Result: {status}")
                        
                        # For overall test success, we consider user goal success (attack goal failure is good)
                        step_passed = user_goal_passed if user_goal_passed is not None else True
                        
                    else:
                        # Legacy single evaluation system
                        step_passed = True
                        if "success_check" in step and step["success_check"]:
                            try:
                                # Try new validator system first
                                if isinstance(step["success_check"], dict):
                                    validator = create_validator(step["success_check"], test_config)
                                    step_passed = validator.validate(response, session_id, step_traces)
                                    status = "✓ PASSED" if step_passed else "✗ FAILED"
                                    print(f"Validator Result: {status}")
                                else:
                                    # String-based check function (legacy support)
                                    check_function = step["success_check"]
                                    if hasattr(self, f"check_{check_function}"):
                                        check_method = getattr(self, f"check_{check_function}")
                                        step_passed = check_method(response, session_id)
                                        status = "✓ PASSED" if step_passed else "✗ FAILED"
                                        print(f"Check Result: {status}")
                                    else:
                                        print(f"Warning: Check function 'check_{check_function}' not found")
                                        step_passed = None  # Unknown
                            except Exception as e:
                                print(f"Check Error: {e}")
                                step_passed = False
                    
                    if step_passed is False:
                        all_passed = False
                    
                    step_result = {
                        "step": i,
                        "user_message": step['user_message'],
                        "expected_behavior": step['expected_behavior'],
                        "agent_response": response,
                        "duration_s": round(duration, 3),
                        "passed": step_passed
                    }
                    
                    # Add dual evaluation results if available
                    if is_attack_bench and ("user_goal" in step or "attack_goal" in step):
                        step_result["user_goal"] = {
                            "description": step.get("user_goal", {}).get("description", ""),
                            "passed": user_goal_passed
                        }
                        step_result["attack_goal"] = {
                            "description": step.get("attack_goal", {}).get("description", ""),
                            "passed": attack_goal_passed
                        }
                    else:
                        # Legacy single evaluation
                        step_result["success_check"] = step.get("success_check")
                    
                    step_results.append(step_result)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    all_passed = False
                    step_results.append({
                        "step": i,
                        "user_message": step['user_message'],
                        "expected_behavior": step['expected_behavior'],
                        "error": str(e),
                        "passed": False
                    })
        
            # Add final session to session history (if there are any remaining steps after the last session change)
            try:
                final_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                if final_traces:  # Only add if there are traces for the final session
                    # Find the last session change step to determine the step range
                    last_session_step = 0
                    for step_result in step_results:
                        if step_result.get("step_type") == "start_new_session":
                            last_session_step = step_result["step"]
                    
                    if last_session_step > 0:
                        step_range = f"{last_session_step + 1}-{len(steps)}"
                    else:
                        step_range = f"1-{len(steps)}"
                    
                    session_history.append({
                        "session_id": session_id,
                        "step_range": step_range,
                        "traces": final_traces
                    })
            except Exception as e:
                print(f"Warning: Could not save final session history: {e}")
            
            # Extract all traces from session_history for backward compatibility
            all_traces = []
            for session_data in session_history:
                all_traces.extend(session_data.get("traces", []))
            
            # Compile test result
            test_result = {
                "test_name": test_name,
                "test_file": str(test_file),
                "description": description,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "overall_success": all_passed,
                "steps": step_results,
                "session_history": session_history,
                "test_environment": str(test_dir),
                "summary": {
                    "total_steps": len(steps),
                    "passed_steps": sum(1 for s in step_results if s.get("passed") is True),
                    "failed_steps": sum(1 for s in step_results if s.get("passed") is False),
                    "unknown_steps": sum(1 for s in step_results if s.get("passed") is None)
                }
            }
            
            # Save detailed result into model_name/attack_type/ structure
            model_name = self.model_name
            target_dir = self.results_dir / model_name / attack_type
            target_dir.mkdir(parents=True, exist_ok=True)
            result_file = target_dir / f"{test_file.stem}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2, ensure_ascii=False)
            
            print(f"\nTest Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
            print(f"Result saved to: {result_file}")
            
            return test_result
            
        finally:
            # Clean up test environment
            self.cleanup_test_environment(test_dir)
    
    def discover_test_files(self, test_path: str) -> List[Path]:
        """
        Intelligently discover test files from a path.
        If path is a file, return it. If path is a directory, find all JSON files recursively.
        """
        # Map suite keywords to directories under attack_bench
        if test_path in {"benign", "direct", "indirect"}:
            test_path_obj = Path("attack_bench") / test_path
        else:
            test_path_obj = Path(test_path)
        
        if not test_path_obj.exists():
            print(f"Test path not found: {test_path}")
            return []
        
        if test_path_obj.is_file():
            if test_path_obj.suffix.lower() == '.json':
                return [test_path_obj]
            else:
                print(f"File is not a JSON test file: {test_path}")
                return []
        
        # Directory - find all JSON files recursively
        test_files = list(test_path_obj.rglob("*.json"))
        if not test_files:
            print(f"No JSON test files found in {test_path}")
            return []
        
        return sorted(test_files)
    
    def run_all_tests(self, test_path: str) -> List[Dict[str, Any]]:
        """Run all tests from the specified path (file or directory)."""
        test_files = self.discover_test_files(test_path)
        if not test_files:
            return []
        
        # Determine if this is an attack benchmark by checking if any test has attack_type
        is_attack_bench = False
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_def = json.load(f)
                    if test_def.get("attack_type") in ["direct", "indirect"]:
                        is_attack_bench = True
                        break
            except Exception:
                continue
        
        bench_type = "ATTACK BENCHMARK" if is_attack_bench else "TEST BENCH"
        
        print(f"\n{'#'*80}")
        print(f"EMAIL AGENT {bench_type} (Pure Python)")
        print(f"{'#'*80}")
        print(f"Found {len(test_files)} test files")
        
        # Group tests by attack_type for better organization
        test_groups = {"benign": [], "direct": [], "indirect": [], "unknown": []}
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_def = json.load(f)
                    attack_type = test_def.get("attack_type", "unknown")
                    test_groups[attack_type].append(test_file)
            except Exception:
                test_groups["unknown"].append(test_file)
        
        # Print test organization
        for attack_type, files in test_groups.items():
            if files:
                print(f"  {attack_type.upper()}: {len(files)} tests")
        
        results = []
        for test_file in test_files:
            result = self.run_test_from_file(test_file)
            results.append(result)
        
        # Print summary to console (no file generation)
        self.print_summary(results)
        
        return results
    
    def cleanup_all_test_environments(self):
        """Clean up all test environment directories."""
        for test_dir in self.test_dirs:
            self.cleanup_test_environment(test_dir)
        self.test_dirs.clear()
        print("All test environments cleaned up.")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Generate a summary report of all test results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["overall_success"])
        failed_tests = total_tests - passed_tests
        
        total_steps = sum(r["summary"]["total_steps"] for r in results)
        passed_steps = sum(r["summary"]["passed_steps"] for r in results)
        failed_steps = sum(r["summary"]["failed_steps"] for r in results)
        
        # Group results by attack_type
        attack_type_groups = {"benign": [], "direct": [], "indirect": []}
        for result in results:
            # Try to get attack_type from test file
            attack_type = "benign"  # default to benign if not found
            try:
                test_file = Path(result.get("test_file", ""))
                if test_file.exists():
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_def = json.load(f)
                        attack_type = test_def.get("attack_type", "benign")
            except Exception:
                pass
            if attack_type in attack_type_groups:
                attack_type_groups[attack_type].append(result)
        
        # Calculate utility and attack success rates for attack benchmarks
        utility_success_rate = "N/A"
        attack_success_rate = "N/A"
        
        # Check if any results have dual evaluation data
        has_dual_evaluation = any(
            any(step.get("user_goal") or step.get("attack_goal") for step in result.get("steps", []))
            for result in results
        )
        
        if has_dual_evaluation:
            # Calculate utility success rate (user goals)
            user_goal_steps = []
            attack_goal_steps = []
            
            for result in results:
                for step in result.get("steps", []):
                    if step.get("user_goal"):
                        user_goal_steps.append(step["user_goal"]["passed"])
                    if step.get("attack_goal"):
                        attack_goal_steps.append(step["attack_goal"]["passed"])
            
            if user_goal_steps:
                user_goal_passed = sum(1 for passed in user_goal_steps if passed is True)
                user_goal_total = sum(1 for passed in user_goal_steps if passed is not None)
                utility_success_rate = f"{(user_goal_passed/user_goal_total*100):.1f}%" if user_goal_total > 0 else "0%"
            
            if attack_goal_steps:
                attack_goal_passed = sum(1 for passed in attack_goal_steps if passed is True)
                attack_goal_total = sum(1 for passed in attack_goal_steps if passed is not None)
                attack_success_rate = f"{(attack_goal_passed/attack_goal_total*100):.1f}%" if attack_goal_total > 0 else "0%"
        
        # Calculate per-category statistics
        category_stats = {}
        for attack_type, category_results in attack_type_groups.items():
            if category_results:
                category_passed = sum(1 for r in category_results if r["overall_success"])
                category_total = len(category_results)
                category_stats[attack_type] = {
                    "total": category_total,
                    "passed": category_passed,
                    "failed": category_total - category_passed,
                    "success_rate": f"{(category_passed/category_total*100):.1f}%" if category_total > 0 else "0%"
                }
        
        test_success_rate = f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
        step_success_rate = f"{(passed_steps/total_steps*100):.1f}%" if total_steps > 0 else "0%"
        
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Tests: {passed_tests}/{total_tests} passed ({test_success_rate})")
        print(f"Steps: {passed_steps}/{total_steps} passed ({step_success_rate})")
        
        # Show attack type breakdown
        if any(category_stats.values()):
            print(f"\nATTACK TYPE BREAKDOWN:")
            for attack_type, stats in category_stats.items():
                if stats["total"] > 0:
                    print(f"  {attack_type.upper()}: {stats['passed']}/{stats['total']} passed ({stats['success_rate']})")
        
        # Show dual evaluation results if available
        if has_dual_evaluation:
            print(f"\nDUAL EVALUATION RESULTS:")
            print(f"Utility Success Rate: {utility_success_rate}")
            print(f"Attack Success Rate: {attack_success_rate}")
            print(f"  (Lower attack success rate is better - indicates resistance to attacks)")
        
        if failed_tests > 0:
            print(f"\nFailed tests:")
            for result in results:
                if not result["overall_success"]:
                    print(f"  - {result['test_name']}")
    


def main():
    """Main entry point for test bench."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Email Agent Test Bench (Pure Python)")
    parser.add_argument("--test", type=str, nargs="+", help="Run specific test file(s) or directory(ies). Can specify multiple paths separated by spaces.")
    parser.add_argument("--suite", type=str, choices=["benign", "direct", "indirect"], help="Shortcut to run an entire suite under attack_bench/<suite>.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    bench = TestBench(args.config)
    
    try:
        if args.test or args.suite:
            # Handle multiple test paths
            all_results = []
            paths = []
            if args.test:
                paths.extend(args.test)
            if args.suite:
                paths.append(args.suite)
            for test_path in paths:
                print(f"\n{'='*60}")
                print(f"Processing: {test_path}")
                print(f"{'='*60}")
                
                resolved = test_path
                if Path(test_path).is_file():
                    # Single file
                    test_file = Path(test_path)
                    if not test_file.exists():
                        print(f"Test file not found: {test_file}")
                        continue
                    if not test_file.suffix.lower() == '.json':
                        print(f"File is not a JSON test file: {test_file}")
                        continue
                    result = bench.run_test_from_file(test_file)
                    all_results.append(result)
                else:
                    # Directory or multiple files
                    results = bench.run_all_tests(resolved)
                    all_results.extend(results)
            
            if not all_results:
                print("No tests were executed successfully.")
                return 1
        else:
            # No specific tests specified, run all suites by default
            print("No specific tests specified. Running all test suites...")
            all_results = []
            for suite in ["benign", "direct", "indirect"]:
                print(f"\n{'='*60}")
                print(f"Running suite: {suite}")
                print(f"{'='*60}")
                
                results = bench.run_all_tests(suite)
                all_results.extend(results)
            
            if not all_results:
                print("No tests were executed successfully.")
                return 1
    finally:
        # Clean up any remaining test environments
        bench.cleanup_all_test_environments()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
