#!/usr/bin/env python3
"""
Adaptive Security Benchmark for Email Agent

This module implements an adaptive benchmark that can evolve attacks when static ones fail.
It uses optimization strategies like DSPy and OpenEvolve to generate more effective attacks.

Key Features:
- Try static attack first, if it fails → trigger optimization
- Use multiple optimization strategies (DSPy, OpenEvolve)
- Only modify attack emails, preserve all other test case data
- Resume from failed step after successful optimization
- Cache successful attacks for future runs
- Detailed logging for optimization process
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from test_bench import TestBench
from agent_core import invoke_agent, clear_agent_cache
from utils import read_trace_events, load_config
from test_validators import create_validator
from optimization_strategies import DSPyOptimizer, OpenEvolveOptimizer


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    success: bool
    optimized_attack_email: Optional[Dict[str, Any]]
    optimization_strategy: str
    iterations: int
    feedback: List[str]
    final_evaluation: Optional[Dict[str, Any]]


class AdaptiveBenchmark:
    """Adaptive benchmark that evolves attacks when static ones fail."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.model_name = (self.config.get("model", {}) or {}).get("model_name")
        if not self.model_name:
            raise ValueError("Config missing model.model_name. Please set it in config.yaml.")
        
        # Set up logging
        self.setup_logging()
        
        # Initialize base test bench
        self.test_bench = TestBench(config_path)
        
        # Cache directory for successful attacks
        self.cache_dir = Path("attack_bench_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize optimization strategies
        self.optimizers = {
            "dspy": DSPyOptimizer(self.config),
            "openevolve": OpenEvolveOptimizer(self.config)
        }
        
        # Set loggers for optimizers
        for optimizer in self.optimizers.values():
            optimizer.set_logger(self.logger)
        
        self.logger.info("Adaptive benchmark initialized with model: %s", self.model_name)
    
    def setup_logging(self):
        """Set up detailed logging for optimization process."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("adaptive_benchmark")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for detailed logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"adaptive_benchmark_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging initialized. Detailed logs: %s", log_file)
    
    def run_adaptive_test(self, test_file: Path) -> Dict[str, Any]:
        """
        Run a test with adaptive attack optimization.
        
        Process:
        1. Try static attack first
        2. If attack fails, try optimization strategies
        3. Resume from failed step with optimized attack
        4. Cache successful attacks
        """
        self.logger.info(f"Starting adaptive test: {test_file.name}")
        
        # Load test definition
        with open(test_file, 'r', encoding='utf-8') as f:
            test_def = json.load(f)
        
        test_name = test_def["name"]
        attack_type = test_def.get("attack_type", "unknown")
        
        # Check if this is an indirect PI attack
        if attack_type != "indirect":
            self.logger.info(f"Test {test_name} is not an indirect attack, running normally")
            return self.test_bench.run_test_from_file(test_file)
        
        # Check for cached version first
        cached_test = self.get_cached_test(test_file)
        if cached_test:
            self.logger.info(f"Using cached test with optimized attack: {test_name}")
            return self.test_bench.run_test_from_file(cached_test)
        
        self.logger.info(f"Running adaptive test for indirect attack: {test_name}")
        
        # Run test with adaptive optimization
        result = self._run_test_with_optimization(test_def, test_file)
        
        # Cache successful attacks
        if result.get("optimization_used"):
            self._cache_successful_attack(test_file, result)
        
        return result
    
    def _run_test_with_optimization(self, test_def: Dict[str, Any], test_file: Path) -> Dict[str, Any]:
        """Run test with optimization when attacks fail."""
        test_name = test_def["name"]
        steps = test_def["steps"]
        initial_data = test_def.get("initial_data", {})
        
        self.logger.info(f"Running test with optimization: {test_name}")
        
        # Clear agent cache for clean state
        clear_agent_cache()
        
        # Create test environment
        test_config = self.test_bench.create_test_environment(test_name, initial_data)
        test_dir = Path(test_config["data"]["mailbox_dir"]).parent
        
        try:
            session_id = f"adaptive_{int(time.time()*1000)}"
            self.logger.info(f"Session ID: {session_id}")
            
            step_results = []
            all_passed = True
            optimization_used = False
            
            for i, step in enumerate(steps, 1):
                self.logger.info(f"Processing step {i}/{len(steps)}")
                
                # Check if this step has an attack goal
                if "attack_goal" in step and step["attack_goal"]:
                    self.logger.info(f"Step {i} has attack goal, testing attack effectiveness")
                    
                    # Run the step and check if attack succeeds
                    step_result = self._run_step_with_attack_check(
                        step, i, session_id, test_config
                    )
                    
                    # Check if attack failed
                    attack_goal_passed = step_result.get("attack_goal", {}).get("passed")
                    if attack_goal_passed is False:
                        self.logger.warning(f"Attack failed at step {i}, attempting optimization")
                        
                        # Try optimization strategies
                        optimization_result = self._optimize_attack(
                            test_def, step, i, session_id, test_config
                        )
                        
                        if optimization_result.success:
                            self.logger.info(f"Optimization successful with {optimization_result.optimization_strategy}")
                            
                            # Update the test with optimized attack
                            test_def = self._update_test_with_optimized_attack(
                                test_def, optimization_result.optimized_attack_email
                            )
                            
                            # Re-run the step with optimized attack
                            step_result = self._run_step_with_attack_check(
                                step, i, session_id, test_config, 
                                optimized_attack_email=optimization_result.optimized_attack_email
                            )
                            
                            optimization_used = True
                            step_result["optimization"] = {
                                "strategy": optimization_result.optimization_strategy,
                                "iterations": optimization_result.iterations,
                                "feedback": optimization_result.feedback
                            }
                        else:
                            self.logger.error(f"All optimization strategies failed for step {i}")
                            step_result["optimization"] = {
                                "strategy": "all_failed",
                                "iterations": 0,
                                "feedback": optimization_result.feedback
                            }
                    
                    step_results.append(step_result)
                    
                    # Check if step passed
                    if step_result.get("passed") is False:
                        all_passed = False
                else:
                    # Regular step without attack goal
                    step_result = self._run_regular_step(step, i, session_id, test_config)
                    step_results.append(step_result)
                    
                    if step_result.get("passed") is False:
                        all_passed = False
            
            # Compile result
            result = {
                "test_name": test_name,
                "test_file": str(test_file),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "overall_success": all_passed,
                "steps": step_results,
                "optimization_used": optimization_used,
                "adaptive_benchmark": True
            }
            
            return result
            
        finally:
            # Clean up test environment
            self.test_bench.cleanup_test_environment(test_dir)
    
    def _run_step_with_attack_check(self, step: Dict[str, Any], step_num: int, 
                                  session_id: str, test_config: Dict[str, Any],
                                  optimized_attack_email: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a step and check if attack succeeds."""
        self.logger.info(f"Running step {step_num} with attack check")
        
        # If we have an optimized attack email, we need to update the test environment
        if optimized_attack_email:
            self._inject_optimized_attack_email(test_config, optimized_attack_email)
        
        # Run the step
        start_time = time.time()
        try:
            result = invoke_agent(
                text=step['user_message'],
                session_id=session_id,
                config=test_config
            )
            duration = time.time() - start_time
            
            response = result["response"]
            self.logger.info(f"Agent response: {response[:200]}...")
            
            # Get traces for evaluation
            try:
                step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
            except Exception as e:
                self.logger.warning(f"Could not read traces: {e}")
                step_traces = []
            
            # Evaluate both user and attack goals
            user_goal_passed = None
            attack_goal_passed = None
            partial_attack_score = None
            
            if "user_goal" in step and step["user_goal"]:
                try:
                    user_goal_check = step["user_goal"]["success_check"]
                    if isinstance(user_goal_check, dict):
                        validator = create_validator(user_goal_check, test_config)
                        user_goal_passed = validator.validate(response, session_id, step_traces)
                    self.logger.info(f"User goal result: {user_goal_passed}")
                except Exception as e:
                    self.logger.error(f"User goal check error: {e}")
                    user_goal_passed = False
            
            if "attack_goal" in step and step["attack_goal"]:
                try:
                    attack_goal_check = step["attack_goal"]["success_check"]
                    if isinstance(attack_goal_check, dict):
                        validator = create_validator(attack_goal_check, test_config)
                        attack_goal_passed = validator.validate(response, session_id, step_traces)
                        try:
                            partial_attack_score = float(validator.score(response, session_id, step_traces))
                        except Exception:
                            partial_attack_score = 1.0 if attack_goal_passed else 0.0
                    self.logger.info(f"Attack goal result: {attack_goal_passed} (partial={partial_attack_score if partial_attack_score is not None else 'n/a'})")
                except Exception as e:
                    self.logger.error(f"Attack goal check error: {e}")
                    attack_goal_passed = False
                    partial_attack_score = 0.0
            
            # Overall step success (user goal success, attack goal failure is good)
            step_passed = user_goal_passed if user_goal_passed is not None else True
            
            return {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "agent_response": response,
                "duration_s": round(duration, 3),
                "passed": step_passed,
                "user_goal": {
                    "description": step.get("user_goal", {}).get("description", ""),
                    "passed": user_goal_passed
                },
                "attack_goal": {
                    "description": step.get("attack_goal", {}).get("description", ""),
                    "passed": attack_goal_passed,
                    "partial_attack_score": partial_attack_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in step {step_num}: {e}")
            return {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "error": str(e),
                "passed": False
            }
    
    def _run_regular_step(self, step: Dict[str, Any], step_num: int, 
                         session_id: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a regular step without attack goals."""
        self.logger.info(f"Running regular step {step_num}")
        
        start_time = time.time()
        try:
            result = invoke_agent(
                text=step['user_message'],
                session_id=session_id,
                config=test_config
            )
            duration = time.time() - start_time
            
            response = result["response"]
            
            return {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "agent_response": response,
                "duration_s": round(duration, 3),
                "passed": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in regular step {step_num}: {e}")
            return {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "error": str(e),
                "passed": False
            }
    
    def _optimize_attack(self, test_def: Dict[str, Any], failed_step: Dict[str, Any], 
                        step_num: int, session_id: str, test_config: Dict[str, Any]) -> OptimizationResult:
        """Try different optimization strategies to find a successful attack."""
        self.logger.info(f"Starting optimization for step {step_num}")
        
        # Get the original attack email
        original_attack_email = None
        if "attack_emails" in test_def.get("initial_data", {}):
            attack_emails = test_def["initial_data"]["attack_emails"]
            if attack_emails:
                original_attack_email = attack_emails[0]
        
        if not original_attack_email:
            self.logger.error("No attack email found in test definition")
            return OptimizationResult(
                success=False,
                optimized_attack_email=None,
                optimization_strategy="none",
                iterations=0,
                feedback=["No attack email found"],
                final_evaluation=None
            )
        
        # Try each optimization strategy
        for strategy_name, optimizer in self.optimizers.items():
            self.logger.info(f"Trying optimization strategy: {strategy_name}")
            
            result = optimizer.optimize_attack(
                original_attack_email, failed_step, step_num, session_id, test_config
            )
            
            if result.success:
                self.logger.info(f"Optimization successful with {strategy_name}")
                return result
            else:
                self.logger.warning(f"Optimization failed with {strategy_name}")
        
        # All strategies failed
        return OptimizationResult(
            success=False,
            optimized_attack_email=None,
            optimization_strategy="all_failed",
            iterations=0,
            feedback=["All optimization strategies failed"],
            final_evaluation=None
        )
    
    
    def _update_test_with_optimized_attack(self, test_def: Dict[str, Any], 
                                         optimized_attack_email: Dict[str, Any]) -> Dict[str, Any]:
        """Update test definition with optimized attack email."""
        if "initial_data" in test_def and "attack_emails" in test_def["initial_data"]:
            test_def["initial_data"]["attack_emails"][0] = optimized_attack_email
        return test_def
    
    def _inject_optimized_attack_email(self, test_config: Dict[str, Any], 
                                     optimized_attack_email: Dict[str, Any]):
        """Inject optimized attack email into the test environment."""
        # This would update the inbox with the new attack email
        # For now, this is a placeholder
        self.logger.info("Injecting optimized attack email into test environment")
        # TODO: Implement actual email injection
        _ = test_config  # Suppress unused argument warning
        _ = optimized_attack_email  # Suppress unused argument warning
    
    def get_cached_test(self, test_file: Path) -> Optional[Path]:
        """Check if there's a cached version of the test with successful attacks."""
        cache_file = self.cache_dir / test_file.relative_to(Path("attack_bench"))
        if cache_file.exists():
            return cache_file
        return None
    
    def _cache_successful_attack(self, test_file: Path, result: Dict[str, Any]):
        """Cache a test with successful attacks for future runs."""
        self.logger.info("Caching successful attack for %s", test_file.name)
        
        # Create cache directory structure
        cache_file = self.cache_dir / test_file.relative_to(Path("attack_bench"))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original test and update with optimized attacks
        with open(test_file, 'r', encoding='utf-8') as f:
            cached_test = json.load(f)
        
        # Add optimization metadata
        cached_test["optimization_metadata"] = {
            "optimized": True,
            "optimization_timestamp": datetime.now().isoformat(),
            "model_name": self.model_name
        }
        
        # Save cached version
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cached_test, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Cached test saved to: %s", cache_file)
    
    def run_adaptive_tests(self, test_path: str) -> List[Dict[str, Any]]:
        """Run adaptive tests on the specified path."""
        test_files = self.test_bench.discover_test_files(test_path)
        if not test_files:
            return []
        
        self.logger.info("Running adaptive tests on %d files", len(test_files))
        
        results = []
        for test_file in test_files:
            result = self.run_adaptive_test(test_file)
            results.append(result)
        
        return results


def main():
    """Main entry point for adaptive benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Security Benchmark")
    parser.add_argument("--test", type=str, nargs="+", help="Run specific test file(s) or directory(ies)")
    parser.add_argument("--suite", type=str, choices=["benign", "direct", "indirect"], help="Run an entire suite")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    benchmark = AdaptiveBenchmark(args.config)
    
    try:
        if args.test or args.suite:
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
                
                results = benchmark.run_adaptive_tests(test_path)
                all_results.extend(results)
            
            if not all_results:
                print("No tests were executed successfully.")
                return 1
        else:
            # Run indirect attacks by default
            print("No specific tests specified. Running indirect attack suite...")
            results = benchmark.run_adaptive_tests("indirect")
            
            if not results:
                print("No tests were executed successfully.")
                return 1
    
    finally:
        # Clean up any remaining test environments
        benchmark.test_bench.cleanup_all_test_environments()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
