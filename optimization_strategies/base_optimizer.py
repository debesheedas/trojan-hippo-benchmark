"""
Base optimizer class for attack optimization strategies.

This module defines the common interface that all optimization strategies
must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    success: bool
    optimized_attack_email: Optional[Dict[str, Any]]
    optimization_strategy: str
    iterations: int
    feedback: List[str]
    final_evaluation: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class BaseOptimizer(ABC):
    """Base class for all optimization strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None  # Will be set by the adaptive benchmark
    
    def set_logger(self, logger):
        """Set the logger for this optimizer."""
        self.logger = logger
    
    @abstractmethod
    def optimize_attack(self, 
                       original_attack_email: Dict[str, Any],
                       failed_step: Dict[str, Any],
                       step_num: int,
                       session_id: str,
                       test_config: Dict[str, Any],
                       environment_state: 'EnvironmentState',
                       max_iterations: int = 10) -> OptimizationResult:
        """
        Optimize an attack email to make it more effective.
        
        Args:
            original_attack_email: The original attack email that failed
            failed_step: The test step where the attack failed
            step_num: The step number where the attack failed
            session_id: The session ID for the test
            test_config: The test configuration
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            OptimizationResult with the optimized attack or failure info
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this optimization strategy."""
        pass
    
    def _log_info(self, message: str):
        """Log an info message."""
        if self.logger:
            self.logger.info(f"[{self.get_strategy_name()}] {message}")
    
    def _log_warning(self, message: str):
        """Log a warning message."""
        if self.logger:
            self.logger.warning(f"[{self.get_strategy_name()}] {message}")
    
    def _log_error(self, message: str):
        """Log an error message."""
        if self.logger:
            self.logger.error(f"[{self.get_strategy_name()}] {message}")
    
    def _log_debug(self, message: str):
        """Log a debug message."""
        if self.logger:
            self.logger.debug(f"[{self.get_strategy_name()}] {message}")
