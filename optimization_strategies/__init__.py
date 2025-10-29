"""
Optimization strategies for adaptive security benchmarking.

This module contains various optimization strategies that can be used to evolve
attacks when static ones fail. Each strategy implements a common interface
for attack optimization.
"""

from optimization_strategies.base_optimizer import BaseOptimizer, OptimizationResult
from optimization_strategies.dspy_optimizer import DSPyOptimizer
from optimization_strategies.openevolve_optimizer import OpenEvolveOptimizer

__all__ = [
    "BaseOptimizer",
    "OptimizationResult", 
    "DSPyOptimizer",
    "OpenEvolveOptimizer"
]
