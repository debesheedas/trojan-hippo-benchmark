"""
Optimization strategies for adaptive security benchmarking.

This module contains various optimization strategies that can be used to evolve
attacks when static ones fail. Each strategy implements a common interface
for attack optimization.
"""

from .base_optimizer import BaseOptimizer, OptimizationResult
from .openevolve_optimizer import OpenEvolveOptimizer

__all__ = [
    "BaseOptimizer",
    "OptimizationResult",
    "OpenEvolveOptimizer",
]
