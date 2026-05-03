"""
Memory Backend Strategies for OpenEvolve Optimizer.

This module provides memory-backend-specific implementations for the OpenEvolve optimizer.
Each memory backend (explicit, rag, mem0, context) can have its own strategy that
customizes the mutator and scorer prompts to account for backend-specific behavior.
"""

from .base_strategy import MemoryBackendStrategy
from .explicit_strategy import ExplicitMemoryStrategy
from .rag_strategy import RAGMemoryStrategy
from .mem0_strategy import Mem0MemoryStrategy
from .context_strategy import ContextMemoryStrategy

__all__ = [
    "MemoryBackendStrategy",
    "ExplicitMemoryStrategy",
    "RAGMemoryStrategy",
    "Mem0MemoryStrategy",
    "ContextMemoryStrategy",
]

# Strategy registry - maps memory backend names to strategy classes
STRATEGY_REGISTRY = {
    "explicit": ExplicitMemoryStrategy,
    "rag": RAGMemoryStrategy,
    "mem0": Mem0MemoryStrategy,
    "context": ContextMemoryStrategy,
    "none": ExplicitMemoryStrategy,  # Use explicit strategy as default for now
}


def get_strategy(memory_backend: str, config: dict) -> MemoryBackendStrategy:
    """
    Get the appropriate strategy for a memory backend.
    
    Args:
        memory_backend: Memory backend name ("explicit", "rag", "mem0", "context", "none")
        config: Full configuration dictionary
        
    Returns:
        MemoryBackendStrategy instance
    """
    strategy_class = STRATEGY_REGISTRY.get(memory_backend, ExplicitMemoryStrategy)
    return strategy_class(config)
