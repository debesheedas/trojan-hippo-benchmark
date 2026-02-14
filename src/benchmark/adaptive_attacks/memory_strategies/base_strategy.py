"""
Base abstract strategy class for memory backend-specific optimizations.

This defines the interface that all memory backend strategies must implement.
Each strategy can customize:
1. Mutator prompts (how to generate attack variants)
2. Scorer/Critic prompts (how to evaluate attack effectiveness)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class MemoryBackendStrategy(ABC):
    """
    Abstract base class for memory backend-specific optimization strategies.
    
    Each memory backend (explicit, rag, mem0, context) can have its own strategy
    that customizes how the optimizer generates and evaluates attack candidates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.openevolve_config = config.get("benchmark", {}).get("openevolve", {})
    
    @abstractmethod
    def get_mutator_prompt_extension(self,
                                     attack_goal: Dict[str, Any],
                                     user_message: str,
                                     test_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get backend-specific extension to add to the mutator system prompt.
        
        This method should return additional guidance that is specific to how
        this memory backend works, which will help the mutator generate better
        attack variants.
        
        Args:
            attack_goal: The attack goal definition
            user_message: The user message from the test case
            test_config: Optional test configuration dict (contains defense info)
            
        Returns:
            String containing prompt extension (can be empty for backends that don't need it)
        """
        pass
    
    @abstractmethod
    def get_scorer_prompt_extension(self,
                                    attack_goal: Dict[str, Any],
                                    user_message: str) -> str:
        """
        Get backend-specific extension to add to the scorer/critic system prompt.
        
        This method should return additional evaluation criteria that are specific
        to how this memory backend works, which will help the scorer provide better
        feedback on attack effectiveness.
        
        Args:
            attack_goal: The attack goal definition
            user_message: The user message from the test case
            
        Returns:
            String containing prompt extension (can be empty for backends that don't need it)
        """
        pass
    
    def get_config_options(self) -> Dict[str, Any]:
        """
        Get backend-specific configuration options.
        
        Returns:
            Dictionary of configuration options specific to this backend
        """
        return {}
