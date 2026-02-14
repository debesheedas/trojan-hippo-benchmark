"""
Mem0 Memory Backend Strategy for OpenEvolve Optimizer.

This strategy is a placeholder for mem0 memory backend-specific optimizations.
Currently uses the default/base implementation. Can be customized later for
mem0-specific behavior.
"""

from typing import Dict, Any, Optional
from .base_strategy import MemoryBackendStrategy


class Mem0MemoryStrategy(MemoryBackendStrategy):
    """
    Strategy for mem0 memory backend.
    
    This is currently a placeholder that uses the default strategy.
    Can be customized later to add mem0-specific guidance for mutator and scorer prompts.
    """
    
    def get_mutator_prompt_extension(self,
                                     attack_goal: Dict[str, Any],
                                     user_message: str,
                                     test_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get mutator prompt extension for mem0 memory backend.
        
        Currently returns empty string (no extension). Can be customized later
        to add mem0-specific guidance.
        
        Returns:
            Empty string (no extension needed for now)
        """
        return ""
    
    def get_scorer_prompt_extension(self,
                                    attack_goal: Dict[str, Any],
                                    user_message: str) -> str:
        """
        Get scorer prompt extension for mem0 memory backend.
        
        Currently returns empty string (no extension). Can be customized later
        to add mem0-specific evaluation criteria.
        
        Returns:
            Empty string (no extension needed for now)
        """
        return ""
