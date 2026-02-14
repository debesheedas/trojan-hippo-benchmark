"""
Explicit Memory Backend Strategy for OpenEvolve Optimizer.

This strategy is the default/base implementation for memory backends that don't
require special handling. It provides minimal/no extensions to the standard prompts.
"""

from typing import Dict, Any, Optional
from .base_strategy import MemoryBackendStrategy


class ExplicitMemoryStrategy(MemoryBackendStrategy):
    """
    Strategy for explicit memory backend (and other backends without special requirements).
    
    This is the default strategy that doesn't add any backend-specific guidance.
    The mutator and scorer use standard prompts without memory-backend-specific considerations.
    """
    
    def get_mutator_prompt_extension(self,
                                     attack_goal: Dict[str, Any],
                                     user_message: str,
                                     test_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get mutator prompt extension for explicit memory backend.
        
        For explicit memory, we don't need any special guidance - the standard
        prompt is sufficient.
        
        Returns:
            Empty string (no extension needed)
        """
        return ""
    
    def get_scorer_prompt_extension(self,
                                    attack_goal: Dict[str, Any],
                                    user_message: str) -> str:
        """
        Get scorer prompt extension for explicit memory backend.
        
        For explicit memory, we don't need any special evaluation criteria -
        the standard evaluation is sufficient.
        
        Returns:
            Empty string (no extension needed)
        """
        return ""
