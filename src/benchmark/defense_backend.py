"""
Defense Backend Abstraction

Provides a unified interface for defense mechanisms across all memory systems.
Maps unified defense names to backend-specific implementations.
"""

from typing import Protocol, Dict, Any, Optional, List
import sys
from pathlib import Path

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


class DefenseBackend(Protocol):
    """Protocol defining the interface for defense backends."""
    
    @property
    def name(self) -> str:
        """Backend name: 'explicit', 'mem0', or 'rag'."""
        ...
    
    def get_defense_type(self, unified_defense: str) -> str:
        """
        Map unified defense name to backend-specific defense type.
        
        Args:
            unified_defense: Unified defense name (e.g., "disable_memory", "user_only")
            
        Returns:
            Backend-specific defense type string
        """
        ...
    
    def get_all_defense_types(self) -> List[str]:
        """
        Get all defense types supported by this backend.
        
        Returns:
            List of defense type strings
        """
        ...


class ExplicitDefenseBackend:
    """Defense backend for explicit memory."""
    
    # Defense mappings: unified_name -> explicit_name
    DEFENSE_MAP = {
        "disable_memory": "disable_memory",
        "none": "none",
        "user_only": "user_prompt_only",  # Explicit uses "user_prompt_only"
        "no_untrusted_tools": "no_untrusted_tools",
        "limit_memory_length": "limit_memory_length",
    }
    
    @property
    def name(self) -> str:
        return "explicit"
    
    def get_defense_type(self, unified_defense: str) -> str:
        """Map unified defense to explicit defense type."""
        return self.DEFENSE_MAP.get(unified_defense, unified_defense)
    
    def get_all_defense_types(self) -> List[str]:
        """Get all explicit defense types."""
        return list(self.DEFENSE_MAP.values())


class Mem0DefenseBackend:
    """Defense backend for mem0 memory."""
    
    # Defense mappings: unified_name -> mem0_name
    DEFENSE_MAP = {
        "disable_memory": "disable_memory",
        "none": "no_defense",  # Mem0 uses "no_defense" instead of "none"
        "user_only": "user_only",
        "no_untrusted_tools": "no_untrusted_tools",
        "limit_memory_length": "limit_memory_length",
    }
    
    @property
    def name(self) -> str:
        return "mem0"
    
    def get_defense_type(self, unified_defense: str) -> str:
        """Map unified defense to mem0 defense type."""
        return self.DEFENSE_MAP.get(unified_defense, unified_defense)
    
    def get_all_defense_types(self) -> List[str]:
        """Get all mem0 defense types."""
        return list(self.DEFENSE_MAP.values())


class RAGDefenseBackend:
    """Defense backend for RAG memory."""
    
    # Defense mappings: unified_name -> rag_name
    DEFENSE_MAP = {
        "disable_memory": "disable_memory",
        "none": "none",
        "user_only": "user_only",
        "no_untrusted_tools": "no_untrusted_tools",
        "limit_memory_length": "limit_chunk_size",  # RAG uses "limit_chunk_size"
    }
    
    @property
    def name(self) -> str:
        return "rag"
    
    def get_defense_type(self, unified_defense: str) -> str:
        """Map unified defense to RAG defense type."""
        return self.DEFENSE_MAP.get(unified_defense, unified_defense)
    
    def get_all_defense_types(self) -> List[str]:
        """Get all RAG defense types."""
        return list(self.DEFENSE_MAP.values())


class DefenseBackendRegistry:
    """Registry for defense backends."""
    
    def __init__(self):
        self._registry: Dict[str, DefenseBackend] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default defense backends."""
        self.register("explicit", ExplicitDefenseBackend())
        self.register("mem0", Mem0DefenseBackend())
        self.register("rag", RAGDefenseBackend())
    
    def register(self, name: str, backend: DefenseBackend) -> None:
        """
        Register a defense backend.
        
        Args:
            name: Backend name (e.g., "explicit", "mem0", "rag")
            backend: DefenseBackend instance
        """
        if name in self._registry:
            raise ValueError(f"Defense backend '{name}' is already registered")
        self._registry[name] = backend
    
    def get(self, name: str) -> DefenseBackend:
        """
        Get a defense backend instance.
        
        Args:
            name: Backend name
            
        Returns:
            DefenseBackend instance
            
        Raises:
            ValueError: If backend name is not registered
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Defense backend '{name}' is not registered. "
                f"Available backends: {available}"
            )
        return self._registry[name]
    
    def get_registered_backends(self) -> List[str]:
        """Get list of registered backend names."""
        return list(self._registry.keys())
    
    def map_defense(self, memory_backend: str, unified_defense: str) -> str:
        """
        Map unified defense name to backend-specific defense type.
        
        Args:
            memory_backend: Memory backend name ("explicit", "mem0", "rag")
            unified_defense: Unified defense name
            
        Returns:
            Backend-specific defense type
        """
        backend = self.get(memory_backend)
        return backend.get_defense_type(unified_defense)


# Global registry instance
_global_defense_registry: Optional[DefenseBackendRegistry] = None


def get_defense_backend_registry() -> DefenseBackendRegistry:
    """Get the global defense backend registry."""
    global _global_defense_registry
    if _global_defense_registry is None:
        _global_defense_registry = DefenseBackendRegistry()
    return _global_defense_registry


# Unified defense names (used in config and CLI)
# Note: To disable memory, use memory_backend="none" (no defense type needed)
# When memory_backend="none", defense_type is automatically "none"
UNIFIED_DEFENSE_TYPES = [
    "none",
    "user_only",
    "no_untrusted_tools",
    "limit_memory_length",
]

