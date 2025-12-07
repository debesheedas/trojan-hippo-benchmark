"""
Unit tests for memory backend abstractions.
"""

import pytest
import json
import shutil
from pathlib import Path
from typing import Dict, Any

# Add src to path
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from benchmark.memory_backend import (
    ExplicitMemoryBackend,
    Mem0MemoryBackend,
    RAGMemoryBackend,
    MemoryBackendRegistry,
    get_memory_backend_registry
)


@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary test directory."""
    return tmp_path / "test_env"


@pytest.fixture
def config_explicit():
    """Config for explicit memory."""
    return {
        "memory": {
            "explicit_memory": {
                "enabled": True,
                "defense_type": "none",
                "memory_file": "data/interactive_agent/agent_memory.json"
            }
        }
    }


@pytest.fixture
def config_mem0():
    """Config for mem0 memory."""
    return {
        "memory": {
            "mem0_memory": {
                "enabled": True,
                "defense_type": "no_defense",
                "vectorstore_path": "data/interactive_agent/mem0_vectorstore",
                "user_id": "vince",
                "agent_id": None
            }
        }
    }


@pytest.fixture
def config_rag():
    """Config for RAG memory."""
    return {
        "memory": {
            "rag_memory": {
                "enabled": True,
                "defense_type": "none",
                "vectorstore_path": "data/interactive_agent/rag_vectorstore"
            }
        }
    }


class TestExplicitMemoryBackend:
    """Tests for ExplicitMemoryBackend."""
    
    def test_name(self):
        """Test backend name."""
        backend = ExplicitMemoryBackend()
        assert backend.name == "explicit"
    
    def test_initialize_with_existing_memory_set(self, test_dir, config_explicit):
        """Test initializing with an existing memory set."""
        # Ensure test_dir exists
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a test memory set
        initial_memory_dir = Path("data/benchmark/initial_explicit_memory")
        initial_memory_dir.mkdir(parents=True, exist_ok=True)
        
        test_memory_set = "test_memory_set_0"
        test_memory_file = initial_memory_dir / f"{test_memory_set}.json"
        test_memory_data = {
            "long_term": [
                "User prefers concise emails",
                "User's name is Vince Kaminski"
            ]
        }
        
        with open(test_memory_file, 'w') as f:
            json.dump(test_memory_data, f)
        
        try:
            backend = ExplicitMemoryBackend()
            backend.initialize(test_memory_set, test_dir, config_explicit)
            
            # Verify memory file was created
            memory_file = test_dir / "agent_memory.json"
            assert memory_file.exists()
            
            # Verify contents
            with open(memory_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_memory_data
        finally:
            # Cleanup
            if test_memory_file.exists():
                test_memory_file.unlink()
    
    def test_initialize_without_memory_set(self, test_dir, config_explicit):
        """Test initializing without a memory set (creates empty)."""
        # Ensure test_dir exists
        test_dir.mkdir(parents=True, exist_ok=True)
        
        backend = ExplicitMemoryBackend()
        backend.initialize("nonexistent_set", test_dir, config_explicit)
        
        # Verify empty memory file was created
        memory_file = test_dir / "agent_memory.json"
        assert memory_file.exists()
        
        with open(memory_file, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == {"long_term": []}
    
    def test_get_memory_state(self, test_dir, config_explicit):
        """Test getting memory state."""
        # Ensure test_dir exists
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create memory file
        memory_file = test_dir / "agent_memory.json"
        memory_data = {
            "long_term": [
                "User prefers concise emails",
                "User's name is Vince"
            ]
        }
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f)
        
        backend = ExplicitMemoryBackend()
        state = backend.get_memory_state(test_dir, config_explicit)
        
        assert len(state) == 2
        assert "concise emails" in state[0]
        assert "Vince" in state[1]
    
    def test_clear_memory(self, test_dir, config_explicit):
        """Test clearing memory."""
        # Ensure test_dir exists
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create memory file with data
        memory_file = test_dir / "agent_memory.json"
        memory_data = {"long_term": ["Some memory"]}
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f)
        
        backend = ExplicitMemoryBackend()
        backend.clear_memory(test_dir)
        
        # Verify cleared
        with open(memory_file, 'r') as f:
            cleared_data = json.load(f)
        assert cleared_data == {"long_term": []}


class TestMemoryBackendRegistry:
    """Tests for MemoryBackendRegistry."""
    
    def test_default_registration(self):
        """Test that default backends are registered."""
        registry = MemoryBackendRegistry()
        backends = registry.get_registered_backends()
        
        assert "explicit" in backends
        assert "mem0" in backends
        assert "rag" in backends
    
    def test_create_explicit(self, config_explicit):
        """Test creating explicit backend."""
        registry = MemoryBackendRegistry()
        backend = registry.create("explicit", config_explicit)
        
        assert backend.name == "explicit"
        assert isinstance(backend, ExplicitMemoryBackend)
    
    def test_create_mem0(self, config_mem0):
        """Test creating mem0 backend."""
        registry = MemoryBackendRegistry()
        backend = registry.create("mem0", config_mem0)
        
        assert backend.name == "mem0"
        assert isinstance(backend, Mem0MemoryBackend)
    
    def test_create_rag(self, config_rag):
        """Test creating RAG backend."""
        registry = MemoryBackendRegistry()
        backend = registry.create("rag", config_rag)
        
        assert backend.name == "rag"
        assert isinstance(backend, RAGMemoryBackend)
    
    def test_create_unknown_backend(self):
        """Test creating unknown backend raises error."""
        registry = MemoryBackendRegistry()
        
        with pytest.raises(ValueError, match="not registered"):
            registry.create("unknown", {})
    
    def test_register_custom_backend(self):
        """Test registering a custom backend."""
        class CustomBackend:
            @property
            def name(self):
                return "custom"
            
            def initialize(self, memory_set, test_dir, config):
                pass
            
            def get_memory_state(self, test_dir, config):
                return []
            
            def clear_memory(self, test_dir):
                pass
        
        registry = MemoryBackendRegistry()
        registry.register("custom", CustomBackend)
        
        backend = registry.create("custom", {})
        assert backend.name == "custom"
        assert isinstance(backend, CustomBackend)
    
    def test_register_duplicate(self):
        """Test registering duplicate backend raises error."""
        registry = MemoryBackendRegistry()
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register("explicit", ExplicitMemoryBackend)
    
    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_memory_backend_registry()
        registry2 = get_memory_backend_registry()
        
        assert registry1 is registry2


class TestBackendIntegration:
    """Integration tests for backends."""
    
    def test_explicit_backend_full_cycle(self, test_dir, config_explicit):
        """Test full cycle: initialize -> get state -> clear."""
        # Ensure test_dir exists
        test_dir.mkdir(parents=True, exist_ok=True)
        
        backend = ExplicitMemoryBackend()
        
        # Initialize
        backend.initialize("nonexistent", test_dir, config_explicit)
        
        # Get state (should be empty)
        state = backend.get_memory_state(test_dir, config_explicit)
        assert state == []
        
        # Manually add memory
        memory_file = test_dir / "agent_memory.json"
        with open(memory_file, 'r') as f:
            data = json.load(f)
        data["long_term"].append("Test memory")
        with open(memory_file, 'w') as f:
            json.dump(data, f)
        
        # Get state again
        state = backend.get_memory_state(test_dir, config_explicit)
        assert len(state) == 1
        assert "Test memory" in state[0]
        
        # Clear
        backend.clear_memory(test_dir)
        state = backend.get_memory_state(test_dir, config_explicit)
        assert state == []

