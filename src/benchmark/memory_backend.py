"""
Memory Backend Abstraction

Provides a unified interface for all memory systems (explicit, mem0, rag).
Uses protocol-based design inspired by prompt-siren's architecture.
"""

import json
import shutil
from pathlib import Path
from typing import Protocol, Dict, Any, List, Optional, Type
from datetime import datetime, timezone
import sys
from agent.utils import debug_info, debug_debug, debug_print_exception

# LangChain imports for RAG (optional, only used when RAG backend is used)
try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None
    FAISS = None

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


class MemoryBackend(Protocol):
    """Protocol defining the interface for memory backends."""
    
    @property
    def name(self) -> str:
        """Backend name: 'explicit', 'mem0', or 'rag'."""
        ...
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """
        Initialize memory from a memory set.
        
        Args:
            memory_set: Name of the memory set (e.g., "memory_set_0")
            test_dir: Test-specific directory for isolated memory storage
            config: Configuration dictionary
        """
        ...
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """
        Get current memory contents as list of strings.
        
        Args:
            test_dir: Test-specific directory
            config: Configuration dictionary
            
        Returns:
            List of memory strings
        """
        ...
    
    def clear_memory(self, test_dir: Path) -> None:
        """
        Clear memory for test isolation.
        
        Args:
            test_dir: Test-specific directory
        """
        ...


class ExplicitMemoryBackend:
    """Backend for explicit (JSON-based) memory."""
    
    @property
    def name(self) -> str:
        return "explicit"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize explicit memory (always starts empty - memory built during test execution).
        
        Args:
            memory_set: Ignored (kept for interface compatibility)
            test_dir: Test-specific directory
            config: Configuration dictionary
        """
        # All tests start with empty memory - memory is built during test execution
        memory_file = test_dir / "agent_memory.json"
        
        # Create empty memory file if it doesn't exist
        if not memory_file.exists():
            empty_memory = {"long_term": []}
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(empty_memory, f, indent=2)
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get explicit memory contents."""
        memory_file = test_dir / "agent_memory.json"
        
        if not memory_file.exists():
            return []
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            long_term_memory = memory_data.get("long_term", [])
            return [str(entry) for entry in long_term_memory]
        except Exception as e:
            debug_info(f"Could not read memory file {memory_file}")
            debug_print_exception(e, context=f"Reading memory file {memory_file}", include_traceback=True)
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear explicit memory."""
        memory_file = test_dir / "agent_memory.json"
        if memory_file.exists():
            empty_memory = {"long_term": []}
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(empty_memory, f, indent=2)


class Mem0MemoryBackend:
    """Backend for mem0 (vectorstore-based) memory."""
    
    @property
    def name(self) -> str:
        return "mem0"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize mem0 memory (always starts empty - memory built during test execution).
        
        Args:
            memory_set: Ignored (kept for interface compatibility)
            test_dir: Test-specific directory
            config: Configuration dictionary
        """
        # All tests start with empty memory - memory is built during test execution
        # The mem0 manager will create an empty vectorstore on first use
        # No initialization needed here
        pass
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get mem0 memory contents.
        
        For efficiency, this method first checks for recent memories added in the current step.
        If no recent memories file exists, it falls back to checking all memories from the vectorstore.
        """
        import json
        
        # First, try to get recent memories (much faster - only checks what was added this step)
        recent_memories_file = test_dir / "mem0_recent_memories.json"
        if recent_memories_file.exists():
            try:
                with open(recent_memories_file, 'r', encoding='utf-8') as f:
                    recent_memories = json.load(f)
                if recent_memories:
                    # Return recent memories - these are what were extracted and added in the current turn
                    return recent_memories
            except Exception as e:
                # If we can't read recent memories, fall back to full retrieval
                debug_debug(f"Could not read recent memories file, falling back to full retrieval")
                debug_print_exception(e, context="Reading recent memories file", include_traceback=True)
        
        # Fallback: Get all memories from the vectorstore
        # This is slower but ensures we check everything if recent memories aren't available
        from agent.backend.mem0_memory import get_mem0_memory_manager
        
        mem0_config = config.get("memory", {}).get("mem0_memory", {})
        vectorstore_path = str(test_dir / "mem0_vectorstore")
        user_id = mem0_config.get("user_id", "vince")
        agent_id = mem0_config.get("agent_id", None)
        
        try:
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                vectorstore_path=vectorstore_path,
                top_k=mem0_config.get("top_k", 3),
                user_id=user_id,
                agent_id=agent_id,
            )
            
            memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=agent_id, limit=1000)
            
            memory_texts = []
            for memory in memories:
                if isinstance(memory, dict):
                    # Try multiple possible keys for mem0 memory content
                    # mem0 may return memories with different key names depending on version
                    memory_text = (
                        memory.get("memory") or
                        memory.get("memories") or
                        memory.get("text") or
                        memory.get("content") or
                        memory.get("fact") or
                        ""
                    )
                    # If still empty, try to get the first string value
                    if not memory_text:
                        for value in memory.values():
                            if isinstance(value, str) and value.strip():
                                memory_text = value
                                break
                    if memory_text:
                        memory_texts.append(str(memory_text))
                else:
                    memory_texts.append(str(memory))
            
            return memory_texts
        except Exception as e:
            debug_info("Could not read mem0 memory")
            debug_print_exception(e, context="Reading mem0 memory", include_traceback=True)
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear mem0 memory."""
        vectorstore_path = test_dir / "mem0_vectorstore"
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)


class RAGMemoryBackend:
    """Backend for RAG (vectorstore-based) memory."""
    
    @property
    def name(self) -> str:
        return "rag"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize RAG memory (always starts empty - memory built during test execution).
        
        Args:
            memory_set: Ignored (kept for interface compatibility)
            test_dir: Test-specific directory
            config: Configuration dictionary
        """
        # All tests start with empty memory - memory is built during test execution
        # The RAG memory manager will create an empty vectorstore on first use
        # No initialization needed here
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get RAG memory contents.
        
        For efficiency, this method first checks for recent chunks added in the current step.
        If no recent chunks file exists, it falls back to checking all documents.
        """
        import json
        
        # First, try to get recent chunks (much faster - only checks what was added this step)
        recent_chunks_file = test_dir / "rag_recent_chunks.json"
        if recent_chunks_file.exists():
            try:
                with open(recent_chunks_file, 'r', encoding='utf-8') as f:
                    recent_chunks = json.load(f)
                if recent_chunks:
                    # Return recent chunks - these are what were added in the current turn
                    return recent_chunks
            except Exception as e:
                # If we can't read recent chunks, fall back to full retrieval
                debug_debug(f"Could not read recent chunks file, falling back to full retrieval")
                debug_print_exception(e, context="Reading recent chunks file", include_traceback=True)
        
        # Fallback: Get all memory chunks from the documents list
        # This is slower but ensures we check everything if recent chunks aren't available
        from agent.backend.rag_memory import get_rag_memory_manager
        
        rag_config = config.get("memory", {}).get("rag_memory", {})
        vectorstore_path = str(test_dir / "rag_vectorstore")
        
        try:
            rag_memory_manager = get_rag_memory_manager(
                embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
                top_k=rag_config.get("top_k", 3),
                chunk_size=rag_config.get("chunk_size", 512),
                vectorstore_path=vectorstore_path,
            )
            
            # Get all memory chunks from the documents list
            # RAGMemoryManager stores all chunks in self.documents
            if hasattr(rag_memory_manager, 'documents') and rag_memory_manager.documents:
                # documents is a list of strings (chunk texts)
                return list(rag_memory_manager.documents)
            
            # Fallback: Try to retrieve from vectorstore using a broad query
            if rag_memory_manager.vectorstore:
                try:
                    results = rag_memory_manager.vectorstore.similarity_search("", k=1000)
                    return [doc.page_content for doc in results]
                except Exception as e:
                    debug_debug("Could not retrieve from vectorstore, returning empty list")
                    debug_print_exception(e, context="Retrieving from RAG vectorstore", include_traceback=True)
            
            return []
        except Exception as e:
            debug_info("Could not read RAG memory")
            debug_print_exception(e, context="Reading RAG memory", include_traceback=True)
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear RAG memory."""
        vectorstore_path = test_dir / "rag_vectorstore"
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)
    


class ContextMemoryBackend:
    """Backend for context (simple list-based) memory."""
    
    @property
    def name(self) -> str:
        return "context"
    
    def initialize(self, memory_set: str, test_dir: Path, config: Dict[str, Any]) -> None:
        """Initialize context memory (always starts empty - memory built during test execution).
        
        Args:
            memory_set: Ignored (kept for interface compatibility)
            test_dir: Test-specific directory
            config: Configuration dictionary
        """
        # All tests start with empty memory - memory is built during test execution
        # The context memory manager will create an empty context file on first use
        # No initialization needed here
        context_file = test_dir / "context_memory.json"
        if not context_file.exists():
            empty_context = {"history": []}
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(empty_context, f, indent=2)
    
    def get_memory_state(self, test_dir: Path, config: Dict[str, Any]) -> List[str]:
        """Get context memory contents.
        
        For efficiency, this method first checks for recent entries added in the current step.
        If no recent entries file exists, it falls back to checking all history.
        """
        import json
        
        # First, try to get recent entries (much faster - only checks what was added this step)
        recent_entries_file = test_dir / "context_recent_chunks.json"
        if recent_entries_file.exists():
            try:
                with open(recent_entries_file, 'r', encoding='utf-8') as f:
                    recent_entries = json.load(f)
                if recent_entries:
                    # Return recent entries - these are what were added in the current turn
                    return recent_entries
            except Exception as e:
                # If we can't read recent entries, fall back to full retrieval
                debug_debug(f"Could not read recent entries file, falling back to full retrieval")
                debug_print_exception(e, context="Reading recent entries file", include_traceback=True)
        
        # Fallback: Get all memory entries from the context file
        # This is slower but ensures we check everything if recent entries aren't available
        from agent.backend.context_memory import get_context_memory_manager
        
        context_config = config.get("memory", {}).get("context_memory", {})
        context_path = str(test_dir / "context_memory.json")
        
        try:
            # Get max_context_length from config if specified
            max_context_length = context_config.get("max_context_length")
            model_name = config.get("agent", {}).get("target_model_name", "gpt-5-mini")
            
            context_manager = get_context_memory_manager(
                context_path=context_path,
                max_context_length=max_context_length,
                model_name=model_name,
            )
            
            # Get all history entries
            if hasattr(context_manager, 'history') and context_manager.history:
                return list(context_manager.history)
            
            return []
        except Exception as e:
            debug_info("Could not read context memory")
            debug_print_exception(e, context="Reading context memory", include_traceback=True)
            return []
    
    def clear_memory(self, test_dir: Path) -> None:
        """Clear context memory."""
        context_path = test_dir / "context_memory.json"
        if context_path.exists():
            context_path.unlink()
        
        # Also clear recent chunks file if it exists
        recent_chunks_file = test_dir / "context_recent_chunks.json"
        if recent_chunks_file.exists():
            recent_chunks_file.unlink()


class MemoryBackendRegistry:
    """Registry for memory backends (inspired by prompt-siren's registry pattern)."""
    
    def __init__(self):
        self._registry: Dict[str, Type[MemoryBackend]] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default memory backends."""
        self.register("explicit", ExplicitMemoryBackend)
        self.register("mem0", Mem0MemoryBackend)
        self.register("rag", RAGMemoryBackend)
        self.register("context", ContextMemoryBackend)
    
    def register(self, name: str, backend_class: Type[MemoryBackend]) -> None:
        """
        Register a memory backend.
        
        Args:
            name: Backend name (e.g., "explicit", "mem0", "rag")
            backend_class: Backend class implementing MemoryBackend protocol
        """
        if name in self._registry:
            raise ValueError(f"Memory backend '{name}' is already registered")
        self._registry[name] = backend_class
    
    def create(self, name: str, config: Dict[str, Any]) -> MemoryBackend:
        """
        Create a memory backend instance.
        
        Args:
            name: Backend name
            config: Configuration dictionary
            
        Returns:
            MemoryBackend instance
            
        Raises:
            ValueError: If backend name is not registered
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Memory backend '{name}' is not registered. "
                f"Available backends: {available}"
            )
        
        backend_class = self._registry[name]
        return backend_class()
    
    def get_registered_backends(self) -> List[str]:
        """Get list of registered backend names."""
        return list(self._registry.keys())


# Global registry instance
_global_registry: Optional[MemoryBackendRegistry] = None


def get_memory_backend_registry() -> MemoryBackendRegistry:
    """Get the global memory backend registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MemoryBackendRegistry()
    return _global_registry

