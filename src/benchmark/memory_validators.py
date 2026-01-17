# import json
# import re
# import fnmatch
# from pathlib import Path
# from typing import List, Dict, Any, Union, Optional

# # Note: These imports work because test_bench.py sets up sys.path before importing this module
# from agent.backend.explicit_memory import get_memory_state_for_test as explicit_get_memory_state
# from agent.backend.mem0_memory import get_memory_state_for_test as mem0_get_memory_state
# from agent.backend.rag_memory import get_memory_state_for_test as rag_get_memory_state
# from agent.backend.context_memory import get_memory_state_for_test as context_get_memory_state
# from agent.utils import debug_info, debug_debug, debug_print_exception
# from benchmark.test_validators import TestValidator


# class MemoryValidator(TestValidator):
#     """
#     Validates that memory was updated correctly by checking the memory file.
    
#     Config options:
#     - check_keywords: List of keywords that should be present in memory
#     - check_absence: List of keywords that should NOT be present in memory
#     - memory_file: Path to memory file (default: "data/agent_memory.json")
#     - require_all_keywords: If True, all keywords must be present (default: False)
#     - min_keywords: Minimum number of keywords that must be present
#     """
    
#     def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
#         super().__init__(config, test_config)
#         # Pre-normalize for performance
#         self._check_keywords = config.get("check_keywords", [])
#         self._check_absence = config.get("check_absence", [])
#         self._memory_file = config.get("memory_file", "data/agent/agent_memory.json")
#         self._require_all_keywords = config.get("require_all_keywords", False)
#         self._min_keywords = config.get("min_keywords", None)
        
#         # Pre-normalize keywords
#         self._keywords_lower = [kw.lower() for kw in self._check_keywords]
#         self._absence_lower = [kw.lower() for kw in self._check_absence]
        
#         # Resolve memory file path for test environment
#         if test_config and "data" in test_config:
#             test_memory_file = test_config["data"].get("memory_file", self._memory_file)
#             if test_memory_file != self._memory_file:
#                 self._memory_file = test_memory_file
    
#     def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
#         try:
#             # Read memory file
#             with open(self._memory_file, 'r', encoding='utf-8') as f:
#                 memory_data = json.load(f)
            
#             long_term_memory = memory_data.get("long_term", [])
#             memory_text = " ".join(long_term_memory).lower()
            
#             # Check for required keywords
#             if self._check_keywords:
#                 if not self._check_keywords_in_text(
#                     memory_text,
#                     self._keywords_lower,
#                     self._require_all_keywords,
#                     self._min_keywords
#                 ):
#                     return False
            
#             # Check for absence of keywords
#             if self._check_absence:
#                 for keyword in self._absence_lower:
#                     if keyword in memory_text:
#                         return False  # Found forbidden keyword
            
#             return True
            
#         except Exception as e:
#             debug_info(f"Could not read memory file {self._memory_file}")
#             debug_print_exception(e, context=f"Reading memory file {self._memory_file}", include_traceback=True)
#             return False


# class Mem0MemoryValidator(TestValidator):
#     """
#     Validates that mem0 memory was updated correctly by checking the mem0 vectorstore.
    
#     Config options:
#     - check_keywords: List of keywords that should be present in mem0 memories
#     - check_absence: List of keywords that should NOT be present in mem0 memories
#     - vectorstore_path: Path to mem0 vectorstore (default: from config)
#     - require_all_keywords: If True, all keywords must be present (default: False)
#     - min_keywords: Minimum number of keywords that must be present
#     - user_id: User ID to filter memories (default: "vince")
#     - agent_id: Agent ID to filter memories (default: "email_agent")
#     """
    
#     def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
#         super().__init__(config, test_config)
#         # Pre-normalize for performance
#         self._check_keywords = config.get("check_keywords", [])
#         self._check_absence = config.get("check_absence", [])
#         self._require_all_keywords = config.get("require_all_keywords", False)
#         self._min_keywords = config.get("min_keywords", None)
#         self._user_id = config.get("user_id", "vince")
#         self._agent_id = config.get("agent_id", "email_agent")
        
#         # Pre-normalize keywords
#         self._keywords_lower = [kw.lower() for kw in self._check_keywords]
#         self._absence_lower = [kw.lower() for kw in self._check_absence]
        
#         # Resolve vectorstore path from test config
#         self._vectorstore_path = None
#         if test_config:
#             mem0_config = test_config.get("memory", {}).get("mem0_memory", {})
#             self._vectorstore_path = mem0_config.get("vectorstore_path")
    
#     def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
#         try:
#             # Import here to avoid circular dependencies
#             from agent.backend.mem0_memory import get_mem0_memory_manager
#             from agent.utils import load_config
            
#             # Get config
#             config = load_config()
#             mem0_config = config.get("memory", {}).get("mem0_memory", {})
            
#             # Use test-specific vectorstore path if provided
#             vectorstore_path = self._vectorstore_path or mem0_config.get("vectorstore_path", "data/agent/mem0_vectorstore")
            
#             # Initialize mem0 memory manager
#             mem0_manager = get_mem0_memory_manager(
#                 llm_provider=mem0_config.get("llm_provider", "openai"),
#                 llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
#                 llm_temperature=mem0_config.get("llm_temperature", 0.0),
#                 embedding_provider=mem0_config.get("embedding_provider", "openai"),
#                 embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
#                 vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
#                 vectorstore_path=vectorstore_path,
#                 top_k=mem0_config.get("top_k", 3),
#                 user_id=self._user_id,
#                 agent_id=self._agent_id,
#             )
            
#             # Get all memories from mem0
#             memories = mem0_manager.get_all_memories(user_id=self._user_id, agent_id=self._agent_id, limit=1000)
            
#             # Extract memory texts
#             memory_texts = []
#             for memory in memories:
#                 if isinstance(memory, dict):
#                     memory_text = memory.get("memory", "")
#                     if memory_text:
#                         memory_texts.append(memory_text)
#                 else:
#                     memory_texts.append(str(memory))
            
#             # Combine all memory texts for searching
#             memory_text = " ".join(memory_texts).lower()
            
#             # Check for required keywords
#             if self._check_keywords:
#                 if not self._check_keywords_in_text(
#                     memory_text,
#                     self._keywords_lower,
#                     self._require_all_keywords,
#                     self._min_keywords
#                 ):
#                     return False
            
#             # Check for absence of keywords
#             if self._check_absence:
#                 for keyword in self._absence_lower:
#                     if keyword in memory_text:
#                         return False  # Found forbidden keyword
            
#             return True
            
#         except Exception as e:
#             debug_info("Could not read mem0 memory")
#             debug_print_exception(e, context="Reading mem0 memory for validation", include_traceback=True)
#             return False


# class UnifiedMemoryValidator(TestValidator):
#     """
#     Validates memory state using any memory backend.
    
#     This validator checks the actual memory contents (not tool calls) by querying
#     the memory backend directly. It works with explicit, mem0, and RAG backends.
    
#     Config options:
#     - check_keywords: List of keywords that should be present in memory
#     - check_absence: List of keywords that should NOT be present in memory
#     - require_all_keywords: If True, all keywords must be present (default: False)
#     - min_keywords: Minimum number of keywords that must be present
#     - memory_backend: Backend name to use (default: from config)
#     """
    
#     def __init__(self, config: Dict[str, Any], test_config: Optional[Dict[str, Any]] = None):
#         super().__init__(config, test_config)
        
#         # Pre-normalize for performance
#         self._check_keywords = config.get("check_keywords", [])
#         self._check_absence = config.get("check_absence", [])
#         self._require_all_keywords = config.get("require_all_keywords", False)
#         self._min_keywords = config.get("min_keywords", None)
        
#         # Pre-normalize keywords
#         self._keywords_lower = [kw.lower() for kw in self._check_keywords]
#         self._absence_lower = [kw.lower() for kw in self._check_absence]
        
#         # Determine memory backend
#         self._backend_name = None
#         self._test_dir = None
        
#         if test_config:
#             # Get backend from config
#             memory_config = test_config.get("memory", {})
#             backend_name = memory_config.get("backend")
            
#             if not backend_name:
#                 # Fallback: detect from enabled flags
#                 if memory_config.get("explicit_memory", {}).get("enabled", False):
#                     backend_name = "explicit"
#                 elif memory_config.get("mem0_memory", {}).get("enabled", False):
#                     backend_name = "mem0"
#                 elif memory_config.get("rag_memory", {}).get("enabled", False):
#                     backend_name = "rag"
#                 elif memory_config.get("context_memory", {}).get("enabled", False):
#                     backend_name = "context"
            
#             if backend_name:
#                 # Store backend name for later use
#                 self._backend_name = backend_name
            
#             # Get test directory from test config
#             data_config = test_config.get("data", {})
#             if "mailbox_dir" in data_config:
#                 # Extract test directory from mailbox_dir (e.g., "test_env_abc123/inbox" -> "test_env_abc123")
#                 mailbox_dir = Path(data_config["mailbox_dir"])
#                 self._test_dir = mailbox_dir.parent
    
#     def validate(self, agent_response: str, session_id: str, traces: List[Dict] = None) -> bool:
#         """Validate memory state using the memory backend."""
#         # Skip validation if no memory backend is enabled
#         if not hasattr(self, '_backend_name') or not self._test_dir:
#             # Check if memory backend is explicitly "none" (no memory enabled)
#             if self.test_config:
#                 memory_config = self.test_config.get("memory", {})
#                 if memory_config.get("backend") == "none":
#                     # No memory backend - validation not applicable
#                     print("Warning: UnifiedMemoryValidator: No memory backend enabled (backend='none') - skipping validation")
#                     return True  # Return True to avoid false negatives when memory is intentionally disabled
#             print("Warning: UnifiedMemoryValidator: No memory backend or test directory available")
#             return False
        
#         try:
#             # Get memory state using backend-specific function
#             backend_name = self._backend_name
#             if backend_name == "explicit":
#                 memory_state = explicit_get_memory_state(self._test_dir, self.test_config or {})
#             elif backend_name == "mem0":
#                 memory_state = mem0_get_memory_state(self._test_dir, self.test_config or {})
#             elif backend_name == "rag":
#                 memory_state = rag_get_memory_state(self._test_dir, self.test_config or {})
#             elif backend_name == "context":
#                 memory_state = context_get_memory_state(self._test_dir, self.test_config or {})
#             else:
#                 print(f"Warning: UnifiedMemoryValidator: Unknown backend '{backend_name}'")
#                 return False
            
#             if not memory_state:
#                 return False
            
#             # Combine all memory entries into a single string for searching
#             memory_text = " ".join(memory_state).lower()
            
#             # Check for required keywords
#             if self._check_keywords:
#                 if not self._check_keywords_in_text(
#                     memory_text,
#                     self._keywords_lower,
#                     self._require_all_keywords,
#                     self._min_keywords
#                 ):
#                     return False
            
#             # Check for absence of keywords
#             if self._check_absence:
#                 for keyword in self._absence_lower:
#                     if keyword in memory_text:
#                         return False  # Found forbidden keyword
            
#             return True
            
#         except Exception as e:
#             debug_info("UnifiedMemoryValidator validation failed")
#             debug_print_exception(e, context="UnifiedMemoryValidator validation", include_traceback=True)
#             return False