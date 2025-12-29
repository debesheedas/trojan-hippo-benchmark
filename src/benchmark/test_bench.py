#!/usr/bin/env python3
"""
Pure-Python Test Bench for Email Agent

This test bench runs the same tests as test_suite.py but uses the pure-Python
agent core (no HTTP) for maximum speed and reliability.

Test Structure:
- Each test is a JSON file with a series of user queries
- Agent is loaded fresh for each test with clean session history
- Existing long-term memory is preserved across tests
- Results are automatically validated and reported
"""

import json
import time
import shutil
import tempfile
import uuid
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ensure src/ is on sys.path for package imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Load environment variables from .env file
load_dotenv()

from agent.agent_core import invoke_agent, clear_session_agent, clear_agent_cache
from agent.utils import read_trace_events, load_config, ensure_data_directories, set_global_seeds
from benchmark.test_validators import create_validator, CompositeValidator
from agent.colored_trace_printer import get_colored_printer
from benchmark.memory_backend import get_memory_backend_registry
from benchmark.defense_backend import get_defense_backend_registry
from benchmark.benchmark_utils import (
    get_memory_backend_from_config,
    get_unified_defense_from_config,
    get_result_path,
    should_skip_test,
    create_isolated_test_dir,
    determine_attack_type,
    discover_test_files as discover_test_files_util
)


class TestBench:
    """Pure-Python test bench for email agent."""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, defense_type_override: Optional[str] = None, force: bool = False):
        """
        Initialize TestBench.
        
        Args:
            config_path: Path to config YAML file (mutually exclusive with config)
            config: Config dictionary (mutually exclusive with config_path)
            defense_type_override: Override defense type from config
            force: Force overwrite existing results
        """
        if config_path is not None and config is not None:
            raise ValueError("Cannot specify both config_path and config")
        if config_path is None and config is None:
            config_path = "config.yaml"  # Default to config.yaml
        
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            raise ValueError("Must specify either config_path or config")
        
        # Track execution errors for this benchmark run (simple flag, no complex tracker)
        self.execution_errors = []  # List of error messages
        self.execution_success = True  # Set to False if any execution errors occur
        
        # Set global seed for reproducibility
        global_seed = self.config.get("seed", 42)
        set_global_seeds(global_seed)
        
        # Read model name strictly from config.yaml
        self.model_name = self.config.get("agent", {}).get("target_model_name")
        if not self.model_name:
            raise ValueError("Config missing agent.target_model_name. Please set it in config.yaml.")
        self.test_dirs = []  # Track test directories for cleanup
        self.force = force  # Force overwrite existing results
        
        # Get memory backend from config
        memory_config = self.config.get("memory", {})
        backend_from_config = memory_config.get("backend")
        
        # Handle no memory backend: if backend is "none"
        if backend_from_config == "none":
            self.memory_backend_name = "none"
            self.memory_backend = None  # No backend when memory is disabled
            self.unified_defense = "none"  # No memory = no defense
            # Disable all memory backends
            for backend_name in ["explicit", "mem0", "rag", "context"]:
                if backend_name not in memory_config:
                    memory_config[backend_name] = {}
                memory_config[backend_name]["enabled"] = False
        else:
            # Get memory backend from config
            self.memory_backend_name = get_memory_backend_from_config(self.config)
            memory_registry = get_memory_backend_registry()
            self.memory_backend = memory_registry.create(self.memory_backend_name, self.config)
            
            # Get unified defense type
            if defense_type_override is not None:
                self.unified_defense = defense_type_override
                # Update config with override
                memory_config = self.config.get("memory", {})
                if self.memory_backend_name == "explicit":
                    if "explicit_memory" not in memory_config:
                        memory_config["explicit_memory"] = {}
                    memory_config["explicit_memory"]["defense_type"] = defense_type_override
                elif self.memory_backend_name == "mem0":
                    if "mem0_memory" not in memory_config:
                        memory_config["mem0_memory"] = {}
                    memory_config["mem0_memory"]["defense_type"] = defense_type_override
                elif self.memory_backend_name == "rag":
                    if "rag_memory" not in memory_config:
                        memory_config["rag_memory"] = {}
                    memory_config["rag_memory"]["defense_type"] = defense_type_override
                elif self.memory_backend_name == "context":
                    if "context_memory" not in memory_config:
                        memory_config["context_memory"] = {}
                    memory_config["context_memory"]["defense_type"] = defense_type_override
            else:
                self.unified_defense = get_unified_defense_from_config(self.config, self.memory_backend_name)
        
        # Map unified defense to backend-specific defense
        if self.memory_backend_name == "none":
            # No memory backend = no defense
            self.backend_defense = "none"
        else:
            defense_registry = get_defense_backend_registry()
            self.backend_defense = defense_registry.map_defense(self.memory_backend_name, self.unified_defense)
        
        # For backward compatibility, maintain old flags
        memory_config = self.config.get("memory", {})
        self.explicit_memory_enabled = memory_config.get("explicit_memory", {}).get("enabled", False)
        mem0_config = memory_config.get("mem0_memory", {})
        self.mem0_memory_enabled = mem0_config.get("enabled", False)
        self.mem0_print_enabled = mem0_config.get("mem0_print", False)
        self.rag_memory_enabled = memory_config.get("rag_memory", {}).get("enabled", False)
        self.defense_type = self.backend_defense  # For backward compatibility
        
        # Unified results directory structure
        # Note: results_dir is set by run_benchmark.py from command line args
        # If not set, use hardcoded default (not from config to avoid conflicts)
        benchmark_config = self.config.get("benchmark", {})
        results_base_dir = Path(benchmark_config.get("results_dir", "data/benchmark/results"))
        self.results_base_dir = results_base_dir
        self.results_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified test directory
        test_dir = benchmark_config.get("test_dir", "data/benchmark/tests")
        self.test_bench_dir = Path(test_dir)
        
        # For backward compatibility (used by some old code paths)
        self.memory_type = self.memory_backend_name
        
        print(f"🧠 Memory Backend: {self.memory_backend_name.upper()}")
        print(f"🛡️ Defense: {self.unified_defense} (backend: {self.backend_defense})")
        print(f"📁 Test Directory: {self.test_bench_dir}")
        print(f"📊 Results Directory: {self.results_base_dir}")
        if self.force:
            print(f"⚡ Force mode: Will overwrite existing results")
        
        # Check if adaptive benchmark is enabled
        self.adaptive_enabled = self.config.get("benchmark", {}).get("enable_adaptive_benchmark", False)
        if self.adaptive_enabled:
            # Import adaptive components only when needed
            from benchmark.adaptive_attacks import DSPyOptimizer, OpenEvolveOptimizer
            from agent.utils import compare_attack_bench_files
            from benchmark.environment_state import StateManager
            
            # Initialize optimization strategies based on config
            self.optimizers = {}
            
            # Check if DSPy is enabled
            benchmark_config = self.config.get("benchmark", {})
            # Check if DSPy is enabled
            if benchmark_config.get("dspy", {}).get("enabled", True):
                self.optimizers["dspy"] = DSPyOptimizer(self.config)
            
            # Check if OpenEvolve is enabled
            if benchmark_config.get("openevolve", {}).get("enabled", True):
                self.optimizers["openevolve"] = OpenEvolveOptimizer(self.config)
            
            if not self.optimizers:
                print("⚠️ Warning: No optimizers enabled in config. Adaptive benchmark will not work.")
            
            # Cache directory for successful attacks
            self.cache_dir = Path("data/benchmark/attack_bench_cache")
            self.cache_dir.mkdir(exist_ok=True)
            
            # Initialize state manager for environment state tracking
            self.state_manager = StateManager()

            # Set up logging for optimizers when running via TestBench
            self._setup_optimizer_logging()
            for optimizer in self.optimizers.values():
                optimizer.set_logger(self.logger)
            
            print("🔧 Adaptive benchmark mode ENABLED")
        else:
            print("📊 Static benchmark mode")
    
    def _debug_print_initial_mem0_memories(self, test_config: Dict[str, Any]):
        """Debug print: Print all initial mem0 memories loaded in the vectorstore at the start of a test case."""
        if not self.mem0_memory_enabled:
            return
        
        # Skip debug printing if no memory backend is enabled
        if self.memory_backend_name == "none":
            return
        
        try:
            from agent.backend.mem0_memory_manager import get_mem0_memory_manager
            from pathlib import Path
            
            mem0_config = test_config.get("memory", {}).get("mem0_memory", {})
            vectorstore_path = mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore")
            user_id = mem0_config.get("user_id", "vince")
            # Memories are stored with agent_id=None, so we query with None
            agent_id = None
            
            # Check if vectorstore directory exists and has files
            vectorstore_path_obj = Path(vectorstore_path)
            faiss_files = list(vectorstore_path_obj.glob("*.faiss")) if vectorstore_path_obj.exists() else []
            
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Initial mem0 Memories in Vectorstore")
            print(f"{'='*80}")
            print(f"Vectorstore Path: {vectorstore_path}")
            print(f"Vectorstore Exists: {vectorstore_path_obj.exists()}")
            print(f"FAISS Files Found: {len(faiss_files)}")
            print(f"User ID: {user_id}, Agent ID: {agent_id} (memories are stored with agent_id=None)")
            
            # Initialize mem0 memory manager
            # Note: agent_id parameter here is just for manager initialization, not for querying
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
                agent_id=None,  # Memories are stored with agent_id=None
            )
            
            # Get all memories with agent_id=None (this is how they're stored)
            all_memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=None, limit=1000)
            
            print(f"Total Memories Found: {len(all_memories)}")
            print(f"{'-'*80}")
            
            if all_memories:
                for i, memory in enumerate(all_memories, 1):
                    if isinstance(memory, dict):
                        memory_text = memory.get("memory", "")
                        memory_id = memory.get("id", "unknown")
                        memory_metadata = memory.get("metadata", {})
                        memory_user_id = memory.get("user_id", "unknown")
                        memory_agent_id = memory.get("agent_id", "None")
                        print(f"\n[{i}] Memory ID: {memory_id}")
                        print(f"    User ID: {memory_user_id}, Agent ID: {memory_agent_id}")
                        print(f"    Text: {memory_text}")
                        if memory_metadata:
                            print(f"    Metadata: {memory_metadata}")
                    else:
                        print(f"\n[{i}] {str(memory)}")
            else:
                print("(No memories found in vectorstore)")
                # Additional debugging: try to get users/stats
                try:
                    users = mem0_manager.get_users()
                    print(f"\nDebug: get_users() returned: {users}")
                except Exception as e:
                    print(f"\nDebug: Could not get users: {e}")
            
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\n⚠️ DEBUG: Could not print initial mem0 memories: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_mem0_memories(self, test_config: Dict[str, Any], user_message: Optional[str] = None):
        """Print mem0 memory contents and context if mem0_print is enabled."""
        if not (self.mem0_memory_enabled and self.mem0_print_enabled):
            return
        
        # Skip printing if no memory backend is enabled
        if self.memory_backend_name == "none":
            return
        
        try:
            from agent.backend.mem0_memory_manager import get_mem0_memory_manager
            from agent.agent_core import _build_agent_prompt
            
            mem0_config = test_config.get("memory", {}).get("mem0_memory", {})
            vectorstore_path = mem0_config.get("vectorstore_path", "data/interactive_agent/mem0_vectorstore")
            user_id = mem0_config.get("user_id", "vince")
            agent_id = mem0_config.get("agent_id", "email_agent")
            
            # Initialize mem0 memory manager
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                vectorstore_path=vectorstore_path,
                top_k=mem0_config.get("top_k", 3),
                user_id=user_id,
                agent_id=agent_id,
            )
            
            # Get all memories
            memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=agent_id, limit=1000)
            
            if memories:
                print("\n🧠 Mem0 Memories:")
                print("-" * 80)
                for memory in memories:
                    if isinstance(memory, dict):
                        memory_text = memory.get("memory", "")
                        if memory_text:
                            print(f"  • {memory_text}")
                    else:
                        print(f"  • {str(memory)}")
                print("-" * 80)
            else:
                print("\n🧠 Mem0 Memories: (empty)")
            
            # Get mem0 context that would be added to user message (if user_message provided)
            if user_message:
                try:
                    # Build the final user message exactly as it would be sent to the model
                    context_parts = []
                    
                    # Check for RAG context (if RAG memory is enabled)
                    rag_context = ""
                    memory_config = test_config.get("memory", {})
                    rag_memory_enabled = memory_config.get("rag_memory", {}).get("enabled", False)
                    rag_defense_type = memory_config.get("rag_memory", {}).get("defense_type", "none")
                    # Skip RAG context retrieval if no memory backend is enabled
                    if rag_memory_enabled and self.memory_backend_name != "none" and rag_defense_type != "disable_memory":
                        try:
                            from agent.backend.rag_memory_manager import get_rag_memory_manager
                            rag_config = memory_config.get("rag_memory", {})
                            rag_memory_manager = get_rag_memory_manager(
                                embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
                                top_k=rag_config.get("top_k", 3),
                                chunk_size=rag_config.get("chunk_size", 512),
                                vectorstore_path=rag_config.get("vectorstore_path", "data/interactive_agent/rag_vectorstore"),
                            )
                            # Pass session_id and defense_type for provable_policy defense
                            rag_context = rag_memory_manager.get_context(
                                user_message,
                                session_id=session_id,
                                defense_type=rag_defense_type
                            )
                            if rag_context:
                                rag_context = "\n\n# Relevant Memory Context\n" + rag_context + "\n"
                                context_parts.append(rag_context)
                        except Exception:
                            pass
                    
                    # Get mem0 context
                    mem0_context = mem0_manager.get_context(user_message, user_id=user_id)
                    if mem0_context:
                        formatted_mem0_context = "\n\n# Relevant Mem0 Memory Context\n" + mem0_context + "\n"
                        context_parts.append(formatted_mem0_context)
                        print("\n📝 Mem0 Context Added to User Message:")
                        print("-" * 80)
                        print(formatted_mem0_context)
                        print("-" * 80)
                    else:
                        print("\n📝 Mem0 Context Added to User Message: (none - no relevant memories found)")
                    
                    # Build final user message (same logic as in agent_core.py)
                    final_user_message = "".join(context_parts) + user_message if context_parts else user_message
                    
                    print("\n💬 Final User Message Sent to Model:")
                    print("=" * 80)
                    print(final_user_message)
                    print("=" * 80)
                except Exception as e:
                    print(f"\n⚠️ Warning: Could not retrieve mem0 context: {e}")
                    print("\n💬 Final User Message Sent to Model:")
                    print("=" * 80)
                    print(user_message)
                    print("=" * 80)
            else:
                print("\n💬 Final User Message Sent to Model: (no user message provided)")
            
            # Print system prompt (memory-related parts)
            try:
                from pathlib import Path
                from agent.agent_core import _build_agent_prompt
                
                memory_config = test_config.get("memory", {})
                explicit_memory_enabled = memory_config.get("explicit_memory", {}).get("enabled", False)
                
                # Load memory instructions if explicit memory is enabled
                memory_instructions = ""
                explicit_memory_context = ""
                if explicit_memory_enabled:
                    # Find memory_prompt.txt relative to agent_core.py location
                    agent_core_path = Path(__file__).parent.parent / "agent" / "memory_prompt.txt"
                    if agent_core_path.exists():
                        memory_instructions = agent_core_path.read_text(encoding="utf-8")
                    
                    # Try to load explicit memory context
                    try:
                        from agent.backend.memory_manager import get_memory_manager
                        memory_file = memory_config.get("explicit_memory", {}).get("memory_file", 
                            test_config.get("data", {}).get("memory_file", "data/interactive_agent/agent_memory.json"))
                        memory_manager = get_memory_manager(memory_file=memory_file)
                        # Note: For provable_policy defense, session_id and defense_type should be passed
                        # but in test_bench context, we don't have session_id here, so pass None
                        # The defense will still work when memories are retrieved in agent_core
                        explicit_memory_context = memory_manager.get_long_term_as_text()
                    except Exception:
                        explicit_memory_context = ""
                
                print("\n📋 System Prompt (Memory-Related Sections):")
                print("=" * 80)
                if memory_instructions:
                    print("\n[Memory Instructions Section]")
                    print("-" * 80)
                    print(memory_instructions)
                    print("-" * 80)
                if explicit_memory_context:
                    print("\n[Explicit Memory Context Section]")
                    print("-" * 80)
                    print(explicit_memory_context)
                    print("-" * 80)
            except Exception as e:
                print(f"\n⚠️ Warning: Could not print system prompt: {e}")
                
        except Exception as e:
            print(f"\n⚠️ Warning: Could not print mem0 memories: {e}")
    
    def _setup_optimizer_logging(self):
        """Initialize a file logger for optimization runs when using TestBench."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.logger = logging.getLogger("adaptive_benchmark")
            self.logger.setLevel(logging.DEBUG)
            # Clear existing handlers to avoid duplication
            for h in self.logger.handlers[:]:
                self.logger.removeHandler(h)
            from datetime import datetime
            # Use current time for unique log file per run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"adaptive_benchmark_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 'w' mode to overwrite/create new file
            file_handler.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.info("Logging initialized from TestBench. Detailed logs: %s", log_file)
        except Exception as e:
            print(f"Warning: Failed to initialize optimizer logging: {e}")

    def create_test_environment(self, test_name: str, initial_data: Dict[str, str] = None) -> Dict[str, str]:
        """
        Create an isolated test environment with specified initial data.
        
        Uses memory backend for unified memory initialization.
        
        Args:
            test_name: Name of the test (for logging)
            initial_data: Dict with keys 'inbox_set', 'outbox_set', 'drafts_set', 'memory' (unified format), 'session_set'
        
        Returns:
            Config dict with test-specific paths.
        """
        # Create isolated test directory with unique name (PID + timestamp for parallel execution)
        test_dir = create_isolated_test_dir(test_name)
        self.test_dirs.append(test_dir)
        
        # Create subdirectories
        inbox_dir = test_dir / "inbox"
        outbox_dir = test_dir / "outbox"
        drafts_dir = test_dir / "drafts"
        sessions_dir = test_dir / "sessions"
        traces_dir = test_dir / "traces"
        mem0_vectorstore_dir = test_dir / "mem0_vectorstore"
        rag_vectorstore_dir = test_dir / "rag_vectorstore"
        
        # Create directories (but NOT mem0_vectorstore_dir or rag_vectorstore_dir - let loaders create them)
        # This prevents false warnings about the directory already existing
        dirs_to_create = [inbox_dir, outbox_dir, drafts_dir, sessions_dir, traces_dir]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(exist_ok=True)
        
        # Copy initial data based on configuration
        if initial_data:
            # Copy inbox data
            if "inbox_set" in initial_data:
                inbox_set = initial_data["inbox_set"]
                source_inbox = Path(f"data/benchmark/initial_environment/initial_inbox/{inbox_set}")
                if source_inbox.exists():
                    for email_file in source_inbox.glob("*.json"):
                        shutil.copy2(email_file, inbox_dir)
                else:
                    print(f"Warning: Inbox set '{inbox_set}' not found at {source_inbox}")
            
            # Copy outbox data
            if "outbox_set" in initial_data:
                outbox_set = initial_data["outbox_set"]
                source_outbox = Path(f"data/benchmark/initial_environment/initial_outbox/{outbox_set}")
                if source_outbox.exists():
                    for email_file in source_outbox.glob("*.json"):
                        shutil.copy2(email_file, outbox_dir)
                else:
                    print(f"Warning: Outbox set '{outbox_set}' not found at {source_outbox}")
            
            # Copy drafts data
            if "drafts_set" in initial_data:
                drafts_set = initial_data["drafts_set"]
                source_drafts = Path(f"data/benchmark/initial_environment/initial_drafts/{drafts_set}")
                if source_drafts.exists():
                    for email_file in source_drafts.glob("*.json"):
                        shutil.copy2(email_file, drafts_dir)
                else:
                    print(f"Warning: Drafts set '{drafts_set}' not found at {source_drafts}")
            
            # Add attack emails to inbox (for attack benchmarks)
            if "attack_emails" in initial_data:
                attack_emails = initial_data["attack_emails"]
                for attack_email in attack_emails:
                    # Generate unique filename for attack email
                    attack_email_id = str(uuid.uuid4())[:8]
                    attack_email_file = inbox_dir / f"attack_{attack_email_id}.json"
                    
                    # Write attack email to inbox
                    with open(attack_email_file, 'w', encoding='utf-8') as f:
                        json.dump(attack_email, f, indent=2, ensure_ascii=False)
                    
                    print(f"Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
        else:
            # Fallback to original behavior if no initial_data specified
            original_mailbox = Path("data/interactive_agent/mailbox")
            if original_mailbox.exists():
                for email_file in original_mailbox.glob("*.json"):
                    shutil.copy2(email_file, inbox_dir)
        
        # Initialize memory using unified backend
        # Skip loading initial memory set if no backend enabled
        if self.memory_backend_name == "none":
            print(f"🛡️ No memory backend enabled (memory_backend: none) - skipping initial memory set loading")
            # Still create empty memory file for compatibility
            memory_file = test_dir / "agent_memory.json"
            if not memory_file.exists():
                empty_memory = {"long_term": []}
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump(empty_memory, f, indent=2)
        else:
            # Get memory set from initial_data (unified format)
            memory_set = None
            if initial_data:
                # Check for unified format first
                if "memory" in initial_data and isinstance(initial_data["memory"], dict):
                    memory_set = initial_data["memory"].get("set")
                # Check for simple memory_set format (e.g., "memory_set": "0")
                elif "memory_set" in initial_data:
                    memory_set = initial_data["memory_set"]
                # Fallback: check for old backend-specific formats (for backward compatibility)
                elif "mem0_memory_set" in initial_data:
                    memory_set = initial_data["mem0_memory_set"]
                elif "rag_memory_set" in initial_data:
                    memory_set = initial_data["rag_memory_set"]
            
            if memory_set:
                # Normalize memory_set to extract just the number
                # Handles formats like: "0", "1", "memory_set_3", "mem0_memory_set_1", etc.
                import re
                match = re.search(r'(\d+)$', str(memory_set))
                if match:
                    normalized_memory_set = match.group(1)  # Just the number
                else:
                    normalized_memory_set = str(memory_set)  # Fallback to original
                
                # Use memory backend to initialize with normalized set number
                # This must succeed - if it fails, the test cannot proceed
                self.memory_backend.initialize(normalized_memory_set, test_dir, self.config)
                print(f"✅ Loaded {self.memory_backend_name} memory set: {normalized_memory_set}")
            
            # Still create empty memory file for compatibility (for explicit backend)
            if self.memory_backend_name == "explicit":
                memory_file = test_dir / "agent_memory.json"
                if not memory_file.exists():
                    empty_memory = {"long_term": []}
                    with open(memory_file, 'w', encoding='utf-8') as f:
                        json.dump(empty_memory, f, indent=2)
        
        # Copy initial session data to test environment
        if initial_data and "session_set" in initial_data:
            session_set = initial_data["session_set"]
            source_sessions = Path(f"data/benchmark/initial_environment/initial_sessions/{session_set}.json")
            if source_sessions.exists():
                shutil.copy2(source_sessions, sessions_dir / "session_index.json")
            else:
                print(f"Warning: Session set '{session_set}' not found at {source_sessions}")
                # Create empty session file as fallback
                empty_sessions = {
                    "sessions": []
                }
                with open(sessions_dir / "session_index.json", 'w', encoding='utf-8') as f:
                    json.dump(empty_sessions, f, indent=2)
        
        # Create test-specific config (deep copy to avoid modifying original)
        import copy
        test_config = copy.deepcopy(self.config)
        # Initialize data section if it doesn't exist
        if "data" not in test_config:
            test_config["data"] = {}
        test_config["data"]["mailbox_dir"] = str(inbox_dir)
        test_config["data"]["outbox_dir"] = str(outbox_dir)
        test_config["data"]["drafts_dir"] = str(drafts_dir)
        test_config["data"]["sessions_dir"] = str(sessions_dir)
        
        # Set memory backend info for validators
        if "memory" not in test_config:
            test_config["memory"] = {}
        test_config["memory"]["backend"] = self.memory_backend_name
        
        # Set backend-specific paths and defense types (only if backend is enabled)
        if self.memory_backend_name == "none":
            # No memory backend enabled
            # Explicitly set backend to "none" and disable all backends
            test_config["memory"]["backend"] = "none"
            for backend_name in ["explicit", "mem0", "rag", "context"]:
                if backend_name not in test_config["memory"]:
                    test_config["memory"][backend_name] = {}
                test_config["memory"][backend_name]["enabled"] = False
            # Also disable backend-specific configs
            for backend_config_key in ["explicit_memory", "mem0_memory", "rag_memory", "context_memory"]:
                if backend_config_key not in test_config["memory"]:
                    test_config["memory"][backend_config_key] = {}
                test_config["memory"][backend_config_key]["enabled"] = False
        elif self.memory_backend_name == "rag":
            if "rag_memory" not in test_config["memory"]:
                test_config["memory"]["rag_memory"] = {}
            test_config["memory"]["rag_memory"]["vectorstore_path"] = str(test_dir / "rag_vectorstore")
            test_config["memory"]["rag_memory"]["defense_type"] = self.backend_defense
        elif self.memory_backend_name == "mem0":
            if "mem0_memory" not in test_config["memory"]:
                test_config["memory"]["mem0_memory"] = {}
            test_config["memory"]["mem0_memory"]["vectorstore_path"] = str(test_dir / "mem0_vectorstore")
            test_config["memory"]["mem0_memory"]["defense_type"] = self.backend_defense
        elif self.memory_backend_name == "context":
            if "context_memory" not in test_config["memory"]:
                test_config["memory"]["context_memory"] = {}
            test_config["memory"]["context_memory"]["context_path"] = str(test_dir / "context_memory.json")
            test_config["memory"]["context_memory"]["defense_type"] = self.backend_defense
        elif self.memory_backend_name == "explicit":
            if "explicit_memory" not in test_config["memory"]:
                test_config["memory"]["explicit_memory"] = {}
            test_config["memory"]["explicit_memory"]["memory_file"] = str(test_dir / "agent_memory.json")
            test_config["memory"]["explicit_memory"]["defense_type"] = self.backend_defense
        
        # Include defense type in trace file name if defense is enabled
        if self.memory_backend_name in ("rag", "mem0", "context"):
            if self.backend_defense == "none" or self.backend_defense == "no_defense":
                trace_filename = "trace.jsonl"
            else:
                trace_filename = f"trace_{self.backend_defense}.jsonl"
        else:
            trace_filename = "trace.jsonl"
        test_config["data"]["trace_file"] = str(traces_dir / trace_filename)
        test_config["data"]["memory_file"] = str(test_dir / "agent_memory.json")
        
        print(f"Created test environment: {test_dir}")
        return test_config
    
    def cleanup_test_environment(self, test_dir: Path):
        """
        Clean up test environment directory.
        
        Handles errors gracefully to ensure cleanup doesn't fail the benchmark.
        """
        if test_dir and test_dir.exists():
            try:
                shutil.rmtree(test_dir)
                print(f"✅ Cleaned up test environment: {test_dir.name}")
            except Exception as e:
                # Log error but don't fail - we'll clean up later if needed
                print(f"⚠️  Warning: Failed to clean up test environment {test_dir.name}: {e}")
                logging.warning(f"Failed to clean up test environment {test_dir}: {e}")
        
    def run_test_from_file(self, test_file: Path) -> Dict[str, Any]:
        """Run a single test from a JSON file (adaptive or static based on config)."""
        if self.adaptive_enabled:
            return self._run_adaptive_test_from_file(test_file)
        else:
            return self._run_static_test_from_file(test_file)
    
    def _run_static_test_from_file(self, test_file: Path) -> Dict[str, Any]:
        # Reset execution error tracking at start of each test
        test_execution_errors = []
        test_execution_success = True
        
        """
        Run a single test from a JSON file (static mode).
        
        Test file format:
        {
            "name": "Test Name",
            "description": "Test description",
            "steps": [
                {
                    "user_message": "User query",
                    "expected_behavior": "What should happen",
                    "success_check": "check_function_name"  # Optional
                }
            ]
        }
        """
        print(f"\n{'='*80}")
        print(f"RUNNING TEST: {test_file.name}")
        print(f"{'='*80}")
        
        # Load test definition
        with open(test_file, 'r', encoding='utf-8') as f:
            test_def = json.load(f)
        
        # All test cases are now in normalized format - no normalization needed
        test_name = test_def["name"]
        description = test_def["description"]
        steps = test_def["steps"]
        initial_data = test_def.get("initial_data", {})
        # Use helper function to determine attack_type (handles memory_only tests)
        attack_type = determine_attack_type(test_file, test_def)
        
        # Check if result already exists (result caching)
        if should_skip_test(
            self.memory_backend_name,
            self.unified_defense,
            self.model_name,
            attack_type,
            test_file,
            self.force,
            self.results_base_dir
        ):
            result_path = get_result_path(
                self.memory_backend_name,
                self.unified_defense,
                self.model_name,
                attack_type,
                test_file,
                self.results_base_dir
            )
            print(f"⏭️  Skipping {test_file.name} - result already exists at {result_path}")
            print(f"   Use --force to overwrite")
            
            # Load and return existing result
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                return existing_result
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing result: {e}")
                # Continue to run the test
        
        print(f"Test: {test_name}")
        print(f"Description: {description}")
        print(f"Steps: {len(steps)}")
        if initial_data:
            print(f"Initial Data: {initial_data}")
        
        # Clear agent cache and session store to ensure clean state for each test
        clear_agent_cache()
        from agent.agent_core import _session_store
        _session_store.clear()  # Clear all session memories for deterministic runs
        
        # Create isolated test environment
        test_config = self.create_test_environment(test_name, initial_data)
        test_dir = Path(test_config["data"]["mailbox_dir"]).parent

        # Debug: Print all initial mem0 memories loaded in the vectorstore
        self._debug_print_initial_mem0_memories(test_config)
        
        try:
            # Generate unique session ID for this test - use test name hash for reproducibility
            import hashlib
            test_hash = hashlib.md5(f"{test_file.name}_{test_name}".encode()).hexdigest()[:8]
            session_id = f"bench_{test_hash}"
            print(f"Session ID: {session_id}")
            
            # Clear any existing agent for this session to ensure fresh state
            clear_session_agent(session_id)
            
            # Initialize session trust managers for both defenses
            from agent.agent_core import SessionTrustManager, ProvablePolicyManager
            SessionTrustManager.initialize_session(session_id)  # Initialize for no_untrusted_tools defense
            ProvablePolicyManager.initialize_session(session_id)  # Initialize for provable_policy defense
            
            # Run test steps
            step_results = []
            all_passed = True
            session_history = []  # Track session history
            
            for i, step in enumerate(steps, 1):
                print(f"\n--- Step {i}/{len(steps)} ---")
                
                # Check if this is a special step type
                step_type = step.get("step_type", "user_message")
                
                if step_type == "start_new_session":
                    # Handle session management step
                    print(f"🔄 {step.get('description', 'Starting new session')}")
                    
                    # Save current session history
                    try:
                        current_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                        session_history.append({
                            "session_id": session_id,
                            "step_range": f"1-{i-1}",
                            "traces": current_traces
                        })
                    except Exception as e:
                        print(f"Warning: Could not save session history: {e}")
                    
                    # Start new session - use deterministic ID based on test and step
                    old_session_id = session_id
                    import hashlib
                    session_hash = hashlib.md5(f"{test_file.name}_{test_name}_step{i}".encode()).hexdigest()[:8]
                    session_id = f"bench_{session_hash}"
                    print(f"New session: {session_id} (was {old_session_id})")
                    
                    # Clear the agent cache for the old session to ensure clean state
                    clear_session_agent(old_session_id)
                    
                    # Initialize new session as trusted (for no_untrusted_tools defense)
                    from agent.agent_core import SessionTrustManager
                    SessionTrustManager.initialize_session(session_id)
                    
                    # Initialize new session as Trusted (T) for provable_policy defense
                    from agent.agent_core import ProvablePolicyManager
                    ProvablePolicyManager.initialize_session(session_id)
                    
                    # Log session change event
                    from agent.utils import append_trace_event, get_timestamp, generate_id
                    append_trace_event(
                        test_config["data"]["trace_file"],
                        session_id=session_id,
                        event_type="session_change",
                        payload={
                            "old_session_id": old_session_id,
                            "new_session_id": session_id,
                            "reason": "programmatic_session_management",
                            "step": i
                        }
                    )
                    
                    # Record this step (no success_check, so no "passed" attribute)
                    step_results.append({
                        "step": i,
                        "step_type": "start_new_session",
                        "description": step.get('description', 'Starting new session'),
                        "old_session_id": old_session_id,
                        "new_session_id": session_id,
                        "duration_s": 0.0
                    })
                    continue
                
                if step_type == "insert_attack_email":
                    # Handle attack email insertion step
                    print(f"📧 {step.get('description', 'Inserting attack email into inbox')}")
                    
                    # Get the attack email from the step
                    attack_email = step.get("attack_email")
                    if not attack_email:
                        print(f"⚠️ Warning: Step {i} has step_type 'insert_attack_email' but no 'attack_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": "Missing attack_email field"
                        })
                        continue
                    
                    # Get inbox directory from test config
                    inbox_dir = Path(test_config["data"]["mailbox_dir"])
                    inbox_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename for attack email
                    attack_email_id = str(uuid.uuid4())[:8]
                    attack_email_file = inbox_dir / f"attack_{attack_email_id}.json"
                    
                    # Write attack email to inbox
                    try:
                        with open(attack_email_file, 'w', encoding='utf-8') as f:
                            json.dump(attack_email, f, indent=2, ensure_ascii=False)
                        
                        print(f"✅ Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
                        
                        # Record this step (no success_check, so no "passed" attribute)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "email_subject": attack_email.get('subject', 'No subject'),
                            "email_from": attack_email.get('from', 'Unknown sender'),
                            "duration_s": 0.0
                        })
                    except Exception as e:
                        print(f"❌ Error inserting attack email: {e}")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": str(e)
                        })
                    continue
                
                if step_type == "insert_benign_email":
                    # Handle benign email insertion step
                    print(f"📧 {step.get('description', 'Inserting benign email into inbox')}")
                    
                    # Get the benign email from the step
                    benign_email = step.get("benign_email")
                    if not benign_email:
                        print(f"⚠️ Warning: Step {i} has step_type 'insert_benign_email' but no 'benign_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "duration_s": 0.0,
                            "error": "Missing benign_email field"
                        })
                        continue
                    
                    # Get inbox directory from test config
                    inbox_dir = Path(test_config["data"]["mailbox_dir"])
                    inbox_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename for benign email
                    benign_email_id = str(uuid.uuid4())[:8]
                    benign_email_file = inbox_dir / f"benign_{benign_email_id}.json"
                    
                    # Write benign email to inbox
                    try:
                        with open(benign_email_file, 'w', encoding='utf-8') as f:
                            json.dump(benign_email, f, indent=2, ensure_ascii=False)
                        
                        print(f"✅ Added benign email: {benign_email.get('subject', 'No subject')} from {benign_email.get('from', 'Unknown sender')}")
                        
                        # Record this step (no success_check, so no "passed" attribute)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "email_subject": benign_email.get('subject', 'No subject'),
                            "email_from": benign_email.get('from', 'Unknown sender'),
                            "duration_s": 0.0
                        })
                    except Exception as e:
                        print(f"❌ Error inserting benign email: {e}")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "duration_s": 0.0,
                            "error": str(e)
                        })
                    continue
                
                # Handle regular user message steps
                printer = get_colored_printer()
                print(printer.format_trace_event({
                    "event_type": "user_input",
                    "payload": {"text": step['user_message']},
                    "ts": ""
                }))
                print(f"Expected: {step['expected_behavior']}")
                
                # Invoke agent (pure Python, no HTTP)
                print(f"🔄 Calling invoke_agent...")
                start_time = time.time()
                
                # Initialize is_attack_bench before try block so it's available in except block
                is_attack_bench = "user_goal" in step or "attack_goal" in step
                
                try:
                    result = invoke_agent(
                        text=step['user_message'],
                        session_id=session_id,
                        config=test_config
                    )
                    duration = time.time() - start_time
                    print(f"✅ invoke_agent returned (took {duration:.2f}s)")
                    
                    response = result["response"]
                    
                    # Collect traces for this step
                    try:
                        step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                        # Print only tool_call and tool_result events (to avoid duplicate user_input/agent_response)
                        for trace_event in step_traces:
                            event_type = trace_event.get('event_type', '')
                            # Skip user_input and agent_response as we print them separately
                            if event_type in ['tool_call', 'tool_result']:
                                printer.print_trace_event(trace_event)
                    except Exception as e:
                        print(f"Warning: Could not read traces for step: {e}")
                        step_traces = []
                    
                    # Print agent response with color
                    print(printer.format_trace_event({
                        "event_type": "agent_response",
                        "payload": {"text": response},
                        "ts": ""
                    }))
                    print(f"Duration: {duration:.2f}s")
                    
                    # Print mem0 memories if enabled
                    self._print_mem0_memories(test_config, user_message=step['user_message'])
                    
                    # Run dual evaluation system (user_goal and attack_goal)
                    user_goal_passed = None
                    attack_goal_passed = None
                    partial_attack_score = None
                    
                    if is_attack_bench and ("user_goal" in step or "attack_goal" in step):
                        # Dual evaluation system for attack benchmarks
                        # Store step results and current step number in test_config for cross-step validators
                        test_config["_step_results"] = step_results
                        test_config["_current_step_num"] = i
                        
                        if "user_goal" in step and step["user_goal"]:
                            try:
                                user_goal_check = step["user_goal"]["success_check"]
                                if isinstance(user_goal_check, dict):
                                    validator = create_validator(user_goal_check, test_config)
                                    # Use validate_with_print if it's a composite validator
                                    if isinstance(validator, CompositeValidator):
                                        print("\n🔍 User Goal Validator Results:")
                                        trace_file = test_config.get("data", {}).get("trace_file") if isinstance(test_config.get("data"), dict) else None
                                        user_goal_passed = validator.validate_with_print(response, session_id, step_traces, trace_file=trace_file)
                                    else:
                                        user_goal_passed = validator.validate(response, session_id, step_traces)
                                        # Print single validator result
                                        from agent.colored_trace_printer import print_validator_result
                                        validator_type = user_goal_check.get("type", "validator")
                                        print_validator_result(validator_type, "User goal", user_goal_passed)
                                        # Log to trace
                                        try:
                                            from agent.utils import append_trace_event
                                            trace_file = test_config.get("data", {}).get("trace_file") if isinstance(test_config.get("data"), dict) else None
                                            if trace_file:
                                                append_trace_event(
                                                    trace_file,
                                                    "validator_results",
                                                    session_id,
                                                    {
                                                        "type": validator_type,
                                                        "name": "User goal",
                                                        "passed": user_goal_passed
                                                    }
                                                )
                                        except Exception:
                                            pass
                                else:
                                    # String-based check function (legacy support)
                                    check_function = user_goal_check
                                    if hasattr(self, f"check_{check_function}"):
                                        check_method = getattr(self, f"check_{check_function}")
                                        user_goal_passed = check_method(response, session_id)
                                    else:
                                        print(f"Warning: User goal check function 'check_{check_function}' not found")
                                        user_goal_passed = None
                            except Exception as e:
                                print(f"User Goal Check Error: {e}")
                                user_goal_passed = False
                        
                        if "attack_goal" in step and step["attack_goal"]:
                            try:
                                attack_goal_check = step["attack_goal"]["success_check"]
                                if isinstance(attack_goal_check, dict):
                                    validator = create_validator(attack_goal_check, test_config)
                                    # Use validate_with_print if it's a composite validator
                                    if isinstance(validator, CompositeValidator):
                                        print("\n🔍 Validator Results:")
                                        trace_file = test_config.get("data", {}).get("trace_file") if isinstance(test_config.get("data"), dict) else None
                                        attack_goal_passed = validator.validate_with_print(response, session_id, step_traces, trace_file=trace_file)
                                    else:
                                        attack_goal_passed = validator.validate(response, session_id, step_traces)
                                        # Print single validator result
                                        from agent.colored_trace_printer import print_validator_result
                                        validator_type = attack_goal_check.get("type", "validator")
                                        print_validator_result(validator_type, "Attack goal", attack_goal_passed)
                                        # Log to trace
                                        try:
                                            from agent.utils import append_trace_event
                                            trace_file = test_config.get("data", {}).get("trace_file") if isinstance(test_config.get("data"), dict) else None
                                            if trace_file:
                                                append_trace_event(
                                                    trace_file,
                                                    "validator_results",
                                                    session_id,
                                                    {
                                                        "type": validator_type,
                                                        "name": "Attack goal",
                                                        "passed": attack_goal_passed
                                                    }
                                                )
                                        except Exception:
                                            pass
                                    
                                    # compute partial score if available
                                    try:
                                        partial_attack_score = float(validator.score(response, session_id, step_traces))
                                    except Exception:
                                        partial_attack_score = 1.0 if attack_goal_passed else 0.0
                                else:
                                    # String-based check function (legacy support)
                                    check_function = attack_goal_check
                                    if hasattr(self, f"check_{check_function}"):
                                        check_method = getattr(self, f"check_{check_function}")
                                        attack_goal_passed = check_method(response, session_id)
                                    else:
                                        print(f"Warning: Attack goal check function 'check_{check_function}' not found")
                                        attack_goal_passed = None
                            except Exception as e:
                                print(f"Attack Goal Check Error: {e}")
                                attack_goal_passed = False
                                partial_attack_score = 0.0
                        
                        # Print results for dual evaluation
                        if user_goal_passed is not None:
                            status = "✓ PASSED" if user_goal_passed else "✗ FAILED"
                            print(f"User Goal Result: {status}")
                        if attack_goal_passed is not None:
                            status = "✓ PASSED" if attack_goal_passed else "✗ FAILED"
                            suffix = f" (partial={partial_attack_score:.2f})" if partial_attack_score is not None else ""
                            print(f"Attack Goal Result: {status}{suffix}")
                        
                        # For overall test success, we consider user goal success (attack goal failure is good)
                        step_passed = user_goal_passed if user_goal_passed is not None else True
                        
                    else:
                        # Legacy single evaluation system
                        step_passed = True
                        if "success_check" in step and step["success_check"]:
                            try:
                                # Store step results and current step number in test_config for cross-step validators
                                test_config["_step_results"] = step_results
                                test_config["_current_step_num"] = i
                                
                                # Try new validator system first
                                if isinstance(step["success_check"], dict):
                                    validator = create_validator(step["success_check"], test_config)
                                    # Use validate_with_print if it's a composite validator
                                    if isinstance(validator, CompositeValidator):
                                        print("\n🔍 Validator Results:")
                                        trace_file = test_config.get("data", {}).get("trace_file") if isinstance(test_config.get("data"), dict) else None
                                        step_passed = validator.validate_with_print(response, session_id, step_traces, trace_file=trace_file)
                                    else:
                                        step_passed = validator.validate(response, session_id, step_traces)
                                        # Print single validator result
                                        from agent.colored_trace_printer import print_validator_result
                                        validator_type = step["success_check"].get("type", "validator")
                                        print_validator_result(validator_type, "Success check", step_passed)
                                        # Log to trace
                                        try:
                                            from agent.utils import append_trace_event
                                            trace_file = test_config.get("data", {}).get("trace_file") if isinstance(test_config.get("data"), dict) else None
                                            if trace_file:
                                                append_trace_event(
                                                    trace_file,
                                                    "validator_results",
                                                    session_id,
                                                    {
                                                        "type": validator_type,
                                                        "name": "Success check",
                                                        "passed": step_passed
                                                    }
                                                )
                                        except Exception:
                                            pass
                                else:
                                    # String-based check function (legacy support)
                                    check_function = step["success_check"]
                                    if hasattr(self, f"check_{check_function}"):
                                        check_method = getattr(self, f"check_{check_function}")
                                        step_passed = check_method(response, session_id)
                                        status = "✓ PASSED" if step_passed else "✗ FAILED"
                                        print(f"Check Result: {status}")
                                    else:
                                        print(f"Warning: Check function 'check_{check_function}' not found")
                                        step_passed = None  # Unknown
                            except Exception as e:
                                print(f"Check Error: {e}")
                                step_passed = False
                    
                    if step_passed is False:
                        all_passed = False
                    
                    step_result = {
                        "step": i,
                        "user_message": step['user_message'],
                        "expected_behavior": step['expected_behavior'],
                        "agent_response": response,
                        "duration_s": round(duration, 3)
                    }
                    
                    # Only include "passed" if this step has a success_check
                    has_success_check = (
                        ("success_check" in step and step["success_check"]) or
                        (is_attack_bench and ("user_goal" in step or "attack_goal" in step))
                    )
                    if has_success_check:
                        step_result["passed"] = step_passed
                    
                    # Add dual evaluation results if available
                    if is_attack_bench and ("user_goal" in step or "attack_goal" in step):
                        step_result["user_goal"] = {
                            "description": step.get("user_goal", {}).get("description", ""),
                            "passed": user_goal_passed
                        }
                        step_result["attack_goal"] = {
                            "description": step.get("attack_goal", {}).get("description", ""),
                            "passed": attack_goal_passed,
                            "partial_attack_score": partial_attack_score
                        }
                    else:
                        # Legacy single evaluation
                        step_result["success_check"] = step.get("success_check")
                    
                    step_results.append(step_result)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    all_passed = False
                    
                    # Track execution exception
                    error_msg = f"Step {i} execution exception: {str(e)}"
                    test_execution_errors.append(error_msg)
                    test_execution_success = False
                    
                    # Only include "passed" if this step has a success_check
                    step_result = {
                        "step": i,
                        "user_message": step['user_message'],
                        "expected_behavior": step['expected_behavior'],
                        "error": str(e)
                    }
                    has_success_check = (
                        ("success_check" in step and step.get("success_check")) or
                        (is_attack_bench and ("user_goal" in step or "attack_goal" in step))
                    )
                    if has_success_check:
                        step_result["passed"] = False
                        # Copy success_check/user_goal/attack_goal so the step is counted correctly
                        if "success_check" in step and step.get("success_check"):
                            step_result["success_check"] = step["success_check"]
                        if is_attack_bench:
                            if "user_goal" in step:
                                step_result["user_goal"] = step["user_goal"]
                            if "attack_goal" in step:
                                step_result["attack_goal"] = step["attack_goal"]
                    step_results.append(step_result)
        
            # Add final session to session history (if there are any remaining steps after the last session change)
            try:
                final_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                if final_traces:  # Only add if there are traces for the final session
                    # Find the last session change step to determine the step range
                    last_session_step = 0
                    for step_result in step_results:
                        if step_result.get("step_type") == "start_new_session":
                            last_session_step = step_result["step"]
                    
                    if last_session_step > 0:
                        step_range = f"{last_session_step + 1}-{len(steps)}"
                    else:
                        step_range = f"1-{len(steps)}"
                    
                    session_history.append({
                        "session_id": session_id,
                        "step_range": step_range,
                        "traces": final_traces
                    })
            except Exception as e:
                print(f"Warning: Could not save final session history: {e}")
            
            # Extract all traces from session_history for backward compatibility
            all_traces = []
            for session_data in session_history:
                all_traces.extend(session_data.get("traces", []))
            
            # Compile test result
            test_result = {
                "test_name": test_name,
                "test_file": str(test_file),
                "description": description,
                "session_id": session_id,
                "timestamp": datetime(2025, 11, 3, 12, 0, 0).isoformat(),
                "overall_success": all_passed,
                "steps": step_results,
                "session_history": session_history,
                "test_environment": str(test_dir),
                "execution_success": test_execution_success,  # True if no execution errors, False otherwise
                "execution_errors": test_execution_errors if test_execution_errors else None  # List of error messages
            }
            
            # Add defense info if mem0 is enabled
            if self.mem0_memory_enabled:
                if self.defense_type == "none":
                    test_result["defense_type"] = "no_defense"
                else:
                    test_result["defense_type"] = self.defense_type
            
            # Save result using unified structure
            result_path = get_result_path(
                self.memory_backend_name,
                self.unified_defense,
                self.model_name,
                attack_type,
                test_file,
                self.results_base_dir
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to result
            test_result["memory_backend"] = self.memory_backend_name
            test_result["defense_type"] = self.unified_defense
            test_result["backend_defense"] = self.backend_defense
            test_result["model_name"] = self.model_name
            test_result["attack_type"] = attack_type
            
            # Calculate total_user_steps and total_successful_user_steps
            # Count steps that have a success_check (either direct success_check or user_goal/attack_goal)
            total_user_steps = 0
            total_successful_user_steps = 0
            
            for step_result in step_results:
                # Check if step has a success_check
                has_success_check = (
                    step_result.get("success_check") is not None or
                    step_result.get("user_goal") is not None or
                    step_result.get("attack_goal") is not None
                )
                
                if has_success_check:
                    total_user_steps += 1
                    # Check if step passed (only present if success_check exists)
                    if step_result.get("passed") is True:
                        total_successful_user_steps += 1
            
            test_result["total_user_steps"] = total_user_steps
            test_result["total_successful_user_steps"] = total_successful_user_steps
            
            # Write result file (each process writes to unique directory, so no conflicts)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2, ensure_ascii=False)
            
            result_file = result_path  # Use unified path for reporting
            
            # Calculate step statistics (only count steps with success_check)
            # Use total_user_steps and total_successful_user_steps which already filter correctly
            failed_user_steps = total_user_steps - total_successful_user_steps
            
            print(f"\nTest Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
            print(f"Steps Summary: {total_successful_user_steps}/{total_user_steps} steps passed")
            if failed_user_steps > 0:
                print(f"  - {total_successful_user_steps} passed, {failed_user_steps} failed")
            print(f"Result saved to: {result_file}")
            
            return test_result
            
        finally:
            # Clean up test environment
            self.cleanup_test_environment(test_dir)
    
    def _run_adaptive_test_from_file(self, test_file: Path) -> Dict[str, Any]:
        """
        Run a single test from a JSON file (adaptive mode).
        
        This method implements the adaptive benchmark logic:
        1. Check for cached version first (stronger attack)
        2. If no cache, try static attack
        3. If static fails, try optimization strategies in order
        4. Cache successful attacks and resume from failed step
        """
        print(f"\n{'='*80}")
        print(f"RUNNING ADAPTIVE TEST: {test_file.name}")
        print(f"{'='*80}")
        
        # Load test definition
        with open(test_file, 'r', encoding='utf-8') as f:
            test_def = json.load(f)
        
        test_name = test_def["name"]
        # Use helper function to determine attack_type (handles memory_only tests)
        attack_type = determine_attack_type(test_file, test_def)
        
        # Check if this is an indirect PI attack (only these get adaptive treatment)
        if attack_type != "indirect":
            print(f"📊 Non-indirect attack, running in static mode")
            return self._run_static_test_from_file(test_file)
        
        # Check for cached version first
        cached_test = self._get_cached_test(test_file)
        if cached_test:
            print(f"💾 Using cached test with optimized attack")
            return self._run_static_test_from_file(cached_test)
        
        print(f"🔧 Running adaptive test for indirect attack: {test_name}")
        
        # Initialize state manager with test data
        self.state_manager.initialize(test_def, self.config)
        
        # Run test with adaptive optimization
        result = self._run_test_with_optimization(test_def, test_file)
        
        # Save detailed result using unified structure (match static behavior)
        try:
            # Use helper function to determine attack_type (handles memory_only tests)
            attack_type = determine_attack_type(test_file, test_def)
            result_path = get_result_path(
                self.memory_backend_name if self.memory_backend_name != "none" else "none",
                self.unified_defense,
                self.model_name,
                attack_type,
                test_file,
                self.results_base_dir
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to result (if not already present)
            if "memory_backend" not in result:
                result["memory_backend"] = self.memory_backend_name if self.memory_backend_name != "none" else "none"
            if "defense_type" not in result:
                result["defense_type"] = self.unified_defense
            if "model_name" not in result:
                result["model_name"] = self.model_name
            if "attack_type" not in result:
                result["attack_type"] = attack_type
            
            # Ensure total_user_steps and total_successful_user_steps are calculated
            # (already calculated in _run_test_with_optimization, but ensure they exist)
            if "total_user_steps" not in result or "total_successful_user_steps" not in result:
                step_results = result.get("steps", [])
                total_user_steps = 0
                total_successful_user_steps = 0
                
                for step_result in step_results:
                    # Check if step has a success_check
                    has_success_check = (
                        step_result.get("success_check") is not None or
                        step_result.get("user_goal") is not None or
                        step_result.get("attack_goal") is not None
                    )
                    
                    if has_success_check:
                        total_user_steps += 1
                        # Check if step passed (only present if success_check exists)
                        if step_result.get("passed") is True:
                            total_successful_user_steps += 1
                
                result["total_user_steps"] = total_user_steps
                result["total_successful_user_steps"] = total_successful_user_steps
            
            # Write result file (each process writes to unique directory, so no conflicts)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {result_path}")
        except Exception as e:
            print(f"Warning: Failed to save adaptive test result: {e}")
        
        # Cache successful attacks
        if result.get("optimization_used"):
            self._cache_successful_attack(test_file, result)
        
        return result
    
    def _get_cached_test(self, test_file: Path) -> Optional[Path]:
        """Check if there's a cached version of the test with successful attacks."""
        cache_file = self.cache_dir / test_file.relative_to(self.test_bench_dir)
        if cache_file.exists():
            return cache_file
        return None
    
    def _run_test_with_optimization(self, test_def: Dict[str, Any], test_file: Path) -> Dict[str, Any]:
        """Run test with optimization when attacks fail."""
        test_name = test_def["name"]
        steps = test_def["steps"]
        initial_data = test_def.get("initial_data", {})
        
        # Clear agent cache for clean state
        clear_agent_cache()
        
        # Create test environment
        test_config = self.create_test_environment(test_name, initial_data)
        test_dir = Path(test_config["data"]["mailbox_dir"]).parent
        
        # Debug: Print all initial mem0 memories loaded in the vectorstore
        self._debug_print_initial_mem0_memories(test_config)
        
        try:
            session_id = f"adaptive_{int(time.time()*1000)}"
            print(f"Session ID: {session_id}")
            
            # Initialize session as trusted (for no_untrusted_tools defense)
            from agent.agent_core import SessionTrustManager
            SessionTrustManager.initialize_session(session_id)
            
            # Initialize session as Trusted (T) for provable_policy defense
            from agent.agent_core import ProvablePolicyManager
            ProvablePolicyManager.initialize_session(session_id)
            
            step_results = []
            all_passed = True
            optimization_used = False
            session_history = []  # Track session history
            
            for i, step in enumerate(steps, 1):
                print(f"\n--- Step {i}/{len(steps)} ---")
                
                # Save state before each step
                self.state_manager.save_state_before_step(i)
                
                # Handle special step types (e.g., start_new_session) similar to static mode
                step_type = step.get("step_type", "user_message")
                if step_type == "start_new_session":
                    print(f"🔄 {step.get('description', 'Starting new session')}")
                    
                    # Save current session history before switching
                    try:
                        current_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                        if current_traces:  # Only add if there are traces
                            # Find the last session change step to determine the step range
                            last_session_step = 0
                            for step_result in step_results:
                                if step_result.get("step_type") == "start_new_session":
                                    last_session_step = step_result["step"]
                            
                            if last_session_step > 0:
                                step_range = f"{last_session_step + 1}-{i-1}"
                            else:
                                step_range = f"1-{i-1}"
                            
                            session_history.append({
                                "session_id": session_id,
                                "step_range": step_range,
                                "traces": current_traces
                            })
                    except Exception as e:
                        print(f"Warning: Could not save session history: {e}")
                    
                    # Start a new session and clear old session agent cache
                    old_session_id = session_id
                    session_id = f"adaptive_{int(time.time()*1000)}"
                    print(f"New session: {session_id} (was {old_session_id})")
                    clear_session_agent(old_session_id)
                    
                    # Initialize new session as trusted (for no_untrusted_tools defense)
                    from agent.agent_core import SessionTrustManager
                    SessionTrustManager.initialize_session(session_id)
                    
                    # Initialize new session as Trusted (T) for provable_policy defense
                    from agent.agent_core import ProvablePolicyManager
                    ProvablePolicyManager.initialize_session(session_id)
                    
                    # Log session change event
                    try:
                        from agent.utils import append_trace_event, get_timestamp, generate_id
                        append_trace_event(
                            test_config["data"]["trace_file"],
                            session_id=session_id,
                            event_type="session_change",
                            payload={
                                "old_session_id": old_session_id,
                                "new_session_id": session_id,
                                "reason": "programmatic_session_management",
                                "step": i
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Could not log session change event: {e}")
                    # Record this step (no success_check, so no "passed" attribute)
                    step_results.append({
                        "step": i,
                        "step_type": "start_new_session",
                        "description": step.get('description', 'Starting new session'),
                        "old_session_id": old_session_id,
                        "new_session_id": session_id,
                        "duration_s": 0.0
                    })
                    continue
                
                if step_type == "insert_attack_email":
                    # Handle attack email insertion step
                    print(f"📧 {step.get('description', 'Inserting attack email into inbox')}")
                    
                    # Get the attack email from the step
                    attack_email = step.get("attack_email")
                    if not attack_email:
                        print(f"⚠️ Warning: Step {i} has step_type 'insert_attack_email' but no 'attack_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": "Missing attack_email field"
                        })
                        continue
                    
                    # Get inbox directory from test config
                    inbox_dir = Path(test_config["data"]["mailbox_dir"])
                    inbox_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename for attack email
                    attack_email_id = str(uuid.uuid4())[:8]
                    attack_email_file = inbox_dir / f"attack_{attack_email_id}.json"
                    
                    # Write attack email to inbox
                    try:
                        with open(attack_email_file, 'w', encoding='utf-8') as f:
                            json.dump(attack_email, f, indent=2, ensure_ascii=False)
                        
                        print(f"✅ Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
                        
                        # Update state manager with new email
                        try:
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Add email to state's inbox_emails list
                                current_state.inbox_emails.append(attack_email)
                        except Exception as e:
                            print(f"Warning: Could not update state manager with new email: {e}")
                        
                        # Record this step (no success_check, so no "passed" attribute)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "email_subject": attack_email.get('subject', 'No subject'),
                            "email_from": attack_email.get('from', 'Unknown sender'),
                            "duration_s": 0.0
                        })
                    except Exception as e:
                        print(f"❌ Error inserting attack email: {e}")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": str(e)
                        })
                    continue
                
                if step_type == "insert_benign_email":
                    # Handle benign email insertion step
                    print(f"📧 {step.get('description', 'Inserting benign email into inbox')}")
                    
                    # Get the benign email from the step
                    benign_email = step.get("benign_email")
                    if not benign_email:
                        print(f"⚠️ Warning: Step {i} has step_type 'insert_benign_email' but no 'benign_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "duration_s": 0.0,
                            "error": "Missing benign_email field"
                        })
                        continue
                    
                    # Get inbox directory from test config
                    inbox_dir = Path(test_config["data"]["mailbox_dir"])
                    inbox_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename for benign email
                    benign_email_id = str(uuid.uuid4())[:8]
                    benign_email_file = inbox_dir / f"benign_{benign_email_id}.json"
                    
                    # Write benign email to inbox
                    try:
                        with open(benign_email_file, 'w', encoding='utf-8') as f:
                            json.dump(benign_email, f, indent=2, ensure_ascii=False)
                        
                        print(f"✅ Added benign email: {benign_email.get('subject', 'No subject')} from {benign_email.get('from', 'Unknown sender')}")
                        
                        # Update state manager with new email
                        try:
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Add email to state's inbox_emails list
                                current_state.inbox_emails.append(benign_email)
                        except Exception as e:
                            print(f"Warning: Could not update state manager with new email: {e}")
                        
                        # Record this step (no success_check, so no "passed" attribute)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "email_subject": benign_email.get('subject', 'No subject'),
                            "email_from": benign_email.get('from', 'Unknown sender'),
                            "duration_s": 0.0
                        })
                    except Exception as e:
                        print(f"❌ Error inserting benign email: {e}")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "duration_s": 0.0,
                            "error": str(e)
                        })
                    continue

                # Check if this step has an attack goal
                if "attack_goal" in step and step["attack_goal"]:
                    print(f"🎯 Step {i} has attack goal, testing attack effectiveness")
                    
                    # Run the step and check if attack succeeds
                    step_result = self._run_step_with_attack_check(
                        step, i, session_id, test_config
                    )
                    
                    # Update environment state with traces and session data after running the step
                    # This ensures the state has the latest traces when we save it before the next step
                    try:
                        current_state = self.state_manager.get_current_state()
                        if current_state:
                            # Update traces from the step result or by reading from trace file
                            if "step_traces" in step_result:
                                current_state.update_step_data(i, step_result["step_traces"])
                            else:
                                # Fallback: read traces from file
                                try:
                                    step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                                    current_state.update_step_data(i, step_traces)
                                except Exception:
                                    pass
                            # Update session ID
                            current_state.session_id = session_id
                    except Exception as e:
                        print(f"Warning: Could not update environment state after step {i}: {e}")
                    
                    # Check if attack failed
                    attack_goal_passed = step_result.get("attack_goal", {}).get("passed")
                    if attack_goal_passed is False:
                        print(f"⚠️ Attack failed at step {i}, attempting optimization")
                        
                        # Find which step inserted the attack email
                        original_attack_email, attack_email_step_num = self._find_attack_email_from_steps(test_def, i)
                        
                        # Restore to the state right after the attack email was inserted
                        # We need to restore multiple times if the attack email was inserted more than one step back
                        restore_success = False
                        if attack_email_step_num and attack_email_step_num > 0:
                            # We need to restore to step attack_email_step_num
                            # The state manager saves state before each step, so prev_state before step i
                            # is the state after step i-1. We need to walk back further.
                            # For now, restore to prev_state (which is before step i, after step i-1)
                            # and if attack_email_step_num == i-1, we're good. Otherwise we need more restoration.
                            if attack_email_step_num == i - 1:
                                # Attack email was inserted in the previous step, so prev_state is correct
                                restore_success = self.state_manager.restore_to_prev_state()
                                print(f"🔄 Restored to state after attack email insertion (step {attack_email_step_num})")
                            else:
                                # Attack email was inserted earlier, we need to restore to initial state
                                # and replay up to the attack email step
                                print(f"🔄 Attack email was inserted at step {attack_email_step_num}, restoring to that point")
                                # Restore to prev_state first
                                self.state_manager.restore_to_prev_state()
                                # Then we need to restore the initial state and replay steps up to attack_email_step_num
                                # For now, use the initial state as a fallback
                                initial_state = self.state_manager.get_initial_state()
                                if initial_state:
                                    self.state_manager.curr_state = initial_state.copy()
                                    # TODO: Replay steps 1 to attack_email_step_num if needed
                                    # For simplicity, we'll rely on the environment being restored correctly
                                    restore_success = True
                                    print(f"🔄 Restored to initial state, will replay steps up to {attack_email_step_num}")
                                else:
                                    restore_success = self.state_manager.restore_to_prev_state()
                        else:
                            # Fallback: restore to previous state
                            restore_success = self.state_manager.restore_to_prev_state()
                            print(f"🔄 Restored to previous state for optimization")
                        
                        if not restore_success:
                            print(f"⚠️ Could not restore to previous state, using current state")
                        
                        # Try optimization strategies in order: basic, dspy, openevolve
                        optimization_result = self._optimize_attack(
                            test_def, step, i, session_id, test_config
                        )
                        
                        if optimization_result.success:
                            print(f"✅ Optimization successful with {optimization_result.optimization_strategy}")
                            
                            # Update the test with optimized attack (update the insert_attack_email step)
                            test_def = self._update_test_with_optimized_attack(
                                test_def, optimization_result.optimized_attack_email, attack_email_step_num
                            )
                            
                            # Re-inject the optimized attack email into the environment
                            # This ensures the file system and environment state are in sync
                            self._inject_optimized_attack_email(test_config, optimization_result.optimized_attack_email)
                            
                            # Update environment state
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Find the attack email in inbox_emails and replace it
                                # or add it if not found
                                inbox_emails = current_state.inbox_emails
                                email_replaced = False
                                for idx, inbox_email in enumerate(inbox_emails):
                                    # Match by subject or from address (heuristic)
                                    if (inbox_email.get("subject") == original_attack_email.get("subject") or
                                        inbox_email.get("from") == original_attack_email.get("from")):
                                        inbox_emails[idx] = optimization_result.optimized_attack_email
                                        email_replaced = True
                                        break
                                if not email_replaced:
                                    # If not found, replace the first attack email or append
                                    current_state.inbox_emails = [optimization_result.optimized_attack_email] + inbox_emails[1:]
                            
                            # CRITICAL FIX: Clear session history to match the fresh state used during scoring
                            # During optimization, each variant was scored with a fresh session_id like "adaptive_xxx_iter5_var1"
                            # We need to clear the current session's history to match that fresh state
                            print(f"🔄 Clearing session history for {session_id} to match fresh optimization state")
                            clear_session_agent(session_id)
                            
                            # Re-run the step with optimized attack
                            step_result = self._run_step_with_attack_check(
                                step, i, session_id, test_config, 
                                optimized_attack_email=optimization_result.optimized_attack_email
                            )
                            
                            optimization_used = True
                            step_result["optimization"] = {
                                "strategy": optimization_result.optimization_strategy,
                                "iterations": optimization_result.iterations,
                                "feedback": optimization_result.feedback,
                                "optimized_attack_email": optimization_result.optimized_attack_email
                            }
                        else:
                            print(f"❌ All optimization strategies failed for step {i}")
                            step_result["optimization"] = {
                                "strategy": "all_failed",
                                "iterations": 0,
                                "feedback": optimization_result.feedback
                            }
                    
                    step_results.append(step_result)
                    
                    # Check if step passed
                    if step_result.get("passed") is False:
                        all_passed = False
                else:
                    # Regular step without attack goal
                    step_result = self._run_regular_step(step, i, session_id, test_config)
                    
                    # Update environment state with traces and session data after running the step
                    try:
                        current_state = self.state_manager.get_current_state()
                        if current_state:
                            # Update traces from the step result or by reading from trace file
                            if "step_traces" in step_result:
                                current_state.update_step_data(i, step_result["step_traces"])
                            else:
                                # Fallback: read traces from file
                                try:
                                    step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                                    current_state.update_step_data(i, step_traces)
                                except Exception:
                                    pass
                            # Update session ID
                            current_state.session_id = session_id
                    except Exception as e:
                        print(f"Warning: Could not update environment state after step {i}: {e}")
                    
                    step_results.append(step_result)
                    
                    if step_result.get("passed") is False:
                        all_passed = False
            
            # Add final session to session history (if there are any remaining steps after the last session change)
            try:
                final_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
                if final_traces:  # Only add if there are traces for the final session
                    # Find the last session change step to determine the step range
                    last_session_step = 0
                    for step_result in step_results:
                        if step_result.get("step_type") == "start_new_session":
                            last_session_step = step_result["step"]
                    
                    if last_session_step > 0:
                        step_range = f"{last_session_step + 1}-{len(steps)}"
                    else:
                        step_range = f"1-{len(steps)}"
                    
                    session_history.append({
                        "session_id": session_id,
                        "step_range": step_range,
                        "traces": final_traces
                    })
            except Exception as e:
                print(f"Warning: Could not save final session history: {e}")
            
            # Get description from test_def
            description = test_def.get("description", "")
            
            # Calculate step statistics
            total_steps = len(step_results)
            passed_steps = sum(1 for s in step_results if s.get("passed") is True)
            failed_steps = total_steps - passed_steps
            
            # Calculate total_user_steps and total_successful_user_steps
            # Count steps that have a success_check (either direct success_check or user_goal/attack_goal)
            total_user_steps = 0
            total_successful_user_steps = 0
            
            for step_result in step_results:
                # Check if step has a success_check
                has_success_check = (
                    step_result.get("success_check") is not None or
                    step_result.get("user_goal") is not None or
                    step_result.get("attack_goal") is not None
                )
                
                if has_success_check:
                    total_user_steps += 1
                    # Check if step passed (only present if success_check exists)
                    if step_result.get("passed") is True:
                        total_successful_user_steps += 1
            
            # Compile result
            result = {
                "test_name": test_name,
                "test_file": str(test_file),
                "description": description,
                "session_id": session_id,
                "timestamp": datetime(2025, 11, 3, 12, 0, 0).isoformat(),
                "overall_success": all_passed,
                "steps": step_results,
                "session_history": session_history,
                "test_environment": str(test_dir),
                "optimization_used": optimization_used,
                "total_user_steps": total_user_steps,
                "total_successful_user_steps": total_successful_user_steps,
                "adaptive_benchmark": True,
            }
            
            # Calculate step statistics (only count steps with success_check)
            failed_user_steps = total_user_steps - total_successful_user_steps
            
            print(f"\nAdaptive Test Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
            print(f"Steps Summary: {total_successful_user_steps}/{total_user_steps} steps passed")
            if failed_user_steps > 0:
                print(f"  - {total_successful_user_steps} passed, {failed_user_steps} failed")
            
            return result
            
        finally:
            # Clean up test environment
            self.cleanup_test_environment(test_dir)
    
    def _run_step_with_attack_check(self, step: Dict[str, Any], step_num: int, 
                                  session_id: str, test_config: Dict[str, Any],
                                  optimized_attack_email: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a step and check if attack succeeds."""
        # Clear recent memory files at start of step (so we only check memories added this step)
        test_dir = Path(test_config.get("data", {}).get("mailbox_dir", "")).parent if test_config.get("data", {}).get("mailbox_dir") else None
        if test_dir:
            # Clear RAG recent chunks
            recent_chunks_file = test_dir / "rag_recent_chunks.json"
            if recent_chunks_file.exists():
                try:
                    recent_chunks_file.unlink()
                except Exception:
                    pass  # Silently fail if we can't delete
            
            # Clear mem0 recent memories
            recent_memories_file = test_dir / "mem0_recent_memories.json"
            if recent_memories_file.exists():
                try:
                    recent_memories_file.unlink()
                except Exception:
                    pass  # Silently fail if we can't delete
        
        # If we have an optimized attack email, we need to update the test environment
        if optimized_attack_email:
            self._inject_optimized_attack_email(test_config, optimized_attack_email)
        
        # Run the step (reuse existing logic from static test)
        start_time = time.time()
        try:
            result = invoke_agent(
                text=step['user_message'],
                session_id=session_id,
                config=test_config
            )
            duration = time.time() - start_time
        except Exception as e:
            # API call failed after retries - this is an execution error
            # Re-raise to be caught by caller and tracked
            raise
            
            response = result["response"]
            print(f"Agent Response: {response}")
            print(f"Duration: {duration:.2f}s")
            
            # Print mem0 memories if enabled
            self._print_mem0_memories(test_config, user_message=step['user_message'])
            
            # Get traces for evaluation
            try:
                step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
            except Exception as e:
                print(f"Warning: Could not read traces: {e}")
                step_traces = []
            
            # Evaluate both user and attack goals
            user_goal_passed = None
            attack_goal_passed = None
            partial_attack_score = 0.0  # Initialize default value
            
            if "user_goal" in step and step["user_goal"]:
                try:
                    user_goal_check = step["user_goal"]["success_check"]
                    if isinstance(user_goal_check, dict):
                        validator = create_validator(user_goal_check, test_config)
                        user_goal_passed = validator.validate(response, session_id, step_traces)
                    print(f"User Goal Result: {'✓ PASSED' if user_goal_passed else '✗ FAILED'}")
                except Exception as e:
                    print(f"User Goal Check Error: {e}")
                    user_goal_passed = False
            
            if "attack_goal" in step and step["attack_goal"]:
                try:
                    attack_goal_check = step["attack_goal"]["success_check"]
                    if isinstance(attack_goal_check, dict):
                        validator = create_validator(attack_goal_check, test_config)
                        attack_goal_passed = validator.validate(response, session_id, step_traces)
                        try:
                            partial_attack_score = float(validator.score(response, session_id, step_traces))
                        except Exception:
                            partial_attack_score = 1.0 if attack_goal_passed else 0.0
                    status = '✓ PASSED' if attack_goal_passed else '✗ FAILED'
                    pas = f"{partial_attack_score:.2f}" if isinstance(partial_attack_score, float) else "n/a"
                    print(f"Attack Goal Result: {status} (partial={pas})")
                except Exception as e:
                    print(f"Attack Goal Check Error: {e}")
                    attack_goal_passed = False
                    partial_attack_score = 0.0
            
            # Overall step success (user goal success, attack goal failure is good)
            step_passed = user_goal_passed if user_goal_passed is not None else True
            
            result = {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "agent_response": response,
                "duration_s": round(duration, 3),
                "step_traces": step_traces,  # Include traces in result for state update
                "user_goal": {
                    "description": step.get("user_goal", {}).get("description", ""),
                    "passed": user_goal_passed
                },
                "attack_goal": {
                    "description": step.get("attack_goal", {}).get("description", ""),
                    "passed": attack_goal_passed,
                    "partial_attack_score": partial_attack_score
                }
            }
            
            # Only include "passed" if this step has a success_check (user_goal or attack_goal)
            has_success_check = ("user_goal" in step and step["user_goal"]) or ("attack_goal" in step and step["attack_goal"])
            if has_success_check:
                result["passed"] = step_passed
            
            return result
            
        except Exception as e:
            print(f"Error in step {step_num}: {e}")
            
            # Track execution exception (will be handled by caller)
            # This is a test execution error, not a test failure
            
            result = {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "error": str(e)
            }
            # Only include "passed" if this step has a success_check
            has_success_check = ("user_goal" in step and step.get("user_goal")) or ("attack_goal" in step and step.get("attack_goal"))
            if has_success_check:
                result["passed"] = False
            return result
    
    def _run_regular_step(self, step: Dict[str, Any], step_num: int, 
                         session_id: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a regular step without attack goals."""
        # Clear recent memory files at start of step (so we only check memories added this step)
        test_dir = Path(test_config.get("data", {}).get("mailbox_dir", "")).parent if test_config.get("data", {}).get("mailbox_dir") else None
        if test_dir:
            # Clear RAG recent chunks
            recent_chunks_file = test_dir / "rag_recent_chunks.json"
            if recent_chunks_file.exists():
                try:
                    recent_chunks_file.unlink()
                except Exception:
                    pass  # Silently fail if we can't delete
            
            # Clear mem0 recent memories
            recent_memories_file = test_dir / "mem0_recent_memories.json"
            if recent_memories_file.exists():
                try:
                    recent_memories_file.unlink()
                except Exception:
                    pass  # Silently fail if we can't delete
        
        start_time = time.time()
        try:
            result = invoke_agent(
                text=step['user_message'],
                session_id=session_id,
                config=test_config
            )
            duration = time.time() - start_time
            
            response = result["response"]
            
            # Get traces for regular steps too
            step_traces = []
            try:
                step_traces = read_trace_events(test_config["data"]["trace_file"], session_id)
            except Exception as e:
                # Ignore trace reading errors for regular steps
                pass
            
            # Print mem0 memories if enabled
            self._print_mem0_memories(test_config, user_message=step['user_message'])
            
            result = {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "agent_response": response,
                "duration_s": round(duration, 3),
                "step_traces": step_traces  # Include traces in result for state update
            }
            
            # Only include "passed" if this step has a success_check
            has_success_check = "success_check" in step and step.get("success_check")
            if has_success_check:
                # For regular steps, we need to evaluate the success_check
                try:
                    from benchmark.test_validators import create_validator
                    # Store current user message in test_config for validators that need it (e.g., semantic judge)
                    test_config["_current_user_message"] = step['user_message']
                    validator = create_validator(step["success_check"], test_config)
                    step_passed = validator.validate(response, session_id, step_traces)
                    result["passed"] = step_passed
                except Exception as e:
                    print(f"Warning: Could not evaluate success_check: {e}")
                    result["passed"] = False
            else:
                # No success_check, so no "passed" attribute
                pass
            
            return result
            
        except Exception as e:
            print(f"Error in regular step {step_num}: {e}")
            result = {
                "step": step_num,
                "user_message": step['user_message'],
                "expected_behavior": step['expected_behavior'],
                "error": str(e)
            }
            # Only include "passed" if this step has a success_check
            has_success_check = "success_check" in step and step.get("success_check")
            if has_success_check:
                result["passed"] = False
            return result
    
    def _find_attack_email_from_steps(self, test_def: Dict[str, Any], 
                                      current_step_num: int) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Find the attack email from the most recent insert_attack_email step.
        
        Args:
            test_def: The test definition
            current_step_num: The current step number (1-indexed)
            
        Returns:
            Tuple of (attack_email, step_num) where step_num is the step number where the email was inserted.
            Returns (None, None) if no attack email found.
        """
        steps = test_def.get("steps", [])
        
        # Walk backwards from current_step_num-1 (step before the failed step) to step 1
        # We're looking for the most recent insert_attack_email step
        for i in range(current_step_num - 1, 0, -1):  # current_step_num-1 down to 1 (1-indexed)
            if i <= len(steps):
                step = steps[i - 1]  # Convert to 0-indexed
                if step.get("step_type") == "insert_attack_email":
                    attack_email = step.get("attack_email")
                    if attack_email:
                        return attack_email, i
                        
        # Fallback: check initial_data for backwards compatibility
        if "attack_emails" in test_def.get("initial_data", {}):
            attack_emails = test_def["initial_data"]["attack_emails"]
            if attack_emails:
                return attack_emails[0], 0  # Step 0 means it's from initial_data
        
        return None, None
    
    def _optimize_attack(self, test_def: Dict[str, Any], failed_step: Dict[str, Any], 
                        step_num: int, session_id: str, test_config: Dict[str, Any]) -> Any:
        """Try different optimization strategies to find a successful attack."""
        from benchmark.adaptive_attacks import OptimizationResult
        
        # Get the original attack email by walking backwards through steps
        original_attack_email, attack_email_step_num = self._find_attack_email_from_steps(test_def, step_num)
        
        if not original_attack_email:
            print(f"❌ No attack email found in test definition (searched up to step {step_num})")
            return OptimizationResult(
                success=False,
                optimized_attack_email=None,
                optimization_strategy="none",
                iterations=0,
                feedback=[f"No attack email found when looking backwards from step {step_num}"],
                final_evaluation=None
            )
        
        print(f"📧 Found attack email from step {attack_email_step_num if attack_email_step_num > 0 else 'initial_data'}")
        
        # Check if any optimizers are available
        if not self.optimizers:
            print("❌ No optimizers available (all disabled in config)")
            return OptimizationResult(
                success=False,
                optimized_attack_email=None,
                optimization_strategy="no_optimizers",
                iterations=0,
                feedback=["No optimizers enabled in configuration"],
                final_evaluation=None
            )
        
        # Get current environment state for optimization
        current_state = self.state_manager.get_current_state()
        if not current_state:
            print("❌ No environment state available for optimization")
            return OptimizationResult(
                success=False,
                optimized_attack_email=None,
                optimization_strategy="no_state",
                iterations=0,
                feedback=["No environment state available"],
                final_evaluation=None
            )
        
        # Try each optimization strategy in order
        for strategy_name, optimizer in self.optimizers.items():
            print(f"🔧 Trying optimization strategy: {strategy_name}")
            
            result = optimizer.optimize_attack(
                original_attack_email, failed_step, step_num, session_id, test_config, current_state
            )
            
            if result.success:
                print(f"✅ Optimization successful with {strategy_name}")
                return result
            else:
                print(f"⚠️ Optimization failed with {strategy_name}")
        
        # All strategies failed
        return OptimizationResult(
            success=False,
            optimized_attack_email=None,
            optimization_strategy="all_failed",
            iterations=0,
            feedback=["All optimization strategies failed"],
            final_evaluation=None
        )
    
    def _update_test_with_optimized_attack(self, test_def: Dict[str, Any], 
                                         optimized_attack_email: Dict[str, Any],
                                         attack_email_step_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Update test definition with optimized attack email.
        
        Args:
            test_def: The test definition to update
            optimized_attack_email: The optimized attack email
            attack_email_step_num: The step number where the attack email was inserted (1-indexed).
                                   If None or 0, updates initial_data.attack_emails for backwards compatibility.
        """
        if attack_email_step_num and attack_email_step_num > 0:
            # Update the insert_attack_email step
            steps = test_def.get("steps", [])
            if attack_email_step_num <= len(steps):
                step = steps[attack_email_step_num - 1]  # Convert to 0-indexed
                if step.get("step_type") == "insert_attack_email":
                    step["attack_email"] = optimized_attack_email
                    print(f"📝 Updated attack email in step {attack_email_step_num}")
        else:
            # Fallback: update initial_data for backwards compatibility
            if "initial_data" in test_def and "attack_emails" in test_def["initial_data"]:
                test_def["initial_data"]["attack_emails"][0] = optimized_attack_email
                print(f"📝 Updated attack email in initial_data")
        return test_def
    
    def _inject_optimized_attack_email(self, test_config: Dict[str, Any], 
                                     optimized_attack_email: Dict[str, Any]):
        """Inject optimized attack email into the test environment."""
        # Use the scorer's injection method to actually write the email to disk
        if self.adaptive_enabled and hasattr(self, 'optimizers') and self.optimizers:
            # Get the first available scorer (they all share the same injection logic)
            first_optimizer = next(iter(self.optimizers.values()))
            if hasattr(first_optimizer, 'scorer'):
                first_optimizer.scorer._inject_attack_email(
                    optimized_attack_email, 
                    test_config, 
                    environment_state=None
                )
    
    def _cache_successful_attack(self, test_file: Path, result: Dict[str, Any]):
        """Cache a test with successful attacks for future runs."""
        print(f"💾 Caching successful attack for {test_file.name}")
        
        # Create cache directory structure
        cache_file = self.cache_dir / test_file.relative_to(self.test_bench_dir)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original test and update with optimized attacks
        with open(test_file, 'r', encoding='utf-8') as f:
            cached_test = json.load(f)
        
        # Update the test with optimized attack emails from the result
        steps = result.get("steps", [])
        for i, step in enumerate(steps):
            if "optimization" in step and "optimized_attack_email" in step["optimization"]:
                optimized_email = step["optimization"]["optimized_attack_email"]
                # Find the corresponding insert_attack_email step and update it
                if i < len(cached_test.get("steps", [])):
                    cached_step = cached_test["steps"][i]
                    # Look backwards for the insert_attack_email step
                    for j in range(i, -1, -1):
                        if j < len(cached_test["steps"]) and cached_test["steps"][j].get("step_type") == "insert_attack_email":
                            cached_test["steps"][j]["attack_email"] = optimized_email
                            print(f"💾 Updated attack email in step {j+1} with optimized version")
                            break
        
        # Add optimization metadata
        cached_test["optimization_metadata"] = {
            "optimized": True,
            "optimization_timestamp": datetime(2025, 11, 3, 12, 0, 0).isoformat(),
            "model_name": self.model_name
        }
        
        # Save cached version
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cached_test, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Cached test saved to: {cache_file}")
    
    def discover_test_files(self, test_path: str) -> List[Path]:
        """
        Intelligently discover test files from a path.
        If path is a file, return it. If path is a directory, find all JSON files recursively.
        
        Uses unified test directory structure: data/benchmark/tests/{suite}/
        
        This method delegates to the shared utility function for consistency.
        """
        return discover_test_files_util(
            test_path=test_path,
            test_dir=self.test_bench_dir,
            verbose=True  # TestBench prints warnings
        )
    
    def run_all_tests(self, test_path: str) -> List[Dict[str, Any]]:
        """Run all tests from the specified path (file or directory)."""
        test_files = self.discover_test_files(test_path)
        if not test_files:
            return []
        
        # Determine if this is an attack benchmark by checking if any test has attack_type
        is_attack_bench = False
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_def = json.load(f)
                    if test_def.get("attack_type") in ["direct", "indirect"]:
                        is_attack_bench = True
                        break
            except Exception:
                continue
        
        bench_type = "ATTACK BENCHMARK" if is_attack_bench else "TEST BENCH"
        
        print(f"\n{'#'*80}")
        print(f"EMAIL AGENT {bench_type} (Pure Python)")
        print(f"{'#'*80}")
        print(f"Found {len(test_files)} test files")
        
        # Group tests by attack_type for better organization
        test_groups = {"benign": [], "direct": [], "indirect": [], "memory_only": [], "unknown": []}
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_def = json.load(f)
                    # Use helper function to determine attack_type (handles memory_only tests)
                    attack_type = determine_attack_type(test_file, test_def)
                    test_groups[attack_type].append(test_file)
            except Exception:
                test_groups["unknown"].append(test_file)
        
        # Print test organization
        for attack_type, files in test_groups.items():
            if files:
                print(f"  {attack_type.upper()}: {len(files)} tests")
        
        results = []
        try:
            for test_file in test_files:
                result = self.run_test_from_file(test_file)
                results.append(result)
            
            # Print summary to console (no file generation)
            self.print_summary(results)
        finally:
            # Always clean up test environments, even if there was an error
            self.cleanup_all_test_environments()
        
        return results
    
    def cleanup_all_test_environments(self):
        """
        Clean up all test environment directories.
        
        Ensures all test environments created during this benchmark run are removed.
        """
        if not self.test_dirs:
            return
        
        print(f"\n🧹 Cleaning up {len(self.test_dirs)} test environment(s)...")
        for test_dir in self.test_dirs:
            self.cleanup_test_environment(test_dir)
        self.test_dirs.clear()
        print("✅ All test environments cleaned up.")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Generate a summary report of all test results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["overall_success"])
        failed_tests = total_tests - passed_tests
        
        # Calculate step statistics - only count steps with success_check
        # Use total_user_steps and total_successful_user_steps from each test result
        total_user_steps = 0
        total_successful_user_steps = 0
        
        for r in results:
            # Use the pre-calculated values from each test result
            total_user_steps += r.get("total_user_steps", 0)
            total_successful_user_steps += r.get("total_successful_user_steps", 0)
        
        # Fallback: if total_user_steps is 0, calculate from step results (backward compatibility)
        if total_user_steps == 0:
            all_steps = []
            for r in results:
                if "steps" in r:
                    all_steps.extend(r["steps"])
            
            # Count only steps with success_check
            for step in all_steps:
                has_success_check = (
                    step.get("success_check") is not None or
                    step.get("user_goal") is not None or
                    step.get("attack_goal") is not None
                )
                if has_success_check:
                    total_user_steps += 1
                    if step.get("passed") is True:
                        total_successful_user_steps += 1
        
        total_steps = total_user_steps  # Use total_user_steps for display
        passed_steps = total_successful_user_steps  # Use total_successful_user_steps for display
        failed_steps = total_user_steps - total_successful_user_steps
        
        # Group results by attack_type
        attack_type_groups = {"benign": [], "direct": [], "indirect": []}
        for result in results:
            # Try to get attack_type from test file
            attack_type = "benign"  # default to benign if not found
            try:
                test_file = Path(result.get("test_file", ""))
                if test_file.exists():
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_def = json.load(f)
                    # Use helper function to determine attack_type (handles memory_only tests)
                    attack_type = determine_attack_type(test_file, test_def)
            except Exception:
                pass
            if attack_type in attack_type_groups:
                attack_type_groups[attack_type].append(result)
        
        # Calculate utility and attack success rates for attack benchmarks
        utility_success_rate = "N/A"
        attack_success_rate = "N/A"
        
        # Check if any results have dual evaluation data
        has_dual_evaluation = any(
            any(step.get("user_goal") or step.get("attack_goal") for step in result.get("steps", []))
            for result in results
        )
        
        if has_dual_evaluation:
            # Calculate utility success rate (user goals)
            user_goal_steps = []
            attack_goal_steps = []
            
            for result in results:
                for step in result.get("steps", []):
                    if step.get("user_goal"):
                        user_goal_steps.append(step["user_goal"]["passed"])
                    if step.get("attack_goal"):
                        attack_goal_steps.append(step["attack_goal"]["passed"])
            
            if user_goal_steps:
                user_goal_passed = sum(1 for passed in user_goal_steps if passed is True)
                user_goal_total = sum(1 for passed in user_goal_steps if passed is not None)
                utility_success_rate = f"{(user_goal_passed/user_goal_total*100):.1f}%" if user_goal_total > 0 else "0%"
            
            if attack_goal_steps:
                attack_goal_passed = sum(1 for passed in attack_goal_steps if passed is True)
                attack_goal_total = sum(1 for passed in attack_goal_steps if passed is not None)
                attack_success_rate = f"{(attack_goal_passed/attack_goal_total*100):.1f}%" if attack_goal_total > 0 else "0%"
        
        # Calculate per-category statistics
        category_stats = {}
        for attack_type, category_results in attack_type_groups.items():
            if category_results:
                category_passed = sum(1 for r in category_results if r["overall_success"])
                category_total = len(category_results)
                category_stats[attack_type] = {
                    "total": category_total,
                    "passed": category_passed,
                    "failed": category_total - category_passed,
                    "success_rate": f"{(category_passed/category_total*100):.1f}%" if category_total > 0 else "0%"
                }
        
        test_success_rate = f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
        step_success_rate = f"{(passed_steps/total_steps*100):.1f}%" if total_steps > 0 else "0%"
        
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Tests: {passed_tests}/{total_tests} passed ({test_success_rate})")
        print(f"Steps: {passed_steps}/{total_steps} passed ({step_success_rate})")
        
        # Show attack type breakdown
        if any(category_stats.values()):
            print(f"\nATTACK TYPE BREAKDOWN:")
            for attack_type, stats in category_stats.items():
                if stats["total"] > 0:
                    print(f"  {attack_type.upper()}: {stats['passed']}/{stats['total']} passed ({stats['success_rate']})")
        
        # Show dual evaluation results if available
        if has_dual_evaluation:
            print(f"\nDUAL EVALUATION RESULTS:")
            print(f"Utility Success Rate: {utility_success_rate}")
            print(f"Attack Success Rate: {attack_success_rate}")
            print(f"  (Lower attack success rate is better - indicates resistance to attacks)")
        
        if failed_tests > 0:
            print(f"\nFailed tests:")
            for result in results:
                if not result["overall_success"]:
                    print(f"  - {result['test_name']}")
    


def main():
    """Main entry point for test bench."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Email Agent Test Bench (Pure Python)")
    parser.add_argument("--test", type=str, nargs="+", help="Run specific test file(s) or directory(ies). Can specify multiple paths separated by spaces.")
    parser.add_argument("--suite", type=str, choices=["benign", "direct", "indirect"], help="Shortcut to run an entire suite under data/benchmark/attack_bench/<suite>.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--defense-type", type=str, help="Override defense_type from config (e.g., 'none', 'user_prompt_only', 'no_untrusted_tools', 'disable_memory'). This allows parallel runs without modifying the global config file.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing results")
    
    args = parser.parse_args()
    
    bench = TestBench(args.config, defense_type_override=args.defense_type, force=args.force)
    
    try:
        if args.test or args.suite:
            # Handle multiple test paths
            all_results = []
            paths = []
            if args.test:
                paths.extend(args.test)
            if args.suite:
                paths.append(args.suite)
            for test_path in paths:
                print(f"\n{'='*60}")
                print(f"Processing: {test_path}")
                print(f"{'='*60}")
                
                resolved = test_path
                if Path(test_path).is_file():
                    # Single file
                    test_file = Path(test_path)
                    if not test_file.exists():
                        print(f"Test file not found: {test_file}")
                        continue
                    if not test_file.suffix.lower() == '.json':
                        print(f"File is not a JSON test file: {test_file}")
                        continue
                    result = bench.run_test_from_file(test_file)
                    all_results.append(result)
                else:
                    # Directory or multiple files
                    results = bench.run_all_tests(resolved)
                    all_results.extend(results)
            
            if not all_results:
                print("No tests were executed successfully.")
                return 1
        else:
            # No specific tests specified, run all suites by default
            print("No specific tests specified. Running all test suites...")
            all_results = []
            for suite in ["benign", "direct", "indirect", "memory_only"]:
                print(f"\n{'='*60}")
                print(f"Running suite: {suite}")
                print(f"{'='*60}")
                
                results = bench.run_all_tests(suite)
                all_results.extend(results)
            
            if not all_results:
                print("No tests were executed successfully.")
                return 1
    finally:
        # Clean up any remaining test environments
        bench.cleanup_all_test_environments()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
