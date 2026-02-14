"""
Test Bench for Email Agent
- Each test is a JSON file with a series of user queries
- Agent is loaded fresh for each test with clean session history
- Results are automatically validated and reported
"""

import copy
import json
import time
import shutil
import uuid
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Suppress deprecation warning from litellm's async cleanup code
# This warning occurs during cleanup when there's no running event loop
# It's harmless and doesn't affect functionality
warnings.filterwarnings(
    "ignore",
    message="There is no current event loop",
    category=DeprecationWarning,
    module="litellm"
)

# Ensure src/ is on sys.path for package imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Load environment variables from .env file
load_dotenv()

from agent.agent_core import invoke_agent, clear_session_agent, clear_agent_cache
from agent.utils import load_config, ensure_data_directories, set_global_seeds
from agent.utils import debug_info, debug_debug, debug_print_exception, debug_print_long_content, set_debug_level, DebugLevel, get_debug_level
from benchmark.test_validators import create_validator, CompositeValidator
from agent.utils import get_colored_printer
from benchmark.benchmark_utils import (
    ensure_email_unread,
    get_memory_backend_from_config,
    get_unified_defense_from_config,
    get_result_path,
    should_skip_test,
    determine_attack_type,
    discover_test_files as discover_test_files_util,
    get_log_path,
    TEST_DIR
)

# Import in-memory storage
try:
    from benchmark.in_memory_storage import InMemoryTestEnvironment, InMemoryMailbox, InMemoryVectorstore
except ImportError:
    InMemoryTestEnvironment = None  # Type: ignore
    InMemoryMailbox = None  # Type: ignore
    InMemoryVectorstore = None  # Type: ignore


class TestBench:
    """Test bench for email agent."""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, defense_type_override: Optional[str] = None, force: bool = False, logs_base_dir: Optional[Path] = None):
        """
        Initialize TestBench.
        
        Args:
            config_path: Path to config YAML file (mutually exclusive with config)
            config: Config dictionary (mutually exclusive with config_path)
            defense_type_override: Override defense type from config
            force: Force overwrite existing results
            logs_base_dir: Base directory for logs (defaults to data/benchmark/logs)
        """
        if config_path is not None and config is not None:
            raise ValueError("Cannot specify both config_path and config")
        if config_path is None and config is None:
            config_path = "agent_config.yaml"  # Default to agent_config.yaml
        
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            raise ValueError("Must specify either config_path or config")
        
        # Track execution errors for this benchmark run (simple flag, no complex tracker)
        self.execution_errors = []  # List of error messages
        self.execution_success = True  # Set to False if any execution errors occur
        self._last_test_result = None  # Store last test result for terminal summary
        
        # Set global seed for reproducibility
        global_seed = self.config.get("seed", 42)
        set_global_seeds(global_seed)
        
        # Initialize debug level from environment (set by run_benchmark.py)
        # This ensures debug level is consistent across all processes
        debug_level_env = os.getenv("DEBUG_LEVEL", "INFO").upper()
        debug_level = DebugLevel.DEBUG if debug_level_env == "DEBUG" else DebugLevel.INFO
        set_debug_level(debug_level)
        
        # Read model name strictly from agent_config.yaml
        self.model_name = self.config.get("agent", {}).get("target_model_name")
        if not self.model_name:
            raise ValueError("Config missing agent.target_model_name. Please set it in agent_config.yaml.")
        # No test directories needed - everything is in-memory
        self.test_dirs = []  # Kept for compatibility but will always be empty
        self.force = force  # Force overwrite existing results
        self.logs_base_dir = Path(logs_base_dir) if logs_base_dir is not None else None  # Base directory for logs (None = use default)
        
        # Get memory backend from config
        memory_config = self.config.get("memory", {})
        backend_from_config = memory_config.get("backend")
        
        # Get memory backend from config (set by get_run_config based on CLI args)
        self.memory_backend_name = backend_from_config or "explicit"
        self.memory_backend = None  # No longer used - we use functions directly
        
        # Get unified defense type (set by get_run_config, or use override)
        if defense_type_override is not None:
            self.unified_defense = defense_type_override
        else:
            self.unified_defense = get_unified_defense_from_config(self.config, self.memory_backend_name)
        
        # Use unified defense name directly (no mapping needed)
        self.backend_defense = self.unified_defense
        
        # Derive enabled flags from memory_backend_name (for backward compatibility)
        self.explicit_memory_enabled = (self.memory_backend_name == "explicit")
        self.mem0_memory_enabled = (self.memory_backend_name == "mem0")
        self.rag_memory_enabled = (self.memory_backend_name == "rag")
        self.mem0_print_enabled = self.config.get("memory", {}).get("mem0_memory", {}).get("mem0_print", False)
        self.defense_type = self.backend_defense  # For backward compatibility
        
        # Unified results directory structure
        # Note: results_dir is set by run_benchmark.py from command line args
        # If not set, use hardcoded default (not from config to avoid conflicts)
        benchmark_config = self.config.get("benchmark", {})
        results_base_dir = Path(benchmark_config.get("results_dir", "data/benchmark/results"))
        self.results_base_dir = results_base_dir
        self.results_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Store default results base dir for attack_bench detection
        self.default_results_base_dir = self.results_base_dir
        
        # Unified test directory (hardcoded constant, not configurable)
        self.test_bench_dir = TEST_DIR
        
        # For backward compatibility (used by some old code paths)
        self.memory_type = self.memory_backend_name
        
        print(f"Memory Backend: {self.memory_backend_name.upper()}")
        print(f"Defense: {self.unified_defense} (backend: {self.backend_defense})")
        print(f"Test Directory: {self.test_bench_dir}")
        print(f"Results Directory: {self.results_base_dir}")
        if self.force:
            print(f"Force mode: Will overwrite existing results")
        
        # Initialize state manager for environment state tracking (used in both static and adaptive modes)
        from benchmark.environment_state import StateManager
        self.state_manager = StateManager()
        
        # Cache directory for successful attacks (needed for both static and adaptive modes)
        # Static mode needs this to check for cached optimized attacks
        self.cache_dir = Path("data/benchmark/attack_bench_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if adaptive benchmark is enabled
        self.adaptive_enabled = self.config.get("benchmark", {}).get("enable_adaptive_benchmark", False)
        if self.adaptive_enabled:
            # Import adaptive components only when needed
            from benchmark.adaptive_attacks import DSPyOptimizer, OpenEvolveOptimizer
            from agent.attack_utils import compare_attack_bench_files
            
            # Initialize optimization strategies based on config
            self.optimizers = {}
            
            # Check if DSPy is enabled
            benchmark_config = self.config.get("benchmark", {})
            # Check if DSPy is enabled
            if benchmark_config.get("dspy", {}).get("enabled", False):
                self.optimizers["dspy"] = DSPyOptimizer(self.config)
            
            # Check if OpenEvolve is enabled
            if benchmark_config.get("openevolve", {}).get("enabled", True):
                self.optimizers["openevolve"] = OpenEvolveOptimizer(self.config)
            
            if not self.optimizers:
                print("WARNING: No optimizers enabled in config. Adaptive benchmark will not work.")

            # Set up logging for optimizers when running via TestBench
            # Note: We don't initialize logging here - it will be set up in run_test_from_file()
            # after the test log file is created, so all logs go to the same file
            self.logger = logging.getLogger("adaptive_benchmark")
            self.logger.setLevel(logging.DEBUG)
            # Don't add handlers here - they'll be added in run_test_from_file() with the correct log file
            for optimizer in self.optimizers.values():
                optimizer.set_logger(self.logger)
            
            print("Adaptive benchmark mode ENABLED")
        else:
            print("Static benchmark mode")
    
    def _debug_print_initial_mem0_memories(self, test_config: Dict[str, Any]):
        """Debug print: Print all initial mem0 memories loaded in the vectorstore at the start of a test case."""
        if not self.mem0_memory_enabled:
            return
        
        # Skip debug printing if no memory backend is enabled
        if self.memory_backend_name == "none":
            return
        
        try:
            from agent.backend.mem0_memory import get_mem0_memory_manager
            
            mem0_config = test_config.get("memory", {}).get("mem0_memory", {})
            user_id = mem0_config.get("user_id", "vince")
            
            print(f"\n{'='*80}")
            print(f"DEBUG: Initial mem0 Memories (in-memory)")
            print(f"{'='*80}")
            print(f"User ID: {user_id}")
            
            # Initialize mem0 memory manager (in-memory)
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-4o-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                top_k=mem0_config.get("top_k", 3),
                user_id=user_id,
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
                    debug_debug("Could not get users from mem0")
                    debug_print_exception(e, context="Getting users from mem0 for debug printing", include_traceback=True)
            
            print(f"{'='*80}\n")
        except Exception as e:
            debug_info("Could not print initial mem0 memories")
            debug_print_exception(e, context="Printing initial mem0 memories", include_traceback=True)
    
    def _dump_environment_state(self, test_config: Dict[str, Any], step_num: int, step_type: str = "user_message", 
                                 user_message: Optional[str] = None, agent_response: Optional[str] = None,
                                 mode: str = "static") -> None:
        """
        Dump complete environment state for debugging.
        
        This function logs the complete state of the environment including:
        - Mailbox contents (inbox, outbox, drafts) - full email content
        - RAG vectorstore contents (all chunks) - full text
        - Agent responses (full, not truncated)
        - Tool results (full)
        - Conversation turns being indexed
        - Retrieved chunks during RAG queries
        
        Args:
            test_config: Test configuration dictionary
            step_num: Current step number
            step_type: Type of step (e.g., "user_message", "insert_attack_email")
            user_message: User message for this step (if applicable)
            agent_response: Agent response for this step (if applicable)
            mode: Mode identifier ("static" or "adaptive")
        
        Note: Only prints when DEBUG_LEVEL=DEBUG (use --debug-level DEBUG to enable)
        """
        # Only dump environment state when debug level is DEBUG
        if get_debug_level() != DebugLevel.DEBUG:
            return
        
        try:
            in_memory_env = test_config.get("in_memory_environment")
            if not in_memory_env:
                debug_info(f"[{mode}] Step {step_num}: No in_memory_environment found")
                return
            
            print(f"\n{'='*80}")
            print(f"[{mode.upper()}] ENVIRONMENT STATE DUMP - Step {step_num} ({step_type})")
            print(f"{'='*80}")
            
            # 1. Mailbox state
            print(f"\n--- MAILBOX STATE ---")
            mailbox = in_memory_env.mailbox
            for folder in ["inbox", "outbox", "drafts"]:
                emails = mailbox.get_emails(folder=folder)
                print(f"\n{folder.upper()} ({len(emails)} emails):")
                for idx, email in enumerate(emails, 1):
                    print(f"  [{idx}] From: {email.get('from', 'Unknown')}")
                    print(f"      Subject: {email.get('subject', 'No subject')}")
                    print(f"      Body length: {len(email.get('body_plain', ''))} chars")
                    # Print full body (not truncated)
                    body = email.get('body_plain', '')
                    if body:
                        print(f"      Body (FULL):\n{body}")
                    print()
            
            # 2. RAG Vectorstore state
            print(f"\n--- RAG VECTORSTORE STATE ---")
            memory_backend = test_config.get("memory", {}).get("backend", "none")
            if memory_backend == "rag":
                rag_vectorstore = in_memory_env.rag_vectorstore
                if rag_vectorstore:
                    vectorstore, documents, chunk_counter = rag_vectorstore.load()
                    print(f"Total chunks in vectorstore: {len(documents)}")
                    print(f"Chunk counter: {chunk_counter}")
                    print(f"Vectorstore initialized: {vectorstore is not None}")
                    
                    # Print all chunks (full text, not truncated)
                    if documents:
                        print(f"\nAll RAG chunks (FULL TEXT):")
                        for idx, chunk in enumerate(documents, 1):
                            print(f"\n  Chunk {idx}/{len(documents)} ({len(chunk)} chars):")
                            print(f"  {chunk}")
                    else:
                        print("  (No chunks in vectorstore)")
                else:
                    print("  (No RAG vectorstore found)")
            else:
                print(f"  (RAG backend not active, backend={memory_backend})")
            
            # 3. Agent response (if provided)
            if agent_response:
                print(f"\n--- AGENT RESPONSE (FULL) ---")
                print(f"Length: {len(agent_response)} chars")
                print(f"Response (FULL, NOT TRUNCATED):\n{agent_response}")
            
            # 4. User message (if provided)
            if user_message:
                print(f"\n--- USER MESSAGE (FULL) ---")
                print(f"Length: {len(user_message)} chars")
                print(f"Message (FULL, NOT TRUNCATED):\n{user_message}")
            
            # 5. Conversation turn that would be indexed (if both user_message and agent_response provided)
            if user_message and agent_response:
                print(f"\n--- CONVERSATION TURN (what gets indexed) ---")
                conversation_turn = f"User: {user_message}\nAssistant: {agent_response}"
                print(f"Length: {len(conversation_turn)} chars")
                print(f"Conversation turn (FULL, NOT TRUNCATED):\n{conversation_turn}")
            
            # 6. Trace events for this step
            session_id = test_config.get("_current_session_id", "unknown")
            traces = in_memory_env.get_traces(session_id)
            if traces:
                print(f"\n--- TRACE EVENTS (last 10 for session {session_id}) ---")
                recent_traces = traces[-10:]
                for idx, trace in enumerate(recent_traces, 1):
                    event_type = trace.get("event_type", "unknown")
                    payload = trace.get("payload", {})
                    print(f"  [{idx}] {event_type}:")
                    if event_type == "tool_result":
                        tool_name = payload.get("tool_name", "unknown")
                        result = payload.get("result", "")
                        print(f"      Tool: {tool_name}")
                        print(f"      Result length: {len(str(result))} chars")
                        # Print full result (not truncated)
                        print(f"      Result (FULL):\n{str(result)}")
                    elif event_type == "tool_call":
                        tool_name = payload.get("tool_name", "unknown")
                        args = payload.get("args", {})
                        print(f"      Tool: {tool_name}")
                        print(f"      Args: {args}")
                    elif event_type == "agent_response":
                        response = payload.get("text", "")
                        print(f"      Response length: {len(response)} chars")
                        print(f"      Response (FULL):\n{response}")
            
            print(f"\n{'='*80}\n")
            sys.stdout.flush()
            
        except Exception as e:
            debug_info(f"[{mode}] Error dumping environment state")
            debug_print_exception(e, context=f"Dumping environment state for step {step_num}", include_traceback=True)
    
    def _print_mem0_memories(self, test_config: Dict[str, Any], user_message: Optional[str] = None):
        """Print mem0 memory contents and context if mem0_print is enabled."""
        if not (self.mem0_memory_enabled and self.mem0_print_enabled):
            return
        
        # Skip printing if no memory backend is enabled
        if self.memory_backend_name == "none":
            return
        
        try:
            from agent.backend.mem0_memory import get_mem0_memory_manager
            from agent.agent_core import _build_agent_prompt
            
            mem0_config = test_config.get("memory", {}).get("mem0_memory", {})
            user_id = mem0_config.get("user_id", "vince")
            
            # Initialize mem0 memory manager (in-memory)
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                top_k=mem0_config.get("top_k", 3),
                user_id=user_id,
            )
            
            # Get all memories
            memories = mem0_manager.get_all_memories(user_id=user_id, agent_id=None, limit=1000)
            
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
                    
                    # Check for RAG context (if RAG memory backend is active)
                    rag_context = ""
                    memory_config = test_config.get("memory", {})
                    rag_defense_type = memory_config.get("rag_memory", {}).get("defense_type", "none")
                    # Skip RAG context retrieval if not using RAG backend
                    if self.memory_backend_name == "rag" and rag_defense_type != "disable_memory":
                        try:
                            from agent.backend.rag_memory import get_rag_memory_manager
                            rag_config = memory_config.get("rag_memory", {})
                            vectorstore = rag_config.get("vectorstore")
                            if vectorstore:
                                rag_memory_manager = get_rag_memory_manager(
                                    vectorstore=vectorstore,
                                    embedding_model=rag_config.get("embedding_model", "text-embedding-3-small"),
                                    top_k=rag_config.get("top_k", 3),
                                    chunk_size=rag_config.get("chunk_size", 512),
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
                        except Exception as e:
                            debug_info("Could not retrieve RAG memory context")
                            debug_print_exception(e, context="Retrieving RAG memory context", include_traceback=True)
                    
                    # Get mem0 context
                    mem0_context = mem0_manager.get_context(user_message, user_id=user_id)
                    if mem0_context:
                        formatted_mem0_context = "\n\n# Relevant Mem0 Memory Context\n" + mem0_context + "\n"
                        context_parts.append(formatted_mem0_context)
                        print("\nMem0 Context Added to User Message:")
                        print("-" * 80)
                        print(formatted_mem0_context)
                        print("-" * 80)
                    else:
                        print("\nMem0 Context Added to User Message: (none - no relevant memories found)")
                    
                    # Build final user message (same logic as in agent_core.py)
                    final_user_message = "".join(context_parts) + user_message if context_parts else user_message
                    
                    print("\nFinal User Message Sent to Model:")
                    print("=" * 80)
                    print(final_user_message)
                    print("=" * 80)
                except Exception as e:
                    debug_info("Could not retrieve mem0 context for debug printing")
                    debug_print_exception(e, context="Retrieving mem0 context for debug printing", include_traceback=True)
                    print("\nFinal User Message Sent to Model:")
                    print("=" * 80)
                    print(user_message)
                    print("=" * 80)
            else:
                print("\nFinal User Message Sent to Model: (no user message provided)")
            
            # Print system prompt (memory-related parts)
            try:
                from pathlib import Path
                from agent.agent_core import _build_agent_prompt
                
                memory_config = test_config.get("memory", {})
                
                # Load memory instructions if explicit memory backend is active
                memory_instructions = ""
                explicit_memory_context = ""
                if self.memory_backend_name == "explicit":
                    # Find memory_prompt.txt relative to agent_core.py location
                    agent_core_path = Path(__file__).parent.parent / "agent" / "memory_prompt.txt"
                    if agent_core_path.exists():
                        memory_instructions = agent_core_path.read_text(encoding="utf-8")
                    
                    # Try to load explicit memory context
                    try:
                        from agent.backend.explicit_memory import get_memory_manager
                        memory_manager = get_memory_manager()
                        explicit_memory_context = memory_manager.get_long_term_as_text()
                    except Exception:
                        explicit_memory_context = ""
                
                print("\nSystem Prompt (Memory-Related Sections):")
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
                debug_info("Could not print system prompt")
                debug_print_exception(e, context="Printing system prompt", include_traceback=True)
                
        except Exception as e:
            debug_info("Could not print mem0 memories")
            debug_print_exception(e, context="Printing mem0 memories", include_traceback=True)
    
    def _setup_optimizer_logging(self, log_file_path: Optional[Path] = None):
        """
        Initialize a file logger for optimization runs when using TestBench.
        
        Args:
            log_file_path: Path to log file. If None, uses StreamHandler to write to stdout 
                          (which is redirected to test log file). If provided, logs will go 
                          to the same file as the test logs (consolidated logging).
                          NOTE: No separate log file is created - all logs go to the test log file.
        """
        try:
            self.logger = logging.getLogger("adaptive_benchmark")
            self.logger.setLevel(logging.DEBUG)
            # Clear existing handlers to avoid duplication
            for h in self.logger.handlers[:]:
                self.logger.removeHandler(h)
            
            if log_file_path:
                # Use the same log file as the test (consolidated logging)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # 'a' mode to append to existing file
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                # Also add StreamHandler to write to stdout (which is redirected to log file)
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.info("Optimizer logging initialized - logs will be written to: %s", log_file_path)
            else:
                # No log file provided - don't create a separate one
                # Just use StreamHandler to write to stdout (which will be redirected to test log file)
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.info("Optimizer logging initialized - logs will be written to stdout (redirected to test log file)")
        except Exception as e:
            debug_info("Failed to initialize optimizer logging")
            debug_print_exception(e, context="Initializing optimizer logging", include_traceback=True)

    def create_test_environment(self, test_name: str, initial_data: Dict[str, str] = None) -> Dict[str, str]:
        """
        Create an isolated in-memory test environment with specified initial data.
        
        Uses in-memory storage to eliminate file system contamination issues.
        
        Args:
            test_name: Name of the test (for logging)
            initial_data: Dict with keys 'inbox_set', 'outbox_set', 'drafts_set', 'memory' (unified format), 'session_set'
        
        Returns:
            Config dict with in-memory structures and paths (for backward compatibility).
        """
        # Create in-memory test environment (completely in-memory, no file I/O)
        in_memory_env = InMemoryTestEnvironment(test_name)
        
        # Note: trace_store is passed explicitly through in_memory_environment
        # Tools access it via config["in_memory_environment"].trace_store
        
        # No temp directories needed - everything is in-memory
        # test_dir is only used for legacy config paths (not actually created)
        test_dir = None
        
        # Copy initial data to in-memory mailbox
        if initial_data:
            # Copy inbox data
            if "inbox_set" in initial_data:
                inbox_set = initial_data["inbox_set"]
                source_inbox = Path(f"data/benchmark/initial_environment/initial_inbox/{inbox_set}")
                if source_inbox.exists():
                    in_memory_env.mailbox.copy_from_filesystem(source_inbox, folder="inbox")
                else:
                    print(f"Warning: Inbox set '{inbox_set}' not found at {source_inbox}")
            
            # Copy outbox data (but skip for attack benchmarks to prevent false positives)
            # Attack benchmarks should start with empty outbox
            if "outbox_set" in initial_data:
                # Only copy if this is not an attack benchmark
                # For attack benchmarks, we want empty outbox
                outbox_set = initial_data["outbox_set"]
                source_outbox = Path(f"data/benchmark/initial_environment/initial_outbox/{outbox_set}")
                if source_outbox.exists():
                    # Check if this is an attack benchmark by checking test_name
                    # Attack benchmarks typically have "attack" in the name or path
                    if "attack" not in test_name.lower():
                        in_memory_env.mailbox.copy_from_filesystem(source_outbox, folder="outbox")
                else:
                    print(f"Warning: Outbox set '{outbox_set}' not found at {source_outbox}")
            
            # Copy drafts data
            if "drafts_set" in initial_data:
                drafts_set = initial_data["drafts_set"]
                source_drafts = Path(f"data/benchmark/initial_environment/initial_drafts/{drafts_set}")
                if source_drafts.exists():
                    in_memory_env.mailbox.copy_from_filesystem(source_drafts, folder="drafts")
                else:
                    print(f"Warning: Drafts set '{drafts_set}' not found at {source_drafts}")
            
            # Add attack emails to inbox (for attack benchmarks).
            # Use ensure_email_unread so they are always unread for "read unread emails" steps.
            if "attack_emails" in initial_data:
                attack_emails = initial_data["attack_emails"]
                for attack_email in attack_emails:
                    in_memory_env.mailbox.add_email(ensure_email_unread(attack_email), folder="inbox")
                    print(f"Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
        else:
            # Fallback: copy from original mailbox if it exists
            original_mailbox = Path("data/agent/mailbox")
            if original_mailbox.exists():
                in_memory_env.mailbox.copy_from_filesystem(original_mailbox, folder="inbox")
        
        # Initialize memory using unified backend
        # All tests start with empty memory - memory is built during test execution
        if self.memory_backend_name == "none":
            print(f"No memory backend enabled (memory_backend: none) - skipping memory initialization")
        else:
            # Initialize empty memory backend - memory will be built during test execution
            print(f"Initialized {self.memory_backend_name} memory backend (empty - memory built during test)")
        
        # Copy initial session data to in-memory session store
        if initial_data and "session_set" in initial_data:
            session_set = initial_data["session_set"]
            source_sessions = Path(f"data/benchmark/initial_environment/initial_sessions/{session_set}.json")
            if source_sessions.exists():
                with open(source_sessions, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                # Store in in-memory session store
                in_memory_env.session_store.set_session("initial", session_data)
            else:
                print(f"Warning: Session set '{session_set}' not found at {source_sessions}")
        
        # Create test-specific config (deep copy to avoid modifying original)
        import copy
        test_config = copy.deepcopy(self.config)
        
        # Initialize data section if it doesn't exist
        if "data" not in test_config:
            test_config["data"] = {}
        
        # Note: in_memory_env is passed directly to functions, not stored in config
        # Config should only contain static, user-configurable settings
        # We keep it in config as a fallback for backward compatibility, but prefer direct parameter passing
        test_config["in_memory_environment"] = in_memory_env  # Backward compatibility fallback
        test_config["mailbox"] = in_memory_env.mailbox  # Backward compatibility fallback
        
        # Set memory backend info for validators
        if "memory" not in test_config:
            test_config["memory"] = {}
        test_config["memory"]["backend"] = self.memory_backend_name
        
        # Set backend-specific in-memory vectorstores and configs
        if self.memory_backend_name == "none":
            # No memory backend - nothing to set up
            pass
        elif self.memory_backend_name == "rag":
            if "rag_memory" not in test_config["memory"]:
                test_config["memory"]["rag_memory"] = {}
            test_config["memory"]["rag_memory"]["vectorstore"] = in_memory_env.rag_vectorstore
            test_config["memory"]["rag_memory"]["defense_type"] = self.backend_defense
        elif self.memory_backend_name == "mem0":
            if "mem0_memory" not in test_config["memory"]:
                test_config["memory"]["mem0_memory"] = {}
            test_config["memory"]["mem0_memory"]["defense_type"] = self.backend_defense
            # Create and store a shared mem0 manager instance to persist memories across invocations
            from agent.backend.mem0_memory import get_mem0_memory_manager
            mem0_config = test_config["memory"]["mem0_memory"]
            mem0_manager = get_mem0_memory_manager(
                llm_provider=mem0_config.get("llm_provider", "openai"),
                llm_model=mem0_config.get("llm_model", "gpt-5-mini"),
                llm_temperature=mem0_config.get("llm_temperature", 0.0),
                embedding_provider=mem0_config.get("embedding_provider", "openai"),
                embedding_model=mem0_config.get("embedding_model", "text-embedding-3-small"),
                vector_store_provider=mem0_config.get("vector_store_provider", "faiss"),
                top_k=mem0_config.get("top_k", 10),
                user_id=mem0_config.get("user_id", "vince"),
            )
            # Store in both in_memory_env and config for consistency
            # Retrieval functions check in_memory_env first, indexing uses config
            in_memory_env.mem0_manager = mem0_manager
            test_config["memory"]["mem0_memory"]["manager"] = mem0_manager
        elif self.memory_backend_name == "context":
            if "context_memory" not in test_config["memory"]:
                test_config["memory"]["context_memory"] = {}
            test_config["memory"]["context_memory"]["defense_type"] = self.backend_defense
            # Create and store a shared context memory manager to persist memories across invocations
            from agent.backend.context_memory import get_context_memory_manager
            context_config = test_config["memory"]["context_memory"]
            context_manager = get_context_memory_manager(
                max_context_length=context_config.get("max_context_length"),
                model_name=test_config.get("agent", {}).get("model_name", "gpt-4o-mini"),
            )
            # Store in both in_memory_env and config for consistency
            # Retrieval functions check in_memory_env first, indexing uses config
            in_memory_env.context_manager = context_manager
            test_config["memory"]["context_memory"]["manager"] = context_manager
        elif self.memory_backend_name == "explicit":
            if "explicit_memory" not in test_config["memory"]:
                test_config["memory"]["explicit_memory"] = {}
            test_config["memory"]["explicit_memory"]["defense_type"] = self.backend_defense
        
        print(f"Created in-memory test environment: {test_name} (completely in-memory, no file I/O)")
        return test_config
        
    def _get_results_base_dir_for_test(self, test_file: Path) -> Path:
        """Get the appropriate results base directory based on test file location."""
        test_file_str = str(test_file)
        if "attack_bench" in test_file_str:
            # For attack_bench tests, use attack_results instead of results
            # Replace "results" with "attack_results" in the path
            if "results" in str(self.default_results_base_dir):
                attack_results_dir = Path(str(self.default_results_base_dir).replace("results", "attack_results"))
            else:
                # Fallback: construct attack_results path
                attack_results_dir = self.default_results_base_dir.parent / "attack_results"
            attack_results_dir.mkdir(parents=True, exist_ok=True)
            return attack_results_dir
        return self.default_results_base_dir
    
    def _get_logs_base_dir_for_test(self, test_file: Path) -> Path:
        """Get the appropriate logs base directory based on test file location."""
        test_file_str = str(test_file)
        default_logs_dir = self.logs_base_dir if self.logs_base_dir is not None else Path("data/benchmark/logs")
        
        if "attack_bench" in test_file_str:
            # For attack_bench tests, use attack_logs instead of logs
            # Replace "logs" with "attack_logs" in the path
            logs_dir_str = str(default_logs_dir)
            if "/logs" in logs_dir_str:
                attack_logs_dir = Path(logs_dir_str.replace("/logs", "/attack_logs"))
            elif logs_dir_str.endswith("logs"):
                attack_logs_dir = Path(logs_dir_str[:-4] + "attack_logs")
            else:
                # Fallback: construct attack_logs path
                attack_logs_dir = default_logs_dir.parent / "attack_logs"
            attack_logs_dir.mkdir(parents=True, exist_ok=True)
            return attack_logs_dir
        return default_logs_dir
    
    def run_test_from_file(self, test_file: Path) -> Dict[str, Any]:
        """Run a single test from a JSON file (adaptive or static based on config)."""
        # DEBUG: Print to original stdout before redirecting (helps diagnose if we get here)
        import sys
        original_stdout_before_redirect = sys.stdout
        debug_debug(f"run_test_from_file() called for test_file={test_file}", file=original_stdout_before_redirect)
        
        # Determine results base dir for this test (attack_bench uses attack_results)
        results_base_dir_for_test = self._get_results_base_dir_for_test(test_file)
        
        # Load test definition to determine attack_type for log path
        with open(test_file, 'r', encoding='utf-8') as f:
            test_def = json.load(f)
        
        attack_type = determine_attack_type(test_file, test_def)
        
        # Set up individual log file for this test case (matches results folder structure)
        # Use attack_logs for attack_bench tests, logs for utility tests
        logs_base_dir = self._get_logs_base_dir_for_test(test_file)
        log_path = get_log_path(
            memory_backend=self.memory_backend_name,
            unified_defense=self.unified_defense,
            model_name=self.model_name,
            attack_type=attack_type,
            test_file=test_file,
            logs_base_dir=logs_base_dir
        )
        
        # Add "_adaptive" suffix to log filename when in adaptive benchmark mode
        if self.adaptive_enabled:
            log_path = log_path.with_name(log_path.stem + "_adaptive" + log_path.suffix)
        
        # Create log directory
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Open log file and redirect output
        log_file = open(log_path, 'w', encoding='utf-8')
        sys.stdout = log_file
        sys.stderr = log_file
        
        # Write initial debug message to log file immediately
        debug_info(f"Log file opened: {log_path}")
        debug_info(f"Starting test: {test_file.name}")
        debug_info(f"Memory backend: {self.memory_backend_name}, Defense: {self.unified_defense}")
        
        # Update optimizer logging to use the same log file (consolidated logging)
        if self.adaptive_enabled and hasattr(self, 'logger'):
            self._setup_optimizer_logging(log_file_path=log_path)
            # Update logger for all optimizers
            for optimizer in self.optimizers.values():
                optimizer.set_logger(self.logger)
        
        try:
            if self.adaptive_enabled:
                return self._run_adaptive_test_from_file(test_file)
            else:
                return self._run_static_test_from_file(test_file)
        finally:
            # Restore stdout/stderr and close log file
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
            
            # Print summary to terminal (original stdout) after log file is closed
            # This ensures all detailed errors are in the log file, only summary goes to terminal
            if self._last_test_result:
                try:
                    all_passed = self._last_test_result.get("overall_success", False)
                    total_user_steps = self._last_test_result.get("total_user_steps", 0)
                    total_successful_user_steps = self._last_test_result.get("total_successful_user_steps", 0)
                    execution_errors = self._last_test_result.get("execution_errors", [])
                    
                    # Print brief summary to terminal
                    status = "✓ PASSED" if all_passed else "✗ FAILED"
                    print(f"Test: {test_file.name} - {status} ({total_successful_user_steps}/{total_user_steps} steps)", file=original_stdout, flush=True)
                    if execution_errors:
                        print(f"  WARNING: {len(execution_errors)} execution error(s) - see log file for details", file=original_stdout, flush=True)
                except Exception:
                    # If we can't get the result, just skip summary
                    pass
    
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
        # Check for cached version first (for attack_bench tests with optimized attacks)
        # This ensures static mode uses optimized attacks that were found during adaptive mode
        test_file_str = str(test_file)
        is_attack_bench = "attack_bench" in test_file_str
        cached_test = None
        if is_attack_bench:
            cached_test = self._get_cached_test(test_file)
            if cached_test:
                print(f"💾 Found cached test with optimized attack, loading it for static mode")
                # Load the cached test definition (which has optimized attack emails)
                with open(cached_test, 'r', encoding='utf-8') as f:
                    test_def = json.load(f)
                print(f"💾 Using optimized attack email from cache")
            else:
                # No cache, load original test file
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_def = json.load(f)
        else:
            # Not an attack_bench test, load original test file
            with open(test_file, 'r', encoding='utf-8') as f:
                test_def = json.load(f)
        
        # All test cases are now in normalized format - no normalization needed
        test_name = test_def["name"]
        description = test_def["description"]
        steps = test_def["steps"]
        initial_data = test_def.get("initial_data", {})
        # Use helper function to determine attack_type (handles memory_only tests)
        attack_type = determine_attack_type(test_file, test_def)
        
        # Get results base dir for this test (attack_bench uses attack_results)
        results_base_dir_for_test = self._get_results_base_dir_for_test(test_file)
        
        # Check if result already exists (result caching)
        if should_skip_test(
            self.memory_backend_name,
            self.unified_defense,
            self.model_name,
            attack_type,
            test_file,
            self.force,
            results_base_dir_for_test
        ):
            result_path = get_result_path(
                self.memory_backend_name,
                self.unified_defense,
                self.model_name,
                attack_type,
                test_file,
                results_base_dir_for_test
            )
            print(f"Skipping {test_file.name} - result already exists at {result_path}")
            print(f"   Use --force to overwrite")
            
            # Load and return existing result
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                
                return existing_result
            except Exception as e:
                debug_info("Could not load existing result (will run test anyway)")
                debug_print_exception(e, context="Loading existing test result", include_traceback=True)
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
        debug_debug(f"About to create test environment for test: {test_name}")
        test_config = self.create_test_environment(test_name, initial_data)
        debug_debug("Test environment created successfully")

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
                print(f"\n--- Step {i}/{len(steps)} ---", flush=True)
                sys.stdout.flush()
                
                # Check if this is a special step type
                step_type = step.get("step_type", "user_message")
                
                if step_type == "start_new_session":
                    # Handle session management step
                    print(f"{step.get('description', 'Starting new session')}")
                    
                    # Save current session history
                    try:
                        in_memory_env = test_config.get("in_memory_environment")
                        current_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
                        session_history.append({
                            "session_id": session_id,
                            "step_range": f"1-{i-1}",
                            "traces": current_traces
                        })
                    except Exception as e:
                        debug_info("Could not save session history")
                        debug_print_exception(e, context="Saving session history", include_traceback=True)
                    
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
                    in_memory_env = test_config.get("in_memory_environment")
                    if in_memory_env:
                        in_memory_env.log_event(
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
                        print(f"WARNING: Step {i} has step_type 'insert_attack_email' but no 'attack_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": "Missing attack_email field"
                        })
                        continue
                    
                    # Add to in-memory mailbox (required - filesystem storage no longer supported)
                    # Use a deep copy so that when the agent later marks emails as read (e.g. in
                    # read_all_emails), the step's attack_email dict is not mutated. Otherwise the
                    # cached test would get metadata.read=true and the attack email would not appear
                    # as unread in future runs.
                    try:
                        mailbox = test_config.get("mailbox")
                        if not mailbox:
                            raise ValueError("mailbox is required in test_config")
                        mailbox.add_email(copy.deepcopy(attack_email), folder="inbox")
                        
                        print(f"Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
                        
                        # DEBUG: Dump environment state after inserting attack email
                        test_config["_current_session_id"] = session_id
                        self._dump_environment_state(
                            test_config=test_config,
                            step_num=i,
                            step_type="insert_attack_email",
                            mode="static"
                        )
                        
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
                        debug_info("Error inserting attack email")
                        debug_print_exception(e, context="Inserting attack email", include_traceback=True)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": str(e)
                        })
                    continue
                
                if step_type == "insert_email":
                    # Handle email insertion step (for benign test cases like memory_tools)
                    print(f"📧 {step.get('description', 'Inserting email into inbox')}")
                    
                    # Get the email from the step
                    email = step.get("email")
                    if not email:
                        print(f"WARNING: Step {i} has step_type 'insert_email' but no 'email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_email",
                            "description": step.get('description', 'Inserting email'),
                            "duration_s": 0.0,
                            "error": "Missing email field"
                        })
                        continue
                    
                    # Add email to in-memory mailbox (required - filesystem storage no longer supported)
                    try:
                        mailbox = test_config.get("mailbox")
                        if not mailbox:
                            raise ValueError("mailbox is required in test_config")
                        mailbox.add_email(email, folder="inbox")
                        
                        print(f"Added email: {email.get('subject', 'No subject')} from {email.get('from', 'Unknown sender')}")
                        
                        # Update state manager with new email
                        try:
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Add email to state's inbox_emails list
                                current_state.inbox_emails.append(email)
                        except Exception as e:
                            debug_info("Could not update state manager with new email")
                            debug_print_exception(e, context="Updating state manager with new email", include_traceback=True)
                        
                        # Record this step (no success_check, so no "passed" attribute)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_email",
                            "description": step.get('description', 'Inserting email'),
                            "email_subject": email.get('subject', 'No subject'),
                            "email_from": email.get('from', 'Unknown sender'),
                            "duration_s": 0.0
                        })
                    except Exception as e:
                        debug_info("Error inserting email")
                        debug_print_exception(e, context="Inserting email", include_traceback=True)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_email",
                            "description": step.get('description', 'Inserting email'),
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
                        print(f"WARNING: Step {i} has step_type 'insert_benign_email' but no 'benign_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "duration_s": 0.0,
                            "error": "Missing benign_email field"
                        })
                        continue
                    
                    # Add benign email to in-memory mailbox (required - filesystem storage no longer supported)
                    try:
                        mailbox = test_config.get("mailbox")
                        if not mailbox:
                            raise ValueError("mailbox is required in test_config")
                        mailbox.add_email(benign_email, folder="inbox")
                        with open(benign_email_file, 'w', encoding='utf-8') as f:
                            json.dump(benign_email, f, indent=2, ensure_ascii=False)
                        
                        print(f"Added benign email: {benign_email.get('subject', 'No subject')} from {benign_email.get('from', 'Unknown sender')}")
                        
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
                        debug_info("Error inserting benign email")
                        debug_print_exception(e, context="Inserting benign email", include_traceback=True)
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
                
                print(f"Calling invoke_agent...", flush=True)
                sys.stdout.flush()
                start_time = time.time()
                
                # Initialize is_attack_bench before try block so it's available in except block
                is_attack_bench = "user_goal" in step or "attack_goal" in step
                
                try:
                    in_memory_env = test_config.get("in_memory_environment")
                    result = invoke_agent(
                        text=step['user_message'],
                        session_id=session_id,
                        config=test_config,
                        in_memory_env=in_memory_env
                    )
                    duration = time.time() - start_time
                    print(f"invoke_agent returned (took {duration:.2f}s)", flush=True)
                    sys.stdout.flush()
                    
                    response = result["response"]
                    
                    # Collect traces for this step
                    try:
                        in_memory_env = test_config.get("in_memory_environment")
                        step_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
                        # Print only tool_call and tool_result events (to avoid duplicate user_input/agent_response)
                        for trace_event in step_traces:
                            event_type = trace_event.get('event_type', '')
                            # Skip user_input and agent_response as we print them separately
                            if event_type in ['tool_call', 'tool_result']:
                                printer.print_trace_event(trace_event)
                        sys.stdout.flush()  # Flush after reading traces
                    except Exception as e:
                        debug_info("Could not read traces for step")
                        debug_print_exception(e, context="Reading traces for step", include_traceback=True)
                        step_traces = []
                    
                    # Print agent response with color
                    print(printer.format_trace_event({
                        "event_type": "agent_response",
                        "payload": {"text": response},
                        "ts": ""
                    }), flush=True)
                    print(f"Duration: {duration:.2f}s", flush=True)
                    sys.stdout.flush()  # Ensure all output is flushed before potentially blocking operations
                    
                    # DEBUG: Dump complete environment state after each step
                    test_config["_current_session_id"] = session_id  # Store for debugging
                    self._dump_environment_state(
                        test_config=test_config,
                        step_num=i,
                        step_type="user_message",
                        user_message=step['user_message'],
                        agent_response=response,
                        mode="static"
                    )
                    
                    # Print mem0 memories if enabled
                    self._print_mem0_memories(test_config, user_message=step['user_message'])
                    sys.stdout.flush()  # Flush after mem0 memories in case it blocks
                    
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
                                    in_memory_env = test_config.get("in_memory_environment")
                                    validator = create_validator(user_goal_check, test_config, in_memory_env)
                                    # Use validate_with_print if it's a composite validator
                                    if isinstance(validator, CompositeValidator):
                                        print("\nUser Goal Validator Results:")
                                        user_goal_passed = validator.validate_with_print(response, session_id, step_traces)
                                    else:
                                        user_goal_passed = validator.validate(response, session_id, step_traces)
                                        # Print single validator result
                                        from agent.utils import print_validator_result
                                        validator_type = user_goal_check.get("type", "validator")
                                        print_validator_result(validator_type, "User goal", user_goal_passed)
                                        # Log to trace
                                        try:
                                            in_memory_env = test_config.get("in_memory_environment")
                                            if in_memory_env:
                                                in_memory_env.log_event(
                                                    session_id,
                                                    "validator_results",
                                                    {
                                                        "type": validator_type,
                                                        "name": "User goal",
                                                        "passed": user_goal_passed
                                                    }
                                                )
                                        except Exception as e:
                                            debug_debug("Could not add user goal validator result to trace")
                                            debug_print_exception(e, context="Adding user goal validator result to trace", include_traceback=True)
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
                                debug_info("User Goal Check Error")
                                debug_print_exception(e, context="Executing user goal check", include_traceback=True)
                                user_goal_passed = False
                        
                        if "attack_goal" in step and step["attack_goal"]:
                            try:
                                attack_goal_check = step["attack_goal"]["success_check"]
                                if isinstance(attack_goal_check, dict):
                                    in_memory_env = test_config.get("in_memory_environment")
                                    validator = create_validator(attack_goal_check, test_config, in_memory_env)
                                    # Use validate_with_print if it's a composite validator
                                    if isinstance(validator, CompositeValidator):
                                        print("\nValidator Results:")
                                        attack_goal_passed = validator.validate_with_print(response, session_id, step_traces)
                                    else:
                                        attack_goal_passed = validator.validate(response, session_id, step_traces)
                                        # Print single validator result
                                        from agent.utils import print_validator_result
                                        validator_type = attack_goal_check.get("type", "validator")
                                        print_validator_result(validator_type, "Attack goal", attack_goal_passed)
                                        # Log to trace
                                        try:
                                            in_memory_env = test_config.get("in_memory_environment")
                                            if in_memory_env:
                                                in_memory_env.log_event(
                                                    session_id,
                                                    "validator_results",
                                                    {
                                                        "type": validator_type,
                                                        "name": "Attack goal",
                                                        "passed": attack_goal_passed
                                                    }
                                                )
                                        except Exception as e:
                                            debug_debug("Could not add user goal validator result to trace")
                                            debug_print_exception(e, context="Adding user goal validator result to trace", include_traceback=True)
                                    
                                    # compute partial score if available
                                    try:
                                        partial_attack_score = float(validator.score(response, session_id, step_traces))
                                    except Exception as e:
                                        debug_debug("Could not compute partial attack score, using default")
                                        debug_print_exception(e, context="Computing partial attack score", include_traceback=True)
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
                                debug_info("Attack Goal Check Error")
                                debug_print_exception(e, context="Executing attack goal check", include_traceback=True)
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
                                    in_memory_env = test_config.get("in_memory_environment")
                                    validator = create_validator(step["success_check"], test_config, in_memory_env)
                                    # Use validate_with_print if it's a composite validator
                                    if isinstance(validator, CompositeValidator):
                                        print("\nValidator Results:")
                                        step_passed = validator.validate_with_print(response, session_id, step_traces)
                                    else:
                                        step_passed = validator.validate(response, session_id, step_traces)
                                        # Print single validator result
                                        from agent.utils import print_validator_result
                                        validator_type = step["success_check"].get("type", "validator")
                                        print_validator_result(validator_type, "Success check", step_passed)
                                        # Log to trace
                                        try:
                                            in_memory_env = test_config.get("in_memory_environment")
                                            if in_memory_env:
                                                in_memory_env.log_event(
                                                    session_id,
                                                    "validator_results",
                                                    {
                                                        "type": validator_type,
                                                        "name": "Success check",
                                                        "passed": step_passed
                                                    }
                                                )
                                        except Exception as e:
                                            debug_debug("Could not add user goal validator result to trace")
                                            debug_print_exception(e, context="Adding user goal validator result to trace", include_traceback=True)
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
                                debug_info("Step Check Error")
                                debug_print_exception(e, context="Executing step check", include_traceback=True)
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
                    # Log error to log file (stdout is redirected)
                    debug_info(f"Error in step {i}: {e}")
                    debug_print_exception(e, context=f"Step {i} execution", include_traceback=True)
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
                                step_result["user_goal"] = {
                                    "description": step["user_goal"].get("description", ""),
                                    "passed": False
                                }
                            if "attack_goal" in step:
                                step_result["attack_goal"] = {
                                    "description": step["attack_goal"].get("description", ""),
                                    "passed": False,
                                    "partial_attack_score": 0.0
                                }
                    step_results.append(step_result)
        
            # Add final session to session history (if there are any remaining steps after the last session change)
            try:
                in_memory_env = test_config.get("in_memory_environment")
                final_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
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
                debug_info("Could not save final session history")
                debug_print_exception(e, context="Saving final session history", include_traceback=True)
            
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
                "execution_success": test_execution_success,  # True if no execution errors, False otherwise
                "execution_errors": test_execution_errors if test_execution_errors else None  # List of error messages
            }
            
            # Add defense info if mem0 is enabled
            if self.mem0_memory_enabled:
                test_result["defense_type"] = self.defense_type
            
            # Save result using unified structure
            # Get results base dir for this test (attack_bench uses attack_results)
            results_base_dir_for_test = self._get_results_base_dir_for_test(test_file)
            
            result_path = get_result_path(
                self.memory_backend_name,
                self.unified_defense,
                self.model_name,
                attack_type,
                test_file,
                results_base_dir_for_test
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
            
            # Store result for terminal summary (after log file is closed)
            self._last_test_result = test_result
            
            return test_result
            
        finally:
            # Clean up test environment
            # No explicit cleanup needed - temp directories are cleaned up automatically
            pass
    
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
        
        # Check if this is an attack_bench test (only these get adaptive treatment)
        test_file_str = str(test_file)
        is_attack_bench = "attack_bench" in test_file_str
        if not is_attack_bench:
            print(f"Non-attack-bench test, running in static mode")
            return self._run_static_test_from_file(test_file)
        
        # Check for cached version first
        cached_test = self._get_cached_test(test_file)
        if cached_test:
            print(f"💾 Found cached test with optimized attack, loading it")
            # Load the cached test definition (which has optimized attack emails)
            with open(cached_test, 'r', encoding='utf-8') as f:
                cached_test_def = json.load(f)
            # Use the cached test definition instead of the original
            test_def = cached_test_def
            print(f"💾 Using optimized attack email from cache")
        
        print(f"Running adaptive test for indirect attack: {test_name}")
        
        # Initialize state manager with test data
        self.state_manager.initialize(test_def, self.config)
        
        # Run test with adaptive optimization (even if using cached test, so we can re-optimize if attack fails)
        result = self._run_test_with_optimization(test_def, test_file)
        
        # Save detailed result using unified structure (match static behavior)
        try:
            # Use helper function to determine attack_type (handles memory_only tests)
            attack_type = determine_attack_type(test_file, test_def)
            # Get results base dir for this test (attack_bench uses attack_results)
            results_base_dir_for_test = self._get_results_base_dir_for_test(test_file)
            result_path = get_result_path(
                self.memory_backend_name if self.memory_backend_name != "none" else "none",
                self.unified_defense,
                self.model_name,
                attack_type,
                test_file,
                results_base_dir_for_test
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
        
        # Cache successful attacks - only cache if optimization was used AND attack goal actually succeeded
        if result.get("optimization_used"):
            # Check if any attack goal actually passed
            attack_succeeded = False
            steps = result.get("steps", [])
            for step_result in steps:
                attack_goal = step_result.get("attack_goal", {})
                if isinstance(attack_goal, dict) and attack_goal.get("passed") is True:
                    attack_succeeded = True
                    break
            
            if attack_succeeded:
                try:
                    self._cache_successful_attack(test_file, result)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to cache successful attack: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"💾 Optimization was used but attack did not succeed - not caching")
        
        return result
    
    def _get_cached_test(self, test_file: Path) -> Optional[Path]:
        """Check if there's a cached version of the test with successful attacks."""
        # Handle test files that may be outside test_bench_dir (e.g., attack_bench tests)
        try:
            # Try relative to test_bench_dir first (for tests in data/benchmark/tests/)
            relative_path = test_file.relative_to(self.test_bench_dir)
            cache_file = self.cache_dir / relative_path
        except ValueError:
            # Test file is not under test_bench_dir (e.g., attack_bench tests)
            # Use path relative to data/benchmark/ instead
            try:
                # Make benchmark_base absolute to handle both absolute and relative test_file paths
                benchmark_base = Path("data/benchmark").resolve()
                test_file_resolved = test_file.resolve()
                relative_path = test_file_resolved.relative_to(benchmark_base)
                cache_file = self.cache_dir / relative_path
            except ValueError:
                # Fallback: use string manipulation to extract path after "data/benchmark/"
                test_file_str = str(test_file)
                if "data/benchmark/" in test_file_str:
                    relative_path = Path(test_file_str.split("data/benchmark/")[-1])
                else:
                    # Use filename only as last resort
                    relative_path = Path(test_file.name)
                cache_file = self.cache_dir / relative_path
        
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
                    print(f"{step.get('description', 'Starting new session')}")
                    
                    # Save current session history before switching
                    try:
                        in_memory_env = test_config.get("in_memory_environment")
                        current_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
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
                        debug_info("Could not save session history")
                        debug_print_exception(e, context="Saving session history", include_traceback=True)
                    
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
                        in_memory_env = test_config.get("in_memory_environment")
                        if in_memory_env:
                            in_memory_env.log_event(
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
                        debug_info("Could not log session change event")
                        debug_print_exception(e, context="Logging session change event", include_traceback=True)
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
                        print(f"WARNING: Step {i} has step_type 'insert_attack_email' but no 'attack_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": "Missing attack_email field"
                        })
                        continue
                    
                    # Add to in-memory mailbox (required - filesystem storage no longer supported).
                    # Use ensure_email_unread so the email is always unread for "read unread emails" steps.
                    try:
                        mailbox = test_config.get("mailbox")
                        if not mailbox:
                            raise ValueError("mailbox is required in test_config")
                        attack_email_copy = ensure_email_unread(attack_email)
                        mailbox.add_email(attack_email_copy, folder="inbox")
                        
                        print(f"Added attack email: {attack_email.get('subject', 'No subject')} from {attack_email.get('from', 'Unknown sender')}")
                        
                        # Update state manager with new email (use copy so state matches mailbox)
                        try:
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Add email to state's inbox_emails list
                                current_state.inbox_emails.append(attack_email_copy)
                        except Exception as e:
                            debug_info("Could not update state manager with new email")
                            debug_print_exception(e, context="Updating state manager with new email", include_traceback=True)
                        
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
                        debug_info("Error inserting attack email")
                        debug_print_exception(e, context="Inserting attack email", include_traceback=True)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_attack_email",
                            "description": step.get('description', 'Inserting attack email'),
                            "duration_s": 0.0,
                            "error": str(e)
                        })
                    continue
                
                if step_type == "insert_email":
                    # Handle email insertion step (for benign test cases like memory_tools)
                    print(f"📧 {step.get('description', 'Inserting email into inbox')}")
                    
                    # Get the email from the step
                    email = step.get("email")
                    if not email:
                        print(f"WARNING: Step {i} has step_type 'insert_email' but no 'email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_email",
                            "description": step.get('description', 'Inserting email'),
                            "duration_s": 0.0,
                            "error": "Missing email field"
                        })
                        continue
                    
                    # Add to in-memory mailbox (required - filesystem storage no longer supported)
                    try:
                        mailbox = test_config.get("mailbox")
                        if not mailbox:
                            raise ValueError("mailbox is required in test_config")
                        mailbox.add_email(email, folder="inbox")
                        
                        print(f"Added email: {email.get('subject', 'No subject')} from {email.get('from', 'Unknown sender')}")
                        
                        # Update state manager with new email
                        try:
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Add email to state's inbox_emails list
                                current_state.inbox_emails.append(email)
                        except Exception as e:
                            debug_info("Could not update state manager with new email")
                            debug_print_exception(e, context="Updating state manager with new email", include_traceback=True)
                        
                        # Record this step (no success_check, so no "passed" attribute)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_email",
                            "description": step.get('description', 'Inserting email'),
                            "email_subject": email.get('subject', 'No subject'),
                            "email_from": email.get('from', 'Unknown sender'),
                            "duration_s": 0.0
                        })
                    except Exception as e:
                        debug_info("Error inserting email")
                        debug_print_exception(e, context="Inserting email", include_traceback=True)
                        step_results.append({
                            "step": i,
                            "step_type": "insert_email",
                            "description": step.get('description', 'Inserting email'),
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
                        print(f"WARNING: Step {i} has step_type 'insert_benign_email' but no 'benign_email' field. Skipping.")
                        step_results.append({
                            "step": i,
                            "step_type": "insert_benign_email",
                            "description": step.get('description', 'Inserting benign email'),
                            "duration_s": 0.0,
                            "error": "Missing benign_email field"
                        })
                        continue
                    
                    # Add to in-memory mailbox (required - filesystem storage no longer supported)
                    try:
                        mailbox = test_config.get("mailbox")
                        if not mailbox:
                            raise ValueError("mailbox is required in test_config")
                        mailbox.add_email(benign_email, folder="inbox")
                        
                        print(f"Added benign email: {benign_email.get('subject', 'No subject')} from {benign_email.get('from', 'Unknown sender')}")
                        
                        # Update state manager with new email
                        try:
                            current_state = self.state_manager.get_current_state()
                            if current_state:
                                # Add email to state's inbox_emails list
                                current_state.inbox_emails.append(benign_email)
                        except Exception as e:
                            debug_info("Could not update state manager with new email")
                            debug_print_exception(e, context="Updating state manager with new email", include_traceback=True)
                        
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
                        debug_info("Error inserting benign email")
                        debug_print_exception(e, context="Inserting benign email", include_traceback=True)
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
                    print(f"Step {i} has attack goal, testing attack effectiveness")
                    
                    # Run the step and check if attack succeeds
                    step_result = self._run_step_with_attack_check(
                        step, i, session_id, test_config
                    )
                    
                    # Handle case where step_result is None (shouldn't happen, but be defensive)
                    if step_result is None:
                        print(f"ERROR: Step {i} returned None result, skipping optimization")
                        step_results.append({
                            "step": i,
                            "error": "Step execution returned None",
                            "passed": False
                        })
                        all_passed = False
                        continue
                    
                    # Update environment state with traces and session data after running the step
                    # This ensures the state has the latest traces when we save it before the next step
                    try:
                        current_state = self.state_manager.get_current_state()
                        if current_state:
                            # Update traces from the step result or by reading from trace file
                            if step_result and "step_traces" in step_result:
                                current_state.update_step_data(i, step_result["step_traces"])
                            else:
                                # Fallback: read traces from in-memory environment
                                try:
                                    in_memory_env = test_config.get("in_memory_environment")
                                    step_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
                                    current_state.update_step_data(i, step_traces)
                                except Exception as e:
                                    debug_debug(f"Could not read traces for step {i}, continuing without traces")
                                    debug_print_exception(e, context=f"Reading traces for step {i} in adaptive test", include_traceback=True)
                            # Update session ID
                            current_state.session_id = session_id
                    except Exception as e:
                        debug_info(f"Could not update environment state after step {i}")
                        debug_print_exception(e, context=f"Updating environment state after step {i}", include_traceback=True)
                    
                    # Check if attack failed
                    # Ensure attack_goal dict exists in step_result
                    attack_goal_dict = step_result.get("attack_goal", {})
                    attack_goal_passed = attack_goal_dict.get("passed") if isinstance(attack_goal_dict, dict) else None
                    if attack_goal_passed is False:
                        print(f"WARNING: Attack failed at step {i}, attempting optimization")
                        
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
                                print(f"Restored to state after attack email insertion (step {attack_email_step_num})")
                            else:
                                # Attack email was inserted earlier, we need to restore to initial state
                                # and replay up to the attack email step
                                print(f"Attack email was inserted at step {attack_email_step_num}, restoring to that point")
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
                                    print(f"Restored to initial state, will replay steps up to {attack_email_step_num}")
                                else:
                                    restore_success = self.state_manager.restore_to_prev_state()
                        else:
                            # Fallback: restore to previous state
                            restore_success = self.state_manager.restore_to_prev_state()
                            print(f"Restored to previous state for optimization")
                        
                        if not restore_success:
                            print(f"WARNING: Could not restore to previous state, using current state")
                        
                        # Try optimization strategies in order: basic, dspy, openevolve
                        optimization_result = self._optimize_attack(
                            test_def, step, i, session_id, test_config
                        )
                        
                        if optimization_result.success:
                            print(f"Optimization successful with {optimization_result.optimization_strategy}")
                            
                            # Update the test with optimized attack (update the insert_attack_email step)
                            test_def = self._update_test_with_optimized_attack(
                                test_def, optimization_result.optimized_attack_email, attack_email_step_num
                            )
                            
                            # Run verification on a completely fresh environment: create new env and replay
                            # steps 0..through_idx (0-based), then run the attack step. This avoids any read/unread
                            # or state leakage from the previous run (same as scorer does for each candidate).
                            through_idx = max(0, i - 2)  # 0-based: replay through step i-2 so step i-1 is last replayed
                            print(f"🔄 Running verification on fresh environment (replay steps 1-{through_idx + 1}, then step {i})...")
                            test_config, session_id = self._run_steps_on_fresh_environment(
                                test_def, through_idx, optimization_result.optimized_attack_email
                            )
                            print(f"✅ Fresh replay complete; running attack step {i}")
                            
                            # Run the attack step (no inject needed; env already has optimized email)
                            step_result = self._run_step_with_attack_check(
                                step, i, session_id, test_config,
                                optimized_attack_email=None
                            )
                            
                            optimization_used = True
                            step_result["optimization"] = {
                                "strategy": optimization_result.optimization_strategy,
                                "iterations": optimization_result.iterations,
                                "feedback": optimization_result.feedback,
                                "optimized_attack_email": optimization_result.optimized_attack_email
                            }
                        else:
                            print(f"ERROR: All optimization strategies failed for step {i}")
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
                                # Fallback: read traces from in-memory environment
                                try:
                                    in_memory_env = test_config.get("in_memory_environment")
                                    step_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
                                    current_state.update_step_data(i, step_traces)
                                except Exception as e:
                                    debug_debug(f"Could not read traces for step {i}, continuing without traces")
                                    debug_print_exception(e, context=f"Reading traces for step {i} in adaptive test", include_traceback=True)
                            # Update session ID
                            current_state.session_id = session_id
                    except Exception as e:
                        debug_info(f"Could not update environment state after step {i}")
                        debug_print_exception(e, context=f"Updating environment state after step {i}", include_traceback=True)
                    
                    step_results.append(step_result)
                    
                    if step_result.get("passed") is False:
                        all_passed = False
            
            # Add final session to session history (if there are any remaining steps after the last session change)
            try:
                in_memory_env = test_config.get("in_memory_environment")
                final_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
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
                debug_info("Could not save final session history")
                debug_print_exception(e, context="Saving final session history", include_traceback=True)
            
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
            # No explicit cleanup needed - temp directories are cleaned up automatically
            pass
    
    def _run_step_with_attack_check(self, step: Dict[str, Any], step_num: int, 
                                  session_id: str, test_config: Dict[str, Any],
                                  optimized_attack_email: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a step and check if attack succeeds."""
        # No file cleanup needed - everything is in-memory
        # Recent memory tracking is handled in-memory by memory backends
        
        # If we have an optimized attack email, we need to update the test environment
        if optimized_attack_email:
            self._inject_optimized_attack_email(test_config, optimized_attack_email)
        
        # Run the step (reuse existing logic from static test)
        start_time = time.time()
        try:
            in_memory_env = test_config.get("in_memory_environment")
            agent_result = invoke_agent(
                text=step['user_message'],
                session_id=session_id,
                config=test_config,
                in_memory_env=in_memory_env
            )
            duration = time.time() - start_time
            
            response = agent_result["response"]
            print(f"Agent Response: {response}")
            print(f"Duration: {duration:.2f}s")
            
            # Print mem0 memories if enabled
            self._print_mem0_memories(test_config, user_message=step['user_message'])
            
            # Get traces for evaluation
            try:
                in_memory_env = test_config.get("in_memory_environment")
                step_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
            except Exception as e:
                debug_info("Could not read traces")
                debug_print_exception(e, context="Reading traces", include_traceback=True)
                step_traces = []
            
            # Evaluate both user and attack goals
            user_goal_passed = None
            attack_goal_passed = None
            partial_attack_score = 0.0  # Initialize default value
            
            if "user_goal" in step and step["user_goal"]:
                try:
                    user_goal_check = step["user_goal"]["success_check"]
                    if isinstance(user_goal_check, dict):
                        in_memory_env = test_config.get("in_memory_environment")
                        validator = create_validator(user_goal_check, test_config, in_memory_env)
                        user_goal_passed = validator.validate(response, session_id, step_traces)
                    print(f"User Goal Result: {'✓ PASSED' if user_goal_passed else '✗ FAILED'}")
                except Exception as e:
                    print(f"User Goal Check Error: {e}")
                    user_goal_passed = False
            
            if "attack_goal" in step and step["attack_goal"]:
                try:
                    attack_goal_check = step["attack_goal"]["success_check"]
                    if isinstance(attack_goal_check, dict):
                        in_memory_env = test_config.get("in_memory_environment")
                        validator = create_validator(attack_goal_check, test_config, in_memory_env)
                        attack_goal_passed = validator.validate(response, session_id, step_traces)
                        try:
                            partial_attack_score = float(validator.score(response, session_id, step_traces))
                        except Exception as e:
                            debug_debug("Could not compute partial attack score, using default")
                            debug_print_exception(e, context="Computing partial attack score", include_traceback=True)
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
                "user_message": step.get('user_message', ''),
                "expected_behavior": step.get('expected_behavior', ''),
                "error": str(e),
                "step_traces": []  # Empty traces on error
            }
            
            # Include user_goal and attack_goal structures even on error
            # This ensures the result structure is consistent
            if "user_goal" in step and step.get("user_goal"):
                result["user_goal"] = {
                    "description": step["user_goal"].get("description", ""),
                    "passed": False
                }
            
            if "attack_goal" in step and step.get("attack_goal"):
                result["attack_goal"] = {
                    "description": step["attack_goal"].get("description", ""),
                    "passed": False,
                    "partial_attack_score": 0.0
                }
            
            # Only include "passed" if this step has a success_check
            has_success_check = ("user_goal" in step and step.get("user_goal")) or ("attack_goal" in step and step.get("attack_goal"))
            if has_success_check:
                result["passed"] = False
            
            return result
    
    def _run_regular_step(self, step: Dict[str, Any], step_num: int, 
                         session_id: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a regular step without attack goals."""
        # No file cleanup needed - everything is in-memory
        # Recent memory tracking is handled in-memory by memory backends
        
        start_time = time.time()
        try:
            in_memory_env = test_config.get("in_memory_environment")
            result = invoke_agent(
                text=step['user_message'],
                session_id=session_id,
                config=test_config,
                in_memory_env=in_memory_env
            )
            duration = time.time() - start_time
            
            response = result["response"]
            
            # Get traces for regular steps too
            step_traces = []
            try:
                in_memory_env = test_config.get("in_memory_environment")
                step_traces = in_memory_env.get_traces(session_id) if in_memory_env else []
            except Exception as e:
                # Ignore trace reading errors for regular steps (non-critical)
                debug_debug("Could not read traces for regular step (non-critical)")
                debug_print_exception(e, context="Reading traces for regular step", include_traceback=True)
            
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
                    in_memory_env = test_config.get("in_memory_environment")
                    validator = create_validator(step["success_check"], test_config, in_memory_env)
                    step_passed = validator.validate(response, session_id, step_traces)
                    result["passed"] = step_passed
                except Exception as e:
                    debug_info("Could not evaluate success_check")
                    debug_print_exception(e, context="Evaluating success_check", include_traceback=True)
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
    
    def _run_steps_on_fresh_environment(
        self,
        test_def: Dict[str, Any],
        through_step_index: int,
        optimized_attack_email: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Create a completely fresh test environment and run steps 0 through through_step_index (0-based, inclusive).
        Uses optimized_attack_email at the insert_attack_email step. Ensures every evaluation starts from a clean
        mailbox and RAG (no read/unread or state leakage from previous runs).
        
        Returns:
            (test_config, session_id) after running the steps.
        """
        test_name = test_def.get("name", "adaptive_test")
        initial_data = test_def.get("initial_data", {})
        # Do not pre-add attack_emails; we add the optimized one at the insert_attack_email step.
        initial_data_no_attack = {k: v for k, v in initial_data.items() if k != "attack_emails"}
        
        fresh_test_config = self.create_test_environment(test_name, initial_data_no_attack or None)
        session_id = f"adaptive_{int(time.time()*1000)}"
        from agent.agent_core import SessionTrustManager, ProvablePolicyManager
        SessionTrustManager.initialize_session(session_id)
        ProvablePolicyManager.initialize_session(session_id)
        
        steps = test_def.get("steps", [])
        in_memory_env = fresh_test_config.get("in_memory_environment")
        
        for step_idx in range(through_step_index + 1):
            if step_idx >= len(steps):
                break
            step = steps[step_idx]
            step_type = step.get("step_type", "user_message")
            
            if step_type == "start_new_session":
                old_session_id = session_id
                session_id = f"adaptive_{int(time.time()*1000)}"
                clear_session_agent(old_session_id)
                SessionTrustManager.initialize_session(session_id)
                ProvablePolicyManager.initialize_session(session_id)
            elif step_type == "insert_attack_email":
                fresh_test_config["mailbox"].add_email(ensure_email_unread(optimized_attack_email), folder="inbox")
            elif step_type == "insert_email":
                email = step.get("email")
                if email:
                    fresh_test_config["mailbox"].add_email(email, folder="inbox")
            elif step_type == "insert_benign_email":
                benign_email = step.get("benign_email")
                if benign_email:
                    fresh_test_config["mailbox"].add_email(benign_email, folder="inbox")
            elif step_type == "user_message":
                user_msg = step.get("user_message", "")
                if user_msg:
                    invoke_agent(user_msg, session_id, fresh_test_config, in_memory_env)
        
        return fresh_test_config, session_id
    
    def _optimize_attack(self, test_def: Dict[str, Any], failed_step: Dict[str, Any], 
                        step_num: int, session_id: str, test_config: Dict[str, Any]) -> Any:
        """Try different optimization strategies to find a successful attack."""
        from benchmark.adaptive_attacks import OptimizationResult
        
        # Get the original attack email by walking backwards through steps
        original_attack_email, attack_email_step_num = self._find_attack_email_from_steps(test_def, step_num)
        
        if not original_attack_email:
            print(f"ERROR: No attack email found in test definition (searched up to step {step_num})")
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
            print("ERROR: No optimizers available (all disabled in config)")
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
            print("ERROR: No environment state available for optimization")
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
            print(f"Trying optimization strategy: {strategy_name}")
            
            result = optimizer.optimize_attack(
                original_attack_email, failed_step, step_num, session_id, test_config, current_state,
                test_def=test_def, attack_email_step_num=attack_email_step_num
            )
            
            if result.success:
                print(f"Optimization successful with {strategy_name}")
                return result
            else:
                print(f"WARNING: Optimization failed with {strategy_name}")
        
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
                    print(f"Updated attack email in step {attack_email_step_num}")
        else:
            # Fallback: update initial_data for backwards compatibility
            if "initial_data" in test_def and "attack_emails" in test_def["initial_data"]:
                test_def["initial_data"]["attack_emails"][0] = optimized_attack_email
                print(f"Updated attack email in initial_data")
        return test_def
    
    def _inject_optimized_attack_email(self, test_config: Dict[str, Any], 
                                     optimized_attack_email: Dict[str, Any]):
        """Inject optimized attack email into the test environment (always unread)."""
        inject_email = ensure_email_unread(optimized_attack_email)
        
        # Inject the optimized attack email directly into the mailbox
        mailbox = test_config.get("mailbox")
        if not mailbox:
            raise ValueError("mailbox is required in test_config for injecting optimized attack email")
        
        # Find and replace the existing attack email in the inbox
        # The attack email is typically the first email inserted, so we'll replace the first one
        # or find one that matches the sender
        attack_email_found = False
        optimized_from = optimized_attack_email.get("from")
        
        # Look for the attack email by matching the sender
        # Attack emails are usually inserted first, so we check the first few emails
        for i, email in enumerate(mailbox.inbox):
            # Replace if it matches the sender (most reliable identifier)
            if email.get("from") == optimized_from:
                mailbox.inbox[i] = inject_email
                attack_email_found = True
                print(f"💾 Replaced attack email in inbox (index {i}) with optimized version")
                break
        
        # If we didn't find a matching email, replace the first one (assuming it's the attack email)
        # or add it if inbox is empty
        if not attack_email_found:
            if len(mailbox.inbox) > 0:
                mailbox.inbox[0] = inject_email
                print(f"💾 Replaced first email in inbox with optimized attack email")
            else:
                mailbox.add_email(inject_email, folder="inbox")
                print(f"💾 Added optimized attack email to inbox")
        
        # CRITICAL: Clear the RAG vectorstore when injecting optimized email
        # The RAG vectorstore was built with the old attack email from previous steps (1-3)
        # We need to clear it so it gets rebuilt with the new optimized email when step 4 runs
        # This ensures the agent retrieves the optimized email from RAG, not the old one
        memory_backend = test_config.get("memory", {}).get("backend", "none")
        if memory_backend == "rag":
            rag_memory_config = test_config.get("memory", {}).get("rag_memory", {})
            vectorstore = rag_memory_config.get("vectorstore")
            if vectorstore:
                try:
                    print(f"🧹 Clearing in-memory RAG vectorstore to rebuild with optimized attack email")
                    vectorstore.clear()
                    print(f"✅ RAG vectorstore cleared successfully")
                    if self.logger:
                        self.logger.debug(f"Cleared in-memory RAG vectorstore to rebuild with optimized attack email")
                except Exception as e:
                    print(f"⚠️ Could not clear RAG vectorstore: {e}")
                    debug_info(f"Could not clear RAG vectorstore: {e}")
                    debug_print_exception(e, context="Clearing RAG vectorstore", include_traceback=True)
            else:
                print(f"ℹ️ No vectorstore found in rag_memory_config (will be created when agent processes emails)")
    
    def _cache_successful_attack(self, test_file: Path, result: Dict[str, Any]):
        """Cache a test with successful attacks for future runs."""
        print(f"💾 Caching successful attack for {test_file.name}")
        
        try:
            # Create cache directory structure
            # Handle test files that may be outside test_bench_dir (e.g., attack_bench tests)
            try:
                # Try relative to test_bench_dir first (for tests in data/benchmark/tests/)
                relative_path = test_file.relative_to(self.test_bench_dir)
                cache_file = self.cache_dir / relative_path
            except ValueError:
                # Test file is not under test_bench_dir (e.g., attack_bench tests)
                # Use path relative to data/benchmark/ instead
                try:
                    # Make benchmark_base absolute to handle both absolute and relative test_file paths
                    benchmark_base = Path("data/benchmark").resolve()
                    test_file_resolved = test_file.resolve()
                    relative_path = test_file_resolved.relative_to(benchmark_base)
                    cache_file = self.cache_dir / relative_path
                except ValueError:
                    # Fallback: use string manipulation to extract path after "data/benchmark/"
                    test_file_str = str(test_file)
                    if "data/benchmark/" in test_file_str:
                        relative_path = Path(test_file_str.split("data/benchmark/")[-1])
                    else:
                        # Use filename only as last resort
                        relative_path = Path(test_file.name)
                    cache_file = self.cache_dir / relative_path
            
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load original test and update with optimized attacks
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
            
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
                "optimization_timestamp": datetime.now().isoformat(),
                "model_name": self.model_name
            }
            
            # Save cached version
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_test, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Cached test saved to: {cache_file}")
        except Exception as e:
            print(f"⚠️  Error in _cache_successful_attack: {e}")
            import traceback
            traceback.print_exc()
            # Don't re-raise - let the outer handler deal with it gracefully
    
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
        """Run all tests from the specified path (file or directory). If path is a directory, all *.json files in it are run."""
        test_files = self.discover_test_files(test_path)
        if not test_files:
            return []
        # Show actual test path (file or directory) so logs/results location is clear when using --test with a folder
        print(f"Running tests from: {test_path} ({len(test_files)} file(s))")
        
        # Determine if this is an attack benchmark by checking if test files are in attack_bench directory
        is_attack_bench = False
        for test_file in test_files:
            test_file_str = str(test_file)
            if "attack_bench" in test_file_str:
                is_attack_bench = True
                break
        
        bench_type = "ATTACK BENCHMARK" if is_attack_bench else "TEST BENCH"
        
        print(f"\n{'#'*80}")
        print(f"EMAIL AGENT {bench_type}")
        print(f"{'#'*80}")
        print(f"Found {len(test_files)} test files")
        
        # Group tests by attack_type for better organization
        test_groups = {"memory_only": [], "assistant_responses": [], "untrusted_probe": [], "untrusted_send": [], "disable_send": [], "memory_tools": [], "long_memory": [], "attack_bench": [], "unknown": []}
        for test_file in test_files:
            try:
                test_file_str = str(test_file)
                # Check if this is an attack_bench test
                if "attack_bench" in test_file_str:
                    test_groups["attack_bench"].append(test_file)
                else:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_def = json.load(f)
                        # Use helper function to determine attack_type (handles utility suite tests)
                        attack_type = determine_attack_type(test_file, test_def)
                        if attack_type and attack_type in test_groups:
                            test_groups[attack_type].append(test_file)
                        else:
                            test_groups["unknown"].append(test_file)
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
        Clean up all temporary trace/session directories.
        
        These are minimal temp directories created by tempfile.mkdtemp() for traces/sessions only.
        Since everything else is in-memory, cleanup is optional (OS will clean up temp dirs eventually).
        We still clean them up explicitly to avoid cluttering the temp directory, but silently.
        """
        if not self.test_dirs:
            return
        
        # Silently clean up temp directories (no verbose messages since they're just temp files)
        for test_dir in self.test_dirs:
            if test_dir and test_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(test_dir)
                except Exception:
                    # Ignore cleanup errors for temp directories - OS will clean them up eventually
                    pass
        self.test_dirs.clear()
    
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
        attack_type_groups = {"memory_only": [], "assistant_responses": [], "untrusted_probe": [], "untrusted_send": [], "disable_send": [], "memory_tools": [], "long_memory": [], "attack_bench": [], "unknown": []}
        for result in results:
            # Try to get attack_type from test file
            attack_type = None  # default to None if not found
            try:
                test_file = Path(result.get("test_file", ""))
                if test_file.exists():
                    test_file_str = str(test_file)
                    # Check if this is an attack_bench test
                    if "attack_bench" in test_file_str:
                        attack_type = "attack_bench"
                    else:
                        with open(test_file, 'r', encoding='utf-8') as f:
                            test_def = json.load(f)
                        # Use helper function to determine attack_type (handles utility suite tests)
                        attack_type = determine_attack_type(test_file, test_def)
            except Exception as e:
                debug_debug(f"Could not parse test file {test_file} to determine attack type for result grouping")
                debug_print_exception(e, context=f"Parsing test file {test_file} for result grouping", include_traceback=True)
            if attack_type and attack_type in attack_type_groups:
                attack_type_groups[attack_type].append(result)
            else:
                attack_type_groups["unknown"].append(result)
        
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
    
    parser = argparse.ArgumentParser(description="Email Agent Test Bench")
    parser.add_argument("--test", type=str, nargs="+", help="Run specific test file(s) or directory(ies). Can specify multiple paths separated by spaces.")
    parser.add_argument("--suite", type=str, choices=["memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"], help="Shortcut to run an entire utility test suite under data/benchmark/tests/<suite>/.")
    parser.add_argument("--config", type=str, default="agent_config.yaml", help="Config file")
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
            # No specific tests specified, run all utility test suites by default
            print("No specific tests specified. Running all utility test suites...")
            all_results = []
            for suite in ["memory_only", "assistant_responses", "untrusted_probe", "untrusted_send", "disable_send", "memory_tools", "long_memory"]:
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
