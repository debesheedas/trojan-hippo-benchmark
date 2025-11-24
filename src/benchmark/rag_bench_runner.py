#!/usr/bin/env python3
"""
RAG Bench Test Runner

Runs RAG poisoning attack test cases from the rag_bench folder.
Provides comprehensive logging of all operations including:
- User messages and agent responses
- Tool calls and results
- RAG memory storage and retrieval
- Session transitions
"""

import json
import time
import shutil
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
from agent.utils import (
    read_trace_events, load_config, ensure_data_directories,
    append_trace_event, USER_EMAIL
)
from benchmark.test_validators import create_validator
from benchmark.rag_poisoning_attack import store_session_in_rag_memory
from agent.backend.rag_memory_manager import get_rag_memory_manager


class RAGBenchRunner:
    """Runner for RAG poisoning attack test cases."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG bench runner."""
        self.config = load_config(config_path)
        self.logger = self._setup_logger()
        self.results = []
        
        # Ensure RAG memory is enabled
        rag_config = self.config.get("memory", {}).get("rag_memory", {})
        if not rag_config.get("enabled", False):
            raise ValueError("RAG memory must be enabled in config.yaml")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up detailed logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rag_bench_{timestamp}.log"
        
        logger = logging.getLogger("rag_bench")
        logger.setLevel(logging.DEBUG)
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging to: {log_file}")
        return logger
    
    def create_test_environment(self, test_name: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create isolated test environment."""
        # Create temporary directory for this test
        test_dir = Path(f"test_env_{uuid.uuid4().hex[:8]}")
        test_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        mailbox_dir = test_dir / "mailbox"
        outbox_dir = test_dir / "outbox"
        drafts_dir = test_dir / "drafts"
        memory_file = test_dir / "agent_memory.json"
        trace_file = test_dir / "trace.jsonl"
        sessions_dir = test_dir / "sessions"
        traces_dir = test_dir / "traces"
        
        for d in [mailbox_dir, outbox_dir, drafts_dir, sessions_dir, traces_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Copy initial data if specified (inbox, outbox, drafts, memory)
        if initial_data:
            # Copy inbox data
            if "inbox_set" in initial_data:
                inbox_set = initial_data["inbox_set"]
                source_inbox = Path(f"data/benchmark/initial_inbox/{inbox_set}")
                if source_inbox.exists():
                    for email_file in source_inbox.glob("*.json"):
                        shutil.copy2(email_file, mailbox_dir)
                    self.logger.info(f"Loaded inbox set: {inbox_set} ({len(list(source_inbox.glob('*.json')))} emails)")
                else:
                    self.logger.warning(f"Inbox set '{inbox_set}' not found at {source_inbox}")
            
            # Copy outbox data
            if "outbox_set" in initial_data:
                outbox_set = initial_data["outbox_set"]
                source_outbox = Path(f"data/benchmark/initial_outbox/{outbox_set}")
                if source_outbox.exists():
                    for email_file in source_outbox.glob("*.json"):
                        shutil.copy2(email_file, outbox_dir)
                    self.logger.info(f"Loaded outbox set: {outbox_set}")
                else:
                    self.logger.warning(f"Outbox set '{outbox_set}' not found at {source_outbox}")
            
            # Copy drafts data
            if "drafts_set" in initial_data:
                drafts_set = initial_data["drafts_set"]
                source_drafts = Path(f"data/benchmark/initial_drafts/{drafts_set}")
                if source_drafts.exists():
                    for email_file in source_drafts.glob("*.json"):
                        shutil.copy2(email_file, drafts_dir)
                    self.logger.info(f"Loaded drafts set: {drafts_set}")
                else:
                    self.logger.warning(f"Drafts set '{drafts_set}' not found at {source_drafts}")
            
            # Load memory data
            if "memory_set" in initial_data:
                memory_set = initial_data["memory_set"]
                source_memory = Path(f"data/benchmark/initial_memory/{memory_set}.json")
                if source_memory.exists():
                    with open(source_memory, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                    with open(memory_file, 'w', encoding='utf-8') as f:
                        json.dump(memory_data, f, indent=2)
                    self.logger.info(f"Loaded memory set: {memory_set}")
                else:
                    self.logger.warning(f"Memory set '{memory_set}' not found at {source_memory}")
        
        # Create test config
        test_config = self.config.copy()
        test_config["data"] = {
            "mailbox_dir": str(mailbox_dir),
            "outbox_dir": str(outbox_dir),
            "drafts_dir": str(drafts_dir),
            "memory_file": str(memory_file),
            "trace_file": str(trace_file)
        }
        
        # Initialize memory file
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump({"short_term": [], "long_term": []}, f, indent=2)
        
        return test_config
    
    def run_test_from_file(self, test_file: Path) -> Dict[str, Any]:
        """Run a single RAG bench test from a JSON file."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"RUNNING RAG BENCH TEST: {test_file.name}")
        self.logger.info(f"{'='*80}")
        
        # Load test definition
        with open(test_file, 'r', encoding='utf-8') as f:
            test_def = json.load(f)
        
        test_name = test_def["name"]
        description = test_def["description"]
        steps = test_def["steps"]
        initial_data = test_def.get("initial_data", {})
        rag_memory_config = test_def.get("rag_memory", {})
        
        self.logger.info(f"Test: {test_name}")
        self.logger.info(f"Description: {description}")
        self.logger.info(f"Steps: {len(steps)}")
        
        # Clear agent cache
        clear_agent_cache()
        from agent.agent_core import _session_store
        _session_store.clear()
        
        # Create isolated test environment
        test_config = self.create_test_environment(test_name, initial_data)
        test_dir = Path(test_config["data"]["mailbox_dir"]).parent
        
        # Update RAG memory config if specified
        if rag_memory_config:
            test_config["memory"]["rag_memory"].update(rag_memory_config)
        
        try:
            # Load initial RAG memory set if specified
            if "rag_memory_set" in initial_data:
                rag_memory_set = initial_data["rag_memory_set"]
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"LOADING INITIAL RAG MEMORY SET: {rag_memory_set}")
                self.logger.info(f"{'='*80}")
                
                try:
                    from benchmark.rag_memory_loader import load_rag_memory_set
                    # Use test-specific vectorstore path
                    test_vectorstore_path = f"{test_config['memory']['rag_memory'].get('vectorstore_path', 'data/interactive_agent/rag_vectorstore')}_{test_name}_{rag_memory_set}"
                    chunks_loaded = load_rag_memory_set(
                        rag_memory_set=rag_memory_set,
                        config=test_config,
                        vectorstore_path=test_vectorstore_path,
                        force_new=True
                    )
                    # Update config to use this vectorstore
                    test_config["memory"]["rag_memory"]["vectorstore_path"] = test_vectorstore_path
                    self.logger.info(f"✓ Loaded {chunks_loaded} chunks into RAG memory")
                except Exception as e:
                    self.logger.error(f"Error loading RAG memory set: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # Generate session IDs
            session1_id = f"rag_bench_session1_{uuid.uuid4().hex[:8]}"
            session2_id = f"rag_bench_session2_{uuid.uuid4().hex[:8]}"
            
            self.logger.info(f"Session 1 ID: {session1_id}")
            self.logger.info(f"Session 2 ID: {session2_id}")
            
            # Clear sessions
            clear_session_agent(session1_id, config=test_config, auto_store_rag=False)
            clear_session_agent(session2_id, config=test_config, auto_store_rag=False)
            
            # Process steps
            step_results = []
            session1_steps = []
            session2_steps = []
            current_session_id = session1_id
            
            for i, step in enumerate(steps, 1):
                step_type = step.get("step_type", "user_message")
                
                if step_type == "session_1":
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"SESSION 1 - Step {i}")
                    self.logger.info(f"{'='*80}")
                    
                    # Insert attack emails
                    attack_emails = step.get("attack_emails", [])
                    if attack_emails:
                        self.logger.info(f"Inserting {len(attack_emails)} attack emails...")
                        inbox_dir = Path(test_config["data"]["mailbox_dir"])
                        for email_idx, attack_email in enumerate(attack_emails):
                            email_id = str(uuid.uuid4())[:8]
                            email_file = inbox_dir / f"attack_{email_idx:03d}_{email_id}.json"
                            with open(email_file, 'w', encoding='utf-8') as f:
                                json.dump(attack_email, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"  ✓ Inserted: {email_file.name}")
                            self.logger.debug(f"    From: {attack_email.get('from')}")
                            self.logger.debug(f"    Subject: {attack_email.get('subject')}")
                    
                    # Run Session 1 queries
                    session1_queries = step.get("session1_queries", [])
                    session1_responses = []
                    
                    for query_idx, query_step in enumerate(session1_queries, 1):
                        user_message = query_step.get("user_message", "")
                        self.logger.info(f"\n--- Session 1 Query {query_idx}/{len(session1_queries)} ---")
                        self.logger.info(f"👤 User: {user_message}")
                        
                        start_time = time.time()
                        try:
                            response = invoke_agent(
                                text=user_message,
                                session_id=session1_id,
                                config=test_config
                            )
                            duration = time.time() - start_time
                            
                            response_text = response.get("response", "")
                            self.logger.info(f"🤖 Agent Response ({len(response_text)} chars, {duration:.2f}s):")
                            self.logger.info(f"{response_text[:500]}{'...' if len(response_text) > 500 else ''}")
                            
                            # Get traces for this query
                            traces = read_trace_events(test_config["data"]["trace_file"], session1_id)
                            query_traces = [t for t in traces if t.get("event_type") in ["user_input", "tool_call", "tool_result", "agent_response"]]
                            
                            # Log tool calls
                            tool_calls = [t for t in query_traces if t.get("event_type") == "tool_call"]
                            if tool_calls:
                                self.logger.info(f"🔧 Tool Calls ({len(tool_calls)}):")
                                for tool_trace in tool_calls:
                                    tool_name = tool_trace.get("payload", {}).get("tool_name", "unknown")
                                    tool_inputs = tool_trace.get("payload", {}).get("inputs", {})
                                    self.logger.info(f"  - {tool_name}({json.dumps(tool_inputs, indent=2)})")
                            
                            session1_responses.append({
                                "query": user_message,
                                "response": response_text,
                                "duration_s": duration,
                                "tool_calls": len(tool_calls),
                                "traces": query_traces
                            })
                            
                        except Exception as e:
                            self.logger.error(f"Error in Session 1 query: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            session1_responses.append({
                                "query": user_message,
                                "response": "",
                                "error": str(e),
                                "duration_s": 0.0
                            })
                    
                    # Store Session 1 in RAG memory
                    chunks_stored = 0
                    if step.get("store_in_rag", False):
                        self.logger.info(f"\n{'='*80}")
                        self.logger.info("STORING SESSION 1 IN RAG MEMORY")
                        self.logger.info(f"{'='*80}")
                        
                        try:
                            chunks_stored = store_session_in_rag_memory(session1_id, test_config)
                            self.logger.info(f"✓ Stored {chunks_stored} conversation chunks in RAG memory")
                            
                            # Log what was stored
                            rag_memory_manager = get_rag_memory_manager(
                                embedding_model=test_config["memory"]["rag_memory"].get("embedding_model", "text-embedding-3-small"),
                                top_k=test_config["memory"]["rag_memory"].get("top_k", 3),
                                chunk_size=test_config["memory"]["rag_memory"].get("chunk_size", 512),
                                vectorstore_path=test_config["memory"]["rag_memory"].get("vectorstore_path", "data/interactive_agent/rag_vectorstore"),
                                force_new=False
                            )
                            
                            # Get session memory to show what was chunked
                            from agent.agent_core import _get_session_memory
                            session_messages = _get_session_memory(session1_id)
                            self.logger.info(f"Session 1 had {len(session_messages)} messages")
                            for msg_idx, msg in enumerate(session_messages):
                                content = msg.get("content", "")[:200]
                                self.logger.debug(f"  Message {msg_idx}: {msg.get('role')} - {content}...")
                            
                        except Exception as e:
                            self.logger.error(f"Error storing Session 1 in RAG: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                    
                    session1_steps.append({
                        "step": i,
                        "step_type": "session_1",
                        "attack_emails": len(attack_emails),
                        "queries": session1_responses,
                        "chunks_stored": chunks_stored
                    })
                
                elif step_type == "evaluation_query":
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"EVALUATION QUERY - Step {i}")
                    self.logger.info(f"{'='*80}")
                    
                    user_message = step.get("user_message", "")
                    evaluation_config = step.get("evaluation", {})
                    ground_truth = evaluation_config.get("ground_truth", [])
                    
                    self.logger.info(f"👤 User: {user_message}")
                    if ground_truth:
                        self.logger.info(f"📋 Ground Truth: {ground_truth[0] if isinstance(ground_truth, list) and ground_truth else ground_truth}")
                    
                    # Get RAG context before query
                    rag_context = ""
                    try:
                        rag_memory_manager = get_rag_memory_manager(
                            embedding_model=test_config["memory"]["rag_memory"].get("embedding_model", "text-embedding-3-small"),
                            top_k=test_config["memory"]["rag_memory"].get("top_k", 3),
                            chunk_size=test_config["memory"]["rag_memory"].get("chunk_size", 512),
                            vectorstore_path=test_config["memory"]["rag_memory"].get("vectorstore_path", "data/interactive_agent/rag_vectorstore"),
                            force_new=False
                        )
                        rag_context = rag_memory_manager.get_context(user_message)
                        if rag_context:
                            self.logger.info(f"📚 RAG Context Retrieved ({len(rag_context)} chars):")
                            self.logger.info(f"{rag_context[:500]}{'...' if len(rag_context) > 500 else ''}")
                    except Exception as e:
                        self.logger.warning(f"Could not retrieve RAG context: {e}")
                    
                    # Use a single session for evaluation queries
                    eval_session_id = f"rag_bench_eval_{uuid.uuid4().hex[:8]}"
                    clear_session_agent(eval_session_id, config=test_config, auto_store_rag=False)
                    
                    start_time = time.time()
                    evaluation_result = None
                    try:
                        response = invoke_agent(
                            text=user_message,
                            session_id=eval_session_id,
                            config=test_config
                        )
                        duration = time.time() - start_time
                        
                        response_text = response.get("response", "")
                        self.logger.info(f"🤖 Agent Response ({len(response_text)} chars, {duration:.2f}s):")
                        self.logger.info(f"{response_text}")
                        
                        # Get traces
                        traces = read_trace_events(test_config["data"]["trace_file"], eval_session_id)
                        query_traces = [t for t in traces if t.get("event_type") in ["user_input", "tool_call", "tool_result", "agent_response"]]
                        
                        # Log tool calls
                        tool_calls = [t for t in query_traces if t.get("event_type") == "tool_call"]
                        if tool_calls:
                            self.logger.info(f"🔧 Tool Calls ({len(tool_calls)}):")
                            for tool_trace in tool_calls:
                                tool_name = tool_trace.get("payload", {}).get("tool_name", "unknown")
                                tool_inputs = tool_trace.get("payload", {}).get("inputs", {})
                                self.logger.info(f"  - {tool_name}")
                                self.logger.info(f"    Inputs: {json.dumps(tool_inputs, indent=4)}")
                        
                        # Evaluate response if ground truth provided
                        if ground_truth:
                            from benchmark.memory_metrics import evaluate_response
                            evaluation_result = evaluate_response(
                                agent_response=response_text,
                                ground_truth=ground_truth,
                                query=user_message,
                                use_post_process=True
                            )
                            
                            metrics = evaluation_result.get("metrics", {})
                            self.logger.info(f"\n📊 Evaluation Metrics:")
                            self.logger.info(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")
                            self.logger.info(f"  F1 Score: {metrics.get('f1', 0):.4f}")
                            self.logger.info(f"  Substring Match: {metrics.get('substring_exact_match', 0):.4f}")
                            if 'rougeL_f1' in metrics:
                                self.logger.info(f"  ROUGE-L F1: {metrics.get('rougeL_f1', 0):.4f}")
                            if 'rougeLsum_f1' in metrics:
                                self.logger.info(f"  ROUGE-Lsum F1: {metrics.get('rougeLsum_f1', 0):.4f}")
                        
                        step_results.append({
                            "step": i,
                            "step_type": "evaluation_query",
                            "user_message": user_message,
                            "response": response_text,
                            "rag_context": rag_context,
                            "duration_s": duration,
                            "tool_calls": [{"name": t.get("payload", {}).get("tool_name"), "inputs": t.get("payload", {}).get("inputs")} for t in tool_calls],
                            "evaluation": evaluation_result,
                            "traces": query_traces
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in evaluation query: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        step_results.append({
                            "step": i,
                            "step_type": "evaluation_query",
                            "user_message": user_message,
                            "response": "",
                            "error": str(e),
                            "duration_s": 0.0,
                            "evaluation": None
                        })
                
                elif step_type == "session_2":
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"SESSION 2 - Step {i}")
                    self.logger.info(f"{'='*80}")
                    
                    user_message = step.get("user_message", "")
                    self.logger.info(f"👤 User: {user_message}")
                    
                    # Get RAG context before query
                    rag_context = ""
                    try:
                        rag_memory_manager = get_rag_memory_manager(
                            embedding_model=test_config["memory"]["rag_memory"].get("embedding_model", "text-embedding-3-small"),
                            top_k=test_config["memory"]["rag_memory"].get("top_k", 3),
                            chunk_size=test_config["memory"]["rag_memory"].get("chunk_size", 512),
                            vectorstore_path=test_config["memory"]["rag_memory"].get("vectorstore_path", "data/interactive_agent/rag_vectorstore"),
                            force_new=False
                        )
                        rag_context = rag_memory_manager.get_context(user_message)
                        if rag_context:
                            self.logger.info(f"📚 RAG Context Retrieved ({len(rag_context)} chars):")
                            self.logger.info(f"{rag_context[:500]}{'...' if len(rag_context) > 500 else ''}")
                    except Exception as e:
                        self.logger.warning(f"Could not retrieve RAG context: {e}")
                    
                    start_time = time.time()
                    try:
                        response = invoke_agent(
                            text=user_message,
                            session_id=session2_id,
                            config=test_config
                        )
                        duration = time.time() - start_time
                        
                        response_text = response.get("response", "")
                        self.logger.info(f"🤖 Agent Response ({len(response_text)} chars, {duration:.2f}s):")
                        self.logger.info(f"{response_text}")
                        
                        # Get traces
                        traces = read_trace_events(test_config["data"]["trace_file"], session2_id)
                        query_traces = [t for t in traces if t.get("event_type") in ["user_input", "tool_call", "tool_result", "agent_response"]]
                        
                        # Log tool calls
                        tool_calls = [t for t in query_traces if t.get("event_type") == "tool_call"]
                        if tool_calls:
                            self.logger.info(f"🔧 Tool Calls ({len(tool_calls)}):")
                            for tool_trace in tool_calls:
                                tool_name = tool_trace.get("payload", {}).get("tool_name", "unknown")
                                tool_inputs = tool_trace.get("payload", {}).get("inputs", {})
                                self.logger.info(f"  - {tool_name}")
                                self.logger.info(f"    Inputs: {json.dumps(tool_inputs, indent=4)}")
                        
                        # Validate goals
                        user_goal = step.get("user_goal", {})
                        attack_goal = step.get("attack_goal", {})
                        
                        user_goal_passed = False
                        attack_goal_passed = False
                        
                        if user_goal.get("success_check"):
                            validator = create_validator(user_goal["success_check"])
                            user_goal_passed = validator.validate(response_text, session2_id, traces)
                            self.logger.info(f"✓ User Goal: {'PASSED' if user_goal_passed else 'FAILED'}")
                        
                        if attack_goal.get("success_check"):
                            validator = create_validator(attack_goal["success_check"])
                            attack_goal_passed = validator.validate(response_text, session2_id, traces)
                            self.logger.info(f"{'⚠️' if attack_goal_passed else '✓'} Attack Goal: {'PASSED (ATTACK SUCCESSFUL)' if attack_goal_passed else 'FAILED (Attack Blocked)'}")
                        
                        session2_steps.append({
                            "step": i,
                            "step_type": "session_2",
                            "user_message": user_message,
                            "response": response_text,
                            "rag_context": rag_context,
                            "duration_s": duration,
                            "tool_calls": [{"name": t.get("payload", {}).get("tool_name"), "inputs": t.get("payload", {}).get("inputs")} for t in tool_calls],
                            "user_goal_passed": user_goal_passed,
                            "attack_goal_passed": attack_goal_passed,
                            "traces": query_traces
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in Session 2 query: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        session2_steps.append({
                            "step": i,
                            "step_type": "session_2",
                            "user_message": user_message,
                            "response": "",
                            "error": str(e),
                            "duration_s": 0.0
                        })
            
            # Aggregate evaluation metrics if this is an evaluation test
            evaluation_metrics = None
            if test_def.get("attack_type") == "evaluation" and step_results:
                from benchmark.memory_metrics import aggregate_metrics
                evaluation_results = [s.get("evaluation") for s in step_results if s.get("evaluation")]
                if evaluation_results:
                    evaluation_metrics = aggregate_metrics(evaluation_results)
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info("AGGREGATED EVALUATION METRICS")
                    self.logger.info(f"{'='*80}")
                    for metric_name, metric_value in evaluation_metrics.items():
                        self.logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Compile results
            result = {
                "test_name": test_name,
                "test_file": str(test_file),
                "description": description,
                "attack_type": attack_type,
                "session1_id": session1_id if session1_steps else None,
                "session2_id": session2_id if session2_steps else None,
                "session1": {
                    "steps": session1_steps
                } if session1_steps else None,
                "session2": {
                    "steps": session2_steps
                } if session2_steps else None,
                "evaluation_queries": {
                    "steps": step_results
                } if step_results and not session1_steps and not session2_steps else None,
                "overall_success": all(s.get("attack_goal_passed", False) for s in session2_steps if s.get("step_type") == "session_2") if session2_steps else None,
                "evaluation_metrics": evaluation_metrics,
                "test_dir": str(test_dir)
            }
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info("TEST COMPLETE")
            self.logger.info(f"{'='*80}")
            if session1_steps:
                self.logger.info(f"Session 1 Steps: {len(session1_steps)}")
            if session2_steps:
                self.logger.info(f"Session 2 Steps: {len(session2_steps)}")
            if step_results and not session1_steps and not session2_steps:
                self.logger.info(f"Evaluation Queries: {len(step_results)}")
            if result.get('overall_success') is not None:
                self.logger.info(f"Attack Success: {result['overall_success']}")
            if evaluation_metrics:
                self.logger.info(f"Evaluation Metrics: {evaluation_metrics}")
            self.logger.info(f"Test Directory: {test_dir}")
            self.logger.info(f"Trace Files: {test_dir / 'traces'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running test: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "test_name": test_name,
                "error": str(e),
                "overall_success": False
            }
        finally:
            # Don't cleanup test_dir - keep it for inspection
            pass
    
    def run_all_tests(self, test_path: str) -> List[Dict[str, Any]]:
        """Run all tests from the specified path."""
        test_path_obj = Path(test_path)
        results = []
        
        if test_path_obj.is_file():
            # Single file
            result = self.run_test_from_file(test_path_obj)
            results.append(result)
        else:
            # Directory - find all JSON files
            test_files = sorted(test_path_obj.glob("*.json"))
            if not test_files:
                self.logger.warning(f"No JSON test files found in {test_path}")
                return results
            
            for test_file in test_files:
                result = self.run_test_from_file(test_file)
                results.append(result)
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Bench Test Runner")
    parser.add_argument("--test", type=str, nargs="+", help="Run specific test file(s) or directory")
    parser.add_argument("--suite", type=str, default="rag_bench", help="Run test suite (default: rag_bench)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    runner = RAGBenchRunner(args.config)
    
    if args.test:
        test_paths = args.test
    elif args.suite:
        test_paths = [f"data/benchmark/{args.suite}"]
    else:
        test_paths = ["data/benchmark/rag_bench"]
    
    all_results = []
    for test_path in test_paths:
        results = runner.run_all_tests(test_path)
        all_results.extend(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in all_results:
        status = "✅ PASS" if result.get("overall_success") else "❌ FAIL"
        print(f"{status} - {result.get('test_name', 'Unknown')}")
        if "test_dir" in result:
            print(f"  Test Directory: {result['test_dir']}")
            print(f"  Traces: {result['test_dir']}/traces/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

