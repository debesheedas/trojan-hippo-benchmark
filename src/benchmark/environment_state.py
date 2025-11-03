"""
Environment State Management for Adaptive Benchmark.

This module handles saving and restoring the exact state of the test environment
at each step, enabling optimization to happen in the correct context.
"""

import copy
from typing import Dict, Any, Optional, List
from pathlib import Path


class EnvironmentState:
    """
    Represents the complete state of the test environment at a given point.
    
    This includes all data that could affect the agent's behavior:
    - Session data (conversation history, memory)
    - Inbox/outbox state
    - Drafts state
    - Any other environment data
    """
    
    def __init__(self):
        # Session data
        self.session_id: Optional[str] = None
        self.session_history: List[Dict[str, Any]] = []
        self.memory_data: Dict[str, Any] = {}
        
        # Email data
        self.inbox_emails: List[Dict[str, Any]] = []
        self.outbox_emails: List[Dict[str, Any]] = []
        self.draft_emails: List[Dict[str, Any]] = []
        
        # Test-specific data
        self.initial_data: Dict[str, Any] = {}
        self.test_config: Dict[str, Any] = {}
        
        # Step-specific data
        self.current_step_num: int = 0
        self.step_traces: List[Dict[str, Any]] = []
        
        # Attack-specific data (for optimization)
        self.attack_emails: List[Dict[str, Any]] = []
        self.original_attack_emails: List[Dict[str, Any]] = []  # Backup of original
        
    def copy(self) -> 'EnvironmentState':
        """Create a deep copy of this environment state."""
        new_state = EnvironmentState()
        
        # Deep copy all attributes
        new_state.session_id = self.session_id
        new_state.session_history = copy.deepcopy(self.session_history)
        new_state.memory_data = copy.deepcopy(self.memory_data)
        new_state.inbox_emails = copy.deepcopy(self.inbox_emails)
        new_state.outbox_emails = copy.deepcopy(self.outbox_emails)
        new_state.draft_emails = copy.deepcopy(self.draft_emails)
        new_state.initial_data = copy.deepcopy(self.initial_data)
        new_state.test_config = copy.deepcopy(self.test_config)
        new_state.current_step_num = self.current_step_num
        new_state.step_traces = copy.deepcopy(self.step_traces)
        new_state.attack_emails = copy.deepcopy(self.attack_emails)
        new_state.original_attack_emails = copy.deepcopy(self.original_attack_emails)
        
        return new_state
    
    def update_from_test_data(self, test_data: Dict[str, Any], test_config: Dict[str, Any]):
        """Update state from test case data."""
        self.initial_data = test_data.get("initial_data", {})
        self.test_config = test_config
        
        # Load attack emails
        self.attack_emails = self.initial_data.get("attack_emails", [])
        self.original_attack_emails = copy.deepcopy(self.attack_emails)
        
        # Load other initial data
        self.inbox_emails = self.initial_data.get("inbox_emails", [])
        self.outbox_emails = self.initial_data.get("outbox_emails", [])
        self.draft_emails = self.initial_data.get("draft_emails", [])
        
    def update_session_data(self, session_id: str, session_history: List[Dict[str, Any]], 
                           memory_data: Dict[str, Any]):
        """Update session-related data."""
        self.session_id = session_id
        self.session_history = copy.deepcopy(session_history)
        self.memory_data = copy.deepcopy(memory_data)
        
    def update_step_data(self, step_num: int, step_traces: List[Dict[str, Any]]):
        """Update step-specific data."""
        self.current_step_num = step_num
        self.step_traces = copy.deepcopy(step_traces)
        
    def inject_attack_email(self, attack_email: Dict[str, Any], replace_index: int = 0):
        """
        Inject an optimized attack email into the environment.
        
        Args:
            attack_email: The optimized attack email to inject
            replace_index: Index of the attack email to replace (default: 0)
        """
        if replace_index < len(self.attack_emails):
            self.attack_emails[replace_index] = copy.deepcopy(attack_email)
        else:
            # If index doesn't exist, append
            self.attack_emails.append(copy.deepcopy(attack_email))
    
    def restore_original_attack_emails(self):
        """Restore the original attack emails (undo optimization)."""
        self.attack_emails = copy.deepcopy(self.original_attack_emails)
    
    def get_attack_email(self, index: int = 0) -> Optional[Dict[str, Any]]:
        """Get the attack email at the specified index."""
        if index < len(self.attack_emails):
            return self.attack_emails[index]
        return None
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get a summary of the current environment state for logging."""
        return {
            "session_id": self.session_id,
            "step_num": self.current_step_num,
            "session_history_length": len(self.session_history),
            "memory_entries": len(self.memory_data),
            "inbox_emails": len(self.inbox_emails),
            "outbox_emails": len(self.outbox_emails),
            "draft_emails": len(self.draft_emails),
            "attack_emails": len(self.attack_emails),
            "step_traces_length": len(self.step_traces)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "session_history": self.session_history,
            "memory_data": self.memory_data,
            "inbox_emails": self.inbox_emails,
            "outbox_emails": self.outbox_emails,
            "draft_emails": self.draft_emails,
            "initial_data": self.initial_data,
            "test_config": self.test_config,
            "current_step_num": self.current_step_num,
            "step_traces": self.step_traces,
            "attack_emails": self.attack_emails,
            "original_attack_emails": self.original_attack_emails
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentState':
        """Create an EnvironmentState from a dictionary."""
        state = cls()
        state.session_id = data.get("session_id")
        state.session_history = data.get("session_history", [])
        state.memory_data = data.get("memory_data", {})
        state.inbox_emails = data.get("inbox_emails", [])
        state.outbox_emails = data.get("outbox_emails", [])
        state.draft_emails = data.get("draft_emails", [])
        state.initial_data = data.get("initial_data", {})
        state.test_config = data.get("test_config", {})
        state.current_step_num = data.get("current_step_num", 0)
        state.step_traces = data.get("step_traces", [])
        state.attack_emails = data.get("attack_emails", [])
        state.original_attack_emails = data.get("original_attack_emails", [])
        return state


class StateManager:
    """
    Manages environment state transitions during test execution.
    
    This class handles the prev_state and curr_state management,
    ensuring we can always restore to the previous step's state.
    """
    
    def __init__(self):
        self.prev_state: Optional[EnvironmentState] = None
        self.curr_state: Optional[EnvironmentState] = None
        self.initial_state: Optional[EnvironmentState] = None
        
    def initialize(self, test_data: Dict[str, Any], test_config: Dict[str, Any]):
        """Initialize the state manager with test data."""
        self.curr_state = EnvironmentState()
        self.curr_state.update_from_test_data(test_data, test_config)
        self.initial_state = self.curr_state.copy()
        
    def save_state_before_step(self, step_num: int):
        """Save the current state before executing a step."""
        if self.curr_state is not None:
            self.prev_state = self.curr_state.copy()
            self.curr_state.current_step_num = step_num
            
    def restore_to_prev_state(self):
        """Restore the current state to the previous step's state."""
        if self.prev_state is not None:
            self.curr_state = self.prev_state.copy()
            return True
        return False
    
    def get_current_state(self) -> Optional[EnvironmentState]:
        """Get the current environment state."""
        return self.curr_state
    
    def get_prev_state(self) -> Optional[EnvironmentState]:
        """Get the previous environment state."""
        return self.prev_state
    
    def get_initial_state(self) -> Optional[EnvironmentState]:
        """Get the initial environment state."""
        return self.initial_state
    
    def update_current_state(self, **kwargs):
        """Update the current state with new data."""
        if self.curr_state is not None:
            if 'session_id' in kwargs and 'session_history' in kwargs and 'memory_data' in kwargs:
                self.curr_state.update_session_data(
                    kwargs['session_id'], 
                    kwargs['session_history'], 
                    kwargs['memory_data']
                )
            if 'step_num' in kwargs and 'step_traces' in kwargs:
                self.curr_state.update_step_data(
                    kwargs['step_num'], 
                    kwargs['step_traces']
                )
    
    def inject_attack_email(self, attack_email: Dict[str, Any], replace_index: int = 0):
        """Inject an attack email into the current state."""
        if self.curr_state is not None:
            self.curr_state.inject_attack_email(attack_email, replace_index)
    
    def restore_original_attack_emails(self):
        """Restore original attack emails in the current state."""
        if self.curr_state is not None:
            self.curr_state.restore_original_attack_emails()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of both states for logging."""
        summary = {
            "prev_state": self.prev_state.get_environment_summary() if self.prev_state else None,
            "curr_state": self.curr_state.get_environment_summary() if self.curr_state else None
        }
        return summary
