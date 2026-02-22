"""
In-memory storage classes for test environments.

This module provides in-memory implementations of mailbox (inbox/outbox/drafts)
and vectorstore storage to eliminate file system dependencies and contamination
issues in benchmark tests.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import uuid


class InMemoryMailbox:
    """
    In-memory mailbox storage for emails.
    
    Stores emails in memory as dictionaries, organized by folder (inbox, outbox, drafts).
    Provides file-like interface for compatibility with existing code.
    """
    
    def __init__(self):
        """Initialize empty mailbox."""
        self.inbox: List[Dict[str, Any]] = []
        self.outbox: List[Dict[str, Any]] = []
        self.drafts: List[Dict[str, Any]] = []
        
        # Track email IDs for uniqueness
        self._email_ids: set = set()
    
    def add_email(self, email: Dict[str, Any], folder: str = "inbox") -> str:
        """
        Add an email to the specified folder.
        
        Args:
            email: Email dictionary
            folder: Folder name ("inbox", "outbox", or "drafts")
            
        Returns:
            Email ID (generated if not present)
        """
        # Generate email ID if not present
        email_id = email.get("_email_id")
        if not email_id:
            email_id = str(uuid.uuid4())[:8]
            email["_email_id"] = email_id
        
        # Ensure uniqueness
        if email_id in self._email_ids:
            email_id = f"{email_id}_{uuid.uuid4().hex[:4]}"
            email["_email_id"] = email_id
        
        self._email_ids.add(email_id)
        
        # Add to appropriate folder
        if folder == "inbox":
            self.inbox.append(email)
        elif folder == "outbox":
            self.outbox.append(email)
        elif folder == "drafts":
            self.drafts.append(email)
        else:
            raise ValueError(f"Unknown folder: {folder}")
        
        return email_id
    
    def get_emails(self, folder: str = "inbox", unread_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get emails from specified folder.
        
        Args:
            folder: Folder name ("inbox", "outbox", or "drafts")
            unread_only: If True, only return unread emails (for inbox)
            
        Returns:
            List of email dictionaries
        """
        if folder == "inbox":
            emails = self.inbox
            if unread_only:
                emails = [e for e in emails if not e.get("metadata", {}).get("read", False)]
        elif folder == "outbox":
            emails = self.outbox
        elif folder == "drafts":
            emails = self.drafts
        else:
            raise ValueError(f"Unknown folder: {folder}")
        
        return emails.copy()  # Return copy to prevent external modification
    
    def mark_as_read(self, email_id: str, folder: str = "inbox") -> bool:
        """
        Mark an email as read.
        
        Args:
            email_id: Email ID to mark as read
            folder: Folder name
            
        Returns:
            True if email was found and marked, False otherwise
        """
        if folder == "inbox":
            emails = self.inbox
        else:
            return False
        
        for email in emails:
            if email.get("_email_id") == email_id:
                if "metadata" not in email:
                    email["metadata"] = {}
                email["metadata"]["read"] = True
                return True
        
        return False
    
    def remove_email(self, email_id: str, folder: str = "inbox") -> bool:
        """
        Remove an email from the specified folder.
        
        Args:
            email_id: Email ID to remove
            folder: Folder name
            
        Returns:
            True if email was found and removed, False otherwise
        """
        if folder == "inbox":
            emails = self.inbox
        elif folder == "outbox":
            emails = self.outbox
        elif folder == "drafts":
            emails = self.drafts
        else:
            return False
        
        for i, email in enumerate(emails):
            if email.get("_email_id") == email_id:
                emails.pop(i)
                self._email_ids.discard(email_id)
                return True
        
        return False
    
    def clear(self, folder: Optional[str] = None):
        """
        Clear emails from specified folder or all folders.
        
        Args:
            folder: Folder name to clear, or None to clear all
        """
        if folder == "inbox":
            self.inbox.clear()
        elif folder == "outbox":
            self.outbox.clear()
        elif folder == "drafts":
            self.drafts.clear()
        elif folder is None:
            self.inbox.clear()
            self.outbox.clear()
            self.drafts.clear()
            self._email_ids.clear()
        else:
            raise ValueError(f"Unknown folder: {folder}")
    
    def copy_from_filesystem(self, source_dir: Path, folder: str = "inbox"):
        """
        Copy emails from filesystem directory to in-memory storage.
        
        Args:
            source_dir: Source directory path
            folder: Target folder name
        """
        if not source_dir.exists():
            return
        
        for email_file in source_dir.glob("*.json"):
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email = json.load(f)
                    self.add_email(email, folder)
            except Exception:
                continue


class InMemoryVectorstore:
    """
    In-memory vectorstore wrapper for FAISS.
    
    Provides in-memory storage for vector embeddings without file system persistence.
    """
    
    def __init__(self, vectorstore=None):
        """
        Initialize in-memory vectorstore.
        
        Args:
            vectorstore: Optional FAISS vectorstore instance
        """
        self.vectorstore = vectorstore
        self.documents: List[str] = []
        self._chunk_counter = 0
    
    def save(self, vectorstore, documents: List[str], chunk_counter: int):
        """
        Save vectorstore state to memory.
        
        Args:
            vectorstore: FAISS vectorstore instance
            documents: List of document strings
            chunk_counter: Current chunk counter value
        """
        self.vectorstore = vectorstore
        self.documents = documents.copy()
        self._chunk_counter = chunk_counter
    
    def load(self) -> tuple:
        """
        Load vectorstore state from memory.
        
        Returns:
            Tuple of (vectorstore, documents, chunk_counter)
        """
        return self.vectorstore, self.documents.copy(), self._chunk_counter
    
    def clear(self):
        """Clear vectorstore state."""
        self.vectorstore = None
        self.documents.clear()
        self._chunk_counter = 0


class InMemoryTraceStore:
    """
    In-memory trace store for session events.
    
    Stores trace events per session_id in memory instead of writing to files.
    """
    
    def __init__(self):
        """Initialize empty trace store."""
        self._traces: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> list of events
    
    def append_event(self, session_id: str, event: Dict[str, Any]) -> None:
        """Append a trace event for a session."""
        if session_id not in self._traces:
            self._traces[session_id] = []
        self._traces[session_id].append(event)
    
    def get_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all trace events for a session."""
        return self._traces.get(session_id, [])
    
    def clear(self) -> None:
        """Clear all traces."""
        self._traces.clear()
    
    def clear_session(self, session_id: str) -> None:
        """Clear traces for a specific session."""
        if session_id in self._traces:
            del self._traces[session_id]


class InMemorySessionStore:
    """
    In-memory session store.
    
    Stores session data in memory instead of writing to files.
    """
    
    def __init__(self):
        """Initialize empty session store."""
        self._sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session data
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data for a session_id."""
        return self._sessions.get(session_id)
    
    def set_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Set session data for a session_id."""
        self._sessions[session_id] = session_data
    
    def clear(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()
    
    def clear_session(self, session_id: str) -> None:
        """Clear session data for a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]


class InMemoryTestEnvironment:
    """
    Complete in-memory test environment.
    
    Contains ALL runtime state for a test execution:
    - Storage: mailbox, vectorstores, traces, sessions
    - Memory managers: created based on which backend is being tested
    
    No file I/O needed - everything is in memory.
    This is the single source of truth for all test runtime state.
    """
    
    def __init__(self, test_name: str):
        """
        Initialize in-memory test environment.
        
        Args:
            test_name: Name of the test
        """
        self.test_name = test_name
        
        # Storage components
        self.mailbox = InMemoryMailbox()
        self.rag_vectorstore = InMemoryVectorstore()
        self.trace_store = InMemoryTraceStore()
        self.session_store = InMemorySessionStore()
        
        # Exact RAG chunks that were injected into the agent's context (set by get_rag_memory_context when in_memory_env is passed)
        self.last_rag_retrieved_chunks: Optional[List[str]] = None
        # Exact mem0 memories retrieved for the last query (set by get_mem0_memory_context) for scorer/anti-reward-hacking
        self.last_mem0_retrieved_memories: Optional[List[str]] = None

        # Memory managers (initialized based on which backend is used)
        # These are None by default and set by test_bench based on memory_backend
        self.rag_manager = None      # RAGMemoryManager instance
        self.mem0_manager = None     # Mem0MemoryManager instance  
        self.context_manager = None  # ContextMemoryManager instance
        self.explicit_manager = None # MemoryManager instance (for explicit memory)
    
    def get_memory_manager(self, backend: str):
        """
        Get the memory manager for the specified backend.
        
        Args:
            backend: Memory backend name ("rag", "mem0", "context", "explicit", "none")
            
        Returns:
            The memory manager instance, or None if not initialized or backend is "none"
        """
        if backend == "rag":
            return self.rag_manager
        elif backend == "mem0":
            return self.mem0_manager
        elif backend == "context":
            return self.context_manager
        elif backend == "explicit":
            return self.explicit_manager
        else:
            return None
    
    def log_event(self, session_id: str, event_type: str, payload: dict, event_id: Optional[str] = None) -> str:
        """
        Log a trace event to the environment's trace store.
        
        This is the preferred way to log events - use this instead of the global
        append_trace_event function.
        
        Args:
            session_id: Session ID for the event
            event_type: Type of event (e.g., "user_input", "tool_call", "agent_response")
            payload: Event data dictionary
            event_id: Optional event ID (generated if not provided)
            
        Returns:
            The event ID
        """
        from datetime import datetime, timezone
        
        if event_id is None:
            event_id = str(uuid.uuid4())
        
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            **payload
        }
        
        self.trace_store.append_event(session_id, event)
        return event_id
    
    def get_traces(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all trace events for a session.
        
        This is the preferred way to read traces - use this instead of the global
        read_trace_events function.
        
        Args:
            session_id: Session ID to get traces for
            
        Returns:
            List of trace event dictionaries
        """
        return self.trace_store.get_events(session_id)
    
    def clear(self):
        """Clear all storage in the environment."""
        self.mailbox.clear()
        self.rag_vectorstore.clear()
        self.trace_store.clear()
        self.session_store.clear()
        self.last_rag_retrieved_chunks = None
        self.last_mem0_retrieved_memories = None

        # Clear memory managers (they maintain their own internal state)
        if self.rag_manager is not None:
            self.rag_manager.clear_all_memory()
        # mem0, context, explicit managers don't have a standard clear method
        # They are recreated fresh for each test
