"""
Session Manager for handling multiple chat sessions.
Manages session storage, retrieval, and memory behavior.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import threading

SESSIONS_DIR = Path("data/interactive_agent/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_INDEX_FILE = Path("data/interactive_agent/session_index.json")
SHORT_TERM_MEMORY_MAX_SIZE = 15


@dataclass
class Session:
    """Represents a chat session."""
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int
    short_term_memory: List[str]
    conversation_history: List[Dict]


class SessionManager:
    """Manages chat sessions with persistent storage."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the session manager."""
        self._load_session_index()
    
    def _load_session_index(self):
        """Load the session index from disk."""
        if SESSION_INDEX_FILE.exists():
            try:
                with open(SESSION_INDEX_FILE, 'r', encoding='utf-8') as f:
                    self.session_index = json.load(f)
            except json.JSONDecodeError:
                self.session_index = {"sessions": [], "next_id": 1}
        else:
            self.session_index = {"sessions": [], "next_id": 1}
    
    def _save_session_index(self):
        """Save the session index to disk."""
        with open(SESSION_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.session_index, f, indent=2)
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return SESSIONS_DIR / f"{session_id}.json"
    
    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        session_id = f"session_{self.session_index['next_id']:03d}"
        self.session_index['next_id'] += 1
        return session_id
    
    def _generate_session_title(self, first_message: str) -> str:
        """Generate a title for a session based on the first message."""
        # Truncate and clean the first message for title
        title = first_message.strip()[:50]
        if len(first_message) > 50:
            title += "..."
        return title
    
    def create_session(self, first_message: str = "") -> Session:
        """Create a new session."""
        session_id = self._generate_session_id()
        # Use fixed date for reproducibility (November 3, 2025)
        now = datetime(2025, 11, 3, 12, 0, 0).isoformat()
        title = self._generate_session_title(first_message) if first_message else "New Chat"
        
        session = Session(
            session_id=session_id,
            title=title,
            created_at=now,
            last_updated=now,
            message_count=0,
            short_term_memory=[],
            conversation_history=[]
        )
        
        # Save session to disk
        self._save_session(session)
        
        # Update session index
        self.session_index["sessions"].append({
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "last_updated": now,
            "message_count": 0
        })
        self._save_session_index()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Session(**data)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions (metadata only)."""
        return self.session_index.get("sessions", [])
    
    def _save_session(self, session: Session):
        """Save a session to disk."""
        session_file = self._get_session_file(session.session_id)
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(session), f, indent=2)
    
    def update_session(self, session: Session):
        """Update a session."""
        # Use fixed date for reproducibility (November 3, 2025)
        session.last_updated = datetime(2025, 11, 3, 12, 0, 0).isoformat()
        self._save_session(session)
        
        # Update session index
        for i, session_meta in enumerate(self.session_index["sessions"]):
            if session_meta["session_id"] == session.session_id:
                self.session_index["sessions"][i] = {
                    "session_id": session.session_id,
                    "title": session.title,
                    "created_at": session.created_at,
                    "last_updated": session.last_updated,
                    "message_count": session.message_count
                }
                break
        self._save_session_index()
    
    def add_message_to_session(self, session_id: str, message: str, is_user: bool = True):
        """Add a message to a session's short-term memory and conversation history."""
        session = self.get_session(session_id)
        if not session:
            return
        
        # Add to short-term memory (rolling buffer)
        session.short_term_memory.append(message)
        if len(session.short_term_memory) > SHORT_TERM_MEMORY_MAX_SIZE:
            session.short_term_memory.pop(0)
        
        # Add to conversation history
        session.conversation_history.append({
            "message": message,
            "is_user": is_user,
            # Use fixed date for reproducibility (November 3, 2025)
            "timestamp": datetime(2025, 11, 3, 12, 0, 0).isoformat()
        })
        
        session.message_count += 1
        self.update_session(session)
    
    def get_session_short_term_memory(self, session_id: str) -> List[str]:
        """Get short-term memory for a session."""
        session = self.get_session(session_id)
        return session.short_term_memory if session else []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()
        
        # Remove from session index
        self.session_index["sessions"] = [
            s for s in self.session_index["sessions"] 
            if s["session_id"] != session_id
        ]
        self._save_session_index()
        return True
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """Update a session's title."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.title = new_title
        self.update_session(session)
        return True


def get_session_manager() -> SessionManager:
    """Get the singleton session manager instance."""
    return SessionManager()
