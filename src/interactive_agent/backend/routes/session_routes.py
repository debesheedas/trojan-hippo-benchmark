"""
Session management API routes.
Handles creating, listing, and managing chat sessions.
"""

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel
from typing import List, Optional

from ..session_manager import get_session_manager, Session

router = APIRouter()
session_manager = get_session_manager()


class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""
    first_message: Optional[str] = ""


class UpdateSessionTitleRequest(BaseModel):
    """Request model for updating session title."""
    title: str


class SessionResponse(BaseModel):
    """Response model for session data."""
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int


class SessionDetailResponse(BaseModel):
    """Response model for detailed session data."""
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int
    short_term_memory: List[str]
    conversation_history: List[dict]


@router.get("/sessions", response_model=List[SessionResponse])
async def get_all_sessions():
    """Get all sessions (metadata only)."""
    try:
        sessions = session_manager.get_all_sessions()
        return [SessionResponse(**session) for session in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    try:
        session = session_manager.create_session(request.first_message)
        return SessionResponse(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=session.message_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str = Path(..., description="Session ID")):
    """Get detailed information about a specific session."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionDetailResponse(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=session.message_count,
            short_term_memory=session.short_term_memory,
            conversation_history=session.conversation_history
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/title")
async def update_session_title(
    session_id: str = Path(..., description="Session ID"),
    request: UpdateSessionTitleRequest = None
):
    """Update a session's title."""
    try:
        success = session_manager.update_session_title(session_id, request.title)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "success", "message": "Session title updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str = Path(..., description="Session ID")):
    """Delete a session."""
    try:
        success = session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "success", "message": "Session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
