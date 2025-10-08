"""
Memory Routes Module
FastAPI routes for memory management with SSE support.
"""

import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.memory_manager import get_memory_manager


# Create router
router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryUpdateRequest(BaseModel):
    """Request model for memory updates."""
    update: str


class MemoryResponse(BaseModel):
    """Response model for memory retrieval."""
    short_term: list
    long_term: list


@router.get("", response_model=MemoryResponse)
async def get_memory():
    """
    Get current memory state.
    
    Returns:
        Current short-term and long-term memory
    """
    try:
        memory_manager = get_memory_manager()
        context = memory_manager.get_context()
        return MemoryResponse(
            short_term=context["short_term"],
            long_term=context["long_term"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("")
async def update_memory(request: MemoryUpdateRequest):
    """
    Add an update to long-term memory.
    
    Args:
        request: Memory update request containing update text
    
    Returns:
        Success message and updated memory state
    """
    try:
        memory_manager = get_memory_manager()
        memory_manager.add_long_term(request.update)
        context = memory_manager.get_context()
        
        return {
            "status": "success",
            "message": "Memory updated",
            "memory": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("")
async def clear_memory():
    """
    Clear all memory (both short-term and long-term).
    
    Returns:
        Success message
    """
    try:
        memory_manager = get_memory_manager()
        memory_manager.clear_all_memory()
        
        return {
            "status": "success",
            "message": "All memory cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stream")
async def stream_memory():
    """
    Server-Sent Events endpoint for live memory updates.
    
    Returns:
        SSE stream of memory changes
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for memory updates."""
        memory_manager = get_memory_manager()
        
        # Create a queue for this client
        queue = asyncio.Queue()
        
        def callback(context):
            """Callback to push updates to queue."""
            try:
                asyncio.create_task(queue.put(context))
            except Exception as e:
                print(f"Error in memory callback: {e}")
        
        # Subscribe to updates
        memory_manager.subscribe(callback)
        
        try:
            # Send initial state
            initial_context = memory_manager.get_context()
            yield f"data: {json.dumps(initial_context)}\n\n"
            
            # Stream updates
            while True:
                try:
                    # Wait for update with timeout
                    context = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(context)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            # Unsubscribe when client disconnects
            memory_manager.unsubscribe(callback)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

