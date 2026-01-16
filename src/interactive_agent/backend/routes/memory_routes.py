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

from agent.backend.explicit_memory import get_memory_manager


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
        # Direct file access to avoid potential deadlocks with memory manager
        import json
        from pathlib import Path
        
        memory_file = Path("data/interactive_agent/agent_memory.json")
        
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                long_term = data.get("long_term", [])
        else:
            long_term = []
        
        return MemoryResponse(
            short_term=[],  # Short-term memory is now handled by LangChain sessions
            long_term=long_term
        )
    except Exception as e:
        print(f"Error in get_memory endpoint: {e}")
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
        # Direct file access to avoid potential deadlocks with memory manager
        import json
        from pathlib import Path
        
        memory_file = Path("data/interactive_agent/agent_memory.json")
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                long_term = data.get("long_term", [])
        else:
            long_term = []
        
        # Handle forget requests
        if request.update.lower().startswith("forget"):
            to_forget = request.update[7:].strip()
            original_count = len(long_term)
            long_term = [item for item in long_term if to_forget.lower() not in item.lower()]
            removed_count = original_count - len(long_term)
            
            # Save updated memory
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "long_term": long_term,
                    "last_updated": "2025-01-08T00:00:00.000000"
                }, f, indent=2, ensure_ascii=False)
            
            # Try to notify memory manager subscribers (non-blocking)
            try:
                memory_manager = get_memory_manager()
                memory_manager._notify_subscribers()
            except Exception as e:
                print(f"Warning: Could not notify memory manager: {e}")
            
            return {
                "status": "success",
                "message": f"Removed {removed_count} fact(s) matching '{to_forget}'",
                "memory": {"short_term": [], "long_term": long_term}
            }
        else:
            # Add new memory (avoid duplicates)
            if request.update not in long_term:
                long_term.append(request.update)
                
                # Save updated memory
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "long_term": long_term,
                        "last_updated": "2025-01-08T00:00:00.000000"
                    }, f, indent=2, ensure_ascii=False)
                
                # Try to notify memory manager subscribers (non-blocking)
                try:
                    memory_manager = get_memory_manager()
                    memory_manager._notify_subscribers()
                except Exception as e:
                    print(f"Warning: Could not notify memory manager: {e}")
                
                return {
                    "status": "success",
                    "message": f"Saved '{request.update}' to long-term memory",
                    "memory": {"short_term": [], "long_term": long_term}
                }
            else:
                return {
                    "status": "success",
                    "message": f"Memory already exists: '{request.update}'",
                    "memory": {"short_term": [], "long_term": long_term}
                }
        
    except Exception as e:
        print(f"Error in update_memory endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("")
async def clear_memory():
    """
    Clear all memory (both short-term and long-term).
    
    Returns:
        Success message
    """
    try:
        # Direct file access to avoid potential deadlocks with memory manager
        import json
        from pathlib import Path
        
        memory_file = Path("data/interactive_agent/agent_memory.json")
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear the memory file directly
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump({
                "long_term": [],
                "last_updated": "2025-01-08T00:00:00.000000"
            }, f, indent=2, ensure_ascii=False)
        
        # Try to notify memory manager subscribers (non-blocking)
        try:
            memory_manager = get_memory_manager()
            memory_manager._notify_subscribers()
        except Exception as e:
            print(f"Warning: Could not notify memory manager: {e}")
        
        return {
            "status": "success",
            "message": "All memory cleared"
        }
    except Exception as e:
        print(f"Error in clear_memory endpoint: {e}")
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
                # Use asyncio.create_task safely
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(queue.put(context))
                else:
                    # If no event loop is running, just log the update
                    print(f"Memory update (no event loop): {context}")
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

