"""
FastAPI routes for the multimodal chatbot orchestrator.

Endpoints:
    POST /orchestrator/chat - Main chat endpoint with optional IR2RGB preprocessing
    POST /orchestrator/ir2rgb - Standalone IR2RGB conversion endpoint
    GET /orchestrator/session/{session_id}/history - Get session history
    GET /orchestrator/sessions - Get user sessions
    GET /orchestrator/sessions/{session_id}/messages - Get session messages
    POST /orchestrator/sessions/{session_id}/summarize - Summarize conversation
    DELETE /orchestrator/session/{session_id} - Clear session
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from app.schemas.orchestrator_schema import (
    ChatRequest, 
    ChatResponse,
    IR2RGBRequest,
    IR2RGBResponse,
)
from app.schemas.message_schema import MessageCreate
from app.schemas.session_schema import SessionCreate, SessionUpdate
from app.schemas.user_memory_schema import UserMemoryUpdate
from app.orchestrator import app, integrate_user_memory
from app.utils.session_manager import SessionManager
from app.services.ir2rgb_service import get_ir2rgb_service, is_ir2rgb_available
from app.services import memory_service
from app.api.deps import get_db_dep
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Main chat endpoint for multimodal chatbot.
    
    Accepts:
        - image_url: URL of image to process
        - query: Optional text query
        - mode: "auto", "grounding", "vqa", or "captioning"
        - session_id: Optional session ID for conversation persistence
        - user_id: Optional user ID for user memory integration
        - needs_ir2rgb: Whether to convert multispectral image to RGB
        - ir2rgb_channels: Channel order for IR2RGB (e.g., ["NIR", "R", "G"])
        - ir2rgb_synthesize: Which channel to synthesize ("R", "G", or "B")
    
    Returns:
        ChatResponse with response content, execution log, and message_id
    """
    try:
        # Initialize session manager with database
        session_manager = SessionManager(app, db)
        
        # Generate session ID if not provided
        session_id = req.session_id or str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Get or create session
        session = await memory_service.get_session(db, session_id)
        if not session:
            session_data = SessionCreate(
                session_id=session_id,
                user_id=req.user_id
            )
            session = await memory_service.create_session(db, session_data)
        
        # Load user memory if user_id provided
        user_memory = None
        if req.user_id:
            user_memory = await memory_service.get_user_memory(db, req.user_id)
            if not user_memory:
                # Create default user memory
                from app.schemas.user_memory_schema import UserMemoryCreate
                user_memory = await memory_service.create_user_memory(
                    db,
                    UserMemoryCreate(user_id=req.user_id)
                )
        
        # Get existing session state from LangGraph
        existing_state = session_manager.get_session_state(session_id)
        
        # Prepare inputs for LangGraph
        inputs = {
            "session_id": session_id,
            "image_url": req.image_url,
            "user_query": req.query,
            "mode": req.mode,
            "needs_ir2rgb": req.needs_ir2rgb,
            "ir2rgb_channels": req.ir2rgb_channels,
            "ir2rgb_synthesize": req.ir2rgb_synthesize,
            "original_image_url": None,
            "messages": existing_state.get("messages", []) if existing_state else [],
            "session_context": existing_state.get("session_context", {}) if existing_state else {},
            "caption_result": None,
            "vqa_result": None,
            "grounding_result": None,
            "execution_log": [],
        }
        
        # Integrate user memory into context
        if user_memory:
            memory_update = integrate_user_memory(inputs, user_memory.model_dump())
            inputs.update(memory_update)
        
        logger.info(f"Processing chat request: session={session_id}, user={req.user_id}, mode={req.mode}, ir2rgb={req.needs_ir2rgb}")
        
        # Invoke workflow
        result = app.invoke(inputs, config=config)
        
        # Determine response type and content
        response_type = "unknown"
        content = None
        
        if req.mode == "captioning":
            response_type = "caption"
            content = result.get("caption_result", "")
        elif req.mode == "vqa":
            response_type = "answer"
            content = result.get("vqa_result", "")
        elif req.mode == "grounding":
            response_type = "boxes"
            content = result.get("grounding_result", {})
        else:  # auto mode - check which result is populated
            if result.get("caption_result"):
                response_type = "caption"
                content = result["caption_result"]
            elif result.get("vqa_result"):
                response_type = "answer"
                content = result["vqa_result"]
            elif result.get("grounding_result"):
                response_type = "boxes"
                content = result["grounding_result"]
        
        # Get the last assistant message from result
        messages = result.get("messages", [])
        assistant_message = None
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    assistant_message = msg
                    break
        
        # Persist messages and update session in background
        background_tasks.add_task(
            persist_messages_and_update_session,
            db,
            session_id,
            req.user_id,
            req.image_url,
            req.query,
            messages,
            result.get("execution_log", []),
            response_type,
            content
        )
        
        # Extract message_id if available
        message_id = None
        if assistant_message and "id" in assistant_message:
            message_id = assistant_message["id"]
        
        # Build response
        response = ChatResponse(
            session_id=session_id,
            response_type=response_type,
            content=content,
            execution_log=result.get("execution_log", []),
            converted_image_url=result.get("image_url") if req.needs_ir2rgb else None,
            original_image_url=result.get("original_image_url"),
            message_id=message_id,
        )
        
        logger.info(f"Chat request completed: session={session_id}, type={response_type}, message_id={message_id}")
        
        return response
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ir2rgb", response_model=IR2RGBResponse)
async def ir2rgb_endpoint(req: IR2RGBRequest):
    """
    Standalone IR2RGB conversion endpoint.
    
    Converts multispectral/FCC satellite images to RGB format
    using pre-trained RPCC weights and LUT.
    
    Args:
        image_url: URL of multispectral image
        channels: Channel order (e.g., ["NIR", "R", "G"])
        synthesize_channel: Which channel to synthesize ("R", "G", or "B")
    
    Returns:
        IR2RGBResponse with base64 encoded RGB image
    """
    try:
        if not is_ir2rgb_available():
            raise HTTPException(
                status_code=503,
                detail="IR2RGB service is not available (model weights not found)"
            )
        
        logger.info(f"IR2RGB conversion request: channels={req.channels}, synthesize={req.synthesize_channel}")
        
        service = get_ir2rgb_service()
        result = service.convert_from_url(
            req.image_url,
            req.channels,
            req.synthesize_channel
        )
        
        if result.get("success"):
            logger.info(f"IR2RGB conversion successful: {result.get('dimensions')}")
            return IR2RGBResponse(
                success=True,
                rgb_image_url=result.get("rgb_image_url"),
                rgb_image_base64=result.get("rgb_image_base64"),
                format=result.get("format"),
                dimensions=result.get("dimensions"),
                size_bytes=result.get("size_bytes"),
            )
        else:
            logger.warning(f"IR2RGB conversion failed: {result.get('error')}")
            return IR2RGBResponse(
                success=False,
                error=result.get("error", "Unknown error")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"IR2RGB endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ir2rgb/status")
async def ir2rgb_status():
    """
    Check if IR2RGB service is available.
    
    Returns:
        dict with availability status and model path
    """
    available = is_ir2rgb_available()
    return {
        "available": available,
        "message": "IR2RGB service is available" if available else "Model weights not found"
    }


@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Retrieve full session history from persistent storage.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session history with messages
    """
    session_manager = SessionManager(app, db)
    history = await session_manager.get_persistent_messages(session_id)
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history)
    }


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Clear session and all its messages from persistent storage.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Confirmation message
    """
    session_manager = SessionManager(app, db)
    success = await session_manager.clear_session(session_id)
    
    if success:
        return {
            "session_id": session_id,
            "message": "Session and all messages cleared successfully"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or could not be cleared"
        )


@router.get("/sessions")
async def get_user_sessions(
    user_id: str,
    limit: Optional[int] = None,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Get all sessions for a user.
    
    Args:
        user_id: User identifier
        limit: Optional limit on number of sessions to return
    
    Returns:
        List of user sessions
    """
    session_manager = SessionManager(app, db)
    sessions = await session_manager.get_user_sessions(user_id, limit=limit)
    return {
        "user_id": user_id,
        "sessions": sessions,
        "count": len(sessions)
    }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None,
    skip: int = 0,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Get all messages for a session.
    
    Args:
        session_id: Session identifier
        limit: Optional limit on number of messages
        skip: Number of messages to skip (for pagination)
    
    Returns:
        List of messages for the session
    """
    messages = await memory_service.get_session_messages(
        db,
        session_id,
        limit=limit,
        skip=skip
    )
    return {
        "session_id": session_id,
        "messages": [msg.model_dump() for msg in messages],
        "count": len(messages)
    }


@router.post("/sessions/{session_id}/summarize")
async def summarize_session(
    session_id: str,
    max_messages: int = 50,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Generate a conversation summary for a session.
    
    Args:
        session_id: Session identifier
        max_messages: Maximum number of messages to include in summary
    
    Returns:
        Summary text
    """
    summary = await memory_service.summarize_conversation(
        db,
        session_id,
        max_messages=max_messages
    )
    
    if summary:
        return {
            "session_id": session_id,
            "summary": summary,
            "message": "Conversation summarized successfully"
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Could not generate summary for session {session_id}. Session may have too few messages."
        )


async def persist_messages_and_update_session(
    db: AsyncIOMotorDatabase,
    session_id: str,
    user_id: Optional[str],
    image_url: str,
    query: Optional[str],
    messages: list,
    execution_log: list,
    response_type: str,
    content
):
    """
    Background task to persist messages and update session.
    
    This is called after the response is sent to the client,
    so it doesn't block the response.
    """
    try:
        # Get the last user and assistant messages
        user_msg = None
        assistant_msg = None
        
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not assistant_msg:
                assistant_msg = msg
            elif msg.get("role") == "user" and not user_msg:
                user_msg = msg
        
        # Persist user message if exists
        if user_msg:
            user_message_data = MessageCreate(
                role="user",
                content=user_msg.get("content", query or "[Image only]"),
                image_url=user_msg.get("image_url", image_url),
                session_id=session_id,
                user_id=user_id,
                metadata={"execution_log": execution_log}
            )
            await memory_service.save_message(db, user_message_data)
            await memory_service.increment_session_message_count(db, session_id)
        
        # Persist assistant message if exists
        if assistant_msg:
            assistant_message_data = MessageCreate(
                role="assistant",
                content=assistant_msg.get("content", str(content)),
                image_url=assistant_msg.get("image_url"),
                response_type=response_type,
                session_id=session_id,
                user_id=user_id,
                metadata=assistant_msg.get("metadata", {})
            )
            saved_msg = await memory_service.save_message(db, assistant_message_data)
            await memory_service.increment_session_message_count(db, session_id)
            
            # Update assistant message in state with message_id
            assistant_msg["id"] = saved_msg.id
        
        # Update session context with execution stats
        # Get existing session to merge stats
        existing_session = await memory_service.get_session(db, session_id)
        existing_context = existing_session.context if existing_session else {}
        existing_stats = existing_context.get("execution_stats", {})
        
        # Increment execution stats
        current_count = existing_stats.get(response_type, 0)
        session_context_updates = {
            "execution_stats": {
                **existing_stats,
                response_type: current_count + 1
            },
            "last_image_url": image_url,
        }
        
        await memory_service.update_session_context(
            db,
            session_id,
            SessionUpdate(context=session_context_updates)
        )
        
        # Update user memory with frequent queries
        if user_id and query:
            await memory_service.update_user_memory(
                db,
                user_id,
                UserMemoryUpdate(frequent_queries=[query])
            )
        
        logger.debug(
            f"Persisted messages for session={session_id}, "
            f"user={user_id}, type={response_type}"
        )
    
    except Exception as e:
        logger.error(f"Failed to persist messages and update session: {e}", exc_info=True)
