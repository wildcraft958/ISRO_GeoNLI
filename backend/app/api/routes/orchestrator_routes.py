"""
FastAPI routes for the multimodal chatbot orchestrator.

Endpoints:
    POST /orchestrator/chat - Main chat endpoint with optional IR2RGB preprocessing
    POST /orchestrator/ir2rgb - Standalone IR2RGB conversion endpoint
    GET /orchestrator/session/{session_id}/history - Get session history
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.schemas.orchestrator_schema import (
    ChatRequest, 
    ChatResponse,
    IR2RGBRequest,
    IR2RGBResponse,
)
from app.orchestrator import app
from app.utils.session_manager import SessionManager
from app.services.ir2rgb_service import get_ir2rgb_service, is_ir2rgb_available

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])
session_manager = SessionManager(app)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for multimodal chatbot.
    
    Accepts:
        - image_url: URL of image to process
        - query: Optional text query
        - mode: "auto", "grounding", "vqa", or "captioning"
        - session_id: Optional session ID for conversation persistence
        - needs_ir2rgb: Whether to convert multispectral image to RGB
        - ir2rgb_channels: Channel order for IR2RGB (e.g., ["NIR", "R", "G"])
        - ir2rgb_synthesize: Which channel to synthesize ("R", "G", or "B")
    
    Returns:
        ChatResponse with response content and execution log
    """
    try:
        # Generate session ID if not provided
        session_id = req.session_id or str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Get existing session state if available
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
        
        logger.info(f"Processing chat request: session={session_id}, mode={req.mode}, ir2rgb={req.needs_ir2rgb}")
        
        # Invoke workflow
        result = app.invoke(inputs, config=config)
        
        # Determine response type
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
        
        # Update message history in background
        background_tasks.add_task(
            update_message_history,
            session_id,
            req.query,
            response_type,
            content
        )
        
        # Build response
        response = ChatResponse(
            session_id=session_id,
            response_type=response_type,
            content=content,
            execution_log=result.get("execution_log", []),
            converted_image_url=result.get("image_url") if req.needs_ir2rgb else None,
            original_image_url=result.get("original_image_url"),
        )
        
        logger.info(f"Chat request completed: session={session_id}, type={response_type}")
        
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
async def get_session_history(session_id: str):
    """
    Retrieve full session history.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session history with messages
    """
    history = session_manager.get_session_history(session_id)
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history)
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session history (for testing/debugging).
    
    Args:
        session_id: Session identifier
    
    Returns:
        Confirmation message
    """
    # Note: This would require implementing a delete method in SessionManager
    # For now, just return a success message
    return {
        "session_id": session_id,
        "message": "Session cleared (note: full implementation pending)"
    }


async def update_message_history(
    session_id: str, 
    query: Optional[str], 
    response_type: str, 
    content
):
    """
    Background task to update message history.
    
    This is called after the response is sent to the client,
    so it doesn't block the response.
    """
    try:
        # Log the interaction (actual persistence happens via LangGraph checkpoints)
        logger.debug(
            f"Message history update: session={session_id}, "
            f"query={query[:50] if query else None}..., type={response_type}"
        )
    except Exception as e:
        logger.error(f"Failed to update message history: {e}")
