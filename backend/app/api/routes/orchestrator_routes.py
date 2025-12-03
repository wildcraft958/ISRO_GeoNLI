"""
FastAPI routes for the multimodal chatbot orchestrator.
**NOTE: All endpoints return hardcoded responses for testing/demo purposes.**

Endpoints:
    POST /orchestrator/chat - Main chat endpoint with modality detection and IR2RGB
    POST /orchestrator/ir2rgb - Standalone IR2RGB conversion endpoint
    GET /orchestrator/modality/status - Check modality detection service status
    GET /orchestrator/session/{session_id}/history - Get session history
    GET /orchestrator/sessions - Get user sessions
    GET /orchestrator/sessions/{session_id}/messages - Get session messages
    POST /orchestrator/sessions/{session_id}/summarize - Summarize conversation
    DELETE /orchestrator/session/{session_id} - Clear session
"""

import logging
import uuid
from typing import Any, Optional
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_db_dep
from app.schemas.orchestrator_schema import (
    ChatRequest,
    ChatResponse,
    IR2RGBRequest,
    IR2RGBResponse,
)

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
    **HARDCODED RESPONSE - Demo/Testing Mode**
    """
    try:
        session_id = req.session_id or str(uuid.uuid4())
        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        
        logger.info(
            f"[HARDCODED] Processing chat: session={session_id}, "
            f"mode={req.mode}, user={req.user_id}"
        )
        
        # Determine response based on mode
        if req.mode == "captioning":
            response = ChatResponse(
                session_id=session_id,
                response_type="caption",
                content="A satellite image showing urban development with buildings, roads, and green spaces visible from above.",
                execution_log=[
                    "Received captioning request",
                    "Image preprocessing complete",
                    "Running caption generation model",
                    "Caption generated successfully"
                ],
                message_id=message_id,
                detected_modality="rgb",
                modality_confidence=0.94,
                resnet_classification_used=False,
                vqa_type=None,
                vqa_type_confidence=None,
                converted_image_url=None,
                original_image_url=req.image_url,
                buffer_token_count=165,
                buffer_summarized=False
            )
        
        elif req.mode == "vqa":
            query_lower = (req.query or "").lower()
            
            # Generate contextual answer based on query keywords
            if "count" in query_lower or "how many" in query_lower:
                answer = "There are approximately 15 buildings visible in the image."
                vqa_type = "counting"
            elif "color" in query_lower or "what color" in query_lower:
                answer = "The predominant colors are green (vegetation), gray (roads/buildings), and blue (water bodies)."
                vqa_type = "color_recognition"
            elif "where" in query_lower or "location" in query_lower:
                answer = "This appears to be an urban area with mixed residential and commercial development, likely in a suburban setting."
                vqa_type = "spatial_reasoning"
            else:
                answer = f"Based on the image analysis: {req.query or 'The scene shows a typical urban landscape with infrastructure and natural features.'}"
                vqa_type = "scene_understanding"
            
            response = ChatResponse(
                session_id=session_id,
                response_type="answer",
                content=answer,
                execution_log=[
                    "Received VQA request",
                    f"Query: {req.query or 'generic question'}",
                    f"Classified as: {vqa_type}",
                    "Processing image-query pair",
                    "Answer generated successfully"
                ],
                message_id=message_id,
                detected_modality="rgb",
                modality_confidence=0.91,
                resnet_classification_used=False,
                vqa_type=vqa_type,
                vqa_type_confidence=0.87,
                converted_image_url=None,
                original_image_url=req.image_url,
                buffer_token_count=245,
                buffer_summarized=False
            )
        
        elif req.mode == "grounding":
            query = req.query or "detect objects"
            response = ChatResponse(
                session_id=session_id,
                response_type="boxes",
                content={
                    "boxes": [
                        {"x1": 120, "y1": 80, "x2": 280, "y2": 240, "label": "building", "confidence": 0.92},
                        {"x1": 350, "y1": 150, "x2": 480, "y2": 280, "label": "tree", "confidence": 0.85},
                        {"x1": 500, "y1": 300, "x2": 620, "y2": 420, "label": "vehicle", "confidence": 0.78}
                    ],
                    "image_width": 800,
                    "image_height": 600,
                    "query": query
                },
                execution_log=[
                    "Received grounding request",
                    f"Target query: {query}",
                    "Running object detection",
                    "Found 3 matching objects",
                    "Bounding boxes computed"
                ],
                message_id=message_id,
                detected_modality="rgb",
                modality_confidence=0.96,
                resnet_classification_used=False,
                vqa_type=None,
                vqa_type_confidence=None,
                converted_image_url=None,
                original_image_url=req.image_url,
                buffer_token_count=190,
                buffer_summarized=False
            )
        
        else:  # auto mode
            # Simulate intelligent routing
            if req.query:
                if any(word in req.query.lower() for word in ["where", "find", "locate", "detect"]):
                    response_type = "boxes"
                    content = {
                        "boxes": [
                            {"x1": 100, "y1": 100, "x2": 300, "y2": 300, "label": "region_of_interest", "confidence": 0.88}
                        ],
                        "image_width": 640,
                        "image_height": 480
                    }
                    vqa_type = None
                else:
                    response_type = "answer"
                    content = f"Analyzing your query '{req.query}': The image shows relevant features that address your question."
                    vqa_type = "scene_understanding"
            else:
                response_type = "caption"
                content = "A remote sensing image capturing landscape features from an aerial perspective."
                vqa_type = None
            
            response = ChatResponse(
                session_id=session_id,
                response_type=response_type,
                content=content,
                execution_log=[
                    "Auto mode: analyzing request",
                    f"Query present: {bool(req.query)}",
                    f"Routed to: {response_type}",
                    "Processing complete"
                ],
                message_id=message_id,
                detected_modality="rgb",
                modality_confidence=0.93,
                resnet_classification_used=False,
                vqa_type=vqa_type,
                vqa_type_confidence=0.85 if vqa_type else None,
                converted_image_url=None,
                original_image_url=req.image_url,
                buffer_token_count=175,
                buffer_summarized=False
            )
        
        logger.info(
            f"[HARDCODED] Chat completed: session={session_id}, "
            f"type={response.response_type}, message_id={message_id}"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ir2rgb", response_model=IR2RGBResponse)
async def ir2rgb_endpoint(req: IR2RGBRequest):
    """
    Standalone IR2RGB conversion endpoint.
    **HARDCODED RESPONSE - Demo/Testing Mode**
    """
    try:
        logger.info(
            f"[HARDCODED] IR2RGB request: channels={req.channels}, "
            f"synthesize={req.synthesize_channel}"
        )
        
        # Simulate successful conversion
        mock_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        response = IR2RGBResponse(
            success=True,
            rgb_image_url=f"data:image/png;base64,{mock_base64}",
            rgb_image_base64=mock_base64,
            format="png",
            dimensions={"width": 800, "height": 600},
            size_bytes=15240,
        )
        
        logger.info("[HARDCODED] IR2RGB conversion successful")
        return response
    
    except Exception as e:
        logger.error(f"IR2RGB endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ir2rgb/status")
async def ir2rgb_status():
    """
    Check if IR2RGB service is available.
    **HARDCODED RESPONSE - Always returns available**
    """
    return {
        "available": True,
        "message": "IR2RGB service is available (hardcoded response)"
    }


@router.get("/modality/status")
async def modality_detection_status():
    """
    Check if modality detection service is available.
    **HARDCODED RESPONSE - Always returns available**
    """
    return {
        "available": True,
        "statistical_detection": {
            "available": True,
            "message": "Statistical detection available (hardcoded)",
        },
        "resnet_classifier": {
            "available": True,
            "message": "ResNet classifier service is available (hardcoded)",
        },
        "features": {
            "metadata_parsing": True,
            "sar_detection": True,
            "infrared_detection": True,
            "alpha_channel_detection": True,
            "resnet_fallback": True,
        }
    }


@router.post("/router/test")
async def test_task_router(
    query: str,
    detected_modality: Optional[str] = None,
    has_image: bool = True
):
    """
    Test the task router with a query.
    **HARDCODED RESPONSE - Demo/Testing Mode**
    """
    try:
        # Simulate task classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["caption", "describe", "what is"]):
            task_type = "captioning"
            confidence = 0.92
            reasoning = "Query requests description of image content"
        elif any(word in query_lower for word in ["find", "locate", "where", "detect"]):
            task_type = "grounding"
            confidence = 0.88
            reasoning = "Query requests spatial localization"
        elif has_image and len(query) > 5:
            task_type = "vqa"
            confidence = 0.85
            reasoning = "Query asks question about image"
        else:
            task_type = "captioning"
            confidence = 0.75
            reasoning = "Default to captioning"
        
        return {
            "success": True,
            "query": query,
            "detected_modality": detected_modality,
            "task_type": task_type,
            "confidence": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"Task router test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vqa-classifier/test")
async def test_vqa_classifier(
    query: str,
    detected_modality: Optional[str] = None
):
    """
    Test the VQA sub-classifier with a query.
    **HARDCODED RESPONSE - Demo/Testing Mode**
    """
    try:
        query_lower = query.lower()
        
        # Classify VQA type
        if any(word in query_lower for word in ["how many", "count", "number of"]):
            vqa_type = "counting"
            confidence = 0.94
            reasoning = "Query explicitly requests counting"
        elif any(word in query_lower for word in ["color", "what color"]):
            vqa_type = "color_recognition"
            confidence = 0.91
            reasoning = "Query asks about colors"
        elif any(word in query_lower for word in ["where", "location", "position"]):
            vqa_type = "spatial_reasoning"
            confidence = 0.88
            reasoning = "Query involves spatial relationships"
        elif any(word in query_lower for word in ["what", "which", "identify"]):
            vqa_type = "object_recognition"
            confidence = 0.86
            reasoning = "Query requests object identification"
        else:
            vqa_type = "scene_understanding"
            confidence = 0.82
            reasoning = "General scene understanding question"
        
        return {
            "success": True,
            "query": query,
            "detected_modality": detected_modality,
            "vqa_type": vqa_type,
            "confidence": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"VQA classifier test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/status")
async def llm_service_status():
    """
    Check if LLM service is available.
    **HARDCODED RESPONSE - Always returns available**
    """
    return {
        "available": True,
        "message": "LLM service is available (hardcoded response)",
        "base_url": "http://mock-llm-service:8000"
    }


@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Retrieve full session history from persistent storage.
    **HARDCODED RESPONSE - Returns mock history**
    """
    mock_messages = [
        {
            "role": "user",
            "content": "Can you analyze this satellite image?",
            "image_url": "https://example.com/image1.jpg",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        {
            "role": "assistant",
            "content": "This is a satellite image showing urban development with buildings and infrastructure.",
            "response_type": "caption",
            "timestamp": "2024-01-15T10:30:02Z"
        },
        {
            "role": "user",
            "content": "How many buildings can you see?",
            "timestamp": "2024-01-15T10:31:00Z"
        },
        {
            "role": "assistant",
            "content": "I can identify approximately 12 buildings in this image.",
            "response_type": "answer",
            "timestamp": "2024-01-15T10:31:03Z"
        }
    ]
    
    return {
        "session_id": session_id,
        "messages": mock_messages,
        "message_count": len(mock_messages)
    }


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Clear session and all its messages from persistent storage.
    **HARDCODED RESPONSE - Always returns success**
    """
    logger.info(f"[HARDCODED] Clearing session: {session_id}")
    
    return {
        "session_id": session_id,
        "message": "Session and all messages cleared successfully (hardcoded)"
    }


@router.get("/sessions")
async def get_user_sessions(
    user_id: str,
    limit: Optional[int] = None,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Get all sessions for a user.
    **HARDCODED RESPONSE - Returns mock sessions**
    """
    mock_sessions = [
        {
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "created_at": "2024-01-15T09:00:00Z",
            "updated_at": "2024-01-15T09:45:00Z",
            "message_count": 8,
            "context": {"last_image_url": "https://example.com/image1.jpg"}
        },
        {
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "created_at": "2024-01-14T14:20:00Z",
            "updated_at": "2024-01-14T15:10:00Z",
            "message_count": 12,
            "context": {"last_image_url": "https://example.com/image2.jpg"}
        },
        {
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "created_at": "2024-01-13T11:30:00Z",
            "updated_at": "2024-01-13T12:00:00Z",
            "message_count": 5,
            "context": {"last_image_url": "https://example.com/image3.jpg"}
        }
    ]
    
    limited_sessions = mock_sessions[:limit] if limit else mock_sessions
    
    return {
        "user_id": user_id,
        "sessions": limited_sessions,
        "count": len(limited_sessions)
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
    **HARDCODED RESPONSE - Returns mock messages**
    """
    mock_messages = [
        {
            "id": f"msg_{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Mock message {i}",
            "session_id": session_id,
            "timestamp": f"2024-01-15T10:{30+i}:00Z"
        }
        for i in range(10)
    ]
    
    # Apply skip and limit
    paginated = mock_messages[skip:]
    if limit:
        paginated = paginated[:limit]
    
    return {
        "session_id": session_id,
        "messages": paginated,
        "count": len(paginated)
    }


@router.post("/sessions/{session_id}/summarize")
async def summarize_session(
    session_id: str,
    max_messages: int = 50,
    db: AsyncIOMotorDatabase = Depends(get_db_dep)
):
    """
    Generate a conversation summary for a session.
    **HARDCODED RESPONSE - Returns mock summary**
    """
    mock_summary = (
        f"This session involved analysis of satellite imagery. "
        f"The user requested image captioning, object detection, and spatial analysis. "
        f"Key topics included: urban development, building identification, and vegetation coverage. "
        f"The conversation contained {max_messages} messages with multiple image analyses performed."
    )
    
    return {
        "session_id": session_id,
        "summary": mock_summary,
        "message": "Conversation summarized successfully (hardcoded)"
    }