from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional, Any, List

from app.core.database import get_db
from app.db.models import ChatSession, Query
from app.services.orchestrator import chat_app

router = APIRouter()

class ChatRequest(BaseModel):
    chat_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="The Clerk User ID")
    query_text: str = Field(..., description="User question")
    mode: str = Field(default="AUTO", description="Mode: AUTO, VQA, etc")

    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "uuid-string",
                "user_id": "user_12345", 
                "query_text": "Find ships",
                "mode": "AUTO"
            }
        }

class ChatResponse(BaseModel):
    response_text: str
    mode_used: str
    metadata: Optional[Any] = None

class QueryDetail(BaseModel):
    """Individual Q&A pair"""
    query_text: str
    response_text: Optional[str]
    response_metadata: Optional[Any] # Bounding boxes, etc.
    created_at: datetime

    class Config:
        from_attributes = True

class ChatHistoryResponse(BaseModel):
    """Full Session Transcript"""
    chat_id: str
    summary_context: Optional[str]
    image_url: str
    image_type: str
    transcript: List[QueryDetail]

@router.post(
    "/orchestration/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK
)
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.chat_id == req.chat_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    # Validation uses consistent names now
    if session.user_id != req.user_id:
        raise HTTPException(status_code=403, detail="Permission denied.")

    initial_state = {
        "chat_id": req.chat_id,
        "user_id": req.user_id, # Consistent
        "query_text": req.query_text,
        "requested_mode": req.mode,
        "image_url": session.image_url,
        "image_type": session.image_type,
        "summary_context": session.summary_context or ""
    }

    try:
        result = await chat_app.ainvoke(initial_state)
        return ChatResponse(
            response_text=result["choices"][0].content,
            mode_used=result["final_mode"],
            metadata=result.get("response_metadata")
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Processing Error")
    
@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    summary="Get full history of a specific chat session"
)
def get_chat_history(chat_id: str, db: Session = Depends(get_db)):
    """
    Fetches the session summary and all Q&A messages for a chat_id.
    """
    # 1. Fetch Session
    session = db.query(ChatSession).filter(ChatSession.chat_id == chat_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # 2. Fetch Queries (Chronological Order)
    queries = db.query(Query)\
        .filter(Query.chat_id == chat_id)\
        .order_by(Query.created_at.asc())\
        .all()

    # 3. Construct Response
    return ChatHistoryResponse(
        chat_id=session.chat_id,
        summary_context=session.summary_context,
        image_url=session.image_url,
        image_type=session.image_type,
        transcript=queries # Pydantic will auto-map the list of Query objects
    )