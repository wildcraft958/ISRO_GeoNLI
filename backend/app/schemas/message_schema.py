from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class MessageCreate(BaseModel):
    """Schema for creating a new message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    image_url: Optional[str] = Field(None, description="URL of associated image")
    response_type: Optional[str] = Field(
        None, 
        description="Type of response: 'caption', 'answer', 'boxes', or None"
    )
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier (optional)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (execution_log, grounding_result, etc.)"
    )


class MessageInDB(BaseModel):
    """Schema for message stored in database."""
    id: str
    role: str
    content: str
    image_url: Optional[str]
    response_type: Optional[str]
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    metadata: Dict[str, Any]


class MessagePublic(MessageInDB):
    """Public schema for message responses."""
    pass

