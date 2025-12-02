from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# Type definitions for strict validation
MessageRole = Literal["user", "assistant"]
ResponseType = Literal["caption", "answer", "boxes"]


class MessageCreate(BaseModel):
    """Schema for creating a new message."""
    role: MessageRole = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    image_url: Optional[str] = Field(None, description="URL of associated image")
    response_type: Optional[ResponseType] = Field(
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
    role: MessageRole
    content: str
    image_url: Optional[str]
    response_type: Optional[ResponseType]
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    metadata: Dict[str, Any]


class MessagePublic(MessageInDB):
    """Public schema for message responses."""
    pass

