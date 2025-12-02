from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Schema for creating a new session."""
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier (optional)")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Session context (conversation_summary, user_preferences, execution_stats)"
    )


class SessionUpdate(BaseModel):
    """Schema for updating a session."""
    context: Optional[Dict[str, Any]] = Field(None, description="Updated session context")
    summary: Optional[str] = Field(None, description="Conversation summary")
    message_count: Optional[int] = Field(None, description="Updated message count")


class SessionInDB(BaseModel):
    """Schema for session stored in database."""
    id: str
    session_id: str
    user_id: Optional[str]
    context: Dict[str, Any]
    summary: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_activity: datetime


class SessionPublic(SessionInDB):
    """Public schema for session responses."""
    pass

