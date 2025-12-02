from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class UserMemoryCreate(BaseModel):
    """Schema for creating user memory."""
    user_id: str = Field(..., description="User identifier")
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="User preferences (default_mode, ir2rgb_defaults, etc.)"
    )


class UserMemoryUpdate(BaseModel):
    """Schema for updating user memory."""
    preferences: Optional[Dict[str, Any]] = Field(None, description="Updated preferences")
    conversation_summaries: Optional[List[str]] = Field(
        None, 
        description="List of conversation summaries"
    )
    frequent_queries: Optional[List[str]] = Field(
        None,
        description="List of frequently used queries"
    )


class UserMemoryInDB(BaseModel):
    """Schema for user memory stored in database."""
    id: str
    user_id: str
    preferences: Dict[str, Any]
    conversation_summaries: List[str]
    frequent_queries: List[str]
    created_at: datetime
    updated_at: datetime


class UserMemoryPublic(UserMemoryInDB):
    """Public schema for user memory responses."""
    pass

