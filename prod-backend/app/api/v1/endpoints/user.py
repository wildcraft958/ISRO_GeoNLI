from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List
from app.core.database import get_db
from app.db.models import User, ChatSession

router = APIRouter()

# --- Pydantic Models ---

class UserCreate(BaseModel):
    """Request model for creating a new user"""
    # RENAMED to user_id
    user_id: str = Field(
        ..., 
        description="Unique identifier from Clerk (e.g. user_2pIx...)",
        example="user_12345",
        min_length=1,
        max_length=255
    )
    email: EmailStr = Field(..., example="user@example.com")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "email": "user@example.com"
            }
        }

class UserResponse(BaseModel):
    id: str = Field(..., description="Internal UUID")
    # RENAMED to user_id
    user_id: str = Field(..., description="Clerk User ID")
    email: EmailStr
    created_at: datetime 

    class Config:
        from_attributes = True

# --- New Response Model for History ---
class UserSessionSummary(BaseModel):
    chat_id: str
    image_url: str
    image_type: str
    created_at: datetime
    summary_preview: Optional[str] = None

    class Config:
        from_attributes = True

# --- Endpoint ---

@router.post(
    "/", 
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user"
)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # 1. Check by ID (Updated to user_id)
    existing_user_by_id = db.query(User).filter(User.user_id == user.user_id).first()
    if existing_user_by_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this user_id already exists"
        )

    # 2. Check by Email
    existing_user_by_email = db.query(User).filter(User.email == user.email).first()
    if existing_user_by_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    # 3. Create (Updated to user_id)
    db_user = User(
        user_id=user.user_id,
        email=user.email
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user

@router.get(
    "/{user_id}/history", 
    response_model=List[UserSessionSummary],
    summary="Get all chat sessions for a user"
)
def get_user_history(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieves a list of all conversation sessions owned by the user.
    """
    # 1. Validate User exists (Optional, but good practice)
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Fetch Sessions (ordered by newest first)
    sessions = db.query(ChatSession)\
        .filter(ChatSession.user_id == user_id)\
        .order_by(ChatSession.created_at.desc())\
        .all()

    # 3. Map to Response
    # Pydantic's from_attributes=True handles the mapping, 
    # but we can manually map summary_context to summary_preview
    results = []
    for s in sessions:
        results.append(UserSessionSummary(
            chat_id=s.chat_id,
            image_url=s.image_url,
            image_type=s.image_type,
            created_at=s.created_at,
            # Just show first 100 chars of summary for the list view
            summary_preview=s.summary_context[:100] + "..." if s.summary_context else ""
        ))
        
    return results