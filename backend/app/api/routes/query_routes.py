from app.api.deps import get_db_dep
from app.schemas.query_schema import QueryCreate,QueryPublic 
from app.services import query_service
from fastapi import APIRouter, Depends, Response, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, List

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/create", response_model=QueryPublic, status_code=status.HTTP_201_CREATED)
async def create_chat_endpoint(
    payload: QueryCreate,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
    response: Response = None,
):
    chat = await query_service.create_query(db, payload)

    if response:
        response.status_code = status.HTTP_201_CREATED
    return chat 


@router.get("/chat/{chat_id}", response_model=List[QueryPublic])
async def get_queries_by_chat_endpoint(
    chat_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
):
    """
    Get all queries for a specific chat.
    
    Args:
        chat_id: The ID of the chat
    
    Returns:
        List of queries for the specified chat
    """
    queries = await query_service.get_queries_by_chat_id(db, chat_id)
    return queries


@router.get("/user/{user_id}", response_model=Dict[str, List[QueryPublic]])
async def get_queries_by_user_endpoint(
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
):
    """
    Get all queries for all chats belonging to a user, grouped by chat_id.
    
    Args:
        user_id: The ID of the user
    
    Returns:
        Dictionary mapping chat_id to list of queries for that chat
    """
    queries_by_chat = await query_service.get_queries_by_user(db, user_id)
    return queries_by_chat


