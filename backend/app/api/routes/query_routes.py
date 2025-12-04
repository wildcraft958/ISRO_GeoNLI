from app.api.deps import get_db_dep
from app.schemas.query_schema import QueryCreate, QueryPublic, QueryUpdate
from app.services import query_service
from fastapi import APIRouter, Depends, Response, status, HTTPException
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


@router.put("/{query_id}", response_model=QueryPublic)
async def update_query_response_endpoint(
    query_id: str,
    payload: QueryUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
):
    """
    Update the response field of a query.
    
    Args:
        query_id: The ID of the query to update
        payload: The update data containing the response
    
    Returns:
        The updated query
    """
    updated_query = await query_service.update_query_response(db, query_id, payload.response)
    if not updated_query:
        raise HTTPException(status_code=404, detail="Query not found")
    return updated_query


