from app.api.deps import get_db_dep
from app.schemas.chat_schema import ChatCreate, ChatPublic
from app.services import chat_service
from fastapi import APIRouter, Depends, HTTPException, Response, status
from motor.motor_asyncio import AsyncIOMotorDatabase

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/create", response_model=ChatPublic, status_code=status.HTTP_201_CREATED)
async def create_chat_endpoint(
    payload: ChatCreate,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
    response: Response = None,
):
    chat = await chat_service.create_chat(db, payload)

    if response:
        response.status_code = status.HTTP_201_CREATED
    return chat 


@router.get("/all", response_model=list[ChatPublic])
async def get_user_chats(
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
):
    chats = await chat_service.get_chats_by_user(db, user_id)
    return chats

# for deletioin of chat
@router.delete("/delete/{chat_id}", status_code=status.HTTP_200_OK)
async def delete_chat_endpoint(
    chat_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
):
    """
    Delete a chat by its ID.
    
    Args:
        chat_id: The ID of the chat to delete
    
    Returns:
        Success message with deleted chat ID
    """
    success = await chat_service.delete_chat(db, chat_id)
    
    if success:
        return {
            "chat_id": chat_id,
            "message": "Chat deleted successfully"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Chat {chat_id} not found or could not be deleted"
        )
