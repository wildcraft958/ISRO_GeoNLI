from app.api.deps import get_db_dep
from app.schemas.chat_schema import ChatCreate, ChatPublic
from app.services import chat_service
from fastapi import APIRouter, Depends, Response, status
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

