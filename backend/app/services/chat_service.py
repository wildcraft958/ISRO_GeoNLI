from typing import Optional

from app.models.chat import chat_to_public
from app.schemas.chat_schema import ChatCreate, ChatPublic
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

COLLECTION_NAME="chats"

async def create_chat(db: AsyncIOMotorDatabase, data: ChatCreate) -> ChatPublic:
    doc = data.model_dump()

    try:
        result = await db[COLLECTION_NAME].insert_one(doc)
    except DuplicateKeyError:
        existing = await db[COLLECTION_NAME].find_one({"image_url": data.image_url})
        return ChatPublic(**chat_to_public(existing))

    created = await db[COLLECTION_NAME].find_one({"_id": result.inserted_id})
    return ChatPublic(**chat_to_public(created))

async def get_chats_by_user(db, user_id: str):
    cursor = db["chats"].find({"user_id": user_id})
    return [ChatPublic(**doc) for doc in await cursor.to_list(length=None)]

