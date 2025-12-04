from datetime import datetime, timezone
from typing import Optional

from app.models.chat import chat_to_public
from app.schemas.chat_schema import ChatCreate, ChatPublic
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

COLLECTION_NAME="chats"

async def create_chat(db: AsyncIOMotorDatabase, data: ChatCreate) -> ChatPublic:
    doc = data.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)

    try:
        result = await db[COLLECTION_NAME].insert_one(doc)
    except DuplicateKeyError:
        existing = await db[COLLECTION_NAME].find_one({"image_url": data.image_url})
        return ChatPublic(**chat_to_public(existing))

    created = await db[COLLECTION_NAME].find_one({"_id": result.inserted_id})
    return ChatPublic(**chat_to_public(created))

async def get_chats_by_user(db, user_id: str):
    cursor = db["chats"].find({"user_id": user_id})
    docs = await cursor.to_list(length=None)
    return [ChatPublic(**chat_to_public(doc)) for doc in docs]

async def delete_chat(db: AsyncIOMotorDatabase, chat_id: str) -> bool:
    """Delete a chat by its ID and all associated queries."""
    from bson import ObjectId
    
    try:
        # First, delete all queries associated with this chat
        queries_deleted = await db["queries"].delete_many({"chat_id": chat_id})
        
        # Then delete the chat itself
        # Try to delete by ObjectId if chat_id is a valid ObjectId
        if ObjectId.is_valid(chat_id):
            result = await db[COLLECTION_NAME].delete_one({"_id": ObjectId(chat_id)})
        else:
            # Fallback: try to delete by string id
            result = await db[COLLECTION_NAME].delete_one({"id": chat_id})
        
        # Return True if chat was deleted (queries deletion is a bonus)
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error deleting chat {chat_id}: {e}")
        return False

