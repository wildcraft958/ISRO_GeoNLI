from typing import Optional, List, Dict, Any
from bson import ObjectId

from app.models.query import query_to_public
from app.schemas.query_schema import QueryCreate, QueryPublic, QueryUpdate 
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

COLLECTION_NAME="queries"

async def create_query(db: AsyncIOMotorDatabase, data: QueryCreate) -> QueryPublic:
    doc = data.model_dump()

    try:
        result = await db[COLLECTION_NAME].insert_one(doc)
    except DuplicateKeyError:
        existing = await db[COLLECTION_NAME].find_one({"parent_id": data.parent_id})
        return QueryPublic(**query_to_public(existing))

    created = await db[COLLECTION_NAME].find_one({"_id": result.inserted_id})
    return QueryPublic(**query_to_public(created))

async def get_queries_by_chat_id(db: AsyncIOMotorDatabase, chat_id: str) -> List[QueryPublic]:
    """Get all queries for a specific chat."""
    cursor = db[COLLECTION_NAME].find({"chat_id": chat_id})
    queries = []
    async for doc in cursor:
        queries.append(QueryPublic(**query_to_public(doc)))
    return queries

async def get_queries_by_user(db: AsyncIOMotorDatabase, user_id: str) -> Dict[str, List[QueryPublic]]:
    """Get all queries for all chats belonging to a user, grouped by chat_id."""
    # First, get all chats for the user
    chats = await db["chats"].find({"user_id": user_id}).to_list(length=None)
    
    # Get all chat IDs
    chat_ids = [str(chat["_id"]) for chat in chats]
    
    # Fetch all queries for these chats
    result: Dict[str, List[QueryPublic]] = {}
    
    for chat_id in chat_ids:
        queries = await get_queries_by_chat_id(db, chat_id)
        if queries:
            result[chat_id] = queries
    
    return result


async def update_query_response(db: AsyncIOMotorDatabase, query_id: str, response: Any) -> Optional[QueryPublic]:
    """Update the response field of a query."""
    try:
        result = await db[COLLECTION_NAME].update_one(
            {"_id": ObjectId(query_id)},
            {"$set": {"response": response}}
        )
        
        if result.modified_count == 0:
            return None
        
        updated = await db[COLLECTION_NAME].find_one({"_id": ObjectId(query_id)})
        if updated:
            return QueryPublic(**query_to_public(updated))
        return None
    except Exception as e:
        print(f"Error updating query: {e}")
        return None

