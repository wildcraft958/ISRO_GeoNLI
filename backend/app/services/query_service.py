from typing import Optional

from app.models.query import query_to_public
from app.schemas.query_schema import QueryCreate, QueryPublic 
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

