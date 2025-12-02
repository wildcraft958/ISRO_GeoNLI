from typing import Optional

from app.models.user import user_to_public
from app.schemas.user_schema import UserCreate, UserPublic
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

COLLECTION_NAME = "users"


async def create_user(db: AsyncIOMotorDatabase, data: UserCreate) -> UserPublic:
    doc = data.model_dump()

    try:
        result = await db[COLLECTION_NAME].insert_one(doc)
    except DuplicateKeyError:
        existing = await db[COLLECTION_NAME].find_one({"email": data.email})
        return UserPublic(**user_to_public(existing))

    created = await db[COLLECTION_NAME].find_one({"_id": result.inserted_id})
    return UserPublic(**user_to_public(created))


async def get_user_by_email(
    db: AsyncIOMotorDatabase, email: str
) -> Optional[UserPublic]:
    doc = await db[COLLECTION_NAME].find_one({"email": email})
    if not doc:
        return None
    return UserPublic(**user_to_public(doc))


async def get_user_by_id(
    db: AsyncIOMotorDatabase, user_id: str
) -> Optional[UserPublic]:
    from bson import ObjectId

    doc = await db[COLLECTION_NAME].find_one({"_id": ObjectId(user_id)})
    if not doc:
        return None
    return UserPublic(**user_to_public(doc))


async def list_users(db: AsyncIOMotorDatabase) -> list[UserPublic]:
    cursor = db[COLLECTION_NAME].find({})
    users = []
    async for doc in cursor:
        users.append(UserPublic(**user_to_public(doc)))
    return users
