from app.core.database import get_db
from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_db_dep() -> AsyncIOMotorDatabase:
    db = get_db()
    return db
