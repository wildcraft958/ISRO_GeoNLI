from app.config import settings
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = settings.MONGO_URL

client = AsyncIOMotorClient(MONGO_URL)

# Choose database
db = client["isro_vision"]


def get_client() -> AsyncIOMotorClient:
    global client
    if client is None:
        client = AsyncIOMotorClient(settings.MONGO_URL)
    return client


def get_db():
    _client = get_client()
    return _client[settings.MONGO_DB_NAME]
