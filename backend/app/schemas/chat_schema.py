from datetime import datetime
from pydantic import BaseModel


class ChatCreate(BaseModel):
    image_url: str
    user_id: str


class ChatInDB(BaseModel):
    id: str
    image_url: str
    user_id: str
    created_at: datetime


class ChatPublic(ChatInDB):
    pass
