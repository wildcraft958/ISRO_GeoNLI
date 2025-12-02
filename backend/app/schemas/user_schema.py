from typing import Optional
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    username: Optional[str] = None


class UserInDB(BaseModel):
    id: str
    email: EmailStr
    username: Optional[str] = None


class UserPublic(UserInDB):
    pass 
