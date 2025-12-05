from fastapi import APIRouter
from app.api.v1.endpoints import upload, user, chat

api_router = APIRouter()

api_router.include_router(upload.router, tags=["Image Upload & Classify"])
api_router.include_router(user.router, tags=["User"])
api_router.include_router(chat.router, tags=["Chat"])