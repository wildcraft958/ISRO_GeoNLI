# app/api/routes/user_routes.py
from app.api.deps import get_db_dep
from app.schemas.user_schema import UserCreate, UserPublic
from app.services import user_service
from fastapi import APIRouter, Depends, Response, status
from motor.motor_asyncio import AsyncIOMotorDatabase

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/sync", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(
    payload: UserCreate,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
    response: Response = None,
):
    existing = await user_service.get_user_by_email(db, payload.email)

    if existing:
        if response:
            response.status_code = status.HTTP_200_OK
        return existing

    user = await user_service.create_user(db, payload)
    if response:
        response.status_code = status.HTTP_201_CREATED
    return user
