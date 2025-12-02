from app.api.deps import get_db_dep
from app.schemas.query_schema import QueryCreate,QueryPublic 
from app.services import query_service
from fastapi import APIRouter, Depends, Response, status
from motor.motor_asyncio import AsyncIOMotorDatabase

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/create", response_model=QueryPublic, status_code=status.HTTP_201_CREATED)
async def create_chat_endpoint(
    payload: QueryCreate,
    db: AsyncIOMotorDatabase = Depends(get_db_dep),
    response: Response = None,
):
    chat = await query_service.create_query(db, payload)

    if response:
        response.status_code = status.HTTP_201_CREATED
    return chat 


