import logging

from app.api.routes import chat_routes, image_upload, query_routes, user_routes, orchestrator_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="ISRO Vision API")
app.include_router(user_routes.router)
app.include_router(query_routes.router)
app.include_router(chat_routes.router)
app.include_router(image_upload.router)
app.include_router(orchestrator_routes.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Server is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
