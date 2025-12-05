# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.api.v1.router import api_router
from app.db.models import Base
from app.core.database import engine

# 1. Create Database Tables
# In production, use Alembic. For dev/prototyping, this auto-creates tables on startup.
Base.metadata.create_all(bind=engine)

# 2. Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Backend for AI Chat Application (Image Upload & Orchestration)"
)

# 3. CORS Configuration
# Allow your frontend (e.g., localhost:3000) to hit this backend
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    # Add your production domain here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Include Routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# 5. Root/Health Check
@app.get("/")
def read_root():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

# Entry point for debugging
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)