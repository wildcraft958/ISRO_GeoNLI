from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from requests import Session
import uvicorn
import logging

from app.core.config import settings
from app.api.v1.router import api_router
from app.db.models import Base
from app.core.database import engine, get_db
from app.core.logging_middleware import TimeLoggingMiddleware
from app.api.v1.endpoints.test_pipeline import test_pipeline
from app.schemas.test_structure import TestPipelineRequest, TestPipelineResponse
# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 1. Create Database Tables
Base.metadata.create_all(bind=engine)

# 2. Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Backend for AI Chat Application (Image Upload & Orchestration)"
)

# 3. Add Middlewares
# Time Logger (Custom)
app.add_middleware(TimeLoggingMiddleware)

# CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Instrumentator (Prometheus Metrics)
# Exposes /metrics endpoint for scraping
Instrumentator().instrument(app).expose(app)

# 5. Include Routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# 6. Root/Health Check
@app.get("/")
def read_root():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

@app.post("/test", response_model=TestPipelineResponse, summary="End-to-End Pipeline Test")
async def test(payload: TestPipelineRequest, db: Session = Depends(get_db)):
    return test_pipeline(payload,db)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)