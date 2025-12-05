from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import logging

from app.core.config import settings
from app.api.v1.router import api_router
from app.db.models import Base
from app.core.database import engine
from app.core.logging_middleware import TimeLoggingMiddleware

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
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    # Add your EC2 public IP or domain here later
    "*" # For dev, restrict in prod
]

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

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)