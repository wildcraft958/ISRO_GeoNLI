import logging
import os

from app.api.routes import chat_routes, image_upload, query_routes, user_routes, orchestrator_routes
from app.services.ir2rgb_service import is_ir2rgb_available
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="ISRO Vision API",
    description="Multimodal Chatbot Orchestrator with IR2RGB preprocessing support",
    version="0.2.0"
)

# Include routers
app.include_router(user_routes.router)
app.include_router(query_routes.router)
app.include_router(chat_routes.router)
app.include_router(image_upload.router)
app.include_router(orchestrator_routes.router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("Starting ISRO Vision API...")
    logger.info(f"IR2RGB service available: {is_ir2rgb_available()}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ISRO Vision API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "ir2rgb": is_ir2rgb_available()
        }
    }
