from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from sam3_app.api.endpoints import router
from sam3_app.core.instances import get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print("Loading SAM3 Model...")
    try:
        get_model().load_model()
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        # We might want to exit or define behavior on failure
        # For now, allow start but endpoints might fail
    yield
    # Shutdown logic if any

app = FastAPI(
    title="SAM3 Agent API",
    description="Standalone Dockerized SAM3 Agent Service",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "SAM3 Agent API is running. Visit /docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sam3_app.main:app", host="0.0.0.0", port=8000, reload=False)
