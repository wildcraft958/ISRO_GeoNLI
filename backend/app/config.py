import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # Database settings
    MONGO_URL: str = os.getenv(
        "MONGO_URL",
        "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.5.2",
    )

    # App settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_secret_key_here")

    # mongo db name
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "isro_vision")
    
    # Modal service settings
    MODAL_BASE_URL: str = os.getenv("MODAL_BASE_URL", "https://default.modal.run")
    
    # OpenAI settings (for auto-router)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LangSmith settings (optional, for monitoring)
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY", None)
    LANGSMITH_PROJECT: Optional[str] = os.getenv("LANGSMITH_PROJECT", None)
    
    # Conversation Buffer settings
    BUFFER_MAX_MESSAGES: int = int(os.getenv("BUFFER_MAX_MESSAGES", "30"))
    BUFFER_MAX_TOKENS: int = int(os.getenv("BUFFER_MAX_TOKENS", "8000"))
    BUFFER_RECENT_MESSAGES: int = int(os.getenv("BUFFER_RECENT_MESSAGES", "10"))
    BUFFER_SUMMARY_THRESHOLD: int = int(os.getenv("BUFFER_SUMMARY_THRESHOLD", "20"))
    
    # LLM settings (for routing and classification)
    LLM_MODEL_NAME: Optional[str] = os.getenv("LLM_MODEL_NAME", None)
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    
    # System prompts for LLM tasks (configurable via environment)
    LLM_ROUTER_SYSTEM_PROMPT: str = os.getenv(
        "LLM_ROUTER_SYSTEM_PROMPT",
        """You are a task classifier for a multimodal image analysis system.
Classify user queries into one of three task types:
- VQA: Questions about image content (what, why, how, describe, count, calculate area)
- GROUNDING: Requests for object location/detection (where, locate, find, bounding boxes)
- CAPTIONING: Requests for general image description (no specific question)

Consider the image modality (rgb, infrared, sar) when classifying.
Return ONLY ONE word: "VQA", "GROUNDING", or "CAPTIONING"."""
    )
    
    LLM_VQA_CLASSIFIER_SYSTEM_PROMPT: str = os.getenv(
        "LLM_VQA_CLASSIFIER_SYSTEM_PROMPT",
        """You are a VQA question type classifier for satellite/remote sensing image analysis.
Classify questions into one of four types:
- YESNO: Questions that can be answered with yes/no (is there, does it have, are there)
- GENERAL: Open-ended questions (what, why, how, describe, explain, identify)
- COUNTING: Questions asking to count objects/features (how many, count the number of)
- AREA: Questions asking for area/measurement calculations (what is the area, calculate area, size of)

Return ONLY ONE word: "YESNO", "GENERAL", "COUNTING", or "AREA"."""
    )

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
