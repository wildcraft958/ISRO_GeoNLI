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

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
