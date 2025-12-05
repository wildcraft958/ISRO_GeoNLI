# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Chat Backend"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = ""
    
    # AWS S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str =""
    AWS_REGION: str = ""
    S3_BUCKET_NAME: str = ""

    # Auth
    CLERK_SECRET_KEY: str = ""

    #Modals
    VQA_BINARY_URL: str = ""
    VQA_NUMERICAL_URL: str = ""
    VQA_GENERAL_URL: str = ""
    CAPTION_URL: str = ""
    GROUNDING_URL: str = ""

    RESNET_URL: str = "/classify"
    GENERIC_LLM_URL: str = ""

    SAR_URL: str = ""

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()