# app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create the engine using the URL from settings
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency generator for FastAPI routes.
    Ensures DB session is closed after request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()