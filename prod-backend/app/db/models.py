from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    # Internal Primary Key (UUID)
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # This is the ID provided by Clerk (e.g., "user_2pIx...")
    user_id = Column(String, unique=True, index=True, nullable=False)
    
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("ChatSession", back_populates="owner")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    chat_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # UPDATED Foreign Key to match the new column name
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    
    image_url = Column(Text, nullable=False)
    image_type = Column(String(10), nullable=False) 
    summary_context = Column(Text, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="sessions")
    queries = relationship("Query", back_populates="session")


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chat_sessions.chat_id"), nullable=False)
    user_id = Column(String, nullable=False) 
    
    query_text = Column(Text, nullable=False)
    query_mode = Column(String(50), nullable=False) 
    response_text = Column(Text, nullable=True) 
    response_metadata = Column(JSONB, nullable=True) 
    
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="queries")