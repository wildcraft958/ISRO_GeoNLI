import asyncio
import logging

from pymongo import ASCENDING, DESCENDING
from app.core.database import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_db():
    """Initialize database with required indexes for all collections."""
    db = get_db()
    logger.info("Initializing database indexes...")

    # ==========================================================================
    # Users Collection
    # ==========================================================================
    users = db["users"]
    await users.create_index("email", unique=True)
    logger.info("Created index on users.email (unique)")

    # ==========================================================================
    # Messages Collection - for chat memory
    # ==========================================================================
    messages = db["messages"]
    
    # Index for retrieving messages by session (most common query)
    await messages.create_index(
        [("session_id", ASCENDING), ("created_at", ASCENDING)],
        name="idx_messages_session_created"
    )
    logger.info("Created compound index on messages (session_id, created_at)")
    
    # Index for user's messages across sessions
    await messages.create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)],
        name="idx_messages_user_created",
        sparse=True  # Only index documents where user_id exists
    )
    logger.info("Created compound index on messages (user_id, created_at)")

    # ==========================================================================
    # Sessions Collection - for session memory
    # ==========================================================================
    sessions = db["sessions"]
    
    # Unique index on session_id (primary lookup)
    await sessions.create_index(
        "session_id",
        unique=True,
        name="idx_sessions_session_id"
    )
    logger.info("Created unique index on sessions.session_id")
    
    # Index for user's sessions ordered by activity
    await sessions.create_index(
        [("user_id", ASCENDING), ("last_activity", DESCENDING)],
        name="idx_sessions_user_activity",
        sparse=True
    )
    logger.info("Created compound index on sessions (user_id, last_activity)")

    # ==========================================================================
    # User Memories Collection - for long-term user memory
    # ==========================================================================
    user_memories = db["user_memories"]
    
    # Unique index on user_id (one memory per user)
    await user_memories.create_index(
        "user_id",
        unique=True,
        name="idx_user_memories_user_id"
    )
    logger.info("Created unique index on user_memories.user_id")

    # ==========================================================================
    # LangGraph Checkpoints Collection
    # ==========================================================================
    checkpoints = db["langgraph_checkpoints"]
    
    # Index for thread_id lookups
    await checkpoints.create_index(
        "thread_id",
        name="idx_checkpoints_thread_id"
    )
    logger.info("Created index on langgraph_checkpoints.thread_id")

    logger.info("Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(init_db())
