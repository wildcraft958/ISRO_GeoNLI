"""
Memory service for managing chat memory, session memory, and user memory.

This service handles persistence of:
- Messages: Individual conversation messages
- Sessions: Session context and metadata
- User Memories: Long-term user preferences and context
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase
from langchain_openai import ChatOpenAI

from app.models.message import message_to_public
from app.models.session import session_to_public
from app.models.user_memory import user_memory_to_public
from app.schemas.message_schema import MessageCreate, MessagePublic
from app.schemas.session_schema import SessionCreate, SessionUpdate, SessionPublic
from app.schemas.user_memory_schema import UserMemoryCreate, UserMemoryUpdate, UserMemoryPublic
from app.config import settings

logger = logging.getLogger(__name__)

# Collection names
MESSAGES_COLLECTION = "messages"
SESSIONS_COLLECTION = "sessions"
USER_MEMORIES_COLLECTION = "user_memories"

# Memory limits (configurable constants)
MAX_CONVERSATION_SUMMARIES = 10
MAX_FREQUENT_QUERIES = 20
MIN_MESSAGES_FOR_SUMMARY = 10
DEFAULT_MAX_MESSAGES = 50


# =============================================================================
# Message Operations
# =============================================================================

async def save_message(
    db: AsyncIOMotorDatabase,
    data: MessageCreate
) -> MessagePublic:
    """Save a message to the database."""
    doc = data.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    
    result = await db[MESSAGES_COLLECTION].insert_one(doc)
    created = await db[MESSAGES_COLLECTION].find_one({"_id": result.inserted_id})
    return MessagePublic(**message_to_public(created))


async def get_session_messages(
    db: AsyncIOMotorDatabase,
    session_id: str,
    limit: Optional[int] = None,
    skip: int = 0
) -> List[MessagePublic]:
    """Retrieve all messages for a session, ordered by creation time."""
    query = {"session_id": session_id}
    cursor = db[MESSAGES_COLLECTION].find(query).sort("created_at", 1).skip(skip)
    
    if limit:
        cursor = cursor.limit(limit)
    
    messages = []
    async for doc in cursor:
        messages.append(MessagePublic(**message_to_public(doc)))
    
    return messages


async def delete_session_messages(
    db: AsyncIOMotorDatabase,
    session_id: str
) -> int:
    """Delete all messages for a session. Returns count of deleted messages."""
    result = await db[MESSAGES_COLLECTION].delete_many({"session_id": session_id})
    return result.deleted_count


# =============================================================================
# Session Operations
# =============================================================================

async def create_session(
    db: AsyncIOMotorDatabase,
    data: SessionCreate
) -> SessionPublic:
    """Create a new session."""
    doc = data.model_dump()
    now = datetime.now(timezone.utc)
    doc["created_at"] = now
    doc["updated_at"] = now
    doc["last_activity"] = now
    doc.setdefault("context", {})
    doc.setdefault("summary", None)
    doc.setdefault("message_count", 0)
    
    result = await db[SESSIONS_COLLECTION].insert_one(doc)
    created = await db[SESSIONS_COLLECTION].find_one({"_id": result.inserted_id})
    return SessionPublic(**session_to_public(created))


async def get_session(
    db: AsyncIOMotorDatabase,
    session_id: str
) -> Optional[SessionPublic]:
    """Retrieve a session by session_id."""
    doc = await db[SESSIONS_COLLECTION].find_one({"session_id": session_id})
    if doc:
        return SessionPublic(**session_to_public(doc))
    return None


async def update_session_context(
    db: AsyncIOMotorDatabase,
    session_id: str,
    update_data: SessionUpdate
) -> Optional[SessionPublic]:
    """Update session context, summary, or message count."""
    update_dict = update_data.model_dump(exclude_unset=True)
    if not update_dict:
        return await get_session(db, session_id)
    
    update_dict["updated_at"] = datetime.now(timezone.utc)
    update_dict["last_activity"] = datetime.now(timezone.utc)
    
    # Handle context merge
    if "context" in update_dict:
        # Merge context dict instead of replacing
        existing = await db[SESSIONS_COLLECTION].find_one({"session_id": session_id})
        if existing and "context" in existing:
            existing_context = existing.get("context", {})
            existing_context.update(update_dict["context"])
            update_dict["context"] = existing_context
    
    result = await db[SESSIONS_COLLECTION].update_one(
        {"session_id": session_id},
        {"$set": update_dict}
    )
    
    if result.matched_count > 0:
        return await get_session(db, session_id)
    return None


async def get_user_sessions(
    db: AsyncIOMotorDatabase,
    user_id: str,
    limit: Optional[int] = None,
    skip: int = 0
) -> List[SessionPublic]:
    """Retrieve all sessions for a user, ordered by last activity."""
    query = {"user_id": user_id}
    cursor = db[SESSIONS_COLLECTION].find(query).sort("last_activity", -1).skip(skip)
    
    if limit:
        cursor = cursor.limit(limit)
    
    sessions = []
    async for doc in cursor:
        sessions.append(SessionPublic(**session_to_public(doc)))
    
    return sessions


async def delete_session(
    db: AsyncIOMotorDatabase,
    session_id: str
) -> bool:
    """Delete a session and all its messages.
    
    Returns:
        True if session was deleted or if session didn't exist but cleanup was performed.
        False only if there was an error during deletion.
    """
    # Delete messages first (always succeeds, even if no messages exist)
    deleted_messages = await delete_session_messages(db, session_id)
    logger.debug(f"Deleted {deleted_messages} messages for session {session_id}")
    
    # Delete session
    result = await db[SESSIONS_COLLECTION].delete_one({"session_id": session_id})
    
    # Return True if session was deleted OR if messages were deleted (cleanup performed)
    # This handles the case where session doesn't exist but messages might
    return result.deleted_count > 0 or deleted_messages > 0


async def increment_session_message_count(
    db: AsyncIOMotorDatabase,
    session_id: str
) -> None:
    """Increment the message count for a session."""
    await db[SESSIONS_COLLECTION].update_one(
        {"session_id": session_id},
        {
            "$inc": {"message_count": 1},
            "$set": {
                "updated_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc)
            }
        }
    )


# =============================================================================
# User Memory Operations
# =============================================================================

async def get_user_memory(
    db: AsyncIOMotorDatabase,
    user_id: str
) -> Optional[UserMemoryPublic]:
    """Retrieve user memory by user_id."""
    doc = await db[USER_MEMORIES_COLLECTION].find_one({"user_id": user_id})
    if doc:
        return UserMemoryPublic(**user_memory_to_public(doc))
    return None


async def create_user_memory(
    db: AsyncIOMotorDatabase,
    data: UserMemoryCreate
) -> UserMemoryPublic:
    """Create a new user memory."""
    doc = data.model_dump()
    now = datetime.now(timezone.utc)
    doc["created_at"] = now
    doc["updated_at"] = now
    doc.setdefault("preferences", {})
    doc.setdefault("conversation_summaries", [])
    doc.setdefault("frequent_queries", [])
    
    result = await db[USER_MEMORIES_COLLECTION].insert_one(doc)
    created = await db[USER_MEMORIES_COLLECTION].find_one({"_id": result.inserted_id})
    return UserMemoryPublic(**user_memory_to_public(created))


async def update_user_memory(
    db: AsyncIOMotorDatabase,
    user_id: str,
    update_data: UserMemoryUpdate
) -> Optional[UserMemoryPublic]:
    """Update user memory (preferences, summaries, frequent queries).
    
    Optimized to fetch existing document once and merge all updates.
    """
    update_dict = update_data.model_dump(exclude_unset=True)
    if not update_dict:
        return await get_user_memory(db, user_id)
    
    update_dict["updated_at"] = datetime.now(timezone.utc)
    
    # Fetch existing document once for all merge operations
    existing = await db[USER_MEMORIES_COLLECTION].find_one({"user_id": user_id})
    
    # Handle preferences merge
    if "preferences" in update_dict and existing:
        existing_prefs = existing.get("preferences", {})
        existing_prefs.update(update_dict["preferences"])
        update_dict["preferences"] = existing_prefs
    
    # Handle list appends for summaries
    if "conversation_summaries" in update_dict and existing:
        existing_summaries = existing.get("conversation_summaries", [])
        new_summaries = update_dict["conversation_summaries"]
        # Append new summaries, keeping last MAX_CONVERSATION_SUMMARIES
        combined = existing_summaries + new_summaries
        update_dict["conversation_summaries"] = combined[-MAX_CONVERSATION_SUMMARIES:]
    
    # Handle list appends for queries
    if "frequent_queries" in update_dict and existing:
        existing_queries = existing.get("frequent_queries", [])
        new_queries = update_dict["frequent_queries"]
        # Append new queries, deduplicate, keep last MAX_FREQUENT_QUERIES
        combined = list(set(existing_queries + new_queries))
        update_dict["frequent_queries"] = combined[-MAX_FREQUENT_QUERIES:]
    
    result = await db[USER_MEMORIES_COLLECTION].update_one(
        {"user_id": user_id},
        {"$set": update_dict}
    )
    
    if result.matched_count > 0:
        return await get_user_memory(db, user_id)
    
    # If not found, create it
    if result.matched_count == 0:
        create_data = UserMemoryCreate(user_id=user_id, preferences=update_dict.get("preferences", {}))
        return await create_user_memory(db, create_data)
    
    return None


# =============================================================================
# Conversation Summarization
# =============================================================================

async def summarize_conversation(
    db: AsyncIOMotorDatabase,
    session_id: str,
    max_messages: int = DEFAULT_MAX_MESSAGES
) -> Optional[str]:
    """
    Summarize a conversation using LLM when it gets too long.
    
    Retrieves recent messages and generates a summary that can replace
    older messages in context.
    """
    try:
        # Get recent messages
        messages = await get_session_messages(db, session_id, limit=max_messages)
        
        if len(messages) < MIN_MESSAGES_FOR_SUMMARY:
            return None
        
        # Build conversation text
        conversation_text = []
        for msg in messages:
            role = msg.role
            content = msg.content[:200]  # Truncate long messages
            if msg.image_url:
                conversation_text.append(f"{role}: [Image: {msg.image_url}] {content}")
            else:
                conversation_text.append(f"{role}: {content}")
        
        conversation_str = "\n".join(conversation_text)
        
        # Generate summary using LLM
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not set, cannot generate summary")
            return None
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            f"Summarize the following conversation in 2-3 sentences, "
            f"focusing on the main topics, user preferences, and key information:\n\n"
            f"{conversation_str}\n\n"
            f"Summary:"
        )
        
        response = llm.invoke(prompt)
        summary = response.content.strip()
        
        # Update session with summary
        await update_session_context(
            db,
            session_id,
            SessionUpdate(summary=summary)
        )
        
        logger.info(f"Generated conversation summary for session {session_id}")
        return summary
    
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}", exc_info=True)
        return None

