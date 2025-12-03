"""
Memory service for managing chat memory, session memory, and user memory.

This service handles persistence of:
- Messages: Individual conversation messages
- Sessions: Session context and metadata
- User Memories: Long-term user preferences and context
- Conversation Buffer: Token-aware sliding window management
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

import tiktoken
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

# Initialize tiktoken encoder (cl100k_base is used by GPT-4, GPT-3.5-turbo)
try:
    _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(f"Failed to load tiktoken encoder: {e}. Using fallback estimation.")
    _tiktoken_encoder = None

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


# =============================================================================
# Conversation Buffer Manager
# =============================================================================

class ConversationBufferManager:
    """
    Manages conversation buffer with token-aware sliding window.
    
    Features:
    - Token counting using tiktoken (cl100k_base encoding)
    - Automatic summarization when thresholds are exceeded
    - Sliding window: keeps recent N messages + summary of older messages
    - Async-compatible for background processing
    """
    
    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        recent_messages: Optional[int] = None,
        summary_threshold: Optional[int] = None
    ):
        """
        Initialize buffer manager with configurable limits.
        
        Args:
            max_messages: Maximum messages before forced truncation (default from settings)
            max_tokens: Maximum tokens before summarization (default from settings)
            recent_messages: Number of recent messages to keep after summarization
            summary_threshold: Message count threshold to trigger summarization
        """
        self.max_messages = max_messages or settings.BUFFER_MAX_MESSAGES
        self.max_tokens = max_tokens or settings.BUFFER_MAX_TOKENS
        self.recent_messages = recent_messages or settings.BUFFER_RECENT_MESSAGES
        self.summary_threshold = summary_threshold or settings.BUFFER_SUMMARY_THRESHOLD
        self.encoder = _tiktoken_encoder
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string using tiktoken.
        
        Falls back to word-based estimation if tiktoken is unavailable.
        """
        if not text:
            return 0
        
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed, using fallback: {e}")
        
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens across all messages in the buffer.
        
        Includes role, content, and metadata in the count.
        """
        total_tokens = 0
        
        for msg in messages:
            # Count content
            content = msg.get("content", "")
            total_tokens += self.count_tokens(content)
            
            # Count role (small overhead)
            role = msg.get("role", "")
            total_tokens += self.count_tokens(role)
            
            # Count image URL if present (as reference)
            image_url = msg.get("image_url", "")
            if image_url:
                # Just count URL string, not the image itself
                total_tokens += self.count_tokens(f"[Image: {image_url[:100]}]")
            
            # Add overhead for message structure (~4 tokens per message)
            total_tokens += 4
        
        return total_tokens
    
    def should_summarize(
        self,
        messages: List[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if buffer should be summarized based on thresholds.
        
        Returns True if:
        - Message count exceeds summary_threshold
        - Token count exceeds max_tokens
        - Message count exceeds max_messages (forced truncation)
        """
        message_count = len(messages)
        
        # Check message count thresholds
        if message_count >= self.max_messages:
            logger.debug(f"Buffer exceeds max_messages ({message_count} >= {self.max_messages})")
            return True
        
        if message_count >= self.summary_threshold:
            logger.debug(f"Buffer exceeds summary_threshold ({message_count} >= {self.summary_threshold})")
            return True
        
        # Check token count
        token_count = self.count_message_tokens(messages)
        if token_count >= self.max_tokens:
            logger.debug(f"Buffer exceeds max_tokens ({token_count} >= {self.max_tokens})")
            return True
        
        return False
    
    def _build_summary_prompt(self, messages_to_summarize: List[Dict[str, Any]]) -> str:
        """Build prompt for summarizing messages."""
        conversation_parts = []
        
        for msg in messages_to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # Truncate very long messages
            image_url = msg.get("image_url")
            
            if image_url:
                conversation_parts.append(f"{role}: [Image provided] {content}")
            else:
                conversation_parts.append(f"{role}: {content}")
        
        conversation_text = "\n".join(conversation_parts)
        
        return (
            f"Summarize the following conversation excerpt in 2-3 concise sentences. "
            f"Focus on key topics discussed, user preferences expressed, and important context:\n\n"
            f"{conversation_text}\n\n"
            f"Summary:"
        )
    
    def _create_summary_message(
        self,
        summary_text: str,
        summarized_count: int,
        previous_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a summary message to prepend to the buffer.
        
        If there was a previous summary, it's combined with the new one.
        """
        if previous_summary:
            combined_summary = f"{previous_summary}\n\nMore recently: {summary_text}"
        else:
            combined_summary = summary_text
        
        return {
            "role": "system",
            "content": f"[Conversation Summary]\n{combined_summary}",
            "is_summary": True,
            "summarized_count": summarized_count,
            "timestamp": datetime.now(timezone.utc),
        }
    
    async def summarize_messages(
        self,
        messages_to_summarize: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generate a summary of the given messages using LLM.
        
        Returns None if summarization fails.
        """
        if not messages_to_summarize:
            return None
        
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not set, cannot generate buffer summary")
            return None
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = self._build_summary_prompt(messages_to_summarize)
            
            response = llm.invoke(prompt)
            summary = response.content.strip()
            
            logger.info(f"Generated buffer summary for {len(messages_to_summarize)} messages")
            return summary
        
        except Exception as e:
            logger.error(f"Failed to generate buffer summary: {e}", exc_info=True)
            return None
    
    async def manage_buffer(
        self,
        messages: List[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for buffer management.
        
        Checks if summarization is needed, and if so:
        1. Summarizes older messages
        2. Keeps recent N messages
        3. Updates session_context with buffer metadata
        
        Returns:
            Tuple of (managed_messages, updated_session_context)
        """
        session_context = session_context or {}
        
        if not self.should_summarize(messages, session_context):
            # No action needed, return as-is with token count update
            token_count = self.count_message_tokens(messages)
            session_context["buffer_token_count"] = token_count
            return messages, session_context
        
        # Separate messages: older ones to summarize, recent ones to keep
        if len(messages) <= self.recent_messages:
            # Not enough messages to summarize
            token_count = self.count_message_tokens(messages)
            session_context["buffer_token_count"] = token_count
            return messages, session_context
        
        # Check if first message is already a summary
        has_existing_summary = (
            messages and 
            messages[0].get("is_summary", False)
        )
        
        if has_existing_summary:
            # Exclude existing summary from messages to process
            existing_summary_msg = messages[0]
            previous_summary = existing_summary_msg.get("content", "").replace("[Conversation Summary]\n", "")
            messages_to_process = messages[1:]
        else:
            previous_summary = session_context.get("buffer_summary")
            messages_to_process = messages
        
        # Split: older messages to summarize, recent to keep
        split_index = len(messages_to_process) - self.recent_messages
        messages_to_summarize = messages_to_process[:split_index]
        recent_messages = messages_to_process[split_index:]
        
        if not messages_to_summarize:
            # Nothing to summarize
            token_count = self.count_message_tokens(messages)
            session_context["buffer_token_count"] = token_count
            return messages, session_context
        
        # Generate summary
        summary_text = await self.summarize_messages(messages_to_summarize)
        
        if summary_text:
            # Create new summary message
            summary_message = self._create_summary_message(
                summary_text,
                summarized_count=len(messages_to_summarize),
                previous_summary=previous_summary
            )
            
            # Build managed buffer: [summary] + recent messages
            managed_messages = [summary_message] + recent_messages
            
            # Update session context
            session_context["buffer_summary"] = summary_message["content"]
            session_context["buffer_token_count"] = self.count_message_tokens(managed_messages)
            session_context["last_summarization_at"] = datetime.now(timezone.utc).isoformat()
            session_context["total_summarized_messages"] = (
                session_context.get("total_summarized_messages", 0) + len(messages_to_summarize)
            )
            
            logger.info(
                f"Buffer managed: {len(messages)} -> {len(managed_messages)} messages, "
                f"summarized {len(messages_to_summarize)} messages"
            )
            
            return managed_messages, session_context
        else:
            # Summarization failed, fall back to simple truncation
            logger.warning("Summarization failed, falling back to truncation")
            
            # Keep only recent messages (no summary)
            managed_messages = messages[-self.recent_messages:]
            session_context["buffer_token_count"] = self.count_message_tokens(managed_messages)
            session_context["buffer_truncated_at"] = datetime.now(timezone.utc).isoformat()
            
            return managed_messages, session_context
    
    async def manage_buffer_async(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str,
        messages: List[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Async version that also persists buffer updates to database.
        
        This is the preferred method for background task integration.
        """
        managed_messages, updated_context = await self.manage_buffer(
            messages, session_context
        )
        
        # Persist context updates to session
        try:
            await update_session_context(
                db,
                session_id,
                SessionUpdate(context=updated_context)
            )
            logger.debug(f"Persisted buffer context for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to persist buffer context: {e}", exc_info=True)
        
        return managed_messages, updated_context


# Singleton instance for easy access
_buffer_manager: Optional[ConversationBufferManager] = None


def get_buffer_manager() -> ConversationBufferManager:
    """Get or create the singleton buffer manager instance."""
    global _buffer_manager
    if _buffer_manager is None:
        _buffer_manager = ConversationBufferManager()
    return _buffer_manager

