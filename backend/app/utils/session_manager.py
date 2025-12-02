"""
SessionManager for managing LangGraph session state and persistent memory.

Integrates LangGraph checkpoints with MongoDB-based persistent storage
for messages, sessions, and user memories.
"""

import logging
from typing import Optional, List, Dict, Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.orchestrator import app
from app.services import memory_service
from app.schemas.session_schema import SessionUpdate

logger = logging.getLogger(__name__)


class SessionManager:
    """Utility for managing LangGraph session state and persistent memory"""
    
    def __init__(self, graph_app=None, db: Optional[AsyncIOMotorDatabase] = None):
        self.app = graph_app or app
        self.db = db
    
    def get_session_history(self, session_id: str) -> List[dict]:
        """
        Retrieve all messages in session from LangGraph state.
        
        Note: For persistent messages, use get_persistent_messages() instead.
        """
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            if state and state.values:
                return state.values.get("messages", [])
            return []
        except Exception as e:
            logger.warning(
                f"Failed to get session history for {session_id}: {e}",
                exc_info=True
            )
            return []
    
    def get_session_state(self, session_id: str) -> Optional[dict]:
        """Retrieve full session state from LangGraph"""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            if state and state.values:
                return state.values
            return None
        except Exception as e:
            logger.warning(
                f"Failed to get session state for {session_id}: {e}",
                exc_info=True
            )
            return None
    
    async def get_persistent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve persistent messages from database for a session."""
        if not self.db:
            logger.warning("Database not available, returning empty list")
            return []
        
        messages = await memory_service.get_session_messages(
            self.db,
            session_id,
            limit=limit
        )
        return [msg.model_dump() for msg in messages]
    
    async def update_session_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update session context in persistent storage.
        
        Args:
            session_id: Session identifier
            context_updates: Dictionary of context fields to update
        
        Returns:
            Updated session context or None if update failed
        """
        if not self.db:
            logger.warning("Database not available, cannot update session context")
            return None
        
        try:
            update_data = SessionUpdate(context=context_updates)
            session = await memory_service.update_session_context(
                self.db,
                session_id,
                update_data
            )
            if session:
                return session.context
            return None
        except Exception as e:
            logger.error(f"Failed to update session context: {e}", exc_info=True)
            return None
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear session and all its messages from persistent storage.
        
        Note: This does not clear LangGraph checkpoints. Use with caution.
        """
        if not self.db:
            logger.warning("Database not available, cannot clear session")
            return False
        
        try:
            return await memory_service.delete_session(self.db, session_id)
        except Exception as e:
            logger.error(f"Failed to clear session: {e}", exc_info=True)
            return False
    
    async def get_session_summary(self, session_id: str) -> Optional[str]:
        """Get conversation summary for a session."""
        if not self.db:
            return None
        
        try:
            session = await memory_service.get_session(self.db, session_id)
            if session:
                return session.summary
            return None
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}", exc_info=True)
            return None
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        if not self.db:
            logger.warning("Database not available, returning empty list")
            return []
        
        try:
            sessions = await memory_service.get_user_sessions(
                self.db,
                user_id,
                limit=limit
            )
            return [session.model_dump() for session in sessions]
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}", exc_info=True)
            return []

