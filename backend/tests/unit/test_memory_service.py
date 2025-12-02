"""
Unit tests for the memory service.

Tests cover:
- Message operations: create, get, list
- Session operations: create, get, update, delete
- User memory operations: get, update
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import memory_service
from app.schemas.message_schema import MessageCreate
from app.schemas.session_schema import SessionCreate, SessionUpdate
from app.schemas.user_memory_schema import UserMemoryUpdate


# =============================================================================
# Message Service Tests
# =============================================================================

class TestMessageService:
    """Tests for message-related operations."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_message(self, mock_db, sample_message_data):
        """Test creating a message."""
        mock_db.messages.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id="test_msg_id")
        )
        
        message_create = MessageCreate(
            role="user",
            content="Test message",
            session_id="test-session"
        )
        
        result = await memory_service.create_message(mock_db, message_create)
        
        assert result is not None
        mock_db.messages.insert_one.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_message(self, mock_db, sample_message_data):
        """Test getting a message by ID."""
        mock_db.messages.find_one = AsyncMock(return_value=sample_message_data)
        
        result = await memory_service.get_message(mock_db, "test_msg_id")
        
        assert result is not None
        assert result["role"] == "user"
        mock_db.messages.find_one.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_message_not_found(self, mock_db):
        """Test getting non-existent message."""
        mock_db.messages.find_one = AsyncMock(return_value=None)
        
        result = await memory_service.get_message(mock_db, "nonexistent")
        
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_messages(self, mock_db, sample_message_data):
        """Test getting all messages for a session."""
        messages = [sample_message_data, {**sample_message_data, "_id": "msg2"}]
        
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=messages)
        mock_db.messages.find = MagicMock(return_value=mock_cursor)
        
        result = await memory_service.get_session_messages(
            mock_db, 
            "test-session-123"
        )
        
        assert len(result) == 2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_session_messages(self, mock_db):
        """Test deleting all messages for a session."""
        mock_db.messages.delete_many = AsyncMock(
            return_value=MagicMock(deleted_count=5)
        )
        
        result = await memory_service.delete_session_messages(
            mock_db,
            "test-session-123"
        )
        
        assert result == 5
        mock_db.messages.delete_many.assert_called_once()


# =============================================================================
# Session Service Tests
# =============================================================================

class TestSessionService:
    """Tests for session-related operations."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session(self, mock_db):
        """Test creating a session."""
        mock_db.sessions.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id="test_session_id")
        )
        
        session_create = SessionCreate(
            session_id="test-session-123",
            user_id="test-user"
        )
        
        result = await memory_service.create_session(mock_db, session_create)
        
        assert result is not None
        mock_db.sessions.insert_one.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session(self, mock_db, sample_session_data):
        """Test getting a session by ID."""
        mock_db.sessions.find_one = AsyncMock(return_value=sample_session_data)
        
        result = await memory_service.get_session(mock_db, "test-session-123")
        
        assert result is not None
        assert result["session_id"] == "test-session-123"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_db):
        """Test getting non-existent session."""
        mock_db.sessions.find_one = AsyncMock(return_value=None)
        
        result = await memory_service.get_session(mock_db, "nonexistent")
        
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_session(self, mock_db):
        """Test updating a session."""
        mock_db.sessions.update_one = AsyncMock(
            return_value=MagicMock(matched_count=1, modified_count=1)
        )
        
        session_update = SessionUpdate(
            summary="Test summary",
            context={"key": "value"}
        )
        
        result = await memory_service.update_session(
            mock_db,
            "test-session-123",
            session_update
        )
        
        assert result is True
        mock_db.sessions.update_one.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_session(self, mock_db):
        """Test deleting a session."""
        mock_db.sessions.delete_one = AsyncMock(
            return_value=MagicMock(deleted_count=1)
        )
        mock_db.messages.delete_many = AsyncMock(
            return_value=MagicMock(deleted_count=5)
        )
        
        result = await memory_service.delete_session(mock_db, "test-session-123")
        
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, mock_db, sample_session_data):
        """Test getting all sessions for a user."""
        sessions = [sample_session_data, {**sample_session_data, "session_id": "session2"}]
        
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=sessions)
        mock_db.sessions.find = MagicMock(return_value=mock_cursor)
        
        result = await memory_service.get_user_sessions(mock_db, "test-user-456")
        
        assert len(result) == 2


# =============================================================================
# User Memory Service Tests
# =============================================================================

class TestUserMemoryService:
    """Tests for user memory operations."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_user_memory(self, mock_db):
        """Test getting user memory."""
        memory_data = {
            "_id": "mem_id",
            "user_id": "test-user",
            "preferences": {"theme": "dark"},
            "context": {},
            "created_at": datetime.now(timezone.utc)
        }
        mock_db.user_memories.find_one = AsyncMock(return_value=memory_data)
        
        result = await memory_service.get_user_memory(mock_db, "test-user")
        
        assert result is not None
        assert result["preferences"]["theme"] == "dark"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_user_memory_not_found(self, mock_db):
        """Test getting non-existent user memory."""
        mock_db.user_memories.find_one = AsyncMock(return_value=None)
        
        result = await memory_service.get_user_memory(mock_db, "nonexistent")
        
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_user_memory_existing(self, mock_db):
        """Test updating existing user memory."""
        mock_db.user_memories.update_one = AsyncMock(
            return_value=MagicMock(matched_count=1, modified_count=1)
        )
        
        memory_update = UserMemoryUpdate(
            preferences={"theme": "light"},
            context={"last_topic": "satellite imagery"}
        )
        
        result = await memory_service.update_user_memory(
            mock_db,
            "test-user",
            memory_update
        )
        
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_user_memory_create_new(self, mock_db):
        """Test creating user memory if not exists."""
        mock_db.user_memories.update_one = AsyncMock(
            return_value=MagicMock(matched_count=0, upserted_id="new_id")
        )
        
        memory_update = UserMemoryUpdate(
            preferences={"theme": "dark"}
        )
        
        result = await memory_service.update_user_memory(
            mock_db,
            "new-user",
            memory_update
        )
        
        # Should use upsert
        assert result is True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_message_with_metadata(self, mock_db):
        """Test creating message with metadata."""
        mock_db.messages.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id="test_id")
        )
        
        message_create = MessageCreate(
            role="assistant",
            content="Response content",
            session_id="test-session",
            metadata={"model": "gpt-4", "tokens": 100}
        )
        
        result = await memory_service.create_message(mock_db, message_create)
        
        assert result is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_session_with_empty_context(self, mock_db):
        """Test session with empty context."""
        mock_db.sessions.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id="test_id")
        )
        
        session_create = SessionCreate(
            session_id="test-session",
            context={}
        )
        
        result = await memory_service.create_session(mock_db, session_create)
        
        assert result is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pagination_parameters(self, mock_db, sample_message_data):
        """Test message retrieval with pagination."""
        messages = [sample_message_data]
        
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=messages)
        mock_db.messages.find = MagicMock(return_value=mock_cursor)
        
        result = await memory_service.get_session_messages(
            mock_db,
            "test-session",
            skip=10,
            limit=20
        )
        
        mock_cursor.skip.assert_called_with(10)
        mock_cursor.limit.assert_called_with(20)


