"""
Unit tests for ConversationBufferManager.

Tests cover:
- Token counting with tiktoken
- Buffer threshold detection
- Sliding window summarization
- Message management
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.memory_service import (
    ConversationBufferManager,
    get_buffer_manager,
)


class TestTokenCounting:
    """Tests for token counting functionality."""
    
    @pytest.mark.unit
    def test_count_tokens_empty_string(self):
        """Test token counting with empty string."""
        manager = ConversationBufferManager()
        assert manager.count_tokens("") == 0
    
    @pytest.mark.unit
    def test_count_tokens_simple_text(self):
        """Test token counting with simple text."""
        manager = ConversationBufferManager()
        tokens = manager.count_tokens("Hello, world!")
        
        # Should return a positive integer
        assert isinstance(tokens, int)
        assert tokens > 0
    
    @pytest.mark.unit
    def test_count_tokens_longer_text(self):
        """Test token counting with longer text."""
        manager = ConversationBufferManager()
        short_text = "Hello"
        long_text = "Hello, this is a much longer text with more words and content."
        
        short_tokens = manager.count_tokens(short_text)
        long_tokens = manager.count_tokens(long_text)
        
        # Longer text should have more tokens
        assert long_tokens > short_tokens
    
    @pytest.mark.unit
    def test_count_message_tokens_empty_list(self):
        """Test message token counting with empty list."""
        manager = ConversationBufferManager()
        assert manager.count_message_tokens([]) == 0
    
    @pytest.mark.unit
    def test_count_message_tokens_single_message(self):
        """Test message token counting with single message."""
        manager = ConversationBufferManager()
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        tokens = manager.count_message_tokens(messages)
        assert tokens > 0
    
    @pytest.mark.unit
    def test_count_message_tokens_with_image_url(self):
        """Test message token counting with image URL."""
        manager = ConversationBufferManager()
        messages = [
            {
                "role": "user",
                "content": "What's in this image?",
                "image_url": "https://example.com/image.jpg"
            }
        ]
        
        tokens = manager.count_message_tokens(messages)
        assert tokens > 0
    
    @pytest.mark.unit
    def test_count_message_tokens_multiple_messages(self):
        """Test message token counting with multiple messages."""
        manager = ConversationBufferManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'm sorry, I don't have access to weather data."},
        ]
        
        tokens = manager.count_message_tokens(messages)
        
        # Should be more than a single message
        single_tokens = manager.count_message_tokens([messages[0]])
        assert tokens > single_tokens


class TestShouldSummarize:
    """Tests for summarization threshold detection."""
    
    @pytest.mark.unit
    def test_should_summarize_below_threshold(self):
        """Test that summarization is not triggered below threshold."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=20,
            max_tokens=8000
        )
        
        # Create 10 messages (below threshold)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        
        assert manager.should_summarize(messages) is False
    
    @pytest.mark.unit
    def test_should_summarize_at_threshold(self):
        """Test that summarization is triggered at threshold."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=20,
            max_tokens=8000
        )
        
        # Create 20 messages (at threshold)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]
        
        assert manager.should_summarize(messages) is True
    
    @pytest.mark.unit
    def test_should_summarize_above_max_messages(self):
        """Test that summarization is triggered above max messages."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=20,
            max_tokens=8000
        )
        
        # Create 35 messages (above max)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(35)
        ]
        
        assert manager.should_summarize(messages) is True
    
    @pytest.mark.unit
    def test_should_summarize_token_threshold(self):
        """Test that summarization is triggered by token count."""
        manager = ConversationBufferManager(
            max_messages=1000,  # High message limit
            summary_threshold=500,  # High threshold
            max_tokens=100  # Low token limit
        )
        
        # Create messages with enough content to exceed token limit
        messages = [
            {"role": "user", "content": "This is a fairly long message with many words. " * 10}
            for i in range(5)
        ]
        
        assert manager.should_summarize(messages) is True


class TestSummaryMessageCreation:
    """Tests for summary message creation."""
    
    @pytest.mark.unit
    def test_create_summary_message_basic(self):
        """Test basic summary message creation."""
        manager = ConversationBufferManager()
        
        summary_msg = manager._create_summary_message(
            summary_text="This is a conversation summary.",
            summarized_count=10
        )
        
        assert summary_msg["role"] == "system"
        assert summary_msg["is_summary"] is True
        assert summary_msg["summarized_count"] == 10
        assert "This is a conversation summary." in summary_msg["content"]
        assert "[Conversation Summary]" in summary_msg["content"]
        assert "timestamp" in summary_msg
    
    @pytest.mark.unit
    def test_create_summary_message_with_previous(self):
        """Test summary message creation with previous summary."""
        manager = ConversationBufferManager()
        
        summary_msg = manager._create_summary_message(
            summary_text="New summary content.",
            summarized_count=5,
            previous_summary="Previous conversation context."
        )
        
        assert "Previous conversation context." in summary_msg["content"]
        assert "New summary content." in summary_msg["content"]
        assert "More recently:" in summary_msg["content"]


class TestBuildSummaryPrompt:
    """Tests for summary prompt building."""
    
    @pytest.mark.unit
    def test_build_summary_prompt_basic(self):
        """Test basic summary prompt building."""
        manager = ConversationBufferManager()
        
        messages = [
            {"role": "user", "content": "What's in the image?"},
            {"role": "assistant", "content": "I see a cat."},
        ]
        
        prompt = manager._build_summary_prompt(messages)
        
        assert "user:" in prompt.lower()
        assert "assistant:" in prompt.lower()
        assert "What's in the image?" in prompt
        assert "I see a cat." in prompt
        assert "Summarize" in prompt
    
    @pytest.mark.unit
    def test_build_summary_prompt_with_image(self):
        """Test summary prompt building with image messages."""
        manager = ConversationBufferManager()
        
        messages = [
            {
                "role": "user",
                "content": "Describe this",
                "image_url": "https://example.com/image.jpg"
            },
        ]
        
        prompt = manager._build_summary_prompt(messages)
        
        assert "[Image provided]" in prompt
    
    @pytest.mark.unit
    def test_build_summary_prompt_truncates_long_content(self):
        """Test that very long messages are truncated in prompt."""
        manager = ConversationBufferManager()
        
        long_content = "x" * 1000  # 1000 characters
        messages = [
            {"role": "user", "content": long_content},
        ]
        
        prompt = manager._build_summary_prompt(messages)
        
        # Should be truncated to 500 chars
        assert len(prompt) < len(long_content) + 200


class TestManageBuffer:
    """Tests for buffer management."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manage_buffer_no_action_needed(self):
        """Test buffer management when no action is needed."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=20,
            recent_messages=10
        )
        
        # Create 5 messages (below threshold)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(5)
        ]
        
        managed_messages, context = await manager.manage_buffer(messages, {})
        
        # Should return messages unchanged
        assert len(managed_messages) == 5
        assert "buffer_token_count" in context
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manage_buffer_not_enough_to_summarize(self):
        """Test buffer management when there aren't enough messages to summarize."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=5,  # Low threshold
            recent_messages=10  # But keep 10 recent
        )
        
        # Create 7 messages (above threshold but <= recent_messages)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(7)
        ]
        
        managed_messages, context = await manager.manage_buffer(messages, {})
        
        # Should return messages unchanged (can't summarize if all are "recent")
        assert len(managed_messages) == 7
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manage_buffer_with_summarization(self):
        """Test buffer management with summarization."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=10,
            recent_messages=5
        )
        
        # Create 15 messages (above threshold)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(15)
        ]
        
        with patch.object(manager, 'summarize_messages', new_callable=AsyncMock) as mock_summarize:
            mock_summarize.return_value = "This is a test summary."
            
            managed_messages, context = await manager.manage_buffer(messages, {})
            
            # Should have summary + recent messages
            assert len(managed_messages) == 6  # 1 summary + 5 recent
            assert managed_messages[0]["is_summary"] is True
            assert managed_messages[0]["summarized_count"] == 10
            
            # Context should be updated
            assert "buffer_summary" in context
            assert "last_summarization_at" in context
            assert context["total_summarized_messages"] == 10
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manage_buffer_summarization_failure_fallback(self):
        """Test buffer management fallback when summarization fails."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=10,
            recent_messages=5
        )
        
        # Create 15 messages
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(15)
        ]
        
        with patch.object(manager, 'summarize_messages', new_callable=AsyncMock) as mock_summarize:
            mock_summarize.return_value = None  # Summarization failed
            
            managed_messages, context = await manager.manage_buffer(messages, {})
            
            # Should fall back to simple truncation
            assert len(managed_messages) == 5  # Just recent messages
            assert "buffer_truncated_at" in context
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manage_buffer_with_existing_summary(self):
        """Test buffer management when there's already a summary message."""
        manager = ConversationBufferManager(
            max_messages=30,
            summary_threshold=10,
            recent_messages=5
        )
        
        # Create messages with existing summary at start
        existing_summary = {
            "role": "system",
            "content": "[Conversation Summary]\nPrevious context.",
            "is_summary": True,
            "summarized_count": 5
        }
        
        messages = [existing_summary] + [
            {"role": "user", "content": f"Message {i}"}
            for i in range(12)
        ]
        
        with patch.object(manager, 'summarize_messages', new_callable=AsyncMock) as mock_summarize:
            mock_summarize.return_value = "New summary content."
            
            managed_messages, context = await manager.manage_buffer(messages, {})
            
            # Should combine previous and new summary
            assert len(managed_messages) == 6  # 1 summary + 5 recent
            assert "Previous context." in managed_messages[0]["content"]
            assert "New summary content." in managed_messages[0]["content"]


class TestSummarizeMessages:
    """Tests for message summarization."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_summarize_messages_empty_list(self):
        """Test summarization with empty message list."""
        manager = ConversationBufferManager()
        
        result = await manager.summarize_messages([])
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_summarize_messages_no_api_key(self):
        """Test summarization without OpenAI API key."""
        manager = ConversationBufferManager()
        
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        
        with patch('app.services.memory_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ""
            
            result = await manager.summarize_messages(messages)
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_summarize_messages_success(self):
        """Test successful message summarization."""
        manager = ConversationBufferManager()
        
        messages = [
            {"role": "user", "content": "What's in the image?"},
            {"role": "assistant", "content": "I see a satellite image of a city."},
        ]
        
        with patch('app.services.memory_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            with patch('app.services.memory_service.ChatOpenAI') as mock_llm_class:
                mock_llm = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "User asked about an image showing a city."
                mock_llm.invoke.return_value = mock_response
                mock_llm_class.return_value = mock_llm
                
                result = await manager.summarize_messages(messages)
                
                assert result == "User asked about an image showing a city."
                mock_llm.invoke.assert_called_once()


class TestGetBufferManager:
    """Tests for singleton buffer manager accessor."""
    
    @pytest.mark.unit
    def test_get_buffer_manager_returns_instance(self):
        """Test that get_buffer_manager returns an instance."""
        manager = get_buffer_manager()
        
        assert isinstance(manager, ConversationBufferManager)
    
    @pytest.mark.unit
    def test_get_buffer_manager_singleton(self):
        """Test that get_buffer_manager returns the same instance."""
        manager1 = get_buffer_manager()
        manager2 = get_buffer_manager()
        
        assert manager1 is manager2


class TestConfigurableThresholds:
    """Tests for configurable buffer thresholds."""
    
    @pytest.mark.unit
    def test_custom_max_messages(self):
        """Test custom max_messages threshold."""
        manager = ConversationBufferManager(max_messages=50)
        assert manager.max_messages == 50
    
    @pytest.mark.unit
    def test_custom_max_tokens(self):
        """Test custom max_tokens threshold."""
        manager = ConversationBufferManager(max_tokens=16000)
        assert manager.max_tokens == 16000
    
    @pytest.mark.unit
    def test_custom_recent_messages(self):
        """Test custom recent_messages count."""
        manager = ConversationBufferManager(recent_messages=15)
        assert manager.recent_messages == 15
    
    @pytest.mark.unit
    def test_custom_summary_threshold(self):
        """Test custom summary_threshold."""
        manager = ConversationBufferManager(summary_threshold=25)
        assert manager.summary_threshold == 25
    
    @pytest.mark.unit
    def test_defaults_from_settings(self):
        """Test that defaults come from settings."""
        with patch('app.services.memory_service.settings') as mock_settings:
            mock_settings.BUFFER_MAX_MESSAGES = 40
            mock_settings.BUFFER_MAX_TOKENS = 10000
            mock_settings.BUFFER_RECENT_MESSAGES = 12
            mock_settings.BUFFER_SUMMARY_THRESHOLD = 25
            
            manager = ConversationBufferManager()
            
            assert manager.max_messages == 40
            assert manager.max_tokens == 10000
            assert manager.recent_messages == 12
            assert manager.summary_threshold == 25

