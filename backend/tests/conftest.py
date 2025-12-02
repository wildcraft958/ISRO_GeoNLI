"""
Shared pytest fixtures and configuration for ISRO Vision API tests.

This module provides:
- FastAPI test client
- Mock MongoDB database
- Sample image data (URLs, base64, numpy arrays)
- Mock external service responses
- Environment variable fixtures
"""

import base64
import io
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Set test environment variables before importing app
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "test_isro_vision")
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")
os.environ.setdefault("AWS_S3_BUCKET_NAME", "test-bucket")


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================

@pytest.fixture
def app():
    """Get the FastAPI application instance."""
    from app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create a FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app):
    """Create an async HTTP client for testing."""
    import httpx
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Mock MongoDB Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    
    # Mock collections
    db.messages = MagicMock()
    db.sessions = MagicMock()
    db.user_memories = MagicMock()
    
    # Mock common operations
    db.messages.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_msg_id"))
    db.messages.find_one = AsyncMock(return_value=None)
    db.messages.find = MagicMock(return_value=AsyncIteratorMock([]))
    db.messages.delete_many = AsyncMock(return_value=MagicMock(deleted_count=0))
    
    db.sessions.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_session_id"))
    db.sessions.find_one = AsyncMock(return_value=None)
    db.sessions.find = MagicMock(return_value=AsyncIteratorMock([]))
    db.sessions.update_one = AsyncMock(return_value=MagicMock(matched_count=1))
    db.sessions.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    
    db.user_memories.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_memory_id"))
    db.user_memories.find_one = AsyncMock(return_value=None)
    db.user_memories.update_one = AsyncMock(return_value=MagicMock(matched_count=1))
    
    return db


class AsyncIteratorMock:
    """Mock async iterator for MongoDB cursors."""
    
    def __init__(self, items: list):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
    
    def sort(self, *args, **kwargs):
        return self
    
    def skip(self, n: int):
        return self
    
    def limit(self, n: int):
        return self


@pytest.fixture
def mock_db_dependency(mock_db):
    """Override the database dependency for testing."""
    async def override_get_db():
        return mock_db
    
    return override_get_db


# =============================================================================
# Sample Image Fixtures
# =============================================================================

@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    # Create a simple 100x100 RGB image with some variation
    np.random.seed(42)
    arr = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a sample grayscale image for testing."""
    np.random.seed(42)
    arr = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Create a sample RGBA image with alpha channel."""
    np.random.seed(42)
    rgb = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    # Create alpha channel with many 0 and 255 values (alpha-like)
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[25:75, 25:75] = 255
    arr = np.dstack([rgb, alpha])
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def sample_sar_like_image() -> np.ndarray:
    """Create a SAR-like image with high speckle noise."""
    np.random.seed(42)
    # SAR images have high coefficient of variation (speckle)
    base = np.random.exponential(scale=50, size=(100, 100))
    arr = np.clip(base, 0, 255).astype(np.uint8)
    return arr


@pytest.fixture
def sample_infrared_image() -> np.ndarray:
    """Create a sample 4-channel infrared/multispectral image."""
    np.random.seed(42)
    # 4-channel image (NIR, R, G, B or similar)
    arr = np.random.randint(50, 200, (100, 100, 4), dtype=np.uint8)
    return arr


@pytest.fixture
def sample_image_bytes(sample_rgb_image) -> bytes:
    """Get sample image as bytes."""
    buffer = io.BytesIO()
    sample_rgb_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes) -> str:
    """Get sample image as base64 string."""
    return base64.b64encode(sample_image_bytes).decode()


@pytest.fixture
def sample_image_data_uri(sample_image_base64) -> str:
    """Get sample image as data URI."""
    return f"data:image/png;base64,{sample_image_base64}"


@pytest.fixture
def sample_image_url() -> str:
    """Sample image URL for testing."""
    return "https://example.com/test_image.jpg"


# =============================================================================
# Mock External Services
# =============================================================================

@pytest.fixture
def mock_requests_get(sample_image_bytes):
    """Mock requests.get for image downloads."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = sample_image_bytes
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_modal_client():
    """Mock the Modal service client."""
    with patch("app.services.modal_client.ModalServiceClient") as mock_class:
        mock_instance = MagicMock()
        mock_instance.call_grounding.return_value = {
            "bboxes": [{"x": 10, "y": 20, "w": 50, "h": 50, "label": "object", "conf": 0.95}]
        }
        mock_instance.call_vqa.return_value = {
            "answer": "This is a test answer from VQA service."
        }
        mock_instance.call_captioning.return_value = {
            "caption": "A test image showing various objects."
        }
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_resnet_classifier():
    """Mock the ResNet classifier client."""
    with patch("app.services.resnet_classifier_client.ResNetClassifierClient") as mock_class:
        mock_instance = MagicMock()
        mock_instance.classify_from_url.return_value = (
            "rgb",
            0.95,
            {"rgb": 0.95, "infrared": 0.03, "sar": 0.02}
        )
        mock_instance.health_check.return_value = True
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai():
    """Mock OpenAI/LangChain LLM calls."""
    with patch("langchain_openai.ChatOpenAI") as mock_llm:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "QA"
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        yield mock_instance


# =============================================================================
# Schema Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_chat_request() -> Dict[str, Any]:
    """Sample chat request data."""
    return {
        "image_url": "https://example.com/test.jpg",
        "query": "What is in this image?",
        "mode": "auto",
        "session_id": "test-session-123",
        "user_id": "test-user-456",
        "modality_detection_enabled": True,
        "needs_ir2rgb": False,
    }


@pytest.fixture
def sample_ir2rgb_request() -> Dict[str, Any]:
    """Sample IR2RGB request data."""
    return {
        "image_url": "https://example.com/multispectral.tif",
        "channels": ["NIR", "R", "G"],
        "synthesize_channel": "B",
    }


@pytest.fixture
def sample_agent_state() -> Dict[str, Any]:
    """Sample agent state for workflow testing."""
    return {
        "session_id": "test-session-123",
        "image_url": "https://example.com/test.jpg",
        "user_query": "What is this?",
        "mode": "auto",
        "modality_detection_enabled": True,
        "detected_modality": None,
        "modality_confidence": None,
        "modality_diagnostics": None,
        "resnet_classification_used": False,
        "needs_ir2rgb": False,
        "ir2rgb_channels": None,
        "ir2rgb_synthesize": "B",
        "original_image_url": None,
        "caption_result": None,
        "vqa_result": None,
        "grounding_result": None,
        "messages": [],
        "session_context": {},
        "execution_log": [],
    }


@pytest.fixture
def sample_session_data() -> Dict[str, Any]:
    """Sample session data."""
    return {
        "_id": "test_id",
        "session_id": "test-session-123",
        "user_id": "test-user-456",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "last_activity": datetime.now(timezone.utc),
        "context": {},
        "summary": None,
        "message_count": 0,
    }


@pytest.fixture
def sample_message_data() -> Dict[str, Any]:
    """Sample message data."""
    return {
        "_id": "test_msg_id",
        "role": "user",
        "content": "What is in this image?",
        "image_url": "https://example.com/test.jpg",
        "session_id": "test-session-123",
        "user_id": "test-user-456",
        "created_at": datetime.now(timezone.utc),
        "metadata": {},
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def freeze_time():
    """Fixture to freeze time for deterministic tests."""
    from freezegun import freeze_time as ft
    return ft


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset modality router singleton
    import app.services.modality_router as mr
    mr._modality_router_service = None
    
    # Reset ResNet client singleton
    import app.services.resnet_classifier_client as rc
    rc._resnet_client = None
    
    # Reset IR2RGB service singleton
    import app.services.ir2rgb_service as ir
    ir._ir2rgb_service = None
    
    yield


@pytest.fixture
def temp_image_file(sample_rgb_image, tmp_path):
    """Create a temporary image file."""
    file_path = tmp_path / "test_image.png"
    sample_rgb_image.save(file_path)
    return file_path


