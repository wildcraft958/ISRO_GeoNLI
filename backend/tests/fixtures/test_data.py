"""
Test data generators and utilities for ISRO Vision API tests.

This module provides functions to generate test data for:
- Images (numpy arrays, PIL Images)
- API requests/responses
- Database documents
- Workflow states
"""

import base64
import io
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from PIL import Image


# =============================================================================
# Image Generators
# =============================================================================

def generate_rgb_image(width: int = 100, height: int = 100, seed: int = 42) -> Image.Image:
    """Generate a synthetic RGB image for testing."""
    np.random.seed(seed)
    arr = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate_grayscale_image(width: int = 100, height: int = 100, seed: int = 42) -> Image.Image:
    """Generate a synthetic grayscale image for testing."""
    np.random.seed(seed)
    arr = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def generate_rgba_image(width: int = 100, height: int = 100, seed: int = 42) -> Image.Image:
    """Generate a synthetic RGBA image with alpha channel."""
    np.random.seed(seed)
    rgb = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Create binary alpha channel
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[25:75, 25:75] = 255
    arr = np.dstack([rgb, alpha])
    return Image.fromarray(arr, mode="RGBA")


def generate_sar_like_image(width: int = 100, height: int = 100, seed: int = 42) -> np.ndarray:
    """Generate a SAR-like image with high speckle noise."""
    np.random.seed(seed)
    # SAR images have high coefficient of variation
    base = np.random.exponential(scale=50, size=(height, width))
    arr = np.clip(base, 0, 255).astype(np.uint8)
    return arr


def generate_multispectral_image(width: int = 100, height: int = 100, channels: int = 4, seed: int = 42) -> np.ndarray:
    """Generate a multispectral image (NIR, R, G, B or similar)."""
    np.random.seed(seed)
    arr = np.random.randint(50, 200, (height, width, channels), dtype=np.uint8)
    return arr


def image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()


def image_to_data_uri(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to data URI."""
    base64_str = image_to_base64(pil_image, format)
    return f"data:image/{format.lower()};base64,{base64_str}"


# =============================================================================
# API Request/Response Generators
# =============================================================================

def generate_chat_request(
    image_url: str = "https://example.com/test.jpg",
    query: str = "What is in this image?",
    mode: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """Generate a ChatRequest payload."""
    request = {
        "image_url": image_url,
        "query": query,
        "mode": mode,
        **kwargs
    }
    return request


def generate_ir2rgb_request(
    image_url: str = "https://example.com/multispectral.tif",
    channels: List[str] = None,
    synthesize_channel: str = "B",
    **kwargs
) -> Dict[str, Any]:
    """Generate an IR2RGBRequest payload."""
    if channels is None:
        channels = ["NIR", "R", "G"]
    
    request = {
        "image_url": image_url,
        "channels": channels,
        "synthesize_channel": synthesize_channel,
        **kwargs
    }
    return request


def generate_chat_response(
    session_id: str = "test-session-123",
    response_type: str = "caption",
    content: Any = "A test image",
    **kwargs
) -> Dict[str, Any]:
    """Generate a ChatResponse payload."""
    response = {
        "session_id": session_id,
        "response_type": response_type,
        "content": content,
        "execution_log": kwargs.pop("execution_log", []),
        **kwargs
    }
    return response


# =============================================================================
# Database Document Generators
# =============================================================================

def generate_session_document(
    session_id: str = "test-session-123",
    user_id: str = "test-user-456",
    **kwargs
) -> Dict[str, Any]:
    """Generate a session document for MongoDB."""
    now = datetime.now(timezone.utc)
    session = {
        "_id": kwargs.pop("_id", f"session_{session_id}"),
        "session_id": session_id,
        "user_id": user_id,
        "created_at": kwargs.pop("created_at", now),
        "updated_at": kwargs.pop("updated_at", now),
        "last_activity": kwargs.pop("last_activity", now),
        "context": kwargs.pop("context", {}),
        "summary": kwargs.pop("summary", None),
        "message_count": kwargs.pop("message_count", 0),
        **kwargs
    }
    return session


def generate_message_document(
    role: str = "user",
    content: str = "Test message",
    session_id: str = "test-session-123",
    user_id: str = "test-user-456",
    **kwargs
) -> Dict[str, Any]:
    """Generate a message document for MongoDB."""
    now = datetime.now(timezone.utc)
    message = {
        "_id": kwargs.pop("_id", f"msg_{now.timestamp()}"),
        "role": role,
        "content": content,
        "session_id": session_id,
        "user_id": user_id,
        "created_at": kwargs.pop("created_at", now),
        "metadata": kwargs.pop("metadata", {}),
        **kwargs
    }
    return message


def generate_user_memory_document(
    user_id: str = "test-user-456",
    **kwargs
) -> Dict[str, Any]:
    """Generate a user memory document for MongoDB."""
    now = datetime.now(timezone.utc)
    memory = {
        "_id": kwargs.pop("_id", f"memory_{user_id}"),
        "user_id": user_id,
        "preferences": kwargs.pop("preferences", {}),
        "context": kwargs.pop("context", {}),
        "created_at": kwargs.pop("created_at", now),
        "updated_at": kwargs.pop("updated_at", now),
        **kwargs
    }
    return memory


# =============================================================================
# Workflow State Generators
# =============================================================================

def generate_agent_state(
    session_id: str = "test-session-123",
    image_url: str = "https://example.com/test.jpg",
    user_query: str = "What is this?",
    mode: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """Generate an AgentState dictionary."""
    state = {
        "session_id": session_id,
        "image_url": image_url,
        "user_query": user_query,
        "mode": mode,
        "modality_detection_enabled": kwargs.pop("modality_detection_enabled", True),
        "detected_modality": kwargs.pop("detected_modality", None),
        "modality_confidence": kwargs.pop("modality_confidence", None),
        "modality_diagnostics": kwargs.pop("modality_diagnostics", None),
        "resnet_classification_used": kwargs.pop("resnet_classification_used", False),
        "needs_ir2rgb": kwargs.pop("needs_ir2rgb", False),
        "ir2rgb_channels": kwargs.pop("ir2rgb_channels", None),
        "ir2rgb_synthesize": kwargs.pop("ir2rgb_synthesize", "B"),
        "original_image_url": kwargs.pop("original_image_url", None),
        "caption_result": kwargs.pop("caption_result", None),
        "vqa_result": kwargs.pop("vqa_result", None),
        "grounding_result": kwargs.pop("grounding_result", None),
        "messages": kwargs.pop("messages", []),
        "session_context": kwargs.pop("session_context", {}),
        "execution_log": kwargs.pop("execution_log", []),
        **kwargs
    }
    return state


# =============================================================================
# Service Response Generators
# =============================================================================

def generate_modal_grounding_response(
    num_boxes: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """Generate a grounding service response."""
    bboxes = []
    for i in range(num_boxes):
        bboxes.append({
            "x": 10 + i * 20,
            "y": 20 + i * 20,
            "w": 50,
            "h": 50,
            "label": f"object_{i}",
            "conf": 0.9 - i * 0.1
        })
    
    return {
        "bboxes": bboxes,
        **kwargs
    }


def generate_modal_vqa_response(
    answer: str = "Yes, there is a building in the image.",
    **kwargs
) -> Dict[str, Any]:
    """Generate a VQA service response."""
    return {
        "answer": answer,
        **kwargs
    }


def generate_modal_captioning_response(
    caption: str = "A satellite image showing urban areas and roads.",
    **kwargs
) -> Dict[str, Any]:
    """Generate a captioning service response."""
    return {
        "caption": caption,
        **kwargs
    }


def generate_resnet_classification_response(
    modality: str = "rgb",
    confidence: float = 0.95,
    **kwargs
) -> Dict[str, Any]:
    """Generate a ResNet classification response."""
    probabilities = {
        "rgb": 0.95 if modality == "rgb" else 0.03,
        "infrared": 0.95 if modality == "infrared" else 0.03,
        "sar": 0.95 if modality == "sar" else 0.02,
    }
    
    return {
        "success": True,
        "modality": modality,
        "confidence": confidence,
        "probabilities": probabilities,
        **kwargs
    }


# =============================================================================
# Utility Functions
# =============================================================================

def create_test_image_file(pil_image: Image.Image, tmp_path, filename: str = "test_image.png") -> str:
    """Save a PIL image to a temporary file and return the path."""
    file_path = tmp_path / filename
    pil_image.save(file_path)
    return str(file_path)


def create_test_npz_file(npz_data: Dict[str, np.ndarray], tmp_path, filename: str = "test_model.npz") -> str:
    """Save numpy arrays to NPZ file and return the path."""
    file_path = tmp_path / filename
    np.savez(file_path, **npz_data)
    return str(file_path)


