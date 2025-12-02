from typing import Any, Mapping
from datetime import datetime


def message_to_public(message: Mapping[str, Any]) -> dict:
    """Convert MongoDB message document to public format."""
    return {
        "id": str(message["_id"]),
        "role": message["role"],
        "content": message["content"],
        "image_url": message.get("image_url"),
        "response_type": message.get("response_type"),
        "session_id": message["session_id"],
        "user_id": message.get("user_id"),
        "created_at": message["created_at"],
        "metadata": message.get("metadata", {}),
    }

