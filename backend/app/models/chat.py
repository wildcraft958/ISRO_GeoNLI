from datetime import datetime, timezone
from typing import Any, Mapping


def chat_to_public(chat: Mapping[str, Any]) -> dict:
    # Handle existing chats that might not have created_at
    created_at = chat.get("created_at")
    if created_at is None:
        # Use current time as fallback for existing chats without created_at
        created_at = datetime.now(timezone.utc)
    
    return {
        "id": str(chat["_id"]),
        "image_url": chat["image_url"],
        "user_id": chat["user_id"],
        "created_at": created_at,
    }
