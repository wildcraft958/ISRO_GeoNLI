from typing import Any, Mapping


def chat_to_public(chat: Mapping[str, Any]) -> dict:
    return {
        "id": str(chat["_id"]),
        "image_url": chat["image_url"],
        "user_id": chat["user_id"],
    }
