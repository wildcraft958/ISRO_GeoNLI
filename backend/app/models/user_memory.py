from typing import Any, Mapping


def user_memory_to_public(user_memory: Mapping[str, Any]) -> dict:
    """Convert MongoDB user memory document to public format."""
    return {
        "id": str(user_memory["_id"]),
        "user_id": user_memory["user_id"],
        "preferences": user_memory.get("preferences", {}),
        "conversation_summaries": user_memory.get("conversation_summaries", []),
        "frequent_queries": user_memory.get("frequent_queries", []),
        "created_at": user_memory["created_at"],
        "updated_at": user_memory.get("updated_at", user_memory["created_at"]),
    }

