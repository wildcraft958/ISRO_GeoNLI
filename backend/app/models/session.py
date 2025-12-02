from typing import Any, Mapping


def session_to_public(session: Mapping[str, Any]) -> dict:
    """Convert MongoDB session document to public format."""
    return {
        "id": str(session["_id"]),
        "session_id": session["session_id"],
        "user_id": session.get("user_id"),
        "context": session.get("context", {}),
        "summary": session.get("summary"),
        "message_count": session.get("message_count", 0),
        "created_at": session["created_at"],
        "updated_at": session.get("updated_at", session["created_at"]),
        "last_activity": session.get("last_activity", session["created_at"]),
    }

