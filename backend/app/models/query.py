from typing import Any, Mapping


def query_to_public(query: Mapping[str, Any]) -> dict:
    return {
        "id": str(query["_id"]),
        "parent_id": query["parent_id"],
        "chat_id": query.get("chat_id"),
        "request": query["request"],
        "response": query["response"],
    }
