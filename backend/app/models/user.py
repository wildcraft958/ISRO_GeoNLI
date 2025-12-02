from typing import Any, Mapping


def user_to_public(user: Mapping[str, Any]) -> dict:
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "username": user.get("first_name"),
    }
