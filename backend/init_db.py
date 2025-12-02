import asyncio

from app.core.database import get_db


async def init_db():
    db = get_db()
    print("Database initialized!")

    users = db["users"]

    await users.create_index("email", unique=True)


if __name__ == "__main__":
    asyncio.run(init_db())
