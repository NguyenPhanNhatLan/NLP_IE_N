from app.db.mongo import get_database

async def get_db():
    return await get_database()
