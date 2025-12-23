import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

from app.core.config import settings

load_dotenv()
logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None


async def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is not None:
        return _client

    try:
        client = AsyncIOMotorClient(settings.mongo_uri)
        await client.admin.command("ping")
        logger.info("MongoDB async connected")
        _client = client
        return client
    except PyMongoError as e:
        raise RuntimeError(f"MongoDB connection failed: {e}") from e


async def get_database() -> AsyncIOMotorDatabase:
    client = await get_mongo_client()
    return client[settings.mongo_db]


async def get_ner_collection() -> AsyncIOMotorCollection:
    db = await get_database()
    return db[settings.ner_collection_name]


async def get_re_collection() -> AsyncIOMotorCollection:
    db = await get_database()
    return db[settings.re_collection_name]


async def init_indexes() -> None:
    db = await get_database()

    collection_names = [
        settings.ner_collection_name,
        settings.re_collection_name,
    ]

    for name in collection_names:
        if not name:
            continue

        await db[name].create_index(
            [("source_id", 1)],
            unique=True,
            name="uniq_source_id",
            partialFilterExpression={"source_id": {"$exists": True, "$type": "string"}},
        )
