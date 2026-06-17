"""Redis-backed chat history (last N messages per conversation)."""

import json

from redis.asyncio import Redis

from src.core.config import MAX_CHAT_MESSAGES, REDIS_URL

_redis: Redis | None = None


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _messages_key(chat_id: str) -> str:
    return f"chat:{chat_id}:messages"


async def get_messages(chat_id: str) -> list[dict[str, str]]:
    redis = await get_redis()
    raw = await redis.lrange(_messages_key(chat_id), -MAX_CHAT_MESSAGES, -1)
    return [json.loads(item) for item in raw]


async def append_messages(chat_id: str, *messages: dict) -> None:
    redis = await get_redis()
    key = _messages_key(chat_id)
    if messages:
        await redis.rpush(key, *[json.dumps(msg) for msg in messages])
        await redis.ltrim(key, -MAX_CHAT_MESSAGES, -1)


async def count_messages(chat_id: str) -> int:
    redis = await get_redis()
    return await redis.llen(_messages_key(chat_id))


async def seed_messages(chat_id: str, messages: list[dict]) -> None:
    """Populate the Redis cache for a conversation (used for one-time hydration)."""
    if not messages:
        return
    redis = await get_redis()
    key = _messages_key(chat_id)
    await redis.rpush(key, *[json.dumps(msg) for msg in messages])
    await redis.ltrim(key, -MAX_CHAT_MESSAGES, -1)
