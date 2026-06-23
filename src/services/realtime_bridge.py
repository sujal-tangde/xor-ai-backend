"""API-process side of realtime events.

Holds the in-memory registry of connected websockets (keyed by user id) and a
long-lived Redis subscriber that relays events published by the worker (see
``realtime``) to the right user's sockets. Lives only in the FastAPI process.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from src.core.config import REDIS_URL
from src.services.realtime import INSIGHTS_CHANNEL

logger = logging.getLogger(__name__)

# user_id -> set of live WebSocket connections for that user.
_connections: dict[str, set[Any]] = defaultdict(set)


def register(user_id: str, ws: Any) -> None:
    _connections[user_id].add(ws)


def unregister(user_id: str, ws: Any) -> None:
    conns = _connections.get(user_id)
    if conns is None:
        return
    conns.discard(ws)
    if not conns:
        _connections.pop(user_id, None)


async def _broadcast(user_id: str, message: dict) -> None:
    for ws in list(_connections.get(user_id, ())):
        try:
            await ws.send_json(message)
        except Exception:
            # The socket is dead/closing — drop it so we stop trying.
            unregister(user_id, ws)


async def run_subscriber() -> None:
    """Relay Redis pub/sub events to connected sockets until cancelled.

    Reconnects on any error so a transient Redis blip doesn't permanently kill
    realtime updates for the lifetime of the server process.
    """
    import redis.asyncio as aioredis

    while True:
        try:
            client = aioredis.from_url(REDIS_URL)
            pubsub = client.pubsub()
            await pubsub.subscribe(INSIGHTS_CHANNEL)
            logger.info("Realtime subscriber listening on %s", INSIGHTS_CHANNEL)
            async for raw in pubsub.listen():
                if raw.get("type") != "message":
                    continue
                try:
                    event = json.loads(raw["data"])
                except (ValueError, TypeError):
                    continue
                user_id = event.get("user_id")
                if not user_id:
                    continue
                await _broadcast(
                    user_id,
                    {
                        "type": event.get("type", "insights_progress"),
                        "project_id": event.get("project_id"),
                        "processed": event.get("processed", 0),
                        "total": event.get("total", 0),
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Realtime subscriber error; reconnecting in 2s")
            await asyncio.sleep(2)
