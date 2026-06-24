"""Cross-process realtime events over Redis pub/sub.

The RQ worker records insights and recomputes KB counts in Postgres, but it runs
in a *separate process* from the FastAPI websocket server, so it cannot touch any
client socket directly. Instead it PUBLISHes small JSON events here; the API
process subscribes (see ``realtime_bridge``) and fans each event out to the
connected websocket(s) of the owning user.

This module is import-safe in the worker: it only touches Redis, never asyncio or
FastAPI. Every publish is best-effort — a Redis hiccup must never fail the job.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

INSIGHTS_CHANNEL = "rt:insights"


def publish_insights_progress(
    user_id: str | None,
    project_id: str,
    processed: int,
    total: int,
) -> None:
    """Publish a project's current insight counts to its owner's sockets."""
    if not user_id:
        return
    try:
        from src.services.queue import get_redis

        payload = json.dumps(
            {
                "type": "insights_progress",
                "user_id": str(user_id),
                "project_id": str(project_id),
                "processed": int(processed or 0),
                "total": int(total or 0),
            }
        )
        get_redis().publish(INSIGHTS_CHANNEL, payload)
    except Exception:
        logger.exception("Failed to publish insights progress for project %s", project_id)


def publish_project_counts(user_id: str | None, project_id: str) -> None:
    """Read the live insight counts for a project and publish them.

    Used right after a new insight row is recorded so the *total* climbs the
    instant the AI identifies an insight — before its (slower) KB recompute runs
    and bumps *processed*. ``total`` is the live ``project_insights`` count;
    ``processed`` is the KB row's counter (0 until the first recompute lands).
    """
    if not user_id:
        return
    try:
        from src.services.file_storage import get_supabase

        client = get_supabase()
        total_res = (
            client.table("project_insights")
            .select("id", count="exact")
            .eq("project_id", project_id)
            .execute()
        )
        total = total_res.count or 0
        kb_res = (
            client.table("project_knowledge_base")
            .select("insights_processed")
            .eq("project_id", project_id)
            .limit(1)
            .execute()
        )
        processed = (kb_res.data[0].get("insights_processed") if kb_res.data else 0) or 0
        publish_insights_progress(user_id, project_id, processed, total)
    except Exception:
        logger.exception("Failed to publish project counts for %s", project_id)
