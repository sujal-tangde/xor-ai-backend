"""Per-upload insight rows (project_insights) and the trigger into KB recompute."""

from __future__ import annotations

import logging
from typing import Any

from src.services.file_storage import get_supabase
from src.services.queue import enqueue_knowledge_base_recompute

logger = logging.getLogger(__name__)


def record_insight(
    project_id: str,
    user_id: str | None,
    file_id: str,
    media_kind: str,
    theory: str,
    structured: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Insert one project_insights row for an upload, then trigger KB recompute.

    Returns the inserted row (with its id), or None if the insert returned no data.
    """
    row = {
        "project_id": project_id,
        "user_id": user_id,
        "file_id": file_id,
        "media_kind": media_kind,
        "theory_context": (theory or "").strip(),
        "structured_context": structured,
    }
    result = get_supabase().table("project_insights").insert(row).execute()
    record = result.data[0] if result.data else None

    insight_id = record.get("id") if record else None
    if insight_id:
        enqueue_knowledge_base_recompute(project_id, str(insight_id))
    else:
        logger.warning(
            "Insight insert for file %s returned no id; KB recompute not enqueued",
            file_id,
        )
    return record
