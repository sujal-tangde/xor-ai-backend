"""Single recomputed whole-project picture (project_knowledge_base).

Each new ``project_insights`` row triggers an incremental merge: the one new
insight is folded into the existing KB row's theory/structured via the LLM, and
the processed counter is bumped (e.g. 19/20 -> 20/20). A Postgres session-level
advisory lock keyed on the project serialises recomputes across RQ workers so two
insights for the same project can't clobber each other's merge.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from src.core.config import DIRECT_URL
from src.services.llm_analysis import merge_dual

logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Normalise a JSONB column value (psycopg2 may hand back str or dict)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _structured_text(value: dict[str, Any] | None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) if value else ""


def recompute_knowledge_base(project_id: str, insight_id: str) -> None:
    """Fold the insight ``insight_id`` into the project's knowledge-base row."""
    if not DIRECT_URL:
        logger.warning("DIRECT_URL not set; skipping KB recompute for %s", project_id)
        return

    conn = psycopg2.connect(DIRECT_URL)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            # Serialise per-project across workers for the whole recompute.
            cur.execute("SELECT pg_advisory_lock(hashtext(%s)::bigint)", (project_id,))

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT user_id, theory_context, structured_context "
                    "FROM project_insights WHERE id = %s",
                    (insight_id,),
                )
                insight = cur.fetchone()
                if not insight:
                    logger.warning("Insight %s not found; skipping KB recompute", insight_id)
                    return

                cur.execute(
                    "SELECT COUNT(*) AS n FROM project_insights WHERE project_id = %s",
                    (project_id,),
                )
                total = int(cur.fetchone()["n"])

                cur.execute(
                    "SELECT theory_context, structured_context, insights_processed "
                    "FROM project_knowledge_base WHERE project_id = %s",
                    (project_id,),
                )
                kb = cur.fetchone()

            new_theory = (insight.get("theory_context") or "").strip()
            new_structured = _as_dict(insight.get("structured_context"))
            user_id = insight.get("user_id")

            existing_theory = (kb.get("theory_context") or "").strip() if kb else ""
            existing_structured = _as_dict(kb.get("structured_context")) if kb else None
            processed = int(kb.get("insights_processed") or 0) if kb else 0

            if not existing_theory and not existing_structured and processed == 0:
                # First insight for the project: KB becomes it verbatim, no LLM call.
                merged_theory = new_theory
                merged_structured = new_structured
            else:
                merged_theory, merged_structured = merge_dual(
                    existing_theory,
                    _structured_text(existing_structured),
                    new_theory,
                    _structured_text(new_structured),
                )
                merged_theory = (merged_theory or "").strip() or existing_theory
                if merged_structured is None:
                    # Merge produced unparseable JSON — keep the previous structured
                    # context rather than corrupting it.
                    logger.warning(
                        "KB merge returned unparseable JSON for project %s; "
                        "keeping previous structured_context",
                        project_id,
                    )
                    merged_structured = existing_structured or new_structured

            processed_now = processed + 1

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_knowledge_base
                        (project_id, user_id, theory_context, structured_context,
                         insights_total, insights_processed, status, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id) DO UPDATE SET
                        user_id = COALESCE(project_knowledge_base.user_id, EXCLUDED.user_id),
                        theory_context = EXCLUDED.theory_context,
                        structured_context = EXCLUDED.structured_context,
                        insights_total = EXCLUDED.insights_total,
                        insights_processed = EXCLUDED.insights_processed,
                        status = EXCLUDED.status,
                        updated_at = now()
                    """,
                    (
                        project_id,
                        user_id,
                        merged_theory,
                        Json(merged_structured) if merged_structured is not None else None,
                        total,
                        processed_now,
                        "ready",
                    ),
                )
            logger.info(
                "KB recompute for project %s: %s/%s insights processed",
                project_id,
                processed_now,
                total,
            )
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtext(%s)::bigint)", (project_id,))
    finally:
        conn.close()
