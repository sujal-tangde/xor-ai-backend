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
import math
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from src.core.config import DIRECT_URL, KB_RELATEDNESS_THRESHOLD
from src.services.llm_analysis import merge_dual

logger = logging.getLogger(__name__)

# Structured keys whose list lengths approximate "how much we know". Used by the
# anti-shrink guard to detect a merge that silently dropped entries.
_RICHNESS_LIST_KEYS = (
    "components",
    "connectors_io",
    "architecture_blocks",
    "design_observations",
    "assumptions",
)


def _kb_text(theory: str, structured: dict[str, Any] | None) -> str:
    """Flatten a (theory, structured) pair into one string for embedding."""
    parts = [(theory or "").strip()]
    if structured:
        parts.append(json.dumps(structured, ensure_ascii=False))
    return "\n\n".join(p for p in parts if p).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _is_related(existing_text: str, new_text: str) -> bool:
    """True if the new insight is at least loosely related to the existing KB.

    Fails open: any embedding error or a disabled threshold returns True so a
    transient problem never blocks a legitimate merge.
    """
    if KB_RELATEDNESS_THRESHOLD <= 0:
        return True
    if not existing_text or not new_text:
        return True
    try:
        from src.services.embeddings import embed_texts

        vec_existing, vec_new = embed_texts([existing_text, new_text])
        score = _cosine(vec_existing, vec_new)
    except Exception:
        logger.exception("Relatedness check failed; folding insight in anyway")
        return True
    logger.info("KB relatedness score %.3f (threshold %.3f)", score, KB_RELATEDNESS_THRESHOLD)
    return score >= KB_RELATEDNESS_THRESHOLD


def _richness(structured: dict[str, Any] | None) -> int:
    """Rough count of recorded facts, used to detect a lossy merge."""
    if not structured:
        return 0
    total = 0
    for key in _RICHNESS_LIST_KEYS:
        value = structured.get(key)
        if isinstance(value, list):
            total += len(value)
    return total


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


def _counts(cur: Any, project_id: str) -> tuple[int, int]:
    """Return (total insight rows, rows already folded into the KB).

    ``processed`` is derived from the ``processed_at`` marker so it can't drift
    away from reality the way the old +1 counter did.
    """
    cur.execute(
        "SELECT count(*) AS total, "
        "count(*) FILTER (WHERE processed_at IS NOT NULL) AS processed "
        "FROM project_insights WHERE project_id = %s",
        (project_id,),
    )
    row = cur.fetchone()
    return int(row["total"] or 0), int(row["processed"] or 0)


def _resync_counts(conn: Any, project_id: str, user_id: str | None) -> None:
    """Recompute the KB row's counters from ground truth and push them live.

    Used on the idempotent skip path (insight already processed) so a duplicate
    job still leaves the displayed total/processed correct without re-merging.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        total, processed = _counts(cur, project_id)
        cur.execute(
            "UPDATE project_knowledge_base "
            "SET insights_total = %s, insights_processed = %s, updated_at = now() "
            "WHERE project_id = %s",
            (total, processed, project_id),
        )
    from src.services.realtime import publish_insights_progress

    publish_insights_progress(user_id, project_id, processed, total)


def reconcile_unprocessed_insights(
    project_id: str | None = None, older_than_seconds: int = 120
) -> int:
    """Re-enqueue KB recompute for every insight still missing a ``processed_at``.

    The safety net for lost work: a recompute that died (worker crash, statement
    timeout) or one that was never enqueued at all (a Redis hiccup swallowed by
    ``enqueue_knowledge_base_recompute``). Because processing is now idempotent,
    re-enqueuing an insight is always safe — an already-folded one is skipped.
    Only rows older than ``older_than_seconds`` are considered so genuinely
    in-flight uploads aren't double-enqueued. Returns the number re-enqueued.
    """
    if not DIRECT_URL:
        return 0

    from src.services.queue import enqueue_knowledge_base_recompute

    conn = psycopg2.connect(DIRECT_URL)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            if project_id:
                cur.execute(
                    "SELECT project_id, id FROM project_insights "
                    "WHERE project_id = %s AND processed_at IS NULL "
                    "AND created_at < now() - make_interval(secs => %s)",
                    (project_id, older_than_seconds),
                )
            else:
                cur.execute(
                    "SELECT project_id, id FROM project_insights "
                    "WHERE processed_at IS NULL "
                    "AND created_at < now() - make_interval(secs => %s)",
                    (older_than_seconds,),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    for proj, insight_id in rows:
        enqueue_knowledge_base_recompute(str(proj), str(insight_id))
    if rows:
        logger.info("Reconcile re-enqueued %s unprocessed insight(s)", len(rows))
    return len(rows)


def recompute_knowledge_base(project_id: str, insight_id: str) -> None:
    """Fold the insight ``insight_id`` into the project's knowledge-base row."""
    if not DIRECT_URL:
        logger.warning("DIRECT_URL not set; skipping KB recompute for %s", project_id)
        return

    conn = psycopg2.connect(DIRECT_URL)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            # This is a background job whose critical section spans an LLM merge,
            # so the per-project advisory lock can be held for many seconds while
            # other workers' jobs queue behind it. Disable the platform
            # statement_timeout on this connection so neither the blocking lock
            # wait nor the merge queries get cancelled mid-flight — that timeout
            # firing on the lock wait was the cause of "canceling statement due to
            # statement timeout". A bounded lock_timeout still keeps a worker from
            # blocking forever if a holder is wedged (advisory locks are also
            # auto-released when their holding connection closes).
            cur.execute("SET statement_timeout = 0")
            cur.execute("SET lock_timeout = '180s'")

        # Serialise per-project across workers for the whole recompute. If the
        # lock can't be acquired within lock_timeout, hand the job back to the
        # queue instead of dropping the merge or tying up this worker.
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(hashtext(%s)::bigint)", (project_id,))
        except psycopg2.errors.LockNotAvailable:
            logger.warning(
                "KB recompute could not acquire lock for project %s within "
                "lock_timeout; re-enqueuing insight %s",
                project_id,
                insight_id,
            )
            from src.services.queue import enqueue_knowledge_base_recompute

            enqueue_knowledge_base_recompute(project_id, insight_id)
            return

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT user_id, theory_context, structured_context, processed_at "
                    "FROM project_insights WHERE id = %s",
                    (insight_id,),
                )
                insight = cur.fetchone()
                if not insight:
                    logger.warning("Insight %s not found; skipping KB recompute", insight_id)
                    return

                # Idempotency guard. processed_at is set only after this insight
                # has been folded into the KB, so a duplicate enqueue or a job
                # retry that follows a successful run must NOT merge it again —
                # that would double-count its facts into the theory/structured.
                # Just resync the derived counters (cheap) and return.
                if insight.get("processed_at") is not None:
                    logger.info(
                        "Insight %s already processed for project %s; resyncing counts only",
                        insight_id,
                        project_id,
                    )
                    _resync_counts(conn, project_id, insight.get("user_id"))
                    return

                cur.execute(
                    "SELECT theory_context, structured_context "
                    "FROM project_knowledge_base WHERE project_id = %s",
                    (project_id,),
                )
                kb = cur.fetchone()

            new_theory = (insight.get("theory_context") or "").strip()
            new_structured = _as_dict(insight.get("structured_context"))
            user_id = insight.get("user_id")

            existing_theory = (kb.get("theory_context") or "").strip() if kb else ""
            existing_structured = _as_dict(kb.get("structured_context")) if kb else None

            if kb is None or (not existing_theory and not existing_structured):
                # First insight for the project: KB becomes it verbatim, no LLM call.
                merged_theory = new_theory
                merged_structured = new_structured
            elif not _is_related(
                _kb_text(existing_theory, existing_structured),
                _kb_text(new_theory, new_structured),
            ):
                # The new upload has nothing to do with this product (e.g. a stray
                # photo). Don't fold it in and risk polluting the KB — keep the
                # existing analysis untouched. The insight row still exists so the
                # upload isn't lost; it's just not merged.
                logger.info(
                    "Insight %s is unrelated to project %s KB; skipping merge",
                    insight_id,
                    project_id,
                )
                merged_theory = existing_theory
                merged_structured = existing_structured
            else:
                merged_theory, merged_structured = merge_dual(
                    existing_theory,
                    _structured_text(existing_structured),
                    new_theory,
                    _structured_text(new_structured),
                )
                merged_theory = (merged_theory or "").strip()

                if merged_structured is None:
                    # Merge produced unparseable JSON — keep the previous structured
                    # context rather than corrupting it.
                    logger.warning(
                        "KB merge returned unparseable JSON for project %s; "
                        "keeping previous structured_context",
                        project_id,
                    )
                    merged_structured = existing_structured or new_structured
                elif _richness(merged_structured) < _richness(existing_structured):
                    # Anti-shrink guard: the merge dropped recorded facts. Folding
                    # in the new upload should never make us know LESS, so reject
                    # the lossy result and keep the richer existing structured.
                    logger.warning(
                        "KB merge shrank structured context for project %s "
                        "(%s -> %s facts); keeping previous structured_context",
                        project_id,
                        _richness(existing_structured),
                        _richness(merged_structured),
                    )
                    merged_structured = existing_structured

                # Same guard for the prose: never let the theory collapse to a
                # fraction of what we already had.
                if existing_theory and len(merged_theory) < 0.6 * len(existing_theory):
                    logger.warning(
                        "KB merge shrank theory for project %s "
                        "(%s -> %s chars); keeping previous theory_context",
                        project_id,
                        len(existing_theory),
                        len(merged_theory),
                    )
                    merged_theory = existing_theory
                elif not merged_theory:
                    merged_theory = existing_theory

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Mark this insight folded-in, then derive both counters from the
                # table itself rather than blindly bumping a stored +1. A lost or
                # double-fired job can no longer drift the count: processed is
                # always exactly "how many insight rows carry a processed_at".
                cur.execute(
                    "UPDATE project_insights SET processed_at = now() "
                    "WHERE id = %s AND processed_at IS NULL",
                    (insight_id,),
                )
                total, processed_now = _counts(cur, project_id)

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
            # Push the new processed/total counts to the user's UI in real time.
            from src.services.realtime import publish_insights_progress

            publish_insights_progress(user_id, project_id, processed_now, total)
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtext(%s)::bigint)", (project_id,))
    finally:
        conn.close()
