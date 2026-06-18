"""pgvector helpers for the image_chunks / file_chunks RAG tables.

Uses the Supabase direct Postgres connection (``DIRECT_URL``) because pgvector's
similarity operators aren't reachable through the Supabase REST client. Inserts
and searches always carry ``project_id`` so a search is pre-filtered to a single
project before cosine ranking.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from src.core.config import DIRECT_URL

logger = logging.getLogger(__name__)

_ALLOWED_TABLES = {"image_chunks", "file_chunks"}


def _vector_literal(vec: list[float]) -> str:
    """Format a float list as a pgvector literal: '[1,2,3]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def insert_chunks(table: str, rows: list[dict[str, Any]]) -> int:
    """Bulk-insert chunk rows into image_chunks/file_chunks.

    Each row: {project_id, user_id, file_id, content, chunk_index, embedding,
    metadata}. ``embedding`` is a list[float]; ``metadata`` is a dict.
    Returns the number of rows inserted.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unknown vector table: {table}")
    if not rows or not DIRECT_URL:
        return 0

    values = [
        (
            row["project_id"],
            row.get("user_id"),
            row["file_id"],
            row["content"],
            row.get("chunk_index"),
            _vector_literal(row["embedding"]),
            json.dumps(row.get("metadata") or {}),
        )
        for row in rows
    ]

    sql = (
        f"INSERT INTO {table} "
        "(project_id, user_id, file_id, content, chunk_index, embedding, metadata) "
        "VALUES %s"
    )
    template = "(%s, %s, %s, %s, %s, %s::vector, %s::jsonb)"

    conn = psycopg2.connect(DIRECT_URL)
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template=template)
        conn.commit()
    finally:
        conn.close()
    return len(values)


def search_chunks(
    table: str,
    project_id: str,
    query_embedding: list[float],
    k: int = 8,
) -> list[dict[str, Any]]:
    """Cosine top-k over a chunk table, pre-filtered by project_id.

    Returns rows with file_id, content, chunk_index, metadata, and distance
    (lower = closer), ordered most-similar first.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unknown vector table: {table}")
    if not DIRECT_URL:
        return []

    vec = _vector_literal(query_embedding)
    sql = (
        "SELECT file_id, content, chunk_index, metadata, "
        "(embedding <=> %s::vector) AS distance "
        f"FROM {table} "
        "WHERE project_id = %s "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    )

    conn = psycopg2.connect(DIRECT_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (vec, project_id, vec, k))
            rows = cur.fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]
