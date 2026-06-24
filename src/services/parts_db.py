"""Read-only lookups against the external parts database (JLCPCB dataset).

A separate Postgres instance (configured via ``PG_*`` in ``.env``, kept distinct
from the app's own ``DIRECT_URL`` database) holding ~7M components with MPN,
stock, and quantity-break pricing. Used to check whether a manufacturer part
number exists and return its sourcing/pricing detail.

Strictly read-only. The connection is pooled and created lazily on first use, so
importing this module never opens a socket. Schema/table come from env, so they
are injected as quoted identifiers (never string-formatted) to avoid injection.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from src.core.config import (
    PG_DATABASE,
    PG_ENABLED,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_TABLE,
    PG_USER,
)

logger = logging.getLogger(__name__)

# Columns returned per component. A whitelist keeps the response shape stable and
# avoids leaking the raw ``jlc_extra`` blob unless we explicitly decide to.
_SELECT_COLUMNS = (
    "mpn",
    "lcsc",
    "manufacturer",
    "description",
    "category",
    "subcategory",
    "package",
    "stock",
    "basic",
    "preferred",
    "datasheet",
    "price",
)

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


class PartsDBError(Exception):
    """Raised when the parts DB is unconfigured/disabled or a query fails."""


def is_enabled() -> bool:
    """True when the parts DB is switched on and minimally configured."""
    return PG_ENABLED and bool(PG_HOST and PG_USER and PG_DATABASE)


def _table_ident() -> sql.Composed:
    """Safely-quoted ``schema.table`` identifier from env config."""
    return sql.Identifier(PG_SCHEMA, PG_TABLE)


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                if not is_enabled():
                    raise PartsDBError(
                        "Parts DB is disabled or not configured (set PG_ENABLED and PG_* in .env)."
                    )
                _pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,
                    host=PG_HOST,
                    port=PG_PORT,
                    user=PG_USER,
                    password=PG_PASSWORD,
                    dbname=PG_DATABASE,
                    connect_timeout=10,
                    # Read-only: a failed transaction can't accidentally write.
                    options="-c default_transaction_read_only=on",
                )
                logger.info("Parts DB pool initialized for %s.%s", PG_SCHEMA, PG_TABLE)
    return _pool


@contextmanager
def _connection():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def ping() -> dict[str, Any]:
    """Cheap health check: confirms the table is reachable. Returns table info."""
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(_table_ident()))
            cur.fetchone()
    return {"ok": True, "table": f"{PG_SCHEMA}.{PG_TABLE}"}


def _price_bounds(price: Any) -> tuple[float | None, float | None]:
    """Min/max unit price across the quantity-break tiers (``[{qFrom,qTo,price}]``)."""
    if not isinstance(price, list):
        return None, None
    values = [
        t["price"]
        for t in price
        if isinstance(t, dict) and isinstance(t.get("price"), (int, float))
    ]
    return (min(values), max(values)) if values else (None, None)


def _shape(row: dict[str, Any]) -> dict[str, Any]:
    """Add convenience fields (price bounds) without dropping the raw tiers."""
    lo, hi = _price_bounds(row.get("price"))
    row["unit_price_min"] = lo
    row["unit_price_max"] = hi
    return row


def lookup_mpns(mpns: list[str]) -> list[dict[str, Any]]:
    """Exact (case-insensitive) existence + detail lookup for a list of MPNs.

    Returns one result per *requested* MPN, preserving input order and de-duping
    case-insensitively: ``{"mpn", "found", "component"}``. ``component`` is the
    matched row (with ``unit_price_min``/``max`` added) or ``None`` if not found.
    Uses the ``lower(mpn)`` functional index, so it stays fast on the 7M-row table.
    """
    # De-dupe case-insensitively while preserving the first-seen original spelling.
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in mpns:
        cleaned = (raw or "").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            ordered.append(cleaned)
    if not ordered:
        return []

    lowered = [m.lower() for m in ordered]
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in _SELECT_COLUMNS)
    query = sql.SQL("SELECT {cols} FROM {table} WHERE lower(mpn) = ANY(%s)").format(
        cols=columns, table=_table_ident()
    )

    with _connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (lowered,))
            rows = cur.fetchall()

    by_lower = {row["mpn"].lower(): _shape(dict(row)) for row in rows}
    return [
        {
            "mpn": original,
            "found": original.lower() in by_lower,
            "component": by_lower.get(original.lower()),
        }
        for original in ordered
    ]
