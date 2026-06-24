"""Parts-DB-backed component pricing for the should-cost report.

Replaces the Mouser distributor lookup as the BOM pricing source. Prices come
from the internal parts database (JLCPCB dataset, see :mod:`src.services.parts_db`)
in **USD** and are converted to **INR** here using the live USD→INR rate, so the
rest of the pipeline keeps treating ``price_breaks`` as INR exactly as it did for
Mouser — no downstream change needed.

Batch-friendly: ``price_mpns`` resolves up to the parts-DB cap (300) MPNs per
query, chunking automatically, so the whole BOM is priced in one or two round
trips instead of one HTTP call per line.

Every public function is defensive: any DB/parse error logs, records a failure,
and yields an empty result rather than raising, so a single failed lookup can
never crash the report. The caller flips affected lines to a rate-card estimate
and continues.
"""

from __future__ import annotations

import logging
from typing import Any

from src.services import parts_db
from src.services.failure_log import record_failure

logger = logging.getLogger(__name__)

# parts_db caps a single lookup at 300 MPNs; chunk anything larger.
_MAX_MPNS = 300


def is_configured() -> bool:
    """True when the parts database is enabled and minimally configured."""
    return parts_db.is_enabled()


def price_for_qty(price_breaks: list[dict[str, Any]], qty: int) -> float | None:
    """Unit price for an order of ``qty`` units from a ``[{qty, price}]`` list.

    Picks the highest break quantity that is <= ``qty`` (the tier the order
    qualifies for); falls back to the lowest break if ``qty`` is below every
    break. The breaks are derived from the parts-DB ``qFrom`` boundaries, so the
    selected tier is exactly the one whose [qFrom, qTo] range contains ``qty``.
    """
    breaks = sorted(
        (
            (int(b["qty"]), float(b["price"]))
            for b in (price_breaks or [])
            if b.get("qty") and b.get("price") is not None
        ),
        key=lambda x: x[0],
    )
    if not breaks:
        return None
    chosen = breaks[0][1]
    for break_qty, unit_price in breaks:
        if break_qty <= qty:
            chosen = unit_price
        else:
            break
    return chosen


def _normalize_component(component: dict[str, Any], fx_rate: float) -> dict[str, Any]:
    """Flatten one parts-DB row into the fields the pipeline keeps (INR prices).

    The parts-DB ``price`` tiers are ``[{qFrom, qTo, price}]`` in USD. We convert
    each tier to a Mouser-style ``{qty, price}`` break with ``qty = qFrom`` and
    ``price = price_usd * fx_rate``, so the rest of the pipeline can keep using
    :func:`price_for_qty` unchanged.
    """
    breaks: list[dict[str, Any]] = []
    for tier in component.get("price") or []:
        if not isinstance(tier, dict):
            continue
        try:
            qty = int(tier.get("qFrom") or 0)
        except (TypeError, ValueError):
            qty = 0
        raw = tier.get("price")
        if qty > 0 and isinstance(raw, (int, float)):
            breaks.append({"qty": qty, "price": round(float(raw) * fx_rate, 4)})
    breaks.sort(key=lambda b: b["qty"])
    return {
        "mpn": component.get("mpn"),
        "manufacturer": component.get("manufacturer"),
        "description": component.get("description"),
        "price_breaks": breaks,
        # The parts DB carries no HSN/HTS code — duty classification fills it in.
        "hsn": None,
        "datasheet_url": component.get("datasheet") or None,
        "availability": component.get("stock"),
        "currency": "INR",
    }


def price_mpns(mpns: list[str], fx_rate: float) -> dict[str, dict[str, Any]]:
    """Batch-price a list of MPNs against the parts DB.

    Returns a map of ``lower(mpn) -> normalized part`` (INR ``price_breaks``) for
    every MPN that was found *and* carried usable price breaks. MPNs that are not
    found, carry no pricing, or fail to look up are simply absent from the map, so
    the caller falls back to a rate-card estimate for those lines.

    Never raises: on any DB error it logs, records a failure, and skips that chunk.
    """
    if not is_configured() or not fx_rate or fx_rate <= 0:
        return {}

    # De-dupe case-insensitively, preserving the first-seen original spelling.
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in mpns:
        cleaned = (raw or "").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            ordered.append(cleaned)
    if not ordered:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ordered), _MAX_MPNS):
        chunk = ordered[start : start + _MAX_MPNS]
        try:
            results = parts_db.lookup_mpns(chunk)
        except Exception as exc:  # noqa: BLE001 - defensive: never crash the pipeline
            logger.warning("Parts DB pricing lookup failed for %d MPNs: %s", len(chunk), exc)
            record_failure(
                "pricing", ", ".join(chunk[:5]) + ("…" if len(chunk) > 5 else ""),
                "Parts database lookup failed — using rate-card estimates",
                error=exc, context={"mpn_count": len(chunk)},
            )
            continue
        for res in results:
            if not (res.get("found") and isinstance(res.get("component"), dict)):
                continue
            part = _normalize_component(res["component"], fx_rate)
            if part.get("price_breaks"):
                out[str(res["mpn"]).strip().lower()] = part
    return out


def price_one(mpn: str, fx_rate: float) -> dict[str, Any] | None:
    """Single-MPN convenience wrapper around :func:`price_mpns` (for the edit flow).

    Returns the normalized part (INR ``price_breaks``) or ``None`` if not found /
    not priced. Never raises.
    """
    mpn = (mpn or "").strip()
    if not mpn:
        return None
    return price_mpns([mpn], fx_rate).get(mpn.lower())
