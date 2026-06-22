"""Mouser Search API client for BOM component pricing.

Used by the should-cost report pipeline to price resolved MPNs. Mouser is
queried with ONLY the ``apiKey`` query param and a ``Content-Type`` header — no
session/cookie header (those are stale and unnecessary).

IMPORTANT: prices returned here are already in INR (the Mouser account/locale is
INR), so they are NEVER FX-converted downstream — only the PCBWay USD quote is.

Every public function is defensive: any network/parse error returns ``None`` (or
an empty result) rather than raising, so a single failed line can never crash the
report. The caller flips that line to a rate-card estimate and continues.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from src.core.config import MOUSER_API_BASE, MOUSER_API_KEY
from src.services.failure_log import record_failure

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 20.0

# Mouser standard quantity price breaks. The volume-curve selection picks the
# highest break <= target volume; for 10,000 that resolves to the 1,000 break.
_BREAK_QUANTITIES = (1, 10, 25, 100, 250, 500, 1000)


def is_configured() -> bool:
    """True when a Mouser API key is present."""
    return bool(MOUSER_API_KEY)


def _to_float_price(raw: Any) -> float | None:
    """Parse a Mouser price string (e.g. '₹152.70', '$1.23', '1,234.50')."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw)
    # Strip everything that isn't a digit, dot, or minus (drops ₹/$/commas/spaces).
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
    if not cleaned or cleaned in {".", "-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_price_breaks(raw_breaks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return sorted [{qty, price}] breaks from Mouser's PriceBreaks list."""
    out: list[dict[str, Any]] = []
    for brk in raw_breaks or []:
        if not isinstance(brk, dict):
            continue
        try:
            qty = int(brk.get("Quantity") or 0)
        except (TypeError, ValueError):
            qty = 0
        price = _to_float_price(brk.get("Price"))
        if qty > 0 and price is not None:
            out.append({"qty": qty, "price": price})
    out.sort(key=lambda b: b["qty"])
    return out


def _extract_hsn(compliance: list[dict[str, Any]] | None) -> str | None:
    """Pull the HSN/HTS code from Mouser ProductCompliance (CNHTS preferred)."""
    if not compliance:
        return None
    by_name: dict[str, str] = {}
    for item in compliance:
        if not isinstance(item, dict):
            continue
        name = str(item.get("ComplianceName") or "").upper()
        value = str(item.get("ComplianceValue") or "").strip()
        if name and value:
            by_name[name] = value
    for key in ("CNHTS", "USHTS", "HSN", "HTS"):
        if by_name.get(key):
            return by_name[key]
    return None


def _normalize_part(part: dict[str, Any]) -> dict[str, Any]:
    """Flatten one Mouser part record into the fields the pipeline keeps."""
    return {
        "mpn": part.get("ManufacturerPartNumber"),
        "manufacturer": part.get("Manufacturer"),
        "description": part.get("Description"),
        "price_breaks": _normalize_price_breaks(part.get("PriceBreaks")),
        "hsn": _extract_hsn(part.get("ProductCompliance")),
        "datasheet_url": part.get("DataSheetUrl") or None,
        "availability": part.get("Availability"),
        "currency": (part.get("PriceBreaks") or [{}])[0].get("Currency")
        if part.get("PriceBreaks")
        else None,
    }


def search_part(mpn: str) -> dict[str, Any] | None:
    """Exact-match search for a single MPN. Returns a normalized part or None.

    Never raises — on any error (timeout, HTTP error, no key, not found) it logs
    and returns None so the caller can fall back to a rate-card estimate.
    """
    mpn = (mpn or "").strip()
    if not mpn or not is_configured():
        return None
    try:
        resp = httpx.post(
            f"{MOUSER_API_BASE}/search/partnumber",
            params={"apiKey": MOUSER_API_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "SearchByPartRequest": {
                    "mouserPartNumber": mpn,
                    "partSearchOptions": "Exact",
                }
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - defensive: never crash the pipeline
        logger.warning("Mouser lookup failed for %s: %s", mpn, exc)
        record_failure(
            "pricing", mpn, "Mouser API request failed — using a rate-card estimate",
            error=exc, context={"mpn": mpn, "endpoint": "search/partnumber"},
        )
        return None

    results = (data or {}).get("SearchResults") or {}
    parts = results.get("Parts") or []
    if not parts:
        record_failure(
            "pricing", mpn, "Mouser returned no matching part — using a rate-card estimate",
            context={"mpn": mpn, "mouser_errors": (data or {}).get("Errors")},
        )
        return None

    # Prefer an exact MPN match; otherwise take the first part with price breaks.
    chosen = None
    for part in parts:
        if str(part.get("ManufacturerPartNumber") or "").strip().lower() == mpn.lower():
            chosen = part
            break
    if chosen is None:
        chosen = next(
            (p for p in parts if p.get("PriceBreaks")),
            parts[0],
        )
    return _normalize_part(chosen)


def price_for_qty(price_breaks: list[dict[str, Any]], qty: int) -> float | None:
    """Unit price for an order of ``qty`` units.

    Picks the highest break quantity that is <= ``qty`` (the tier the order
    qualifies for). For 10,000 this resolves to the 1,000 break, since 1,000 is
    Mouser's largest standard break. Falls back to the lowest break if ``qty`` is
    below every break.
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
