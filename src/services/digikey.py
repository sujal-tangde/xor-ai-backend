"""DigiKey Product Information API client for MPN resolution and pricing.

Used by the report-generation flow to resolve component markings into real
manufacturer part numbers and to fetch volume pricing for the BOM. The OAuth
client-credentials token is cached in-process and refreshed shortly before it
expires. All network access goes through a short-lived ``httpx`` client so the
worker/event loop is never blocked on a dangling connection.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from src.core.config import (
    DIGIKEY_API_BASE,
    DIGIKEY_CLIENT_ID,
    DIGIKEY_CLIENT_SECRET,
    DIGIKEY_LOCALE_CURRENCY,
    DIGIKEY_LOCALE_LANGUAGE,
    DIGIKEY_LOCALE_SITE,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 20.0
# Refresh the token this many seconds before it actually expires.
_TOKEN_REFRESH_MARGIN = 60.0

_token_lock = threading.Lock()
_token_value: str | None = None
_token_expiry: float = 0.0


def is_configured() -> bool:
    """True when DigiKey credentials are present in the environment."""
    return bool(DIGIKEY_CLIENT_ID and DIGIKEY_CLIENT_SECRET)


def _get_token() -> str:
    """Return a valid bearer token, fetching/refreshing it as needed."""
    global _token_value, _token_expiry
    if not is_configured():
        raise RuntimeError("DigiKey credentials are not configured.")

    now = time.monotonic()
    with _token_lock:
        if _token_value and now < _token_expiry - _TOKEN_REFRESH_MARGIN:
            return _token_value

        resp = httpx.post(
            f"{DIGIKEY_API_BASE}/v1/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": DIGIKEY_CLIENT_ID,
                "client_secret": DIGIKEY_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_value = data["access_token"]
        _token_expiry = now + float(data.get("expires_in", 600))
        return _token_value


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "X-DIGIKEY-Client-Id": DIGIKEY_CLIENT_ID,
        "X-DIGIKEY-Locale-Site": "IN",
        "X-DIGIKEY-Locale-Language": "DIGIKEY_LOCALE_LANGUAGE",
        "X-DIGIKEY-Locale-Currency": "INR",
        "Content-Type": "application/json",
    }


def _first_variation(product: dict[str, Any]) -> dict[str, Any]:
    variations = product.get("ProductVariations") or []
    return variations[0] if variations else {}


def _normalize_part(product: dict[str, Any]) -> dict[str, Any]:
    """Flatten a DigiKey product object into the fields we keep."""
    variation = _first_variation(product)
    status = (product.get("ProductStatus") or {}).get("Status")
    return {
        "mpn": product.get("ManufacturerProductNumber"),
        "manufacturer": (product.get("Manufacturer") or {}).get("Name"),
        "description": (product.get("Description") or {}).get("ProductDescription"),
        "package": (variation.get("PackageType") or {}).get("Name"),
        "quantity_available": product.get("QuantityAvailable"),
        "price_breaks": variation.get("StandardPricing") or [],
        "status": status,
        "discontinued": bool(product.get("Discontinued")),
        "end_of_life": bool(product.get("EndOfLife")),
        "datasheet_url": product.get("DatasheetUrl"),
        "product_url": product.get("ProductUrl"),
    }


def search_keyword(keywords: str, limit: int = 8) -> list[dict[str, Any]]:
    """Keyword search for candidate parts. Returns a list of normalized parts."""
    keywords = (keywords or "").strip()
    if not keywords:
        return []
    resp = httpx.post(
        f"{DIGIKEY_API_BASE}/products/v4/search/keyword",
        headers=_headers(),
        json={"Keywords": keywords, "Limit": max(1, min(limit, 50))},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    products = resp.json().get("Products") or []
    return [_normalize_part(p) for p in products]


def product_details(mpn: str) -> dict[str, Any] | None:
    """Full pricing/details for a resolved MPN, or None if not found."""
    mpn = (mpn or "").strip()
    if not mpn:
        return None
    resp = httpx.get(
        f"{DIGIKEY_API_BASE}/products/v4/search/{httpx.URL(mpn)}/productdetails",
        headers=_headers(),
        timeout=_HTTP_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    product = resp.json().get("Product")
    return _normalize_part(product) if product else None


def price_for_qty(price_breaks: list[dict[str, Any]], qty: int) -> float | None:
    """Unit price for an order of ``qty`` units from a list of price breaks.

    Picks the unit price of the highest break quantity that is <= qty (i.e. the
    price tier the order actually qualifies for). Falls back to the lowest break
    if qty is below every break.
    """
    if not price_breaks:
        return None
    try:
        breaks = sorted(
            (
                (int(b.get("BreakQuantity", 1) or 1), float(b.get("UnitPrice")))
                for b in price_breaks
                if b.get("UnitPrice") is not None
            ),
            key=lambda x: x[0],
        )
    except (TypeError, ValueError):
        return None
    if not breaks:
        return None
    chosen = breaks[0][1]
    for break_qty, unit_price in breaks:
        if break_qty <= qty:
            chosen = unit_price
        else:
            break
    return chosen
