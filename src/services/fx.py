"""USD → INR conversion via Frankfurter (no API key).

Used ONLY to convert the PCBWay USD fab quote to INR. Mouser component prices are
already INR and must NOT pass through here.

The rate is fetched once and cached per request by the caller (the pipeline holds
the returned value for the whole run). A process-wide ``last known`` rate is kept
so that if Frankfurter is briefly down we degrade to the most recent good rate
before finally falling back to the hardcoded ``REPORT_USD_INR_FALLBACK``.
"""

from __future__ import annotations

import logging
import threading

import httpx

from src.core.config import FRANKFURTER_API_BASE, REPORT_USD_INR_FALLBACK
from src.services.failure_log import record_failure

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 12.0

_lock = threading.Lock()
_last_known_rate: float | None = None

# Source tags surfaced in the report / methodology.
SOURCE_LIVE = "frankfurter"
SOURCE_CACHED = "cached"
SOURCE_FALLBACK = "fallback"


def fetch_usd_inr() -> dict[str, object]:
    """Return ``{"rate": float, "source": str, "live": bool}`` for USD→INR.

    Never raises. On failure it returns the last-known cached rate (if any),
    otherwise the hardcoded fallback, with a ``source`` the report discloses.
    """
    global _last_known_rate
    try:
        resp = httpx.get(
            f"{FRANKFURTER_API_BASE}/latest",
            params={"base": "USD", "symbols": "INR"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rate = float((resp.json().get("rates") or {}).get("INR"))
        if rate > 0:
            with _lock:
                _last_known_rate = rate
            return {"rate": round(rate, 4), "source": SOURCE_LIVE, "live": True}
    except Exception as exc:  # noqa: BLE001 - defensive
        logger.warning("Frankfurter FX lookup failed: %s", exc)
        with _lock:
            cached = _last_known_rate
        record_failure(
            "fx", "Frankfurter USD/INR",
            "Live FX rate unavailable — using cached rate"
            if cached is not None else "Live FX rate unavailable — using fallback rate (~85)",
            error=exc,
            context={"fallback_used": "cached" if cached is not None else "hardcoded",
                     "cached_rate": cached, "hardcoded_rate": REPORT_USD_INR_FALLBACK},
        )
        if cached is not None:
            return {"rate": round(cached, 4), "source": SOURCE_CACHED, "live": False}
        return {"rate": float(REPORT_USD_INR_FALLBACK), "source": SOURCE_FALLBACK, "live": False}

    with _lock:
        cached = _last_known_rate
    if cached is not None:
        return {"rate": round(cached, 4), "source": SOURCE_CACHED, "live": False}
    return {
        "rate": float(REPORT_USD_INR_FALLBACK),
        "source": SOURCE_FALLBACK,
        "live": False,
    }
