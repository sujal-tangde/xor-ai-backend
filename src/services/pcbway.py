"""PCBWay partner API client for PCB fabrication quotes.

Quotes the bare board at each order quantity in the volume curve. PCBWay returns
USD prices, which the pipeline converts to INR via :mod:`src.services.fx`.

If PCBWay is unavailable (no key, timeout, error, or an unrecognized response),
:func:`quote_board` returns ``None`` and the pipeline falls back to an internal
area × layers rate-card heuristic (:func:`heuristic_quote_inr`). Either way the
report always gets a fab cost; the source/tag just flips from live to estimate.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.core.config import PCBWAY_API_BASE, PCBWAY_API_KEY
from src.services.failure_log import record_failure

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 25.0


def is_configured() -> bool:
    return bool(PCBWAY_API_KEY)


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_payload(pcb: dict[str, Any], qty: int) -> dict[str, Any]:
    """Map the KB ``pcb`` block to a PCBWay PcbQuotation request payload."""
    pcb = pcb or {}
    dims = pcb.get("dimensions_mm") or {}
    length = _num(dims.get("length"), 50) or 50
    width = _num(dims.get("width"), 50) or 50
    thickness = _num(dims.get("thickness"), 1.6) or 1.6
    layers = int(_num(pcb.get("layer_count_estimate"), 2) or 2)
    soldermask = pcb.get("soldermask_color") or "Green"
    surface = pcb.get("surface_finish") or "HASL with lead"
    return {
        "Length": round(length, 2),
        "width": round(width, 2),
        "layers": layers,
        "Thickness": thickness,
        "SolderMask": soldermask,
        "surface": surface,
        "Material": "FR-4",
        "Qty": qty,
        "Unit": "mm",
        "Quantity": qty,
    }


def _extract_usd_price(data: dict[str, Any]) -> float | None:
    """Best-effort extraction of a USD price from PCBWay's response shape."""
    if not isinstance(data, dict):
        return None
    # Common shapes: {"result": {"price": "12.34"}} / {"Price": 12.34} /
    # {"result": {"priceList":[{"Price": ...}]}}.
    for key in ("Price", "price", "totalPrice", "TotalPrice"):
        val = _num(data.get(key))
        if val is not None:
            return val
    result = data.get("result") if isinstance(data.get("result"), dict) else None
    if result:
        for key in ("Price", "price", "totalPrice", "TotalPrice"):
            val = _num(result.get(key))
            if val is not None:
                return val
        price_list = result.get("priceList") or result.get("PriceList")
        if isinstance(price_list, list):
            total = 0.0
            found = False
            for row in price_list:
                if isinstance(row, dict):
                    val = _num(row.get("Price") or row.get("price"))
                    if val is not None:
                        total += val
                        found = True
            if found:
                return total
    return None


def quote_board(pcb: dict[str, Any], qty: int) -> dict[str, Any] | None:
    """Live PCBWay quote for one quantity. Returns ``{usd, qty, raw}`` or None.

    Never raises — any failure returns None so the pipeline uses the heuristic.
    """
    if not is_configured():
        return None
    try:
        resp = httpx.post(
            f"{PCBWAY_API_BASE}/api/Pcb/PcbQuotation",
            headers={
                "api-key": PCBWAY_API_KEY,
                "Content-Type": "application/json",
            },
            json=build_payload(pcb, qty),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - defensive
        logger.warning("PCBWay quote failed for qty %s: %s", qty, exc)
        record_failure(
            "fab_quote", "PCBWay quote",
            "PCBWay API request failed — using the internal fab cost model",
            error=exc, context={"qty": qty, "payload": build_payload(pcb, qty)},
        )
        return None

    usd = _extract_usd_price(data)
    if usd is None:
        logger.warning("PCBWay returned no recognizable price for qty %s", qty)
        record_failure(
            "fab_quote", "PCBWay quote",
            "PCBWay returned no recognizable price — using the internal fab cost model",
            context={"qty": qty, "response_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
        return None
    return {"usd": round(usd, 4), "qty": qty, "raw": data}


def heuristic_quote_inr(pcb: dict[str, Any], qty: int) -> float:
    """Internal fab cost fallback: area (cm²) × layers × rate, INR per board.

    Cheaper per board at higher volume. Always returns a positive number so the
    report's fab line is never empty.
    """
    pcb = pcb or {}
    dims = pcb.get("dimensions_mm") or {}
    length = _num(dims.get("length"), 50) or 50
    width = _num(dims.get("width"), 50) or 50
    layers = int(_num(pcb.get("layer_count_estimate"), 2) or 2)
    area_cm2 = (length / 10.0) * (width / 10.0)

    # INR per cm² per layer, declining with volume; plus a small per-board floor.
    if qty <= 1:
        rate = 9.0
        floor = 900.0
    elif qty <= 100:
        rate = 2.2
        floor = 110.0
    elif qty <= 1000:
        rate = 1.2
        floor = 45.0
    else:
        rate = 0.9
        floor = 30.0
    return round(max(floor, area_cm2 * max(layers, 1) * rate), 2)
