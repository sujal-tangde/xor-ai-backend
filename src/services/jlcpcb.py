"""JLCPCB OpenAPI client for PCB fabrication quotes.

Replaces the old PCBWay client. Quotes the bare board at each order quantity in
the volume curve via ``POST /overseas/openapi/pcb/calculate``. JLCPCB returns the
board cost in USD (``data.pcbCostInfo.totalFee`` — the total for the order qty),
which we convert to a per-board USD figure and then to INR via
:mod:`src.services.fx`.

Requests are authenticated with an HMAC-SHA256 signature over
``METHOD\\nPATH\\nTIMESTAMP\\nNONCE\\nBODY\\n`` (the Python equivalent of the
Postman pre-request script), sent in a ``JOP`` Authorization header. The exact
serialized body that is signed is the exact body that is sent.

If JLCPCB is unavailable (no creds, timeout, error, or an unrecognized response),
:func:`quote_board` returns ``None`` and the pipeline falls back to an internal
area × layers rate-card heuristic (:func:`heuristic_quote_inr`). Either way the
report always gets a fab cost; the source/tag just flips from live to estimate.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any

import httpx

from src.core.config import (
    JLCPCB_ACCESS_KEY,
    JLCPCB_API_BASE,
    JLCPCB_APP_ID,
    JLCPCB_CALCULATE_PATH,
    JLCPCB_COUNTRY,
    JLCPCB_SECRET_KEY,
)
from src.services.failure_log import record_failure

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 25.0
_NONCE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# JLCPCB's minimum order quantity. A volume below this is quoted at the minimum
# and divided by the minimum, giving the realistic per-board cost at min order.
_MIN_QTY = 5


def is_configured() -> bool:
    return bool(JLCPCB_APP_ID and JLCPCB_ACCESS_KEY and JLCPCB_SECRET_KEY)


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _nonce() -> str:
    return "".join(secrets.choice(_NONCE_CHARS) for _ in range(32))


def _sign(method: str, path: str, body: str) -> dict[str, str]:
    """Build the signed ``JOP`` Authorization header for one request.

    Mirrors the Postman pre-request script exactly:
        stringToSign = METHOD\\nPATH\\nTIMESTAMP\\nNONCE\\nBODY\\n
        signature    = Base64( HMAC-SHA256(stringToSign, secretKey) )
    """
    timestamp = str(int(time.time()))
    nonce = _nonce()
    string_to_sign = f"{method}\n{path}\n{timestamp}\n{nonce}\n{body}\n"
    digest = hmac.new(
        JLCPCB_SECRET_KEY.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature = base64.b64encode(digest).decode("ascii")
    authorization = (
        f'JOP appid="{JLCPCB_APP_ID}",accesskey="{JLCPCB_ACCESS_KEY}",'
        f'nonce="{nonce}",timestamp="{timestamp}",signature="{signature}"'
    )
    return {"Content-Type": "application/json", "Authorization": authorization}


def build_request_payload(pcb: dict[str, Any], qty: int) -> dict[str, Any]:
    """Map the KB ``pcb`` block to a JLCPCB ``/pcb/calculate`` request body.

    Only the cost-driving structural fields are derived from the KB (layer count,
    dimensions, thickness, qty); the rest use JLCPCB's standard defaults (green
    soldermask, HASL finish, 1oz copper) — exactly what a should-cost baseline
    assumes when the teardown doesn't specify otherwise.
    """
    pcb = pcb or {}
    dims = pcb.get("dimensions_mm") or {}
    length = _num(dims.get("length"), 50) or 50
    width = _num(dims.get("width"), 50) or 50
    thickness = _num(dims.get("thickness"), 1.6) or 1.6
    layer = int(_num(pcb.get("layer_count_estimate"), 2) or 2)
    return {
        "orderType": 1,
        "achieveDate": 5,
        "country": JLCPCB_COUNTRY,
        "fileKey": "",
        "pcbParam": {
            "plateType": 1,
            "layer": layer,
            "length": round(length, 2),
            "width": round(width, 2),
            "qty": int(qty),
            "thickness": thickness,
            "materialDetails": 0,
            "pcbColor": 0,
            "surfaceFinish": 0,
            "viaCovering": 1,
            "goldFinger": 0,
            "panelFlag": 0,
            "differentDesign": 1,
            "copperWeight": 1,
            "serviceConfigVos": [],
        },
    }


def template_params(pcb: dict[str, Any]) -> dict[str, Any]:
    """The fab parameters surfaced in the report's PCB Fabrication section.

    Kept in the same shape the report template reads (Material/layers/Length/
    width/Thickness/SolderMask/surface), derived from the KB where available.
    """
    pcb = pcb or {}
    dims = pcb.get("dimensions_mm") or {}
    return {
        "Length": round(_num(dims.get("length"), 50) or 50, 2),
        "width": round(_num(dims.get("width"), 50) or 50, 2),
        "layers": int(_num(pcb.get("layer_count_estimate"), 2) or 2),
        "Thickness": _num(dims.get("thickness"), 1.6) or 1.6,
        "SolderMask": pcb.get("soldermask_color") or "Green",
        "surface": pcb.get("surface_finish") or "HASL with lead",
        "Material": "FR-4",
    }


def _extract_total_usd(data: dict[str, Any]) -> float | None:
    """Pull the board cost (USD, total for the order qty) from the response."""
    if not isinstance(data, dict):
        return None
    payload = data.get("data") if isinstance(data.get("data"), dict) else None
    if not payload:
        return None
    # Preferred: the (possibly promotional) board cost JLCPCB actually charges.
    cost_info = payload.get("pcbCostInfo") if isinstance(payload.get("pcbCostInfo"), dict) else {}
    for source in (cost_info.get("totalFee"), payload.get("priceWithoutFreight")):
        val = _num(source)
        if val is not None and val > 0:
            return val
    # Fall back to the pre-discount price if that's all that's present.
    origin = payload.get("originPcbCostInfo") if isinstance(payload.get("originPcbCostInfo"), dict) else {}
    return _num(origin.get("totalFee"))


def quote_board(pcb: dict[str, Any], qty: int) -> dict[str, Any] | None:
    """Live JLCPCB quote for one quantity. Returns per-board USD or ``None``.

    Returns ``{"usd": <per-board USD>, "qty": <qty quoted>, "raw": data}``.
    Never raises — any failure returns None so the pipeline uses the heuristic.
    """
    if not is_configured():
        return None

    sent_qty = max(int(qty), _MIN_QTY)
    payload = build_request_payload(pcb, sent_qty)
    body = json.dumps(payload, ensure_ascii=False)  # sign + send the SAME bytes
    headers = _sign("POST", JLCPCB_CALCULATE_PATH, body)
    url = f"{JLCPCB_API_BASE.rstrip('/')}{JLCPCB_CALCULATE_PATH}"

    try:
        resp = httpx.post(url, headers=headers, content=body.encode("utf-8"), timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - defensive
        logger.warning("JLCPCB quote failed for qty %s: %s", qty, exc)
        record_failure(
            "fab_quote", "JLCPCB quote",
            "JLCPCB API request failed — using the internal fab cost model",
            error=exc, context={"qty": qty, "sent_qty": sent_qty},
        )
        return None

    if not (isinstance(data, dict) and data.get("success") and data.get("code") == 200):
        logger.warning("JLCPCB returned an unsuccessful response for qty %s: %s",
                       qty, data.get("message") if isinstance(data, dict) else data)
        record_failure(
            "fab_quote", "JLCPCB quote",
            "JLCPCB returned an unsuccessful response — using the internal fab cost model",
            context={"qty": qty, "message": data.get("message") if isinstance(data, dict) else None},
        )
        return None

    total_usd = _extract_total_usd(data)
    if total_usd is None or total_usd <= 0:
        logger.warning("JLCPCB returned no recognizable price for qty %s", qty)
        record_failure(
            "fab_quote", "JLCPCB quote",
            "JLCPCB returned no recognizable price — using the internal fab cost model",
            context={"qty": qty},
        )
        return None

    # totalFee is the cost for the whole order (sent_qty boards) — divide to a
    # per-board figure so every volume-curve point is a true per-unit fab cost.
    per_board_usd = total_usd / sent_qty
    return {"usd": round(per_board_usd, 4), "qty": sent_qty, "total_usd": round(total_usd, 4), "raw": data}


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
