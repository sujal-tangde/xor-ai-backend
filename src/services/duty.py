"""Customs-duty pass: HSN → BCD / SWS / IGST → landed INR cost.

Rates are hardcoded here (no DB table), keyed by the 4-digit HSN prefix. The HSN
comes from the Mouser ``ProductCompliance`` CNHTS field; when it's missing we use
``DEFAULT_RATES`` (and only ask the LLM to classify when CNHTS is entirely
absent — that classification lives in the pipeline, not here).

Landed-cost math (SWS is 10% *of the BCD amount*; IGST is on the duty-inclusive
value):

    bcd_amount  = price * BCD
    sws_amount  = bcd_amount * SWS
    igst_amount = (price + bcd_amount + sws_amount) * IGST
    landed      = price + bcd_amount + sws_amount + igst_amount
"""

from __future__ import annotations

import re
from typing import Any

# Keyed by 4-digit HSN prefix.
tariff_rates: dict[str, dict[str, float]] = {
    "8542": {"BCD": 0.00, "SWS": 0.10, "IGST": 0.18},  # ICs
    "8504": {"BCD": 0.10, "SWS": 0.10, "IGST": 0.18},  # power supplies / inductors
    "8532": {"BCD": 0.10, "SWS": 0.10, "IGST": 0.18},  # capacitors
    "8533": {"BCD": 0.10, "SWS": 0.10, "IGST": 0.18},  # resistors
    "8541": {"BCD": 0.10, "SWS": 0.10, "IGST": 0.18},  # diodes / LEDs / transistors
    "8536": {"BCD": 0.10, "SWS": 0.10, "IGST": 0.18},  # connectors / switches
}

DEFAULT_RATES: dict[str, float] = {"BCD": 0.10, "SWS": 0.10, "IGST": 0.18}


def hsn_prefix(hsn: str | None) -> str | None:
    """First 4 digits of an HSN/HTS code, or None if not derivable."""
    if not hsn:
        return None
    digits = re.sub(r"\D", "", str(hsn))
    return digits[:4] if len(digits) >= 4 else None


def rates_for_hsn(hsn: str | None) -> tuple[dict[str, float], bool]:
    """Return ``(rates, matched)``. ``matched`` is False when DEFAULT_RATES used."""
    prefix = hsn_prefix(hsn)
    if prefix and prefix in tariff_rates:
        return tariff_rates[prefix], True
    return DEFAULT_RATES, False


def rate_label(rates: dict[str, float]) -> str:
    """BOM ``BCD/IGST`` cell label, e.g. ``'0% / 18%'``."""
    bcd = rates.get("BCD", 0.0) * 100
    igst = rates.get("IGST", 0.0) * 100
    return f"{bcd:.0f}% / {igst:.0f}%"


def landed_cost(price: float, rates: dict[str, float]) -> dict[str, Any]:
    """Compute the landed unit cost and its duty components from a base price."""
    price = float(price or 0.0)
    bcd_amount = price * rates.get("BCD", 0.0)
    sws_amount = bcd_amount * rates.get("SWS", 0.0)
    igst_amount = (price + bcd_amount + sws_amount) * rates.get("IGST", 0.0)
    landed = price + bcd_amount + sws_amount + igst_amount
    return {
        "base": round(price, 4),
        "bcd_amount": round(bcd_amount, 4),
        "sws_amount": round(sws_amount, 4),
        "igst_amount": round(igst_amount, 4),
        "landed": round(landed, 4),
        "rates": rates,
        "rate_label": rate_label(rates),
    }
