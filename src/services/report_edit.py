"""Report edit / regenerate flow (Step 12).

Operates on the SAVED structured JSON (``report_json``), never a fresh teardown
run. An edit is classified into:
  - prose-only      → rewrite one narrative field, no recompute.
  - data (no API)   → patch ``_compute`` (qty, volume, rate, remove line, user
                      price) then recompute the deterministic numbers downstream.
  - sourcing        → one targeted Mouser call (re-price / add a line) then
                      recompute.

Edits never silently change unrelated numbers — only the requested fields and
their direct roll-ups move. A user-supplied price is re-tagged ``Est`` /
user-provided so provenance stays honest.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.services import mouser, report_pipeline
from src.services.llm_analysis import invoke_llm, parse_json_object

logger = logging.getLogger(__name__)

_PROSE_FIELDS = {
    "executive_summary": ("product", "summary_prose"),
    "architecture_analysis": ("architecture", "prose"),
    "architecture": ("architecture", "prose"),
    "cost_driver_insight": ("architecture", "insight"),
    "insight": ("architecture", "insight"),
    "market_context": ("marketContext", "prose"),
    "market": ("marketContext", "prose"),
    "data_confidence": ("dataConfidence", "prose"),
}

_CLASSIFY_PROMPT = """You are editing an existing electronics should-cost report. The user asked for a change. \
Classify it into concrete operations against the report state below. Do NOT compute new numbers — only describe \
WHAT to change; the system recomputes costs deterministically.

Operation types you may emit (in an "operations" array):
- {{"op": "set_volume", "volume": 5000}}                      target production volume
- {{"op": "set_qty", "match": "<mpn/designator/text>", "qty": 3}}
- {{"op": "remove_line", "match": "<mpn/designator/text>"}}
- {{"op": "set_unit_price", "match": "<...>", "price_inr": 12.5}}   user-supplied price (will be tagged Est)
- {{"op": "set_rate", "target": "per_joint"|"setup"|"stencil", "value": 0.2}}   assembly rate card
- {{"op": "rewrite_prose", "field": "executive_summary"|"architecture_analysis"|"market_context"|"data_confidence"|"cost_driver_insight", "instruction": "<how>"}}
- {{"op": "reprice_line", "match": "<...>", "mpn": "<MPN to price on Mouser>"}}   needs a live lookup
- {{"op": "add_line", "mpn": "<MPN or null>", "qty": 1, "description": "<...>", "category": "ic"}}   needs lookup

"match" is matched case-insensitively against a line's MPN, designator or description.
Set "needs_sourcing": true if ANY operation is reprice_line or add_line (a live API call is required).

Return ONLY JSON: {{"operations": [...], "needs_sourcing": false, "summary": "<one line of what changed>"}}

REPORT STATE:
{state}

USER REQUEST:
{request}
"""


def _state_summary(report_json: dict[str, Any]) -> dict[str, Any]:
    compute = report_json.get("_compute") or {}
    lines = compute.get("lines") or []
    assembly = compute.get("assembly") or {}
    return {
        "volume": compute.get("volume"),
        "assembly_rates": {
            "per_joint": assembly.get("rate_per_joint_inr"),
            "setup": assembly.get("setup_fee_inr"),
            "stencil": assembly.get("stencil_fee_inr"),
        },
        "lines": [
            {"mpn": ln.get("mpn"), "designator": ln.get("designator"),
             "description": ln.get("description"), "qty": ln.get("qty")}
            for ln in lines
        ][:60],
        "prose_fields": list(_PROSE_FIELDS.keys()),
    }


def classify_edit(request: str, report_json: dict[str, Any]) -> dict[str, Any]:
    """Classify a free-text edit request into operations. Falls back to prose."""
    try:
        raw = invoke_llm(
            [{"role": "user", "content": _CLASSIFY_PROMPT.format(
                state=json.dumps(_state_summary(report_json), ensure_ascii=False)[:6000],
                request=request[:1500],
            )}],
            max_tokens=1200,
        )
        data = parse_json_object(raw)
    except Exception:
        logger.warning("Edit classification failed; treating as prose edit", exc_info=True)
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("operations"), list):
        return {
            "operations": [
                {"op": "rewrite_prose", "field": "executive_summary", "instruction": request}
            ],
            "needs_sourcing": False,
            "summary": "Updated the report narrative.",
        }
    return data


# --------------------------------------------------------------------------- #
# Line matching + appliers
# --------------------------------------------------------------------------- #
def _match_line(lines: list[dict[str, Any]], match: str) -> dict[str, Any] | None:
    needle = (match or "").strip().lower()
    if not needle:
        return None
    for ln in lines:
        for key in ("mpn", "designator", "ref_des", "description"):
            val = str(ln.get(key) or "").lower()
            if val and (needle in val or val in needle):
                return ln
    return None


def _clear_prices(line: dict[str, Any]) -> None:
    line["price_by_volume"] = {}
    line["landed_by_volume"] = {}


def _rewrite_prose(report_json: dict[str, Any], field: str, instruction: str) -> bool:
    path = _PROSE_FIELDS.get((field or "").strip().lower())
    if not path:
        return False
    section, key = path
    current = (report_json.get(section) or {}).get(key) or ""
    prompt = (
        "Rewrite this should-cost report passage per the instruction. Keep it factual and in the same "
        "professional tone. Do NOT introduce new numbers. Output ONLY the rewritten passage, no preamble.\n\n"
        f"INSTRUCTION: {instruction}\n\nCURRENT PASSAGE:\n{current}"
    )
    try:
        new_text = invoke_llm([{"role": "user", "content": prompt}], max_tokens=1200).strip()
    except Exception:
        logger.warning("Prose rewrite failed for %s", field, exc_info=True)
        return False
    if new_text:
        report_json.setdefault(section, {})[key] = new_text
        return True
    return False


def apply_edit(report_json: dict[str, Any], request: str, emit=None) -> tuple[dict[str, Any], str]:
    """Apply an edit to a saved report_json. Returns ``(report_json, summary)``."""
    emit = emit or (lambda *a, **k: None)
    classification = classify_edit(request, report_json)
    operations = classification.get("operations") or []
    compute = report_json.setdefault("_compute", {})
    lines = compute.setdefault("lines", [])
    assembly = compute.setdefault("assembly", {})

    recompute_needed = False
    changes: list[str] = []

    for op in operations:
        if not isinstance(op, dict):
            continue
        kind = op.get("op")

        if kind == "rewrite_prose":
            if _rewrite_prose(report_json, op.get("field", ""), op.get("instruction", request)):
                changes.append(f"rewrote {op.get('field')}")

        elif kind == "set_volume":
            try:
                compute["volume"] = max(1, int(op["volume"]))
                recompute_needed = True
                changes.append(f"target volume → {compute['volume']:,}")
            except (KeyError, TypeError, ValueError):
                pass

        elif kind == "set_qty":
            ln = _match_line(lines, op.get("match", ""))
            if ln is not None:
                try:
                    ln["qty"] = max(1, int(op["qty"]))
                    recompute_needed = True
                    changes.append(f"qty of {ln.get('mpn') or ln.get('designator')} → {ln['qty']}")
                except (KeyError, TypeError, ValueError):
                    pass

        elif kind == "remove_line":
            ln = _match_line(lines, op.get("match", ""))
            if ln is not None:
                lines.remove(ln)
                recompute_needed = True
                changes.append(f"removed {ln.get('mpn') or ln.get('designator') or ln.get('description')}")

        elif kind == "set_unit_price":
            ln = _match_line(lines, op.get("match", ""))
            if ln is not None:
                try:
                    ln["user_price"] = float(op["price_inr"])
                    ln["tag"] = "Est"
                    ln["source"] = "User-provided"
                    ln["confidence"] = "user"
                    ln["note"] = "user-supplied price"
                    _clear_prices(ln)
                    recompute_needed = True
                    changes.append(f"price of {ln.get('mpn') or ln.get('designator')} → ₹{ln['user_price']}")
                except (KeyError, TypeError, ValueError):
                    pass

        elif kind == "set_rate":
            target = op.get("target")
            try:
                value = float(op["value"])
            except (KeyError, TypeError, ValueError):
                continue
            field = {"per_joint": "rate_per_joint_inr", "setup": "setup_fee_inr",
                     "stencil": "stencil_fee_inr"}.get(target)
            if field:
                assembly[field] = value
                joints = int(assembly.get("total_joints") or 0)
                assembly["per_unit_inr"] = round(joints * assembly.get("rate_per_joint_inr", 0.0), 4)
                assembly["nre_inr"] = round(
                    assembly.get("setup_fee_inr", 0.0) + assembly.get("stencil_fee_inr", 0.0), 2
                )
                recompute_needed = True
                changes.append(f"assembly {target} rate → {value}")

        elif kind == "reprice_line":
            emit("rendering", "in_progress", "Re-fetching a live price…")
            ln = _match_line(lines, op.get("match", ""))
            mpn = (op.get("mpn") or (ln.get("mpn") if ln else None) or "").strip()
            if ln is not None and mpn:
                part = mouser.search_part(mpn)
                if part and part.get("price_breaks"):
                    ln["mpn"] = part.get("mpn") or mpn
                    ln["make"] = part.get("manufacturer") or ln.get("make")
                    ln["price_breaks"] = part["price_breaks"]
                    ln["hsn"] = part.get("hsn") or ln.get("hsn")
                    ln["tag"] = "Live"
                    ln["source"] = "Mouser"
                    ln["confidence"] = "high"
                    ln["user_price"] = None
                    ln["note"] = ""
                    _clear_prices(ln)
                    recompute_needed = True
                    changes.append(f"re-priced {ln['mpn']} live")
                else:
                    changes.append(f"could not find a live price for {mpn}")

        elif kind == "add_line":
            mpn = (op.get("mpn") or "").strip() or None
            category = (op.get("category") or "other").lower()
            try:
                qty = max(1, int(op.get("qty") or 1))
            except (TypeError, ValueError):
                qty = 1
            new_line = {
                "ref_des": op.get("description") or mpn or "new",
                "designator": "—",
                "category": category,
                "qty": qty,
                "pkg": "—",
                "description": op.get("description") or (mpn or "Added component"),
                "mpn": mpn,
                "make": None,
                "hsn": None,
                "datasheet_url": None,
                "price_by_volume": {},
                "price_breaks": None,
                "source": "Rate-card",
                "tag": "Est",
                "confidence": "low",
                "note": "added during edit",
                "pins": report_pipeline.pins_for_package("—", category),
            }
            if mpn:
                emit("rendering", "in_progress", "Pricing the new component…")
                part = mouser.search_part(mpn)
                if part and part.get("price_breaks"):
                    new_line.update({
                        "mpn": part.get("mpn") or mpn,
                        "make": part.get("manufacturer"),
                        "description": part.get("description") or new_line["description"],
                        "price_breaks": part["price_breaks"],
                        "hsn": part.get("hsn"),
                        "tag": "Live",
                        "source": "Mouser",
                        "confidence": "high",
                        "note": "",
                    })
            # Duty rates for the new line.
            from src.services import duty as duty_mod
            hsn = new_line["hsn"] or report_pipeline._CATEGORY_HSN.get(category)
            new_line["hsn"] = hsn
            rates, _ = duty_mod.rates_for_hsn(hsn)
            new_line["rates"] = rates
            new_line["bcd_igst"] = duty_mod.rate_label(rates)
            lines.append(new_line)
            recompute_needed = True
            changes.append(f"added {new_line.get('mpn') or new_line.get('description')}")

    if recompute_needed:
        report_pipeline.recompute_numeric(report_json)
        # Keep top-level volume/meta in sync for the panel header.
        report_json.setdefault("meta", {})["volume"] = compute.get("volume")

    summary = classification.get("summary") or (", ".join(changes) if changes else "Updated the report.")
    return report_json, summary
