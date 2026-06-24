"""Should-cost report pipeline — the deterministic compute half.

Runs the stages after KB read + HILT (MPN resolve → parallel price/fab/estimate/
market → FX → duty → assembly → volume curve → aggregate) and returns one
structured JSON document (see :func:`aggregate`). The agent tool drives this and
owns progress emission + rendering/storage; this module is pure compute plus the
LLM-estimator calls (which themselves degrade gracefully).

Governing rule: **the LLM narrates, the code computes.** Every number here is
computed in Python and carries a source tag (``Live``/``Est``) and confidence;
anything the APIs can't price is flagged in ``dataQuality`` and tagged ``Est`` —
never hallucinated. No stage can crash the run: every external call has a
fallback and every gap is recorded as a plain-language note.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from src.core.config import (
    ASSEMBLY_RATE_PER_JOINT_INR,
    ASSEMBLY_SETUP_FEE_INR,
    ASSEMBLY_STENCIL_FEE_INR,
    REPORT_VOLUME_CURVE,
)
from src.services import duty, fx, jlcpcb, parts_pricing, report_estimators
from src.services.failure_log import record_failure

logger = logging.getLogger(__name__)

# emit(stage, status, message, meta=None)
Emit = Callable[..., None]


def _noop(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# Rate card (INR) for generic passives and any line we can't price live.
# --------------------------------------------------------------------------- #
_RATE_CARD: dict[str, float] = {
    "resistor": 0.5, "capacitor": 1.2, "inductor": 9.0, "ferrite": 3.0,
    "diode": 2.5, "led": 4.0, "transistor": 3.5, "ic": 90.0, "mcu": 180.0,
    "connector": 14.0, "crystal": 7.0, "oscillator": 25.0, "switch": 6.0,
    "sensor": 60.0, "module": 220.0, "electromechanical": 30.0, "other": 10.0,
}

# Volume discount applied to rate-card base prices (and used as a default HSN
# heading per category for duty when nothing better is known).
_VOLUME_DISCOUNT = {1: 1.0, 100: 0.82, 1000: 0.65, 10000: 0.55}

_CATEGORY_HSN = {
    "resistor": "8533", "capacitor": "8532", "inductor": "8504", "ferrite": "8504",
    "diode": "8541", "led": "8541", "transistor": "8541", "ic": "8542", "mcu": "8542",
    "sensor": "8542", "module": "8542", "connector": "8536", "switch": "8536",
    "crystal": "8541", "oscillator": "8541", "electromechanical": "8536", "other": "8542",
}

_GENERIC_CATEGORIES = {"resistor", "capacitor", "inductor", "ferrite"}


def _discount_for(volume: int) -> float:
    keys = sorted(_VOLUME_DISCOUNT)
    chosen = _VOLUME_DISCOUNT[keys[0]]
    for k in keys:
        if k <= volume:
            chosen = _VOLUME_DISCOUNT[k]
    return chosen


def _categorize(component: dict[str, Any], resolution: dict[str, Any] | None) -> str:
    if resolution and resolution.get("category"):
        cat = str(resolution["category"]).lower().strip()
        if cat in _RATE_CARD:
            return cat
    text = " ".join(
        str(component.get(k) or "") for k in ("type", "function", "value")
    ).lower()
    for key in ("mcu", "microcontroller"):
        if key in text:
            return "mcu"
    for cat in ("resistor", "capacitor", "inductor", "ferrite", "diode", "led",
                "transistor", "connector", "crystal", "oscillator", "switch",
                "sensor", "module"):
        if cat in text:
            return cat
    if re.search(r"\bic\b|amplifier|regulator|driver|soc|chip", text):
        return "ic"
    return "other"


def _rate_card_price(category: str, volume: int) -> float:
    base = _RATE_CARD.get(category, _RATE_CARD["other"])
    return round(base * _discount_for(volume), 4)


# --------------------------------------------------------------------------- #
# Assembly: estimate pins per package for the joint count.
# --------------------------------------------------------------------------- #
def pins_for_package(package: str | None, category: str | None = None) -> int:
    """Estimate solder joints (pins) for a package string."""
    pkg = str(package or "").upper().strip()
    if not pkg:
        # Fall back to a category default.
        return {"resistor": 2, "capacitor": 2, "inductor": 2, "ferrite": 2,
                "diode": 2, "led": 2, "transistor": 3, "crystal": 2,
                "connector": 4, "ic": 8, "mcu": 32}.get(category or "", 2)
    # Two-terminal chip packages.
    if re.fullmatch(r"0\d{3}", pkg) or pkg in {"1206", "1210", "2010", "2512", "0201", "0402", "0603", "0805"}:
        return 2
    if pkg in {"SOD-123", "SOD-323", "SMA", "SMB", "SMC", "MELF"}:
        return 2
    if pkg.startswith("SOT"):
        m = re.search(r"SOT-?(\d+)", pkg)
        if m:
            try:
                return max(3, int(m.group(1)) if int(m.group(1)) <= 12 else 3)
            except ValueError:
                pass
        return 3
    # Packages carrying an explicit pin count: QFN-48, TQFP-64, SOIC-8, ESOP-8…
    m = re.search(r"-(\d{1,3})\b", pkg) or re.search(r"(\d{1,3})$", pkg)
    if m:
        try:
            pins = int(m.group(1))
            if 2 <= pins <= 400:
                return pins
        except ValueError:
            pass
    if "QFN" in pkg or "QFP" in pkg or "BGA" in pkg:
        return 32
    if "SOIC" in pkg or "SOP" in pkg or "MSOP" in pkg:
        return 8
    return 2


# --------------------------------------------------------------------------- #
# Stage C — price each BOM line (resolved-MPN → parts DB; else rate card)
# --------------------------------------------------------------------------- #
def _qty(component: dict[str, Any]) -> int:
    try:
        return max(1, int(component.get("qty_per_unit") or 1))
    except (TypeError, ValueError):
        return 1


def _resolve_line_mpn(
    component: dict[str, Any],
    resolution: dict[str, Any] | None,
    category: str,
) -> tuple[str | None, bool]:
    """Pick the MPN to price for a line and whether it's a generic passive."""
    resolved_mpn = None
    is_generic = False
    if resolution:
        is_generic = bool(resolution.get("is_generic_passive"))
        resolved_mpn = (resolution.get("resolved_mpn") or "").strip() or None
    if not resolved_mpn:
        resolved_mpn = (component.get("mpn") or "").strip() or None
    if category in _GENERIC_CATEGORIES and not (component.get("mpn") or "").strip():
        is_generic = True
    return resolved_mpn, is_generic


def _price_one_line(
    index: int,
    component: dict[str, Any],
    resolution: dict[str, Any] | None,
    volumes: list[int],
    parts_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve + price one component across all volumes. Never raises.

    Pricing comes from ``parts_map`` (a pre-fetched parts-DB lookup keyed by
    ``lower(mpn)``, prices already in INR); lines with no match fall back to the
    rate card.
    """
    ref_des = component.get("ref_des") or f"#{index + 1}"
    category = _categorize(component, resolution)
    qty = _qty(component)
    description = (
        component.get("function")
        or component.get("type")
        or category.title()
    )
    pkg = component.get("package") or "—"

    resolved_mpn, is_generic = _resolve_line_mpn(component, resolution, category)

    line: dict[str, Any] = {
        "ref_des": ref_des,
        "designator": ref_des,
        "category": category,
        "qty": qty,
        "pkg": pkg,
        "description": description,
        "mpn": resolved_mpn,
        "make": component.get("manufacturer"),
        "hsn": None,
        "datasheet_url": None,
        "price_by_volume": {},
        "price_breaks": None,
        "source": "Rate-card",
        "tag": "Est",
        "confidence": "low",
        "note": "",
        "pins": pins_for_package(pkg, category),
    }

    part = None
    if not is_generic and resolved_mpn:
        part = parts_map.get(resolved_mpn.lower())

    if part and part.get("price_breaks"):
        # Keep the resolved MPN's original casing — the parts DB stores it lowercased.
        line["mpn"] = resolved_mpn or part.get("mpn")
        line["make"] = part.get("manufacturer") or line["make"]
        if part.get("description"):
            line["description"] = part["description"]
        line["hsn"] = part.get("hsn")
        line["datasheet_url"] = part.get("datasheet_url")
        line["source"] = "JLCPCB"
        line["tag"] = "Live"
        line["confidence"] = "high"
        line["price_breaks"] = part["price_breaks"]
        for v in volumes:
            p = parts_pricing.price_for_qty(part["price_breaks"], v)
            if p is not None:
                line["price_by_volume"][v] = round(p, 4)
        if not line["price_by_volume"]:
            # Found the part but no usable break — fall back to rate card.
            line["source"] = "Rate-card"
            line["tag"] = "Est"
            line["confidence"] = "low"
            line["price_breaks"] = None
            line["note"] = "live price break unavailable"
    else:
        if is_generic:
            line["note"] = "generic passive — rate-card price"
        elif resolved_mpn:
            line["note"] = "live price not found — rate-card estimate"
        else:
            line["note"] = "part not resolved — rate-card estimate"

    if not line["price_by_volume"]:
        for v in volumes:
            line["price_by_volume"][v] = _rate_card_price(category, v)

    return line


def _collect_mpns(
    components: list[dict[str, Any]],
    resolutions: dict[str, dict[str, Any]],
) -> list[str]:
    """Resolved, non-generic MPNs to batch-price against the parts DB."""
    mpns: list[str] = []
    for idx, component in enumerate(components):
        ref_des = component.get("ref_des") or f"#{idx + 1}"
        resolution = resolutions.get(str(ref_des))
        category = _categorize(component, resolution)
        resolved_mpn, is_generic = _resolve_line_mpn(component, resolution, category)
        if resolved_mpn and not is_generic:
            mpns.append(resolved_mpn)
    return mpns


def _run_pricing(
    components: list[dict[str, Any]],
    resolutions: dict[str, dict[str, Any]],
    volumes: list[int],
    fx_rate: float,
    emit: Emit,
) -> list[dict[str, Any]]:
    """Price all BOM lines, emitting per-item progress (runs on the caller thread).

    All resolved MPNs are priced in one batched parts-DB lookup (USD→INR via
    ``fx_rate``) before lines are built, so the DB is hit once for the whole BOM.
    """
    total = len(components)
    if total == 0:
        return []

    parts_map = parts_pricing.price_mpns(_collect_mpns(components, resolutions), fx_rate)

    indexed: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(8, total)) as ex:
        futures = {}
        for idx, component in enumerate(components):
            ref_des = component.get("ref_des") or f"#{idx + 1}"
            resolution = resolutions.get(str(ref_des))
            futures[ex.submit(_price_one_line, idx, component, resolution, volumes, parts_map)] = idx
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                line = fut.result()
            except Exception:  # noqa: BLE001 - never let one line crash pricing
                logger.warning("Pricing a BOM line crashed; skipping", exc_info=True)
                line = None
            done += 1
            if line is not None:
                indexed.append((idx, line))
                label = line.get("mpn") or line.get("description") or line.get("ref_des")
                src = "the parts database" if line["source"] == "JLCPCB" else "an estimate"
                emit(
                    "pricing", "in_progress",
                    f"Pricing component {done} of {total} ({label}) via {src}…",
                    {"current": done, "total": total},
                )
    # Preserve original component order, then number the lines.
    indexed.sort(key=lambda pair: pair[0])
    lines = [line for _, line in indexed]
    for i, line in enumerate(lines, start=1):
        line["sno"] = i
    return lines


# --------------------------------------------------------------------------- #
# Stage E — PCB fab quote (JLCPCB live USD → INR, else heuristic INR)
# --------------------------------------------------------------------------- #
def _run_fab(pcb: dict[str, Any], volumes: list[int], fx_info: dict[str, Any]) -> dict[str, Any]:
    """Quote the bare board at each volume. Returns INR per board + provenance.

    Uses the already-fetched ``fx_info`` (shared with component pricing) so the
    whole run converts USD→INR at a single, consistent rate.
    """
    fab_by_volume: dict[int, float] = {}
    live = False
    rate = float(fx_info["rate"])
    for v in volumes:
        q = jlcpcb.quote_board(pcb, v)
        if q is not None:
            fab_by_volume[v] = round(q["usd"] * rate, 2)
            live = True
        else:
            fab_by_volume[v] = jlcpcb.heuristic_quote_inr(pcb, v)
    return {
        "fab_by_volume": fab_by_volume,
        "live": live,
        "fx": fx_info,
        "source": "JLCPCB" if live else "Internal fab model",
        "tag": "Live" if live else "Est",
        "params": jlcpcb.template_params(pcb),
    }


# --------------------------------------------------------------------------- #
# Stage G — market context (Tavily web → LLM summary)
# --------------------------------------------------------------------------- #
def _tavily_snippets(query: str) -> str:
    try:
        from src.agent.tools.tavily_search_tool import get_tavily_search_tool

        tool = get_tavily_search_tool()
        if tool is None:
            return ""
        result = tool.invoke({"query": query})
        if isinstance(result, dict):
            parts: list[str] = []
            if result.get("answer"):
                parts.append(str(result["answer"]))
            for item in result.get("results") or []:
                if isinstance(item, dict):
                    for key in ("content", "snippet", "raw_content"):
                        if item.get(key):
                            parts.append(str(item[key]))
            return "\n".join(parts)[:8000]
        return str(result)[:8000]
    except Exception as exc:
        logger.warning("Tavily market lookup failed", exc_info=True)
        record_failure(
            "market_context", "Tavily web search",
            "Web search for market context failed — proceeding without web comparables",
            error=exc, context={"query": query},
        )
        return ""


def _run_market(product_label: str, structured: dict[str, Any]) -> dict[str, Any]:
    query = (
        f"{product_label} retail price India and key component prices, "
        "supply and sourcing risk"
    )
    snippets = _tavily_snippets(query)
    summary = report_estimators.summarize_market(product_label, structured, snippets)
    summary["source"] = "Tavily (web)"
    summary["tag"] = "Est"
    return summary


# --------------------------------------------------------------------------- #
# Duty pass + assembly + volume curve (pure compute)
# --------------------------------------------------------------------------- #
def _apply_duty(lines: list[dict[str, Any]], volumes: list[int]) -> None:
    """Attach duty rates + landed prices per volume to each line."""
    # Classify HSN only for lines the parts DB gave none for.
    missing = [ln for ln in lines if not duty.hsn_prefix(ln.get("hsn"))]
    classified: dict[str, str] = {}
    if missing:
        to_classify = [
            {"ref_des": ln["ref_des"], "type": ln.get("description"),
             "function": ln.get("description"), "package": ln.get("pkg")}
            for ln in missing
        ]
        classified = report_estimators.classify_hsns(to_classify)

    for ln in lines:
        hsn = ln.get("hsn")
        if not duty.hsn_prefix(hsn):
            hsn = classified.get(ln["ref_des"]) or _CATEGORY_HSN.get(ln["category"])
            ln["hsn"] = hsn
        rates, matched = duty.rates_for_hsn(hsn)
        ln["rates"] = rates
        ln["bcd_igst"] = duty.rate_label(rates)
        ln["landed_by_volume"] = {}
        for v in volumes:
            base = ln["price_by_volume"].get(v, 0.0)
            ln["landed_by_volume"][v] = round(duty.landed_cost(base, rates)["landed"], 4)


def _assembly_model(lines: list[dict[str, Any]]) -> dict[str, Any]:
    total_joints = sum(int(ln["qty"]) * int(ln.get("pins") or 2) for ln in lines)
    nre = ASSEMBLY_SETUP_FEE_INR + ASSEMBLY_STENCIL_FEE_INR
    per_unit = round(total_joints * ASSEMBLY_RATE_PER_JOINT_INR, 4)
    return {
        "total_joints": total_joints,
        "setup_fee_inr": ASSEMBLY_SETUP_FEE_INR,
        "stencil_fee_inr": ASSEMBLY_STENCIL_FEE_INR,
        "rate_per_joint_inr": ASSEMBLY_RATE_PER_JOINT_INR,
        "nre_inr": round(nre, 2),
        "per_unit_inr": per_unit,
        "tag": "Est",
        "source": "EMS rate card",
    }


def _bom_landed_per_unit(lines: list[dict[str, Any]], volume: int) -> float:
    return round(
        sum(ln["landed_by_volume"].get(volume, 0.0) * int(ln["qty"]) for ln in lines), 2
    )


def _fab_at(fab: dict[str, Any], volume: int) -> float:
    """Fab cost at ``volume``; nearest priced volume if it wasn't quoted directly
    (so an edit to a new target volume needs no fresh JLCPCB call)."""
    by_vol = fab.get("fab_by_volume") or {}
    if volume in by_vol:
        return by_vol[volume]
    if not by_vol:
        return 0.0
    nearest = min(by_vol, key=lambda v: abs(int(v) - volume))
    return by_vol[nearest]


def _stage_amounts_at(
    lines: list[dict[str, Any]],
    fab: dict[str, Any],
    assembly: dict[str, Any],
    non_quotable: dict[str, Any],
    volume: int,
) -> list[tuple[str, float]]:
    """The six per-unit RECURRING manufacturing-stage costs for a SINGLE unit.

    One-time NRE (tooling, firmware dev, line setup) is reported separately and is
    NOT amortized into the unit cost — so this figure is the true cost of one
    unit's inputs and is volume-independent.
    """
    fw, enc, fa = non_quotable["firmware"], non_quotable["enclosure"], non_quotable["final_assembly"]
    v = max(1, int(volume))
    return [
        ("Component BOM (landed)", _bom_landed_per_unit(lines, v)),
        ("PCB fabrication", round(_fab_at(fab, v), 2)),
        ("PCB assembly", round(assembly["per_unit_inr"], 2)),
        ("Firmware / programming", round(fw["per_unit_inr"], 2)),
        ("Enclosure & mechanical", round(enc["per_unit_inr"], 2)),
        ("Final assembly, test, pack", round(fa["per_unit_inr"], 2)),
    ]


def _build_volume_curve(
    lines: list[dict[str, Any]],
    fab: dict[str, Any],
    assembly: dict[str, Any],
    non_quotable: dict[str, Any],
    curve_volumes: list[int],
) -> dict[str, Any]:
    per_volume = {v: _stage_amounts_at(lines, fab, assembly, non_quotable, v) for v in curve_volumes}
    stage_names = [name for name, _ in next(iter(per_volume.values()))] if per_volume else []
    rows = [
        {"stage": name, "values": [per_volume[v][i][1] for v in curve_volumes]}
        for i, name in enumerate(stage_names)
    ]
    totals = [round(sum(amt for _, amt in per_volume[v]), 2) for v in curve_volumes]
    return {"volumes": curve_volumes, "rows": rows, "totals": totals}


def _stage_breakdown(
    lines: list[dict[str, Any]],
    fab: dict[str, Any],
    assembly: dict[str, Any],
    non_quotable: dict[str, Any],
    volume: int,
) -> dict[str, Any]:
    """Per-stage cost at the EXACT selected volume, with bar proportions."""
    amounts = _stage_amounts_at(lines, fab, assembly, non_quotable, volume)
    total = sum(amt for _, amt in amounts) or 1.0
    rows = [
        {"stage": name, "amount": amt, "pct": round(100.0 * amt / total, 1) if total else 0.0}
        for name, amt in amounts
    ]
    return {"rows": rows, "total": round(sum(amt for _, amt in amounts), 2), "volume": int(volume)}


def _line_base_price(line: dict[str, Any], volume: int) -> float:
    """Base (pre-duty) unit price for a line at ``volume`` — no API calls."""
    if line.get("user_price") is not None:
        return float(line["user_price"])
    breaks = line.get("price_breaks")
    if breaks:
        p = parts_pricing.price_for_qty(breaks, volume)
        if p is not None:
            return round(p, 4)
    return _rate_card_price(line.get("category") or "other", volume)


def ensure_line_prices(line: dict[str, Any], volumes: list[int]) -> None:
    """Fill ``price_by_volume`` + ``landed_by_volume`` for any missing volume.

    Recompute-safe: derives prices from stored breaks / user price / rate card and
    applies the line's already-determined duty rates. Used on initial build and on
    every edit (e.g. a new target volume) without re-hitting any API.
    """
    rates = line.get("rates") or duty.DEFAULT_RATES
    pbv = line.setdefault("price_by_volume", {})
    lbv = line.setdefault("landed_by_volume", {})
    for v in volumes:
        if v not in pbv:
            pbv[v] = _line_base_price(line, v)
        if v not in lbv:
            lbv[v] = round(duty.landed_cost(pbv[v], rates)["landed"], 4)


def build_numeric_sections(
    lines: list[dict[str, Any]],
    fab: dict[str, Any],
    assembly: dict[str, Any],
    non_quotable: dict[str, Any],
    volume: int,
    curve_volumes: list[int],
) -> dict[str, Any]:
    """Build BOM rows, stage breakdown, volume curve and headline metrics.

    Single code path shared by the initial aggregate and the edit/recompute flow,
    so an edit never silently drifts from how the report was first computed.
    """
    needed = sorted(set(curve_volumes + [volume]))
    for ln in lines:
        ensure_line_prices(ln, needed)

    volume_curve = _build_volume_curve(lines, fab, assembly, non_quotable, curve_volumes)
    stages = _stage_breakdown(lines, fab, assembly, non_quotable, volume)

    bom_rows = []
    bom_subtotal = 0.0
    for i, ln in enumerate(lines, start=1):
        ln["sno"] = i
        unit = ln["landed_by_volume"].get(volume, 0.0)
        ext = round(unit * int(ln["qty"]), 2)
        bom_subtotal += ext
        bom_rows.append({
            "sno": i,
            "mpn": ln.get("mpn") or "—",
            "make": ln.get("make") or "—",
            "description": ln.get("description") or "—",
            "designator": ln.get("designator") or "—",
            "qty": ln["qty"],
            "pkg": ln.get("pkg") or "—",
            "unit_inr": round(unit, 2),
            "bcd_igst": ln.get("bcd_igst") or "—",
            "ext_inr": ext,
            "source": ln.get("source") or "—",
            "tag": ln.get("tag") or "Est",
            "confidence": ln.get("confidence") or "low",
            "note": ln.get("note") or "",
            "datasheet_url": ln.get("datasheet_url"),
        })

    fw, enc, fa = non_quotable["firmware"], non_quotable["enclosure"], non_quotable["final_assembly"]
    one_time_nre = round(
        assembly["nre_inr"] + fw["nre_inr"] + enc["nre_inr"] + fa["nre_inr"], 2
    )

    def total_at(v: int) -> float | None:
        return volume_curve["totals"][curve_volumes.index(v)] if v in curve_volumes else None

    live_count = sum(1 for ln in lines if ln.get("tag") == "Live")
    metrics = {
        "ex_works_at_1000": total_at(1000),
        "ex_works_at_10000": total_at(10000),
        "ex_works_selected": total_at(volume) or stages["total"],
        "one_time_nre_inr": one_time_nre,
        "selected_volume": volume,
    }
    return {
        "volume_curve": volume_curve,
        "stages": stages,
        "bom_rows": bom_rows,
        "bom_subtotal": round(bom_subtotal, 2),
        "metrics": metrics,
        "live_count": live_count,
        "fab_selected_inr": (fab.get("fab_by_volume") or {}).get(volume),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(
    structured: dict[str, Any],
    theory: str,
    product_label: str,
    volume: int,
    emit: Emit = _noop,
) -> dict[str, Any]:
    """Run stages 3–9 and return the aggregated structured JSON for the report."""
    structured = structured or {}
    components = [c for c in (structured.get("components") or []) if isinstance(c, dict)]
    pcb = structured.get("pcb") or {}
    data_quality: list[str] = []

    curve_volumes = list(REPORT_VOLUME_CURVE)
    price_volumes = sorted(set(curve_volumes + [volume]))

    # --- Step 3: resolve MPNs ------------------------------------------------ #
    emit("resolving_mpns", "started", "Identifying part numbers…")
    resolutions = report_estimators.resolve_mpns(components)
    if not resolutions and components:
        data_quality.append(
            "Some part numbers could not be confidently identified, so a few components "
            "use category estimates rather than a specific manufacturer part."
        )
    emit("resolving_mpns", "done", "Part numbers identified.")

    # --- Step 4: PARALLEL price (C) | fab (E) | non-quotable (F) | market (G) - #
    # Fetch the USD→INR rate once and share it across component pricing and the fab
    # quote, so the whole run converts at a single consistent rate.
    fx_info = fx.fetch_usd_inr()
    fx_rate = float(fx_info["rate"])

    emit("pricing", "started", "Pricing components from the parts database…")
    emit("fab_quote", "started", "Getting PCB fab quote from JLCPCB…")
    emit("non_quotable", "started", "Estimating firmware, tooling & labour…")
    emit("market_context", "started", "Gathering market & sourcing data from the web…")

    with ThreadPoolExecutor(max_workers=3) as ex:
        fab_future = ex.submit(_run_fab, pcb, price_volumes, fx_info)
        nq_future = ex.submit(report_estimators.estimate_non_quotable, structured)
        mkt_future = ex.submit(_run_market, product_label, structured)

        # C (pricing) runs on this thread so its per-item progress streams cleanly.
        lines = _run_pricing(components, resolutions, price_volumes, fx_rate, emit)
        live_count = sum(1 for ln in lines if ln["tag"] == "Live")
        est_count = len(lines) - live_count
        if not parts_pricing.is_configured():
            data_quality.append(
                "Live component pricing was unavailable for this run, so the bill of "
                "materials uses industry rate-card estimates — confirm before quoting."
            )
        elif est_count:
            data_quality.append(
                f"Live pricing wasn't available for {est_count} component"
                f"{'s' if est_count != 1 else ''}, so these use industry estimates — "
                "confirm before quoting."
            )
        emit("pricing", "done", f"Priced {len(lines)} components ({live_count} live).")

        try:
            fab = fab_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fab quote failed entirely; using heuristic", exc_info=True)
            record_failure(
                "fab_quote", "PCB fabrication",
                "Fab quoting stage failed entirely — using the internal fab cost model",
                error=exc, context={"volumes": price_volumes},
            )
            fab = {"fab_by_volume": {v: jlcpcb.heuristic_quote_inr(pcb, v) for v in price_volumes},
                   "live": False, "fx": fx_info, "source": "Internal fab model",
                   "tag": "Est", "params": jlcpcb.template_params(pcb)}
        if not fab["live"]:
            data_quality.append(
                "The PCB fabrication quote is based on our internal cost model rather "
                "than a live supplier quote for this run."
            )
            emit("fab_quote", "warning", "Live fab quote unavailable — using our internal model.")
        else:
            emit("fab_quote", "done", "PCB fab quote received.")

        try:
            non_quotable = nq_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Non-quotable estimate failed", exc_info=True)
            record_failure(
                "non_quotable", "Non-quotable estimate",
                "Non-quotable stage failed entirely — using default reference rates",
                error=exc,
            )
            non_quotable = {**report_estimators._NON_QUOTABLE_FALLBACK, "fallback": True}
        if non_quotable.get("fallback"):
            data_quality.append(
                "Firmware, tooling and labour costs use default industry reference rates "
                "for this run."
            )
        emit("non_quotable", "done", "Firmware, tooling & labour estimated.")

        try:
            market = mkt_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Market context failed", exc_info=True)
            record_failure(
                "market_context", product_label or "market",
                "Market context stage failed entirely — omitting web comparables",
                error=exc,
            )
            market = {"observations": [], "comparables": [], "margin_band": None,
                      "had_web": False, "source": "Tavily (web)", "tag": "Est"}
        if not market.get("had_web") or not market.get("comparables"):
            data_quality.append(
                "Retail market comparables couldn't be gathered this time, so the margin "
                "estimate is approximate."
            )
        emit("market_context", "done", "Market context gathered.")

    # --- Step 5: FX (shared rate already applied to pricing + fab; surface it) - #
    fx_info = fab.get("fx") or fx_info
    emit("fx", "started", "Applying USD→INR exchange rate to component & fab costs…")
    if fx_info.get("source") == fx.SOURCE_CACHED:
        data_quality.append("Exchange rate is from a recent cached value rather than today's live rate.")
    elif fx_info.get("source") == fx.SOURCE_FALLBACK:
        data_quality.append("A standard fallback exchange rate was used as today's live rate was unavailable.")
    emit("fx", "done", f"Exchange rate applied (1 USD ≈ ₹{fx_info.get('rate')}).")

    # --- Step 6: duty pass --------------------------------------------------- #
    emit("duty", "started", "Calculating customs duty & landed cost…")
    _apply_duty(lines, price_volumes)
    emit("duty", "done", "Customs duty & landed cost applied.")

    # --- Step 7: assembly model --------------------------------------------- #
    emit("assembly", "started", "Modelling PCB assembly cost…")
    assembly = _assembly_model(lines)
    emit("assembly", "done", "Assembly cost modelled.")

    # --- Step 8: volume curve ------------------------------------------------ #
    emit("volume_curve", "started", "Running the cost model across volumes…")
    numeric = build_numeric_sections(lines, fab, assembly, non_quotable, volume, curve_volumes)
    emit("volume_curve", "done", "Volume curve computed.")

    # --- Step 9: aggregate --------------------------------------------------- #
    return aggregate(
        structured=structured,
        theory=theory,
        product_label=product_label,
        volume=volume,
        lines=lines,
        fab=fab,
        assembly=assembly,
        non_quotable=non_quotable,
        market=market,
        fx_info=fx_info,
        numeric=numeric,
        curve_volumes=curve_volumes,
        data_quality=data_quality,
        emit=emit,
    )


def aggregate(
    *,
    structured: dict[str, Any],
    theory: str,
    product_label: str,
    volume: int,
    lines: list[dict[str, Any]],
    fab: dict[str, Any],
    assembly: dict[str, Any],
    non_quotable: dict[str, Any],
    market: dict[str, Any],
    fx_info: dict[str, Any],
    numeric: dict[str, Any],
    curve_volumes: list[int],
    data_quality: list[str],
    emit: Emit = _noop,
) -> dict[str, Any]:
    """Assemble the final structured JSON (the single source for template-fill)."""
    volume_curve = numeric["volume_curve"]
    stages = numeric["stages"]
    bom_rows = numeric["bom_rows"]
    bom_subtotal = numeric["bom_subtotal"]
    metrics = numeric["metrics"]
    one_time_nre = metrics["one_time_nre_inr"]

    def total_at(v: int) -> float | None:
        if v in curve_volumes:
            return volume_curve["totals"][curve_volumes.index(v)]
        return None

    product = structured.get("product") or {}
    enclosure = structured.get("enclosure") or {}

    # Subsystems from architecture blocks.
    subsystems = []
    for block in (structured.get("architecture_blocks") or []):
        if isinstance(block, dict) and block.get("block"):
            subsystems.append({
                "subsystem": block.get("block"),
                "basis": block.get("description") or "",
                "confidence": "Med",
            })

    methodology = [
        {"stage": "Component identification", "source": "Teardown photos + spec",
         "method": "Vision MPN read + parametric match", "confidence": "High"},
        {"stage": "Component pricing", "source": "JLCPCB parts database",
         "method": "Live qty-break lookup (USD→INR)",
         "confidence": "High" if any(l["tag"] == "Live" for l in lines) else "Low"},
        {"stage": "Landed / customs", "source": "HSN tariff table",
         "method": "Per-MPN HSN → BCD / SWS / IGST", "confidence": "Medium"},
        {"stage": "PCB fabrication", "source": fab.get("source", "JLCPCB"),
         "method": "Live quote → INR" if fab.get("live") else "Area × layers rate-card",
         "confidence": "High" if fab.get("live") else "Medium"},
        {"stage": "PCB assembly", "source": "EMS rate card",
         "method": "Per-joint model; one-time NRE separate", "confidence": "Medium"},
        {"stage": "Firmware / enclosure / labour", "source": "Indian EMS reference rates",
         "method": "Per-unit estimate; one-time NRE separate", "confidence": "Low-Med"},
        {"stage": "Market context", "source": "Tavily (web)",
         "method": "Web comparables (low confidence)", "confidence": "Low"},
    ]

    # Figures handed to the prose LLM (narration only — no recompute). This is a
    # SINGLE-UNIT report, so no volume-scaling figures are exposed to the writer.
    figures = {
        "currency": "INR",
        "basis": "single unit (recurring cost; one-time NRE reported separately)",
        "unit_cost_per_unit_recurring": stages["total"],
        "one_time_nre_inr_separate": one_time_nre,
        "bom_cost_per_unit_inr": round(bom_subtotal, 2),
        "stage_breakdown": stages["rows"],
        "live_lines": sum(1 for l in lines if l["tag"] == "Live"),
        "total_lines": len(lines),
        "margin_band": market.get("margin_band"),
    }

    emit("rendering", "started", "Writing the report narrative…")
    prose = report_estimators.write_prose(
        {"name": product_label, "product": product, "theory": (theory or "")[:3000]},
        figures,
        data_quality,
    )

    all_live = not data_quality
    if all_live:
        prose["data_confidence"] = (
            "All component prices and fab costs were sourced live for this report."
        )

    report_json = {
        "meta": {
            "title": f"Reverse Engineering Cost Report — {product_label}",
            "product_label": product_label,
            "currency": "INR",
            "volume": volume,
            "inputs": _inputs_label(structured),
        },
        "product": {
            "name": product.get("name") or product_label,
            "subtitle": product.get("primary_function") or product.get("type") or "",
            "summary_prose": prose["executive_summary"],
            "key_findings": prose["key_findings"],
            "overview": _overview_kv(product, enclosure, structured.get("pcb") or {}, lines),
            "subsystems": subsystems,
        },
        "architecture": {
            "prose": prose["architecture_analysis"],
            "blocks": [
                {"block": b.get("block"), "description": b.get("description") or ""}
                for b in (structured.get("architecture_blocks") or [])
                if isinstance(b, dict) and b.get("block")
            ],
            "insight": prose["cost_driver_insight"],
        },
        "metrics": metrics,
        "stages": stages,
        "bom": {
            "rows": bom_rows,
            "subtotal_inr": round(bom_subtotal, 2),
            "live_count": numeric["live_count"],
            "est_count": len(lines) - numeric["live_count"],
        },
        "fab": {
            "params": fab.get("params") or {},
            "fab_by_volume": fab.get("fab_by_volume") or {},
            "selected_inr": (fab.get("fab_by_volume") or {}).get(volume),
            "source": fab.get("source"),
            "tag": fab.get("tag"),
            "live": fab.get("live"),
        },
        "assembly": assembly,
        "non_quotable": non_quotable,
        "volumeCurve": volume_curve,
        "marketContext": {
            "prose": prose["market_context"],
            "comparables": market.get("comparables") or [],
            "observations": market.get("observations") or [],
            "margin_band": market.get("margin_band"),
            "source": market.get("source"),
            "tag": "Est",
        },
        "methodology": methodology,
        "dataQuality": data_quality,
        "dataConfidence": {"prose": prose["data_confidence"], "all_live": all_live},
        "fx": fx_info,
        # Raw state for the edit/regenerate flow — recompute operates on this, never
        # a fresh teardown run. Not consumed by the template.
        "_compute": {
            "lines": lines,
            "fab": fab,
            "assembly": assembly,
            "non_quotable": non_quotable,
            "fx": fx_info,
            "curve_volumes": curve_volumes,
            "product_label": product_label,
            "volume": volume,
        },
    }
    return report_json


def recompute_numeric(report_json: dict[str, Any]) -> dict[str, Any]:
    """Rebuild every numeric section from ``_compute`` after a data edit.

    Only the requested fields and their direct downstream roll-ups change — prose,
    methodology, market and overview sections are left untouched. Returns the same
    dict, mutated in place.
    """
    state = report_json.get("_compute") or {}
    lines = state.get("lines") or []
    fab = state.get("fab") or {}
    assembly = state.get("assembly") or {}
    non_quotable = state.get("non_quotable") or {}
    curve_volumes = state.get("curve_volumes") or list(REPORT_VOLUME_CURVE)
    volume = int(state.get("volume") or (curve_volumes[2] if len(curve_volumes) > 2 else curve_volumes[-1]))

    numeric = build_numeric_sections(lines, fab, assembly, non_quotable, volume, curve_volumes)

    report_json["stages"] = numeric["stages"]
    report_json["volumeCurve"] = numeric["volume_curve"]
    report_json["metrics"] = numeric["metrics"]
    report_json["bom"] = {
        "rows": numeric["bom_rows"],
        "subtotal_inr": numeric["bom_subtotal"],
        "live_count": numeric["live_count"],
        "est_count": len(lines) - numeric["live_count"],
    }
    fab_section = report_json.setdefault("fab", {})
    fab_section["fab_by_volume"] = fab.get("fab_by_volume") or {}
    fab_section["selected_inr"] = numeric["fab_selected_inr"]
    fab_section["params"] = fab.get("params") or fab_section.get("params") or {}
    fab_section["source"] = fab.get("source", fab_section.get("source"))
    fab_section["tag"] = fab.get("tag", fab_section.get("tag"))
    fab_section["live"] = fab.get("live", fab_section.get("live"))
    report_json["assembly"] = assembly
    report_json["non_quotable"] = non_quotable
    meta = report_json.setdefault("meta", {})
    meta["volume"] = volume
    return report_json


def _inputs_label(structured: dict[str, Any]) -> str:
    n = len(structured.get("components") or [])
    return f"{n} identified components" if n else "teardown analysis"


def _overview_kv(
    product: dict[str, Any],
    enclosure: dict[str, Any],
    pcb: dict[str, Any],
    lines: list[dict[str, Any]],
) -> list[dict[str, str]]:
    kv: list[dict[str, str]] = []

    def add(label: str, value: Any) -> None:
        if value not in (None, "", []):
            kv.append({"k": label, "v": str(value)})

    add("Product type", product.get("type") or product.get("product_class"))
    add("Primary function", product.get("primary_function"))
    add("Enclosure", enclosure.get("material") and (
        f"{enclosure.get('material')}"
        + (f", {enclosure.get('finish')}" if enclosure.get("finish") else "")
    ))
    dims = pcb.get("dimensions_mm") or {}
    if dims.get("length") and dims.get("width"):
        add("PCB size", f"{dims.get('length')} × {dims.get('width')} mm")
    add("PCB layers", pcb.get("layer_count_estimate"))
    add("Surface finish", pcb.get("surface_finish"))
    add("Unique BOM lines", len(lines))
    add("Total placements", sum(int(l["qty"]) for l in lines))
    return kv
