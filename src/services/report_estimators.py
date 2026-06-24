"""LLM-backed estimation + narration for the should-cost report.

This is the *narration / soft-estimate* half of the pipeline — everything the
APIs can't give us directly. The hard compute (duty math, landed cost, assembly
joints, volume amortization, table fill) lives in code (the pipeline + template),
never here. Per the governing principle: **the LLM narrates, the code computes.**

Every function is defensive: on any LLM failure it logs and returns a safe
default so the report always generates. Callers fold failures into ``dataQuality``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.services.failure_log import record_failure
from src.services.llm_analysis import invoke_llm, parse_json_object

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Step 3 — MPN resolution
# --------------------------------------------------------------------------- #
_RESOLVE_PROMPT = """You are an electronics sourcing expert. For each component below (from a teardown of a \
physical product), propose the most likely REAL manufacturer part number (MPN) so it can be priced against a \
parts database (JLCPCB/LCSC catalog).

Rules:
- Use the component's type, package, function, value and top_mark to infer the MPN.
- Generic passives (plain resistors/capacitors/inductors with no distinctive marking) usually have NO meaningful \
single MPN — mark "is_generic_passive": true for these and do NOT invent an MPN.
- For real ICs/connectors/crystals/modules, propose a concrete orderable MPN. If a top_mark is present, resolve it.
- Where ambiguous, give a primary MPN plus up to 2 alternates.
- Never fabricate a part you cannot justify from the evidence — if you truly cannot resolve it, set \
"resolved_mpn": null and explain in "rationale".
- "category" must be one of: resistor, capacitor, inductor, ferrite, diode, led, transistor, ic, mcu, \
connector, crystal, oscillator, switch, sensor, module, electromechanical, other.

Return ONLY a JSON object of this shape:
{{"resolutions": [{{"ref_des": "...", "resolved_mpn": "...|null", "alternates": ["..."], \
"is_generic_passive": false, "category": "ic", "rationale": "..."}}]}}

COMPONENTS (JSON):
{components}
"""


def resolve_mpns(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a map ref_des/index -> resolution. Falls back to per-component guesses."""
    slim = []
    for idx, c in enumerate(components or []):
        if not isinstance(c, dict):
            continue
        slim.append(
            {
                "ref_des": c.get("ref_des") or f"#{idx + 1}",
                "type": c.get("type"),
                "package": c.get("package"),
                "value": c.get("value"),
                "function": c.get("function"),
                "top_mark": c.get("top_mark") or c.get("markings"),
                "existing_mpn": c.get("mpn"),
            }
        )
    if not slim:
        return {}
    try:
        raw = invoke_llm(
            [{"role": "user", "content": _RESOLVE_PROMPT.format(
                components=json.dumps(slim, ensure_ascii=False)[:9000]
            )}],
            max_tokens=3000,
        )
        data = parse_json_object(raw) or {}
        resolutions = data.get("resolutions") or []
    except Exception as exc:
        logger.warning("MPN resolution LLM call failed; using heuristic fallback", exc_info=True)
        record_failure(
            "resolving_mpns", "MPN resolution",
            "Could not auto-resolve part numbers — falling back to category estimates",
            error=exc, context={"component_count": len(slim)},
        )
        resolutions = []

    out: dict[str, dict[str, Any]] = {}
    for res in resolutions:
        if isinstance(res, dict) and res.get("ref_des"):
            out[str(res["ref_des"])] = res
    return out


# --------------------------------------------------------------------------- #
# Step 6 (assist) — HSN classification, ONLY when the parts DB gave no HSN at all
# --------------------------------------------------------------------------- #
_HSN_PROMPT = """You are a customs-classification assistant for electronic components imported into India. \
For each component below, give the most likely 4-digit HSN heading (Indian customs). Use these common headings:
- 8542 integrated circuits
- 8541 diodes, LEDs, transistors, photo devices
- 8533 resistors
- 8532 capacitors
- 8504 inductors / transformers / power supply
- 8536 connectors, switches, relays (< 1000V)
Pick the closest 4-digit heading; if unsure use "8542" for active silicon or "8536" for electromechanical.

Return ONLY JSON: {{"hsn": [{{"ref_des": "...", "hsn4": "8542"}}]}}

COMPONENTS (JSON):
{components}
"""


def classify_hsns(components: list[dict[str, Any]]) -> dict[str, str]:
    """Map ref_des -> 4-digit HSN for components the parts DB couldn't supply one for."""
    if not components:
        return {}
    slim = [
        {
            "ref_des": c.get("ref_des") or f"#{i + 1}",
            "type": c.get("type"),
            "function": c.get("function"),
            "package": c.get("package"),
        }
        for i, c in enumerate(components)
        if isinstance(c, dict)
    ]
    try:
        raw = invoke_llm(
            [{"role": "user", "content": _HSN_PROMPT.format(
                components=json.dumps(slim, ensure_ascii=False)[:6000]
            )}],
            max_tokens=1500,
        )
        data = parse_json_object(raw) or {}
        rows = data.get("hsn") or []
    except Exception as exc:
        logger.warning("HSN classification failed; duty will use defaults", exc_info=True)
        record_failure(
            "duty", "HSN classification",
            "Could not classify HSN codes — applying default duty rates",
            error=exc, context={"component_count": len(slim)},
        )
        return {}
    out: dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("ref_des") and row.get("hsn4"):
            out[str(row["ref_des"])] = str(row["hsn4"])
    return out


# --------------------------------------------------------------------------- #
# Step 4F — Non-quotable estimates (firmware / enclosure tooling / labour / pack)
# --------------------------------------------------------------------------- #
_NON_QUOTABLE_PROMPT = """You are an Indian EMS (electronics manufacturing services) cost estimator. The APIs \
cannot price these non-quotable cost blocks, so estimate them from typical Indian EMS reference rates. Separate \
ONE-TIME NRE (non-recurring engineering: tooling, firmware development, line setup) from PER-UNIT recurring cost.

Use realistic Indian rates (INR). Be conservative and label everything an estimate.

Cost blocks (give each as INR):
- firmware: nre_inr (firmware development/bring-up, one-time), per_unit_inr (flashing + functional test time).
- enclosure: nre_inr (injection mould / tooling, one-time — large for moulded plastic), per_unit_inr (moulded \
part material + finishing). If the enclosure is simple/off-the-shelf, keep nre small.
- final_assembly: nre_inr (jig/fixture + line setup, one-time), per_unit_inr (manual assembly labour + final \
test + retail packaging).

Return ONLY JSON:
{{"firmware": {{"nre_inr": 0, "per_unit_inr": 0}},
  "enclosure": {{"nre_inr": 0, "per_unit_inr": 0, "process": "...", "material": "..."}},
  "final_assembly": {{"nre_inr": 0, "per_unit_inr": 0}},
  "notes": ["short assumption notes"]}}

PRODUCT CONTEXT (JSON):
{context}
"""

_NON_QUOTABLE_FALLBACK = {
    "firmware": {"nre_inr": 80000.0, "per_unit_inr": 8.0},
    "enclosure": {"nre_inr": 250000.0, "per_unit_inr": 60.0, "process": "injection moulding", "material": "ABS"},
    "final_assembly": {"nre_inr": 20000.0, "per_unit_inr": 35.0},
    "notes": ["Non-quotable blocks use default Indian EMS reference rates (live estimate unavailable)."],
}


def estimate_non_quotable(structured: dict[str, Any]) -> dict[str, Any]:
    """LLM estimate of firmware/enclosure/final-assembly NRE + per-unit (INR)."""
    context = {
        "product": (structured or {}).get("product") or {},
        "enclosure": (structured or {}).get("enclosure") or {},
        "pcb": (structured or {}).get("pcb") or {},
        "component_count": len((structured or {}).get("components") or []),
    }
    try:
        raw = invoke_llm(
            [{"role": "user", "content": _NON_QUOTABLE_PROMPT.format(
                context=json.dumps(context, ensure_ascii=False)[:6000]
            )}],
            max_tokens=1500,
        )
        data = parse_json_object(raw)
    except Exception as exc:
        logger.warning("Non-quotable estimate failed; using fallback rates", exc_info=True)
        record_failure(
            "non_quotable", "Non-quotable estimate",
            "Could not estimate firmware/tooling/labour — using default reference rates",
            error=exc, context={"component_count": context.get("component_count")},
        )
        data = None
    if not isinstance(data, dict):
        return {**_NON_QUOTABLE_FALLBACK, "fallback": True}

    def _block(key: str, defaults: dict[str, Any]) -> dict[str, Any]:
        block = data.get(key) if isinstance(data.get(key), dict) else {}
        out = dict(defaults)
        for k in ("nre_inr", "per_unit_inr"):
            try:
                if block.get(k) is not None:
                    out[k] = float(block[k])
            except (TypeError, ValueError):
                pass
        for k in ("process", "material"):
            if block.get(k):
                out[k] = block[k]
        return out

    return {
        "firmware": _block("firmware", _NON_QUOTABLE_FALLBACK["firmware"]),
        "enclosure": _block("enclosure", _NON_QUOTABLE_FALLBACK["enclosure"]),
        "final_assembly": _block("final_assembly", _NON_QUOTABLE_FALLBACK["final_assembly"]),
        "notes": data.get("notes") if isinstance(data.get("notes"), list) else [],
        "fallback": False,
    }


# --------------------------------------------------------------------------- #
# Step 4G — Market context via Tavily (web), summarized by the LLM
# --------------------------------------------------------------------------- #
_MARKET_SUMMARY_PROMPT = """You are a hardware market analyst. Below are raw web search snippets about a product \
and its components, plus the product context. Summarize ONLY what the snippets support — never invent figures.

Produce JSON with:
- "observations": 3-5 short sourcing/supply/market bullet strings (concentration risk, lead time, price \
volatility, second-source advice) grounded in the snippets.
- "comparables": up to 3 retail comparables [{{"name": "...", "retail_mrp_inr": 0, "note": "..."}}] ONLY if the \
snippets mention retail/MRP prices; otherwise return an empty list.
- "margin_band": short string like "45-60%" ONLY if retail comparables exist; else null.

Return ONLY that JSON object.

PRODUCT CONTEXT (JSON):
{context}

WEB SNIPPETS:
{snippets}
"""


def summarize_market(product_label: str, structured: dict[str, Any], snippets: str) -> dict[str, Any]:
    """Turn raw Tavily snippets into structured market context. Tagged Est elsewhere."""
    if not snippets or not snippets.strip():
        return {"observations": [], "comparables": [], "margin_band": None, "had_web": False}
    context = {"product_label": product_label, "product": (structured or {}).get("product") or {}}
    try:
        raw = invoke_llm(
            [{"role": "user", "content": _MARKET_SUMMARY_PROMPT.format(
                context=json.dumps(context, ensure_ascii=False)[:3000],
                snippets=snippets[:6000],
            )}],
            max_tokens=1500,
        )
        data = parse_json_object(raw) or {}
    except Exception as exc:
        logger.warning("Market summary failed", exc_info=True)
        record_failure(
            "market_context", product_label or "market",
            "Could not summarize web market context — omitting comparables",
            error=exc,
        )
        return {"observations": [], "comparables": [], "margin_band": None, "had_web": False}
    return {
        "observations": data.get("observations") if isinstance(data.get("observations"), list) else [],
        "comparables": data.get("comparables") if isinstance(data.get("comparables"), list) else [],
        "margin_band": data.get("margin_band"),
        "had_web": True,
    }


# --------------------------------------------------------------------------- #
# Step 10 — Prose only (Executive Summary, Architecture, Market Context, Data
# Confidence). The LLM writes prose; it NEVER formats numbers/tables.
# --------------------------------------------------------------------------- #
_PROSE_PROMPT = """You are an expert electronics should-cost analyst and technical writer. Write the PROSE \
sections of a should-cost report. You are given the product analysis and the FINAL computed cost figures — \
do NOT recompute, restate, or tabulate numbers; reference them only in narrative where useful. Write in a \
professional, factual tone. All money is INR (₹). Never invent component prices.

IMPORTANT: This is a SINGLE-UNIT cost report. Do NOT discuss production volumes, order quantities, volume \
discounts, economies of scale, or how cost changes at 1k/10k units. Speak only about the per-unit cost. Do NOT \
use the term "ex-works" — call it the "per-unit cost" or "unit cost". The one-time NRE \
(``one_time_nre_inr_separate``) is a SEPARATE one-time investment — it is NOT part of the per-unit cost; never add \
it into the unit figure or say the unit costs hundreds of thousands.

Write these fields (Markdown prose, no headings, no tables, no code fences):
- "executive_summary": 3-5 sentences: what the product is and the headline cost story.
- "key_findings": array of 3-4 short bullet strings about the main cost drivers / confidence.
- "architecture_analysis": 2-4 sentences on the electronics architecture / topology.
- "cost_driver_insight": 1-2 sentences naming the dominant cost lever.
- "market_context": 2-3 sentences positioning cost vs the comparable retail tier and sourcing risk.
- "data_confidence": 2-3 plain-language sentences for a "Data Confidence & Notes" callout, written from the \
data-quality notes provided. NEVER use technical terms (no "API", "timeout", "null", "exception", HTTP codes, \
module names). If everything was sourced live, say so positively. If some figures used estimates, say so plainly \
and advise confirming before quoting.

Return ONLY a JSON object with exactly those keys.

PRODUCT (JSON):
{product}

COMPUTED FIGURES (JSON):
{figures}

DATA-QUALITY NOTES (plain strings):
{data_quality}
"""

_PROSE_FALLBACK = {
    "executive_summary": "This report reconstructs the per-unit manufacturing cost of the product from a "
    "physical teardown, combining live component pricing, a PCB fabrication quote, customs duty and "
    "estimated assembly and tooling costs. All figures are single-unit landed cost in Indian Rupees.",
    "key_findings": [
        "The bill of materials is the largest share of the per-unit cost.",
        "Active components and the enclosure are the dominant cost lines.",
        "Some figures are industry estimates where live data was unavailable — confirm before quoting.",
    ],
    "architecture_analysis": "The electronics are organized into the functional blocks identified during the "
    "teardown, built around the main processing/control component and its supporting power and I/O circuitry.",
    "cost_driver_insight": "The highest-value active components and the enclosure tooling are the dominant "
    "cost levers in this design.",
    "market_context": "The reconstructed per-unit landed cost leaves a typical margin band against comparable "
    "retail units; sourcing risk centers on single-sourced active components.",
    "data_confidence": "Most figures in this report are based on live sourcing data. Where a live figure was "
    "unavailable, an industry estimate was used — please confirm those before using the numbers in a quotation.",
}


def write_prose(
    product: dict[str, Any],
    figures: dict[str, Any],
    data_quality: list[str],
) -> dict[str, Any]:
    """Generate the prose fields. Falls back to neutral copy on any failure."""
    try:
        raw = invoke_llm(
            [{"role": "user", "content": _PROSE_PROMPT.format(
                product=json.dumps(product or {}, ensure_ascii=False)[:5000],
                figures=json.dumps(figures or {}, ensure_ascii=False)[:5000],
                data_quality=json.dumps(data_quality or [], ensure_ascii=False)[:3000],
            )}],
            max_tokens=2000,
        )
        data = parse_json_object(raw)
    except Exception as exc:
        logger.warning("Prose generation failed; using fallback copy", exc_info=True)
        record_failure(
            "rendering", "Report narrative",
            "Could not generate the report narrative — using neutral fallback copy",
            error=exc,
        )
        data = None
    if not isinstance(data, dict):
        return dict(_PROSE_FALLBACK)
    out = dict(_PROSE_FALLBACK)
    for key in _PROSE_FALLBACK:
        if data.get(key):
            out[key] = data[key]
    if not isinstance(out["key_findings"], list):
        out["key_findings"] = _PROSE_FALLBACK["key_findings"]
    return out
