"""Orchestration for should-cost report generation.

Pulls the project knowledge base, decides what (if anything) we must ask the
user, resolves and prices the BOM via DigiKey, computes transparent heuristic
estimates for the non-BOM cost blocks, and composes the final report markdown
with the LLM. The HILT ``interrupt()`` itself lives in the agent tool; this
module is pure (callbacks aside) so it can be unit-reasoned about in isolation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from src.core.config import REPORT_MAX_QUESTIONS
from src.services import digikey
from src.services.llm_analysis import invoke_llm, parse_json_object

logger = logging.getLogger(__name__)

Progress = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:  # pragma: no cover - default sink
    pass


# --------------------------------------------------------------------------- #
# Gap assessment -> HILT questions
# --------------------------------------------------------------------------- #
_ASSESS_PROMPT = """You are preparing a professional electronics should-cost report for a physical product. \
Below is everything currently known about the product (a prose theory summary and a structured JSON breakdown), \
plus the user's request.

This is a SINGLE-UNIT cost report — do NOT ask about production volume, order quantity, or annual demand; \
those are irrelevant here.

Decide whether there is ENOUGH information to produce a credible should-cost report, or whether a SMALL number of \
targeted questions to the user would materially improve it. Only ask about MATERIAL physical unknowns that move the \
per-unit cost and that the user can actually answer from the product in hand, for example:
- An expensive or unidentified IC whose marking is illegible (ask the user to read the top-mark or upload a clearer photo).
- PCB layer count or board dimensions, when missing and needed for fab costing.
- Enclosure material or manufacturing process, when missing and needed for tooling/enclosure costing.

Rules:
- Ask AT MOST {max_questions} questions. Fewer is better. If the data is already sufficient, return an empty list.
- Never ask for Gerbers, CAD, schematics, firmware, or production volume — only physical things a teardown/photo/measurement can provide.
- Each question may request a short text answer (kind "text") or ask the user to upload a file/photo (kind "file").
- Give each question a short, stable, descriptive "id" (e.g. "ic_u3_marking", "pcb_layers", "enclosure_material").
- Mark a question optional when the report can still be produced without it.

Return ONLY a JSON object of this exact shape, nothing else:
{{"questions": [{{"id": "pcb_layers", "prompt": "...", "kind": "text", "optional": true, "why": "..."}}]}}

USER REQUEST:
{user_request}

THEORY SUMMARY:
{theory}

STRUCTURED JSON:
{structured}
"""


def assess_missing(
    theory: str,
    structured: dict[str, Any] | None,
    user_request: str,
) -> list[dict[str, Any]]:
    """Ask the LLM which (if any) minimal questions to put to the user."""
    prompt = _ASSESS_PROMPT.format(
        max_questions=REPORT_MAX_QUESTIONS,
        user_request=(user_request or "Generate a should-cost report.")[:2000],
        theory=(theory or "(none)")[:8000],
        structured=json.dumps(structured or {}, ensure_ascii=False)[:8000],
    )
    try:
        raw = invoke_llm([{"role": "user", "content": prompt}], max_tokens=1200)
        data = parse_json_object(raw) or {}
    except Exception:
        logger.exception("Report gap assessment failed; proceeding without questions")
        return []
    questions = data.get("questions")
    if not isinstance(questions, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for index, q in enumerate(questions[:REPORT_MAX_QUESTIONS]):
        if not isinstance(q, dict):
            continue
        prompt_text = str(q.get("prompt") or "").strip()
        if not prompt_text:
            continue
        kind = "file" if str(q.get("kind")).lower() == "file" else "text"
        cleaned.append(
            {
                "id": str(q.get("id") or f"q{index + 1}"),
                "prompt": prompt_text,
                "kind": kind,
                "optional": bool(q.get("optional", True)),
                "why": str(q.get("why") or "").strip(),
            }
        )
    return cleaned


# --------------------------------------------------------------------------- #
# BOM resolution + pricing via DigiKey
# --------------------------------------------------------------------------- #
def _search_term(component: dict[str, Any]) -> str:
    """Build a DigiKey search term: prefer markings, else type+package+value."""
    mark = component.get("top_mark") or component.get("markings")
    if mark:
        return str(mark).strip()
    parts = [
        str(component.get(k)).strip()
        for k in ("type", "package", "value")
        if component.get(k)
    ]
    return " ".join(parts).strip()


def _pick_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best candidate: prefer active, then most available stock."""
    if not candidates:
        return None
    active = [
        c
        for c in candidates
        if not c.get("discontinued")
        and not c.get("end_of_life")
        and str(c.get("status") or "").lower() == "active"
    ]
    pool = active or candidates
    return max(pool, key=lambda c: c.get("quantity_available") or 0)


def resolve_and_price_bom(
    structured: dict[str, Any] | None,
    volume: int,
    progress: Progress = _noop,
) -> dict[str, Any]:
    """Resolve component markings to MPNs and price the BOM at ``volume``."""
    components = (structured or {}).get("components") or []
    lines: list[dict[str, Any]] = []
    notes: list[str] = []
    subtotal = 0.0
    configured = digikey.is_configured()
    if not configured:
        notes.append("DigiKey API not configured — BOM pricing was not run.")

    for component in components:
        if not isinstance(component, dict):
            continue
        label = component.get("type") or component.get("function") or "Component"
        ref_des = component.get("ref_des") or ""
        try:
            qty = int(component.get("qty_per_unit") or 1)
        except (TypeError, ValueError):
            qty = 1
        existing_mpn = (component.get("mpn") or "").strip()

        line: dict[str, Any] = {
            "ref_des": ref_des,
            "label": label,
            "mpn": existing_mpn or None,
            "qty_per_unit": qty,
            "unit_price": None,
            "line_cost": None,
            "source": "unresolved",
            "status": None,
            "note": "",
        }

        if not configured:
            lines.append(line)
            continue

        progress(
            {
                "stage": "digikey",
                "message": f"Pricing {label}{f' ({ref_des})' if ref_des else ''}…",
            }
        )

        part: dict[str, Any] | None = None
        try:
            if existing_mpn:
                part = digikey.product_details(existing_mpn)
            if part is None:
                term = _search_term(component)
                if term:
                    part = _pick_candidate(digikey.search_keyword(term, limit=8))
        except Exception as exc:
            logger.warning("DigiKey lookup failed for %s: %s", label, exc)
            line["note"] = "DigiKey lookup failed."
            lines.append(line)
            continue

        if part is None:
            line["note"] = "No matching part found on DigiKey."
            lines.append(line)
            continue

        unit_price = digikey.price_for_qty(part.get("price_breaks") or [], volume)
        line["mpn"] = part.get("mpn") or existing_mpn or None
        line["status"] = part.get("status")
        line["source"] = "digikey"
        if part.get("discontinued") or part.get("end_of_life"):
            note = "Marked obsolete/EOL on DigiKey — substitute recommended."
            line["note"] = note
            notes.append(f"{line['mpn'] or label}: {note}")
        if unit_price is not None:
            line["unit_price"] = round(unit_price, 5)
            line["line_cost"] = round(unit_price * qty, 5)
            subtotal += line["line_cost"]
        else:
            line["note"] = (line["note"] + " No price break available.").strip()
        lines.append(line)

    return {
        "lines": lines,
        "per_unit_subtotal_usd": round(subtotal, 4) if lines else None,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Heuristic estimates for the non-BOM cost blocks
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_costs(
    structured: dict[str, Any] | None,
    bom: dict[str, Any],
    volume: int,
) -> dict[str, Any]:
    """Assemble the ``costs`` structure consumed by the report prompt."""
    structured = structured or {}
    costs: dict[str, Any] = {"volume": volume, "bom": bom}
    present: list[float] = []
    missing: list[str] = []

    subtotal = bom.get("per_unit_subtotal_usd")
    if isinstance(subtotal, (int, float)):
        present.append(float(subtotal))
    else:
        missing.append("BOM pricing")

    # PCB fab: area (cm^2) x layers x rate. Needs board dimensions + layer count.
    pcb = structured.get("pcb") or {}
    dims = pcb.get("dimensions_mm") or {}
    length, width = _num(dims.get("length")), _num(dims.get("width"))
    layers = _num(pcb.get("layer_count_estimate")) or 2
    if length and width:
        area_cm2 = (length / 10.0) * (width / 10.0)
        rate = 0.022 if volume < 1000 else 0.014
        per_unit = max(0.50, area_cm2 * layers * rate)
        costs["pcb_fab"] = {
            "per_unit_usd": round(per_unit, 4),
            "basis": "estimate",
            "parameters": {
                "board_area_cm2": round(area_cm2, 2),
                "layers": int(layers),
                "rate_usd_per_cm2_layer": rate,
                "volume": volume,
            },
        }
        present.append(per_unit)
    else:
        costs["pcb_fab"] = None
        missing.append("PCB fab (board dimensions/layer count required)")

    # Assembly: per-placement SMT cost + amortized setup.
    components = structured.get("components") or []
    placements = 0
    for component in components:
        if isinstance(component, dict):
            try:
                placements += int(component.get("qty_per_unit") or 1)
            except (TypeError, ValueError):
                placements += 1
    if placements:
        per_place = 0.015 if volume < 1000 else 0.010
        setup = 250.0 / max(volume, 1)
        per_unit = round(placements * per_place + setup, 4)
        costs["assembly"] = {
            "per_unit_usd": per_unit,
            "basis": "estimate",
            "parameters": {
                "placements": placements,
                "cost_per_placement_usd": per_place,
                "setup_amortized_usd": round(setup, 4),
                "volume": volume,
            },
        }
        present.append(per_unit)
    else:
        costs["assembly"] = None
        missing.append("Assembly (component count required)")

    # Enclosure: only estimate when material + dimensions are known.
    enclosure = structured.get("enclosure") or {}
    material = enclosure.get("material")
    edims = enclosure.get("dimensions_mm") or {}
    el, ew, eh = _num(edims.get("length")), _num(edims.get("width")), _num(edims.get("height"))
    if material and el and ew and eh:
        volume_cm3 = (el / 10.0) * (ew / 10.0) * (eh / 10.0)
        rate = 0.004 if volume < 1000 else 0.0025
        per_unit = round(max(0.30, volume_cm3 * rate), 4)
        costs["enclosure"] = {
            "per_unit_usd": per_unit,
            "basis": "estimate",
            "parameters": {
                "material": material,
                "process": enclosure.get("process"),
                "bounding_volume_cm3": round(volume_cm3, 2),
                "volume": volume,
            },
        }
        present.append(per_unit)
    else:
        costs["enclosure"] = None
        missing.append("Enclosure (material + dimensions required)")

    if present:
        costs["total"] = {
            "per_unit_usd": round(sum(present), 4),
            "missing_blocks": missing,
        }
    else:
        costs["total"] = None
    return costs


# --------------------------------------------------------------------------- #
# Final markdown composition
# --------------------------------------------------------------------------- #
_COMPOSE_SYSTEM = "You are an expert electronics should-cost analyst and technical writer."


def compose_report_markdown(
    report_prompt: str,
    theory: str,
    structured: dict[str, Any] | None,
    costs: dict[str, Any],
    volume: int,
    web_context: str | None,
    *,
    precomputed: dict[str, str] | None = None,
) -> str:
    """Compose the final report markdown from all gathered data."""
    pre = precomputed or {}
    precomputed_block = ""
    if pre:
        precomputed_block = (
            "\n\nPRE-COMPUTED INR TABLES AND FIGURES (paste verbatim; never use USD):\n"
            f"\n### Executive Summary table\n{pre.get('executive_table', '')}\n"
            f"\n### BOM table\n{pre.get('bom_table', '')}\n"
        )
        if pre.get("bom_notes"):
            precomputed_block += f"\n### BOM notes\n{pre['bom_notes']}\n"
        precomputed_block += (
            f"\n### Fab Costing (INR)\n{pre.get('fab_text', '')}\n"
            f"\n### Assembly (INR)\n{pre.get('assembly_text', '')}\n"
            f"\n### Enclosure (INR)\n{pre.get('enclosure_text', '')}\n"
            f"\n### Total should-cost (INR)\n{pre.get('total_text', '')}\n"
        )

    data_block = json.dumps(
        {
            "volume": volume,
            "theory": theory,
            "structured": structured or {},
            "components": (structured or {}).get("components") or [],
            "dimensions": (structured or {}).get("pcb", {}),
            "materials": (structured or {}).get("enclosure", {}),
            "enclosure": (structured or {}).get("enclosure", {}),
            "costs": costs,
            "web_context": web_context or "",
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = (
        report_prompt.replace("{volume}", str(volume))
        + precomputed_block
        + "\n\nDATA (use ONLY this; output ONLY the report markdown — no preamble, no code fences):\n"
        + data_block
    )
    raw = invoke_llm(
        [
            {"role": "system", "content": _COMPOSE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=8192,
    )
    return raw.strip()
