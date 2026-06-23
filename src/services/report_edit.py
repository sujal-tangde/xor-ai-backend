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
import re
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
Your job is to translate ANY edit request into one or more concrete operations against the report state below, \
choosing the most specific operation(s) that capture the user's intent. Do NOT compute new numbers — only describe \
WHAT to change; the system recomputes costs deterministically.

Guiding rules:
- Pick the MOST SPECIFIC operation. Renaming a SECTION's heading is NOT the same as renaming the whole report — \
do not use set_title for a section heading, and do not use set_section_title for the report's overall title.
- Emit MULTIPLE operations when the user asks for several changes in one message.
- The report has a fixed set of sections (executive, product_overview, architecture, cost_by_stage, bom, \
fab_assembly, market, methodology); you can rename their headings, hide them, rewrite their narrative text, or \
edit their data, but you cannot invent brand-new section types.
- If a request is narrative/wording for a section that has an editable prose field, use rewrite_prose. If it does \
not map to any operation below, fall back to a rewrite_prose on the closest section so something reasonable happens.

Operation types you may emit (in an "operations" array):
- {{"op": "set_title", "title": "<new report title>"}}        rename the WHOLE report (the cover/document title only)
- {{"op": "set_section_title", "section": "methodology", "title": "<new heading>"}}   rename ONE section's HEADING \
text (e.g. the user says: change section "08 · Methodology & Confidence" to "08 · Myth & Confi" → \
{{"op":"set_section_title","section":"methodology","title":"Myth & Confi"}}; give ONLY the heading words, never the \
leading number). section is one of: executive | product_overview | architecture | cost_by_stage | bom | \
fab_assembly | market | methodology
- {{"op": "remove_section", "section": "architecture"}}       hide a whole section. section is one of the keys above
- {{"op": "add_image", "position": "after_executive", "caption": "<optional caption>", "url": "<image URL or null>"}}   \
embed an image in the report. Set "url" when the user PASTED an image URL/link in their message; otherwise leave it \
null and the user's ATTACHED image(s) are used. Use position "after_executive" if they want it below the executive \
summary, else "end". Emit whenever the user asks to attach/add/embed/include/insert an image, photo or picture.
- {{"op": "remove_image", "match": "<optional caption/url substring, or null>"}}   remove an embedded image. \
Emit whenever the user asks to remove/delete/get rid of an image, photo or picture (e.g. "remove the image below \
the executive summary"). Leave "match" null to remove ALL images, or set it to part of a caption/URL to remove one.
- {{"op": "set_volume", "volume": 5000}}                      target production volume
- {{"op": "set_qty", "match": "<mpn/designator/text>", "qty": 3}}
- {{"op": "remove_line", "match": "<mpn/designator/text>"}}
- {{"op": "set_unit_price", "match": "<...>", "price_inr": 12.5}}   user-supplied price (will be tagged Est)
- {{"op": "set_stage_cost", "target": "assembly"|"pcb_fabrication"|"firmware"|"enclosure"|"final_assembly", "value_inr": 90}}   \
set the PER-UNIT cost of one manufacturing stage in the "Cost by Manufacturing Stage" section DIRECTLY to a number. \
Use this whenever the user names a stage and a target rupee amount, e.g. "make PCB assembly cost ₹90", "set the \
enclosure cost to 60", "PCB assembly costing to 90". target "assembly" = the "PCB assembly" row. Prefer this over \
set_rate when the user gives a target TOTAL per-unit amount for a stage (set_rate is only for the per-joint/setup/\
stencil rate-card inputs, not the final per-unit figure). You cannot set "Component BOM (landed)" this way — that row \
is derived from BOM lines; edit lines/prices instead.
- {{"op": "set_rate", "target": "per_joint"|"setup"|"stencil", "value": 0.2}}   assembly rate card (per-joint/setup/stencil inputs)
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
    non_quotable = compute.get("non_quotable") or {}
    fab = compute.get("fab") or {}
    volume = compute.get("volume")

    def _nq(block: str) -> Any:
        return (non_quotable.get(block) or {}).get("per_unit_inr")

    fab_by_volume = fab.get("fab_by_volume") or {}
    fab_cost = fab_by_volume.get(volume)
    if fab_cost is None and fab_by_volume:
        fab_cost = next(iter(fab_by_volume.values()))

    # Current per-unit cost of each manufacturing stage (section "Cost by
    # Manufacturing Stage"). Exposed so the classifier can map a request like
    # "make PCB assembly cost ₹90" to a set_stage_cost on the right target
    # instead of guessing a per-joint rate.
    stage_costs = {
        "pcb_fabrication": fab_cost,
        "assembly": assembly.get("per_unit_inr"),
        "firmware": _nq("firmware"),
        "enclosure": _nq("enclosure"),
        "final_assembly": _nq("final_assembly"),
    }
    return {
        "volume": volume,
        "stage_costs_per_unit_inr": stage_costs,
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


# Canonical section keys (match the renderer) + the aliases a model might emit.
_SECTION_ALIASES = {
    "executive": "executive", "executive_summary": "executive", "summary": "executive",
    "product_overview": "product_overview", "product": "product_overview", "overview": "product_overview",
    "architecture": "architecture", "architecture_analysis": "architecture", "arch": "architecture",
    "cost_by_stage": "cost_by_stage", "cost_by_manufacturing_stage": "cost_by_stage",
    "stages": "cost_by_stage", "manufacturing_stage": "cost_by_stage",
    "bom": "bom", "bill_of_materials": "bom",
    "fab_assembly": "fab_assembly", "fabrication": "fab_assembly", "assembly": "fab_assembly",
    "pcb_fabrication": "fab_assembly", "fab": "fab_assembly",
    "market": "market", "market_context": "market",
    "methodology": "methodology", "confidence": "methodology", "methodology_confidence": "methodology",
}


def _canonical_section(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    if not key:
        return None
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    # Tolerate a leading section number / extra words ("3_architecture_analysis").
    # Prefer the longest alias that appears as a token-substring of the key.
    for alias in sorted(_SECTION_ALIASES, key=len, reverse=True):
        if alias in key:
            return _SECTION_ALIASES[alias]
    return None


def _add_images(report_json: dict[str, Any], image_refs: list[dict[str, Any]],
                position: str, caption: str | None) -> int:
    if not image_refs:
        return 0
    images = report_json.setdefault("images", [])
    existing_urls = {img.get("url") for img in images if isinstance(img, dict)}
    pos = "after_executive" if str(position).lower() == "after_executive" else "end"
    added = 0
    for ref in image_refs:
        url = ref.get("url")
        if not url or url in existing_urls:
            continue
        images.append({"url": url, "caption": caption or ref.get("name") or "", "position": pos})
        existing_urls.add(url)
        added += 1
    return added


# Report-title renames are the single most common edit and the classifier
# regularly confuses them with a SECTION-heading rename (the prompt leans hard on
# that distinction). Detect the unambiguous phrasings deterministically so a title
# change never depends on the LLM — and skip the classifier call entirely (and its
# tokens) when the whole request is just a title rename.
_TITLE_RE = re.compile(
    r"""(?ix)             # case-insensitive, verbose
    \b(?:rename|re-?title|change|set|update|make|edit|call)\b
    [^\n]*?               # ... up to the title cue
    \b(?:report\s+title|report|title|it)\b
    \s*(?:to|:|=|as|->|→)\s*
    ["“”'’]?\s*(?P<title>.+?)\s*["“”'’]?\s*$
    """,
)

# "call it X" / "call the report X" — the new name follows directly, no connector.
_TITLE_CALL_RE = re.compile(
    r"""(?ix)
    \bcall\s+(?:it|the\s+report)\s+
    ["“”'’]?\s*(?P<title>.+?)\s*["“”'’]?\s*$
    """,
)


def _detect_title_rename(request: str) -> str | None:
    """Return the new report title for an unambiguous title-rename request, else None.

    Conservative: ignores requests that mention a SECTION (those are section-heading
    renames, handled by the classifier's ``set_section_title``).
    """
    text = (request or "").strip()
    if not text:
        return None
    low = text.lower()
    if "section" in low or "heading" in low or "subsection" in low:
        return None
    m = _TITLE_RE.search(text) or _TITLE_CALL_RE.search(text)
    if not m:
        return None
    title = (m.group("title") or "").strip().strip("\"'“”‘’").strip()
    # Reject degenerate captures ("change it", "set the report") and anything that
    # is clearly not a title (very long, or itself a different edit verb).
    if not title or len(title) > 120:
        return None
    return title


def apply_edit(
    report_json: dict[str, Any],
    request: str,
    emit=None,
    image_refs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Apply an edit to a saved report_json. Returns ``(report_json, summary)``.

    ``image_refs`` are the user's attached images (``[{"url","name"}]``) available
    to embed when the edit asks to attach an image.
    """
    emit = emit or (lambda *a, **k: None)

    forced_title = _detect_title_rename(request)
    # If the request is JUST a title rename, skip the LLM classifier entirely.
    title_only = forced_title is not None and len(request.strip()) <= 80
    if title_only:
        operations: list[Any] = [{"op": "set_title", "title": forced_title}]
        classification: dict[str, Any] = {"operations": operations, "summary": ""}
    else:
        classification = classify_edit(request, report_json)
        operations = list(classification.get("operations") or [])
        if forced_title is not None:
            # Deterministic title wins: prepend our set_title and drop any
            # title/section-title op the classifier emitted, so "title" can never
            # be misrouted into a section-heading rename.
            operations = [
                op for op in operations
                if not (isinstance(op, dict) and op.get("op") in ("set_title", "set_section_title"))
            ]
            operations.insert(0, {"op": "set_title", "title": forced_title})
    compute = report_json.setdefault("_compute", {})
    lines = compute.setdefault("lines", [])
    assembly = compute.setdefault("assembly", {})

    recompute_needed = False
    changes: list[str] = []

    for op in operations:
        if not isinstance(op, dict):
            continue
        kind = op.get("op")

        if kind == "set_title":
            new_title = str(op.get("title") or "").strip()
            if new_title:
                report_json.setdefault("meta", {})["title"] = new_title
                changes.append(f'title → "{new_title}"')

        elif kind == "set_section_title":
            section = _canonical_section(str(op.get("section") or ""))
            new_title = str(op.get("title") or "").strip()
            if section and new_title:
                report_json.setdefault("section_titles", {})[section] = new_title
                changes.append(f'{section.replace("_", " ")} heading → "{new_title}"')

        elif kind == "remove_section":
            section = _canonical_section(str(op.get("section") or ""))
            if section:
                hidden = report_json.setdefault("hidden_sections", [])
                if section not in hidden:
                    hidden.append(section)
                    changes.append(f"removed the {section.replace('_', ' ')} section")

        elif kind == "add_image":
            # Prefer the user's attached image(s); fall back to a URL pasted in the
            # request when nothing was attached, so "add this image: <url>" works too.
            refs = list(image_refs or [])
            op_url = str(op.get("url") or "").strip()
            if not refs and op_url.lower().startswith(("http://", "https://")):
                refs = [{"url": op_url, "name": op.get("caption") or ""}]
            added = _add_images(
                report_json, refs,
                op.get("position", "end"), op.get("caption"),
            )
            if added:
                changes.append(f"attached {added} image(s)")

        elif kind == "remove_image":
            existing_images = report_json.get("images") or []
            if existing_images:
                match = str(op.get("match") or "").strip().lower()
                if match:
                    kept = [
                        im for im in existing_images
                        if match not in str(im.get("caption") or "").lower()
                        and match not in str(im.get("url") or "").lower()
                    ]
                else:
                    kept = []  # no match given → remove all images
                removed = len(existing_images) - len(kept)
                if removed:
                    report_json["images"] = kept
                    changes.append(f"removed {removed} image(s)")

        elif kind == "rewrite_prose":
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

        elif kind == "set_stage_cost":
            target = str(op.get("target") or "").strip().lower()
            try:
                value = round(max(0.0, float(op["value_inr"])), 4)
            except (KeyError, TypeError, ValueError):
                continue
            non_quotable = compute.setdefault("non_quotable", {})
            if target in ("assembly", "pcb_assembly", "smt", "smt_assembly"):
                # Override the assembly per-unit directly. Keep the rate-card inputs
                # consistent so a later "change the per-joint rate" still behaves.
                assembly["per_unit_inr"] = value
                joints = int(assembly.get("total_joints") or 0)
                if joints > 0:
                    assembly["rate_per_joint_inr"] = round(value / joints, 6)
                recompute_needed = True
                changes.append(f"PCB assembly cost → ₹{value:,.2f}/unit")
            elif target in ("pcb_fabrication", "fabrication", "fab", "pcb_fab"):
                fab = compute.setdefault("fab", {})
                by_vol = fab.setdefault("fab_by_volume", {})
                vols = list(by_vol.keys()) or [compute.get("volume") or 1]
                for v in vols:
                    by_vol[v] = value
                recompute_needed = True
                changes.append(f"PCB fabrication cost → ₹{value:,.2f}/unit")
            elif target in ("firmware", "programming", "firmware_programming"):
                non_quotable.setdefault("firmware", {})["per_unit_inr"] = value
                recompute_needed = True
                changes.append(f"firmware/programming cost → ₹{value:,.2f}/unit")
            elif target in ("enclosure", "mechanical", "enclosure_mechanical"):
                non_quotable.setdefault("enclosure", {})["per_unit_inr"] = value
                recompute_needed = True
                changes.append(f"enclosure & mechanical cost → ₹{value:,.2f}/unit")
            elif target in ("final_assembly", "final", "test_pack", "assembly_test_pack"):
                non_quotable.setdefault("final_assembly", {})["per_unit_inr"] = value
                recompute_needed = True
                changes.append(f"final assembly/test/pack cost → ₹{value:,.2f}/unit")

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

    # Summary MUST reflect what actually changed, not the classifier's optimistic
    # self-description — otherwise the tool reports success ("Updated the title…")
    # for edits that applied nothing, and the assistant confidently relays a change
    # that never happened. Empty string => nothing was changed in this pass (the
    # caller still checks for image attachments before declaring failure).
    summary = ", ".join(changes) if changes else ""
    return report_json, summary
