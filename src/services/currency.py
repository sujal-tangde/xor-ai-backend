"""USD → INR conversion for should-cost reports (spot rate via Tavily)."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.core.config import REPORT_USD_INR_FALLBACK

logger = logging.getLogger(__name__)

_RATE_MIN = 50.0
_RATE_MAX = 150.0

_RATE_PATTERNS = (
    re.compile(
        r"1\s*(?:US\$|\$|USD)\s*(?:=|is|:|to|in)\s*([\d,]+(?:\.\d+)?)\s*(?:INR|₹|Rs\.?)",
        re.I,
    ),
    re.compile(r"(?:US\$|\$|USD)\s*/\s*INR\s*[=:]\s*([\d,]+(?:\.\d+)?)", re.I),
    re.compile(
        r"([\d]{2,3}(?:\.\d{1,4})?)\s*(?:INR|₹|Rs\.?)\s*(?:per|for|to)\s*(?:1\s*)?(?:US\$|\$|USD)",
        re.I,
    ),
    re.compile(
        r"(?:exchange\s*rate|conversion)[^\d]{0,40}([\d]{2,3}(?:\.\d{1,4})?)",
        re.I,
    ),
)

_SOURCE_LABELS = {
    "digikey": "DigiKey",
    "web": "Web estimate (low confidence)",
    "unresolved": "Unresolved",
}

_TABLE_RE = re.compile(r"\n\|[^\n]+\|\n\|[-:| ]+\|(?:\n\|[^\n]+\|)*")


def _parse_rate_from_text(text: str) -> float | None:
    if not text:
        return None
    for pattern in _RATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if _RATE_MIN <= value <= _RATE_MAX:
            return value
    return None


def _tavily_search(query: str) -> str:
    try:
        from src.agent.tools.tavily_search_tool import get_tavily_search_tool

        tool = get_tavily_search_tool()
        if tool is None:
            return ""
        result = tool.invoke({"query": query})
        if isinstance(result, dict):
            parts: list[str] = []
            answer = result.get("answer")
            if answer:
                parts.append(str(answer))
            for item in result.get("results") or []:
                if isinstance(item, dict):
                    for key in ("content", "snippet", "raw_content"):
                        if item.get(key):
                            parts.append(str(item[key]))
            return "\n".join(parts)
        return str(result)
    except Exception:
        logger.warning("Tavily FX lookup failed", exc_info=True)
        return ""


def fetch_usd_inr_rate() -> tuple[float, str]:
    """Return (rate, source). Uses Tavily when configured, else env fallback."""
    text = _tavily_search("current USD to INR exchange rate today 1 USD in Indian rupees")
    rate = _parse_rate_from_text(text)
    if rate is not None:
        logger.info("USD/INR rate from Tavily: %.4f", rate)
        return round(rate, 4), "tavily"
    logger.warning(
        "Could not parse USD/INR from Tavily; using fallback %.4f",
        REPORT_USD_INR_FALLBACK,
    )
    return float(REPORT_USD_INR_FALLBACK), "fallback"


def _to_inr(amount: float | int | None, rate: float, *, places: int = 2) -> float | None:
    if amount is None:
        return None
    return round(float(amount) * rate, places)


def _bom_subtotal_inr(bom: dict[str, Any]) -> float | None:
    for key in ("per_unit_subtotal", "per_unit_subtotal_inr"):
        if bom.get(key) is not None:
            return float(bom[key])
    return None


def _block_per_unit_inr(block: dict[str, Any] | None) -> float | None:
    if not isinstance(block, dict):
        return None
    for key in ("per_unit", "per_unit_inr"):
        if block.get(key) is not None:
            return float(block[key])
    return None


def _line_unit_price_inr(line: dict[str, Any]) -> float | None:
    for key in ("unit_price", "unit_price_inr"):
        if line.get(key) is not None:
            return float(line[key])
    return None


def _line_cost_inr(line: dict[str, Any]) -> float | None:
    for key in ("line_cost", "line_cost_inr"):
        if line.get(key) is not None:
            return float(line[key])
    return None


def _total_per_unit_inr(total: dict[str, Any] | None) -> float | None:
    if not isinstance(total, dict):
        return None
    for key in ("per_unit", "per_unit_inr"):
        if total.get(key) is not None:
            return float(total[key])
    return None


def format_inr(amount: float | int | None) -> str:
    if amount is None:
        return "—"
    return f"₹{float(amount):,.2f}"


def convert_costs_to_inr(costs: dict[str, Any], rate: float, *, source: str) -> dict[str, Any]:
    """Return an INR-only costs dict for report composition (no USD monetary fields)."""
    bom_in = costs.get("bom") if isinstance(costs.get("bom"), dict) else {}
    lines_out: list[dict[str, Any]] = []
    for line in bom_in.get("lines") or []:
        if not isinstance(line, dict):
            continue
        lines_out.append(
            {
                "ref_des": line.get("ref_des") or "",
                "label": line.get("label") or "Component",
                "mpn": line.get("mpn"),
                "qty_per_unit": line.get("qty_per_unit") or 1,
                "unit_price": _to_inr(line.get("unit_price"), rate),
                "line_cost": _to_inr(line.get("line_cost"), rate),
                "source": line.get("source") or "unresolved",
                "status": line.get("status"),
                "note": line.get("note") or "",
            }
        )

    bom_out: dict[str, Any] = {
        "lines": lines_out,
        "per_unit_subtotal": _to_inr(bom_in.get("per_unit_subtotal_usd"), rate),
        "notes": list(bom_in.get("notes") or []),
    }

    def _block_inr(block: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(block, dict):
            return None
        per_unit = _to_inr(block.get("per_unit_usd"), rate)
        if per_unit is None:
            return None
        params_in = block.get("parameters") if isinstance(block.get("parameters"), dict) else {}
        params_out = dict(params_in)
        for key, value in list(params_in.items()):
            if key.endswith("_usd") and isinstance(value, (int, float)):
                params_out[key.replace("_usd", "_inr")] = _to_inr(value, rate, places=4)
                params_out.pop(key, None)
        return {
            "per_unit": per_unit,
            "basis": block.get("basis"),
            "parameters": params_out,
        }

    total_in = costs.get("total") if isinstance(costs.get("total"), dict) else {}
    total_out = None
    if total_in.get("per_unit_usd") is not None:
        total_out = {
            "per_unit": _to_inr(total_in.get("per_unit_usd"), rate),
            "missing_blocks": list(total_in.get("missing_blocks") or []),
        }

    return {
        "volume": costs.get("volume"),
        "currency": "INR",
        "fx": {
            "usd_inr": rate,
            "source": source,
            "note": "DigiKey list prices are USD; all report figures below are converted to INR.",
        },
        "bom": bom_out,
        "pcb_fab": _block_inr(costs.get("pcb_fab")),
        "assembly": _block_inr(costs.get("assembly")),
        "enclosure": _block_inr(costs.get("enclosure")),
        "total": total_out,
    }


def build_executive_summary_table(costs: dict[str, Any]) -> str:
    rows: list[tuple[str, float | None]] = []
    bom = costs.get("bom") if isinstance(costs.get("bom"), dict) else {}
    subtotal = _bom_subtotal_inr(bom)
    if subtotal is not None:
        rows.append(("BOM per-unit subtotal", subtotal))
    for key, label in (
        ("pcb_fab", "PCB fab"),
        ("assembly", "Assembly"),
        ("enclosure", "Enclosure"),
    ):
        block = costs.get(key)
        per_unit = _block_per_unit_inr(block if isinstance(block, dict) else None)
        if per_unit is not None:
            rows.append((label, per_unit))
    total = costs.get("total") if isinstance(costs.get("total"), dict) else {}
    total_amt = _total_per_unit_inr(total)

    lines = [
        "| Cost Block | Amount (INR) |",
        "|---|---:|",
    ]
    for label, amount in rows:
        lines.append(f"| {label} | {format_inr(amount)} |")
    if total_amt is not None:
        lines.append(f"| **Total should-cost** | **{format_inr(total_amt)}** |")
    return "\n".join(lines)


def build_bom_table(costs: dict[str, Any]) -> str:
    bom = costs.get("bom") if isinstance(costs.get("bom"), dict) else {}
    lines_data = bom.get("lines") or []
    if not lines_data:
        return ""

    lines = [
        "| Ref Des | Component | MPN | Qty/Unit | Unit Price (INR) | Line Cost (INR) | Source | Notes |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for line in lines_data:
        if not isinstance(line, dict):
            continue
        source = _SOURCE_LABELS.get(str(line.get("source") or "").lower(), line.get("source") or "")
        mpn = line.get("mpn") or "—"
        note = (line.get("note") or "").replace("|", "/")
        lines.append(
            "| {ref} | {label} | {mpn} | {qty} | {unit} | {cost} | {source} | {note} |".format(
                ref=line.get("ref_des") or "—",
                label=line.get("label") or "Component",
                mpn=mpn,
                qty=line.get("qty_per_unit") or 1,
                unit=format_inr(_line_unit_price_inr(line)),
                cost=format_inr(_line_cost_inr(line)),
                source=source,
                note=note or "—",
            )
        )
    subtotal = _bom_subtotal_inr(bom)
    if subtotal is not None:
        lines.append(
            f"| | | | | **Subtotal** | **{format_inr(subtotal)}** | | |"
        )
    return "\n".join(lines)


def build_precomputed_cost_blocks(costs: dict[str, Any]) -> dict[str, str]:
    """Markdown fragments the LLM must not override."""
    bom = costs.get("bom") if isinstance(costs.get("bom"), dict) else {}
    bom_notes = bom.get("notes") or []
    notes_md = ""
    if bom_notes:
        notes_md = "\n".join(f"- {note}" for note in bom_notes)

    fab = costs.get("pcb_fab")
    assembly = costs.get("assembly")
    enclosure = costs.get("enclosure")
    total = costs.get("total") if isinstance(costs.get("total"), dict) else {}

    def _block_text(block: dict[str, Any] | None, label: str) -> str:
        per_unit = _block_per_unit_inr(block)
        if per_unit is None:
            return f"Not yet determined. Data required for {label.lower()} costing."
        params = (block or {}).get("parameters") or {}
        param_lines = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in params.items())
        return (
            f"**Per-unit estimate (INR): {format_inr(per_unit)}**\n\n"
            f"{param_lines}"
        ).strip()

    total_text = ""
    total_amt = _total_per_unit_inr(total)
    if total_amt is not None:
        total_text = f"**Total should-cost per unit (INR): {format_inr(total_amt)}**"
        missing = total.get("missing_blocks") or []
        if missing:
            total_text += f"\n\nMissing blocks: {', '.join(missing)}."

    return {
        "executive_table": build_executive_summary_table(costs),
        "bom_table": build_bom_table(costs),
        "bom_notes": notes_md,
        "fab_text": _block_text(fab, "PCB fab"),
        "assembly_text": _block_text(assembly, "assembly"),
        "enclosure_text": _block_text(enclosure, "enclosure"),
        "total_text": total_text,
    }


def _section_body(markdown: str, heading_prefix: str) -> tuple[str, int, int] | None:
    match = re.search(
        rf"^## {re.escape(heading_prefix)}[^\n]*$",
        markdown,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return None
    start = match.end()
    next_heading = re.search(r"^## ", markdown[start:], flags=re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(markdown)
    return markdown[start:end], start, end


def _replace_first_table(body: str, table: str) -> str:
    if not table:
        return body
    if _TABLE_RE.search(body):
        return _TABLE_RE.sub("\n\n" + table, body, count=1)
    return body.rstrip() + "\n\n" + table + "\n"


def enforce_inr_markdown(markdown: str, costs: dict[str, Any]) -> str:
    """Inject programmatic INR tables and scrub any USD labels the LLM emitted."""
    blocks = build_precomputed_cost_blocks(costs)
    md = markdown or ""

    replacements = {
        "Amount (USD)": "Amount (INR)",
        "Unit Price (USD)": "Unit Price (INR)",
        "Line Cost (USD)": "Line Cost (INR)",
        "(USD)": "(INR)",
    }
    for old, new in replacements.items():
        md = md.replace(old, new)
    md = re.sub(r"\bUSD\b", "INR", md)

    exec_section = _section_body(md, "Executive Summary")
    if exec_section and blocks["executive_table"]:
        body, start, end = exec_section
        md = md[:start] + _replace_first_table(body, blocks["executive_table"]) + md[end:]

    bom_section = _section_body(md, "BOM")
    if bom_section and blocks["bom_table"]:
        body, start, end = bom_section
        body = _replace_first_table(body, blocks["bom_table"])
        if blocks["bom_notes"] and blocks["bom_notes"] not in body:
            body = body.rstrip() + "\n\n" + blocks["bom_notes"] + "\n"
        md = md[:start] + body + md[end:]

    for heading, key in (
        ("Fab Costing", "fab_text"),
        ("Assembly", "assembly_text"),
        ("Enclosure and Total Should-Cost", "enclosure_text"),
    ):
        section = _section_body(md, heading)
        if not section or not blocks[key]:
            continue
        body, start, end = section
        if blocks[key] not in body:
            body = blocks[key] + "\n\n" + body.lstrip()
        md = md[:start] + body + md[end:]

    total_section = _section_body(md, "Enclosure and Total Should-Cost")
    if total_section and blocks["total_text"]:
        body, start, end = total_section
        if blocks["total_text"] not in body:
            body = body.rstrip() + "\n\n" + blocks["total_text"] + "\n"
        md = md[:start] + body + md[end:]

    return md.strip()


def costs_for_display(costs: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize stored costs to INR for preview/PDF, including legacy USD-only rows."""
    if not isinstance(costs, dict):
        return {}
    if costs.get("currency") == "INR":
        return costs
    bom = costs.get("bom") if isinstance(costs.get("bom"), dict) else {}
    if bom.get("per_unit_subtotal_inr") or bom.get("per_unit_subtotal"):
        return {**costs, "currency": "INR"}
    if bom.get("per_unit_subtotal_usd") or (
        isinstance(costs.get("total"), dict) and costs["total"].get("per_unit_usd") is not None
    ):
        fx = costs.get("fx") if isinstance(costs.get("fx"), dict) else {}
        rate = fx.get("usd_inr")
        source = fx.get("source") or "fallback"
        if not rate:
            rate, source = fetch_usd_inr_rate()
        return convert_costs_to_inr(costs, float(rate), source=str(source))
    return costs


def report_subtitle(volume: int | None, costs: dict[str, Any] | None) -> str | None:
    parts: list[str] = []
    if volume:
        parts.append(f"Estimated production volume: {volume:,} units")
    fx = (costs or {}).get("fx") or {}
    rate = fx.get("usd_inr")
    if rate:
        parts.append(f"All costs in INR (1 USD = {float(rate):.2f} INR)")
    return " · ".join(parts) if parts else None
