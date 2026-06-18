"""The ``report_generation`` agent tool.

Generates a professional should-cost PDF report for a project. It:

1. Pulls the project knowledge base (theory + structured context).
2. Uses HILT (``interrupt()``) to ask the user a SMALL number of clarifying
   questions when the data is insufficient — the user can answer, skip, or
   upload a file. Answered questions are persisted and folded back into the
   insight pipeline in the background.
3. Resolves + prices the BOM via the DigiKey API.
4. Optionally pulls extra product/market context from Tavily web search.
5. Composes the report markdown, renders it to PDF, stores it, and streams the
   result to the frontend (right-side preview + download).

Progress is streamed to the UI via the LangGraph custom stream channel; the
question round is delivered via the graph interrupt mechanism.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent.tools.validation import invalid_project_id_message, is_uuid
from src.core.config import REPORT_DEFAULT_VOLUME
from src.services import currency, report_builder, reports
from src.services.projects_service import get_project_knowledge_base, get_project_name
from src.services.queue import enqueue_qa_insight

logger = logging.getLogger(__name__)

TOOL_NAME = "report_generation"
TOOL_LABEL = "report generation"

# The report structure/prompt. {volume} is substituted at composition time.
REPORT_PROMPT = """You are generating a professional electronics should-cost analysis report in clean GitHub-flavored Markdown.

Use EXACTLY these top-level sections (## headings), in this order:
1. Executive Summary
2. Product Overview
3. Architecture Analysis
4. BOM
5. Fab Costing
6. Assembly
7. Enclosure and Total Should-Cost for {volume} units
8. Market Context
9. Design Observations
10. Citations

Section requirements:
- Executive Summary: 2-4 sentences on what the product is, then paste the PRE-COMPUTED executive summary table verbatim (see below). Do NOT create your own cost table.
- BOM: Paste the PRE-COMPUTED BOM table verbatim, then list `costs.bom.notes` if any. Do NOT build your own BOM pricing table.
- Fab Costing / Assembly / Enclosure: use the PRE-COMPUTED INR figures provided below; add brief narrative citing `dimensions`, `materials`, `enclosure` facts where relevant.
- Total should-cost: use the PRE-COMPUTED total figure provided below.

Rules:
- NEVER use USD. All monetary values are INR only. The `costs` object has `currency: "INR"`.
- Use ONLY the pre-computed tables and INR numbers provided — never invent figures.
- Format money with the ₹ symbol (e.g. ₹300.48).
- In Citations, cite DigiKey, the USD→INR rate from `costs.fx`, and uploaded documents.
- Note assumptions honestly. BOM DigiKey prices were converted using `costs.fx.usd_inr`."""


def _writer():
    """Return the LangGraph custom-stream writer, or a noop if unavailable."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # pragma: no cover - no streaming context
        return lambda _payload: None


def _emit(stage: str, message: str) -> None:
    try:
        _writer()({"kind": "report_progress", "stage": stage, "message": message})
    except Exception:  # pragma: no cover - streaming best-effort
        pass


def _cfg(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable", {}) or {}


def _parse_volume(text: str) -> int | None:
    """Pull a production volume from free text like '10k units' / 'volume 5000'."""
    if not text:
        return None
    match = re.search(r"(\d[\d,\.]*)\s*([kK])?\s*(?:units|pcs|pieces|qty|volume)?", text)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    if match.group(2):
        value *= 1000
    return int(value) if value > 0 else None


def _normalize_answer(raw: Any) -> dict[str, Any]:
    """Coerce one resumed answer into {answer, file_ids, status}."""
    if isinstance(raw, dict):
        answer = (raw.get("answer") or "").strip() if raw.get("answer") else ""
        file_ids = raw.get("file_ids") or None
        status = raw.get("status") or ("skipped" if not answer and not file_ids else "answered")
        return {"answer": answer, "file_ids": file_ids, "status": status}
    if isinstance(raw, str):
        text = raw.strip()
        return {"answer": text, "file_ids": None, "status": "answered" if text else "skipped"}
    return {"answer": "", "file_ids": None, "status": "skipped"}


def _tavily_context(query: str) -> str | None:
    """Best-effort web context for the report (returns a compact string)."""
    try:
        from src.agent.tools.tavily_search_tool import get_tavily_search_tool

        tool_obj = get_tavily_search_tool()
        if tool_obj is None:
            return None
        result = tool_obj.invoke({"query": query})
        if isinstance(result, dict):
            answer = result.get("answer")
            if answer:
                return str(answer)
            return json.dumps(result, ensure_ascii=False)[:4000]
        return str(result)[:4000]
    except Exception:
        logger.warning("Tavily context fetch failed", exc_info=True)
        return None


def _generate_modification(
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    modification_request: str,
) -> str:
    """Apply a text modification to the latest report for this conversation."""
    base = reports.latest_report_for_conversation(conversation_id) if conversation_id else None
    if base is None:
        return (
            "There is no existing report in this conversation to modify yet. "
            "Ask me to generate a report first."
        )
    _emit("modify", "Applying your requested changes to the report…")
    from src.services.llm_analysis import invoke_llm

    prompt = (
        "Revise the following should-cost report markdown per the user's request. "
        "Keep the same section structure and all factual figures unless the request "
        "explicitly changes them. Never invent new prices. All amounts are in INR — "
        "keep them in INR unless the user explicitly asks otherwise. Output ONLY the revised "
        "markdown — no code fences.\n\nUSER REQUEST:\n"
        + modification_request
        + "\n\nCURRENT REPORT MARKDOWN:\n"
        + (base.get("markdown") or "")
    )
    new_markdown = reports.normalize_report_markdown(
        invoke_llm([{"role": "user", "content": prompt}], max_tokens=8192).strip()
    )
    stored_costs = base.get("costs")
    if isinstance(stored_costs, dict) and stored_costs.get("currency") == "INR":
        new_markdown = currency.enforce_inr_markdown(new_markdown, stored_costs)
    title = base.get("title") or "Should-Cost Report"
    volume = base.get("volume")
    subtitle = currency.report_subtitle(volume, base.get("costs"))
    pdf_bytes = reports.render_pdf(new_markdown, title=title, subtitle=subtitle)
    updated = reports.update_report(
        base["id"], markdown_text=new_markdown, pdf_bytes=pdf_bytes
    )
    report_id = (updated or base)["id"]
    _writer()(
        {
            "kind": "report_ready",
            "report_id": report_id,
            "title": title,
            "markdown": new_markdown,
            "volume": volume,
            "fx_rate": (base.get("costs") or {}).get("fx", {}).get("usd_inr"),
        }
    )
    return (
        f'Updated the report "{title}". The revised PDF is shown on the right and is '
        "ready to download. Let me know if you'd like any other changes."
    )


@tool(TOOL_NAME)
def report_generation_tool(
    request: str,
    config: RunnableConfig = None,  # type: ignore[assignment]
    modification_request: str | None = None,
) -> str:
    """Generate (or modify) a professional should-cost PDF report for the product.

    Call this whenever the user asks to generate a report, a should-cost report,
    a BOM cost report, a cost breakdown PDF, or anything similar. It reads the
    project's accumulated knowledge base, asks the user a few clarifying
    questions only if needed, prices the BOM via DigiKey, renders a PDF, and
    streams it to the user with a download option.

    Args:
        request: The user's request, verbatim (used to infer intent/volume).
        modification_request: If the user is asking to CHANGE an existing report
            that was already generated in this conversation, put their change
            request here; the existing report is revised instead of rebuilt.
    """
    cfg = _cfg(config)
    project_id = cfg.get("project_id")
    user_id = cfg.get("user_id")
    conversation_id = cfg.get("conversation_id")

    if not project_id:
        return "No project is associated with this conversation, so I can't build a report."
    if not is_uuid(str(project_id)):
        return invalid_project_id_message(str(project_id))

    # Modification path: revise the existing report instead of rebuilding.
    if modification_request and modification_request.strip():
        return _generate_modification(
            str(project_id), conversation_id, user_id, modification_request.strip()
        )

    _emit("start", "Starting report generation…")
    _emit("context", "Reading what we know about the product…")

    kb = get_project_knowledge_base(str(project_id))
    if not kb or (not (kb.get("theory_context") or "").strip() and not kb.get("structured_context")):
        return (
            "I don't have enough analyzed information about this product yet to build a "
            "report. Please upload photos of the PCB and product (and any datasheets) "
            "so I can analyze them first, then ask me to generate the report."
        )

    theory = (kb.get("theory_context") or "").strip()
    structured = kb.get("structured_context")
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            structured = None
    if not isinstance(structured, dict):
        structured = {}

    project_name = get_project_name(str(project_id)) or "this product"

    # ---- HILT: ask only the minimal missing questions -------------------- #
    _emit("assess", "Checking whether I need any details from you…")
    questions = report_builder.assess_missing(theory, structured, request)

    answers_by_id: dict[str, dict[str, Any]] = {}
    if questions:
        from langgraph.types import interrupt

        _emit("questions", f"I have {len(questions)} quick question(s) for you.")
        resumed = interrupt({"type": "report_questions", "questions": questions})
        # `resumed` is whatever the websocket layer passed to Command(resume=...).
        raw_answers = resumed.get("answers") if isinstance(resumed, dict) else resumed
        if isinstance(raw_answers, dict):
            answers_by_id = {k: _normalize_answer(v) for k, v in raw_answers.items()}
        elif isinstance(raw_answers, list):
            for q, a in zip(questions, raw_answers):
                answers_by_id[q["id"]] = _normalize_answer(a)

        # Persist Q&A and fold answers back into the insight pipeline.
        qa_pairs: list[dict[str, Any]] = []
        for q in questions:
            ans = answers_by_id.get(q["id"], {"answer": "", "file_ids": None, "status": "skipped"})
            try:
                reports.save_question_answer(
                    str(project_id),
                    conversation_id,
                    user_id,
                    question=q["prompt"],
                    kind=q["kind"],
                    answer=ans["answer"] or None,
                    file_ids=ans["file_ids"],
                    status=ans["status"],
                )
            except Exception:
                logger.warning("Failed to persist report question", exc_info=True)
            qa_pairs.append(
                {
                    "question": q["prompt"],
                    "answer": ans["answer"],
                    "file_ids": ans["file_ids"],
                }
            )
        try:
            enqueue_qa_insight(str(project_id), user_id, qa_pairs)
        except Exception:
            logger.warning("Failed to enqueue Q&A insight ingestion", exc_info=True)

    # ---- Determine production volume ------------------------------------- #
    volume = _parse_volume(request)
    if volume is None:
        for qid, ans in answers_by_id.items():
            if "volume" in qid.lower() or "qty" in qid.lower() or "unit" in qid.lower():
                volume = _parse_volume(ans["answer"])
                if volume:
                    break
    assumed_volume = volume is None
    volume = volume or REPORT_DEFAULT_VOLUME

    # ---- Resolve + price the BOM via DigiKey ---------------------------- #
    _emit("digikey", "Resolving and pricing components via DigiKey…")
    bom = report_builder.resolve_and_price_bom(structured, volume, progress=lambda p: _emit(p.get("stage", "digikey"), p.get("message", "")))
    costs = report_builder.estimate_costs(structured, bom, volume)
    if assumed_volume:
        bom.setdefault("notes", []).append(
            f"Production volume not specified — assumed {volume:,} units."
        )

    # ---- USD → INR conversion (spot rate via Tavily) ------------------- #
    _emit("fx", "Fetching USD → INR exchange rate…")
    usd_inr, fx_source = currency.fetch_usd_inr_rate()
    costs = currency.convert_costs_to_inr(costs, usd_inr, source=fx_source)

    # ---- Optional web context ------------------------------------------- #
    web_context = None
    product_label = ((structured.get("product") or {}).get("name")) or project_name
    if product_label and product_label != "this product":
        _emit("web", "Pulling extra market/product context from the web…")
        web_context = _tavily_context(
            f"{product_label} electronics product specifications and market price"
        )

    # ---- Compose + render ----------------------------------------------- #
    _emit("compose", "Writing the report…")
    precomputed = currency.build_precomputed_cost_blocks(costs)
    markdown_text = reports.normalize_report_markdown(
        report_builder.compose_report_markdown(
            REPORT_PROMPT,
            theory,
            structured,
            costs,
            volume,
            web_context,
            precomputed=precomputed,
        )
    )
    markdown_text = currency.enforce_inr_markdown(markdown_text, costs)

    title = f"Should-Cost Report — {product_label}" if product_label else "Should-Cost Report"
    subtitle = currency.report_subtitle(volume, costs)
    _emit("pdf", "Rendering the PDF…")
    pdf_bytes = reports.render_pdf(markdown_text, title=title, subtitle=subtitle)

    record = reports.create_report(
        str(project_id),
        conversation_id,
        user_id,
        title=title,
        volume=volume,
        markdown_text=markdown_text,
        costs=costs,
        pdf_bytes=pdf_bytes,
    )
    report_id = record["id"] if record else None

    _writer()(
        {
            "kind": "report_ready",
            "report_id": report_id,
            "title": title,
            "markdown": markdown_text,
            "volume": volume,
            "fx_rate": costs.get("fx", {}).get("usd_inr"),
        }
    )

    summary = (
        f'I generated the should-cost report "{title}" (assumed {volume:,} units). '
        "It's shown on the right and ready to download. "
        "Would you like any modifications — different volume, more detail in a section, "
        "or anything else?"
    )
    return summary
