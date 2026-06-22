"""The ``report_generation`` agent tool.

Generates (or edits) a professional should-cost PDF report for a project. It:

1.  Reads the project knowledge base (vision-derived components/pcb/enclosure/…).
2.  Uses HILT (``interrupt()``) to ask a SMALL number of clarifying questions
    when material data is missing; answers are persisted + folded back into KB.
3.  Resolves MPNs, then in parallel prices the BOM (Mouser), quotes the PCB
    (PCBWay), estimates non-quotable blocks, and gathers market context (Tavily).
4.  Applies FX (PCBWay only), customs duty, the assembly model and the volume
    curve, aggregates one structured JSON, fills the locked HTML template,
    renders a PDF, uploads it to the ``reports`` bucket, and streams the result.

Every stage emits a ``report_progress`` event (start → done/warning) and degrades
gracefully — the report always generates; any gap is disclosed in plain language.

The governing principle: **the LLM narrates, the code computes.** All numbers are
computed in code with a ``Live``/``Est`` source tag; nothing is hallucinated.
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
from src.services import report_builder, report_edit, report_pipeline, report_template, reports
from src.services.failure_log import record_failure
from src.services.projects_service import get_project_knowledge_base, get_project_name
from src.services.queue import enqueue_qa_insight

logger = logging.getLogger(__name__)

TOOL_NAME = "report_generation"
TOOL_LABEL = "report generation"

# Cumulative (start, end) progress fraction per stage for the overall bar. The
# tool computes the emitted ``progress`` from this table so the pipeline only has
# to name the stage + status (+ optional meta for interpolation within a stage).
_STAGE_RANGE: dict[str, tuple[float, float]] = {
    "reading_kb": (0.00, 0.04),
    "hilt": (0.04, 0.08),
    "resolving_mpns": (0.08, 0.20),
    "pricing": (0.20, 0.45),
    "fab_quote": (0.45, 0.50),
    "non_quotable": (0.50, 0.55),
    "market_context": (0.55, 0.62),
    "fx": (0.62, 0.66),
    "duty": (0.66, 0.74),
    "assembly": (0.74, 0.80),
    "volume_curve": (0.80, 0.88),
    "rendering": (0.88, 0.94),
    "storing": (0.94, 0.99),
    "report_ready": (1.0, 1.0),
}


def _writer():
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # pragma: no cover - no streaming context
        return lambda _payload: None


def _make_emit():
    """Capture the stream writer once (in the node context) and return an emit fn.

    The returned ``emit`` is safe to call from worker threads (it closes over the
    captured writer rather than re-reading a contextvar that wouldn't propagate).
    """
    writer = _writer()

    def emit(stage: str, status: str, message: str, meta: dict[str, Any] | None = None) -> None:
        start, end = _STAGE_RANGE.get(stage, (0.0, 1.0))
        if status == "in_progress" and meta and meta.get("total"):
            try:
                frac = float(meta["current"]) / float(meta["total"])
            except (TypeError, ValueError, ZeroDivisionError):
                frac = 0.0
            progress = round(start + (end - start) * max(0.0, min(1.0, frac)), 3)
        elif status in ("started",):
            progress = round(start, 3)
        else:  # done | warning | error
            progress = round(end, 3)
        try:
            writer({
                "kind": "report_progress",
                "stage": stage,
                "status": status,
                "message": message,
                "progress": progress,
                "meta": meta or {},
            })
        except Exception:  # pragma: no cover - streaming best-effort
            pass

    return emit, writer


def _cfg(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable", {}) or {}


def _parse_volume(text: str) -> int | None:
    """Pull a production volume ONLY when the user states it explicitly.

    Requires an explicit quantity context — a unit word, a 'k' suffix, or a
    'volume/qty of N' phrase — so a stray number in the message (e.g. "9") is
    never mistaken for a production volume. Otherwise returns None (the caller
    defaults to a per-unit basis).
    """
    if not text:
        return None
    patterns = (
        r"(\d[\d,\.]*)\s*([kK])\b",  # "10k"
        r"(\d[\d,\.]*)\s*(?:units?|pcs|pieces|qty|quantity)\b",  # "5000 units"
        r"(?:volume|qty|quantity)\s*(?:of|=|:|is)?\s*(\d[\d,\.]*)\s*([kK])?",  # "volume of 5000"
        r"\bfor\s+(\d[\d,\.]*)\s*([kK])?\s*(?:units?|pcs|pieces)\b",  # "for 1000 units"
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except (ValueError, IndexError):
            continue
        groups = match.groups()
        if len(groups) > 1 and groups[1] and groups[1].lower() == "k":
            value *= 1000
        if value > 0:
            return int(value)
    return None


def _normalize_answer(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        answer = (raw.get("answer") or "").strip() if raw.get("answer") else ""
        file_ids = raw.get("file_ids") or None
        status = raw.get("status") or ("skipped" if not answer and not file_ids else "answered")
        return {"answer": answer, "file_ids": file_ids, "status": status}
    if isinstance(raw, str):
        text = raw.strip()
        return {"answer": text, "file_ids": None, "status": "answered" if text else "skipped"}
    return {"answer": "", "file_ids": None, "status": "skipped"}


def _structured_from_kb(kb: dict[str, Any]) -> dict[str, Any]:
    structured = kb.get("structured_context")
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            structured = None
    return structured if isinstance(structured, dict) else {}


def _render_store_stream(
    *,
    report_json: dict[str, Any],
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    title: str,
    volume: int,
    emit,
    writer,
    existing_report_id: str | None = None,
) -> str | None:
    """Render HTML→PDF, upload to the bucket, persist JSON, and stream report_ready."""
    emit("rendering", "in_progress", "Building the report document…")
    # HTML backs the downloadable PDF (full-fidelity CSS); markdown backs the
    # in-app preview panel (no CSS quirks).
    html = report_template.render_html(report_json)
    markdown_text = report_template.render_markdown(report_json)

    pdf_url = None
    report_id = existing_report_id
    try:
        pdf_bytes = reports.render_pdf_from_html(html)
    except Exception as exc:
        logger.warning("PDF render failed", exc_info=True)
        record_failure(
            "rendering", title or "report",
            "Could not render the report PDF — the report was saved without a PDF",
            error=exc, context={"project_id": project_id, "report_id": existing_report_id},
        )
        pdf_bytes = None

    emit("storing", "started", "Saving your report…")
    try:
        if existing_report_id:
            pdf_path = None
            if pdf_bytes is not None:
                pdf_path, pdf_url = reports.upload_report_pdf(existing_report_id, pdf_bytes)
            reports.update_report(
                existing_report_id,
                title=title,
                volume=volume,
                report_json=report_json,
                html=html,
                markdown_text=markdown_text,
                pdf_path=pdf_path,
                pdf_url=pdf_url,
            )
            report_id = existing_report_id
        else:
            record = reports.create_report(
                project_id, conversation_id, user_id,
                title=title, volume=volume, report_json=report_json,
                html=html, markdown_text=markdown_text, pdf_path=None, pdf_url=None,
            )
            report_id = record["id"] if record else None
            if report_id and pdf_bytes is not None:
                pdf_path, pdf_url = reports.upload_report_pdf(report_id, pdf_bytes)
                reports.update_report(report_id, pdf_path=pdf_path, pdf_url=pdf_url)
        emit("storing", "done", "Report saved.")
    except Exception as exc:
        logger.warning("Report storage failed", exc_info=True)
        record_failure(
            "storing", title or "report",
            "Could not finish saving the report (upload/persist failed)",
            error=exc, context={"project_id": project_id, "report_id": report_id},
        )
        emit("storing", "error", "We couldn't finish saving the report — please try again.")
        return report_id

    emit("report_ready", "done", "Report ready.")
    writer({
        "kind": "report_ready",
        "report_id": report_id,
        "title": title,
        "markdown": markdown_text,
        "pdf_url": pdf_url,
        "volume": volume,
        "fx_rate": (report_json.get("fx") or {}).get("rate"),
    })
    return report_id


def _generate_modification(
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    modification_request: str,
) -> str:
    """Edit the latest report for this conversation, operating on its saved JSON."""
    emit, writer = _make_emit()
    base = reports.latest_report_for_conversation(conversation_id) if conversation_id else None
    if base is None or not base.get("report_json"):
        return (
            "There's no existing report in this conversation to modify yet. "
            "Ask me to generate a report first, then I can tweak it."
        )

    report_json = base["report_json"]
    if isinstance(report_json, str):
        try:
            report_json = json.loads(report_json)
        except json.JSONDecodeError:
            return "The saved report couldn't be read for editing — please regenerate it."

    emit("rendering", "started", "Applying your requested changes…")
    report_json, summary = report_edit.apply_edit(report_json, modification_request, emit)

    title = base.get("title") or (report_json.get("meta") or {}).get("title") or "Should-Cost Report"
    volume = (report_json.get("meta") or {}).get("volume") or base.get("volume") or REPORT_DEFAULT_VOLUME

    _render_store_stream(
        report_json=report_json,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        title=title,
        volume=int(volume),
        emit=emit,
        writer=writer,
        existing_report_id=base["id"],
    )
    return (
        f'Updated the report — {summary}. The revised version is shown on the right and the PDF is '
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
    a BOM cost report, a cost breakdown PDF, or similar. It reads the project's
    knowledge base, asks a few clarifying questions only if needed, prices the BOM
    (Mouser), quotes the PCB (PCBWay), applies duty and assembly costs across a
    volume curve, renders a PDF in the locked report format, and streams it.

    Args:
        request: The user's request, verbatim (used to infer intent/volume).
        modification_request: If the user is asking to CHANGE a report already
            generated in this conversation (e.g. "change the target volume to
            5000", "remove the LED line", "shorten the executive summary"), put
            their change request here; the existing report is edited in place.
    """
    cfg = _cfg(config)
    project_id = cfg.get("project_id")
    user_id = cfg.get("user_id")
    conversation_id = cfg.get("conversation_id")

    if not project_id:
        return "No project is associated with this conversation, so I can't build a report."
    if not is_uuid(str(project_id)):
        return invalid_project_id_message(str(project_id))

    # ---- Edit path: revise the existing report instead of rebuilding. ------- #
    if modification_request and modification_request.strip():
        return _generate_modification(
            str(project_id), conversation_id, user_id, modification_request.strip()
        )

    emit, writer = _make_emit()

    # ---- Step 1: read KB ---------------------------------------------------- #
    emit("reading_kb", "started", "Reading what we know about the product…")
    kb = get_project_knowledge_base(str(project_id))
    if not kb or (not (kb.get("theory_context") or "").strip() and not kb.get("structured_context")):
        return (
            "I don't have enough analyzed information about this product yet to build a "
            "report. Please upload photos of the PCB and product (and any datasheets) "
            "so I can analyze them first, then ask me to generate the report."
        )
    theory = (kb.get("theory_context") or "").strip()
    structured = _structured_from_kb(kb)
    project_name = get_project_name(str(project_id)) or "this product"
    product_label = ((structured.get("product") or {}).get("name")) or project_name
    emit("reading_kb", "done", "Product knowledge loaded.")

    # ---- Step 2: HILT gap questions ----------------------------------------- #
    emit("hilt", "started", "Checking whether I need any details from you…")
    questions = report_builder.assess_missing(theory, structured, request)
    answers_by_id: dict[str, dict[str, Any]] = {}
    if questions:
        from langgraph.types import interrupt

        resumed = interrupt({"type": "report_questions", "questions": questions})
        raw_answers = resumed.get("answers") if isinstance(resumed, dict) else resumed
        if isinstance(raw_answers, dict):
            answers_by_id = {k: _normalize_answer(v) for k, v in raw_answers.items()}
        elif isinstance(raw_answers, list):
            for q, a in zip(questions, raw_answers):
                answers_by_id[q["id"]] = _normalize_answer(a)

        qa_pairs: list[dict[str, Any]] = []
        for q in questions:
            ans = answers_by_id.get(q["id"], {"answer": "", "file_ids": None, "status": "skipped"})
            try:
                reports.save_question_answer(
                    str(project_id), conversation_id, user_id,
                    question=q["prompt"], kind=q["kind"],
                    answer=ans["answer"] or None, file_ids=ans["file_ids"], status=ans["status"],
                )
            except Exception:
                logger.warning("Failed to persist report question", exc_info=True)
            qa_pairs.append({"question": q["prompt"], "answer": ans["answer"], "file_ids": ans["file_ids"]})
        try:
            enqueue_qa_insight(str(project_id), user_id, qa_pairs)
        except Exception:
            logger.warning("Failed to enqueue Q&A insight ingestion", exc_info=True)
    emit("hilt", "done", "Got what I need.")

    # ---- Determine production volume ---------------------------------------- #
    # The report is always for a SINGLE unit. We do not parse a production volume
    # from the request — costs are the recurring per-unit cost, with one-time NRE
    # reported separately (never amortized over a quantity).
    volume = 1

    # ---- Steps 3–9: run the pipeline ---------------------------------------- #
    report_json = report_pipeline.run_pipeline(
        structured=structured,
        theory=theory,
        product_label=product_label,
        volume=volume,
        emit=emit,
    )
    # The report is always single-unit: per-unit recurring cost, with one-time NRE
    # (tooling/firmware/line setup) reported separately — never amortized in.
    report_json.setdefault("dataQuality", []).append(
        "Figures are the recurring cost to build a single unit. One-time tooling, firmware "
        "and line-setup (NRE) is listed separately and is not included in the per-unit cost."
    )

    # ---- Steps 10–11: render, store, stream --------------------------------- #
    title = report_json.get("meta", {}).get("title") or f"Should-Cost Report — {product_label}"
    _render_store_stream(
        report_json=report_json,
        project_id=str(project_id),
        conversation_id=conversation_id,
        user_id=user_id,
        title=title,
        volume=volume,
        emit=emit,
        writer=writer,
    )

    live = report_json.get("bom", {}).get("live_count", 0)
    total_lines = live + report_json.get("bom", {}).get("est_count", 0)
    note = "" if not report_json.get("dataQuality") else " Some figures use industry estimates where live data wasn't available — see the Data Confidence notes."
    return (
        f'I generated the single-unit should-cost report "{title}" '
        f"({live} of {total_lines} BOM lines priced live).{note} "
        "It's the recurring cost to build one unit, with one-time tooling/NRE listed separately. "
        "It's shown on the right and the PDF is ready to download — want a line removed or more detail in a section?"
    )
