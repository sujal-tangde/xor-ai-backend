"""The ``get_report`` agent tool.

Fetches an EXISTING should-cost report from the database and re-displays it in the
conversation — without regenerating or editing anything. Use it whenever the user
wants to see, open, pull up, or re-display a report that was already produced
(e.g. "show me my report", "open the cost report", "pull up the report again").

It reads the saved ``reports`` row (the structured JSON, stored markdown and PDF
URL), streams a ``report_ready`` event so the preview panel shows the report on the
right with its download button, and returns a short confirmation. Nothing is
recomputed: the report is served exactly as it was last saved.

To CREATE a report use ``report_generation``; to CHANGE one use ``report_edit``.
This tool only retrieves and shows an existing report.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent.tools.report_tool import _cfg, _writer
from src.agent.tools.validation import invalid_project_id_message, is_uuid
from src.services import report_template, reports

logger = logging.getLogger(__name__)

TOOL_NAME = "get_report"
TOOL_LABEL = "fetching report"


def _coerce_report_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@tool(TOOL_NAME)
def get_report_tool(
    report_id: str | None = None,
    config: RunnableConfig = None,  # type: ignore[assignment]
) -> str:
    """Fetch an existing should-cost report from the database and show it.

    Call this whenever the user wants to VIEW or RE-OPEN a report that already
    exists — "show me the report", "open my cost report", "pull the report up
    again", "can I see the report?". It does NOT generate or change anything; it
    loads the saved report and re-displays it on the right with a download button.

    Resolution order:
      1. ``report_id`` if given (scoped to the current user).
      2. Otherwise the latest report for THIS conversation.
      3. Otherwise the latest report for the project (a report made in an earlier
         chat for the same product is still the right one to show).

    If no report exists anywhere for the project, the tool says so — in that case
    call ``report_generation`` to create one first.

    Args:
        report_id: The specific report's ID to fetch. Omit it to fetch the most
            recent report for this conversation (or project). Only pass an ID the
            user explicitly referenced.
    """
    cfg = _cfg(config)
    project_id = cfg.get("project_id")
    user_id = cfg.get("user_id")
    conversation_id = cfg.get("conversation_id")

    if not project_id:
        return "No project is associated with this conversation, so I can't fetch a report."
    if not is_uuid(str(project_id)):
        return invalid_project_id_message(str(project_id))

    record: dict[str, Any] | None = None
    if report_id and str(report_id).strip():
        record = reports.get_report(str(report_id).strip(), user_id)
        if record is None:
            return (
                f"I couldn't find a report with that ID for you. Try asking me to show "
                "the latest report, or generate a new one."
            )
    else:
        if conversation_id:
            record = reports.latest_report_for_conversation(conversation_id)
        if record is None:
            record = reports.latest_report_for_project(str(project_id), user_id)

    if record is None:
        return (
            "There's no saved report for this product yet. Ask me to generate a "
            "report and I'll create one."
        )

    report_json = _coerce_report_json(record.get("report_json"))

    # Prefer the stored markdown; fall back to rendering it from the structured JSON
    # for older reports saved before markdown was persisted, so the panel always
    # has something to show.
    markdown_text = record.get("markdown") or ""
    if not markdown_text and report_json:
        try:
            markdown_text = report_template.render_markdown(report_json)
        except Exception:
            logger.warning("Could not render markdown for report %s", record.get("id"), exc_info=True)
            markdown_text = ""
    markdown_text = reports.normalize_report_markdown(markdown_text) if markdown_text else ""

    title = record.get("title") or "Should-Cost Report"

    # Stream the report so the preview panel renders it on the right with its
    # download button — the same event the generate/edit flows emit.
    writer = _writer()
    try:
        writer({
            "kind": "report_ready",
            "report_id": record.get("id"),
            "title": title,
            "markdown": markdown_text,
            "pdf_url": record.get("pdf_url"),
            "volume": record.get("volume"),
            "fx_rate": (report_json.get("fx") or {}).get("rate"),
        })
    except Exception:  # pragma: no cover - streaming best-effort
        logger.warning("Failed to stream fetched report %s", record.get("id"), exc_info=True)

    pdf_note = (
        " The PDF is ready to download."
        if record.get("pdf_url")
        else " (No PDF is attached to this report — ask me to regenerate it if you need one.)"
    )
    return (
        f'Here\'s your report "{title}" — it\'s shown on the right.{pdf_note} '
        "Let me know if you'd like to change anything."
    )
