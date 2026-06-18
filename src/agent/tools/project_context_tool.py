"""LangChain tool to fetch a project's recomputed whole-product knowledge base."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from src.agent.tools.validation import invalid_project_id_message, is_uuid
from src.services.projects_service import get_project_knowledge_base

logger = logging.getLogger(__name__)

TOOL_NAME = "get_project_context"
TOOL_LABEL = "project context"


def _structured_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


@tool(TOOL_NAME)
def get_project_context_tool(project_id: str) -> str:
    """Retrieve the project's recomputed whole-product analysis.

    Every image and document uploaded to a project is analyzed into a per-upload
    insight, and those insights are continuously merged into a single cumulative
    understanding of the product, stored as a prose theory summary plus a
    structured JSON breakdown (identity, enclosure, exhaustive component list,
    connectors, architecture, etc.).

    Call when the user asks about the overall project, the product as a whole,
    the full bill of materials / component list, the system architecture, or
    anything spanning more than a single upload. Do NOT call for greetings,
    small talk, or messages that do not need product data. Pass the project UUID
    provided in the message context.
    """
    if not project_id:
        return "No project ID provided."
    if not is_uuid(project_id):
        return invalid_project_id_message(project_id)

    try:
        kb = get_project_knowledge_base(project_id)
    except Exception as exc:  # pragma: no cover - surfaced to the agent
        logger.exception("get_project_knowledge_base failed for %s", project_id)
        return f"Could not load the project knowledge base: {exc}"
    if kb is None:
        return (
            f"Project {project_id}: no knowledge base yet — nothing has been "
            "analyzed for this project, or processing is still in progress."
        )

    theory = (kb.get("theory_context") or "").strip()
    structured = _structured_text(kb.get("structured_context"))
    total = kb.get("insights_total")
    processed = kb.get("insights_processed")

    if not theory and not structured:
        return (
            f"Project {project_id}: knowledge base exists but is still being "
            f"computed ({processed}/{total} uploads folded in). Try again shortly."
        )

    sections = [
        f"Project knowledge base (ID: {project_id}) — "
        f"{processed}/{total} uploads folded in:"
    ]
    if theory:
        sections.append(f"Theory analysis:\n{theory}")
    if structured:
        sections.append(f"Structured analysis (JSON):\n{structured}")
    return "\n\n---\n\n".join(sections)
