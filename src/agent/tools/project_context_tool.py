"""LangChain tool to fetch a project's accumulated hardware analysis context."""

from __future__ import annotations

from langchain_core.tools import tool

from src.services.projects_service import get_project_context

TOOL_NAME = "get_project_context"
TOOL_LABEL = "project context"


@tool(TOOL_NAME)
def get_project_context_tool(project_id: str) -> str:
    """Retrieve the project's accumulated hardware analysis context.

    Every image uploaded to a project is analyzed and merged into a single
    cumulative understanding of the product, stored as a prose theory summary
    plus a structured JSON breakdown (identity, enclosure, exhaustive component
    list, connectors, architecture, etc.).

    Call this whenever the user asks about the overall project, the product as a
    whole, the full bill of materials / component list, the system architecture,
    or anything that spans more than a single uploaded image. Pass the project
    UUID provided in the message context.
    """
    if not project_id:
        return "No project ID provided."

    project = get_project_context(project_id)
    if project is None:
        return f"Project {project_id} not found."

    name = project.get("name") or "Unknown project"
    theory = (project.get("context") or "").strip()
    structured = (project.get("structured_context") or "").strip()

    if not theory and not structured:
        return (
            f"Project: {name} (ID: {project_id})\n"
            "No accumulated analysis yet — no images have been analyzed for this "
            "project, or analysis is still in progress."
        )

    sections = [f"Project: {name} (ID: {project_id})"]
    if theory:
        sections.append(f"Theory analysis:\n{theory}")
    if structured:
        sections.append(f"Structured analysis (JSON):\n{structured}")
    return "\n\n---\n\n".join(sections)
