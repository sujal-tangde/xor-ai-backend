"""LangChain tools for direct (non-RAG) lookups of uploads and their insights."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from src.agent.tools.validation import (
    invalid_file_id_message,
    invalid_project_id_message,
    is_uuid,
)
from src.services.file_storage import get_files_by_ids
from src.services.projects_service import get_insight, get_insights, list_uploads

logger = logging.getLogger(__name__)

LIST_UPLOADS_NAME = "list_project_uploads"
LIST_UPLOADS_LABEL = "project uploads"
GET_INSIGHT_NAME = "get_upload_insight"
GET_INSIGHT_LABEL = "upload insight"
GET_INSIGHTS_BY_IDS_NAME = "get_insights_by_file_ids"
GET_INSIGHTS_BY_IDS_LABEL = "attached file insights"


def _structured_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_insight(record: dict[str, Any]) -> str:
    media = record.get("media_kind") or "upload"
    theory = (record.get("theory_context") or "").strip()
    structured = _structured_text(record.get("structured_context"))
    parts = [f"Media kind: {media}"]
    if theory:
        parts.append(f"Theory analysis:\n{theory}")
    parts.append(f"Structured analysis (JSON):\n{structured}")
    return "\n\n".join(parts)


@tool(LIST_UPLOADS_NAME)
def list_project_uploads(project_id: str) -> str:
    """List every file and image uploaded to this project with its processing status.

    Use this whenever you need to know what uploads exist before fetching a
    specific one, or when the user asks "what have I uploaded" / "which images do
    you have". When the user references a file or image by name or number, call
    this first to resolve the name to a file_id, then call get_upload_insight.
    Pass the project UUID from the message context.
    """
    if not project_id:
        return "No project ID provided."
    if not is_uuid(project_id):
        return invalid_project_id_message(project_id)

    try:
        uploads = list_uploads(project_id)
    except Exception as exc:  # pragma: no cover - surfaced to the agent
        logger.exception("list_uploads failed for project %s", project_id)
        return f"Could not list uploads: {exc}"
    if not uploads:
        return "No files or images have been uploaded to this project yet."

    lines = ["Uploads in this project:"]
    for u in uploads:
        lines.append(
            f"- file_id={u.get('id')} | name={u.get('name')} | "
            f"kind={u.get('file_type')} | status={u.get('processing_status') or 'unknown'}"
        )
    return "\n".join(lines)


@tool(GET_INSIGHT_NAME)
def get_upload_insight(project_id: str, file_id: str) -> str:
    """Retrieve one specific upload's full theory + structured analysis.

    Call this when the user references a specific image or file (resolve its name
    to a file_id via list_project_uploads first). Returns the full structured JSON
    intact — do not expect it to be summarized. Pass the project UUID and the
    file_id.
    """
    if not project_id or not file_id:
        return "Both project_id and file_id are required."
    if not is_uuid(project_id):
        return invalid_project_id_message(project_id)
    if not is_uuid(file_id):
        return invalid_file_id_message(file_id)

    try:
        record = get_insight(project_id, file_id)
        if record is None:
            statuses = {
                f["id"]: f.get("processing_status") for f in get_files_by_ids([file_id])
            }
            status = statuses.get(file_id)
            if status and status != "complete":
                return (
                    f"File {file_id}: analysis is still {status}. Ask the user to wait "
                    "a moment and try again."
                )
            return f"No insight found for file {file_id} in project {project_id}."
    except Exception as exc:  # pragma: no cover - surfaced to the agent
        logger.exception("get_upload_insight failed for file %s", file_id)
        return f"Could not load that upload's analysis: {exc}"

    return f"Insight for file {file_id}:\n\n{_format_insight(record)}"


@tool(GET_INSIGHTS_BY_IDS_NAME)
def get_insights_by_file_ids(project_id: str, file_ids: list[str]) -> str:
    """Retrieve full theory + structured analysis for several attached file_ids.

    Call this when the frontend attaches one or more file_ids to the user's
    message (the user selected specific images/files in the UI). This is the
    GROUND TRUTH for those files — prioritize it over any RAG search results when
    answering about them. Returns full structured JSON per file, intact. If a
    file's processing is not yet complete, tell the user its analysis is still
    pending. Pass the project UUID and the list of file_ids.
    """
    if not project_id or not file_ids:
        return "project_id and at least one file_id are required."
    if not is_uuid(project_id):
        return invalid_project_id_message(project_id)

    valid_ids = [fid for fid in file_ids if is_uuid(fid)]
    invalid_ids = [fid for fid in file_ids if not is_uuid(fid)]

    sections: list[str] = []
    for bad in invalid_ids:
        sections.append(invalid_file_id_message(bad))

    if not valid_ids:
        return "\n\n---\n\n".join(sections) if sections else (
            "No valid file IDs were provided."
        )

    try:
        insights = {r["file_id"]: r for r in get_insights(project_id, valid_ids)}
        statuses = {
            f["id"]: f.get("processing_status") for f in get_files_by_ids(valid_ids)
        }
    except Exception as exc:  # pragma: no cover - surfaced to the agent
        logger.exception("get_insights_by_file_ids failed for project %s", project_id)
        return f"Could not load the attached files' analysis: {exc}"

    for file_id in valid_ids:
        record = insights.get(file_id)
        if record is not None:
            sections.append(f"File {file_id}:\n{_format_insight(record)}")
            continue
        status = statuses.get(file_id)
        if status is None:
            sections.append(f"File {file_id}: not found in this project.")
        elif status != "complete":
            sections.append(
                f"File {file_id}: analysis still {status} — not ready yet. "
                "Tell the user its analysis is still pending."
            )
        else:
            sections.append(f"File {file_id}: no insight stored.")
    return "\n\n---\n\n".join(sections)
