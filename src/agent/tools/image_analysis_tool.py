"""LangChain tool to fetch stored hardware image analysis by file ID."""

from __future__ import annotations

from langchain_core.tools import tool

from src.services.file_storage import get_files_by_ids


def _format_analysis_record(record: dict) -> str:
    name = record.get("name") or "Unknown"
    file_id = record.get("id") or ""
    file_type = record.get("file_type") or ""
    status = record.get("image_analysis_status") or "unknown"
    analysis = (record.get("image_analysis") or "").strip()

    if file_type != "image":
        return f"File: {name} (ID: {file_id})\nStatus: not an image — no hardware analysis available."

    if status == "processing":
        return (
            f"File: {name} (ID: {file_id})\n"
            "Status: analysis still in progress. Ask the user to wait a moment and try again."
        )

    if status == "failed":
        return (
            f"File: {name} (ID: {file_id})\n"
            "Status: analysis failed. No stored analysis is available for this image."
        )

    if not analysis:
        return (
            f"File: {name} (ID: {file_id})\n"
            "Status: no analysis text stored for this image yet."
        )

    return f"File: {name} (ID: {file_id})\nAnalysis:\n{analysis}"


@tool
def get_image_analysis(file_ids: list[str]) -> str:
    """Retrieve stored hardware/teardown analysis for uploaded image files.

    Call this when the user asks about images they attached, or about PCB components,
    enclosures, chips, connectors, or other hardware details visible in their uploads.
    Pass the file UUIDs provided in the user's message context.
    """
    if not file_ids:
        return "No file IDs provided."

    records = get_files_by_ids(file_ids)
    found_ids = {str(record.get("id")) for record in records}

    sections = [_format_analysis_record(record) for record in records]

    missing = [file_id for file_id in file_ids if file_id not in found_ids]
    for file_id in missing:
        sections.append(f"File ID: {file_id}\nStatus: not found in database.")

    return "\n\n---\n\n".join(sections)
