"""LangChain RAG tools: semantic search over image_chunks / file_chunks."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from src.agent.tools.validation import invalid_project_id_message, is_uuid
from src.services.embeddings import embed_query
from src.services.vector_store import search_chunks

logger = logging.getLogger(__name__)

SEARCH_IMAGE_CHUNKS_NAME = "search_image_chunks"
SEARCH_IMAGE_CHUNKS_LABEL = "image search"
SEARCH_FILE_CHUNKS_NAME = "search_file_chunks"
SEARCH_FILE_CHUNKS_LABEL = "document search"

TOP_K = 8


def _format_results(rows: list[dict[str, Any]], kind: str) -> str:
    if not rows:
        return f"No matching {kind} content found in this project."
    lines = [f"Top {kind} matches:"]
    for row in rows:
        meta = row.get("metadata") or {}
        name = meta.get("file_name") if isinstance(meta, dict) else None
        header = f"[file_id={row.get('file_id')}"
        if name:
            header += f" | {name}"
        header += "]"
        lines.append(f"{header}\n{(row.get('content') or '').strip()}")
    return "\n\n---\n\n".join(lines)


def _search(table: str, project_id: str, query: str, kind: str) -> str:
    if not project_id:
        return "No project ID provided."
    if not is_uuid(project_id):
        return invalid_project_id_message(project_id)
    if not query or not query.strip():
        return "No search query provided."
    try:
        vector = embed_query(query)
        rows = search_chunks(table, project_id, vector, k=TOP_K)
    except Exception as exc:  # pragma: no cover - surfaced to the agent
        logger.exception("RAG search over %s failed", table)
        return f"Search failed: {exc}"
    return _format_results(rows, kind)


@tool(SEARCH_IMAGE_CHUNKS_NAME)
def search_image_chunks(project_id: str, query: str) -> str:
    """Semantic search across the analysis of ALL images in this project.

    Use for fuzzy questions about visual content where you don't know which
    specific image holds the answer — e.g. "is there a crystal near the MCU",
    "what ICs are on any of the boards", "any visible markings on connectors".
    Pass the project UUID and a natural-language query.
    """
    return _search("image_chunks", project_id, query, "image")


@tool(SEARCH_FILE_CHUNKS_NAME)
def search_file_chunks(project_id: str, query: str) -> str:
    """Semantic search across the text of ALL documents in this project.

    Use for fuzzy questions about document content (datasheets, manuals, box
    text, specs) where you don't know which file holds the answer — e.g. "what is
    the rated supply voltage", "what does the manual say about the antenna", "any
    certifications mentioned". Pass the project UUID and a natural-language query.
    """
    return _search("file_chunks", project_id, query, "document")
