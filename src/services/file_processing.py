"""Background processing entrypoints (run by the RQ worker).

``process_image`` and ``process_document`` are enqueued by the upload API. Each
runs the heavy work for one upload — analysis, embedding into the RAG chunk
tables, and recording a per-upload insight — while keeping the
``uploaded_files.processing_status`` column up to date so the UI can poll.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.services.embeddings import embed_texts
from src.services.file_storage import get_supabase
from src.services.insights import record_insight
from src.services.llm_analysis import (
    analyze_document_chunk,
    analyze_image_dual,
    reduce_document_partials,
)
from src.services.text_extraction import chunk_document, split_text
from src.services.vector_store import insert_chunks

logger = logging.getLogger(__name__)

# Bound the document map step so a huge file can't fan out into hundreds of LLM
# calls. Chunks are batched up to this many characters per map call, capped at
# MAX_MAP_BATCHES batches (extra text still gets embedded for RAG).
MAP_BATCH_CHARS = 6000
MAX_MAP_BATCHES = 20


def _set_status(file_id: str, status: str) -> None:
    try:
        get_supabase().table("uploaded_files").update(
            {"processing_status": status}
        ).eq("id", file_id).execute()
    except Exception:
        logger.exception("Failed to set processing_status=%s for file %s", status, file_id)


def _semantic_text(theory: str, structured: dict[str, Any] | None) -> str:
    """Build the embeddable semantic text for an image from its analysis."""
    parts = [(theory or "").strip()]
    if structured:
        parts.append("Structured facts:\n" + json.dumps(structured, ensure_ascii=False))
    return "\n\n".join(p for p in parts if p)


def _batch_chunks(chunks: list[str]) -> list[str]:
    """Group text chunks into batches bounded by MAP_BATCH_CHARS."""
    batches: list[str] = []
    current: list[str] = []
    length = 0
    for chunk in chunks:
        if current and length + len(chunk) > MAP_BATCH_CHARS:
            batches.append("\n\n".join(current))
            current = []
            length = 0
        current.append(chunk)
        length += len(chunk)
    if current:
        batches.append("\n\n".join(current))
    return batches


def process_image(
    file_id: str,
    project_id: str,
    user_id: str | None,
    compressed_jpeg: bytes,
    file_name: str,
) -> None:
    """Image pipeline: vision analysis -> embed semantic text -> insight."""
    logger.info("Processing image %s for project %s", file_id, project_id)
    _set_status(file_id, "processing")
    try:
        theory, structured = analyze_image_dual(compressed_jpeg)

        semantic = _semantic_text(theory, structured)
        chunks = split_text(semantic) or ([semantic] if semantic else [])
        if chunks:
            embeddings = embed_texts(chunks)
            rows = [
                {
                    "project_id": project_id,
                    "user_id": user_id,
                    "file_id": file_id,
                    "content": chunk,
                    "chunk_index": i,
                    "embedding": embeddings[i],
                    "metadata": {"file_name": file_name, "media_kind": "image"},
                }
                for i, chunk in enumerate(chunks)
            ]
            insert_chunks("image_chunks", rows)

        record_insight(project_id, user_id, file_id, "image", theory, structured)
        _set_status(file_id, "complete")
        logger.info("Completed image %s", file_id)
    except Exception:
        logger.exception("Image processing failed for %s", file_id)
        _set_status(file_id, "failed")
        raise


def process_document(
    file_id: str,
    project_id: str,
    user_id: str | None,
    raw_bytes: bytes,
    ext: str,
    file_name: str,
) -> None:
    """Document pipeline: extract+chunk -> embed -> map/reduce analysis -> insight."""
    logger.info("Processing document %s (.%s) for project %s", file_id, ext, project_id)
    _set_status(file_id, "processing")
    try:
        chunks = chunk_document(raw_bytes, ext)

        if chunks:
            embeddings = embed_texts(chunks)
            rows = [
                {
                    "project_id": project_id,
                    "user_id": user_id,
                    "file_id": file_id,
                    "content": chunk,
                    "chunk_index": i,
                    "embedding": embeddings[i],
                    "metadata": {"file_name": file_name, "media_kind": "document"},
                }
                for i, chunk in enumerate(chunks)
            ]
            insert_chunks("file_chunks", rows)

        # Map + reduce over batched chunks to one (theory, structured).
        batches = _batch_chunks(chunks)
        if len(batches) > MAX_MAP_BATCHES:
            logger.warning(
                "Document %s has %s map batches; capping to %s for analysis",
                file_id,
                len(batches),
                MAX_MAP_BATCHES,
            )
            batches = batches[:MAX_MAP_BATCHES]

        partials: list[dict[str, Any]] = []
        for batch in batches:
            theory, structured = analyze_document_chunk(batch)
            partials.append({"theory": theory, "structured": structured})

        if not partials:
            theory, structured = "", None
        elif len(partials) == 1:
            theory, structured = partials[0]["theory"], partials[0]["structured"]
        else:
            theory, structured = reduce_document_partials(partials)

        record_insight(project_id, user_id, file_id, "document", theory, structured)
        _set_status(file_id, "complete")
        logger.info("Completed document %s", file_id)
    except Exception:
        logger.exception("Document processing failed for %s", file_id)
        _set_status(file_id, "failed")
        raise
